"""Script-level checks that need no model: argument guards and the analysis table."""
import json
import subprocess
import sys

from scripts.analyze import truncation_rate


def _run(*args):
    return subprocess.run([sys.executable, "-m", "scripts.run", *args],
                          capture_output=True, text=True)


def test_train_split_must_skip_adapter_slice():
    """HANDOFF issue 5: dev runs on train must not overlap the adapter's train[:2000]."""
    r = _run("--method", "single_cot", "--split", "train", "--offset", "0", "--n", "1")
    assert r.returncode == 2 and "--offset >= 2000" in r.stderr


def test_m0_drop_requires_latent_m0():
    r = _run("--method", "latent_mas", "--latent_steps", "10", "--m0_cache", "drop", "--n", "1")
    assert r.returncode == 2 and "only applies" in r.stderr


def test_truncation_rate_ignores_judger():
    rows = [{"per_agent_tokens": [256, 100, 256, 512]},   # judger (last) is never counted
            {"per_agent_tokens": [10, 20, 30, 40]}]
    assert truncation_rate(rows, 256) == 2 / 6
    assert truncation_rate([{"per_agent_tokens": [40]}], 256) is None   # single_cot


def test_analyze_end_to_end(tmp_path):
    """analyze.py on two tiny synthetic runs: table prints, comparison and truncation column appear."""
    def write(name, rows, cap=256):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (tmp_path / f"{name}.config.json").write_text(json.dumps({"agent_max_tokens": cap}))

    base = [{"id": f"p{i}", "correct": i % 2 == 0, "output_tokens": 700, "forward_passes": 704,
             "latency_s": 2.0, "per_agent_tokens": [256, 200, 200, 44]} for i in range(6)]
    lat = [{"id": f"p{i}", "correct": True, "output_tokens": 300, "forward_passes": 340,
            "latency_s": 1.0, "per_agent_tokens": [300]} for i in range(6)]
    write("text_mas_seed0", base)
    write("latent_mas_m10_seed0", lat)
    r = subprocess.run([sys.executable, "-m", "scripts.analyze", "--dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "6 problems shared by all" in r.stdout
    assert "agents at 256-token cap: 33%" in r.stdout
    assert "57%" in r.stdout            # 1 - 300/700 output tokens saved


def _timed_run(dirpath, tag, sec, fwd, n=2):
    with open(dirpath / f"{tag}_seed0.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"p{i}", "latency_s": sec, "forward_passes": fwd}) + "\n")


def test_plan_estimates_unmeasured_latent_m(tmp_path):
    from scripts import plan
    _timed_run(tmp_path, "single_cot", 10.0, 300)
    _timed_run(tmp_path, "latent_mas_m10", 20.0, 200)
    stats = plan.summarize_runs(str(tmp_path))
    assert stats["latent_mas_m10"] == {"n": 2, "sec": 20.0, "fwd": 200.0}
    assert plan.estimate(stats, "latent_mas_m10") == (20.0, False)
    sec, est = plan.estimate(stats, "latent_mas_m20")        # +30 passes at 0.1 s/pass
    assert est and abs(sec - 23.0) < 1e-9
    assert plan.estimate(stats, "latent_mas_m0_drop") == (10.0, True)   # proxy: single_cot
    assert plan.estimate(stats, "text_mas") == (None, True)            # never guessed


def test_plan_schedule_and_commands():
    from scripts import plan
    jobs = [{"hours": h} for h in (5, 4, 3, 2)]
    assert sorted(plan.schedule(jobs, 2)) == [7, 7]
    job = {"tag": "latent_mas_m0_drop", "size": "3B", "split": "train", "offset": 5000,
           "n": 300, "lane": 1}
    cmd = plan.command(job, {"3B": "Qwen/Qwen2.5-3B-Instruct"})
    assert cmd == ("CUDA_VISIBLE_DEVICES=1 python -m scripts.run --method latent_mas "
                   "--latent_steps 0 --model Qwen/Qwen2.5-3B-Instruct --m0_cache drop "
                   "--split train --n 300 --offset 5000")
    raw = plan.command({**job, "tag": "latent_mas_m10_raw", "split": "test", "offset": 0},
                       {"3B": "M"})
    assert raw.endswith("--latent_steps 10 --model M --realign raw --split test --n 300")


def test_plan_cli(tmp_path):
    for tag, sec, fwd in (("single_cot", 10, 300), ("text_mas", 40, 900), ("latent_mas_m10", 20, 200)):
        _timed_run(tmp_path, tag, sec, fwd)
    r = subprocess.run([sys.executable, "-m", "scripts.plan", "--dir", str(tmp_path),
                        "--emit_commands"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Total GPU-hours" in r.stdout and "# --- GPU 1" in r.stdout
    assert "Not estimated (time them first): 1.5B single_cot" in r.stdout   # no --dir_small


def _scored_run(dirpath, tag, correct):
    with open(dirpath / f"{tag}_seed0.jsonl", "w") as f:
        for i, c in enumerate(correct):
            f.write(json.dumps({"id": f"p{i}", "correct": c}) + "\n")


def test_select_m_rule(tmp_path):
    """Pre-registered rule: highest dev accuracy among plain latent runs; ties -> smaller m;
    ablation rows and text MAS are never candidates."""
    from scripts import select_m
    _scored_run(tmp_path, "latent_mas_m1", [1, 1, 0, 1])
    _scored_run(tmp_path, "latent_mas_m3", [1, 1, 1, 0])        # ties m=1 -> m=1 wins
    _scored_run(tmp_path, "latent_mas_m10", [0, 0, 0, 0])
    _scored_run(tmp_path, "latent_mas_m10_raw", [1, 1, 1, 1])   # ablation: ignored
    _scored_run(tmp_path, "latent_mas_m0_drop", [1, 1, 1, 1])   # ablation: ignored
    _scored_run(tmp_path, "text_mas", [1, 1, 1, 1])             # not latent: ignored
    chosen, table = select_m.select_m(select_m.load_latent_runs(str(tmp_path)))
    assert chosen == 1
    assert [m for m, _, _ in table] == [1, 3, 10]


def test_select_m_refuses_small_n(tmp_path):
    _scored_run(tmp_path, "latent_mas_m1", [1, 1])
    r = subprocess.run([sys.executable, "-m", "scripts.select_m", "--dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "finish the dev runs" in r.stderr


def test_plan_dev_grid_includes_small_m():
    from scripts import plan
    tags = [j["tag"] for j in plan.build_grid(10, 300, 1319, 5000) if j["phase"] == "T6 dev"]
    for m in (0, 1, 2, 3, 5, 10, 20, 40):
        assert f"latent_mas_m{m}" in tags


def test_plan_from_summary_csv_dev_phase(tmp_path):
    from scripts import plan
    csv = tmp_path / "t3.csv"
    csv.write_text("model,run,n,acc,boxed,junk,out_tok,fwd,sec,peak_mb,agents_at_cap\n"
                   "Qwen2.5-3B-Instruct,latent_mas_m1_seed0,5,1.0,1.0,0/5,180,187.4,9.9,6113,\n"
                   "Qwen2.5-1.5B-Instruct,single_cot_seed0,5,1.0,1.0,0/5,250,251,10.0,3007,\n")
    stats = plan.summarize_csv(str(csv), "Qwen2.5-3B-Instruct")
    assert stats == {"latent_mas_m1": {"n": 5, "sec": 9.9, "fwd": 187.4}}
    r = subprocess.run([sys.executable, "-m", "scripts.plan", "--summary_csv", str(csv),
                        "--phase", "dev"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "T6 dev" in r.stdout and "T7 final" not in r.stdout
