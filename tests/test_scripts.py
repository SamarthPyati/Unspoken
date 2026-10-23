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
