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
