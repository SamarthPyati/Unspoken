"""Generate notebooks/kaggle_quick_dev_test.ipynb (plain JSON, no nbformat needed).

Option B for a tight deadline: a reduced dev sweep, the pre-registered m rule, then a test
SUBSET, in one Kaggle session. Edit the cells here, not the .ipynb, then regenerate:
  python notebooks/build_kaggle_quick.py notebooks/kaggle_quick_dev_test.ipynb
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# Unspoken on Kaggle: quick dev sweep + test subset (option B)

For a tight deadline, one session (**~4.5–5 h** on 2x T4) that produces real numbers for the paper:

| Phase | Data | Runs (Qwen2.5-3B) |
|---|---|---|
| Dev sweep (reduced) | GSM8K train 5,000–5,099 (100 problems) | text MAS, single CoT, latent m = 1, 2, 3 |
| Choose m | the dev latent runs | pre-registered rule (`scripts/select_m.py`), `--min_n 100` |
| Test subset | GSM8K test 0–299 (300 problems) | text MAS, single CoT, latent MAS at the chosen m |

**Deviations from the pre-registered plan** (the paper must state them, with the reason: deadline):
100 instead of 300 dev problems; dev grid reduced to m = 1, 2, 3 (m ≥ 5 was incoherent in every
sanity run); the test set is evaluated on its first 300 problems now, and these same problems are
reused (not re-run) when the full 1,319-problem evaluation is done later.

**Run it:** GPU T4 x2, Internet on, `REPO_URL` below, then **Save Version → Save & Run All
(Commit)**, once. If the session ends early, attach its output to a new session: finished problems
are skipped.

GPU 0 runs text MAS (dev, then test). GPU 1 runs the latent dev runs, applies the rule as soon as
they finish (without waiting for GPU 0), then runs the test subset at the chosen m and single CoT.
""")

code(r"""
# ---- configuration ---------------------------------------------------------------
REPO_URL = "https://github.com/SamarthPyati/Unspoken.git"   # "" -> use a .bundle under /kaggle/input
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_DEV, DEV_OFFSET = 100, 5000
N_TEST = 300
DEV_M = [1, 2, 3]
BUDGET_HOURS = 11.0
WORK = "/kaggle/working/unspoken"

import glob, json, os, re, shutil, subprocess, sys, threading, time
T0 = time.time()

def sh(cmd, check=True, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=kw.pop("cwd", WORK if os.path.isdir(WORK) else None),
                       text=True, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r

def hours_left():
    return BUDGET_HOURS - (time.time() - T0) / 3600
""")

md("## 0. Code, environment, tests, resume")

code(r"""
if os.path.isdir(WORK):
    print("repo already present:", WORK)
elif REPO_URL:
    url = REPO_URL
    try:
        from kaggle_secrets import UserSecretsClient
        url = url.replace("https://", f"https://{UserSecretsClient().get_secret('GITHUB_TOKEN')}@")
    except Exception:
        pass
    sh(f"git clone --quiet {url} {WORK}", cwd="/kaggle/working")
else:
    bundles = glob.glob("/kaggle/input/**/*.bundle", recursive=True)
    assert bundles, "No REPO_URL and no .bundle under /kaggle/input."
    sh(f"git clone --quiet {bundles[0]} {WORK}", cwd="/kaggle/working")
    sh("git checkout --quiet main")
sh("git log -1 --format='commit %h  %s  (%ad)' --date=short")
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
sh("grep -v '^gradio' requirements.txt > /tmp/req.txt && python -m pip install -q -r /tmp/req.txt")
r = sh("python -m pytest -q -p no:cacheprovider", check=False)
assert r.returncode == 0, "Tests failed: fix before spending GPU hours."
""")

code(r"""
SHORT = MODEL.split("/")[-1]
DEV_DIR, TEST_DIR = f"{WORK}/results/{SHORT}/train", f"{WORK}/results/{SHORT}"
os.makedirs(DEV_DIR, exist_ok=True)
# Resume: copy finished problems from a previous session's output (never overwrite).
for p in glob.glob(f"/kaggle/input/**/results/{SHORT}/**/*", recursive=True):
    if not os.path.isfile(p):
        continue
    sub = "train" if os.path.basename(os.path.dirname(p)) == "train" else ""
    dst = os.path.join(TEST_DIR, sub, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst)

def rows(path):
    return sum(1 for l in open(path) if l.strip()) if os.path.exists(path) else 0
print({os.path.relpath(p, TEST_DIR): rows(p) for p in glob.glob(f"{TEST_DIR}/**/*.jsonl", recursive=True)} or "starting fresh")

from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download(MODEL); load_dataset("openai/gsm8k", "main")
print(f"setup done in {(time.time() - T0) / 60:.1f} min")
""")

md("""
## 1. Run both GPUs

Each GPU works through its list. GPU 1's list includes the m selection, which runs as soon as the
three latent dev runs exist, so the test-subset latent run can start without waiting for GPU 0.
""")

code(r"""
# Exclusive GPUs, or nothing (see the T6 attempt-1 incident in HANDOFF).
alive = [p for p in globals().get("_Q_PROCS", []) if p.poll() is None]
assert not alive, f"{len(alive)} run(s) from an earlier execution are still going (!pkill -f scripts.run)"
busy = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
                      shell=True, capture_output=True, text=True).stdout.strip()
assert not busy, "GPUs already in use (stop them with !pkill -f scripts.run, or restart):\n" + busy
_Q_PROCS = []

DEV = f"--model {MODEL} --split train --offset {DEV_OFFSET} --n {N_DEV}"
TEST = f"--model {MODEL} --split test --n {N_TEST}"
chosen = {}

def select_m():
    # The pre-registered rule, on the reduced dev sweep (deviation: --min_n N_DEV instead of 300).
    r = subprocess.run(f"python -m scripts.select_m --dir {DEV_DIR} --min_n {N_DEV}", shell=True,
                       cwd=WORK, capture_output=True, text=True)
    open(f"{WORK}/select_m.txt", "w").write(r.stdout + r.stderr)
    print(r.stdout + r.stderr, flush=True)
    found = re.search(r"chosen m = (\d+)", r.stdout)
    assert r.returncode == 0 and found, "m selection failed (see select_m.txt)"
    chosen["m"] = int(found.group(1))
    return f"--method latent_mas --latent_steps {chosen['m']} {TEST}"

LANES = {
    0: [("dev text_mas", f"--method text_mas {DEV}"),
        ("test text_mas", f"--method text_mas {TEST}")],
    1: [*[(f"dev latent m{m}", f"--method latent_mas --latent_steps {m} {DEV}") for m in DEV_M],
        ("dev single_cot", f"--method single_cot {DEV}"),
        ("test latent (chosen m)", select_m),          # resolved when reached
        ("test single_cot", f"--method single_cot {TEST}")],
}
os.makedirs(f"{WORK}/logs", exist_ok=True)
status = {}

def lane(gpu, steps):
    for i, (name, args) in enumerate(steps):
        if hours_left() < 0.05:
            status[name] = "stopped (budget, resumable)"; continue
        if callable(args):
            try:
                args = args()
            except Exception as exc:
                status[name] = f"failed: {exc}"; print(f"[GPU {gpu}] {name}: {exc}", flush=True); continue
        t, log = time.time(), f"{WORK}/logs/q_gpu{gpu}_{i:02d}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.log"
        with open(log, "w") as f:
            f.write(f"# python -m scripts.run {args}\n"); f.flush()
            p = subprocess.Popen(f"python -m scripts.run {args}", shell=True, cwd=WORK, stdout=f,
                                 stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            _Q_PROCS.append(p)
            try:
                rc = p.wait(timeout=max(60, hours_left() * 3600))
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(); rc = "stopped (budget, resumable)"
        status[name] = rc
        print(f"[GPU {gpu}] {'ok ' if rc == 0 else rc} {(time.time() - t) / 60:6.1f} min  {name}", flush=True)

threads = [threading.Thread(target=lane, args=(g, s)) for g, s in LANES.items()]
[t.start() for t in threads]; [t.join() for t in threads]
failed = {k: v for k, v in status.items() if v != 0 and not str(v).startswith("stopped")}
print(f"\nafter {(time.time() - T0) / 3600:.2f} h: chosen m = {chosen.get('m')}, status = {status}")
if failed:
    print("\n" + "!" * 70 + f"\nRUNS FAILED (see logs/): {failed}\n" + "!" * 70)
    for path in sorted(glob.glob(f"{WORK}/logs/q_*.log")):
        print(f"\n--- {os.path.basename(path)}\n" + "".join(open(path).readlines()[-3:]))
""")

md("## 2. Tables: dev sweep, test subset, and the H1 indication")

code(r"""
tables = {}
for name, d in (("dev (train 5000-5099)", DEV_DIR), (f"test subset (0-{N_TEST - 1})", TEST_DIR)):
    r = sh(f"python -m scripts.analyze --dir {d} --baseline text_mas_seed0", check=False, capture_output=True)
    tables[name] = r.stdout + r.stderr
    print(f"==================== {name}\n{tables[name]}")
open(f"{WORK}/analysis.txt", "w").write("\n".join(f"== {k}\n{v}" for k, v in tables.items()))
""")

code(r"""
sys.path.insert(0, WORK)
from src.metrics import paired_diff_ci, relative_reduction, mcnemar
def load(p): return {r["id"]: r for r in map(json.loads, (l for l in open(p) if l.strip()))}
h1 = "H1 indication: not computable yet"
try:
    text = load(f"{TEST_DIR}/text_mas_seed0.jsonl")
    lat = load(f"{TEST_DIR}/latent_mas_m{chosen['m']}_seed0.jsonl")
    ids = sorted(set(text) & set(lat))
    assert ids, "no shared problems"
    d = paired_diff_ci([lat[i]["correct"] for i in ids], [text[i]["correct"] for i in ids])
    mc = mcnemar([lat[i]["correct"] for i in ids], [text[i]["correct"] for i in ids])
    saved = relative_reduction([text[i]["output_tokens"] for i in ids], [lat[i]["output_tokens"] for i in ids])
    fwd = relative_reduction([text[i]["forward_passes"] for i in ids], [lat[i]["forward_passes"] for i in ids])
    h1 = (f"H1 indication on the test subset ({len(ids)}/{N_TEST} problems, m={chosen['m']}): "
          f"accuracy latent - text = {d['mean']:+.3f} [95% CI {d['lo']:+.3f}, {d['hi']:+.3f}], "
          f"McNemar p = {mc['p_value']:.3g}; output tokens saved = {saved:.0%}; forward passes saved = {fwd:.0%}.\n"
          f"  accuracy, point estimate >= -0.03: {d['mean'] >= -0.03}\n"
          f"  accuracy, CI lower bound >= -0.03: {d['lo'] >= -0.03}\n"
          f"  tokens saved >= 50%:              {saved >= 0.5}")
except Exception as exc:
    h1 += f" ({exc})"
print(h1); open(f"{WORK}/h1_subset.txt", "w").write(h1 + "\n")
""")

md("## 3. Package")

code(r"""
out = "/kaggle/working/unspoken_kaggle_quick.zip"
sh(f"rm -f {out} && zip -qr {out} results logs analysis.txt h1_subset.txt select_m.txt")
print(f"wrote {out} ({os.path.getsize(out) / 2**20:.1f} MB)\n")
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=WORK, text=True).strip()
if failed:
    raise RuntimeError(f"DID NOT COMPLETE: runs failed {failed}. The zip holds the logs.")
done = all(v == 0 for v in status.values()) and len(status) == sum(len(s) for s in LANES.values())
print("---- paste into spec/HANDOFF.md (Change log) ----" if done else "PARTIAL: attach this output to a new session to resume.")
if done:
    print(f"- {time.strftime('%Y-%m-%d')}: Option B on Kaggle 2x T4 (commit {commit}, {SHORT}): reduced dev sweep "
          f"(train {DEV_OFFSET}-{DEV_OFFSET + N_DEV - 1}; text MAS, single CoT, latent m={DEV_M}); pre-registered rule "
          f"with --min_n {N_DEV} chose m={chosen.get('m')}; test subset 0-{N_TEST - 1}. {h1.splitlines()[0]} "
          "Deviations (deadline): dev n=100 not 300, grid m=1-3, test subset reported before the full test run.")
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    c["source"] = [line + "\n" for line in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
