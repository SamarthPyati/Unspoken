"""Generate notebooks/kaggle_T6_dev.ipynb (plain JSON, no nbformat needed).

Edit the cells here, not the .ipynb, then regenerate from the repo root:
  python notebooks/build_kaggle_t6.py notebooks/kaggle_T6_dev.ipynb
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
# Unspoken on Kaggle: T6 dev sweep (choose m)

Runs the HANDOFF T6 grid on Qwen2.5-3B on the **dev slice** (GSM8K train, offset 5000, n=300),
never the test set. It then applies the **pre-registered rule** (`scripts/select_m.py`, logged
before any dev result existed) to choose `m` for the final runs, and plans T7 with that `m`.

| Grid (13 runs) | |
|---|---|
| Baselines | `single_cot`, `text_mas` |
| Latent steps | `m ∈ {0, 1, 2, 3, 5, 10, 20, 40}` (1–3 added on 2026-10-23, HANDOFF issue 4) |
| Ablations | `m=0 drop`, `m=10 raw`, `m=10 norm_only` |

Expected: **~7.6 h** on 2x T4 (from the T3 timings), so one session.

**Before you run.** Use the same settings as the T1/T3/T5 notebook: **GPU T4 x2**, **Internet on**,
and the code as a `.bundle` input or `REPO_URL`. Then **Save Version → Save & Run All (Commit)**, so
it runs in the background, on a fresh machine, without the browser open.

**Run it once per session.** Don't press *Run All* again in an interactive session while runs are
going: earlier runs keep their GPUs even after a re-run or restart, the new ones run out of memory,
and any timings measured meanwhile are wrong. The grid cell now refuses to start if the GPUs are busy.

**If the session ends early**, nothing is lost: every problem is written as it finishes. Add this
notebook's output (or the zip) as an input to a new session and run again. It copies the finished
problems back and resumes. Don't feed it the T3 sanity results; start T6 from empty.
""")

code(r"""
# ---- configuration ---------------------------------------------------------------
REPO_URL = ""             # "" -> use a .bundle under /kaggle/input
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_DEV, DEV_OFFSET = 300, 5000
BUDGET_HOURS = 11.0       # stop launching/kill jobs after this; Kaggle's hard limit is 12 h
WORK = "/kaggle/working/unspoken"

import glob, json, os, shutil, subprocess, sys, threading, time
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

md("## 0. Code, environment, tests")

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
# Resume: copy finished T6 problems from a previous session's output (never overwrite).
RUN_DIR = f"{WORK}/results/{MODEL.split('/')[-1]}/train"
os.makedirs(RUN_DIR, exist_ok=True)
prev = [p for p in glob.glob("/kaggle/input/**/results/*/train/*", recursive=True)
        if os.path.basename(os.path.dirname(os.path.dirname(p))) == MODEL.split("/")[-1]]
for p in prev:
    dst = os.path.join(RUN_DIR, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst)
done = {os.path.basename(p): sum(1 for l in open(p) if l.strip())
        for p in glob.glob(f"{RUN_DIR}/*.jsonl")}
print("resumed:" if done else "starting fresh (no previous T6 results in /kaggle/input)", done or "")

from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download(MODEL); load_dataset("openai/gsm8k", "main")
print(f"setup done in {(time.time() - T0) / 60:.1f} min")
""")

md("""
## 1. Run the grid on both GPUs

`scripts.plan` assigns runs to the GPUs from the committed T3 timings. Within each GPU the runs
that decide `m` go first (text MAS and m = 1, 2, 3), so if the session is cut short the important
rows exist. Progress is logged per run in `logs/`.
""")

code(r"""
sys.path.insert(0, WORK)
from scripts import plan

stats = {"3B": plan.summarize_csv(f"{WORK}/spec/kaggle/t3_summary.csv", MODEL.split("/")[-1]), "1.5B": {}}
jobs = [j for j in plan.build_grid(10, N_DEV, 0, DEV_OFFSET) if j["phase"] == "T6 dev"]
for j in jobs:
    sec, _ = plan.estimate(stats["3B"], j["tag"])
    j["hours"] = (sec or 15.0) * j["n"] / 3600
lanes = plan.schedule(jobs, 2)
PRIORITY = ["text_mas", "latent_mas_m1", "latent_mas_m2", "latent_mas_m3", "single_cot",
            "latent_mas_m0_drop", "latent_mas_m0", "latent_mas_m5", "latent_mas_m10",
            "latent_mas_m20", "latent_mas_m40", "latent_mas_m10_raw", "latent_mas_m10_norm_only"]
LANES = {g: sorted((j for j in jobs if j["lane"] == g), key=lambda j: PRIORITY.index(j["tag"]))
         for g in range(2)}
for g, js in LANES.items():
    print(f"GPU {g} (~{lanes[g]:.1f} h): " + ", ".join(j["tag"] for j in js))

# Exclusive GPUs, or nothing. Runs left over from an earlier execution of this cell keep running
# after a re-run or kernel restart; sharing a T4 with them runs out of memory and, worse, inflates
# latency (one of the four metrics). Seen on 2026-10-23: three 3B processes per GPU, all OOM.
alive = [p for p in globals().get("_T6_PROCS", []) if p.poll() is None]
assert not alive, (f"{len(alive)} run(s) from an earlier execution of this cell are still going. "
                   "Let them finish, or stop them with: !pkill -f scripts.run")
busy = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
                      shell=True, capture_output=True, text=True).stdout.strip()
assert not busy, ("The GPUs are already in use, so timings would be wrong:\n" + busy +
                  "\nStop those processes (e.g. !pkill -f scripts.run) or restart the session, then rerun.")
_T6_PROCS = []

os.makedirs(f"{WORK}/logs", exist_ok=True)
status = {}

def lane(gpu, jobs):
    for i, j in enumerate(jobs):
        cmd = plan.command(j, {"3B": MODEL}).split(" ", 1)[1]   # drop CUDA_VISIBLE_DEVICES=, set below
        if hours_left() < 0.05:
            status[j["tag"]] = "skipped (budget)"; print(f"[GPU {gpu}] SKIP {j['tag']} (budget)", flush=True)
            continue
        t, log = time.time(), f"{WORK}/logs/t6_gpu{gpu}_{i:02d}_{j['tag']}.log"
        with open(log, "w") as f:
            f.write(f"# {cmd}\n"); f.flush()
            p = subprocess.Popen(cmd, shell=True, cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            _T6_PROCS.append(p)
            try:
                rc = p.wait(timeout=max(60, hours_left() * 3600))
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(); rc = "stopped (budget, resumable)"
        status[j["tag"]] = rc
        print(f"[GPU {gpu}] {'ok ' if rc == 0 else rc} {(time.time() - t) / 60:6.1f} min  {j['tag']}", flush=True)

threads = [threading.Thread(target=lane, args=(g, js)) for g, js in LANES.items()]
[t.start() for t in threads]; [t.join() for t in threads]
print(f"\ngrid finished or budget reached after {(time.time() - T0) / 3600:.2f} h")
failed = {k: v for k, v in status.items() if v not in (0,) and not str(v).startswith(("skipped", "stopped"))}
if failed:
    print("\n" + "!" * 70 + f"\nRUNS FAILED (see logs/; the tail of each is shown below): {failed}\n" + "!" * 70)
    for path in sorted(glob.glob(f"{WORK}/logs/t6_*.log")):
        if any(path.endswith(f"_{tag}.log") for tag in failed):
            print(f"\n--- {os.path.basename(path)}\n" + "".join(open(path).readlines()[-3:]))
""")

md("""
## 2. Results on the dev slice

The `boxed` column is the quickest coherence signal (HANDOFF issue 4). Then the pre-registered rule
chooses `m`; it refuses to choose until every latent run has the full 300 problems.
""")

code(r"""
# check=False: with no results (e.g. every run failed) we still want to reach packaging,
# so the logs get downloaded.
r = sh(f"python -m scripts.analyze --dir {RUN_DIR} --baseline text_mas_seed0", check=False,
       capture_output=True)
print(r.stdout, r.stderr); open(f"{WORK}/t6_analysis.txt", "w").write(r.stdout + r.stderr)
sel = sh(f"python -m scripts.select_m --dir {RUN_DIR}", check=False, capture_output=True)
print(sel.stdout, sel.stderr); open(f"{WORK}/t6_select_m.txt", "w").write(sel.stdout + sel.stderr)
import re
m_chosen = int(re.search(r"chosen m = (\d+)", sel.stdout).group(1)) if sel.returncode == 0 else None
print("m chosen by the pre-registered rule:", m_chosen)
""")

code(r"""
# Read a few answers per run: the first 2 problems, 250 characters each.
for path in sorted(glob.glob(f"{RUN_DIR}/*.jsonl")):
    print(f"\n==================== {os.path.basename(path)}")
    for line in list(open(path))[:2]:
        rec = json.loads(line)
        print(f"--- {rec['id']} gold={rec['gold']} pred={rec['pred']} correct={rec['correct']}")
        print(rec["output"][:250].strip())
""")

md("## 3. Plan T7 with the chosen m, and package")

code(r"""
if m_chosen is not None:
    r = sh(f"python -m scripts.plan --summary_csv spec/kaggle/t3_summary.csv --phase final "
           f"--final_m {m_chosen} --emit_commands", capture_output=True)
    print(r.stdout); open(f"{WORK}/t7_plan.md", "w").write(r.stdout)
else:
    print("m not chosen yet (dev runs incomplete): rerun this notebook with this output as input.")
    open(f"{WORK}/t7_plan.md", "w").write("m not chosen yet\n")

out = "/kaggle/working/unspoken_kaggle_T6.zip"
sh(f"rm -f {out} && zip -qr {out} results logs t6_analysis.txt t6_select_m.txt t7_plan.md")
print(f"wrote {out} ({os.path.getsize(out) / 2**20:.1f} MB)\n")
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=WORK, text=True).strip()
if failed:
    # No change-log line for a broken run: it must not read like a result.
    raise RuntimeError(f"T6 DID NOT COMPLETE: runs failed {failed}. The zip holds the logs for "
                       "debugging; do not use its results (and do not resume from them).")
print("---- paste into spec/HANDOFF.md (Change log) ----")
print(f"- {time.strftime('%Y-%m-%d')}: T6 dev sweep on Kaggle 2x T4 ({MODEL.split('/')[-1]}, train offset {DEV_OFFSET}, "
      f"n={N_DEV}, commit {commit}); {(time.time() - T0) / 3600:.1f} h. Pre-registered rule chose "
      f"m = {m_chosen}. Table: t6_analysis.txt. Run status: {status}.")
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
