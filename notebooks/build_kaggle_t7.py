"""Generate notebooks/kaggle_T7_final.ipynb (plain JSON, no nbformat needed).

Edit the cells here, not the .ipynb, then regenerate from the repo root:
  python notebooks/build_kaggle_t7.py notebooks/kaggle_T7_final.ipynb
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
# Unspoken on Kaggle: T7 final runs (test set)

The runs that test H1: the **full GSM8K test set (1,319 problems)**, for Qwen2.5-3B (primary) and
Qwen2.5-1.5B (scale check), with `single_cot`, `text_mas` and `latent_mas` at the `m` chosen by the
T6 dev sweep.

**This touches the test set, so run it only after T6 has chosen `m`.** Set `FINAL_M` below. If you
attach the T6 output, the notebook checks that `FINAL_M` matches the `m` the pre-registered rule chose.

Expected: **~23–28 h** on 2x T4, so **2–3 sessions** of up to 11 h each.

**Each session**
1. Settings: **GPU T4 x2**, **Internet on**. Set `REPO_URL` (or attach the code bundle).
2. **Attach the previous session's output** (not needed for the first session). Finished problems
   are copied back and skipped.
3. **Save Version → Save & Run All (Commit).** Run it once per session; the GPU guard refuses to
   start if anything else is using the GPUs.
4. Download `unspoken_kaggle_T7.zip`. When every run reaches 1,319, the notebook prints the
   change-log line and the H1 check.

The runs advance in **rounds of 200 problems**: every method gets problems 1–200, then 201–400, and
so on. If a session ends early, all methods have covered the same problems, so the interim tables
compare like with like.
""")

code(r"""
# ---- configuration ---------------------------------------------------------------
REPO_URL = "https://github.com/SamarthPyati/Unspoken.git"   # "" -> use a .bundle under /kaggle/input
FINAL_M = None            # REQUIRED: the m chosen by T6 (scripts/select_m.py), e.g. 1
M_SMALL = None            # m for the 1.5B rows; None = same as FINAL_M (team decision, HANDOFF)
ALLOW_M_OVERRIDE = False  # True only if the team logged a reason to differ from T6's choice
MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
N_TEST, CHUNK = 1319, 200
BUDGET_HOURS = 11.0
WORK = "/kaggle/working/unspoken"

import glob, json, os, re, shutil, subprocess, sys, threading, time
T0 = time.time()
assert FINAL_M is not None, "Set FINAL_M to the m chosen by the T6 dev sweep before touching the test set."
M_SMALL = FINAL_M if M_SMALL is None else M_SMALL

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

code(r"""
# Cross-check FINAL_M against the T6 output, if attached.
sel = glob.glob("/kaggle/input/**/t6_select_m.txt", recursive=True)
if sel:
    found = re.search(r"chosen m = (\d+)", open(sel[0]).read())
    t6_m = int(found.group(1)) if found else None
    print(f"T6 rule chose m = {t6_m}  ({sel[0]})")
    assert t6_m == FINAL_M or ALLOW_M_OVERRIDE, (
        f"FINAL_M={FINAL_M} but T6 chose m={t6_m}. Use T6's m, or set ALLOW_M_OVERRIDE=True "
        "after logging the reason in spec/HANDOFF.md.")
else:
    print("No t6_select_m.txt attached: make sure FINAL_M is the m T6 chose.")
print(f"final settings: 3B latent m={FINAL_M}, 1.5B latent m={M_SMALL}")
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
# Resume: copy finished TEST-split problems from a previous T7 session's output (never overwrite).
# Only results/<model>/*.jsonl (test); dev results live in results/<model>/train/ and are ignored.
SHORT = {MODEL: MODEL.split("/")[-1], MODEL_SMALL: MODEL_SMALL.split("/")[-1]}
for m in SHORT.values():
    os.makedirs(f"{WORK}/results/{m}", exist_ok=True)
    for p in glob.glob(f"/kaggle/input/**/results/{m}/*", recursive=True):
        dst = f"{WORK}/results/{m}/{os.path.basename(p)}"
        if os.path.isfile(p) and not os.path.exists(dst):
            shutil.copy(p, dst)

def progress():
    return {f"{m}/{os.path.basename(p)[:-6]}": sum(1 for l in open(p) if l.strip())
            for m in SHORT.values() for p in glob.glob(f"{WORK}/results/{m}/*.jsonl")}
print("resumed:" if progress() else "starting fresh", progress() or "")

from huggingface_hub import snapshot_download
from datasets import load_dataset
for m in SHORT:
    snapshot_download(m)
load_dataset("openai/gsm8k", "main")
print(f"setup done in {(time.time() - T0) / 60:.1f} min")
""")

md("""
## 1. Run, in rounds of 200 problems, on both GPUs

`scripts.plan` assigns the six runs to the two GPUs from the committed T3 timings. Each GPU then
loops: every run on that GPU advances to problem 200, then 400, and so on up to 1,319. `scripts.run`
skips problems already written, so each step only adds the new ones.
""")

code(r"""
sys.path.insert(0, WORK)
from scripts import plan

stats = {"3B": plan.summarize_csv(f"{WORK}/spec/kaggle/t3_summary.csv", SHORT[MODEL]),
         "1.5B": plan.summarize_csv(f"{WORK}/spec/kaggle/t3_summary.csv", SHORT[MODEL_SMALL])}
jobs = [j for j in plan.build_grid(FINAL_M, 0, N_TEST, 0) if j["phase"] == "T7 final"]
for j in jobs:
    if j["size"] == "1.5B" and j["tag"].startswith("latent_mas"):
        j["tag"] = f"latent_mas_m{M_SMALL}"
    sec, _ = plan.estimate(stats[j["size"]], j["tag"])
    j["hours"] = (sec or 30.0) * j["n"] / 3600
lanes = plan.schedule(jobs, 2)
LANES = {g: [j for j in jobs if j["lane"] == g] for g in range(2)}
for g, js in LANES.items():
    print(f"GPU {g} (~{lanes[g]:.1f} h total): " + ", ".join(f"{j['size']} {j['tag']}" for j in js))

# Exclusive GPUs, or nothing (see the T6 attempt-1 incident in HANDOFF).
alive = [p for p in globals().get("_T7_PROCS", []) if p.poll() is None]
assert not alive, f"{len(alive)} run(s) from an earlier execution are still going (!pkill -f scripts.run)"
busy = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
                      shell=True, capture_output=True, text=True).stdout.strip()
assert not busy, "GPUs already in use (stop them with !pkill -f scripts.run, or restart):\n" + busy
_T7_PROCS = []

os.makedirs(f"{WORK}/logs", exist_ok=True)
status = {}
MODELS = {"3B": MODEL, "1.5B": MODEL_SMALL}

def lane(gpu, jobs):
    for target in list(range(CHUNK, N_TEST, CHUNK)) + [N_TEST]:
        for j in jobs:
            key = f"{j['size']} {j['tag']}"
            if status.get(key) not in (None, 0):
                continue                     # this run failed or stopped earlier: don't retry blindly
            if hours_left() < 0.05:
                status[key] = "stopped (budget, resumable)"; continue
            out = f"{WORK}/results/{SHORT[MODELS[j['size']]]}/{j['tag']}_seed0.jsonl"
            if os.path.exists(out) and sum(1 for l in open(out) if l.strip()) >= target:
                status.setdefault(key, 0); continue   # round already done (resumed): skip the model load
            cmd = plan.command({**j, "n": target}, MODELS).split(" ", 1)[1]
            log = f"{WORK}/logs/t7_gpu{gpu}_{j['size']}_{j['tag']}.log"
            with open(log, "a") as f:
                f.write(f"\n# {cmd}\n"); f.flush()
                p = subprocess.Popen(cmd, shell=True, cwd=WORK, stdout=f, stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
                _T7_PROCS.append(p)
                try:
                    rc = p.wait(timeout=max(60, hours_left() * 3600))
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(); rc = "stopped (budget, resumable)"
            status[key] = rc
        print(f"[GPU {gpu}] round to {target:4d} done at {(time.time() - T0) / 3600:5.2f} h  "
              + ", ".join(f"{k}: {v}" for k, v in status.items() if k in {f"{j['size']} {j['tag']}" for j in jobs}),
              flush=True)

threads = [threading.Thread(target=lane, args=(g, js)) for g, js in LANES.items()]
[t.start() for t in threads]; [t.join() for t in threads]

prog = progress()
complete = len(prog) == len(jobs) and all(v >= N_TEST for v in prog.values())
failed = {k: v for k, v in status.items() if v != 0 and not str(v).startswith("stopped")}
print(f"\nafter {(time.time() - T0) / 3600:.2f} h:", prog)
if failed:
    print("\n" + "!" * 70 + f"\nRUNS FAILED (see logs/): {failed}\n" + "!" * 70)
    for path in sorted(glob.glob(f"{WORK}/logs/t7_*.log")):
        if any(path.endswith(f"_{k.replace(' ', '_')}.log") for k in failed):
            print(f"\n--- {os.path.basename(path)}\n" + "".join(open(path).readlines()[-3:]))
""")

md("""
## 2. Tables (interim until every run reaches 1,319)

`analyze` compares only problems that every run in a folder has finished, so interim tables are fair.
""")

code(r"""
tables = {}
for m in SHORT.values():
    r = sh(f"python -m scripts.analyze --dir results/{m} --baseline text_mas_seed0", check=False,
           capture_output=True)
    tables[m] = r.stdout + r.stderr
    print(f"==================== {m}\n{tables[m]}")
open(f"{WORK}/t7_analysis.txt", "w").write("\n".join(f"== {m}\n{t}" for m, t in tables.items()))
""")

md("""
### H1 check (3B, latent vs text MAS)

H1 (pre-registered): accuracy **within 3 points** of text MAS **and at least 50% fewer output
tokens**. The pre-registration does not say whether "within 3 points" applies to the point
estimate or to the lower end of the 95% CI, so both are printed. The team should fix which one
counts before reading this cell on the complete data (HANDOFF).
""")

code(r"""
sys.path.insert(0, WORK)
from src.metrics import paired_diff_ci, relative_reduction

def load(p): return {r["id"]: r for r in map(json.loads, (l for l in open(p) if l.strip()))}
d3 = f"{WORK}/results/{SHORT[MODEL]}"
try:
    text, lat = load(f"{d3}/text_mas_seed0.jsonl"), load(f"{d3}/latent_mas_m{FINAL_M}_seed0.jsonl")
    ids = sorted(set(text) & set(lat))
    assert ids, "no shared problems yet"
except (FileNotFoundError, AssertionError) as e:   # e.g. a run failed: still reach packaging
    ids = []
    h1 = f"H1: not computable yet ({e})"
    print(h1); open(f"{WORK}/t7_h1.txt", "w").write(h1 + "\n")
if ids:
    d = paired_diff_ci([lat[i]["correct"] for i in ids], [text[i]["correct"] for i in ids])
    saved = relative_reduction([text[i]["output_tokens"] for i in ids], [lat[i]["output_tokens"] for i in ids])
    fwd = relative_reduction([text[i]["forward_passes"] for i in ids], [lat[i]["forward_passes"] for i in ids])
    h1 = (f"H1 on {len(ids)}/{N_TEST} problems{'' if len(ids) >= N_TEST else ' (INTERIM)'}: "
          f"accuracy difference latent - text = {d['mean']:+.3f} [95% CI {d['lo']:+.3f}, {d['hi']:+.3f}]; "
          f"output tokens saved = {saved:.0%}; forward passes saved = {fwd:.0%}.\n"
          f"  accuracy, point estimate >= -0.03: {d['mean'] >= -0.03}\n"
          f"  accuracy, CI lower bound >= -0.03: {d['lo'] >= -0.03}\n"
          f"  tokens saved >= 50%:              {saved >= 0.5}")
    print(h1); open(f"{WORK}/t7_h1.txt", "w").write(h1 + "\n")
""")

md("## 3. Package")

code(r"""
out = "/kaggle/working/unspoken_kaggle_T7.zip"
sh(f"rm -f {out} && zip -qr {out} results logs t7_analysis.txt t7_h1.txt")
print(f"wrote {out} ({os.path.getsize(out) / 2**20:.1f} MB)\n")
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=WORK, text=True).strip()
if failed:
    raise RuntimeError(f"T7 DID NOT COMPLETE: runs failed {failed}. The zip holds the logs; "
                       "do not resume from failed runs before reading them.")
if not complete:
    print(f"T7 PARTIAL after this session: {prog}\n-> attach this notebook's output to the next "
          "session and run again; it resumes where it stopped.")
else:
    print("---- paste into spec/HANDOFF.md (Change log) ----")
    print(f"- {time.strftime('%Y-%m-%d')}: T7 final runs complete on Kaggle 2x T4 (test, n={N_TEST}, "
          f"commit {commit}): 3B and 1.5B x single_cot, text_mas, latent_mas (3B m={FINAL_M}, "
          f"1.5B m={M_SMALL}). Tables: t7_analysis.txt. {h1.splitlines()[0]}")
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
