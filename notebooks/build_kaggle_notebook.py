"""Generate notebooks/kaggle_T1_T3_T5.ipynb (plain JSON, no nbformat needed).

Edit the cells here, not the .ipynb, then regenerate from the repo root:
  python notebooks/build_kaggle_notebook.py notebooks/kaggle_T1_T3_T5.ipynb
"""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})

md(r"""
# Unspoken on Kaggle: T1 (environment), T3 (sanity runs), T5 (run plan)

One GPU session that closes three HANDOFF tasks:

| Task | What this notebook does | Done when |
|---|---|---|
| **T1** | Installs, runs the full test suite on the T4, records versions, writes `requirements.lock` | `pytest` fully green on Kaggle |
| **T3** | 5-problem sanity runs of every main method on 3B (plus timing runs on 1.5B), on both GPUs in parallel | outputs read by hand, `\boxed{}` present, `tie_word_embeddings` and latency recorded |
| **T5** | Turns the measured latencies into a schedule for the T6 dev grid and T7 final runs | runtime table split across 2 GPUs and the weekly quota |

**Before you run**
1. *Settings → Accelerator:* **GPU T4 x2**. *Settings → Internet:* **On** (models and GSM8K download from the Hugging Face Hub).
2. Get the code here, either way:
   * **GitHub:** set `REPO_URL` in the next cell (private repo: add a `GITHUB_TOKEN` Kaggle secret).
   * **Kaggle dataset (no GitHub needed):** on your machine run `git bundle create unspoken.bundle --all`, upload the file as a private dataset and *Add Input* it here. The bundle keeps the git history, so every result records its commit.
3. *Run All.* Expect about 45–75 min, mostly model downloads and text MAS. The total is printed at the end.

**Data hygiene.** Sanity runs use GSM8K **train** from offset 5000 (`--split train --offset 5000`), never the test set: reading test outputs while deciding settings would leak (HANDOFF issue 5).

At the end, download **`unspoken_kaggle_T1_T3_T5.zip`** from the Output panel.
""")

code(r"""
# ---- configuration ---------------------------------------------------------------
REPO_URL = ""            # e.g. "https://github.com/<you>/unspoken.git"; leave "" to use a .bundle input
N_SANITY = 5             # problems per sanity run (T3 asks for 5)
DEV_OFFSET = 5000        # train offset for dev/sanity runs; must be >= 2000 (adapter slice)
MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
RUN_SMALL = True         # also time the 1.5B rows (needed for the T7 estimate)
WORK = "/kaggle/working/unspoken"

import glob, json, os, subprocess, sys, time
T0 = time.time()

def sh(cmd, check=True, **kw):
    # Run a shell command from the repo root, echo it, stream output into the notebook.
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=kw.pop("cwd", WORK if os.path.isdir(WORK) else None),
                       text=True, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r
""")

md("## 0. Get the code and check the GPUs")

code(r"""
if os.path.isdir(WORK):
    print("repo already present:", WORK)
elif REPO_URL:
    url = REPO_URL
    try:  # private GitHub repo: Kaggle secret GITHUB_TOKEN
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        url = url.replace("https://", f"https://{token}@")
    except Exception:
        pass
    sh(f"git clone --quiet {url} {WORK}", cwd="/kaggle/working")
else:
    bundles = glob.glob("/kaggle/input/**/*.bundle", recursive=True)
    assert bundles, "No REPO_URL and no .bundle under /kaggle/input: see 'Before you run'."
    sh(f"git clone --quiet {bundles[0]} {WORK}", cwd="/kaggle/working")
    sh("git checkout --quiet main")

sh("git log -1 --format='commit %h  %s  (%ad)' --date=short")
sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv")
""")

md("""
## 1. T1: install, test, pin versions

Kaggle's preinstalled torch already satisfies `requirements.txt`, so pip only adds what is missing.
`gradio` is skipped: it is only needed for the demo.
""")

code(r"""
sh("grep -v '^gradio' requirements.txt > /tmp/req.txt && python -m pip install -q -r /tmp/req.txt")
r = sh("python -m pytest -v -p no:cacheprovider", check=False)
assert r.returncode == 0, "Tests failed: stop here and fix them before any experiment (HANDOFF T1)."
print("\nT1: test suite green on Kaggle")
""")

code(r"""
import platform, torch, transformers
sh("python -m pip freeze > requirements.lock")
env = {
    "date": time.strftime("%Y-%m-%d"),
    "git_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=WORK, text=True).strip(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
json.dump(env, open(f"{WORK}/env_report.json", "w"), indent=2)
print(json.dumps(env, indent=2))
assert env["gpus"], "No GPU visible: set Accelerator to GPU T4 x2."
""")

md("""
## 2. T3: sanity runs (dev slice, both GPUs in parallel)

Models and GSM8K are downloaded first, so the timings exclude download time and the two GPU
lanes don't race on the same files.
""")

code(r"""
from huggingface_hub import snapshot_download
from datasets import load_dataset
for m in [MODEL] + ([MODEL_SMALL] if RUN_SMALL else []):
    snapshot_download(m)
load_dataset("openai/gsm8k", "main")
print(f"downloads done ({(time.time() - T0) / 60:.1f} min so far)")
""")

code(r"""
import threading
S = f"--split train --offset {DEV_OFFSET} --n {N_SANITY}"
# Lane 0 gets the slow text-MAS runs; lane 1 gets everything else. Main methods first (T3),
# then the latent-step diagnostics for HANDOFF issue 4 (m=1, m=5), then the 1.5B timing rows.
LANES = {
    0: [f"--method text_mas --model {MODEL} {S}"]
       + ([f"--method text_mas --model {MODEL_SMALL} {S}"] if RUN_SMALL else []),
    1: [f"--method single_cot --model {MODEL} {S}",
        f"--method latent_mas --latent_steps 10 --model {MODEL} {S}",
        f"--method latent_mas --latent_steps 5 --model {MODEL} {S}",
        f"--method latent_mas --latent_steps 1 --model {MODEL} {S}",
        f"--method latent_mas --latent_steps 0 --m0_cache drop --model {MODEL} {S}"]
       + ([f"--method single_cot --model {MODEL_SMALL} {S}",
           f"--method latent_mas --latent_steps 10 --model {MODEL_SMALL} {S}"] if RUN_SMALL else []),
}
os.makedirs(f"{WORK}/logs", exist_ok=True)
status = {}

def lane(gpu, jobs):
    for i, args in enumerate(jobs):
        log = f"{WORK}/logs/gpu{gpu}_{i:02d}.log"
        t = time.time()
        with open(log, "w") as f:
            f.write(f"# python -m scripts.run {args}\n"); f.flush()
            rc = subprocess.run(f"python -m scripts.run {args}", shell=True, cwd=WORK, stdout=f,
                                stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
        status[args] = rc
        print(f"[GPU {gpu}] {'ok ' if rc == 0 else 'FAIL'} {time.time() - t:6.0f}s  {args}", flush=True)

threads = [threading.Thread(target=lane, args=(g, j)) for g, j in LANES.items()]
[t.start() for t in threads]; [t.join() for t in threads]
failed = [a for a, rc in status.items() if rc != 0]
assert not failed, f"Runs failed, see logs/: {failed}"
print(f"\nall sanity runs done ({(time.time() - T0) / 60:.1f} min so far)")
""")

md("""
### T3 checks

Automatic checks first, then **read the outputs yourself**. The "junk" flag is only a heuristic:
the share of CJK characters in an English answer. Coherence is judged by a person.
""")

code(r"""
import re
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
def load(p): return [json.loads(l) for l in open(p) if l.strip()]

ties = sorted({l.strip() for p in glob.glob(f"{WORK}/logs/*.log") for l in open(p) if "tie_word_embeddings" in l})
print("recorded:", ties)

rows = []
for path in sorted(glob.glob(f"{WORK}/results/*/train/*.jsonl")):
    rs = load(path)
    model = path.split("/results/")[1].split("/")[0]
    tag = os.path.basename(path)[:-6]
    junk = sum(len(CJK.findall(r["output"])) / max(1, len(r["output"])) > 0.05 for r in rs)
    cfg = json.load(open(path.replace(".jsonl", ".config.json")))
    agents = [n for r in rs for n in r["per_agent_tokens"][:-1]]
    trunc = f"{sum(n >= cfg['agent_max_tokens'] for n in agents) / len(agents):.0%}" if tag.startswith("text_mas") else ""
    rows.append(dict(model=model, run=tag, n=len(rs), acc=sum(r["correct"] for r in rs) / len(rs),
                     boxed=sum("\\boxed{" in r["output"] for r in rs) / len(rs), junk=f"{junk}/{len(rs)}",
                     out_tok=sum(r["output_tokens"] for r in rs) / len(rs),
                     fwd=sum(r["forward_passes"] for r in rs) / len(rs),
                     sec=sum(r["latency_s"] for r in rs) / len(rs),
                     peak_mb=max(r["peak_mem_mb"] for r in rs), agents_at_cap=trunc))

import pandas as pd
table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
""")

code(r"""
# T3 'done when' checklist (automatic part). Coherence of latent_mas still needs a human read below.
main3b = table[(table.model == MODEL.split("/")[-1])]
def row(run): return main3b[main3b.run == run].iloc[0]
checks = {
    "baselines box their answers (single_cot, text_mas)":
        all(row(r).boxed == 1.0 for r in ["single_cot_seed0", "text_mas_seed0"]),
    "latent_mas m10 boxes its answers": row("latent_mas_m10_seed0").boxed == 1.0,
    "latent_mas m10 shows no junk text (heuristic)": row("latent_mas_m10_seed0").junk.startswith("0/"),
    "tie_word_embeddings recorded": bool(ties),
    "per-problem latency recorded": bool(len(table)) and table.sec.gt(0).all(),
}
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print("\nA FAIL on latent_mas m10 reproduces HANDOFF issue 4 (seen on the Mac). It is a finding, not a "
      "notebook error: record it and compare the m=1 / m=5 rows.")
""")

code(r"""
# Read the outputs: the first 300 characters of every judger answer.
for path in sorted(glob.glob(f"{WORK}/results/{MODEL.split('/')[-1]}/train/*.jsonl")):
    print(f"\n==================== {os.path.basename(path)}")
    for r in load(path):
        print(f"--- {r['id']}  gold={r['gold']} pred={r['pred']} correct={r['correct']}")
        print(r["output"][:300].strip())
""")

md("""
## 3. T5: run plan

`scripts.plan` reads the timings above, estimates the runs it hasn't measured (marked `~`), and
packs the T6 dev grid and the T7 final runs onto the two GPUs. `--final_m 10` is a placeholder
until T6 chooses `m`. Check your real weekly GPU quota in Kaggle settings and change `--quota_hours`
if it differs.
""")

code(r"""
small = f"--dir_small results/{MODEL_SMALL.split('/')[-1]}/train" if RUN_SMALL else ""
r = sh(f"python -m scripts.plan --dir results/{MODEL.split('/')[-1]}/train {small} "
       f"--model {MODEL} --model_small {MODEL_SMALL} --final_m 10 --gpus 2 --emit_commands",
       capture_output=True)
print(r.stdout)
open(f"{WORK}/plan.md", "w").write(r.stdout)
""")

md("""
## 4. Package everything for download

The zip contains `results/` (copy into your local `results/`; append-only), `requirements.lock`
(commit it: T1), `env_report.json`, `plan.md`, the T3 summary table and all logs.
Then update `spec/HANDOFF.md` with the snippet printed below.
""")

code(r"""
table.to_csv(f"{WORK}/t3_summary.csv", index=False)
out = "/kaggle/working/unspoken_kaggle_T1_T3_T5.zip"
sh(f"rm -f {out} && zip -qr {out} results requirements.lock env_report.json plan.md t3_summary.csv logs")
print(f"wrote {out}  ({os.path.getsize(out) / 2**20:.1f} MB)\n")

t = {r["run"].replace("_seed0", ""): r["sec"] for r in rows if r["model"] == MODEL.split("/")[-1]}
print("---- paste into spec/HANDOFF.md (Change log) ----")
print(f"- {env['date']}: T1 on Kaggle ({', '.join(env['gpus'])}): pytest green; python {env['python']}, "
      f"torch {env['torch']} (CUDA {env['cuda']}), transformers {env['transformers']}; requirements.lock "
      f"committed. Commit {env['git_commit']}.")
print(f"- {env['date']}: T3 sanity (train offset {DEV_OFFSET}, n={N_SANITY}, {MODEL.split('/')[-1]}, T4 fp16): "
      + ", ".join(f"{k} {v:.0f} s" for k, v in t.items()) + f" per problem; {'; '.join(ties)}.")
print(f"\nnotebook total: {(time.time() - T0) / 60:.0f} min")
""")

nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
