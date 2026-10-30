# Unspoken (`latent-proto`)

A small, testable prototype of one claim from our Research Methodology project:

> **H1 (pre-registered).** On GSM8K, a multi-agent pipeline that exchanges latent working
> memory (the KV cache) reaches accuracy within 3 points of the same pipeline exchanging text,
> while generating at least 50% fewer output tokens.

The method re-implements LatentMAS (Zou et al., arXiv 2511.20639,
[Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)) from scratch at small scale, so
the team understands and controls every line. The cross-model adapter (`src/adapter.py`) is our
own extension.

## Why save tokens?

Generating text is the expensive part of running an LLM. Every output token is one full,
**sequential** forward pass through the model. Prompt tokens are cheaper because they are processed
in parallel, but they still cost compute and memory. So output tokens drive:

- **Latency**: the answer cannot arrive faster than the tokens can be generated one by one.
- **Compute, energy and money**: hosted APIs bill per token; self-hosted GPUs pay in time.
- **Context**: everything written has to be held in the KV cache and read again later.

**Multi-agent systems make this worse.** A planner → critic → refiner → judger pipeline doesn't
just write one answer. Every agent writes a message in full, and the next agent has to read the
whole transcript again. Most of those tokens are agents talking to each other, not the answer the
user sees.

**The idea we test:** let the agents pass their *working memory* (the KV cache) instead of
writing to each other. The intermediate agents "think" in hidden-state space, and only the final
judger writes text. If accuracy holds, all of the inter-agent writing and re-reading disappears.
H1 makes that concrete: at least 50% fewer output tokens, at most 3 points of accuracy lost.

## Results (Qwen2.5-3B, GSM8K, Kaggle T4)

**Short version: latent communication cut output tokens, forward passes and latency by about
75%, but cost 7.3 accuracy points. H1 is half met.** These are real runs, not sanity checks, but on
a reduced plan: under the deadline we ran a 100-problem dev sweep and the first 300 test problems
(the plan and its deviations were logged in [`spec/HANDOFF.md`](spec/HANDOFF.md) before running).

**Choosing m (dev sweep, 100 GSM8K train problems 5,000–5,099).** The pre-registered rule
(`scripts/select_m.py`: highest dev accuracy, ties to the smaller m) picked **m = 1**:

| Method | Dev accuracy [95% CI] | Answers ending in `\boxed{}` |
|---|---|---|
| Text MAS | 94% [89, 98] | 99% |
| Single CoT | 90% [84, 95] | 94% |
| **Latent MAS, m = 1** | **82% [74, 89]** | 94% |
| Latent MAS, m = 2 | 69% [60, 78] | 74% |
| Latent MAS, m = 3 | 42% [32, 52] | 29% |

**Test subset (GSM8K test problems 0–299, used once, m = 1).** Means per problem:

| Method | Accuracy [95% CI] | Output tokens | Forward passes | Latency |
|---|---|---|---|---|
| Text MAS (baseline) | 84.7% [80.7, 88.7] | 882 | 886 | 44.9 s |
| **Latent MAS, m = 1** | **77.3% [72.3, 82.0]** | **219 (−75%)** | **226 (−75%)** | **11.3 s (−75%)** |
| Single-agent CoT | 84.7% [80.3, 88.7] | 328 | 329 | 17.0 s |

**H1 on the test subset:**

| Criterion | Threshold | Measured | Met? |
|---|---|---|---|
| Output tokens saved | ≥ 50% | 75% | **yes** |
| Accuracy, latent − text | ≥ −3 points | −7.3 [−12.0, −2.3], McNemar p = 0.005 | **no** (point estimate and CI) |

Text MAS alone was right on 39 problems, latent MAS alone on 17. Peak GPU memory was about the same
(6.1 vs 6.4 GB).

### What the numbers show

1. **Most multi-agent tokens are agents talking to each other.** In text MAS, 72% of output tokens
   (635 of 882: planner 158, critic 253, refiner 224) are inter-agent messages; the judger's answer
   is 246. Each agent also re-reads the transcript: 1,662 prompt tokens per problem, against 454 for
   latent MAS.
2. **Latent MAS removes that writing, so the saving is real in compute too.** Only the judger
   writes text; the three latent steps (m = 1) replace about 635 generated tokens. Forward passes
   and latency fall by the same 75% as tokens.
3. **The judger still reasons in text.** The method removes inter-agent *messages*, not the final
   written reasoning. Compressing a single model's own chain of thought into latent steps is a
   different line of work (e.g. Coconut) and out of scope here.
4. **On small models, more latent steps break the answer.** Dev accuracy falls from 82% (m = 1) to
   42% (m = 3), and the share of answers that even end in `\boxed{}` falls with it. Qwen3-4B (the
   reference's model family) stays coherent at m = 10 and fp32 fails like fp16, so this is a
   small-model effect, not a bug. At m = 1 the latent errors are coherent solutions with a wrong
   number, not garbled text.
5. **Two caveats.** 53% of text-MAS agent messages hit the 256-token cap, which handicaps the text
   baseline. And single-agent CoT matched text MAS at about a third of its tokens, so on GSM8K at
   this scale the multi-agent setup itself buys no accuracy.

**Trade-offs:**
- **Interpretability:** you can read a critic's text, but you can't read a latent step.
- **Compute is not free:** each latent step is a full forward pass; it just writes no token. That
  is why this project always reports forward passes and latency next to tokens.
- **Same model only:** a KV cache can't move between different models; that is what our adapter
  extension explores (smoke-tested only: held-out cosine 0.676).

**Not yet run** (pre-registered, left for later): all 1,319 test problems, the 1.5B repeat, the
heterogeneous 1.5B → 3B run with a full adapter fit, and the realign and m = 0 ablations.

**Before changing anything, read** [`spec/CLAUDE.md`](spec/CLAUDE.md) (rules and invariants),
[`spec/HANDOFF.md`](spec/HANDOFF.md) (status, open issues, task queue, change log) and
[`spec/docs/RESEARCH_CONTEXT.md`](spec/docs/RESEARCH_CONTEXT.md) (why the experiment is designed this way).

---

## Status (30 Oct 2026)

| | |
|---|---|
| Tests | `pytest` green: 34 tests, including the KV-cache mechanics tests |
| Kaggle runs | Option B done on 2x T4 (4.99 h): dev sweep (100 problems), m chosen by the rule, test subset (300 problems). Raw outputs in `spec/kaggle/option_b/` |
| Result | H1 half met on 3B: −75% tokens, −7.3 accuracy points (see Results above) |
| Demo | `python -m scripts.demo`: chat with text agents vs Unspoken side by side, live token counts, a savings scoreboard and a Stop button |
| Next | Full test set (`notebooks/kaggle_T7_final.ipynb`, `FINAL_M = 1`), 1.5B repeat, text MAS with a higher token cap, adapter full fit |

---

## Quick start

### 1. Install

Python 3.10–3.12 (3.14 is too new for some wheels).

```bash
# local (macOS / Linux), with uv
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# or plain pip (Kaggle / Colab)
pip install -r requirements.txt
```

### 2. Check the mechanics (about 15 s, CPU, no download)

```bash
python -m pytest -v
```

`tests/test_mechanics.py` uses a tiny random Qwen2 model. The key test is
`test_cache_continuation_matches_full_prompt`: decoding prompt B on a cache of prompt A must equal
decoding A+B from scratch. **If it fails, stop**: every latent result would be invalid.

On Kaggle, after the first green run, pin the versions: `pip freeze > requirements.lock`.

### 3. Five-problem sanity run

Always from the repo root, always with `python -m`:

```bash
python -m scripts.run --method single_cot --n 5
python -m scripts.run --method text_mas   --n 5
python -m scripts.run --method latent_mas --n 5 --latent_steps 10
python -m scripts.analyze --dir results/Qwen2.5-3B-Instruct
```

Then **read the outputs by hand**: open the `.jsonl` files and check that the `output` fields are
coherent and contain `\boxed{...}`. Accuracy on 5 problems means nothing; garbled text means a lot.

The default model is `Qwen/Qwen2.5-3B-Instruct`, downloaded from the Hugging Face Hub on first
use (about 6 GB). Device is picked automatically: `cuda` > `mps` (Apple GPU) > `cpu`.

---

## Running the experiment

Do the steps in order; each catches a different class of bug cheaply.

| Step | What | Command sketch |
|---|---|---|
| 1 | Mechanics tests | `python -m pytest -v` |
| 2 | Sanity (n=5), read outputs | see above |
| 3 | **Dev runs**: choose `m` and realign mode here only | `--split train --offset 5000 --n 300` |
| 4 | **Final runs**: chosen settings, full test set | `--split test --n 1319` |
| 5 | Analysis | `python -m scripts.analyze --dir results/Qwen2.5-3B-Instruct` |

Dev runs use GSM8K **train**, never test. `run.py` refuses `--split train` with
`--offset < 2000`, because `train[:2000]` is reserved for fitting the adapter.

### Experiment matrix

| Run | Flags | Purpose |
|---|---|---|
| Single CoT | `--method single_cot` | Baseline 1 |
| Text MAS | `--method text_mas` | Baseline 2 (the main comparison) |
| Latent MAS | `--method latent_mas --latent_steps {0,5,10,20,40}` | Main result + steps ablation |
| m=0, reference semantics | `--method latent_mas --latent_steps 0 --m0_cache drop` | Judger gets no cache at all |
| No correction | `--method latent_mas --realign raw` | Does realignment matter? |
| Norm only | `--method latent_mas --realign norm_only` | Separates rotation from rescaling |
| Hetero (stretch) | `--method hetero_mas --model …1.5B… --model_b …3B… --adapter …` | Our extension |
| Model scale | repeat the main rows with `--model Qwen/Qwen2.5-1.5B-Instruct` | Does the effect survive at small scale? |

On Kaggle's 2x T4, run two jobs in parallel with `CUDA_VISIBLE_DEVICES=0` and `=1`.

### `scripts/run.py` flags

| Flag | Default | Notes |
|---|---|---|
| `--method` | (required) | `single_cot`, `text_mas`, `latent_mas`, `hetero_mas` |
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | latent agents' model (or the only model) |
| `--split` | `test` | `train` for dev runs (needs `--offset >= 2000`) |
| `--n`, `--offset` | `300`, `0` | which slice of the split; use the same slice for every method |
| `--latent_steps` | `10` | `m`, latent steps per non-judger agent |
| `--realign` | `full` | `full`, `norm_only`, `raw` |
| `--m0_cache` | `keep` | with `--latent_steps 0`: `keep` earlier agents' prompts in the cache, or `drop` it (reference behaviour) |
| `--max_new_tokens` | `512` | judger / single-agent budget |
| `--agent_max_tokens` | `256` | text-MAS non-judger budget |
| `--temperature`, `--seed` | `0.0`, `0` | greedy by default; sampling only for robustness checks |
| `--device`, `--device_b` | auto | `--device_b cuda:1` puts hetero's `model_b` on a second GPU |
| `--model_b`, `--adapter` | none | hetero_mas only |
| `--out` | auto | override the output path |

The defaults (`m=10`, 512/256 tokens, greedy) are part of the pre-registered experiment.
Change them only with a reason logged in the HANDOFF change log.

### Outputs

```
results/<model>/<tag>_seed<k>.jsonl          test-split runs (what analyze reads)
results/<model>/train/<tag>_seed<k>.jsonl    dev runs; kept apart so they never enter the test table
<same name>.config.json                      args, device, git commit, torch/transformers versions
```

`<tag>` is the method plus settings, e.g. `latent_mas_m10`, `latent_mas_m10_raw`, `latent_mas_m0_drop`.
Files are **resumable**: rerunning skips problem ids already written, so a timed-out Kaggle
session loses nothing. **Results are append-only**: never delete or edit them.

Each line is one problem: `output`, `pred`, `gold`, `correct`, `output_tokens`, `per_agent_tokens`,
`prompt_tokens`, `forward_passes`, `latent_steps`, `latency_s`, `peak_mem_mb` (CUDA only), `trace`.

### Analysis

```bash
python -m scripts.analyze --dir results/Qwen2.5-3B-Instruct --baseline text_mas_seed0
```

This prints a table with accuracy and 95% bootstrap CI, mean output tokens, forward passes,
latency, and, against the baseline, the paired accuracy difference with CI, McNemar's exact p
and % output tokens saved. For text-MAS runs it also prints how often agents hit their token cap.
With two or more `latent_mas_m*` runs it saves `ablation_latent_steps.png`. Only problems
present in **every** run are compared.

### Cross-model adapter (stretch goal)

```bash
python -m scripts.fit_adapter --model_a Qwen/Qwen2.5-1.5B-Instruct \
    --model_b Qwen/Qwen2.5-3B-Instruct --out cache/adapter_1.5B_to_3B.pt
python -m scripts.run --method hetero_mas --n 5 --model Qwen/Qwen2.5-1.5B-Instruct \
    --model_b Qwen/Qwen2.5-3B-Instruct --adapter cache/adapter_1.5B_to_3B.pt
```

Fit on GSM8K train only. Report the held-out cosine it prints. Near 0 means the map is broken;
a decent cosine with no accuracy gain is a legitimate negative result.

### Live demo (chat)

```bash
python -m scripts.demo                                     # opens http://127.0.0.1:7860
python -m scripts.demo --precompute 5                      # optional: cache 5 questions first
python -m scripts.demo --share                             # public gradio.live link (e.g. from Kaggle)
```

A chat with two answers to every question, side by side:

- **Left, 📝 Text agents:** you watch the planner, critic and refiner write their messages to
  each other live, one collapsible bubble per agent with its token count, then the final answer.
- **Right, 🧠 Unspoken:** the same agents think silently (`3 latent steps, 0 tokens`), and only the
  final answer is written.
- **Scoreboard:** tokens written, answer time and forward passes for this question, plus totals for
  the whole session (with accuracy on GSM8K examples, which are scored against the gold answer).

Type any math word problem, or click one of the GSM8K examples. **⏹ Stop** (shown while agents are
generating) halts the model at the next token; the partial text stays visible, marked as stopped,
and is never cached or scored. *Settings* (collapsed) holds `m`
(default 1, the setting that is coherent on Qwen2.5-3B), the realign mode and cache replay.

- Both columns run through the experiment's own `pipelines.text_mas` / `latent_mas` and scorer.
  Live text comes from an observe-only hook that a test proves leaves outputs unchanged.
- Each question is answered on its own: earlier turns are not fed back in, so prompts stay
  identical to the experiment's.
- The methods run one after the other on the same GPU (latent first), and each shows its own time.
- Results are cached in `cache/demo_results.json`; precompute what you will show live.
- For honesty, also show a larger `m` (m=3 already breaks on Qwen2.5-3B; m=10 is the pre-registered default).
  One question illustrates the mechanism; it is not evidence for H1.

### Kaggle (T4 x2)

[`notebooks/kaggle_T1_T3_T5.ipynb`](notebooks/kaggle_T1_T3_T5.ipynb) does HANDOFF tasks T1, T3 and T5
in one GPU session:
- **T1:** install, run the full test suite on the T4, write `requirements.lock`.
- **T3:** 5-problem sanity runs on 3B, plus timing runs on 1.5B, on both GPUs in parallel.
- **T5:** build a run plan for the dev and final grid.

1. Get the code to Kaggle. Either push to GitHub and set `REPO_URL` in the first cell, or make a
   bundle and upload it as a private Kaggle dataset (it keeps the git history, so results record their commit):
   ```bash
   git bundle create unspoken.bundle --all
   ```
2. New notebook → *File → Import notebook* → this `.ipynb`. *Settings:* **GPU T4 x2**, **Internet on**.
   *Add Input:* your bundle dataset (if you're not using GitHub).
3. *Run All* (about 45–75 min). Download `unspoken_kaggle_T1_T3_T5.zip` from the Output panel.
4. Locally: unzip `results/` into `results/`, commit `requirements.lock`, and paste the printed
   change-log lines into `spec/HANDOFF.md`.

Sanity runs use the dev slice (`--split train --offset 5000`), never the test set.

**T6 dev sweep:** [`notebooks/kaggle_T6_dev.ipynb`](notebooks/kaggle_T6_dev.ipynb) runs the dev grid
(m ∈ {0,1,2,3,5,10,20,40}, the realign and m=0 ablations, both baselines; ~7.6 h on 2x T4). It then
picks `m` with the pre-registered rule in `scripts/select_m.py` and plans T7 with it. Use *Save & Run
All (Commit)*. If the session ends early, add its output as an input to a new session and it resumes.

**Option B (deadline plan, what produced the results above):**
[`notebooks/kaggle_quick_dev_test.ipynb`](notebooks/kaggle_quick_dev_test.ipynb) runs a reduced dev
sweep (100 problems, m ∈ {1,2,3} plus both baselines), picks `m` with `scripts/select_m.py --min_n 100`,
then runs the three methods on test problems 0–299 and prints the H1 check (~5 h on 2x T4).

**T7 final runs:** [`notebooks/kaggle_T7_final.ipynb`](notebooks/kaggle_T7_final.ipynb) runs the full
test set (1,319 problems) for 3B and 1.5B: `single_cot`, `text_mas` and `latent_mas` at the `m` T6
chose. Expect about 23–28 h, so 2–3 sessions.
- **m is required:** set `FINAL_M` first. With T6's output attached, it checks `FINAL_M` matches T6's choice.
- **Even coverage:** runs advance in rounds of 200 problems, so an interrupted session leaves every
  method covering the same problems.
- **Resume:** attach the previous session's output and it continues where it stopped.
- **Result:** when every run reaches 1,319, it prints the H1 check and the change-log line.

To plan runs from any timed results yourself:

```bash
python -m scripts.plan --dir results/Qwen2.5-3B-Instruct/train \
    --dir_small results/Qwen2.5-1.5B-Instruct/train --emit_commands
```

It prints GPU-hours per run, how the runs split across 2 GPUs, the number of Kaggle sessions and
weeks of quota, and the exact `CUDA_VISIBLE_DEVICES=…` commands for each GPU.

---

## Guidance: rules that keep the results valid

Full list in [`spec/CLAUDE.md`](spec/CLAUDE.md). The short version:

1. **Fair comparison.** Prompts are identical across text and latent modes; only the channel differs.
   Never tune a prompt for one method.
2. **One backend.** Every method goes through `src/latent.py` and the same `Counter`. No
   `model.generate()` or vLLM for a subset of methods. Batch size 1.
3. **Four metrics, always.** Output tokens, forward passes, latency, peak memory. Each latent step
   is a full forward pass, so token savings alone flatter latent methods.
4. **Don't move the goalposts.** H1's thresholds are fixed. A failed H1 is a valid result.
5. **Data hygiene.** Choose `m` and realign mode on dev (train) runs; report test only once.
6. **No quantization** for latent runs; it distorts the hidden states being passed.
7. **Fix → test.** Any change in `latent.py`, `modeling.py` or `pipelines.py` comes with a
   tiny-model test in `tests/`. Anything that could change results goes in the HANDOFF change log.
8. **Final numbers come from CUDA.** MPS and CPU are for debugging: peak memory is only measured
   on CUDA, and kernels differ across backends.

### Known limitations

- Qwen2.5 1.5B/3B **tie** input and output embeddings, so the realignment matrix is about the
  identity (`run.py` prints `tie_word_embeddings`). `raw` vs `norm_only` is the meaningful ablation.
- Intermediate agents' latent thoughts are not human-readable; only the judger's text can be inspected.
- At `m=0` with `--m0_cache keep`, the cache holds three unfinished assistant turns. On 3B the judger
  then tends to continue the chat instead of answering, so `drop` is the cleaner "no thinking" row.
- The adapter is a linear map fit on next-token pairs; it may not carry reasoning content.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `test_mechanics.py` is skipped | torch or transformers is not importable; check `python -c "import torch, transformers"` |
| `latent_mas` output is multilingual junk | Expected for Qwen2.5 at larger `m` (HANDOFF issue 4). It is not fp16: fp32 behaves the same. Try smaller `m` on a **dev** slice. |
| `--split train needs --offset >= 2000` | By design: `train[:2000]` is the adapter's fitting data |
| `git_commit: "unknown"` in `.config.json` | The folder is not a git repository; `git init` and commit so runs are traceable |
| Out of GPU memory in `hetero_mas` | Put the judger model on the second T4: `--device_b cuda:1` |
| Slow runs | Batch size 1 is deliberate. Time an `--n 5` run and extrapolate before launching the full grid |

---

## Layout

```
src/
  data.py        GSM8K loading, answer extraction and scoring (shared by all methods)
  prompts.py     role prompts; identical for text and latent modes
  modeling.py    model loading, device choice, training-free latent realignment
  latent.py      the three primitives: prefill, latent_think, decode (+ Counter)
  pipelines.py   single_cot | text_mas | latent_mas | hetero_mas
  adapter.py     stretch goal: closed-form cross-model latent map
  metrics.py     bootstrap CIs, paired difference, McNemar test
scripts/
  run.py         run one method on one slice, resumable JSONL output
  fit_adapter.py fit the cross-model map on GSM8K train
  analyze.py     results table, significance, truncation rate, ablation plot
  demo.py        Gradio chat: text agents vs latent agents side by side, live, with a savings scoreboard
  plan.py        run plan: GPU-hours, 2-GPU schedule and commands from timed runs
  select_m.py    pre-registered rule that picks m from the dev sweep
notebooks/
  kaggle_T1_T3_T5.ipynb   one Kaggle session: tests, sanity runs, run plan
  kaggle_T6_dev.ipynb     dev sweep, m selection, T7 plan
  kaggle_quick_dev_test.ipynb  option B: reduced dev sweep, m rule, test subset 0-299
  kaggle_T7_final.ipynb   final test-set runs in 200-problem rounds, H1 check
  build_*.py              generators for the notebooks (edit these, not the .ipynb)
tests/           pytest: data, metrics, mechanics (tiny random model), script guards
spec/            CLAUDE.md, HANDOFF.md, docs/RESEARCH_CONTEXT.md
```
