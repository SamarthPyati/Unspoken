# Unspoken (`latent-proto`)

A small, testable prototype of one claim from our Research Methodology project:

> **H1 (pre-registered).** On GSM8K, a multi-agent pipeline that exchanges latent working
> memory (the KV cache) reaches accuracy within 3 points of the same pipeline exchanging text,
> while generating at least 50% fewer output tokens.

The method re-implements LatentMAS (Zou et al., arXiv 2511.20639,
[Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)) from scratch at small scale, so
the team understands and controls every line. The cross-model adapter (`src/adapter.py`) is our
own extension.

**Before changing anything, read** [`spec/CLAUDE.md`](spec/CLAUDE.md) (rules and invariants),
[`spec/HANDOFF.md`](spec/HANDOFF.md) (status, open issues, task queue, change log) and
[`spec/docs/RESEARCH_CONTEXT.md`](spec/docs/RESEARCH_CONTEXT.md) (why the experiment is designed this way).

---

## Status (16 Oct 2026)

| | |
|---|---|
| Tests | `pytest` green: 21 tests, including the KV-cache mechanics tests |
| End-to-end | Every method and script runs end to end on Qwen2.5-3B (Apple M4, MPS) |
| Baselines | `single_cot` and `text_mas` give coherent, boxed answers |
| **Open finding** | `latent_mas` is coherent at small `m` (1–5) on Qwen2.5-3B but **degrades at the default `m=10`** and breaks at any `m ≥ 1` on 1.5B. It stays coherent on Qwen3-4B (the reference's model family), so the code is correct; the model and `m` choice is a decision for the team. See HANDOFF issue 4. |
| Still to do | Kaggle T4 run of the tests + `requirements.lock` (T1); dev sweep (T6); final runs (T7) |

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
tests/           pytest: data, metrics, mechanics (tiny random model), script guards
spec/            CLAUDE.md, HANDOFF.md, docs/RESEARCH_CONTEXT.md
```
