# Unspoken (repo: `latent-proto`)

Student research prototype for the Research Methodology course (RV University).
It tests one falsifiable claim: a multi-agent LLM pipeline that passes **latent
working memory (the KV cache)** between agents keeps its accuracy while generating
far fewer output tokens than the same pipeline passing **text**.

The method re-implements LatentMAS (Zou et al., arXiv 2511.20639,
github.com/Gen-Verse/LatentMAS) at small scale. Our own extension is a
cross-model latent adapter (`src/adapter.py`).

- **Current status and task queue:** `HANDOFF.md`. Read it first every session.
- **Why the experiment is designed this way:** `docs/RESEARCH_CONTEXT.md`. Read it
  before changing anything that affects what gets measured.

## Commands (always from repo root, always `python -m`)

```bash
pip install -r requirements.txt
pytest -v                                   # all tests
pytest tests/test_mechanics.py -v           # KV-handoff correctness; tiny random model, CPU, no downloads
python -m scripts.run --method single_cot --n 5
python -m scripts.run --method text_mas   --n 5
python -m scripts.run --method latent_mas --n 5 --latent_steps 10 --realign full   # full | norm_only | raw
python -m scripts.fit_adapter --model_a Qwen/Qwen2.5-1.5B-Instruct --model_b Qwen/Qwen2.5-3B-Instruct --out cache/adapter_1.5B_to_3B.pt
python -m scripts.run --method hetero_mas --n 5 --model Qwen/Qwen2.5-1.5B-Instruct --model_b Qwen/Qwen2.5-3B-Instruct --adapter cache/adapter_1.5B_to_3B.pt
python -m scripts.analyze --dir results/Qwen2.5-3B-Instruct --baseline text_mas_seed0
```

Results go to `results/<model>/<tag>_seed<k>.jsonl` (one JSON line per problem,
resumable) plus a `.config.json` with args, git commit and torch version.

## Environment

- Target: Kaggle 2x T4 (16 GB each). T4 has no native bf16, so models run in **fp16**.
  Two GPUs means two independent runs in parallel (`CUDA_VISIBLE_DEVICES=0` / `=1`).
- Main model `Qwen/Qwen2.5-3B-Instruct`; small model `Qwen/Qwen2.5-1.5B-Instruct`.
- `test_data.py` and `test_metrics.py` need only numpy/scipy. `test_mechanics.py`
  needs torch + transformers but no GPU and no model download.

## Layout

| Path | Role |
|---|---|
| `src/data.py` | GSM8K loading, answer extraction, scoring (shared by every method) |
| `src/prompts.py` | Role prompts (planner, critic, refiner, judger); identical across text and latent modes |
| `src/modeling.py` | Model loading; training-free realignment matrix `W_a` (3 modes) |
| `src/latent.py` | The only three primitives: `prefill`, `latent_think`, `decode`; `Counter` for metrics |
| `src/pipelines.py` | `single_cot`, `text_mas`, `latent_mas`, `hetero_mas`; all return the same dict |
| `src/adapter.py` | Closed-form ridge map from model A hidden states to model B input embeddings |
| `src/metrics.py` | Bootstrap CIs over problems, paired-difference CI, McNemar exact test |
| `scripts/` | `run.py` (experiments), `fit_adapter.py`, `analyze.py` (table + ablation plot) |
| `tests/` | pytest; mechanics tests use a tiny random `Qwen2ForCausalLM` |

## Invariants: do not break these

1. **Fair comparison.** Prompts are identical across modes; only the channel between
   agents differs. Never tune a prompt for one method.
2. **One backend, one loop.** Every method generates through `src/latent.py` and counts
   through the same `Counter`. No `model.generate()`, vLLM or other engine for a subset of methods.
3. **Batch size 1.** Do not add batching until results are trusted and the mechanics tests
   cover padding and position ids.
4. **One scorer.** All methods are scored by `data.extract_prediction` + `data.is_correct`.
5. **Four metrics, always.** Output tokens, forward passes, latency, peak memory. Never report
   token savings without forward passes; each latent step is a full forward pass.
6. **Pre-registered thresholds.** H1 is "accuracy within 3 points of text MAS, at least 50% fewer
   output tokens". Do not change these numbers after seeing results.
7. **Data hygiene.** The adapter is fit on GSM8K train only. Hyperparameters (`m`, realign mode)
   are chosen on a dev slice, never on the problems used for final numbers (see HANDOFF issue 5).
8. **Results are append-only.** Never delete or rewrite files in `results/`. If you change the
   JSONL record schema, update `scripts/analyze.py` in the same change.
9. **Defaults are part of the experiment.** `latent_steps=10`, `max_new_tokens=512`,
   `agent_max_tokens=256`, greedy decoding (`temperature=0`). Change only with a logged reason.

## Conventions

- Python 3.10+, type hints, docstrings that explain *why*. The student team must be able to
  defend every line in a viva, so prefer explicit, commented code over clever code.
- Any fix in `latent.py`, `modeling.py` or `pipelines.py` comes with a test in `tests/`
  using the tiny random model.
- No new dependencies without a one-line justification in the commit message.
- Record anything that changes results in the **Change log** in `HANDOFF.md` (date + what + why);
  the write-up cites it.
- Do not quantize models (4-bit/8-bit) for latent runs; it distorts the hidden states being passed.
- When unsure about research design (what to measure, what counts as a result), stop and ask
  the human rather than deciding. Engineering choices you can make yourself; log them.
