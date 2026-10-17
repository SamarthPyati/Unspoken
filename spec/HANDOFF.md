# Handoff: Unspoken / latent-proto

**Date:** 16 Oct 2026. **From:** design and implementation sessions in claude.ai. **To:** Claude Code.

## TL;DR

The code is written but has **never run against a real model**. PyTorch could not be
installed where it was written. What has actually passed: `tests/test_data.py` and
`tests/test_metrics.py` (CPU, no torch). What has never run: `tests/test_mechanics.py`
and every script against a model. Treat the first session as debugging, not experimenting.

## Status by module

| Module | State |
|---|---|
| `src/data.py` | Tested (`test_data.py`) |
| `src/metrics.py` | Tested (`test_metrics.py`) |
| `src/prompts.py` | Written; not exercised with a real tokenizer |
| `src/modeling.py` | Written; realign math checked numerically in numpy only |
| `src/latent.py` | Written; mechanics tests exist but have never executed |
| `src/pipelines.py` | Written; never executed |
| `src/adapter.py`, `scripts/fit_adapter.py` | Written; never executed (stretch goal) |
| `scripts/run.py`, `scripts/analyze.py` | Written; `analyze.py` never run on real output |
| Gradio demo (side-by-side text vs latent, tokens saved) | `scripts/demo.py`; tested (`test_demo_*`), run on 3B locally |
| `requirements.lock` | **Not created**; do this after the first green test run |

## Provenance: what matches the reference and what doesn't

Checked on 16 Oct 2026 against `Gen-Verse/LatentMAS` (`models.py`, `methods/latent_mas.py`):

**Matches the reference**
- Realignment: `W_a = (W_outᵀW_out + λI)⁻¹ W_outᵀW_in`, then rescale to the mean input-embedding norm.
- Sequential agents: each non-judger agent's full chat-templated prompt (with generation prompt)
  is appended to **one shared cache**, followed by `m` latent steps. Only the judger decodes text.
  The reference also does not close each agent turn with `<|im_end|>`; neither do we.

**Deliberate deviations**
- Batch size 1 and a hand-written decode loop (exact forward-pass counting, no padding bugs).
- We do **not** pass `cache_position`; we rely on transformers inferring positions from the cache.
  The reference passes `cache_position = arange(past_len, past_len + new_len)` explicitly.
  See issue 1.
- **`m = 0` means something different.** The reference drops the cache entirely when
  `latent_steps == 0` (the judger sees only its own prompt). Ours keeps earlier agents' prompts in
  the cache. See issue 6.
- Realign has three modes (`full`, `norm_only`, `raw`); the reference has on/off only. Reason:
  Qwen2.5 1.5B/3B tie input and output embeddings, so `W_a ≈ I` and "on vs off" is meaningless
  there. Numpy check: ‖W_a − I‖ ≈ 2e-8 when tied vs ≈ 8 when untied. `run.py` prints
  `tie_word_embeddings`; record it.

## Known issues and suspicions (priority order)

1. **[verify] Cache positions across transformers versions.** `test_cache_continuation_matches_full_prompt`
   is the guard: decoding prompt B on a cache of prompt A must equal decoding A+B from scratch.
   If it fails, first add explicit `cache_position` (and matching `position_ids` if needed) in
   `latent._forward`, mirroring the reference. Also confirm `DynamicCache.get_seq_length()` exists
   in the installed version.
   **Status (2026-10-16):** passes locally on torch 2.14.1 / transformers 5.20.0 (macOS, CPU).
   Still to confirm on Kaggle (T1).
2. **[fixed 2026-10-16] Wrong trace value in `latent_mas`.** `cache_len_before_judger` was computed after
   `decode`, so it included the judger's prompt tokens. Now read via `cache.get_seq_length()` before
   the judger's `decode`. Test: `test_latent_mas_trace_excludes_judger`.
3. **[fixed 2026-10-16] `torch_dtype=` deprecation.** `modeling.dtype_kwarg` picks `dtype=` on
   transformers >= 4.56 and `torch_dtype=` below. Test: `test_dtype_kwarg_matches_transformers_version`.
4. **[finding, decision needed] Latent thinking degrades at m = 10 on Qwen2.5. Not an fp16 bug.**
   Investigated 2026-10-16 (macOS; CPU and MPS give the same behaviour):
   - fp16 vs fp32 makes no difference; `full` vs `raw` realign makes no difference.
   - Plumbing verified: `inputs_embeds` on a cache gives logits identical to `input_ids`;
     `‖W_a − I‖ ≈ 5e-6` (tied). Our `latent_think` / `latent_mas` were compared line by line with
     `Gen-Verse/LatentMAS` `models.py::generate_latent_batch` and `methods/latent_mas.py`: same
     realign maths, same loop, same shared cache, same unclosed agent turns.
   - Coherence sweep (greedy, first 2 test problems, judger capped at 200 tokens):

     | Model | m=0 (keep) | m=1 | m=2 | m=5 | m=10 |
     |---|---|---|---|---|---|
     | Qwen2.5-1.5B | coherent | junk | junk | junk | junk |
     | Qwen2.5-3B | hallucinates a new "user" turn | coherent, 2/2 | – | coherent 1/2, junk 1/2 | junk |
     | Qwen3-4B (reference's smallest) | – | – | – | – | coherent `<think>` reasoning |

   - Conclusion: the implementation is correct (it works on the reference's model family); Qwen2.5
     at 1.5B/3B is not robust to many latent steps. Realigned latent inputs have max cosine ≈ 0.15 to
     any real token embedding (real tokens ≈ 0.70 to their neighbour), i.e. they are off-manifold.
   - **Ask the human:** keep Qwen2.5-3B and let the dev sweep (T6) pick m (likely small), and/or add
     Qwen3-4B as the main model (4B fp16 ≈ 8 GB, fits a T4; note Qwen3 emits `<think>` by default,
     so `max_new_tokens=512` may truncate). The pre-registered default `m=10` is NOT changed.
     Also not changed: the reference's latent judger prompt adds "The latent information might contain
     irrelevant contents. Ignore it if it is not helpful"; ours doesn't. Changing a prompt is a design call.
   - Original fp16 advice still stands: never switch to quantization.
5. **[fixed 2026-10-16] Dev/test leakage.** `run.py --split {test,train}`. Train runs must use
   `--offset >= 2000` (enforced; `adapter.ADAPTER_TRAIN_TEXTS`), and they write to
   `results/<model>/train/` so they can never be mixed into the test table. `fit_adapter` refuses
   `--n_texts > 2000`. Dev runs: `--split train --offset 5000 --n 300`. Test: `test_scripts.py`.
6. **[fixed 2026-10-16, option (c)] `m = 0` semantics.** Both are available as separate ablation rows:
   `--m0_cache keep` (default, our original meaning: earlier agents' prompts stay in the cache) and
   `--m0_cache drop` (reference: judger gets an empty cache; earlier agents are skipped, since their
   output would be discarded, so forward passes count only the judger). Tag suffix `_drop`.
   Observed: on 3B, m=0 `keep` makes the judger continue the chat as a new user turn (see issue 4),
   so `drop` is probably the cleaner "no thinking" row. Test: `test_latent_mas_m0_keep_vs_drop`.
7. **[measured 2026-10-16] Text-MAS truncation.** `analyze.py` now prints, for text_mas runs, the share
   of non-judger agents that hit `agent_max_tokens` (read from the run's `.config.json`). Smoke run
   (1.5B, 1 problem): 3/3 agents at the 256 cap. If this holds at n=300, raise the cap and log why.
8. **[perf] Runtime unknown.** Batch-1 decoding with `output_hidden_states=True` on every step is
   slow. Time the n=5 sanity runs and extrapolate before committing to the full grid. Any speed-up
   (e.g. requesting hidden states only during latent steps) must not change outputs; prove it with
   a test comparing outputs before/after. Measured 2026-10-16, Apple M4 (MPS, fp16), Qwen2.5-3B, mean per problem
   over 3 problems: single_cot 42 s, text_mas 149 s, latent_mas m10 24 s, hetero_mas 54 s.
   Kaggle T4 timings still needed before planning T5.
9. **[fixed 2026-10-16] hetero device placement.** `run.py --device_b cuda:1` puts `model_b` on a
   second GPU; `_Timer` now syncs every device used and sums peak memory over GPUs.

## Task queue (in order)

| # | Task | Done when |
|---|---|---|
| T1 | Environment + mechanics | `pytest -v` fully green (mechanics included) on Kaggle; torch/transformers versions written in this file; `requirements.lock` committed |
| T2 | Fix issues 2 and 3 | Tests added; Change log updated |
| T3 | Sanity runs, `--n 5`, all three main methods on 3B | Judger outputs contain `\boxed{}`; latent_mas output is coherent text; `tie_word_embeddings` value recorded; per-problem latency noted |
| T4 | ~~Resolve issues 5 and 6~~ done 2026-10-16 (5: `--split`; 6: option (c)); human to confirm | `--split` added (if agreed); m=0 semantics decided and logged |
| T5 | Run plan | Runtime estimate per method x n; table of runs split across the 2 GPUs and Kaggle weekly quota |
| T6 | Dev runs | m ∈ {0,5,10,20,40}; realign `raw` vs `norm_only` vs `full`; single_cot; text_mas |
| T7 | Final runs | Chosen settings on the full GSM8K test set (1,319) for 3B; repeat main rows on 1.5B |
| T8 | Stretch: adapter | `fit_adapter` held-out cosine reported. Near 0 = broken map (debug); decent cosine with no accuracy gain = legitimate negative result |
| T9 | Analysis | `analyze.py` table with CIs, McNemar p, % tokens saved, forward passes, latency; ablation plot |
| T10 | Demo (done 2026-10-17: `python -m scripts.demo`) | Gradio app: same question, text MAS vs latent MAS side by side, token and forward-pass counts |

## Open questions for the human (ask; don't guess)

- Prototype deadline (decides whether T7 full runs and T8 happen).
- Compute: Kaggle/Colab only, or a department GPU server?
- Who on the four-person team owns which part (evaluation, core method, adapter, baselines + demo)?
- Issues 5 and 6 above.
- Add ARC-Challenge as a second benchmark? Only if time allows.

## Change log

- 2026-10-16: Initial implementation in claude.ai.
- 2026-10-16: Realign flag replaced by three modes (`full`, `norm_only`, `raw`) after the
  tied-embeddings finding.
- 2026-10-16: Statistics plan changed from 3 random seeds to greedy decoding + bootstrap over
  problems + McNemar (results are deterministic under greedy).
- 2026-10-16: Mechanics test skips cleanly when torch is missing or broken.
- 2026-10-16: First green `pytest -v` including mechanics (local macOS CPU, Python 3.12,
  torch 2.14.1, transformers 5.20.0). Not yet the Kaggle run T1 requires.
- 2026-10-16: Fixed issue 2. `latent_mas` trace `cache_len_before_judger` now excludes the
  judger's prompt. Changes only the logged trace field, not accuracy or the four metrics.
- 2026-10-16: Fixed issue 3. `load_model` passes `dtype=` on transformers >= 4.56, `torch_dtype=`
  otherwise. Chosen by version, not try/except, because old versions silently ignore unknown kwargs
  (that would load fp32). No behaviour change: weights still load in the requested dtype.
- 2026-10-16: Issue 5 fixed: `--split` added; train runs need `--offset >= 2000` and go to
  `results/<model>/train/`. Why: m and realign must be chosen off the test set.
- 2026-10-16: Issue 6 resolved as option (c): `--m0_cache keep|drop`, both as ablation rows. Default
  `keep` (unchanged). Why: keeps our original meaning and adds the reference's for comparison.
- 2026-10-16: Issue 7: `analyze.py` reports the text-MAS agent truncation rate.
- 2026-10-16: Issue 9: `--device_b`; `_Timer` syncs all devices and sums CUDA peak memory.
- 2026-10-16: `--device` (default cuda > mps > cpu) for run.py and fit_adapter, so the pipeline can
  be debugged on Apple GPUs. Peak memory is 0 off CUDA; final numbers come from CUDA only.
  `.config.json` now also records device and transformers version.
- 2026-10-16: Issue 4 investigation (no code change to the method): latent thinking degrades at
  m=10 on Qwen2.5 1.5B/3B but not on Qwen3-4B; see issue 4. Model/m choice is open for the human.
- 2026-10-16: End-to-end smoke test passed (MPS, Qwen2.5-3B, n=3, not results): every method and
  ablation flag, a dev-split run, fit_adapter (1.5B->3B, 100 texts: held-out cosine 0.676),
  hetero_mas, and analyze (table + ablation plot). text_mas agents hit the 256 cap 56% of the time.
  latent_mas m10 junk on 3/3 (issue 4); m0 `drop` and hetero_mas coherent 3/3.
- 2026-10-17: T10 demo: `scripts/demo.py` (Gradio). Runs the experiment's own `text_mas` /
  `latent_mas`, scores with `data.extract_prediction`/`is_correct`, shows all four metrics side
  by side, caches results in `cache/demo_results.json`. Demo only; does not write to `results/`.
  New dependency `gradio` (demo only).
