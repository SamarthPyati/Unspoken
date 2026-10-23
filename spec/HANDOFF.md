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
   - **Confirmed on Kaggle T4 (2026-10-23, CUDA fp16, dev slice train[5000:5005]):** 3B m=1 5/5 correct
     and coherent (180 output tokens vs text MAS 744, 76% fewer; 9.9 s vs 40.3 s per problem);
     m=5 0/5 and m=10 0/5, both incoherent; m=0 `drop` 5/5. 1.5B m=10 0/5. Same pattern as on the Mac,
     so it is not a backend (MPS) artefact. Caveat: at n=5, m=1 cannot be told apart from m=0 `drop`
     or single_cot (all 5/5); the dev sweep must show whether 1 latent step adds anything.
   - **Decision needed before T6:** the pre-registered grid m ∈ {0,5,10,20,40} skips the only working
     region. Proposal for the human: add m ∈ {1,2,3} to the dev grid (adding rows is not changing H1's
     thresholds). The notebook's CJK "junk" heuristic missed m=10's whitespace junk; rely on the
     `\boxed{}` rate and reading outputs.
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
   (1.5B, 1 problem): 3/3 agents at the 256 cap. Kaggle sanity (dev, n=5): 3B 27%, 1.5B 7%.
   If ~27% holds at n=300, decide whether to raise the cap (and log why).
8. **[perf] Runtime unknown.** Batch-1 decoding with `output_hidden_states=True` on every step is
   slow. Time the n=5 sanity runs and extrapolate before committing to the full grid. Any speed-up
   (e.g. requesting hidden states only during latent steps) must not change outputs; prove it with
   a test comparing outputs before/after.
   **Measured 2026-10-23 (3B, MPS):** `output_hidden_states=True` costs <1% per decode step
   (75.5–80.4 vs 75.7–75.8 ms/step), so that speed-up is **not worth the risk**; code unchanged. Measured 2026-10-16, Apple M4 (MPS, fp16), Qwen2.5-3B, mean per problem
   over 3 problems: single_cot 42 s, text_mas 149 s, latent_mas m10 24 s, hetero_mas 54 s.
   Kaggle T4 timings still needed before planning T5.
9. **[fixed 2026-10-16] hetero device placement.** `run.py --device_b cuda:1` puts `model_b` on a
   second GPU; `_Timer` now syncs every device used and sums peak memory over GPUs.

## Task queue (in order)

| # | Task | Done when |
|---|---|---|
| T1 | ~~Environment + mechanics~~ done 2026-10-23 on Kaggle (see Change log; `requirements.lock`) | `pytest -v` fully green (mechanics included) on Kaggle; torch/transformers versions written in this file; `requirements.lock` committed |
| T2 | ~~Fix issues 2 and 3~~ done 2026-10-16 | Tests added; Change log updated |
| T3 | ~~Sanity runs~~ done 2026-10-23 on Kaggle (dev slice). Baselines PASS; latent_mas **m=10 FAILS** the coherence criterion (0/5, issue 4); m=1 coherent 5/5. Summary: `spec/kaggle/t3_summary.csv` | Judger outputs contain `\boxed{}`; latent_mas output is coherent text; `tie_word_embeddings` value recorded; per-problem latency noted |
| T4 | ~~Resolve issues 5 and 6~~ done 2026-10-16 (5: `--split`; 6: option (c)); human to confirm | `--split` added (if agreed); m=0 semantics decided and logged |
| T5 | ~~Run plan~~ done 2026-10-23: `spec/kaggle/plan.md`: 55.6 GPU-h, ~28 h wall-clock on 2x T4, ~2.5 sessions, <1 week of quota. Grid still lacks m=1 (needs a decision, issue 4) | Runtime estimate per method x n; table of runs split across the 2 GPUs and Kaggle weekly quota |
| T6 | Dev runs: `notebooks/kaggle_T6_dev.ipynb` (~7.6 h on 2x T4, one session) | m ∈ {0,1,2,3,5,10,20,40}; realign `raw` vs `norm_only` vs `full`; m=0 drop; single_cot; text_mas; m chosen by `scripts/select_m.py` |
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
- 2026-10-23: `scripts/plan.py` (T5): reads timed runs, extrapolates unmeasured latent m from
  seconds per forward pass (marked `~`), packs the T6/T7 grid onto 2 GPUs and prints per-GPU
  commands. `notebooks/kaggle_T1_T3_T5.ipynb`: one Kaggle session for T1 (tests, versions,
  requirements.lock), T3 (sanity runs on the dev slice `train` offset 5000, not test) and T5.
  Dry-run locally from a `git bundle`.
- 2026-10-23: T1 on Kaggle (Tesla T4, Tesla T4): pytest green; python 3.13.15, torch 2.11.0+cu128 (CUDA 12.8), transformers 5.16.1; requirements.lock committed. Commit aa2b35a.
- 2026-10-23: T3 sanity (train offset 5000, n=5, Qwen2.5-3B-Instruct, T4 fp16): latent_mas_m0_drop 13 s, latent_mas_m10 10 s, latent_mas_m1 10 s, latent_mas_m5 11 s, single_cot 13 s, text_mas 40 s per problem; tie_word_embeddings = True.
- 2026-10-23: T3 outcome: accuracy (n=5, dev) single_cot 5/5, text_mas 5/5, latent m0_drop 5/5,
  m1 5/5, m5 0/5, m10 0/5; 1.5B single_cot 5/5, text_mas 5/5, latent m10 0/5. Peak memory 3B ≈ 6.1–6.4 GB,
  1.5B ≈ 3.0–3.3 GB per T4. `requirements.lock` is a freeze of the whole Kaggle image (920 packages,
  Python 3.13): use it to reproduce on Kaggle, not for local installs.
- 2026-10-23: T5 plan from Kaggle timings (`spec/kaggle/plan.md`): 55.6 GPU-h, two lanes of ~28 h.
  T7 latent row uses the placeholder m=10 until T6 picks m.
- 2026-10-23: **Pre-registered m-selection rule (written before any T6 result):** among the dev
  `latent_mas_m<k>` runs (realign `full`, cache `keep`), choose the m with the highest dev accuracy on
  the problems all of them finished; ties go to the smaller m. Text MAS, H1's thresholds and ablation
  rows play no part. Implemented in `scripts/select_m.py`, which refuses to choose on < 300 problems.
  The team may amend this rule only before T6 results are seen.
  Known edge case: if every latent run scores the same (e.g. all 0%), the tie-break picks the
  smallest m, which can be m=0 (no latent thinking). Decide now whether that should instead be
  reported as "no working m" (H1 then fails by construction).
- 2026-10-23: Dev grid gains m ∈ {1,2,3} (`scripts/plan.py`), because Qwen2.5-3B is coherent only at
  very small m (issue 4). Additive, dev slice only; H1 thresholds and the default m=10 untouched.
  Open: whether the m chosen on 3B also applies to the 1.5B final rows (T7).
- 2026-10-23: `analyze.py` prints a `\boxed{}` rate per run (coherence signal, issue 4).
- 2026-10-23: `notebooks/kaggle_T6_dev.ipynb` (+ builder): T6 grid on both GPUs, decision rows first,
  11 h budget guard, resume from a previous session's output, then analyze + select_m + T7 plan.
- 2026-10-23: **T6 attempt 1 failed; its results are void.** Commit 755b801, Kaggle 2x T4. All 13 runs
  exited with CUDA OOM: each GPU held two other 3B processes (~6.3 GB each) from an earlier execution
  of the same notebook in the interactive session (result configs started 14:47, before the failing
  run), so three 3B models competed for one 15 GB T4. Partial rows (m1 98, text_mas 25, m0 15, m2 11,
  single_cot 9, m3 9 problems) have no duplicate ids, but their **latencies are contaminated**
  (single_cot 32–35 s/problem vs 12.9 s in T3), so they were not imported and must not be resumed from.
  The notebook's own change-log line ("m = None", all status 1) is superseded by this entry.
  Fix: the grid cell now refuses to start if `nvidia-smi` shows any compute process or a run from an
  earlier execution is still alive; failed runs print their log tails, the logs are still zipped, and
  the notebook raises "T6 DID NOT COMPLETE" instead of printing a change-log line. Same GPU guard in
  the T1/T3/T5 notebook. Verified by local dry runs: busy GPU (refuses), failing runs (raises, logs
  zipped), normal run (completes). Next: rerun T6 from empty with Save & Run All (Commit).
