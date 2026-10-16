# Research context

Background for engineering decisions. When code and this document disagree about
*what is being measured*, stop and ask the human.

## The broader problem and this prototype's scope

The team's Research Methodology problem proposes a latent-space efficiency layer for LLMs
with five subsystems: latent reasoning, latent communication between agents, latent memory,
multimodal compression, and an agent-to-UI layer. A student team cannot prototype all five in
a semester, so the prototype tests **only latent reasoning and latent communication**.
Memory, multimodal compression and the UI layer are out of scope and are future work in the
write-up. Do not add code for them.

## Hypothesis (pre-registered; do not change)

> **H1.** On GSM8K, a multi-agent pipeline that exchanges latent working memory (KV cache)
> reaches accuracy within 3 points of the same pipeline exchanging text, while generating
> at least 50% fewer output tokens.

A failure is a valid result. "The effect holds at 3B but collapses at 1.5B" is a reportable
finding. Never adjust thresholds, prompts or settings to rescue H1.

## Conditions

| Condition | Description |
|---|---|
| `single_cot` | One prompt, one model, text chain-of-thought (sanity baseline) |
| `text_mas` | Planner → critic → refiner → judger. Each agent writes text; the transcript is pasted into the next prompt; each agent starts from an empty cache |
| `latent_mas` | Same four roles. Non-judger agents do `m` latent steps into one shared cache; only the judger writes text |
| `hetero_mas` (stretch) | Latent agents on model A; their hidden states mapped by the adapter into model B's embedding space as a soft prefix for B's judger |

## Mechanism in one paragraph

A latent step takes the last-layer hidden state `h`, maps it to `h · W_a`, rescales it to
the mean norm of real input embeddings, and feeds it back as the next input embedding. No
token is sampled. `W_a = (W_outᵀW_out + λI)⁻¹ W_outᵀW_in` (λ = 1e-5) maps the output space
back into the input space. The KV cache built this way is the message to the next agent.
When input and output embeddings are tied (Qwen2.5 1.5B/3B), `W_a ≈ I`, so the meaningful
ablation is `raw` (no correction at all) against `norm_only`.

## Metrics

Per problem: correctness (exact numeric match), output tokens, forward passes, latent steps,
prompt tokens, latency (CUDA-synchronised), peak GPU memory. Output tokens alone flatter latent
methods because every latent step costs a forward pass, so all four efficiency numbers are reported.

## Statistics

- Greedy decoding makes each method deterministic, so uncertainty comes from which problems
  were sampled: **bootstrap over problems** (10,000 resamples, 95% CI).
- Method vs method on the same problems: **paired-difference bootstrap CI** and
  **McNemar's exact test** (uses only problems where the two methods disagree).
- `analyze.py` compares only problems that every run has finished.

## Ablations

- Latent steps `m ∈ {0, 5, 10, 20, 40}`. The accuracy-vs-`m` curve is likely the key figure.
- Realign mode: `full`, `norm_only`, `raw`.
- Model scale: 3B (primary) and 1.5B.
- Sanity: `m = 0` should behave close to a judger with no prior thinking (see HANDOFF issue 6
  for exactly what our `m = 0` means).

## The extension (team's novel contribution)

KV caches cannot move between different models. The adapter learns a linear map from model A's
last-layer hidden state at position t to model B's input embedding of token t+1, fit in closed
form (ridge regression) on GSM8K train text. Both models must share a tokenizer (two Qwen2.5 sizes).
A natural comparison or future-work item is RecursiveMAS (arXiv 2604.25917), which trains a small
residual link end-to-end on task loss instead.

## Threats to validity the write-up must address

- Implementation bugs that mimic effects (cache positions, attention masks): mechanics tests.
- Unfair baseline: identical prompts, same backend, text-agent truncation rate reported.
- Hidden compute: forward passes and latency reported alongside tokens.
- Dev/test leakage when choosing `m` (HANDOFF issue 5).
- Small-model effects may not transfer to the paper's larger models.
- Intermediate latent "thoughts" are not human-readable; only the judger's text can be inspected.

## What the write-up needs from the code

Results table (accuracy with CI, tokens, forward passes, latency, memory, difference vs text MAS
with p-value, % tokens saved); ablation plot over `m`; realign-mode comparison; `tie_word_embeddings`
for each model; adapter held-out cosine; library versions and git commit for every run
(already in each `.config.json`); a short failure analysis of problems where latent and text disagree.

## References

- Zou et al., LatentMAS: latent collaboration in multi-agent systems. arXiv 2511.20639;
  code: github.com/Gen-Verse/LatentMAS.
- RecursiveMAS. arXiv 2604.25917.
- Cobbe et al., 2021. Training Verifiers to Solve Math Word Problems (GSM8K).
