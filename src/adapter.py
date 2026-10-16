"""Stretch goal: a cross-model latent adapter (your team's novel contribution).

Idea: LatentMAS's realignment maps model A's output space into A's own input
space. Here we learn the same kind of map ACROSS models: from A's last-layer
hidden state at position t to B's input embedding of token t+1 -- i.e. "the
embedding B would have received if A had spoken its next token aloud".

Because the two models share a tokenizer, pairs come for free from any text.
The fit is closed-form ridge regression (no gradient training), so it runs in
minutes on a T4. Fit on GSM8K *train*, never on the test questions.

RecursiveMAS (arXiv 2604.25917) instead trains a two-layer residual
"RecursiveLink" end-to-end on task loss. Comparing against that is a natural
"future work" or a second ablation if you have time.
"""
from typing import List, Tuple

import torch

# The adapter is fit on GSM8K train[:ADAPTER_TRAIN_TEXTS]. Dev runs on the
# train split must start at or after this offset (scripts/run.py enforces it),
# so hyperparameters are never chosen on the adapter's own fitting data.
ADAPTER_TRAIN_TEXTS = 2000


@torch.no_grad()
def collect_pairs(model_a, model_b, tok, texts: List[str], max_tokens: int = 256,
                  max_pairs: int = 200_000) -> Tuple[torch.Tensor, torch.Tensor]:
    dev_a = next(model_a.parameters()).device
    emb_b = model_b.get_input_embeddings().weight
    xs, ys, total = [], [], 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_tokens)["input_ids"].to(dev_a)
        if ids.shape[1] < 2:
            continue
        h = model_a(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]  # [T, d_a]
        nxt = ids[0, 1:].to(emb_b.device)
        xs.append(h[:-1].float().cpu())
        ys.append(emb_b[nxt].float().cpu())
        total += ids.shape[1] - 1
        if total >= max_pairs:
            break
    return torch.cat(xs), torch.cat(ys)


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """Solve (X^T X + lambda I) M = X^T Y. Returns M of shape [d_a, d_b]."""
    d = x.shape[1]
    return torch.linalg.solve(x.T @ x + ridge * torch.eye(d), x.T @ y)


def heldout_cosine(m: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    """Sanity metric: mean cosine between mapped A-states and true B-embeddings.
    Report it alongside task accuracy; a map that scores near 0 here is broken."""
    pred = x @ m
    return torch.nn.functional.cosine_similarity(pred, y, dim=-1).mean().item()


def check_shared_tokenizer(tok_a, tok_b):
    if tok_a.get_vocab() != tok_b.get_vocab():
        raise ValueError("Cross-model adapter needs both models to share a tokenizer.")
