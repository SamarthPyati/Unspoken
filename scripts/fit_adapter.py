"""Fit the cross-model adapter on GSM8K TRAIN and report held-out cosine.

  python -m scripts.fit_adapter --model_a Qwen/Qwen2.5-1.5B-Instruct \
      --model_b Qwen/Qwen2.5-3B-Instruct --out cache/adapter_1.5B_to_3B.pt
"""
import argparse
import os

import torch
from datasets import load_dataset

from src.modeling import load_model, default_device
from src.adapter import (ADAPTER_TRAIN_TEXTS, collect_pairs, fit_ridge, heldout_cosine,
                         check_shared_tokenizer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", required=True)
    p.add_argument("--model_b", required=True)
    p.add_argument("--n_texts", type=int, default=ADAPTER_TRAIN_TEXTS)
    p.add_argument("--max_pairs", type=int, default=100_000)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.n_texts > ADAPTER_TRAIN_TEXTS:
        p.error(f"--n_texts > {ADAPTER_TRAIN_TEXTS} would overlap the dev slice "
                f"(run.py --split train --offset {ADAPTER_TRAIN_TEXTS}...)")

    device = args.device or default_device()
    model_a, tok_a = load_model(args.model_a, device)
    model_b, tok_b = load_model(args.model_b, device)
    check_shared_tokenizer(tok_a, tok_b)

    train = load_dataset("openai/gsm8k", "main", split="train")
    texts = [r["question"] + "\n" + r["answer"] for r in train.select(range(args.n_texts))]
    split = int(0.9 * len(texts))

    x, y = collect_pairs(model_a, model_b, tok_a, texts[:split], max_pairs=args.max_pairs)
    xv, yv = collect_pairs(model_a, model_b, tok_a, texts[split:], max_pairs=args.max_pairs // 9)
    m = fit_ridge(x, y, args.ridge)
    print(f"pairs: train={len(x)} val={len(xv)}")
    print(f"held-out cosine: {heldout_cosine(m, xv, yv):.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"map": m, "model_a": args.model_a, "model_b": args.model_b,
                "ridge": args.ridge}, args.out)


if __name__ == "__main__":
    main()
