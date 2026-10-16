"""Run one method on one slice of GSM8K and append results to a JSONL file.

Examples (from the repo root):
  python -m scripts.run --method single_cot --n 300
  python -m scripts.run --method text_mas   --n 300
  python -m scripts.run --method latent_mas --n 300 --latent_steps 10
  python -m scripts.run --method latent_mas --n 300 --latent_steps 10 --realign raw
  python -m scripts.run --method hetero_mas --n 300 --model Qwen/Qwen2.5-1.5B-Instruct \
        --model_b Qwen/Qwen2.5-3B-Instruct --adapter cache/adapter_1.5B_to_3B.pt
  # dev runs (choose m / realign here, never on test):
  python -m scripts.run --method latent_mas --split train --offset 5000 --n 300 --latent_steps 5
  # m = 0, reference semantics (judger gets no cache):
  python -m scripts.run --method latent_mas --latent_steps 0 --m0_cache drop

Test-split results go to results/<model>/, train-split (dev) results to
results/<model>/train/, so the two can never end up in the same table.
The output file is resumable: rerunning skips problem ids already written,
so a Kaggle session that times out loses nothing.
"""
import argparse
import json
import os
import subprocess
import time
from types import SimpleNamespace

import torch
import transformers
from tqdm import tqdm

from src.adapter import ADAPTER_TRAIN_TEXTS
from src.data import load_gsm8k, extract_prediction, is_correct
from src.modeling import load_model, build_realign, default_device
from src import pipelines


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["single_cot", "text_mas", "latent_mas", "hetero_mas"])
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--model_b", default=None, help="judger model for hetero_mas")
    p.add_argument("--adapter", default=None, help="path from scripts/fit_adapter.py")
    p.add_argument("--split", default="test", choices=["test", "train"],
                   help="test = final numbers only; train (with --offset >= 2000) = dev runs")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--latent_steps", type=int, default=10)
    p.add_argument("--realign", default="full", choices=["full", "norm_only", "raw"])
    p.add_argument("--m0_cache", default="keep", choices=["keep", "drop"],
                   help="latent_mas with --latent_steps 0: keep earlier agents' prompts in the "
                        "cache (ours) or drop the cache like the reference (HANDOFF issue 6)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--agent_max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    p.add_argument("--device_b", default=None,
                   help="device for --model_b (e.g. cuda:1 if both models don't fit one T4)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # Data hygiene (HANDOFF issue 5): dev runs on train must not reuse the
    # problems the cross-model adapter is fit on.
    if args.split == "train" and args.offset < ADAPTER_TRAIN_TEXTS:
        p.error(f"--split train needs --offset >= {ADAPTER_TRAIN_TEXTS}: "
                f"train[:{ADAPTER_TRAIN_TEXTS}] is reserved for fitting the adapter")
    if args.m0_cache == "drop" and not (args.method == "latent_mas" and args.latent_steps == 0):
        p.error("--m0_cache drop only applies to --method latent_mas --latent_steps 0")

    torch.manual_seed(args.seed)
    device = args.device or default_device()
    device_b = args.device_b or device
    tag = args.method
    if args.method in ("latent_mas", "hetero_mas"):
        tag += f"_m{args.latent_steps}" + ("" if args.realign == "full" else f"_{args.realign}")
        if args.m0_cache == "drop":
            tag += "_drop"
    # Dev (train-split) results live in their own sub-folder so they can never be
    # mixed into the test-set table by `analyze --dir results/<model>`.
    run_dir = f"results/{args.model.split('/')[-1]}"
    if args.split != "test":
        run_dir += f"/{args.split}"
    out_path = args.out or f"{run_dir}/{tag}_seed{args.seed}.jsonl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path.replace(".jsonl", ".config.json"), "w") as f:
        json.dump({**vars(args), "device": device, "device_b": device_b,
                   "git_commit": git_commit(), "torch": torch.__version__,
                   "transformers": transformers.__version__,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    model, tok = load_model(args.model, device)
    print(f"tie_word_embeddings = {model.config.tie_word_embeddings}")
    gen = torch.Generator(device=device).manual_seed(args.seed) if args.temperature > 0 else None
    cfg = SimpleNamespace(max_new_tokens=args.max_new_tokens, agent_max_tokens=args.agent_max_tokens,
                          latent_steps=args.latent_steps, temperature=args.temperature, generator=gen,
                          m0_cache=args.m0_cache)

    if args.method in ("latent_mas", "hetero_mas"):
        w_a, norm_a = build_realign(model, mode=args.realign)
    if args.method == "hetero_mas":
        model_b, _ = load_model(args.model_b, device_b)
        saved = torch.load(args.adapter)
        cross_map = saved["map"].to(device_b)
        norm_b = model_b.get_input_embeddings().weight.float().norm(dim=1).mean()

    items = [it for it in load_gsm8k(args.split, args.n, args.offset) if it["id"] not in done]
    correct = 0
    with open(out_path, "a") as f:
        for i, it in enumerate(tqdm(items, desc=tag)):
            q = it["question"]
            if args.method == "single_cot":
                r = pipelines.single_cot(model, tok, q, cfg)
            elif args.method == "text_mas":
                r = pipelines.text_mas(model, tok, q, cfg)
            elif args.method == "latent_mas":
                r = pipelines.latent_mas(model, tok, q, cfg, w_a, norm_a)
            else:
                r = pipelines.hetero_mas(model, model_b, tok, q, cfg, w_a, norm_a, cross_map, norm_b)
            pred = extract_prediction(r["output"])
            r.update(id=it["id"], gold=it["gold"], pred=pred, correct=is_correct(pred, it["gold"]))
            correct += r["correct"]
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  running accuracy: {correct / (i + 1):.3f}")


if __name__ == "__main__":
    main()
