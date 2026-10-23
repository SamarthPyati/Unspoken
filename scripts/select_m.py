"""Choose m for the final runs from the dev sweep, by a rule fixed BEFORE the dev results exist.

  python -m scripts.select_m --dir results/Qwen2.5-3B-Instruct/train

Rule (logged in spec/HANDOFF.md on 2026-10-23, before T6 ran):
  among the `latent_mas_m<k>` dev runs (realign `full`, cache `keep`: the
  pre-registered defaults), pick the m with the highest dev accuracy on the
  problems every such run finished; break ties by the smaller m, which costs
  fewer forward passes.

The rule looks only at latent runs. It does not compare against text MAS or
H1's thresholds, so the choice cannot be steered toward confirming H1.
Ablation rows (`_raw`, `_norm_only`, `_drop`) are reported, never selected.
"""
import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

from src.metrics import bootstrap_ci


def load_latent_runs(run_dir: str) -> Dict[int, Dict[str, dict]]:
    """m -> {problem id: record} for plain latent_mas_m<k> runs only."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "latent_mas_m*_seed*.jsonl"))):
        match = re.fullmatch(r"latent_mas_m(\d+)_seed\d+", os.path.basename(path)[:-6])
        if not match:
            continue   # ablation rows: _raw, _norm_only, _drop
        with open(path) as f:
            runs[int(match.group(1))] = {r["id"]: r for r in map(json.loads, (l for l in f if l.strip()))}
    return runs


def select_m(runs: Dict[int, Dict[str, dict]]) -> Tuple[int, List[Tuple[int, float, int]]]:
    """Apply the rule. Returns (chosen m, [(m, accuracy, n)] sorted by m)."""
    if not runs:
        raise ValueError("no latent_mas_m<k> runs found")
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        raise ValueError("no problem is shared by every latent run")
    table = sorted((m, sum(r[i]["correct"] for i in common) / len(common), len(common))
                   for m, r in runs.items())
    best = max(table, key=lambda row: (row[1], -row[0]))   # highest acc, then smallest m
    return best[0], table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="dev results, e.g. results/<model>/train")
    p.add_argument("--min_n", type=int, default=300,
                   help="refuse to choose on fewer shared problems than this")
    args = p.parse_args()

    runs = load_latent_runs(args.dir)
    chosen, table = select_m(runs)
    n = table[0][2]
    print(f"{'m':>4s}  {'dev acc [95% CI]':24s} n")
    for m, acc, _ in table:
        ci = bootstrap_ci([runs[m][i]["correct"] for i in set.intersection(*(set(r) for r in runs.values()))])
        print(f"{m:4d}  {acc:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]   {n}" + ("   <- chosen" if m == chosen else ""))
    if n < args.min_n:
        raise SystemExit(f"\nonly {n} shared dev problems (< --min_n {args.min_n}): "
                         "finish the dev runs before choosing m")
    print(f"\nchosen m = {chosen}. Next: python -m scripts.plan ... --final_m {chosen}")


if __name__ == "__main__":
    main()
