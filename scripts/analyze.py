"""Turn result files into the numbers and figure for your report.

  python -m scripts.analyze --dir results/Qwen2.5-3B-Instruct --baseline text_mas_seed0

Only problems present in EVERY file are compared, so a half-finished run
can't bias the table.
"""
import argparse
import glob
import json
import os
import re

from src.metrics import bootstrap_ci, mcnemar, paired_diff_ci, relative_reduction


def load(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, (l for l in f if l.strip()))}


def agent_cap(jsonl_path, default=256):
    """agent_max_tokens the run used, read from its .config.json."""
    try:
        with open(jsonl_path.replace(".jsonl", ".config.json")) as f:
            return json.load(f).get("agent_max_tokens", default)
    except FileNotFoundError:
        return default


def truncation_rate(rows, cap):
    """Share of NON-judger agents whose text hit the token cap (HANDOFF issue 7).
    A high rate means the text baseline is being cut short, which is unfair to it."""
    agents = [n for r in rows for n in r["per_agent_tokens"][:-1]]
    return sum(n >= cap for n in agents) / len(agents) if agents else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--baseline", default="text_mas_seed0")
    p.add_argument("--plot", default="ablation_latent_steps.png")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.dir}/*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl result files in {args.dir}")
    runs = {os.path.basename(f)[:-6]: load(f) for f in files}
    caps = {os.path.basename(f)[:-6]: agent_cap(f) for f in files}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(runs)} runs, {len(common)} problems shared by all\n")
    if not common:
        raise SystemExit("no problem id is shared by every run; nothing to compare")
    base = runs.get(args.baseline)

    header = (f"{'run':32s} {'acc [95% CI]':22s} {'boxed':>5s} {'out tok':>8s} {'fwd':>6s} {'sec':>6s} "
              f"{'vs base: dAcc, p, tok saved'}")
    print(header + "\n" + "-" * len(header))
    for name, r in runs.items():
        acc = [r[i]["correct"] for i in common]
        ci = bootstrap_ci(acc)
        tok = [r[i]["output_tokens"] for i in common]
        fwd = [r[i]["forward_passes"] for i in common]
        sec = [r[i]["latency_s"] for i in common]
        # Share of answers with a \boxed{}: the quickest signal that a judger's text is
        # degenerate (HANDOFF issue 4); a low rate means "read the outputs".
        boxed = sum("\\boxed{" in r[i].get("output", "") for i in common) / len(common)
        line = (f"{name:32s} {ci['mean']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]   {boxed:5.0%} "
                f"{sum(tok)/len(tok):8.1f} {sum(fwd)/len(fwd):6.0f} {sum(sec)/len(sec):6.2f}")
        if name.startswith("text_mas"):
            tr = truncation_rate([r[i] for i in common], caps[name])
            if tr is not None:
                line += f"  [agents at {caps[name]}-token cap: {tr:.0%}]"
        if base is not None and name != args.baseline:
            b_acc = [base[i]["correct"] for i in common]
            d = paired_diff_ci(acc, b_acc)
            mc = mcnemar(acc, b_acc)
            saved = relative_reduction([base[i]["output_tokens"] for i in common], tok)
            line += f"  {d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}], p={mc['p_value']:.3g}, {saved:.0%}"
        print(line)

    latent = {int(m.group(1)): r for n, r in runs.items()
              if (m := re.search(r"latent_mas_m(\d+)_seed", n))}
    if len(latent) >= 2:
        import matplotlib.pyplot as plt
        steps = sorted(latent)
        cis = [bootstrap_ci([latent[s][i]["correct"] for i in common]) for s in steps]
        plt.errorbar(steps, [c["mean"] for c in cis],
                     yerr=[[c["mean"] - c["lo"] for c in cis], [c["hi"] - c["mean"] for c in cis]],
                     marker="o", capsize=4, label="latent_mas")
        if base is not None:
            plt.axhline(sum(base[i]["correct"] for i in common) / len(common),
                        linestyle="--", label=args.baseline)
        plt.xlabel("latent steps per agent (m)")
        plt.ylabel("GSM8K accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.dir, args.plot), dpi=150)
        print(f"\nsaved {os.path.join(args.dir, args.plot)}")


if __name__ == "__main__":
    main()
