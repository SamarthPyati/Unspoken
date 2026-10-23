"""Run plan (HANDOFF task T5): turn timed sanity runs into a schedule for the dev and final grid.

  python -m scripts.plan --dir results/Qwen2.5-3B-Instruct/train \
      --dir_small results/Qwen2.5-1.5B-Instruct/train --emit_commands

It reads the mean latency per problem of every run in --dir (and --dir_small
for the 1.5B rows), estimates the runs it has not measured, packs all jobs
onto the available GPUs (longest first) and reports GPU-hours, wall-clock,
Kaggle sessions and weeks of quota.

Estimates for an unmeasured latent m use the measured seconds per forward
pass of the closest measured latent run: each latent step is one forward pass
per agent (3 agents), so going from m to m' adds 3 * (m' - m) passes. Output
length also changes with m, so treat those rows as rough; they are marked "~".
"""
import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

N_AGENTS_THINKING = 3   # planner, critic, refiner; the judger decodes text


def summarize_runs(run_dir: str) -> Dict[str, Dict[str, float]]:
    """tag -> {n, sec, fwd} (means per problem) for every *.jsonl in run_dir."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if not rows:
            continue
        tag = re.sub(r"_seed\d+$", "", os.path.basename(path)[:-6])
        out[tag] = {"n": len(rows),
                    "sec": sum(r["latency_s"] for r in rows) / len(rows),
                    "fwd": sum(r["forward_passes"] for r in rows) / len(rows)}
    return out


def _latent_m(tag: str) -> Optional[int]:
    """latent_mas_m10 -> 10. Only plain `full`-realign tags are used as anchors."""
    m = re.fullmatch(r"latent_mas_m(\d+)", tag)
    return int(m.group(1)) if m else None


def estimate(stats: Dict[str, Dict[str, float]], tag: str) -> Tuple[Optional[float], bool]:
    """(seconds per problem, is_estimate). Measured if the tag was run; otherwise extrapolated."""
    if tag in stats:
        return stats[tag]["sec"], False
    if tag.endswith("_drop"):
        # m=0 drop = the judger alone on an empty cache: closest measured proxy is single_cot.
        return (stats["single_cot"]["sec"], True) if "single_cot" in stats else (None, True)
    base = re.match(r"latent_mas_m(\d+)", tag)
    if not base:
        return None, True
    target_m = int(base.group(1))
    anchors = {m: s for t, s in stats.items() if (m := _latent_m(t)) is not None}
    if not anchors:
        return None, True
    m0 = min(anchors, key=lambda m: abs(m - target_m))
    a = anchors[m0]
    sec_per_fwd = a["sec"] / a["fwd"]
    fwd = max(1.0, a["fwd"] + N_AGENTS_THINKING * (target_m - m0))
    return sec_per_fwd * fwd, True


def build_grid(final_m: int, n_dev: int, n_test: int, dev_offset: int) -> List[Dict]:
    """The HANDOFF T6 (dev) and T7 (final) runs."""
    dev = (["single_cot", "text_mas"] +
           [f"latent_mas_m{m}" for m in (0, 5, 10, 20, 40)] +
           ["latent_mas_m0_drop", "latent_mas_m10_raw", "latent_mas_m10_norm_only"])
    final = ["single_cot", "text_mas", f"latent_mas_m{final_m}"]
    jobs = [{"phase": "T6 dev", "size": "3B", "tag": t, "split": "train", "offset": dev_offset,
             "n": n_dev} for t in dev]
    jobs += [{"phase": "T7 final", "size": "3B", "tag": t, "split": "test", "offset": 0,
              "n": n_test} for t in final]
    jobs += [{"phase": "T7 final", "size": "1.5B", "tag": t, "split": "test", "offset": 0,
              "n": n_test} for t in final]
    return jobs


def schedule(jobs: List[Dict], lanes: int) -> List[float]:
    """Longest-processing-time-first onto `lanes` GPUs. Sets job['lane']; returns lane hours."""
    load = [0.0] * lanes
    for job in sorted(jobs, key=lambda j: -(j["hours"] or 0)):
        i = load.index(min(load))
        job["lane"] = i
        load[i] += job["hours"] or 0
    return load


def command(job: Dict, models: Dict[str, str]) -> str:
    """The scripts.run command for one job, pinned to its GPU lane."""
    t = job["tag"]
    args = [f"--model {models[job['size']]}"]
    if t.startswith("latent_mas"):
        m = re.match(r"latent_mas_m(\d+)(.*)", t)
        args.insert(0, f"--method latent_mas --latent_steps {m.group(1)}")
        suffix = m.group(2)
        if suffix in ("_raw", "_norm_only"):
            args.append(f"--realign {suffix[1:]}")
        elif suffix == "_drop":
            args.append("--m0_cache drop")
    else:
        args.insert(0, f"--method {t}")
    args.append(f"--split {job['split']} --n {job['n']}" +
                (f" --offset {job['offset']}" if job["offset"] else ""))
    return f"CUDA_VISIBLE_DEVICES={job['lane']} python -m scripts.run " + " ".join(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="timed 3B runs (e.g. results/<3B>/train)")
    p.add_argument("--dir_small", default=None, help="timed 1.5B runs")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--model_small", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--final_m", type=int, default=10,
                   help="m for the final latent row; placeholder until T6 picks it")
    p.add_argument("--n_dev", type=int, default=300)
    p.add_argument("--n_test", type=int, default=1319)
    p.add_argument("--dev_offset", type=int, default=5000)
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--session_hours", type=float, default=11.0,
                   help="usable hours per Kaggle session (hard limit 12)")
    p.add_argument("--quota_hours", type=float, default=30.0,
                   help="Kaggle GPU hours per week; check your account, it changes")
    p.add_argument("--emit_commands", action="store_true")
    args = p.parse_args()

    stats = {"3B": summarize_runs(args.dir),
             "1.5B": summarize_runs(args.dir_small) if args.dir_small else {}}
    jobs = build_grid(args.final_m, args.n_dev, args.n_test, args.dev_offset)
    for job in jobs:
        sec, est = estimate(stats[job["size"]], job["tag"])
        job["sec"], job["est"] = sec, est
        job["hours"] = None if sec is None else sec * job["n"] / 3600
    lanes = schedule(jobs, args.gpus)

    print("| Phase | Model | Run | n | s/problem | GPU-hours | GPU |")
    print("|---|---|---|---|---|---|---|")
    for j in jobs:
        sec = "not measured" if j["sec"] is None else f"{'~' if j['est'] else ''}{j['sec']:.1f}"
        hrs = "?" if j["hours"] is None else f"{j['hours']:.1f}"
        print(f"| {j['phase']} | {j['size']} | {j['tag']} | {j['n']} | {sec} | {hrs} | {j['lane']} |")

    gpu_hours = sum(j["hours"] or 0 for j in jobs)
    wall = max(lanes)
    missing = [f"{j['size']} {j['tag']}" for j in jobs if j["hours"] is None]
    print(f"\nTotal GPU-hours: {gpu_hours:.1f}")
    print(f"Wall-clock on {args.gpus} GPUs (longest lane): {wall:.1f} h "
          f"(lanes: {', '.join(f'{h:.1f}' for h in lanes)})")
    print(f"Kaggle sessions of {args.session_hours:g} h: {wall / args.session_hours:.1f}; "
          f"weeks of a {args.quota_hours:g} h quota: {wall / args.quota_hours:.1f} "
          "(assumes quota is charged per session hour, not per GPU: verify on Kaggle)")
    if missing:
        print(f"Not estimated (time them first): {', '.join(missing)}")

    if args.emit_commands:
        models = {"3B": args.model, "1.5B": args.model_small}
        print("\n# Commands, grouped by GPU. Runs are resumable: rerun the same command "
              "in the next session to continue.")
        for lane in range(args.gpus):
            print(f"\n# --- GPU {lane}  (~{lanes[lane]:.1f} h)")
            for j in sorted((j for j in jobs if j["lane"] == lane), key=lambda j: j["phase"]):
                print(command(j, models))


if __name__ == "__main__":
    main()
