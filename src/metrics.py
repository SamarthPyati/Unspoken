"""Statistics for the write-up.

With greedy decoding a method gives the same answer every run, so variance
comes from WHICH problems were sampled. We therefore bootstrap over problems
(resample problems with replacement) rather than over random seeds.
Accuracy differences between two methods on the same problems are tested
with McNemar's exact test, which only uses the problems where they disagree.
"""
from typing import Dict, Sequence

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                   alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """CI for mean(a - b) on the SAME problems (e.g. accuracy or token difference)."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, alpha, seed)


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else binomtest(only_a, n, 0.5).pvalue
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p)}


def relative_reduction(baseline: Sequence[float], method: Sequence[float]) -> float:
    """1 - mean(method)/mean(baseline). 0.6 means 60% fewer."""
    return float(1 - np.mean(method) / np.mean(baseline))
