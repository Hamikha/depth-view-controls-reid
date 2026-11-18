"""Statistics for comparing experiment arms across repeated identity splits.

The go/no-go rule is written here, before any real result exists, so it cannot
be adjusted afterwards to suit the numbers.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import mean, stdev


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion, returned in percent."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= hits <= total:
        raise ValueError("hits must lie between 0 and total")
    p = hits / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def sign_flip_pvalue(differences: list[float]) -> float:
    """Exact two-sided paired sign-flip permutation test.

    It makes no normality assumption, which matters with only about 10 splits.
    With n pairs the smallest possible p-value is 2 / 2**n, so at least 6 splits
    are needed for p < 0.05 to be reachable.
    """
    n = len(differences)
    if n == 0:
        raise ValueError("differences is empty")
    if n > 20:
        raise ValueError("exact enumeration is limited to 20 splits")
    observed = abs(sum(differences))
    extreme = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, differences))) >= observed - 1e-12
    )
    return extreme / 2**n


# Two-sided 95% critical values of Student's t, by degrees of freedom (n - 1 splits).
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}


def paired_interval(differences: list[float]) -> tuple[float, float]:
    """95% confidence interval of the mean paired difference (Student's t).

    It shows how large an effect the data can rule out, which a p-value alone does not.
    """
    n = len(differences)
    if n < 2:
        raise ValueError("at least two differences are needed")
    half = T_975.get(n - 1, 1.96) * stdev(differences) / math.sqrt(n)
    centre = mean(differences)
    return centre - half, centre + half


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, for reading a family of comparisons together."""
    n = len(p_values)
    adjusted = [0.0] * n
    running = 0.0
    for position, index in enumerate(sorted(range(n), key=p_values.__getitem__)):
        running = max(running, min(1.0, (n - position) * p_values[index]))
        adjusted[index] = running
    return adjusted


@dataclass
class ArmComparison:
    metric: str
    n_splits: int
    baseline_mean: float
    baseline_std: float
    treatment_mean: float
    treatment_std: float
    mean_difference: float
    p_value: float
    passes_gate: bool
    ci_low: float = float("nan")
    ci_high: float = float("nan")


def compare_arms(baseline: dict[int, float], treatment: dict[int, float], metric: str = "rank1",
                 alpha: float = 0.05) -> ArmComparison:
    """Compare two arms on the identity splits they share.

    Pass only if the treatment improves the mean, the improvement exceeds the
    standard deviation of the baseline across splits, and the paired test has
    p < alpha.
    """
    shared = sorted(set(baseline) & set(treatment))
    if len(shared) < 2:
        raise ValueError("at least two shared splits are needed")
    base = [baseline[s] for s in shared]
    treat = [treatment[s] for s in shared]
    diffs = [t - b for t, b in zip(treat, base)]
    mean_diff, base_std = mean(diffs), stdev(base)
    p_value = sign_flip_pvalue(diffs)
    ci_low, ci_high = paired_interval(diffs)
    return ArmComparison(
        metric=metric,
        n_splits=len(shared),
        baseline_mean=mean(base),
        baseline_std=base_std,
        treatment_mean=mean(treat),
        treatment_std=stdev(treat),
        mean_difference=mean_diff,
        p_value=p_value,
        passes_gate=mean_diff > 0 and mean_diff > base_std and p_value < alpha,
        ci_low=ci_low,
        ci_high=ci_high,
    )
