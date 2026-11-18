"""Focal comparisons of the paper (Families A and B) under four inferential readings.

For each comparison, from the per-split result files:
  - mean paired Rank-1 difference and the Student t 95% interval (as in the reports);
  - exact two-sided sign-flip p-value and the Holm adjustment within the family and backbone;
  - the decision rule written into reid3d/stats.py before any result existed
    (gain > standard deviation of the baseline across splits, and raw p < 0.05);
  - the Nadeau-Bengio corrected resampled t interval and test, which accounts for the overlap
    of training and test identities between random splits (variance factor 1/J + n_test/n_train);
  - the 95% interval obtained by inverting the sign-flip test (values d0 for which the test
    applied to the differences minus d0 gives p >= 0.05).

    python analysis/focal_comparisons.py results > analysis/focal_comparisons_output_v5.txt

Reads result files only; writes nothing except to standard output.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import mean, stdev

CODE = Path(__file__).resolve().parents[1]  # the repository root, which holds analyze.py and reid3d/
sys.path.insert(0, str(CODE))

from analyze import collect, load  # noqa: E402
from reid3d.stats import T_975, holm_adjust, paired_interval, sign_flip_pvalue  # noqa: E402

# Identities per split: training / test.
SIZES = {"prid2011": (89, 89), "ilidsvid": (150, 150)}
NAMES = {"prid2011": "PRID-2011", "ilidsvid": "iLIDS-VID"}
LIGHT, STRONG = ("rgb", "real", None), ("rgb", "real_aug", None)
LIGHT_V, STRONG_V = ("rgb", "real_synth_weak", "estimated"), ("rgb", "real_synth", "estimated")
LIGHT_V_FLAT = ("rgb", "real_synth_weak", "flat")

LIGHT_V_FLATM, STRONG_V_FLATM = ("rgb", "real_synth_weak", "flatm"), ("rgb", "real_synth", "flatm")
LIGHT_V_SHUF, STRONG_V_SHUF = ("rgb", "real_synth_weak", "shuffled"), ("rgb", "real_synth", "shuffled")
LIGHT_P, STRONG_P = ("rgb", "real_persp", None), ("rgb", "real_aug_persp", None)

FAMILIES = {
    "A": [(d, k, "Strong+V vs Strong", STRONG, STRONG_V) for d in SIZES for k in (1, 2)]
    + [(d, k, "Light+V vs Light", LIGHT, LIGHT_V) for d in SIZES for k in (1, 2)],
    "B (first flat control)": [(d, k, "estimated vs flat (Light+V)", LIGHT_V_FLAT, LIGHT_V) for d in SIZES for k in (1, 2)],
    # Declared in Section 4.4 before the fix run: one Holm family of eight each.
    "C1": [(d, k, "est vs matched flat (Light+V)", LIGHT_V_FLATM, LIGHT_V) for d in SIZES for k in (1, 2)]
    + [(d, k, "est vs matched flat (Strong+V)", STRONG_V_FLATM, STRONG_V) for d in SIZES for k in (1, 2)],
    "C2": [(d, k, "est vs shuffled (Light+V)", LIGHT_V_SHUF, LIGHT_V) for d in SIZES for k in (1, 2)]
    + [(d, k, "est vs shuffled (Strong+V)", STRONG_V_SHUF, STRONG_V) for d in SIZES for k in (1, 2)],
    "C3": [(d, k, "Light+V vs Light+P", LIGHT_P, LIGHT_V) for d in SIZES for k in (1, 2)]
    + [(d, k, "Strong+V vs Strong+P", STRONG_P, STRONG_V) for d in SIZES for k in (1, 2)],
    "B (redefined: Light+V part of C1)": [(d, k, "est vs matched flat (Light+V)", LIGHT_V_FLATM, LIGHT_V)
                                          for d in SIZES for k in (1, 2)],
}


def t_quantile_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of Student's t via the regularised incomplete beta function."""
    x = df / (df + t * t)
    return betainc(df / 2, 0.5, x)


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) by continued fraction (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x > (a + 1) / (a + b + 2):
        return 1.0 - betainc(b, a, 1 - x)
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    f = d
    for m in range(1, 300):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1.0 + numerator / c
            c = c if abs(c) > 1e-30 else 1e-30
            f *= c * d
    return front * f / a


def nadeau_bengio(differences: list[float], n_train: int, n_test: int) -> tuple[float, float, float, float]:
    """Corrected resampled t: interval, t statistic and two-sided p (df = J - 1)."""
    j = len(differences)
    centre, s = mean(differences), stdev(differences)
    se = math.sqrt((1 / j + n_test / n_train) * s * s)
    half = T_975[j - 1] * se
    t = centre / se if se > 0 else float("inf")
    return centre - half, centre + half, t, t_quantile_two_sided(t, j - 1)


def sign_flip_interval(differences: list[float], step: float = 0.01) -> tuple[float, float]:
    """95% interval by inverting the exact sign-flip test on a grid of shifts."""
    centre = mean(differences)
    span = max(differences) - min(differences) + 1.0
    grid = [centre - span + i * step for i in range(int(2 * span / step) + 1)]
    kept = [d0 for d0 in grid if sign_flip_pvalue([d - d0 for d in differences]) >= 0.05]
    return min(kept), max(kept)


def main(results: Path) -> None:
    patterns = [str(results / f"{prefix}_*.jsonl") for prefix in ("q1", "bb", "fx")]
    table = collect(load([p for p in patterns if list(results.glob(Path(p).name))]))
    print(__doc__.split("\n\n")[0])
    for backbone in ("resnet50", "dinov2_s"):
        for family, members in FAMILIES.items():
            rows = []
            for dataset, k, label, baseline, treatment in members:
                runs = table[(dataset, None, 1500, str(k), backbone)]
                if baseline not in runs or treatment not in runs:
                    continue  # this control was not run for this backbone
                base, treat = runs[baseline], runs[treatment]
                shared = sorted(set(base) & set(treat))
                diffs = [treat[s]["rank1"] - base[s]["rank1"] for s in shared]
                base_values = [base[s]["rank1"] for s in shared]
                p = sign_flip_pvalue(diffs)
                low, high = paired_interval(diffs)
                n_train, n_test = SIZES[dataset]
                nb_low, nb_high, nb_t, nb_p = nadeau_bengio(diffs, n_train, n_test)
                sf_low, sf_high = sign_flip_interval(diffs)
                gate = mean(diffs) > 0 and mean(diffs) > stdev(base_values) and p < 0.05
                rows.append((dataset, k, label, len(shared), mean(diffs), low, high, p, stdev(base_values), gate,
                             nb_low, nb_high, nb_t, nb_p, sf_low, sf_high))
            if not rows:
                continue
            adjusted = holm_adjust([r[7] for r in rows])
            nb_adjusted = holm_adjust([r[13] for r in rows])
            print(f"\nBackbone {backbone}, Family {family} ({len(rows)} comparisons; Holm within this family)")
            print(f"{'dataset':<10}{'k':>2}  {'comparison':<28}{'n':>3}{'diff':>7}  {'t 95% CI':<16}{'sign-flip CI':<16}"
                  f"{'p':>7}{'Holm':>7}{'base SD':>8}{'gate':>6}  {'NB 95% CI':<17}{'NB t':>6}{'NB p':>7}{'NB Holm':>8}")
            for row, holm, nb_holm in zip(rows, adjusted, nb_adjusted):
                (dataset, k, label, n, diff, low, high, p, base_sd, gate, nb_low, nb_high, nb_t, nb_p, sf_low, sf_high) = row
                print(f"{NAMES[dataset]:<10}{k:>2}  {label:<28}{n:>3}{diff:>+7.1f}  [{low:+5.1f}, {high:+5.1f}]  "
                      f"[{sf_low:+5.1f}, {sf_high:+5.1f}]  {p:>6.3f}{holm:>7.3f}{base_sd:>8.1f}{'PASS' if gate else 'fail':>6}  "
                      f"[{nb_low:+5.1f}, {nb_high:+5.1f}]  {nb_t:>6.2f}{nb_p:>7.3f}{nb_holm:>8.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results")
