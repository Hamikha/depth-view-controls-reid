"""Post-hoc tests of the shape effect (Table S14), Rank-1 and mAP.

1. Between datasets: is the shape effect (estimated minus matched flat views) larger on
   iLIDS-VID than on PRID-2011? Splits are independent across datasets, so this is a
   two-sample comparison of the ten per-split differences of each dataset: Welch interval
   and an exact permutation test over all 184,756 ways to split the 20 values in two.
2. Within a dataset: is the shape effect larger with real images mixed in (Light+V) than
   with views alone (Views-only)? Paired by split: sign-flip test and t interval.
3. mAP version of the C1 contrast (estimated minus matched flat), for Table S14.

    python analysis/dataset_contrast.py results > analysis/dataset_contrast_output.txt

Reads result files only; writes nothing except to standard output.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from focal_comparisons import t_quantile_two_sided  # noqa: E402  (two-sided p of Student t)

ARMS = {"Light+V": "real_synth_weak", "Strong+V": "real_synth", "Views-only (Light)": "synth_weak"}
NAMES = {"prid2011": "PRID-2011", "ilidsvid": "iLIDS-VID"}


def load(results: Path) -> dict:
    table = {}
    for path in sorted(results.glob("q1_*.jsonl")) + sorted(results.glob("fx_*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            c = row["config"]
            if c["iterations"] != 1500 or c.get("backbone", "resnet50") != "resnet50" or c["modality"] != "rgb":
                continue
            table[(c["dataset"], c["k"], c["arm"], c["synth_tag"], c["split_seed"])] = row
    return table


def t_critical(df: float) -> float:
    """97.5% Student t quantile by bisection on the two-sided p."""
    low, high = 1.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if t_quantile_two_sided(middle, df) > 0.05 else (low, middle)
    return (low + high) / 2


def sign_flip(x: list[float]) -> float:
    observed = abs(sum(x))
    hits = sum(abs(sum(s * v for s, v in zip(signs, x))) >= observed - 1e-12
               for signs in itertools.product((1, -1), repeat=len(x)))
    return hits / 2 ** len(x)


def permutation(a: list[float], b: list[float]) -> float:
    pool, n = a + b, len(a)
    observed = abs(mean(b) - mean(a))
    hits = total = 0
    for chosen in itertools.combinations(range(len(pool)), n):
        rest = [pool[i] for i in range(len(pool)) if i not in chosen]
        total += 1
        hits += abs(mean(rest) - mean([pool[i] for i in chosen])) >= observed - 1e-12
    return hits / total


def holm(ps: list[float]) -> list[float]:
    order, out, running = sorted(range(len(ps)), key=ps.__getitem__), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def main(results: Path) -> None:
    table = load(results)
    splits = range(10)

    def shape(dataset, k, arm, metric):
        return [table[(dataset, k, arm, "estimated", s)][metric] - table[(dataset, k, arm, "flatm", s)][metric]
                for s in splits]

    for metric in ("rank1", "mAP"):
        print(f"\n=== {metric} ===")
        print("1. Shape effect (estimated - matched flat), iLIDS-VID minus PRID-2011 "
              "(Welch 95% interval, exact permutation p; Holm over the 4 mixed-arm rows)")
        rows = []
        for name in ("Light+V", "Strong+V", "Views-only (Light)"):
            for k in (1, 2):
                a, b = shape("prid2011", k, ARMS[name], metric), shape("ilidsvid", k, ARMS[name], metric)
                va, vb = stdev(a) ** 2 / len(a), stdev(b) ** 2 / len(b)
                se = math.sqrt(va + vb)
                df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
                diff, half = mean(b) - mean(a), t_critical(df) * se
                rows.append((name, k, mean(a), mean(b), diff, diff - half, diff + half, df, permutation(a, b)))
        mixed = [r for r in rows if r[0] != "Views-only (Light)"]
        adjusted = dict(zip([(r[0], r[1]) for r in mixed], holm([r[8] for r in mixed])))
        for name, k, pa, pb, diff, lo, hi, df, p in rows:
            note = f", Holm {adjusted[(name, k)]:.3f}" if (name, k) in adjusted else ""
            print(f"  {name:<19} k={k}: PRID {pa:+.1f}, iLIDS {pb:+.1f}, difference {diff:+.1f} [{lo:+.1f}, {hi:+.1f}] "
                  f"(Welch df {df:.1f}), permutation p {p:.3f}{note}")
        print("2. Shape effect with real images mixed (Light+V) minus views alone (Views-only), paired by split")
        for dataset in ("prid2011", "ilidsvid"):
            for k in (1, 2):
                x = [m - v for m, v in zip(shape(dataset, k, ARMS["Light+V"], metric),
                                           shape(dataset, k, ARMS["Views-only (Light)"], metric))]
                half = 2.262 * stdev(x) / math.sqrt(len(x))
                print(f"  {NAMES[dataset]:<9} k={k}: {mean(x):+.1f} [{mean(x) - half:+.1f}, {mean(x) + half:+.1f}], "
                      f"sign-flip p {sign_flip(x):.3f}")
        if metric == "mAP":
            print("3. C1 in mAP: estimated - matched flat (paired t interval, sign-flip p; Holm over the 8)")
            c1 = []
            for dataset in ("prid2011", "ilidsvid"):
                for k in (1, 2):
                    for name in ("Light+V", "Strong+V"):
                        x = shape(dataset, k, ARMS[name], metric)
                        half = 2.262 * stdev(x) / math.sqrt(len(x))
                        c1.append((dataset, k, name, mean(x), mean(x) - half, mean(x) + half, sign_flip(x)))
            for row, adj in zip(c1, holm([r[6] for r in c1])):
                dataset, k, name, m, lo, hi, p = row
                print(f"  {NAMES[dataset]:<9} k={k} {name:<9}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}], p {p:.3f}, Holm {adj:.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results")
