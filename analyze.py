"""Summarise result files, apply the go/no-go rule, and report the study's questions.

    python analyze.py results/*.jsonl
    python analyze.py results/q1_*.jsonl --question q1      do rendered views help?
    python analyze.py results/q2_*.jsonl --question q2      measured versus estimated depth
    python analyze.py results/*.jsonl    --question q3      accuracy against cost
    python analyze.py results/q1_*.jsonl results/ab_*.jsonl --question angles   view-angle ablation
    python analyze.py results/*.jsonl --question fix   matched flat control, shuffled depth, 2D perspective
    python analyze.py results/q2_tvrid.jsonl --baseline est_depth/real_aug --treatment depth/real_aug

Conditions are written modality/arm or modality/arm/view-source.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from reid3d.stats import compare_arms, holm_adjust

Condition = tuple[str, str, "str | None"]


def load(patterns: list[str]) -> list[dict]:
    rows = []
    for pattern in patterns:
        for path in glob.glob(pattern) or [pattern]:
            for line in Path(path).read_text(encoding="utf8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # blank, or damaged by an interrupted save
    return rows


def collect(rows: list[dict]) -> dict:
    """Group runs as table[setting][condition][split] = run; a later run replaces an earlier one."""
    table: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        c = row["config"]
        setting = (c["dataset"], c.get("low_res"), c["iterations"], str(c["k"]), c.get("backbone", "resnet50"))
        condition = (c["modality"], c["arm"], c.get("synth_tag"))
        table[setting][condition][c["split_seed"]] = row
    return table


def setting_order(setting: tuple):
    dataset, low_res, iterations, k, backbone = setting
    return backbone, dataset, low_res if low_res is not None else 10**6, iterations, float("inf") if k == "None" else int(k)


def setting_text(setting: tuple) -> str:
    dataset, low_res, _, k, backbone = setting
    network = "" if backbone == "resnet50" else f" {backbone}"
    return f"{dataset}{network} res={low_res if low_res is not None else 'full'} k={k}"


def condition_text(condition: Condition) -> str:
    return "/".join(part for part in condition if part)


def parse_condition(text: str) -> Condition:
    parts = text.split("/")
    if len(parts) not in {2, 3}:
        raise ValueError(f"expected modality/arm or modality/arm/view-source, got '{text}'")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else None


def format_stat(values: list[float]) -> str:
    spread = f" +/- {stdev(values):4.1f}" if len(values) > 1 else ""
    return f"{mean(values):5.1f}{spread}"


def print_summary(table: dict) -> None:
    header = f"{'setting':<32}{'iters':>6}  {'condition':<26}{'splits':>6}  {'Rank-1':<15}{'Rank-5':<15}{'mAP':<15}"
    print(header)
    print("-" * len(header))
    for setting in sorted(table, key=setting_order):
        for condition in sorted(table[setting], key=condition_text):
            runs = list(table[setting][condition].values())
            print(f"{setting_text(setting):<32}{setting[2]:>6}  {condition_text(condition):<26}{len(runs):>6}  "
                  f"{format_stat([r['rank1'] for r in runs]):<15}{format_stat([r['rank5'] for r in runs]):<15}"
                  f"{format_stat([r['mAP'] for r in runs]):<15}")


def q1_view_arms(available, arms=frozenset({"synth", "real_synth"})) -> list[Condition]:
    return [c for c in sorted(available, key=condition_text) if c[0] == "rgb" and c[1] in arms]


def q1_weak_pairs(available) -> list[tuple[Condition, Condition]]:
    """What the views add when both sides use flip and shift only."""
    baseline: Condition = ("rgb", "real", None)
    return [(baseline, c) for c in q1_view_arms(available, {"synth_weak", "real_synth_weak"})]


SHAPE_SOURCES = ("estimated", "measured", "mesh")


def flat_control_pairs(available) -> list[tuple[Condition, Condition]]:
    """Views from real 3D shape against the same rotations of a flat plane.

    Tags may carry a suffix for the angle set, such as flat_a15 and estimated_a15;
    only views made with the same angles are compared.
    """
    pairs = []
    for flat in sorted(available, key=condition_text):
        if not (flat[2] or "").startswith("flat"):
            continue
        suffix = flat[2][len("flat"):]
        shaped = {source + suffix for source in SHAPE_SOURCES}
        pairs += [(flat, c) for c in sorted(available, key=condition_text) if c[:2] == flat[:2] and c[2] in shaped]
    return pairs


def angle_pairs(available) -> list[tuple[Condition, Condition]]:
    """Each angle set against the 2D baseline with the same augmentation."""
    baselines = {"real_synth": ("rgb", "real_aug", None), "real_synth_weak": ("rgb", "real", None)}
    return [(baselines[c[1]], c) for c in sorted(available, key=condition_text)
            if c[0] == "rgb" and c[1] in baselines and c[2] is not None]


def augmentation_pairs(available) -> list[tuple[Condition, Condition]]:
    """Flip and shift only against strong augmentation, without views."""
    return [(("rgb", "real_aug", None), ("rgb", "real", None))]


def tag_pairs(base: str, other: str):
    """Views from source `base` against views from source `other`, same arm and same angle set.

    Tags carry an optional angle-set suffix, such as flatm_a15 against estimated_a15.
    """
    def pairs(available) -> list[tuple[Condition, Condition]]:
        found = []
        for first in sorted(available, key=condition_text):
            tag = first[2] or ""
            if not (tag == base or tag.startswith(base + "_")):
                continue
            partner = (first[0], first[1], other + tag[len(base):])
            if partner in available:
                found.append((first, partner))
        return found

    return pairs


PERSPECTIVE_BASELINES = {"real_synth": ("rgb", "real_aug_persp", None), "real_synth_weak": ("rgb", "real_persp", None)}


def perspective_pairs(available) -> list[tuple[Condition, Condition]]:
    """A random 2D perspective warp against the same augmentation without it."""
    return [(("rgb", "real", None), ("rgb", "real_persp", None)), (("rgb", "real_aug", None), ("rgb", "real_aug_persp", None))]


def views_against_perspective(available) -> list[tuple[Condition, Condition]]:
    """Rendered views against the random 2D perspective warp, with the same augmentation otherwise."""
    return [(PERSPECTIVE_BASELINES[c[1]], c) for c in sorted(available, key=condition_text)
            if c[0] == "rgb" and c[1] in PERSPECTIVE_BASELINES and c[2] is not None and "_a" not in c[2]]


def q1_pairs(available) -> list[tuple[Condition, Condition]]:
    """What the views add: both sides use the same strong augmentation."""
    baseline: Condition = ("rgb", "real_aug", None)
    return [(baseline, c) for c in q1_view_arms(available)]


def q1_best_2d_pairs(metric: str):
    """Do the views beat whichever 2D baseline scored higher in each setting?

    When flip/shift wins, this changes the augmentation as well as adding views. Picking the
    baseline by its score also favours the baseline, so a gain here is a conservative one.
    """
    def pairs(available) -> list[tuple[Condition, Condition]]:
        baselines = [c for c in (("rgb", "real", None), ("rgb", "real_aug", None)) if c in available]
        if not baselines:
            return []
        best = max(baselines, key=lambda c: mean(run[metric] for run in available[c].values()))
        return [(best, c) for c in q1_view_arms(available, {"synth", "real_synth", "synth_weak", "real_synth_weak"})]

    return pairs


def q2_input_pairs(available) -> list[tuple[Condition, Condition]]:
    """Does measured depth beat estimated depth as the network's input?"""
    return [(("est_depth", "real_aug", None), ("depth", "real_aug", None))]


def q2_fusion_pairs(available) -> list[tuple[Condition, Condition]]:
    """Does adding either kind of depth to colour help, at any fusion weight?"""
    return [(("rgb", "real_aug", None), c) for c in sorted(available, key=condition_text)
            if c[0].startswith(("rgbd", "rgbe")) and c[1] == "real_aug"]


def q2_view_pairs(modalities: tuple[str, ...]):
    """Do rendered views add to training on each of these inputs?"""
    def pairs(available) -> list[tuple[Condition, Condition]]:
        return [((c[0], "real_aug", None), c) for c in sorted(available, key=condition_text)
                if c[0] in modalities and c[1] == "real_synth"]

    return pairs


def q2_source_pairs(available) -> list[tuple[Condition, Condition]]:
    """Are views from measured depth better than views from estimated depth, for colour?"""
    return [(("rgb", "real_synth", "estimated"), ("rgb", "real_synth", "measured"))]


def print_comparisons(table: dict, pairs_for, metric: str, title: str) -> None:
    """Run paired tests over all settings and print them with Holm-adjusted p-values."""
    found = []
    for setting in sorted(table, key=setting_order):
        if setting[2] == 0:
            continue  # untrained networks are a floor for the tables, not a method to test
        conditions = table[setting]
        for baseline, treatment in pairs_for(conditions):
            if baseline not in conditions or treatment not in conditions:
                continue
            base = {split: run[metric] for split, run in conditions[baseline].items()}
            treat = {split: run[metric] for split, run in conditions[treatment].items()}
            if len(set(base) & set(treat)) >= 2:
                found.append((setting, baseline, treatment, compare_arms(base, treat, metric)))
    print(f"\n{title} ({metric}, {len(found)} comparisons)")
    if not found:
        print("  No matching runs with at least two shared splits.")
        return
    for (setting, baseline, treatment, result), adjusted in zip(found, holm_adjust([f[3].p_value for f in found])):
        verdict = "PASS" if result.passes_gate else "fail"
        note = "  (too few splits for p<0.05)" if result.n_splits < 6 else ""
        print(f"  {setting_text(setting):<30} {condition_text(baseline):<20} -> {condition_text(treatment):<26} "
              f"{result.baseline_mean:5.1f} -> {result.treatment_mean:5.1f} (diff {result.mean_difference:+5.1f}, "
              f"95% CI [{result.ci_low:+.1f}, {result.ci_high:+.1f}], "
              f"p={result.p_value:.3f}, Holm p={adjusted:.3f}, n={result.n_splits}) {verdict}{note}")


def print_costs(table: dict) -> None:
    """Question 3: accuracy against training and deployment cost, and by viewpoint gap."""
    print("\nAccuracy against cost. Train prep is per training image; test time is per frame at deployment.")
    header = (f"{'setting':<32}{'condition':<26}{'Rank-1':>7}{'mAP':>7}{'train min':>11}{'train prep ms':>15}"
              f"{'test ms/frame':>15}{'within':>8}{'ascent':>8}{'oblique':>9}")
    print(header)
    print("-" * len(header))
    for setting in sorted(table, key=setting_order):
        for condition in sorted(table[setting], key=condition_text):
            runs = list(table[setting][condition].values())
            average = lambda name: mean(r.get(name) or 0.0 for r in runs)  # noqa: E731
            scenario = lambda name: (f"{mean(r['scenarios'][name]['rank1'] for r in runs):.1f}"  # noqa: E731
                                     if all(name in r.get("scenarios", {}) for r in runs) else "-")
            test_ms = average("forward_ms_per_frame") + average("test_prep_ms_per_frame")
            print(f"{setting_text(setting):<32}{condition_text(condition):<26}{average('rank1'):>7.1f}{average('mAP'):>7.1f}"
                  f"{average('train_seconds') / 60:>11.1f}{average('train_prep_ms_per_image'):>15.1f}{test_ms:>15.2f}"
                  f"{scenario('within_camera'):>8}{scenario('ascent_descent'):>8}{scenario('flat_oblique'):>9}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--question", choices=["q1", "q2", "q3", "angles", "fix"], help="print the report for one question of the study")
    parser.add_argument("--baseline", help="condition used as the baseline, such as rgb/real_aug")
    parser.add_argument("--treatment", help="condition tested against it, such as rgb/real_synth/estimated")
    parser.add_argument("--metric", default="rank1", choices=["rank1", "rank5", "mAP"])
    args = parser.parse_args(argv)

    rows = load(args.files)
    if not rows:
        parser.error("no result rows found in the given files")
    table = collect(rows)
    print_summary(table)
    if args.question == "q1":
        print_comparisons(table, q1_pairs, args.metric, "Q1: rendered views against 2D, both with strong augmentation")
        print_comparisons(table, q1_weak_pairs, args.metric, "Q1: rendered views against 2D, both with flip and shift only")
        print_comparisons(table, q1_best_2d_pairs(args.metric), args.metric,
                          "Q1: rendered views against the better 2D baseline in each setting")
        print_comparisons(table, flat_control_pairs, args.metric, "Q1 control: views with 3D shape against flat views")
        print_comparisons(table, augmentation_pairs, args.metric, "Q1: flip and shift only against strong augmentation")
    elif args.question == "angles":
        print_comparisons(table, angle_pairs, args.metric, "Angles: views at each angle set against 2D, same augmentation")
        print_comparisons(table, flat_control_pairs, args.metric, "Angles: 3D shape against flat views, same angles")
    elif args.question == "fix":
        print_comparisons(table, tag_pairs("flatm", "estimated"), args.metric,
                          "Fix: estimated-depth views against matched flat views (same person mask, centre and hole filling)")
        print_comparisons(table, tag_pairs("flatm", "measured"), args.metric, "Fix: sensor-depth views against matched flat views")
        print_comparisons(table, tag_pairs("flat", "flatm"), args.metric,
                          "Fix: matched flat views against the first flat control")
        print_comparisons(table, tag_pairs("shuffled", "estimated"), args.metric,
                          "Fix: the image's own estimated depth against depth borrowed from another image")
        print_comparisons(table, angle_pairs, args.metric, "Fix: every view source against 2D, same augmentation")
        print_comparisons(table, perspective_pairs, args.metric, "Fix: random 2D perspective warp against 2D")
        print_comparisons(table, views_against_perspective, args.metric,
                          "Fix: rendered views against the random 2D perspective warp")
    elif args.question == "q2":
        # Each question is its own family for the Holm adjustment.
        print_comparisons(table, q2_input_pairs, args.metric, "Q2: measured against estimated depth as the input")
        print_comparisons(table, q2_fusion_pairs, args.metric, "Q2: colour fused with depth against colour alone")
        print_comparisons(table, q2_view_pairs(("rgb",)), args.metric, "Q2: rendered views added to colour")
        print_comparisons(table, q2_view_pairs(("depth", "est_depth")), args.metric, "Q2: rendered views added to depth inputs")
        print_comparisons(table, q2_source_pairs, args.metric, "Q2: views from measured depth against views from estimated depth")
        print_comparisons(table, flat_control_pairs, args.metric, "Q2 control: views with 3D shape against flat views")
    elif args.question == "q3":
        print_costs(table)
    if args.baseline and args.treatment:
        pair = (parse_condition(args.baseline), parse_condition(args.treatment))
        print_comparisons(table, lambda _: [pair], args.metric, "Go/no-go")


if __name__ == "__main__":
    main()
