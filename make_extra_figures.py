"""Paper figures for the view-angle test and the second backbone.

    python make_extra_figures.py results/*.jsonl --out figures

Both figures plot paired Rank-1 gains with 95% t-intervals over the 10 splits.
Nothing is replaced: if a figure already exists, a numbered name is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import collect, load  # noqa: E402
import make_result_figures as base  # noqa: E402
from make_result_figures import DATASET_NAMES, KS, panel_width, save  # noqa: E402
from reid3d.stats import paired_interval  # noqa: E402

ANGLE_SETS = [("±15°", "_a15"), ("±30°", "_a30"), ("±15° and ±30°", ""), ("±45°", "_a45")]
SOURCES = [("Matched flat views", "flatm", "#999999", "o"), ("Estimated-depth views", "estimated", "#0072B2", "s")]
BACKBONES = [("ResNet-50", "resnet50", "-"), ("DINOv2 ViT-S/14", "dinov2_s", "--")]


def gain(table, dataset, k, backbone, baseline, treated):
    setting = (dataset, None, 1500, str(k), backbone)
    a, b = table.get(setting, {}).get(baseline, {}), table.get(setting, {}).get(treated, {})
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    differences = [b[s]["rank1"] - a[s]["rank1"] for s in shared]
    low, high = paired_interval(differences)
    centre = mean(differences)
    return centre, centre - low, high - centre


def angle_figure(table, datasets, out: Path, one_row: bool = False) -> Path:
    """Gain of Light+V over Light for each angle set, flat against estimated depth, at k = 1 and 2."""
    panels = [(d, k) for d in datasets for k in (1, 2)]
    columns = len(panels) if one_row else (2 if len(panels) > 2 else len(panels))
    rows = -(-len(panels) // columns)
    height = 2.5 if one_row else 2.6 * rows
    figure, axes = plt.subplots(rows, columns, figsize=(panel_width(columns, 3.25), height), squeeze=False,
                                sharey=True)
    names = ["±15°", "±30°", "both", "±45°"] if one_row else [name for name, _ in ANGLE_SETS]
    for letter, axis, (dataset, k) in zip("abcdefgh", axes.flat, panels):
        for offset, (label, source, colour, marker) in zip((-0.12, 0.12), SOURCES):
            xs, centres, errors = [], [], []
            for index, (_, suffix) in enumerate(ANGLE_SETS):
                result = gain(table, dataset, k, "resnet50", ("rgb", "real", None),
                              ("rgb", "real_synth_weak", source + suffix))
                if result:
                    xs.append(index + offset)
                    centres.append(result[0])
                    errors.append(result[1:])
            if xs:
                axis.errorbar(xs, centres, yerr=list(zip(*errors)), fmt=marker, color=colour, markersize=4,
                              capsize=2, linewidth=1.2, label=label)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(ANGLE_SETS)), names, rotation=0 if one_row else 30,
                        ha="center" if one_row else "right", fontsize=8)
        title = (f"({letter}) {DATASET_NAMES.get(dataset, dataset)}, k = {k}" if one_row
                 else f"({letter}) {DATASET_NAMES.get(dataset, dataset)}, {k} image{'s' if k > 1 else ''}")
        axis.set_title(title, fontsize=8, loc="left")
        axis.grid(alpha=0.3, linewidth=0.5)
        # sharey hides inner tick labels; show them on every panel, at the size of the x tick labels.
        axis.tick_params(axis="y", labelleft=True, labelsize=8)
    for row in axes:
        row[0].set_ylabel("Rank-1 gain over Light\n(points)" if one_row else "Rank-1 gain over 2D\n(Light+V minus Light, points)",
                          fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.1 if one_row else 0.07, 1, 1))
    return save(figure, out, "Fig4_view_angles")


def backbone_figure(table, datasets, out: Path) -> Path:
    """Gain of Light+V over Light against k, for both backbones and both view sources."""
    figure, axes = plt.subplots(1, len(datasets), figsize=(panel_width(len(datasets)), 3.1), squeeze=False, sharey=True)
    for letter, axis, dataset in zip("abcd", axes[0], datasets):
        for backbone_label, backbone, style in BACKBONES:
            for (source_label, source, colour, marker), offset in zip(SOURCES, (-0.03, 0.03)):
                ks, centres, errors = [], [], []
                for k in KS:
                    result = gain(table, dataset, k, backbone, ("rgb", "real", None),
                                  ("rgb", "real_synth_weak", source))
                    if result:
                        ks.append(k * 10 ** (offset + (0.01 if backbone != "resnet50" else -0.01)))
                        centres.append(result[0])
                        errors.append(result[1:])
                if ks:
                    axis.errorbar(ks, centres, yerr=list(zip(*errors)), linestyle=style, color=colour, marker=marker,
                                  markersize=4, capsize=2, linewidth=1.2,
                                  label=f"{backbone_label}, {source_label.lower()}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xscale="log", xticks=KS, xticklabels=[str(k) for k in KS], xlabel="Training images per identity",
                 )
        axis.set_title(f"({letter}) {DATASET_NAMES.get(dataset, dataset)}", loc="left")
        axis.grid(alpha=0.3, linewidth=0.5)
        axis.tick_params(labelleft=True)  # sharey hides inner tick labels; show them on every panel
    axes[0][0].set_ylabel("Rank-1 gain over 2D\n(Light+V minus Light, points)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    return save(figure, out, "Fig5_backbone")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cmc", action="store_true", help="write 900 dpi TIFF (plus a PNG preview) for submission")
    parser.add_argument("--flat", choices=["flatm", "flat"], default="flatm",
                        help="flat reference: matched flat views (default) or the first, center-pivot flat views")
    parser.add_argument("--angles-one-row", action="store_true", help="draw the angle figure as one row of panels")
    args = parser.parse_args(argv)
    if args.flat == "flat":
        SOURCES[0] = ("Center-pivot flat views", "flat", "#999999", "o")
    if args.cmc:
        base.FORMATS.clear()
        base.FORMATS.update({"tif": 900, "png": 200})
    table = collect(load(args.files))
    datasets = [d for d in DATASET_NAMES if any(setting[0] == d for setting in table)]
    args.out.mkdir(parents=True, exist_ok=True)
    print("Wrote", angle_figure(table, datasets, args.out, one_row=args.angles_one_row))
    print("Wrote", backbone_figure(table, datasets, args.out))


if __name__ == "__main__":
    main()
