"""Paper figures from result files: Rank-1 against images per person, and the effect of 3D shape.

    python make_result_figures.py results/q1_*.jsonl --out figures

Nothing is replaced: if a figure already exists, a numbered name is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from analyze import collect, load  # noqa: E402
from reid3d.stats import paired_interval  # noqa: E402

# Every text element in Times New Roman (journal and co-author request), maths text included.
TIMES = "Times New Roman"


def use_times() -> str:
    """Set Times New Roman for all figure text; stop if the font is missing rather than fall back silently."""
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=TIMES), fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(f"{TIMES} is not installed; the paper figures need it") from error
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [TIMES],
        "mathtext.fontset": "custom", "mathtext.rm": TIMES, "mathtext.it": f"{TIMES}:italic",
        "mathtext.bf": f"{TIMES}:bold", "mathtext.sf": TIMES,
        "axes.axisbelow": True,  # grid lines behind the data
    })
    return path


TIMES_PATH = use_times()

DATASET_NAMES = {"prid2011": "PRID-2011", "ilidsvid": "iLIDS-VID"}
KS = (1, 2, 5, 10)
# Color-blind-safe pairs: hue for augmentation, solid or dashed for views.
CURVES = [
    ("Light", ("rgb", "real", None), "#0072B2", "--", "o"),
    ("Light+V", ("rgb", "real_synth_weak", "estimated"), "#0072B2", "-", "o"),
    ("Strong", ("rgb", "real_aug", None), "#D55E00", "--", "s"),
    ("Strong+V", ("rgb", "real_synth", "estimated"), "#D55E00", "-", "s"),
]
SHAPE = [("Light+V", "real_synth_weak", "#0072B2", "o"), ("Strong+V", "real_synth", "#D55E00", "s"),
         ("Views-only (Light)", "synth_weak", "#009E73", "^")]
# View source the shape effect is measured against: "flatm" is the flat plane with the person mask,
# rotation center and hole filling of estimated depth; "flat" is the first, center-pivot control.
FLAT = {"tag": "flatm", "label": "matched flat"}


def unused_path(path: Path) -> Path:
    candidate, number = path, 1
    while candidate.exists():
        number += 1
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
    return candidate


# CMC limits figures to 16.51 cm (6.5 in) wide and asks for line art at 900 dpi or more.
MAX_WIDTH = 6.5
FORMATS = {"pdf": None, "png": 200}


def panel_width(panels: int, each: float = 3.4) -> float:
    return min(each * panels, MAX_WIDTH)


def save(figure, out: Path, name: str) -> Path:
    """Write the figure in every format of FORMATS; returns the first file written."""
    first = None
    stem = unused_path(out / f"{name}.{next(iter(FORMATS))}").stem
    for suffix, dpi in FORMATS.items():
        target = out / f"{stem}.{suffix}"
        options = {"dpi": dpi} if dpi else {}
        if suffix in ("tif", "tiff"):
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(target, **options)
        first = first or target
    plt.close(figure)
    return first


def runs(table, dataset, k, condition):
    return table.get((dataset, None, 1500, str(k), "resnet50"), {}).get(condition, {})


def accuracy_figure(table, datasets, out: Path) -> Path:
    figure, axes = plt.subplots(1, len(datasets), figsize=(panel_width(len(datasets)), 2.9), squeeze=False)
    for letter, axis, dataset in zip("abcd", axes[0], datasets):
        for label, condition, colour, style, marker in CURVES:
            ks = [k for k in KS if runs(table, dataset, k, condition)]
            values = [mean(r["rank1"] for r in runs(table, dataset, k, condition).values()) for k in ks]
            # 2D baselines (dashed) get hollow markers so they differ from the view curves at a glance.
            axis.plot(ks, values, style, color=colour, marker=marker, markersize=4, linewidth=1.5, label=label,
                      markerfacecolor="white" if style == "--" else colour)
        untrained = table.get((dataset, None, 0, "1", "resnet50"), {}).get(("rgb", "real_aug", None), {})
        if untrained:
            axis.axhline(mean(r["rank1"] for r in untrained.values()), color="grey", linewidth=0.8, linestyle=":")
            axis.text(1, mean(r["rank1"] for r in untrained.values()), " untrained", color="grey", fontsize=8,
                      ha="left", va="bottom")
        axis.set(xscale="log", xticks=KS, xticklabels=[str(k) for k in KS], xlabel="Training images per identity",
                 )
        axis.set_title(f"({letter}) {DATASET_NAMES.get(dataset, dataset)}", loc="left")
        axis.grid(alpha=0.3, linewidth=0.5)
    axes[0][0].set_ylabel("Rank-1 (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, handlelength=3.5)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return save(figure, out, "rank1_vs_images")


def shape_figure(table, datasets, out: Path) -> Path:
    figure, axes = plt.subplots(1, len(datasets), figsize=(panel_width(len(datasets)), 2.9), squeeze=False, sharey=True)
    for letter, axis, dataset in zip("abcd", axes[0], datasets):
        for offset, (label, arm, colour, marker) in zip((-0.06, 0.0, 0.06), SHAPE):
            ks, centres, errors = [], [], []
            for k in KS:
                flat = runs(table, dataset, k, ("rgb", arm, FLAT["tag"]))
                shaped = runs(table, dataset, k, ("rgb", arm, "estimated"))
                shared = sorted(set(flat) & set(shaped))
                if len(shared) < 2:
                    continue
                differences = [shaped[s]["rank1"] - flat[s]["rank1"] for s in shared]
                low, high = paired_interval(differences)
                ks.append(k * 10 ** offset)
                centres.append(mean(differences))
                errors.append((mean(differences) - low, high - mean(differences)))
            if ks:
                axis.errorbar(ks, centres, yerr=list(zip(*errors)), color=colour, marker=marker, markersize=4,
                              linewidth=1.2, capsize=2, label=label)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xscale="log", xticks=KS, xticklabels=[str(k) for k in KS], xlabel="Training images per identity",
                 )
        axis.set_title(f"({letter}) {DATASET_NAMES.get(dataset, dataset)}", loc="left")
        axis.grid(alpha=0.3, linewidth=0.5)
        axis.tick_params(labelleft=True)  # sharey hides inner tick labels; show them on every panel
    axes[0][0].set_ylabel(f"Rank-1 difference (points)\nestimated − {FLAT['label']}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return save(figure, out, "shape_effect")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cmc", action="store_true", help="write 900 dpi TIFF (plus a PNG preview) for submission")
    parser.add_argument("--flat", choices=["flatm", "flat"], default="flatm",
                        help="flat reference: matched flat views (default) or the first, center-pivot flat views")
    args = parser.parse_args(argv)
    if args.cmc:
        FORMATS.clear()
        FORMATS.update({"tif": 900, "png": 200})
    FLAT.update(tag=args.flat, label="matched flat" if args.flat == "flatm" else "center-pivot flat")
    table = collect(load(args.files))
    datasets = [d for d in DATASET_NAMES if any(setting[0] == d for setting in table)]
    args.out.mkdir(parents=True, exist_ok=True)
    print("Wrote", accuracy_figure(table, datasets, args.out))
    print("Wrote", shape_figure(table, datasets, args.out))


if __name__ == "__main__":
    main()
