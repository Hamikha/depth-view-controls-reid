"""Run the whole proposed study, one stage at a time.

The study climbs a ladder of how much 3D is used:

    level 0  2D images only, with strong augmentation
    level 1  views rendered from depth estimated from RGB
    level 2  views rendered from measured sensor depth (TVRID)
    level 3  views rendered from full 3D meshes (optional, made on a larger GPU)

Stages:
    prepare  estimate depth and render views, including flat views that hold no 3D shape (a control)
    q1       do rendered views help with 1 to 10 images per person? (PRID-2011, iLIDS-VID)
    q2       does measured depth beat estimated depth? (TVRID, full and low resolution)
             Both include the controls: flat views, views with flip and shift only,
             weighted fusion, and the untrained network.
    report   write the tables for Q1, Q2 and Q3 (accuracy against cost)
    extra    two checks a reviewer will ask for (PRID-2011, iLIDS-VID), with their reports:
             view angles  views at +/-15, +/-30 and +/-45 degrees, flat and from estimated depth
             backbone     the main conditions again with DINOv2 ViT-S instead of ResNet-50
             Not part of "all"; run it on its own.
    fix      the controls a referee asked for, with their report (results fx_*.jsonl):
             matched flat views  the flat control with the person, rotation centre and hole filling
                                 of the estimated views, so only the shape differs (all three datasets)
             shuffled depth      views from another image's estimated depth: does the correct shape matter?
             2D perspective      a random projective warp of the real images, the plain 2D rival of views
             Not part of "all"; run it on its own, after "extra".

Every stage can be restarted, because finished work is skipped. Use --dry-run to
see the commands and a rough time estimate first. On a large GPU, --workers runs
several identity splits at once, and --stop-at enforces a time limit.

    python run_paper.py --stage prepare --prid D:/data/prid --tvrid D:/data/tvrid --work D:/work --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEW_ARMS = {"synth", "real_synth", "synth_weak", "real_synth_weak"}
PERSPECTIVE_ARMS = ["real_persp", "real_aug_persp"]
WEIGHTED_FUSIONS = ["rgbd10", "rgbd20", "rgbd30", "rgbe10", "rgbe20", "rgbe30"]
ANGLE_SETS = {"a15": ["-15:0", "15:0"], "a30": ["-30:0", "30:0"], "a45": ["-45:0", "45:0"]}
VIDEO_DATASETS = ("prid2011", "ilidsvid")


def single_modalities(modalities: list[str]) -> set[str]:
    """The models that must be trained: a fusion needs colour and one kind of depth."""
    singles: set[str] = set()
    for modality in modalities:
        if modality.startswith("rgbd"):
            singles |= {"rgb", "depth"}
        elif modality.startswith("rgbe"):
            singles |= {"rgb", "est_depth"}
        else:
            singles.add(modality)
    return singles


@dataclass
class Step:
    command: list[str]
    trainings: int = 0
    report_to: Path | None = None
    group: str | None = None  # steps of one group run in order; different groups may run at once


def script(name: str, *arguments) -> list[str]:
    return [sys.executable, str(HERE / name), *map(str, arguments)]


def common(args) -> list:
    return ["--splits", *args.splits, "--k-max", max(args.k)]


def prepare_steps(args, datasets: dict[str, Path]) -> list[Step]:
    steps: list[Step] = []
    for name, root in datasets.items():
        folder = args.work / name
        is_rgbd = name == "tvrid"
        # The low-resolution versions matter only for TVRID: the video benchmarks are already low-resolution.
        for low_res in ([None, args.low_res] if is_rgbd else [None]):
            suffix = f"_lr{low_res}" if low_res else ""
            resolution = ["--low-res", low_res] if low_res else []
            estimated = folder / f"estimated_depth{suffix}"
            needs_test_frames = ["--eval-frames", args.eval_frames] if is_rgbd else []
            steps.append(Step(script("estimate_depth.py", "--dataset", name, "--data-root", root, "--out", estimated,
                                     *common(args), *needs_test_frames, *resolution)))
            steps.append(Step(script("render_views.py", "--dataset", name, "--data-root", root, "--geometry", "estimated",
                                     "--est-root", estimated, "--out", folder / f"views_estimated{suffix}",
                                     *common(args), *resolution)))
            steps.append(Step(script("render_views.py", "--dataset", name, "--data-root", root, "--geometry", "flat",
                                     "--est-root", estimated, "--out", folder / f"views_flat{suffix}",
                                     *common(args), *resolution)))
            if is_rgbd:
                steps.append(Step(script("render_views.py", "--dataset", name, "--data-root", root, "--geometry", "measured",
                                         "--out", folder / f"views_measured{suffix}", *common(args), *resolution)))
    return steps


def grid(args, dataset: str, root: Path, output: str, split: str, modalities: list[str], arms: list[str],
         views: dict[str, Path] | None = None, est_root: Path | None = None, low_res: int | None = None,
         untrained: bool = False, ks: list[int] | None = None, backbone: str = "resnet50") -> Step:
    """One grid for one identity split. Each split has its own file, so splits can run at once.

    `untrained` scores the network before any training, which every table needs as its floor.
    """
    ks = ks or args.k
    ks = ks[:1] if untrained else ks
    command = script("run_grid.py", "--dataset", dataset, "--data-root", root, "--modalities", *modalities, "--arms", *arms,
                     "--k", *ks, "--splits", split, "--iterations", 0 if untrained else args.iterations,
                     "--eval-frames", args.eval_frames, "--out", args.results / f"{output}.split{split}.jsonl")
    if views:
        command += ["--views", *(f"{tag}={path}" for tag, path in views.items())]
    if est_root:
        command += ["--est-root", str(est_root)]
    if low_res:
        command += ["--low-res", str(low_res)]
    if backbone != "resnet50":
        command += ["--backbone", backbone]
    if args.stop_at is not None:
        command += ["--stop-at", str(args.stop_at)]
    # Fusions reuse models trained for the single modalities within one grid, so they add no training there.
    variants = sum(len(views or {}) if arm in VIEW_ARMS else 1 for arm in arms)
    trainings = 0 if untrained else len(ks) * len(single_modalities(modalities)) * variants
    return Step(command, trainings, group=f"split{split}")


def q1_steps(args, datasets: dict[str, Path]) -> list[Step]:
    steps = []
    for split, (name, root) in itertools.product(args.splits, datasets.items()):
        if name == "tvrid":
            continue
        views = {"estimated": args.work / name / "views_estimated"}
        if name in args.mesh_views:
            views["mesh"] = args.mesh_views[name]
        output, estimated = f"q1_{name}", {"estimated": args.work / name / "views_estimated"}
        steps += [
            grid(args, name, root, output, split, ["rgb"], ["real", "real_aug", "synth", "real_synth"], views),
            # Controls: is it the 3D shape that helps, and does the gain survive with flip and shift only?
            grid(args, name, root, output, split, ["rgb"], ["real_synth"], {"flat": args.work / name / "views_flat"}),
            grid(args, name, root, output, split, ["rgb"], ["synth_weak", "real_synth_weak"],
                 {**estimated, "flat": args.work / name / "views_flat"}),
            grid(args, name, root, output, split, ["rgb"], ["real_aug"], untrained=True),
        ]
    return steps


def q2_steps(args, datasets: dict[str, Path]) -> list[Step]:
    if "tvrid" not in datasets:
        return []
    root, folder, steps = datasets["tvrid"], args.work / "tvrid", []
    for split, low_res in itertools.product(args.splits, (None, args.low_res)):
        suffix = f"_lr{low_res}" if low_res else ""
        estimated_depth = folder / f"estimated_depth{suffix}"
        measured_views, estimated_views = folder / f"views_measured{suffix}", folder / f"views_estimated{suffix}"
        output = f"q2_tvrid{suffix}"
        steps += [
            grid(args, "tvrid", root, output, split, ["rgb", "depth", "est_depth", "rgbd", "rgbe"], ["real_aug"],
                 est_root=estimated_depth, low_res=low_res),
            grid(args, "tvrid", root, output, split, ["depth"], ["real_synth"], {"measured": measured_views}, low_res=low_res),
            grid(args, "tvrid", root, output, split, ["est_depth"], ["real_synth"], {"estimated": estimated_views},
                 est_root=estimated_depth, low_res=low_res),
            grid(args, "tvrid", root, output, split, ["rgb"], ["real_synth"],
                 {"measured": measured_views, "estimated": estimated_views}, low_res=low_res),
            # Controls: flat views, the untrained network, and fusion with less weight on depth.
            grid(args, "tvrid", root, output, split, ["rgb"], ["real_synth"], {"flat": folder / f"views_flat{suffix}"},
                 low_res=low_res),
            grid(args, "tvrid", root, output, split, ["rgb", "depth", "est_depth", "rgbd", "rgbe"], ["real_aug"],
                 est_root=estimated_depth, low_res=low_res, untrained=True),
            grid(args, "tvrid", root, output, split, WEIGHTED_FUSIONS, ["real_aug"], est_root=estimated_depth, low_res=low_res),
        ]
    return steps


def extra_prepare_steps(args, datasets: dict[str, Path]) -> list[Step]:
    """Views for the angle check, and the DINOv2 weights fetched once before parallel runs start."""
    steps = []
    for name in VIDEO_DATASETS:
        if name not in datasets:
            continue
        root, folder = datasets[name], args.work / name
        estimated = folder / "estimated_depth"
        # The depth estimated for the main study (up to 10 images per person) is reused; nothing in it is rewritten.
        sampling = ["--splits", *args.splits, "--k-max", max(args.angle_k)]
        for tag, angles in ANGLE_SETS.items():
            for geometry in ("estimated", "flat"):
                steps.append(Step(script("render_views.py", "--dataset", name, "--data-root", root, "--geometry", geometry,
                                         "--est-root", estimated, "--out", folder / f"views_{geometry}_{tag}", *sampling,
                                         f"--angles={','.join(angles)}")))
    steps.append(Step([sys.executable, "-c", "from reid3d.model import Dinov2Encoder; Dinov2Encoder(True); print('DINOv2 ready')"]))
    return steps


def extra_steps(args, datasets: dict[str, Path]) -> list[Step]:
    steps = []
    for split, name in itertools.product(args.splits, VIDEO_DATASETS):
        if name not in datasets:
            continue
        root, folder = datasets[name], args.work / name
        angle_views = {f"{geometry}_{tag}": folder / f"views_{geometry}_{tag}"
                       for tag in ANGLE_SETS for geometry in ("estimated", "flat")}
        default_views = {"estimated": folder / "views_estimated", "flat": folder / "views_flat"}
        steps += [
            grid(args, name, root, f"ab_{name}", split, ["rgb"], ["real_synth", "real_synth_weak"], angle_views, ks=args.angle_k),
            grid(args, name, root, f"bb_{name}", split, ["rgb"], ["real", "real_aug", "real_synth", "real_synth_weak"],
                 default_views, backbone="dinov2_s"),
            grid(args, name, root, f"bb_{name}", split, ["rgb"], ["real_aug"], untrained=True, backbone="dinov2_s"),
        ]
    return steps


def fix_prepare_steps(args, datasets: dict[str, Path]) -> list[Step]:
    """Views for the matched flat control and the shuffled-depth control. Existing view folders are not touched."""
    steps = []
    for name in VIDEO_DATASETS:
        if name not in datasets:
            continue
        root, folder = datasets[name], args.work / name
        estimated = folder / "estimated_depth"
        render = lambda geometry, out, *extra: Step(script(  # noqa: E731
            "render_views.py", "--dataset", name, "--data-root", root, "--geometry", geometry, "--est-root", estimated,
            "--out", folder / out, *extra))
        steps.append(render("flat_matched", "views_flatm", *common(args)))
        steps.append(render("shuffled", "views_shuffled", "--splits", *args.splits, "--k-max", max(args.angle_k)))
        for tag, angles in ANGLE_SETS.items():
            steps.append(render("flat_matched", f"views_flatm_{tag}", "--splits", *args.splits, "--k-max", max(args.angle_k),
                                f"--angles={','.join(angles)}"))
    if "tvrid" in datasets:
        root, folder = datasets["tvrid"], args.work / "tvrid"
        for low_res in (None, args.low_res):
            suffix = f"_lr{low_res}" if low_res else ""
            resolution = ["--low-res", low_res] if low_res else []
            steps.append(Step(script("render_views.py", "--dataset", "tvrid", "--data-root", root, "--geometry", "flat_matched",
                                     "--est-root", folder / f"estimated_depth{suffix}", "--out", folder / f"views_flatm{suffix}",
                                     *common(args), *resolution)))
    return steps


def fix_steps(args, datasets: dict[str, Path]) -> list[Step]:
    steps = []
    for split, name in itertools.product(args.splits, VIDEO_DATASETS):
        if name not in datasets:
            continue
        root, folder = datasets[name], args.work / name
        output, flat = f"fx_{name}", {"flatm": folder / "views_flatm"}
        steps += [
            grid(args, name, root, output, split, ["rgb"], ["real_synth", "real_synth_weak", "synth_weak"], flat),
            grid(args, name, root, output, split, ["rgb"], ["real_synth", "real_synth_weak"],
                 {"shuffled": folder / "views_shuffled"}, ks=args.angle_k),
            grid(args, name, root, output, split, ["rgb"], PERSPECTIVE_ARMS),
            grid(args, name, root, output, split, ["rgb"], ["real_synth", "real_synth_weak"],
                 {f"flatm_{tag}": folder / f"views_flatm_{tag}" for tag in ANGLE_SETS}, ks=args.angle_k),
            grid(args, name, root, output, split, ["rgb"], ["real_synth", "real_synth_weak"], flat, backbone="dinov2_s"),
            grid(args, name, root, output, split, ["rgb"], ["real_persp"], backbone="dinov2_s"),
        ]
    if "tvrid" in datasets:
        root, folder = datasets["tvrid"], args.work / "tvrid"
        for split, low_res in itertools.product(args.splits, (None, args.low_res)):
            suffix = f"_lr{low_res}" if low_res else ""
            steps.append(grid(args, "tvrid", root, f"fx_tvrid{suffix}", split, ["rgb"], ["real_synth"],
                              {"flatm": folder / f"views_flatm{suffix}"}, low_res=low_res))
    return steps


def fix_report_steps(args) -> list[Step]:
    """One report over every result file, so each control is compared with the runs it controls."""
    if not (args.dry_run or any(args.results.glob("fx_*.jsonl"))):
        return []
    return [Step(script("analyze.py", args.results / "*.jsonl", "--question", "fix"), report_to=args.results / "report_fix.txt")]


def extra_report_steps(args) -> list[Step]:
    steps = []
    if args.dry_run or any(args.results.glob("ab_*.jsonl")):
        steps.append(Step(script("analyze.py", args.results / "q1_*.jsonl", args.results / "ab_*.jsonl", "--question", "angles"),
                          report_to=args.results / "report_angles.txt"))
    if args.dry_run or any(args.results.glob("bb_*.jsonl")):
        steps.append(Step(script("analyze.py", args.results / "bb_*.jsonl", "--question", "q1"),
                          report_to=args.results / "report_backbone.txt"))
    return steps


def report_steps(args) -> list[Step]:
    """A report is written only when its results exist, so a partial study can still be reported."""
    steps = []
    for name, pattern in (("q1", "q1_*.jsonl"), ("q2", "q2_*.jsonl"), ("q3", "q[12]_*.jsonl")):
        if args.dry_run or any(args.results.glob(pattern)):
            steps.append(Step(script("analyze.py", args.results / pattern, "--question", name),
                              report_to=args.results / f"report_{name}.txt"))
    return steps


def parse_mesh_views(items: list[str]) -> dict[str, Path]:
    mapping = {}
    for item in items:
        dataset, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"--mesh-views expects dataset=path, got '{item}'")
        mapping[dataset] = Path(path)
    return mapping


def unused_path(path: Path) -> Path:
    """The path itself, or a numbered name beside it if a file is already there."""
    candidate, number = path, 1
    while candidate.exists():
        number += 1
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
    return candidate


def run_step(step: Step, dry_run: bool) -> None:
    print(" ".join(f'"{part}"' if " " in part else part for part in step.command), flush=True)
    if dry_run:
        return
    if step.report_to is None:
        subprocess.run(step.command, check=True, cwd=HERE)
        return
    step.report_to.parent.mkdir(parents=True, exist_ok=True)
    target = unused_path(step.report_to)
    with target.open("x", encoding="utf8") as handle:  # "x" refuses to replace an existing file
        subprocess.run(step.command, check=True, cwd=HERE, stdout=handle, text=True, encoding="utf8")
    print(f"Wrote {target}", flush=True)


def run(steps: list[Step], dry_run: bool, workers: int = 1) -> None:
    """Run steps in order, except that different groups may overlap when workers > 1."""
    for grouped, section in itertools.groupby(steps, key=lambda step: step.group is not None):
        section = list(section)
        if not grouped or dry_run or workers <= 1:
            for step in section:
                run_step(step, dry_run)
            continue
        groups: dict[str, list[Step]] = defaultdict(list)
        for step in section:
            groups[step.group].append(step)
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda group: [run_step(step, False) for step in group], groups.values()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["prepare", "q1", "q2", "report", "all", "extra", "fix"])
    parser.add_argument("--prid", type=Path, help="PRID-2011 folder")
    parser.add_argument("--ilids", type=Path, help="iLIDS-VID folder")
    parser.add_argument("--tvrid", type=Path, help="TVRID folder")
    parser.add_argument("--work", type=Path, required=True, help="folder for derived data; keep it outside cloud-synced folders")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--mesh-views", nargs="*", default=[], help="optional level 3 views, as dataset=path")
    parser.add_argument("--splits", nargs="+", default=[str(i) for i in range(10)])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--low-res", type=int, default=64, help="simulated camera height for TVRID, in pixels")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--eval-frames", type=int, default=16)
    parser.add_argument("--angle-k", nargs="+", type=int, default=[1, 2], help="images per person for the angle check")
    parser.add_argument("--workers", type=int, default=1, help="identity splits to run at the same time")
    parser.add_argument("--stop-at", type=float, default=None, help="Unix time after which no new model is started")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and a time estimate")
    args = parser.parse_args(argv)
    args.mesh_views = parse_mesh_views(args.mesh_views)

    named = {"prid2011": args.prid, "ilidsvid": args.ilids, "tvrid": args.tvrid}
    datasets = {name: path for name, path in named.items() if path is not None and path.exists()}
    for name, path in named.items():
        if name not in datasets:
            print(f"Skipping {name}: {'no folder given' if path is None else f'{path} does not exist'}", flush=True)
    if not datasets:
        parser.error("no dataset folder was found")

    steps: list[Step] = []
    if args.stage in {"prepare", "all"}:
        steps += prepare_steps(args, datasets)
    if args.stage in {"q1", "all"}:
        steps += q1_steps(args, datasets)
    if args.stage in {"q2", "all"}:
        steps += q2_steps(args, datasets)
    if args.stage == "extra":
        steps += extra_prepare_steps(args, datasets) + extra_steps(args, datasets)
    if args.stage == "fix":
        steps += fix_prepare_steps(args, datasets) + fix_steps(args, datasets)
    trainings = sum(step.trainings for step in steps)
    if trainings:
        print(f"Planned trainings: {trainings}, including any already finished, which are skipped. "
              f"For the time, multiply by the 'One model' figure from speed_test.py.", flush=True)
    run(steps, args.dry_run, args.workers)
    if args.stage in {"report", "all"}:  # planned only now, so that results made by this run are reported
        run(report_steps(args), args.dry_run)
    if args.stage == "extra":
        run(extra_report_steps(args), args.dry_run)
    if args.stage == "fix":
        run(fix_report_steps(args), args.dry_run)


if __name__ == "__main__":
    main()
