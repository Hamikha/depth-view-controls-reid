"""Estimate depth from RGB for the frames an experiment needs, and save it.

Results are stored as 16-bit pseudo-depth, so training reads them exactly like
sensor depth. Finished frames are skipped, so an interrupted run can be restarted.
When the dataset also has measured depth, estimated depth is scored against it.

    python estimate_depth.py --dataset tvrid --data-root D:/data/tvrid --out D:/work/tvrid/estimated_depth --eval-frames 16
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from reid3d import data as D
from reid3d import geometry as G
from reid3d.monocular import DEFAULT_MODEL, MonocularDepth, processing_size


def save_depth16(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth).save(path, compress_level=1)


def frames_to_process(tracklets: list[D.Tracklet], splits: list[int], k_max: int | None,
                      eval_frames: int | None, include_eval: bool) -> list[Path]:
    frames = D.training_frames(tracklets, splits, k_max)
    if include_eval:
        frames |= D.evaluation_frames(tracklets, eval_frames)
    return sorted(frames)


def estimate(model, frames: list[Path], data_root: Path, out: Path, output_size: tuple[int, int],
             input_size: tuple[int, int], batch_size: int, low_res: int | None, measured: dict[Path, Path],
             force: bool, check_every: int = 5) -> dict:
    """Run the depth model over `frames`, returning timing and accuracy statistics."""
    targets = {frame: D.estimated_depth_path(frame, data_root, out) for frame in frames}
    todo = [frame for frame in frames if force or not targets[frame].exists()]
    reader = D.reader_for("rgb", low_res)
    seconds, absrel, pearson = 0.0, [], []
    with ThreadPoolExecutor(8) as pool:
        for batch_number, start in enumerate(range(0, len(todo), batch_size)):
            chunk = todo[start:start + batch_size]
            images = D.load_images(chunk, input_size, reader)
            started = time.perf_counter()
            disparity = model(images)
            if disparity.is_cuda:
                torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            resized = F.interpolate(disparity, output_size, mode="bilinear", align_corners=False)
            encoded = G.disparity_to_depth16(resized).cpu().numpy()[:, 0]
            list(pool.map(lambda item: save_depth16(targets[item[0]], item[1]), zip(chunk, encoded)))
            if measured and batch_number % check_every == 0:
                sensor = D.load_images([measured[f] for f in chunk], output_size,
                                       lambda p, size: D.read_raw_depth(p, size)).to(resized.device)
                scores = G.alignment_metrics(resized, sensor)
                absrel += scores["absrel"]
                pearson += scores["pearson"]
    stats = {"frames_processed": len(todo), "frames_total": len(frames)}
    if todo:
        stats["ms_per_image"] = round(1000.0 * seconds / len(todo), 3)
    if absrel:
        stats.update(sensor_frames_checked=len(absrel), sensor_absrel=round(mean(absrel), 4),
                     sensor_pearson=round(mean(pearson), 4))
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(D.DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--k-max", type=int, default=10, help="largest number of training images per person")
    parser.add_argument("--eval-frames", type=int, default=None,
                        help="also process this many test frames per tracklet (needed for the est_depth modality)")
    parser.add_argument("--low-res", type=int, default=None, help="simulated camera height in pixels")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="recompute frames that already exist")
    args = parser.parse_args(argv)

    data_root, out = Path(args.data_root), Path(args.out)
    tracklets = D.DATASETS[args.dataset](data_root)
    frames = frames_to_process(tracklets, args.splits, args.k_max, args.eval_frames, args.eval_frames is not None)
    measured = {rgb: depth for t in tracklets for rgb, depth in zip(t.paths, t.aux_paths)}
    height, width = D.IMAGE_SIZE[args.dataset]
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    stats = estimate(MonocularDepth(device, args.model), frames, data_root, out, (height, width),
                     processing_size(height, width), args.batch_size, args.low_res, measured, args.force)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else {}
    manifest.update({"model": args.model, "dataset": args.dataset, "low_res": args.low_res, **stats})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
