"""Render new camera views of the training frames from depth.

`--geometry measured` uses a real depth sensor and `--geometry estimated` uses the
output of estimate_depth.py. Three controls use no correct 3D shape:

    flat          a flat plane, rotated about the image centre (the first version of the control)
    flat_matched  a flat plane, with the person, rotation centre and hole filling taken from
                  the image's estimated depth, so it differs from `estimated` only in shape
    shuffled      the estimated depth of another training image, with the person, rotation
                  centre and hole filling again taken from the image's own estimated depth

Views are written where training expects them:

    <out>/<frame path, no extension>/view_00.png    colour view
    <out>/<depth path, no extension>/view_00.png    depth view (16-bit)

Only training frames are rendered, and views that are mostly unobserved are dropped.

    python render_views.py --dataset prid2011 --data-root D:/data/prid --geometry estimated \\
        --est-root D:/work/prid/estimated_depth --out D:/work/prid/views_estimated
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
from PIL import Image

from reid3d import data as D
from reid3d import geometry as G

# Ground-level cameras differ mainly in azimuth. Overhead cameras differ by tilt in
# any direction, and a top view hides the sides of the body, so its angles are smaller.
DEFAULT_ANGLES = {
    "prid2011": ["-30:0", "-15:0", "15:0", "30:0"],
    "ilidsvid": ["-30:0", "-15:0", "15:0", "30:0"],
    "tvrid": ["-20:0", "-10:0", "10:0", "20:0", "0:-20", "0:-10", "0:10", "0:20"],
}
# Estimated depth has no scale. These far-to-near ratios reflect each camera setup: a
# pedestrian in front of a nearby background, or a head seen from above against the floor.
DEFAULT_FAR = {"prid2011": 1.3, "ilidsvid": 1.3, "tvrid": 2.0}
# TVRID crops are 300 px wide at the native scale of a D455 colour stream, whose
# focal length at 640x480 is about 385 px. This is approximate; see --focal.
TVRID_NATIVE_FOCAL = 385.0
TVRID_NATIVE_CROP = 300.0


def parse_angles(items: list[str]) -> list[tuple[float, float]]:
    """Read yaw:pitch items. Items may also be joined by commas, as in --angles=-15:0,15:0,
    which is how a negative first angle must be written on the command line."""
    angles = []
    for item in (part for entry in items for part in entry.split(",") if part):
        yaw, _, pitch = item.partition(":")
        angles.append((float(yaw), float(pitch or 0.0)))
    return angles


def default_focal(dataset: str, geometry: str, height: int, width: int) -> float:
    if dataset == "tvrid" and geometry == "measured":
        return TVRID_NATIVE_FOCAL * height / TVRID_NATIVE_CROP
    return 1.2 * max(height, width)


def write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.moveaxis(array, 0, -1)).save(path, compress_level=1)


def write_depth(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.round().clip(0, 65535).astype(np.uint16)).save(path, compress_level=1)


ESTIMATED_LAYOUT = {"flat_matched", "shuffled"}
GEOMETRIES = ["measured", "estimated", "flat", *sorted(ESTIMATED_LAYOUT)]


def donors(depth_paths: list[Path]) -> list[Path]:
    """For the shuffled control: each frame borrows the depth of the frame half the list away.

    Frames are sorted by path, so the donor is almost always another person, and the
    choice is fixed, so every run and every split uses the same pairs.
    """
    half = len(depth_paths) // 2
    return [depth_paths[(i + half) % len(depth_paths)] for i in range(len(depth_paths))]


def render_frames(frames: list[Path], depth_paths: list[Path], data_root: Path, depth_root: Path, out: Path,
                  geometry: str, size: tuple[int, int], camera: G.Camera, angles: list[tuple[float, float]],
                  batch_size: int, min_visible: float, low_res: int | None, near: float, far: float,
                  device: str, force: bool) -> dict:
    """Render every frame, returning view counts, visibility and timing."""
    rgb_reader = D.reader_for("rgb", low_res)
    raw_reader = lambda path, target: D.read_raw_depth(path, target, low_res)  # noqa: E731
    donor_paths = donors(depth_paths) if geometry == "shuffled" else depth_paths
    pending = [(rgb, depth, donor) for rgb, depth, donor in zip(frames, depth_paths, donor_paths)
               if force or not any(D.view_folder(rgb, data_root, out).glob("view_*.png"))]
    written = dropped = 0
    seconds = 0.0
    visibility: list[float] = []
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            rgb = D.load_images([c[0] for c in chunk], size, rgb_reader).to(device, non_blocking=True)
            depth = D.load_images([c[1] for c in chunk], size, raw_reader).to(device, non_blocking=True)
            if geometry == "shuffled":
                donor = D.load_images([c[2] for c in chunk], size, raw_reader).to(device, non_blocking=True)
            started = time.perf_counter()
            layout = None
            if geometry != "measured":
                estimated = G.depth_from_disparity(1.0 / depth.clamp_min(1.0), near, far)
                layout = estimated if geometry in ESTIMATED_LAYOUT else None
            if geometry == "estimated":
                depth = estimated
            elif geometry in ("flat", "flat_matched"):
                depth = torch.full_like(depth, (near + far) / 2)
            elif geometry == "shuffled":
                depth = G.depth_from_disparity(1.0 / donor.clamp_min(1.0), near, far)
            rendered = G.render_views(rgb, depth, camera, angles, layout=layout)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            # Pseudo-depth is stored in thousandths so it survives 16-bit storage.
            depth_scale = 1.0 if geometry == "measured" else 1000.0
            colour_views = rendered.rgb.cpu().numpy()
            depth_views = (rendered.depth * depth_scale).cpu().numpy()
            visible = rendered.visible.cpu().numpy()
            jobs = []
            for i, (rgb_path, depth_path, _) in enumerate(chunk):
                visibility += visible[i].tolist()
                for view in range(len(angles)):
                    if visible[i, view] < min_visible:
                        dropped += 1
                        continue
                    written += 1
                    name = f"view_{view:02d}.png"
                    jobs.append(pool.submit(write_rgb, D.view_folder(rgb_path, data_root, out) / name, colour_views[i, view]))
                    jobs.append(pool.submit(write_depth, D.view_folder(depth_path, depth_root, out) / name, depth_views[i, view, 0]))
            for job in jobs:
                job.result()
    stats = {"frames_rendered": len(pending), "frames_total": len(frames), "views_written": written, "views_dropped": dropped}
    if pending:
        stats.update(render_ms_per_frame=round(1000.0 * seconds / len(pending), 3), mean_visible=round(mean(visibility), 4))
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(D.DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry", required=True, choices=GEOMETRIES)
    parser.add_argument("--est-root", help="output folder of estimate_depth.py, needed by every geometry except measured")
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--angles", nargs="+", help="views as yaw:pitch in degrees, for example 0:30 0:-30, "
                        "or --angles=-15:0,15:0 when the first value is negative")
    parser.add_argument("--focal", type=float, help="focal length in pixels at the rendered size")
    parser.add_argument("--near", type=float, default=1.0, help="nearest pseudo-depth for estimated geometry")
    parser.add_argument("--far", type=float, help="farthest pseudo-depth for estimated geometry (default depends on the dataset)")
    parser.add_argument("--min-visible", type=float, default=0.5, help="drop views with less of the image observed")
    parser.add_argument("--low-res", type=int, default=None, help="simulated camera height in pixels")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="re-render frames that already have views")
    args = parser.parse_args(argv)
    if args.geometry != "measured" and not args.est_root:
        parser.error(f"--geometry {args.geometry} needs --est-root")

    data_root, out = Path(args.data_root), Path(args.out)
    tracklets = D.DATASETS[args.dataset](data_root)
    frames = sorted(D.training_frames(tracklets, args.splits, args.k_max))
    if args.geometry == "measured":
        lookup = {rgb: depth for t in tracklets for rgb, depth in zip(t.paths, t.aux_paths)}
        missing = [frame for frame in frames if frame not in lookup]
        if missing:
            parser.error(f"{args.dataset} has no measured depth for its frames")
        depth_paths, depth_root = [lookup[frame] for frame in frames], data_root
    else:
        depth_root = Path(args.est_root)
        depth_paths = [D.estimated_depth_path(frame, data_root, depth_root) for frame in frames]

    height, width = D.IMAGE_SIZE[args.dataset]
    focal = args.focal or default_focal(args.dataset, args.geometry, height, width)
    angles = parse_angles(args.angles or DEFAULT_ANGLES[args.dataset])
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    stats = render_frames(frames, depth_paths, data_root, depth_root, out, args.geometry, (height, width),
                          G.Camera.centered(height, width, focal), angles, args.batch_size, args.min_visible,
                          args.low_res, args.near, args.far or DEFAULT_FAR[args.dataset], device, args.force)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8")) if manifest_path.exists() else {}
    manifest.update({"dataset": args.dataset, "geometry": args.geometry, "angles": angles, "focal": focal,
                     "low_res": args.low_res, "min_visible": args.min_visible, **stats})
    source_ms = 0.0
    if args.geometry == "estimated" or args.geometry in ESTIMATED_LAYOUT:
        source = Path(args.est_root) / "manifest.json"
        if source.exists():
            source_ms = json.loads(source.read_text(encoding="utf8")).get("ms_per_image", 0.0)
    if "render_ms_per_frame" in manifest:
        manifest["total_ms_per_image"] = round(manifest["render_ms_per_frame"] + source_ms, 3)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
