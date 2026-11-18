"""Dataset parsing, identity splits, few-image sampling and image caching.

Design rules that protect the validity of the experiments:

* Train and test identities never overlap, and `make_split` is checked in tests.
* Few-image subsets are nested: the k=1 image is part of the k=2 set, and so on.
  Results at different k are then properly comparable.
* Every arm of one split sees the same images, so arms can be compared pairwise.
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class Tracklet:
    pid: int
    cam: int
    paths: list[Path]
    aux_paths: list[Path] = field(default_factory=list)  # e.g. aligned depth frames
    direction: str = ""  # TVRID only: "IN" or "OUT"


# ------------------------------------------------------------------ datasets
def _sorted_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def load_prid2011(root: str | Path, min_frames: int = 27) -> list[Tracklet]:
    """PRID-2011 multi-shot data, using the standard 178-person protocol.

    Only persons 1-200 appear in both cameras. Following the common protocol, a
    person is kept only if both tracklets have more than `min_frames` frames.
    """
    root = Path(root)
    base = next((p for p in (root / "multi_shot", root / "prid_2011" / "multi_shot") if p.is_dir()), None)
    if base is None:
        raise FileNotFoundError(f"could not find multi_shot/ under {root}")
    tracklets: list[Tracklet] = []
    for pid in range(1, 201):
        folders = [base / cam / f"person_{pid:04d}" for cam in ("cam_a", "cam_b")]
        if not all(f.is_dir() for f in folders):
            continue
        frames = [_sorted_images(f) for f in folders]
        if all(len(x) > min_frames for x in frames):
            tracklets += [Tracklet(pid, cam, paths) for cam, paths in enumerate(frames)]
    return tracklets


def load_ilidsvid(root: str | Path) -> list[Tracklet]:
    """iLIDS-VID: 300 persons, each with one tracklet per camera."""
    root = Path(root)
    base = next((p for p in (root / "i-LIDS-VID" / "sequences", root / "sequences") if p.is_dir()), None)
    if base is None:
        raise FileNotFoundError(f"could not find sequences/ under {root}")
    tracklets: list[Tracklet] = []
    persons = sorted(p.name for p in (base / "cam1").iterdir() if p.is_dir())
    for name in persons:
        match = re.fullmatch(r"person(\d+)", name)
        folders = [base / cam / name for cam in ("cam1", "cam2")]
        if match and all(f.is_dir() for f in folders):
            tracklets += [Tracklet(int(match.group(1)), c, _sorted_images(f)) for c, f in enumerate(folders)]
    return tracklets


TVRID_CAMERAS = {"flat": 0, "upward": 1, "downward": 2, "upsideDown": 3}  # official camera ids


def load_tvrid(root: str | Path) -> list[Tracklet]:
    """TVRID top-view RGB-Depth data: train/<person>/<camera>/<passage>/ frames.

    Each frame has an aligned pair, *_RGB.png and *_depth.png (16-bit, millimetres).
    Each camera and passage of a person becomes one tracklet. Only the official
    `train` partition is used here, and it is split again by identity.
    """
    root = Path(root)
    base = root / "train" if (root / "train").is_dir() else root
    persons = sorted(p for p in base.iterdir() if p.is_dir())
    if not persons:
        raise FileNotFoundError(f"no person folders found under {base}")
    camera_ids = TVRID_CAMERAS
    tracklets: list[Tracklet] = []
    for person in persons:
        for camera in sorted(c for c in person.iterdir() if c.is_dir()):
            for passage in sorted(q for q in camera.iterdir() if q.is_dir()):
                rgb = sorted(passage.glob("*_RGB.png"))
                paired = [(f, f.with_name(f.name.replace("_RGB.png", "_depth.png"))) for f in rgb]
                paired = [(r, d) for r, d in paired if d.exists()]
                if paired:
                    tracklets.append(Tracklet(int(person.name), camera_ids[camera.name], [r for r, _ in paired],
                                              [d for _, d in paired], passage.name))
    return tracklets


DATASETS = {"prid2011": load_prid2011, "ilidsvid": load_ilidsvid, "tvrid": load_tvrid}
# TVRID is scored on named camera scenarios; the two-camera video benchmarks use
# their standard camera-0 to camera-1 protocol.
SCENARIO_DATASETS = {"tvrid"}
IMAGE_SIZE = {"prid2011": (256, 128), "ilidsvid": (256, 128), "tvrid": (192, 192)}


# ------------------------------------------------------------------ splits
@dataclass(frozen=True)
class Split:
    seed: int
    train_pids: tuple[int, ...]
    test_pids: tuple[int, ...]


def make_split(tracklets: list[Tracklet], seed: int, train_fraction: float = 0.5) -> Split:
    """Random identity-disjoint split. The same seed always gives the same split."""
    pids = sorted({t.pid for t in tracklets})
    random.Random(f"split-{seed}").shuffle(pids)
    cut = round(len(pids) * train_fraction)
    if not 0 < cut < len(pids):
        raise ValueError("train_fraction leaves an empty train or test set")
    return Split(seed, tuple(sorted(pids[:cut])), tuple(sorted(pids[cut:])))


def sample_images(tracklets: list[Tracklet], pids: tuple[int, ...], k: int | None, seed: int) -> list[tuple[Path, int, int]]:
    """Choose up to k images per identity, returned as (path, pid, cam).

    k=None takes every frame. Otherwise cameras are visited in turn, so k=1 is
    one camera and k=2 is one image from each. A short tracklet simply yields
    fewer images.
    """
    wanted = set(pids)
    by_pid: dict[int, dict[int, list[Path]]] = {}
    for t in tracklets:
        if t.pid in wanted:
            by_pid.setdefault(t.pid, {}).setdefault(t.cam, []).extend(t.paths)
    chosen: list[tuple[Path, int, int]] = []
    for pid in sorted(by_pid):
        rng = random.Random(f"sample-{seed}-{pid}")
        cams = sorted(by_pid[pid])
        rng.shuffle(cams)
        order: dict[int, list[Path]] = {}
        for cam in cams:
            frames = sorted(by_pid[pid][cam])
            rng.shuffle(frames)
            order[cam] = frames
        if k is None:
            chosen += [(p, pid, cam) for cam in cams for p in order[cam]]
            continue
        taken = 0
        position = dict.fromkeys(cams, 0)
        while taken < k and any(position[c] < len(order[c]) for c in cams):
            for cam in cams:
                if taken < k and position[cam] < len(order[cam]):
                    chosen.append((order[cam][position[cam]], pid, cam))
                    position[cam] += 1
                    taken += 1
    return chosen


def evenly_spaced(paths: list[Path], count: int | None) -> list[Path]:
    """Pick `count` frames spread evenly along a tracklet (None keeps them all)."""
    if count is None or len(paths) <= count:
        return list(paths)
    return [paths[i] for i in np.linspace(0, len(paths) - 1, count).round().astype(int)]


def training_frames(tracklets: list[Tracklet], splits: list[int], k_max: int | None) -> set[Path]:
    """Every frame the listed splits can train on, for any k up to k_max.

    Few-image subsets are nested, so k_max covers all smaller k. Generating views
    only for these frames saves time and cannot leak test images into training.
    """
    frames: set[Path] = set()
    for seed in splits:
        split = make_split(tracklets, seed)
        frames |= {path for path, _, _ in sample_images(tracklets, split.train_pids, k_max, seed)}
    return frames


def evaluation_frames(tracklets: list[Tracklet], count: int | None) -> set[Path]:
    """The frames used to represent each tracklet at test time."""
    return {path for t in tracklets for path in evenly_spaced(t.paths, count)}


# ------------------------------------------------------------------ images
def _low_resolution(size: tuple[int, int], low_res: int) -> tuple[int, int]:
    """PIL (width, height) of a simulated camera with `low_res` pixels of height."""
    height, width = size
    return max(1, round(low_res * width / height)), low_res


def read_image(path: Path, size: tuple[int, int], low_res: int | None = None) -> np.ndarray:
    """Decode to a CHW uint8 array resized to (height, width).

    With `low_res` the image first passes through that height, simulating a more
    distant or lower-quality camera while keeping the network input size.
    """
    height, width = size
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if low_res:
            rgb = rgb.resize(_low_resolution(size, low_res), Image.BICUBIC)
        array = np.asarray(rgb.resize((width, height), Image.BILINEAR), dtype=np.uint8)
    return np.moveaxis(array, -1, 0).copy()


def read_raw_depth(path: Path, size: tuple[int, int], low_res: int | None = None) -> np.ndarray:
    """Decode a 16-bit depth map to float32 (1, H, W) in the file's units, with 0 for invalid."""
    height, width = size
    with Image.open(path) as image:
        small = image.resize(_low_resolution(size, low_res), Image.NEAREST) if low_res else image
        depth = np.asarray(small.resize((width, height), Image.NEAREST), dtype=np.float32)
    return depth[None].copy()


def read_depth(path: Path, size: tuple[int, int], low_res: int | None = None) -> np.ndarray:
    """Encode a 16-bit depth map as a 3-channel uint8 image for an RGB backbone.

    Inverse depth is scaled by the 2nd-98th percentile of the valid pixels, per
    image. Invalid pixels (value 0) stay black. Nearest-neighbour resizing avoids
    inventing depth values across object edges.
    """
    depth = read_raw_depth(path, size, low_res)[0]
    valid = depth > 0
    encoded = np.zeros(depth.shape, dtype=np.uint8)
    if valid.sum() > 16:
        inverse = 1.0 / depth[valid]
        low, high = np.percentile(inverse, (2, 98))
        encoded[valid] = (np.clip((inverse - low) / max(high - low, 1e-9), 0, 1) * 254 + 1).astype(np.uint8)
    return np.stack([encoded] * 3)


READERS = {"rgb": read_image, "depth": read_depth, "est_depth": read_depth}


def reader_for(modality: str, low_res: int | None = None):
    """The image reader of a modality, optionally simulating a low-resolution camera."""
    return partial(READERS[modality], low_res=low_res) if low_res else READERS[modality]


ESTIMATED_SUFFIX = "_estdepth.png"


def estimated_depth_path(rgb_path: Path, data_root: str | Path, est_root: str | Path) -> Path:
    """Where the depth estimated from one RGB frame is stored."""
    relative = Path(rgb_path).relative_to(data_root)
    return Path(est_root) / relative.with_name(relative.stem + ESTIMATED_SUFFIX)


def modality_root(modality: str, data_root: str | Path, est_root: str | Path | None) -> Path:
    """The folder that a modality's frame paths are relative to."""
    if modality == "est_depth":
        if est_root is None:
            raise ValueError("modality 'est_depth' needs est_root")
        return Path(est_root)
    return Path(data_root)


def for_modality(tracklets: list[Tracklet], modality: str, data_root: str | Path | None = None,
                 est_root: str | Path | None = None) -> list[Tracklet]:
    """Return tracklets whose `paths` point at the frames of the given modality.

    The order of frames is the same in every modality, so seeded sampling picks
    the same moments in time for RGB, measured depth and estimated depth.
    """
    if modality == "rgb":
        return tracklets
    if modality == "depth":
        if not all(t.aux_paths for t in tracklets):
            raise ValueError("this dataset has no depth frames")
        return [Tracklet(t.pid, t.cam, t.aux_paths, t.paths, t.direction) for t in tracklets]
    if modality == "est_depth":
        if data_root is None or est_root is None:
            raise ValueError("modality 'est_depth' needs data_root and est_root")
        return [Tracklet(t.pid, t.cam, [estimated_depth_path(p, data_root, est_root) for p in t.paths],
                         t.aux_paths, t.direction) for t in tracklets]
    raise ValueError(f"unknown modality: {modality}")


def load_images(paths: list[Path], size: tuple[int, int], reader=read_image, workers: int = 8) -> torch.Tensor:
    """Decode images in parallel into one tensor (N, C, H, W)."""
    if not paths:
        raise ValueError("no images to load")
    with ThreadPoolExecutor(workers) as pool:
        arrays = list(pool.map(lambda p: reader(p, size), paths))
    return torch.from_numpy(np.stack(arrays))


def view_folder(frame_path: Path, root: str | Path, views_root: str | Path) -> Path:
    """Folder holding the rendered views of one frame."""
    return Path(views_root) / Path(frame_path).relative_to(root).with_suffix("")


def synthetic_views(real_path: Path, root: str | Path, views_root: str | Path) -> list[Path]:
    """Rendered views of one frame: <views_root>/<frame relative to root, no extension>/view_*.png.

    `root` is the folder the frame path is relative to, so the renderer stays
    independent of training.
    """
    folder = view_folder(real_path, root, views_root)
    return sorted(folder.glob("view_*.png")) if folder.is_dir() else []
