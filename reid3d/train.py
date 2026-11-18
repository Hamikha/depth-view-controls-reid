"""Train and evaluate one experimental condition (dataset, split, k, arm, modality).

The loop is built for a small GPU. Few-image training sets live on the GPU as
uint8, so there is no DataLoader and no CPU bottleneck. Training length is fixed
in iterations rather than epochs, so every k receives the same amount of
optimisation and differences between k are not caused by training time.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import augment as aug
from . import data as D
from .metrics import cosine_similarity_matrix, evaluate_similarity, fuse_similarities
from .model import ReIDNet, batch_hard_triplet

# The "_weak" arms use rendered views with flip and shift only, to match the `real` arm.
# The "_persp" arms add a random projective warp to real images: the 2D baseline for rendered views.
ARMS = {"real", "real_aug", "synth", "real_synth", "synth_weak", "real_synth_weak", "real_persp", "real_aug_persp"}
SYNTH_ARMS = {"synth", "real_synth", "synth_weak", "real_synth_weak"}
VIEW_ONLY_ARMS = {"synth", "synth_weak"}
MIXED_ARMS = {"real_synth", "real_synth_weak"}
WEAK_ARMS = {"real", "synth_weak", "real_synth_weak", "real_persp"}
PERSPECTIVE_ARMS = {"real_persp", "real_aug_persp"}
# A fusion name ending in a number gives the second modality that weight in percent: rgbd20 is 80% rgb, 20% depth.
FUSION_PAIRS = {"rgbd": ("rgb", "depth"), "rgbe": ("rgb", "est_depth")}
SECOND_WEIGHTS = {"": 0.5, "10": 0.1, "20": 0.2, "30": 0.3}
FUSIONS = {f"{name}{suffix}": pair for name, pair in FUSION_PAIRS.items() for suffix in SECOND_WEIGHTS}
FUSION_WEIGHTS = {f"{name}{suffix}": weight for name in FUSION_PAIRS for suffix, weight in SECOND_WEIGHTS.items()}
SINGLE_MODALITIES = ("rgb", "depth", "est_depth")


@dataclass
class Config:
    dataset: str
    data_root: str
    arm: str = "real_aug"
    modality: str = "rgb"  # rgb, depth, est_depth, or a fusion: rgbd (rgb+depth), rgbe (rgb+est_depth)
    k: int | None = 1
    split_seed: int = 0
    iterations: int = 1500
    identities_per_batch: int = 12
    images_per_identity: int = 4
    height: int = 256
    width: int = 128
    lr: float = 3.5e-4
    weight_decay: float = 5e-4
    id_loss: str = "ce"
    triplet_weight: float = 1.0
    label_smoothing: float = 0.1
    eval_frames: int | None = 16
    query_frames: int | None = None  # 1 gives the single-shot test protocol
    pretrained: bool = True
    synth_root: str | None = None
    synth_tag: str | None = None  # name of the view source, such as "estimated", "measured" or "mesh"
    est_root: str | None = None  # folder of depth maps estimated from RGB
    low_res: int | None = None  # simulated camera height in pixels
    device: str = "cuda"
    gpu_budget_gb: float = 2.0
    amp: bool = True
    backbone: str = "resnet50"

    def key(self) -> str:
        """Unique name of the condition, used so finished runs can be skipped."""
        return (f"{self.dataset}|{self.modality}|{self.arm}|k={self.k}|split={self.split_seed}|it={self.iterations}"
                f"|loss={self.id_loss}|qf={self.query_frames}|views={self.synth_tag}|res={self.low_res}"
                f"|ef={self.eval_frames}"
                # The default backbone adds nothing, so results saved before this option still match.
                + (f"|bb={self.backbone}" if self.backbone != "resnet50" else ""))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Bank:
    """Training images held on the GPU when they fit, otherwise in RAM."""

    def __init__(self, images: torch.Tensor, device: torch.device, budget_gb: float):
        self.device = device
        fits = images.numel() / 1e9 <= budget_gb and device.type == "cuda"
        self.images = images.to(device) if fits else images

    def __getitem__(self, index: torch.Tensor) -> torch.Tensor:
        batch = self.images[index.to(self.images.device)]
        return batch.to(self.device, non_blocking=True)


class PKSampler:
    """Draw P identities and K images each, optionally mixing real and synthetic.

    With very few images per identity K exceeds the number available, so images
    are drawn with replacement; augmentation makes the repeats differ.
    """

    def __init__(self, labels: torch.Tensor, is_synth: torch.Tensor, p: int, k: int, mix: bool, seed: int):
        self.p, self.k, self.mix = p, k, mix
        self.rng = np.random.default_rng(seed)
        labels, is_synth = labels.cpu().numpy(), is_synth.cpu().numpy()
        self.identities = np.unique(labels)
        self.real = {c: np.flatnonzero((labels == c) & ~is_synth) for c in self.identities}
        self.synth = {c: np.flatnonzero((labels == c) & is_synth) for c in self.identities}

    def _pick(self, pool: np.ndarray, n: int) -> np.ndarray:
        return self.rng.choice(pool, n, replace=len(pool) < n)

    def batch(self) -> torch.Tensor:
        replace = len(self.identities) < self.p
        chosen = self.rng.choice(self.identities, self.p, replace=replace)
        picks: list[np.ndarray] = []
        for c in chosen:
            real, synth = self.real[c], self.synth[c]
            if self.mix and len(real) and len(synth):
                half = self.k // 2
                picks += [self._pick(real, self.k - half), self._pick(synth, half)]
            else:
                picks.append(self._pick(real if len(real) else synth, self.k))
        return torch.from_numpy(np.concatenate(picks))


def build_training_set(cfg: Config, tracklets: list[D.Tracklet], split: D.Split, modality: str):
    """Load the images of this arm, returning images, labels, flags and counts.

    Rendered views are read as they are, because the rendering step already
    applied any simulated low resolution to its source images.
    """
    size = (cfg.height, cfg.width)
    real = D.sample_images(tracklets, split.train_pids, cfg.k, cfg.split_seed)
    classes = {pid: i for i, pid in enumerate(sorted({pid for _, pid, _ in real}))}
    images, labels, flags = [], [], []
    missing = rendered = 0
    if cfg.arm in SYNTH_ARMS:
        if not cfg.synth_root:
            raise ValueError(f"arm '{cfg.arm}' needs synth_root")
        root = D.modality_root(modality, cfg.data_root, cfg.est_root)
        paths, view_labels = [], []
        for path, pid, _ in real:
            views = D.synthetic_views(path, root, cfg.synth_root)
            missing += not views
            paths += views
            view_labels += [classes[pid]] * len(views)
        if not paths:
            raise ValueError(f"no rendered {modality} views found under {cfg.synth_root}")
        rendered = len(paths)
        images.append(D.load_images(paths, size, D.READERS[modality]))
        labels += view_labels
        flags += [True] * len(paths)
    if cfg.arm not in VIEW_ONLY_ARMS:
        images.append(D.load_images([p for p, _, _ in real], size, D.reader_for(modality, cfg.low_res)))
        labels += [classes[pid] for _, pid, _ in real]
        flags += [False] * len(real)
    return (torch.cat(images), torch.tensor(labels), torch.tensor(flags),
            {"train_images": len(labels), "train_real": len(real), "train_views": rendered,
             "num_classes": len(classes), "missing_renders": missing})


def lr_at(step: int, total: int, warmup: int = 100) -> float:
    if step < warmup:
        return 0.1 + 0.9 * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def train_model(cfg: Config, images: torch.Tensor, labels: torch.Tensor, flags: torch.Tensor,
                num_classes: int, device: torch.device) -> ReIDNet:
    model = ReIDNet(num_classes, pretrained=cfg.pretrained, id_loss=cfg.id_loss, backbone=cfg.backbone).to(device)
    groups = model.parameter_groups(cfg.lr)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    bank = Bank(images, device, cfg.gpu_budget_gb)
    labels = labels.to(device)
    mix = cfg.arm in MIXED_ARMS
    sampler = PKSampler(labels, flags, cfg.identities_per_batch, cfg.images_per_identity, mix, cfg.split_seed)
    if cfg.arm in PERSPECTIVE_ARMS:
        transform = aug.weak_perspective if cfg.arm in WEAK_ARMS else aug.strong_perspective
    else:
        transform = aug.weak if cfg.arm in WEAK_ARMS else aug.strong

    model.train()
    for step in range(cfg.iterations):
        scale = lr_at(step, cfg.iterations)
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * scale
        index = sampler.batch()
        batch, target = transform(bank[index]), labels[index.to(device)]
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            logits, features = model(batch, target)
            loss = F.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
            loss = loss + cfg.triplet_weight * batch_hard_triplet(features, target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return model


@torch.no_grad()
def extract_features(cfg: Config, model: ReIDNet, tracklets: list[D.Tracklet], frames: int | None,
                     device: torch.device, modality: str, clock: dict | None = None, batch_size: int = 128):
    """One feature per tracklet: the mean over its frames, including flipped frames."""
    model.eval()
    size = (cfg.height, cfg.width)
    paths, owners = [], []
    for i, tracklet in enumerate(tracklets):
        selected = D.evenly_spaced(tracklet.paths, frames)
        paths += selected
        owners += [i] * len(selected)
    images = D.load_images(paths, size, D.reader_for(modality, cfg.low_res))
    features = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        batch = aug.normalize(aug.to_float(images[start:start + batch_size].to(device, non_blocking=True)))
        with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
            out = model(batch) + model(batch.flip(-1))
        features.append(F.normalize(out.float(), dim=1))
    if clock is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        clock["seconds"] += time.perf_counter() - started
        clock["frames"] += len(images)
    owners_tensor = torch.tensor(owners, device=device)
    features_tensor = torch.cat(features)
    pooled = torch.zeros(len(tracklets), features_tensor.size(1), device=device)
    pooled.index_add_(0, owners_tensor, features_tensor)
    return F.normalize(pooled, dim=1)


def test_tracklets(cfg: Config, tracklets: list[D.Tracklet], split: D.Split):
    """Return (query, gallery) test tracklets for this dataset's protocol."""
    wanted = set(split.test_pids)
    test = sorted((t for t in tracklets if t.pid in wanted), key=lambda t: (t.pid, t.cam, t.direction))
    if cfg.dataset in D.SCENARIO_DATASETS:
        return test, test
    return [t for t in test if t.cam == 0], [t for t in test if t.cam == 1]


def similarity_matrix(cfg: Config, model: ReIDNet, queries: list[D.Tracklet], gallery: list[D.Tracklet],
                      modality: str, device: torch.device, clock: dict | None = None) -> torch.Tensor:
    q_frames = cfg.query_frames if cfg.query_frames is not None else cfg.eval_frames
    q_features = extract_features(cfg, model, queries, q_frames, device, modality, clock)
    g_features = extract_features(cfg, model, gallery, cfg.eval_frames, device, modality, clock)
    return cosine_similarity_matrix(q_features, g_features)


def tvrid_scenarios(tracklets: list[D.Tracklet]) -> dict[str, list[tuple[list[int], list[int]]]]:
    """Index lists for the three TVRID retrieval scenarios.

    Each scenario matches one group of tracklets against a different group, in
    both directions. Same-camera people from other groups are never in the
    gallery, following the benchmark, so the scene background cannot dominate.
    """
    def select(cam: int, direction: str | None = None) -> list[int]:
        return [i for i, t in enumerate(tracklets) if t.cam == cam and (direction is None or t.direction == direction)]

    camera = D.TVRID_CAMERAS
    scenarios: dict[str, list[tuple[list[int], list[int]]]] = {"within_camera": [], "ascent_descent": [], "flat_oblique": []}
    for cam in sorted({t.cam for t in tracklets}):
        directions = sorted({t.direction for t in tracklets if t.cam == cam})
        if len(directions) == 2:
            first, second = select(cam, directions[0]), select(cam, directions[1])
            scenarios["within_camera"] += [(first, second), (second, first)]
    for name, (a, b) in {"ascent_descent": ("upward", "downward"), "flat_oblique": ("flat", "upsideDown")}.items():
        left, right = select(camera[a]), select(camera[b])
        if left and right:
            scenarios[name] += [(left, right), (right, left)]
    return {name: pairs for name, pairs in scenarios.items() if pairs}


def score(cfg: Config, similarity: torch.Tensor, queries: list[D.Tracklet], gallery: list[D.Tracklet]) -> dict:
    device = similarity.device
    tensor = lambda values: torch.tensor(values, device=device)
    if cfg.dataset not in D.SCENARIO_DATASETS:
        return evaluate_similarity(similarity, tensor([t.pid for t in queries]), tensor([t.pid for t in gallery])).as_dict()

    pids = tensor([t.pid for t in queries])
    names = ("rank1", "rank5", "rank10", "rank20", "mAP")
    per_scenario: dict[str, dict[str, float]] = {}
    total_queries = 0
    for scenario, pairs in tvrid_scenarios(queries).items():
        results = []
        for q_index, g_index in pairs:
            q, g = tensor(q_index), tensor(g_index)
            results.append(evaluate_similarity(similarity[q][:, g], pids[q], pids[g]).as_dict())
        per_scenario[scenario] = {n: sum(r[n] for r in results) / len(results) for n in names}
        total_queries += sum(r["num_queries"] for r in results)
    overall = {n: sum(v[n] for v in per_scenario.values()) / len(per_scenario) for n in names}
    return {**overall, "num_queries": total_queries, "num_gallery": len(gallery), "scenarios": per_scenario}


def train_and_score(cfg: Config, tracklets: list[D.Tracklet], split: D.Split, modality: str,
                    device: torch.device) -> dict:
    """Train one single-modality model and score it on the test identities."""
    seed_everything(cfg.split_seed)
    view = D.for_modality(tracklets, modality, cfg.data_root, cfg.est_root)
    started = time.perf_counter()
    images, labels, flags, counts = build_training_set(cfg, view, split, modality)
    model = train_model(cfg, images, labels, flags, counts["num_classes"], device)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    queries, gallery = test_tracklets(cfg, view, split)
    clock = {"seconds": 0.0, "frames": 0}
    similarity = similarity_matrix(cfg, model, queries, gallery, modality, device, clock)
    result = score(cfg, similarity, queries, gallery)
    eval_seconds = time.perf_counter() - started
    del model, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"similarity": similarity.cpu(), "result": result, "counts": counts, "queries": queries, "gallery": gallery,
            "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "forward_ms_per_frame": 1000.0 * clock["seconds"] / max(1, clock["frames"])}


def manifest_value(root: str | None, key: str) -> float | None:
    """Read a recorded cost from the manifest written by a preparation tool."""
    path = Path(root) / "manifest.json" if root else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8")).get(key)


def preparation_costs(cfg: Config, modalities: tuple[str, ...]) -> dict:
    """Milliseconds of preparation per image, at training time and at test time."""
    estimation = manifest_value(cfg.est_root, "ms_per_image") if "est_depth" in modalities else None
    views = manifest_value(cfg.synth_root, "total_ms_per_image") if cfg.arm in SYNTH_ARMS else None
    known = [value for value in (estimation, views) if value is not None]
    return {"train_prep_ms_per_image": max(known) if known else 0.0, "test_prep_ms_per_frame": estimation or 0.0}


def run_condition(cfg: Config, tracklets: list[D.Tracklet] | None = None, cache: dict | None = None) -> dict:
    """Run one full condition and return a JSON-serialisable result record.

    Pass the same `cache` across conditions and a fusion reuses single-modality
    models that were already trained, instead of training them again.
    """
    if cfg.arm not in ARMS:
        raise ValueError(f"unknown arm '{cfg.arm}', expected one of {sorted(ARMS)}")
    modalities = FUSIONS.get(cfg.modality, (cfg.modality,))
    if not set(modalities) <= set(SINGLE_MODALITIES):
        raise ValueError(f"unknown modality '{cfg.modality}'")
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if tracklets is None:
        tracklets = D.DATASETS[cfg.dataset](cfg.data_root)
    split = D.make_split(tracklets, cfg.split_seed)
    cache = {} if cache is None else cache
    entries = {}
    for modality in modalities:
        name = replace(cfg, modality=modality).key()
        if name not in cache:
            cache[name] = train_and_score(cfg, tracklets, split, modality, device)
        entries[modality] = cache[name]

    first = entries[modalities[0]]
    if cfg.modality in FUSIONS:
        second = FUSION_WEIGHTS[cfg.modality]
        fused = fuse_similarities([entries[m]["similarity"].to(device) for m in modalities], [1.0 - second, second])
        result = score(cfg, fused, first["queries"], first["gallery"])
        extras = {f"{m}_{name}": entries[m]["result"][name] for m in modalities for name in ("rank1", "mAP")}
    else:
        result, extras = first["result"], {}
    return {"key": cfg.key(), "config": asdict(cfg), **first["counts"], **result, **extras,
            **preparation_costs(cfg, modalities),
            "forward_ms_per_frame": round(sum(e["forward_ms_per_frame"] for e in entries.values()), 3),
            "train_seconds": round(sum(e["train_seconds"] for e in entries.values()), 1),
            "eval_seconds": round(sum(e["eval_seconds"] for e in entries.values()), 1)}
