"""Vectorised re-ID retrieval metrics (CMC and mAP).

The whole computation is done with tensor operations, so it runs on the GPU and
needs no per-query Python loop. `tests/test_metrics.py` checks it against a slow
brute-force reference.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    rank1: float
    rank5: float
    rank10: float
    rank20: float
    mAP: float
    num_queries: int
    num_gallery: int

    def as_dict(self) -> dict:
        return asdict(self)


def cosine_similarity_matrix(query: torch.Tensor, gallery: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Cosine similarity between L2-normalised query and gallery features."""
    query = F.normalize(query, dim=1)
    gallery = F.normalize(gallery, dim=1)
    parts = [query[i:i + chunk] @ gallery.T for i in range(0, query.size(0), chunk)]
    return torch.cat(parts, dim=0)


def fuse_similarities(sims: list[torch.Tensor], weights: list[float] | None = None) -> torch.Tensor:
    """Score-level fusion of several modalities.

    Each similarity matrix is z-scored per query first, so one modality cannot
    dominate only because its scores have a larger spread.
    """
    if not sims:
        raise ValueError("fuse_similarities needs at least one matrix")
    weights = weights or [1.0 / len(sims)] * len(sims)
    if len(weights) != len(sims):
        raise ValueError("one weight is needed per similarity matrix")
    out = torch.zeros_like(sims[0])
    for sim, weight in zip(sims, weights):
        mean = sim.mean(dim=1, keepdim=True)
        std = sim.std(dim=1, keepdim=True).clamp_min(1e-8)
        out += weight * (sim - mean) / std
    return out


@torch.no_grad()
def evaluate_similarity(
    sim: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    q_cams: torch.Tensor | None = None,
    g_cams: torch.Tensor | None = None,
    exclude_same_camera: bool = False,
    q_tracks: torch.Tensor | None = None,
    g_tracks: torch.Tensor | None = None,
    ranks: tuple[int, ...] = (1, 5, 10, 20),
) -> RetrievalResult:
    """Compute CMC and mAP from a query-by-gallery similarity matrix.

    A gallery item is removed for a query when it is the same tracklet as the
    query, or, if `exclude_same_camera` is set, when it shows the same person
    from the same camera. Queries with no remaining correct match are skipped,
    as is standard, and `num_queries` reports how many were really scored.
    """
    device = sim.device
    q_pids, g_pids = q_pids.to(device), g_pids.to(device)
    order = sim.argsort(dim=1, descending=True)
    same_pid = g_pids[order] == q_pids.unsqueeze(1)

    remove = torch.zeros_like(same_pid)
    if exclude_same_camera:
        if q_cams is None or g_cams is None:
            raise ValueError("camera ids are required for exclude_same_camera")
        same_cam = g_cams.to(device)[order] == q_cams.to(device).unsqueeze(1)
        remove |= same_pid & same_cam
    if q_tracks is not None and g_tracks is not None:
        remove |= g_tracks.to(device)[order] == q_tracks.to(device).unsqueeze(1)

    valid = ~remove
    match = same_pid & valid
    has_match = match.any(dim=1)
    if not has_match.any():
        raise ValueError("no query has a valid correct match in the gallery")

    match, valid = match[has_match].float(), valid[has_match].float()
    position = valid.cumsum(dim=1)  # 1-based rank counting only valid items
    cumulative_hits = match.cumsum(dim=1)
    precision = cumulative_hits / position.clamp_min(1)
    average_precision = (precision * match).sum(dim=1) / match.sum(dim=1)

    # Rank of the first correct match, per query.
    first_hit_rank = (position * match + (1 - match) * (position.max() + 1)).min(dim=1).values
    cmc = {k: (first_hit_rank <= k).float().mean().item() for k in ranks}

    return RetrievalResult(
        rank1=100 * cmc[1],
        rank5=100 * cmc[5],
        rank10=100 * cmc[10],
        rank20=100 * cmc[20],
        mAP=100 * average_precision.mean().item(),
        num_queries=int(has_match.sum().item()),
        num_gallery=int(sim.size(1)),
    )
