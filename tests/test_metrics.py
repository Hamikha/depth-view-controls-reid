"""Check the vectorised metrics against a slow, obviously correct reference."""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reid3d.metrics import evaluate_similarity, fuse_similarities  # noqa: E402
from reid3d.stats import compare_arms, sign_flip_pvalue, wilson_interval  # noqa: E402


def brute_force(sim, q_pids, g_pids, q_cams, g_cams, exclude_same_camera):
    first_ranks, aps = [], []
    for i in range(sim.size(0)):
        order = sim[i].argsort(descending=True).tolist()
        kept = [j for j in order if not (exclude_same_camera and g_pids[j] == q_pids[i] and g_cams[j] == q_cams[i])]
        matches = [g_pids[j] == q_pids[i] for j in kept]
        if not any(matches):
            continue
        first_ranks.append(matches.index(True) + 1)
        hits, total = 0, 0.0
        for rank, is_match in enumerate(matches, 1):
            if is_match:
                hits += 1
                total += hits / rank
        aps.append(total / hits)
    return first_ranks, aps


@pytest.mark.parametrize("exclude", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force(seed, exclude):
    generator = torch.Generator().manual_seed(seed)
    sim = torch.rand(40, 120, generator=generator)
    q_pids = torch.randint(0, 15, (40,), generator=generator)
    g_pids = torch.randint(0, 15, (120,), generator=generator)
    q_cams = torch.randint(0, 3, (40,), generator=generator)
    g_cams = torch.randint(0, 3, (120,), generator=generator)
    result = evaluate_similarity(sim, q_pids, g_pids, q_cams, g_cams, exclude_same_camera=exclude)
    first_ranks, aps = brute_force(sim, q_pids.tolist(), g_pids.tolist(), q_cams.tolist(), g_cams.tolist(), exclude)
    assert result.num_queries == len(first_ranks)
    assert result.rank1 == pytest.approx(100 * sum(r <= 1 for r in first_ranks) / len(first_ranks), abs=1e-4)
    assert result.rank5 == pytest.approx(100 * sum(r <= 5 for r in first_ranks) / len(first_ranks), abs=1e-4)
    assert result.rank20 == pytest.approx(100 * sum(r <= 20 for r in first_ranks) / len(first_ranks), abs=1e-4)
    assert result.mAP == pytest.approx(100 * sum(aps) / len(aps), abs=1e-3)


def test_perfect_and_worst_ranking():
    pids = torch.arange(10)
    perfect = evaluate_similarity(torch.eye(10), pids, pids)
    assert perfect.rank1 == 100 and perfect.mAP == 100
    worst = evaluate_similarity(1 - torch.eye(10), pids, pids)
    assert worst.rank1 == 0 and worst.rank5 == 0
    assert worst.mAP == pytest.approx(10.0)


def test_same_tracklet_is_removed():
    sim = torch.tensor([[0.9, 0.8, 0.1]])
    result = evaluate_similarity(sim, torch.tensor([0]), torch.tensor([0, 1, 0]),
                                 q_tracks=torch.tensor([7]), g_tracks=torch.tensor([7, 8, 9]))
    assert result.rank1 == 0  # the query's own tracklet must not count as a hit
    assert result.rank5 == 100


def test_fusion_is_scale_invariant():
    generator = torch.Generator().manual_seed(0)
    a, b = torch.rand(5, 9, generator=generator), torch.rand(5, 9, generator=generator)
    assert torch.allclose(fuse_similarities([a, b]), fuse_similarities([a, b * 1000.0]), atol=1e-5)


def test_chance_level_of_random_scores():
    # Rank-20 from random scores on a 100-person gallery should be close to 20%.
    generator = torch.Generator().manual_seed(1)
    pids = torch.arange(100).repeat(30)
    sim = torch.rand(3000, 100, generator=generator)
    result = evaluate_similarity(sim, pids, torch.arange(100))
    assert 18 < result.rank20 < 22


def test_wilson_reproduces_paper_interval():
    low, high = wilson_interval(8, 66)
    assert low == pytest.approx(6.3, abs=0.1)
    assert high == pytest.approx(22.1, abs=0.1)


def test_sign_flip_exact_values():
    assert sign_flip_pvalue([1.0] * 10) == pytest.approx(2 / 1024)
    assert sign_flip_pvalue([1.0, -1.0]) == 1.0


def test_gate_needs_a_real_consistent_gain():
    baseline = {s: 50.0 + s % 3 for s in range(10)}
    clear_gain = {s: baseline[s] + 5.0 for s in range(10)}
    assert compare_arms(baseline, clear_gain).passes_gate
    tiny_gain = {s: baseline[s] + 0.1 for s in range(10)}
    assert not compare_arms(baseline, tiny_gain).passes_gate  # below between-split variation
    worse = {s: baseline[s] - 5.0 for s in range(10)}
    assert not compare_arms(baseline, worse).passes_gate
