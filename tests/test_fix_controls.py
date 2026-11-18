"""Tests for the controls a referee asked for: matched flat views, shuffled depth and a 2D perspective warp."""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import analyze  # noqa: E402
import render_views  # noqa: E402
import run_paper  # noqa: E402
from reid3d import augment  # noqa: E402
from reid3d import data as D  # noqa: E402
from reid3d import geometry as G  # noqa: E402
from reid3d import train as T  # noqa: E402
from test_proposed_study import (DEVICE, fake_depth_model, fake_prid, fake_tvrid, make_results,  # noqa: E402,F401
                                 prepare_tvrid, result_row)


def person_scene():
    """A bright person left of centre, nearer than the background."""
    rgb = torch.full((1, 3, 64, 48), 60, dtype=torch.uint8)
    rgb[:, :, 8:56, 6:22] = 220
    depth = torch.full((1, 1, 64, 48), 1.3)
    depth[:, :, 8:56, 6:22] = 1.0
    return rgb, depth


def test_matched_flat_views_take_the_person_layout_from_estimated_depth(monkeypatch):
    rgb, depth = person_scene()
    camera = G.Camera.centered(64, 48, 1.2 * 64)
    masks = []
    original = G.split_near_far
    monkeypatch.setattr(G, "split_near_far", lambda z: masks.append(original(z)) or masks[-1])
    flat = torch.full_like(depth, 1.15)
    G.render_views(rgb, depth, camera, [(30.0, 0.0)])
    matched = G.render_views(rgb, flat, camera, [(30.0, 0.0)], layout=depth)
    G.render_views(rgb, flat, camera, [(30.0, 0.0)])
    estimated_mask, matched_mask, first_flat_mask = masks
    assert torch.equal(estimated_mask, matched_mask)
    assert 0.1 < estimated_mask.float().mean().item() < 0.5
    # The first flat control had no person at all: every pixel counted as near.
    assert first_flat_mask.all()
    # Matched flat views still hold no shape: their depth is one rotated plane, so 1/z is affine in the pixel position.
    # (Warped points land on whole pixels, so the fit is close but not exact.) A real shape fails the same test.
    def plane_residual(view: torch.Tensor) -> float:
        rows, columns = torch.nonzero(view > 0, as_tuple=True)
        design = torch.stack([columns.double(), rows.double(), torch.ones_like(rows, dtype=torch.double)], dim=1)
        inverse = 1.0 / view[rows, columns].double()
        fit = torch.linalg.lstsq(design, inverse.unsqueeze(1)).solution
        return (design @ fit).squeeze(1).sub(inverse).abs().max().item()

    assert plane_residual(matched.depth[0, 0, 0]) < 0.03
    assert plane_residual(G.render_views(rgb, depth, camera, [(30.0, 0.0)]).depth[0, 0, 0]) > 0.1


def test_shuffled_depth_is_borrowed_from_another_frame():
    paths = [pathlib.Path(f"p{i}.png") for i in range(7)]
    borrowed = render_views.donors(paths)
    assert sorted(borrowed) == sorted(paths) and all(a != b for a, b in zip(paths, borrowed))
    assert render_views.donors(paths) == borrowed  # the same pairs every time


def test_matched_flat_and_shuffled_views_are_rendered_and_found_by_training(tmp_path, fake_tvrid):
    estimated, _, _ = prepare_tvrid(fake_tvrid, tmp_path)
    tracklets = D.load_tvrid(fake_tvrid)
    frame = D.sample_images(tracklets, D.make_split(tracklets, 0).train_pids, 1, 0)[0][0]
    for geometry in ("flat_matched", "shuffled"):
        out = tmp_path / f"views_{geometry}"
        render_views.main(["--dataset", "tvrid", "--data-root", str(fake_tvrid), "--splits", "0", "--k-max", "2",
                           "--device", DEVICE, "--geometry", geometry, "--est-root", str(estimated), "--out", str(out),
                           "--min-visible", "0", "--angles", "0:15", "0:-15"])
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["geometry"] == geometry and manifest["views_written"] == 2 * manifest["frames_total"]
        assert len(D.synthetic_views(frame, fake_tvrid, out)) == 2


def test_perspective_warp_is_batched_bounded_and_off_when_not_selected():
    batch = torch.rand(6, 3, 32, 16)
    assert torch.allclose(augment.random_perspective(batch, p=0.0), batch, atol=1e-5)
    warped = augment.random_perspective(batch, p=1.0)
    assert warped.shape == batch.shape and torch.isfinite(warped).all()
    assert warped.min() >= batch.min() - 1e-5 and warped.max() <= batch.max() + 1e-5
    assert not torch.allclose(warped, batch, atol=1e-3)


def test_perspective_strength_lies_between_the_fifteen_and_thirty_degree_views():
    """The largest corner shift of the warp (0.15) is that of a plane rotated by about 23 degrees of yaw."""
    height, width = 256, 128
    focal = 1.2 * max(height, width)
    camera = G.Camera.centered(height, width, focal)
    plane = torch.full((1, 1, height, width), 1.15)
    corners = G.lift_points(plane[:, 0], camera)[0, [0, 0, -1, -1], [0, -1, 0, -1]]
    pivot = torch.tensor([0.0, 0.0, 1.15])

    def largest_shift(yaw: float) -> float:
        rotation = G.rotation_matrix(yaw, 0.0, "cpu")
        moved = (corners - pivot) @ rotation.T + pivot
        column = focal * moved[:, 0] / moved[:, 2] + camera.cx
        row = focal * moved[:, 1] / moved[:, 2] + camera.cy
        shift = torch.stack([(column - (focal * corners[:, 0] / corners[:, 2] + camera.cx)) / (width / 2),
                             (row - (focal * corners[:, 1] / corners[:, 2] + camera.cy)) / (height / 2)], dim=1)
        return shift.abs().max().item()

    assert largest_shift(15.0) < 0.15 < largest_shift(30.0)
    assert abs(largest_shift(23.0) - 0.15) < 0.01


def test_perspective_arms_train_with_the_matching_augmentation(tmp_path, fake_tvrid, monkeypatch):
    tracklets = D.load_tvrid(fake_tvrid)
    split = D.make_split(tracklets, 0)
    settings = dict(dataset="tvrid", data_root=str(fake_tvrid), k=2, height=64, width=64)
    used = []
    for name in ("weak", "strong", "weak_perspective", "strong_perspective"):
        original = getattr(augment, name)
        monkeypatch.setattr(augment, name, lambda batch, name=name, original=original: used.append(name) or original(batch))
    for arm in ("real_persp", "real_aug_persp"):
        images, labels, flags, counts = T.build_training_set(T.Config(arm=arm, **settings), tracklets, split, "rgb")
        assert counts["train_views"] == 0 and not flags.any()
        T.train_model(T.Config(arm=arm, iterations=1, identities_per_batch=4, pretrained=False, device=DEVICE, **settings),
                      images, labels, flags, counts["num_classes"], torch.device(DEVICE))
    assert used == ["weak_perspective", "strong_perspective"]


def test_fix_stage_plans_the_controls(tmp_path, fake_prid, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_paper.subprocess, "run", lambda command, **kwargs: calls.append([str(part) for part in command]))
    run_paper.main(["--stage", "fix", "--prid", str(fake_prid), "--work", str(tmp_path / "w"),
                    "--results", str(tmp_path / "r"), "--splits", "0"])
    # per split: matched flat 3 arms x 4 k, shuffled 2 arms x 2 k, perspective 2 arms x 4 k,
    # matched flat angles 2 arms x 3 sets x 2 k, DINOv2 matched flat 2 arms x 4 k and perspective 1 arm x 4 k
    assert "Planned trainings: 48" in capsys.readouterr().out
    renders = [c for c in calls if c[1].endswith("render_views.py")]
    geometries = sorted(c[c.index("--geometry") + 1] for c in renders)
    assert geometries == ["flat_matched"] * 4 + ["shuffled"]
    outs = {pathlib.Path(c[c.index("--out") + 1]).name for c in renders}
    assert outs == {"views_flatm", "views_shuffled", "views_flatm_a15", "views_flatm_a30", "views_flatm_a45"}
    grids = [c for c in calls if c[1].endswith("run_grid.py")]
    assert all(pathlib.Path(c[c.index("--out") + 1]).name.startswith("fx_") for c in grids)
    assert any("real_aug_persp" in c for c in grids)


def test_fix_report_pairs_each_control_with_the_views_it_controls(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    rows = []
    for split in range(8):
        wobble = split % 2
        rows += [result_row("rgb", "real", None, split, 50.0 + wobble),
                 result_row("rgb", "real_persp", None, split, 51.0 + wobble),
                 result_row("rgb", "real_synth_weak", "estimated", split, 56.0 + wobble),
                 result_row("rgb", "real_synth_weak", "flat", split, 53.0 + wobble),
                 result_row("rgb", "real_synth_weak", "flatm", split, 54.0 + wobble),
                 result_row("rgb", "real_synth_weak", "shuffled", split, 52.0 + wobble),
                 result_row("rgb", "real_synth_weak", "estimated_a15", split, 55.0 + wobble),
                 result_row("rgb", "real_synth_weak", "flatm_a15", split, 53.0 + wobble)]
    make_results(path, rows)
    analyze.main([str(path), "--question", "fix"])
    output = capsys.readouterr().out
    shape = output.split("hole filling)")[1].split("Fix:")[0]
    assert "(rank1, 2 comparisons)" in shape and "flatm_a15 -> rgb/real_synth_weak/estimated_a15" in shape.replace("  ", " ")
    assert "flatm -> rgb/real_synth_weak/estimated_a15" not in shape.replace("  ", " ")
    borrowed = output.split("borrowed from another image")[1].split("Fix:")[0]
    assert "shuffled -> rgb/real_synth_weak/estimated" in borrowed.replace("  ", " ") and "diff  +4.0" in borrowed
    rivals = output.split("against the random 2D perspective warp")[1]
    assert "rgb/real_persp" in rivals and "_a15" not in rivals and "(rank1, 4 comparisons)" in rivals
