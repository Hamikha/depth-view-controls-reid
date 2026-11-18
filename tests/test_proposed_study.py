"""End-to-end checks of the proposed study on small fake datasets.

The depth network is replaced by a stub, so these tests need no model download.
"""
import json
import pathlib
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import analyze  # noqa: E402
import estimate_depth  # noqa: E402
import mesh_views  # noqa: E402
import render_views  # noqa: E402
import run_grid  # noqa: E402
import run_paper  # noqa: E402
from reid3d import data as D  # noqa: E402
from reid3d import monocular  # noqa: E402
from reid3d import train as T  # noqa: E402
from reid3d.stats import holm_adjust  # noqa: E402
from test_data_and_training import make_fake_prid, make_fake_tvrid  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StubDepth:
    """Stands in for the depth network: brighter pixels are treated as closer."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.float().mean(dim=1, keepdim=True).div(255).add(0.05).to(DEVICE)


@pytest.fixture(autouse=True)
def fake_depth_model(monkeypatch):
    monkeypatch.setattr(estimate_depth, "MonocularDepth", lambda device, name: StubDepth())


@pytest.fixture(scope="module")
def fake_tvrid(tmp_path_factory):
    return make_fake_tvrid(tmp_path_factory.mktemp("tvrid"))


@pytest.fixture(scope="module")
def fake_prid(tmp_path_factory):
    return make_fake_prid(tmp_path_factory.mktemp("prid"), persons=16, frames=30)


def prepare_tvrid(root, work, splits=("0",), k_max=2):
    estimated, measured_views, estimated_views = work / "est", work / "views_measured", work / "views_estimated"
    common = ["--dataset", "tvrid", "--data-root", str(root), "--splits", *splits, "--k-max", str(k_max), "--device", DEVICE]
    estimate_depth.main([*common, "--out", str(estimated), "--eval-frames", "2", "--batch-size", "4"])
    render_views.main([*common, "--geometry", "measured", "--out", str(measured_views), "--angles", "0:15", "0:-15"])
    render_views.main([*common, "--geometry", "estimated", "--est-root", str(estimated), "--out", str(estimated_views), "--min-visible", "0",
                       "--angles", "0:15", "0:-15"])
    return estimated, measured_views, estimated_views


def test_low_resolution_keeps_size_but_removes_detail(tmp_path):
    noise = np.random.default_rng(0).integers(0, 255, (128, 64, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(noise).save(path)
    sharp = D.read_image(path, (128, 64))
    blurred = D.read_image(path, (128, 64), low_res=16)
    assert blurred.shape == sharp.shape
    assert np.abs(np.diff(blurred.astype(int), axis=2)).mean() < 0.5 * np.abs(np.diff(sharp.astype(int), axis=2)).mean()


def test_estimated_depth_uses_the_same_moments_as_rgb(tmp_path, fake_tvrid):
    tracklets = D.load_tvrid(fake_tvrid)
    split = D.make_split(tracklets, 0)
    view = D.for_modality(tracklets, "est_depth", fake_tvrid, tmp_path)
    rgb = D.sample_images(tracklets, split.train_pids, 4, 0)
    estimated = D.sample_images(view, split.train_pids, 4, 0)
    assert [p.name.replace("_RGB.png", "") for p, _, _ in rgb] == [p.name.replace("_RGB_estdepth.png", "") for p, _, _ in estimated]
    assert D.for_modality(tracklets, "depth")[0].paths[0].name.endswith("_depth.png")


def test_training_frames_are_nested_and_only_from_training_people(fake_prid):
    tracklets = D.load_prid2011(fake_prid)
    small, large = D.training_frames(tracklets, [0], 2), D.training_frames(tracklets, [0], 5)
    assert small <= large
    test_people = set(D.make_split(tracklets, 0).test_pids)
    people = {int(frame.parent.name.split("_")[1]) for frame in large}
    assert not people & test_people


def test_condition_names_separate_view_sources_and_resolutions():
    base = T.Config(dataset="prid2011", data_root=".", arm="real_synth", synth_tag="estimated")
    assert len({base.key(), T.replace(base, synth_tag="mesh").key(), T.replace(base, low_res=64).key(),
                T.replace(base, eval_frames=2).key()}) == 4


def test_estimated_depth_is_saved_scored_and_resumable(tmp_path, fake_tvrid):
    out = tmp_path / "est"
    arguments = ["--dataset", "tvrid", "--data-root", str(fake_tvrid), "--out", str(out), "--splits", "0", "--k-max", "1",
                 "--eval-frames", "2", "--batch-size", "4", "--device", DEVICE]
    estimate_depth.main(arguments)
    manifest = json.loads((out / "manifest.json").read_text())
    saved = list(out.rglob("*_estdepth.png"))
    assert saved and manifest["frames_processed"] == len(saved) and "sensor_absrel" in manifest
    assert D.read_depth(saved[0], (64, 64)).max() > 0
    estimate_depth.main(arguments)
    assert json.loads((out / "manifest.json").read_text())["frames_processed"] == 0


def test_rendering_writes_colour_and_depth_views_only_for_training_people(tmp_path, fake_tvrid):
    _, measured, estimated = prepare_tvrid(fake_tvrid, tmp_path)
    tracklets = D.load_tvrid(fake_tvrid)
    split = D.make_split(tracklets, 0)
    frame = D.sample_images(tracklets, split.train_pids, 1, 0)[0][0]
    assert len(D.synthetic_views(frame, fake_tvrid, measured)) == 2
    depth_frame = frame.with_name(frame.name.replace("_RGB.png", "_depth.png"))
    assert len(D.synthetic_views(depth_frame, fake_tvrid, measured)) == 2
    assert np.asarray(Image.open(D.synthetic_views(depth_frame, fake_tvrid, measured)[0])).dtype == np.uint16
    rendered_people = {int(p.relative_to(measured).parts[1]) for p in measured.rglob("view_*.png")}
    assert not rendered_people & set(split.test_pids)
    assert json.loads((estimated / "manifest.json").read_text())["total_ms_per_image"] > 0


def test_measured_geometry_needs_a_depth_sensor(tmp_path, fake_prid):
    with pytest.raises(SystemExit):
        render_views.main(["--dataset", "prid2011", "--data-root", str(fake_prid), "--geometry", "measured",
                           "--out", str(tmp_path / "v"), "--splits", "0", "--k-max", "1", "--device", DEVICE])


def test_real_synth_training_set_uses_views_of_training_people_only(tmp_path, fake_tvrid, monkeypatch):
    _, measured, _ = prepare_tvrid(fake_tvrid, tmp_path)
    tracklets = D.load_tvrid(fake_tvrid)
    split = D.make_split(tracklets, 0)
    captured = []
    original = D.load_images

    def spy(paths, size, reader=D.read_image, workers=8):
        captured.extend(paths)
        return original(paths, size, reader, workers)

    monkeypatch.setattr(D, "load_images", spy)
    cfg = T.Config(dataset="tvrid", data_root=str(fake_tvrid), arm="real_synth", k=2, synth_root=str(measured), height=64, width=64)
    images, labels, flags, counts = T.build_training_set(cfg, tracklets, split, "rgb")
    assert counts["train_real"] == len(split.train_pids) * 2
    assert counts["train_views"] == counts["train_real"] * 2 and flags.sum().item() == counts["train_views"]
    assert images.shape[0] == labels.shape[0] == counts["train_images"]
    people = {int(p.parts[p.parts.index("train") + 1]) for p in captured}
    assert people == set(split.train_pids)


def test_missing_views_stop_the_run_instead_of_silently_training_without_them(tmp_path, fake_tvrid):
    (tmp_path / "empty").mkdir()
    tracklets = D.load_tvrid(fake_tvrid)
    cfg = T.Config(dataset="tvrid", data_root=str(fake_tvrid), arm="real_synth", k=1, synth_root=str(tmp_path / "empty"),
                   height=64, width=64)
    with pytest.raises(ValueError, match="no rendered"):
        T.build_training_set(cfg, tracklets, D.make_split(tracklets, 0), "rgb")


def test_fusion_reuses_trained_models(tmp_path, fake_tvrid, monkeypatch):
    estimated, _, _ = prepare_tvrid(fake_tvrid, tmp_path)
    calls = []
    original = T.train_model
    monkeypatch.setattr(T, "train_model", lambda cfg, *rest: calls.append(cfg.modality) or original(cfg, *rest))
    tracklets, cache = D.load_tvrid(fake_tvrid), {}
    settings = dict(dataset="tvrid", data_root=str(fake_tvrid), k=1, iterations=2, identities_per_batch=4, pretrained=False,
                    eval_frames=2, est_root=str(estimated), device=DEVICE, height=64, width=64)
    records = {m: T.run_condition(T.Config(modality=m, **settings), tracklets, cache) for m in ("rgb", "depth", "est_depth", "rgbd", "rgbe")}
    assert len(calls) == 3  # the two fusions trained nothing new
    assert {"rgb_rank1", "est_depth_rank1"} <= set(records["rgbe"])
    assert records["est_depth"]["test_prep_ms_per_frame"] > 0
    assert records["depth"]["test_prep_ms_per_frame"] == 0
    assert records["rgb"]["forward_ms_per_frame"] > 0


def test_grid_and_reports_run_end_to_end(tmp_path, fake_tvrid, capsys):
    estimated, measured, views_estimated = prepare_tvrid(fake_tvrid, tmp_path)
    out = tmp_path / "q2.jsonl"
    common = ["--dataset", "tvrid", "--data-root", str(fake_tvrid), "--k", "1", "--splits", "0", "--iterations", "2",
              "--eval-frames", "2", "--no-pretrained", "--device", DEVICE, "--out", str(out)]
    run_grid.main([*common, "--modalities", "depth", "est_depth", "rgb", "rgbd", "--arms", "real_aug", "--est-root", str(estimated)])
    run_grid.main([*common, "--modalities", "rgb", "--arms", "real_synth",
                   "--views", f"measured={measured}", f"estimated={views_estimated}"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 6
    assert {r["config"]["synth_tag"] for r in rows} == {None, "measured", "estimated"}
    assert all(r["train_prep_ms_per_image"] > 0 for r in rows if r["config"]["synth_tag"])

    # Restarting skips everything already finished.
    run_grid.main([*common, "--modalities", "rgb", "--arms", "real_synth",
                   "--views", f"measured={measured}", f"estimated={views_estimated}"])
    assert len(out.read_text().splitlines()) == 6

    capsys.readouterr()
    analyze.main([str(out), "--question", "q3"])
    assert "rgb/real_synth/estimated" in capsys.readouterr().out
    analyze.main([str(out), "--question", "q2"])
    assert "No matching runs" in capsys.readouterr().out  # one split cannot be tested


def make_results(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def result_row(modality, arm, tag, split, rank1):
    config = dict(dataset="prid2011", low_res=None, iterations=1500, k=1, modality=modality, arm=arm, synth_tag=tag, split_seed=split)
    return dict(config=config, rank1=rank1, rank5=rank1, mAP=rank1)


def test_question_reports_compare_the_planned_conditions(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    rows = []
    for split in range(8):
        rows.append(result_row("rgb", "real_aug", None, split, 50.0 + split % 2))
        rows.append(result_row("rgb", "real_synth", "estimated", split, 60.0 + split % 2))
    make_results(path, rows)
    analyze.main([str(path), "--question", "q1"])
    output = capsys.readouterr().out
    assert "rgb/real_aug" in output and "rgb/real_synth/estimated" in output and "PASS" in output and "Holm" in output


def test_q1_also_tests_against_the_better_2d_baseline(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    rows = []
    for split in range(8):
        rows.append(result_row("rgb", "real", None, split, 70.0 + split % 2))
        rows.append(result_row("rgb", "real_aug", None, split, 50.0 + split % 2))
        rows.append(result_row("rgb", "real_synth", "estimated", split, 60.0 + split % 2))
    make_results(path, rows)
    analyze.main([str(path), "--question", "q1"])
    same_augmentation, better_baseline = capsys.readouterr().out.split("better 2D baseline")
    better_baseline = better_baseline.split("Q1 control")[0]  # later sections compare the 2D baselines themselves
    assert "PASS" in same_augmentation
    assert "rgb/real " in better_baseline and "rgb/real_aug" not in better_baseline and "fail" in better_baseline


def test_holm_adjustment_known_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_mesh_bridge_round_trip(tmp_path, fake_prid):
    exported, generated, views = tmp_path / "export", tmp_path / "gen", tmp_path / "views"
    mesh_views.main(["export", "--dataset", "prid2011", "--data-root", str(fake_prid), "--out", str(exported),
                     "--splits", "0", "--k-max", "1"])
    mapping = json.loads((exported / "frames.json").read_text())
    assert len(mapping) == len(list((exported / "bounding_box_train").iterdir()))
    assert all("C0" not in name for name in mapping)
    name, relative = next(iter(mapping.items()))
    generated.mkdir()
    image = Image.fromarray(np.full((32, 16, 3), 120, np.uint8))
    for view in (0, 1, 2):
        image.save(generated / f"{name}_{view:02d}.jpg")
        image.save(generated / f"{name}_{view:02d}_visibility.png")
    keep = generated / "train_cvcf.txt"
    keep.write_text(f"bounding_box_train/{name}_00.jpg\nbounding_box_train/{name}_02.jpg\n")
    mesh_views.main(["import", "--generated", str(generated), "--frames", str(exported / "frames.json"), "--out", str(views),
                     "--keep-list", str(keep), "--ms-per-image", "2500"])
    found = D.synthetic_views(fake_prid / relative, fake_prid, views)
    assert [p.name for p in found] == ["view_00.png", "view_02.png"]  # the filtered view and the masks are excluded
    assert json.loads((views / "manifest.json").read_text())["total_ms_per_image"] == 2500


def test_paper_runner_plan(tmp_path, fake_prid, fake_tvrid, capsys):
    base = ["--prid", str(fake_prid), "--tvrid", str(fake_tvrid), "--work", str(tmp_path / "w"), "--results", str(tmp_path / "r"),
            "--splits", "0", "1", "--k", "1", "2", "--dry-run"]
    run_paper.main(["--stage", "q1", *base])
    q1 = capsys.readouterr().out
    # 2 splits x 2 k x (real, real_aug, synth, real_synth, flat views, and two arms with flip and shift only, from both view sources)
    assert "Planned trainings: 36" in q1
    assert "--iterations 0" in q1 and "flat=" in q1 and "real_synth_weak" in q1
    assert "q1_prid2011.split0.jsonl" in q1 and "q1_prid2011.split1.jsonl" in q1
    assert "Skipping ilidsvid" in q1
    run_paper.main(["--stage", "q2", *base])
    q2 = capsys.readouterr().out
    # 2 splits x 2 k x 2 resolutions x (3 single modalities + 4 view conditions + flat views + 3 models for weighted fusion)
    assert "Planned trainings: 88" in q2 and "--low-res" in q2 and "rgbd20" in q2
    run_paper.main(["--stage", "prepare", *base])
    prepare = capsys.readouterr().out
    assert "estimate_depth.py" in prepare and "--geometry flat" in prepare
    assert not (tmp_path / "w").exists()  # a dry run changes nothing


def test_processing_size_uses_patch_multiples():
    assert monocular.processing_size(256, 128) == (518, 252)
    height, width = monocular.processing_size(192, 192)
    assert height % 14 == 0 and width % 14 == 0


def test_erasing_is_batched_and_stays_in_bounds():
    from reid3d import augment
    batch = torch.full((64, 3, 128, 64), 0.5, device=DEVICE)
    erased = augment.random_erase(batch, p=1.0)
    changed = (erased != batch).any(dim=1)
    fraction = changed.float().mean(dim=(1, 2))
    assert changed.flatten(1).any(dim=1).all() and fraction.max().item() <= 0.5
    untouched = augment.random_erase(batch, p=0.0)
    assert torch.equal(untouched, batch)
    # Each erased region is a solid rectangle.
    mask = changed[0]
    rows, cols = mask.any(dim=1).nonzero().flatten(), mask.any(dim=0).nonzero().flatten()
    box = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    assert box.float().mean().item() > 0.99


def test_a_damaged_result_line_is_ignored_and_never_removed(tmp_path):
    out = tmp_path / "results.jsonl"
    original = '{"key": "a"}\n{"key": "b"}\n{"key": "c'
    out.write_text(original, encoding="utf8")
    assert run_grid.finished_keys(out) == {"a", "b"}
    assert out.read_text(encoding="utf8") == original
    run_grid.append_result(out, {"key": "d"})
    assert out.read_text(encoding="utf8") == original + '\n{"key": "d"}\n'
    assert run_grid.finished_keys(out) == {"a", "b", "d"}


def test_a_new_report_never_replaces_an_earlier_one(tmp_path, monkeypatch):
    earlier = tmp_path / "report_q1.txt"
    earlier.write_text("earlier", encoding="utf8")
    monkeypatch.setattr(run_paper.subprocess, "run", lambda command, **kwargs: kwargs["stdout"].write("newer"))
    run_paper.run_step(run_paper.Step(["analyze"], report_to=earlier), dry_run=False)
    assert earlier.read_text(encoding="utf8") == "earlier"
    assert (tmp_path / "report_q1_2.txt").read_text(encoding="utf8") == "newer"


def test_time_limit_stops_before_starting_a_model(tmp_path, fake_prid, monkeypatch, capsys):
    monkeypatch.setattr(run_grid, "run_condition", lambda *a, **k: pytest.fail("no model should start after the limit"))
    run_grid.main(["--dataset", "prid2011", "--data-root", str(fake_prid), "--k", "1", "--splits", "0", "--arms", "real",
                   "--out", str(tmp_path / "r.jsonl"), "--stop-at", "1", "--device", DEVICE])
    assert "Time limit reached" in capsys.readouterr().out
    assert not (tmp_path / "r.jsonl").exists()


def test_parallel_workers_keep_each_splits_steps_in_order(tmp_path, fake_tvrid, monkeypatch):
    import threading
    calls, lock = [], threading.Lock()

    def record(command, **kwargs):
        with lock:
            calls.append(command)

    monkeypatch.setattr(run_paper.subprocess, "run", record)
    run_paper.main(["--stage", "q2", "--tvrid", str(fake_tvrid), "--work", str(tmp_path / "w"), "--results", str(tmp_path / "r"),
                    "--splits", "0", "1", "--k", "1", "--workers", "2", "--stop-at", "99"])
    assert len(calls) == 2 * 2 * 7  # splits x resolutions x grids
    for split in ("0", "1"):
        own = [c for c in calls if c[c.index("--splits") + 1] == split]
        first_grid = own[0][own[0].index("--modalities") + 1:own[0].index("--arms")]
        assert "rgbd" in first_grid  # the fusion grid of a split comes first and is not reordered
        assert all("--stop-at" in c for c in own)


def test_reports_are_written_only_for_questions_with_results(tmp_path, fake_prid, monkeypatch):
    results = tmp_path / "r"
    results.mkdir()
    (results / "q1_prid2011.split0.jsonl").write_text("{}\n", encoding="utf8")
    calls = []
    monkeypatch.setattr(run_paper.subprocess, "run", lambda command, **kwargs: calls.append(command))
    run_paper.main(["--stage", "report", "--prid", str(fake_prid), "--work", str(tmp_path / "w"), "--results", str(results)])
    questions = [c[c.index("--question") + 1] for c in calls]
    assert questions == ["q1", "q3"]  # no Q2 report, because there are no Q2 results


def test_stage_all_reports_the_results_made_during_the_same_run(tmp_path, fake_prid, monkeypatch):
    calls = []

    def record(command, **kwargs):
        calls.append(command)
        if command[1].endswith("run_grid.py"):
            out = run_paper.Path(command[command.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}\n", encoding="utf8")

    monkeypatch.setattr(run_paper.subprocess, "run", record)
    run_paper.main(["--stage", "all", "--prid", str(fake_prid), "--work", str(tmp_path / "w"), "--results", str(tmp_path / "r"),
                    "--splits", "0", "--k", "1"])
    assert [c[c.index("--question") + 1] for c in calls if "--question" in c] == ["q1", "q3"]


def test_flat_views_are_a_plane_with_no_3d_shape(tmp_path, fake_tvrid):
    from reid3d import geometry as G
    estimated, _, _ = prepare_tvrid(fake_tvrid, tmp_path)
    flat = tmp_path / "views_flat"
    render_views.main(["--dataset", "tvrid", "--data-root", str(fake_tvrid), "--splits", "0", "--k-max", "2", "--device", DEVICE,
                       "--geometry", "flat", "--est-root", str(estimated), "--out", str(flat), "--angles", "0:15", "0:-15"])
    manifest = json.loads((flat / "manifest.json").read_text())
    assert manifest["geometry"] == "flat" and manifest["views_written"] == 2 * manifest["frames_total"]
    tracklets = D.load_tvrid(fake_tvrid)
    frame = D.sample_images(tracklets, D.make_split(tracklets, 0).train_pids, 1, 0)[0][0]
    assert len(D.synthetic_views(frame, fake_tvrid, flat)) == 2

    # A flat plane ignores the image content: two different images are warped by exactly the same mapping.
    generator = torch.Generator().manual_seed(0)
    rgb = torch.randint(0, 255, (2, 3, 48, 48), generator=generator, dtype=torch.uint8)
    rgb[1, :, :, :24] = 0
    depth = torch.full((2, 1, 48, 48), 1.5)
    camera = G.Camera.centered(48, 48, 60.0)
    views = G.render_views(rgb, depth, camera, [(20.0, 0.0)])
    assert views.visible.min().item() > 0.6
    assert torch.equal(views.depth[0] > 0, views.depth[1] > 0)


def test_weak_view_arms_use_views_with_flip_and_shift_only(tmp_path, fake_tvrid, monkeypatch):
    _, measured, _ = prepare_tvrid(fake_tvrid, tmp_path)
    tracklets = D.load_tvrid(fake_tvrid)
    split = D.make_split(tracklets, 0)
    settings = dict(dataset="tvrid", data_root=str(fake_tvrid), k=2, synth_root=str(measured), height=64, width=64)
    mixed = T.build_training_set(T.Config(arm="real_synth_weak", **settings), tracklets, split, "rgb")
    views_only = T.build_training_set(T.Config(arm="synth_weak", **settings), tracklets, split, "rgb")
    strong = T.build_training_set(T.Config(arm="real_synth", **settings), tracklets, split, "rgb")
    assert torch.equal(mixed[0], strong[0])  # same images as the strong arm; only the augmentation differs
    assert (~mixed[2]).sum().item() == mixed[3]["train_real"] and views_only[2].all()

    from reid3d import augment
    monkeypatch.setattr(augment, "strong", lambda batch: pytest.fail("a weak arm must not use strong augmentation"))
    images, labels, flags, counts = mixed
    T.train_model(T.Config(arm="real_synth_weak", iterations=1, identities_per_batch=4, pretrained=False, device=DEVICE, **settings),
                  images, labels, flags, counts["num_classes"], torch.device(DEVICE))


def test_weighted_fusion_names_and_weights(tmp_path, fake_tvrid):
    from reid3d.metrics import fuse_similarities
    assert T.FUSIONS["rgbd20"] == ("rgb", "depth") and T.FUSIONS["rgbe30"] == ("rgb", "est_depth")
    assert T.FUSION_WEIGHTS["rgbd"] == 0.5 and T.FUSION_WEIGHTS["rgbe10"] == 0.1
    first, second = torch.randn(5, 7), torch.randn(5, 7)
    assert torch.equal(fuse_similarities([first, second], [1.0, 0.0]).argsort(dim=1), first.argsort(dim=1))
    assert torch.allclose(fuse_similarities([first, second], [0.5, 0.5]), fuse_similarities([first, second]))

    estimated, _, _ = prepare_tvrid(fake_tvrid, tmp_path)
    out = tmp_path / "sweep.jsonl"
    run_grid.main(["--dataset", "tvrid", "--data-root", str(fake_tvrid), "--k", "1", "--splits", "0", "--iterations", "2",
                   "--eval-frames", "2", "--no-pretrained", "--device", DEVICE, "--out", str(out), "--est-root", str(estimated),
                   "--modalities", "rgbd10", "rgbd30", "rgbe20", "--arms", "real_aug"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["config"]["modality"] for r in rows] == ["rgbd10", "rgbd30", "rgbe20"]
    assert rows[0]["rgb_rank1"] == rows[1]["rgb_rank1"] == rows[2]["rgb_rank1"]  # one colour model, reused by all three


def test_untrained_grid_scores_the_network_without_training(tmp_path, fake_prid, monkeypatch):
    calls = []
    monkeypatch.setattr(run_paper.subprocess, "run", lambda command, **kwargs: calls.append([str(part) for part in command]))
    run_paper.main(["--stage", "q1", "--prid", str(fake_prid), "--work", str(tmp_path / "w"), "--results", str(tmp_path / "r"),
                    "--splits", "0", "--k", "1", "2"])
    untrained = [c for c in calls if c[c.index("--iterations") + 1] == "0"]
    assert len(untrained) == 1
    command = untrained[0]
    assert command[command.index("--k") + 1:command.index("--splits")] == ["1"]
    assert command[command.index("--arms") + 1:command.index("--k")] == ["real_aug"]


def test_reports_cover_the_controls_and_skip_untrained_runs(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    rows = []
    for split in range(8):
        wobble = split % 2
        rows += [result_row("rgb", "real", None, split, 60.0 + wobble), result_row("rgb", "real_aug", None, split, 55.0 + wobble),
                 result_row("rgb", "real_synth", "estimated", split, 62.0 + wobble), result_row("rgb", "real_synth", "flat", split, 58.0 + wobble),
                 result_row("rgb", "real_synth_weak", "estimated", split, 66.0 + wobble),
                 result_row("rgbd20", "real_aug", None, split, 57.0 + wobble)]
        floor = result_row("rgb", "real_aug", None, split, 40.0)
        floor["config"]["iterations"] = 0
        rows.append(floor)
        floor = result_row("rgb", "real_synth", "flat", split, 10.0)
        floor["config"]["iterations"] = 0
        rows.append(floor)
    make_results(path, rows)
    analyze.main([str(path), "--question", "q1"])
    q1 = capsys.readouterr().out
    weak = q1.split("both with flip and shift only")[1].split("Q1:")[0]
    assert "rgb/real " in weak and "rgb/real_synth_weak/estimated" in weak and "+6.0" in weak
    control = q1.split("Q1 control")[1]
    assert "rgb/real_synth/flat" in control and "rgb/real_synth/estimated" in control and "+4.0" in control
    assert "(rank1, 1 comparisons)" in control  # the untrained rows are listed in the table but never tested
    analyze.main([str(path), "--question", "q2"])
    q2 = capsys.readouterr().out
    assert "rgbd20/real_aug" in q2.split("colour fused with depth")[1].split("Q2:")[0]


def test_dinov2_backbone_trains_and_resnet_keys_are_unchanged(tmp_path, fake_prid):
    pytest.importorskip("transformers")
    from reid3d.model import ReIDNet
    old = T.Config(dataset="prid2011", data_root="x")
    assert "bb=" not in old.key()  # results saved before the option still count as finished
    assert old.key() + "|bb=dinov2_s" == T.Config(dataset="prid2011", data_root="x", backbone="dinov2_s").key()

    model = ReIDNet(5, pretrained=False, backbone="dinov2_s")
    logits, features = model(torch.randn(4, 3, 256, 128), torch.tensor([0, 1, 2, 3]))
    assert logits.shape == (4, 5) and features.shape == (4, 384)
    assert model.eval()(torch.randn(2, 3, 256, 128)).shape == (2, 384)

    out = tmp_path / "bb.jsonl"
    run_grid.main(["--dataset", "prid2011", "--data-root", str(fake_prid), "--k", "1", "--splits", "0", "--iterations", "1",
                   "--eval-frames", "2", "--no-pretrained", "--device", DEVICE, "--out", str(out), "--arms", "real",
                   "--backbone", "dinov2_s"])
    row = json.loads(out.read_text().splitlines()[0])
    assert row["config"]["backbone"] == "dinov2_s" and row["key"].endswith("|bb=dinov2_s")


def test_extra_stage_plans_the_angle_check_and_the_second_backbone(tmp_path, fake_prid, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run_paper.subprocess, "run", lambda command, **kwargs: calls.append([str(part) for part in command]))
    run_paper.main(["--stage", "extra", "--prid", str(fake_prid), "--work", str(tmp_path / "w"),
                    "--results", str(tmp_path / "r"), "--splits", "0"])
    # per split: angles 2 k x 2 arms x 6 view sets; backbone 4 k x (2 arms + 2 arms x 2 view sets)
    assert "Planned trainings: 48" in capsys.readouterr().out
    renders = [c for c in calls if c[1].endswith("render_views.py")]
    assert len(renders) == 6 and all(c[-1].startswith("--angles=") for c in renders)
    assert any(c[-1] == "--angles=-45:0,45:0" and c[c.index("--out") + 1].endswith("views_flat_a45") for c in renders)
    assert render_views.parse_angles(["-45:0,45:0"]) == [(-45.0, 0.0), (45.0, 0.0)]
    assert any("Dinov2Encoder(True)" in " ".join(c) for c in calls)
    grids = [c for c in calls if c[1].endswith("run_grid.py")]
    angles = [c for c in grids if c[c.index("--out") + 1].startswith(str(tmp_path / "r" / "ab_"))]
    assert angles and angles[0][angles[0].index("--k") + 1:angles[0].index("--splits")] == ["1", "2"]
    assert "--backbone" not in angles[0] and "estimated_a30=" in " ".join(angles[0])
    backbone = [c for c in grids if "--backbone" in c]
    assert len(backbone) == 2 and all(c[c.index("--backbone") + 1] == "dinov2_s" for c in backbone)


def test_angle_report_compares_only_views_made_with_the_same_angles(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    rows = []
    for split in range(8):
        wobble = split % 2
        rows += [result_row("rgb", "real_aug", None, split, 50.0 + wobble), result_row("rgb", "real", None, split, 52.0 + wobble),
                 result_row("rgb", "real_synth", "flat_a15", split, 55.0 + wobble),
                 result_row("rgb", "real_synth", "estimated_a15", split, 57.0 + wobble),
                 result_row("rgb", "real_synth", "flat_a45", split, 54.0 + wobble),
                 result_row("rgb", "real_synth", "estimated_a45", split, 60.0 + wobble)]
    make_results(path, rows)
    analyze.main([str(path), "--question", "angles"])
    output = capsys.readouterr().out
    shape = output.split("same angles")[1]
    assert "rgb/real_synth/flat_a15" in shape and "(rank1, 2 comparisons)" in shape
    assert "flat_a15 -> rgb/real_synth/estimated_a45" not in shape.replace("  ", " ")
    assert "95% CI [+2.0, +2.0]" in shape or "95% CI [" in shape
    against_2d = output.split("same augmentation")[1].split("Angles:")[0]
    assert "(rank1, 4 comparisons)" in against_2d


def test_a_second_backbone_is_reported_separately(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    rows = []
    for split in range(3):
        rows.append(result_row("rgb", "real_aug", None, split, 50.0))
        other = result_row("rgb", "real_aug", None, split, 70.0)
        other["config"]["backbone"] = "dinov2_s"
        rows.append(other)
    make_results(path, rows)
    analyze.main([str(path)])
    output = capsys.readouterr().out
    assert "prid2011 res=full k=1" in output and "prid2011 dinov2_s res=full k=1" in output


def test_example_view_figure_is_written_and_never_replaces_one(tmp_path, fake_prid):
    import make_example_views
    work = tmp_path / "work"
    common = ["--dataset", "prid2011", "--data-root", str(fake_prid), "--splits", "0", "--k-max", "1", "--device", DEVICE]
    estimated = work / "prid2011" / "estimated_depth"
    estimate_depth.main([*common, "--out", str(estimated), "--batch-size", "4"])
    for geometry in ("flat", "estimated"):
        render_views.main([*common, "--geometry", geometry, "--est-root", str(estimated), "--out",
                           str(work / "prid2011" / f"views_{geometry}"), "--angles=-15:0,15:0"])
    arguments = ["--dataset", "prid2011", "--data-root", str(fake_prid), "--work", str(work), "--out", str(tmp_path / "fig"),
                 "--people", "2"]
    make_example_views.main(arguments)
    make_example_views.main(arguments)
    written = sorted(p.name for p in (tmp_path / "fig").iterdir())
    assert written == ["example_views_prid2011.png", "example_views_prid2011_2.png"]
    width, height = Image.open(tmp_path / "fig" / written[0]).size
    assert width > 3 * 128 and height > 4 * 256  # 3 columns, 2 people x 2 sources
