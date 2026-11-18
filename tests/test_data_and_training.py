"""Data-protocol checks and an end-to-end run on small synthetic images.

The fake dataset gives each identity a distinct colour and stripe pattern, so a
working pipeline must score far above chance and a broken one will not.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reid3d import data as D  # noqa: E402
from reid3d.train import Config, run_condition  # noqa: E402


def make_fake_prid(root: pathlib.Path, persons: int = 24, frames: int = 30) -> pathlib.Path:
    rng = np.random.default_rng(0)
    for pid in range(1, persons + 1):
        colour = rng.integers(40, 215, 3)
        stripe = rng.integers(40, 215, 3)
        period = int(rng.integers(6, 22))
        for cam, name in enumerate(("cam_a", "cam_b")):
            folder = root / "multi_shot" / name / f"person_{pid:04d}"
            folder.mkdir(parents=True)
            for frame in range(frames):
                image = np.tile(colour, (128, 64, 1)).astype(np.int16)
                image[(np.arange(128) % period) < period // 2] = stripe
                image += rng.integers(-18, 18, image.shape) + 10 * cam
                Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(folder / f"{frame:04d}.png")
    return root


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    return make_fake_prid(tmp_path_factory.mktemp("prid"))


def test_short_tracklets_are_dropped(tmp_path):
    root = make_fake_prid(tmp_path / "data", persons=3)
    for frame in list((root / "multi_shot" / "cam_b" / "person_0002").iterdir())[4:]:
        frame.unlink()
    assert sorted({t.pid for t in D.load_prid2011(root)}) == [1, 3]


def test_split_is_disjoint_and_reproducible(fake_root):
    tracklets = D.load_prid2011(fake_root)
    first = D.make_split(tracklets, 3)
    assert not set(first.train_pids) & set(first.test_pids)
    assert set(first.train_pids) | set(first.test_pids) == {t.pid for t in tracklets}
    assert first == D.make_split(tracklets, 3)
    assert first != D.make_split(tracklets, 4)


def test_few_image_subsets_are_nested_and_cross_camera(fake_root):
    tracklets = D.load_prid2011(fake_root)
    pids = D.make_split(tracklets, 0).train_pids
    one = D.sample_images(tracklets, pids, 1, 0)
    two = D.sample_images(tracklets, pids, 2, 0)
    five = D.sample_images(tracklets, pids, 5, 0)
    assert len(one) == len(pids) and len(two) == 2 * len(pids)
    assert {p for p, _, _ in one} <= {p for p, _, _ in two} <= {p for p, _, _ in five}
    for pid in pids:
        assert {c for _, q, c in two if q == pid} == {0, 1}  # k=2 uses both cameras


def test_sampling_never_touches_test_identities(fake_root):
    tracklets = D.load_prid2011(fake_root)
    split = D.make_split(tracklets, 1)
    assert {pid for _, pid, _ in D.sample_images(tracklets, split.train_pids, None, 1)} == set(split.train_pids)


@pytest.mark.parametrize("arm", ["real", "real_aug"])
def test_end_to_end_learns_above_chance(fake_root, arm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(dataset="prid2011", data_root=str(fake_root), arm=arm, k=2, iterations=60, identities_per_batch=8,
                 pretrained=False, eval_frames=4, device=device, height=128, width=64)
    record = run_condition(cfg)
    assert record["num_gallery"] == 12 and record["num_queries"] == 12
    assert record["rank1"] > 3 * 100 / 12  # at least three times random guessing


def test_synthetic_arm_requires_renders(fake_root):
    cfg = Config(dataset="prid2011", data_root=str(fake_root), arm="synth", iterations=2, pretrained=False, device="cpu")
    with pytest.raises(ValueError, match="synth_root"):
        run_condition(cfg)


def make_fake_tvrid(root: pathlib.Path, persons: int = 12) -> pathlib.Path:
    rng = np.random.default_rng(1)
    for pid in range(persons):
        colour = rng.integers(40, 215, 3)
        height_mm = int(rng.integers(900, 2400))
        for camera in ("flat", "upward"):
            for passage in ("passage1", "passage2"):
                folder = root / "train" / f"{pid:04d}" / camera / passage
                folder.mkdir(parents=True)
                for frame in range(6):
                    rgb = np.tile(colour, (64, 64, 1)) + rng.integers(-15, 15, (64, 64, 3))
                    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(folder / f"t{frame}_RGB.png")
                    depth = np.full((64, 64), height_mm, np.uint16) + rng.integers(0, 30, (64, 64)).astype(np.uint16)
                    depth[(np.arange(64) % (4 + pid)) < 2] += 400  # identity cue that survives normalisation
                    depth[:6, :6] = 0  # invalid pixels, as in a real sensor
                    Image.fromarray(depth).save(folder / f"t{frame}_depth.png")
    return root


def test_tvrid_pairs_rgb_with_depth(tmp_path):
    tracklets = D.load_tvrid(make_fake_tvrid(tmp_path / "tv", persons=2))
    assert len(tracklets) == 2 * 2 * 2
    assert all(a.name.replace("_RGB", "_depth") == b.name for t in tracklets for a, b in zip(t.paths, t.aux_paths))
    assert D.for_modality(tracklets, "depth")[0].paths[0].name.endswith("_depth.png")


def test_depth_encoding_keeps_invalid_pixels_black(tmp_path):
    root = make_fake_tvrid(tmp_path / "tv", persons=1)
    frame = next((root / "train").rglob("*_depth.png"))
    encoded = D.read_depth(frame, (64, 64))
    assert encoded.shape == (3, 64, 64) and encoded.dtype == np.uint8
    assert (encoded[:, :6, :6] == 0).all() and encoded.max() > 0


def test_rgbd_fusion_end_to_end(tmp_path):
    root = make_fake_tvrid(tmp_path / "tv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(dataset="tvrid", data_root=str(root), arm="real_aug", modality="rgbd", k=2, iterations=40,
                 identities_per_batch=6, pretrained=False, eval_frames=2, device=device, height=64, width=64)
    record = run_condition(cfg)
    assert {"rgb_rank1", "depth_rank1", "rgb_mAP", "depth_mAP"} <= set(record)
    assert set(record["scenarios"]) == {"within_camera"}
    assert record["num_queries"] == 2 * 2 * 6  # 2 cameras, both directions, 6 test persons
    assert record["rank1"] > 100 / 6
