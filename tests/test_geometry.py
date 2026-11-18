"""Check view rendering against hand-computed and brute-force answers."""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reid3d import data as D  # noqa: E402
from reid3d import geometry as G  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def column_coded_image(height, width):
    """An image whose red channel identifies the source column."""
    image = torch.zeros(1, 3, height, width, dtype=torch.uint8, device=DEVICE)
    image[:, 0] = (torch.arange(width, device=DEVICE) * 255 // max(1, width - 1)).byte().expand(height, -1)
    image[:, 1] = 128
    return image


def test_rotation_matrix_is_a_proper_rotation():
    assert torch.allclose(G.rotation_matrix(0, 0), torch.eye(3))
    rotation = G.rotation_matrix(25, -40)
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-6)
    assert torch.det(rotation).item() == pytest.approx(1.0, abs=1e-6)


def test_identity_view_returns_the_input():
    rgb = torch.randint(0, 256, (2, 3, 32, 24), device=DEVICE, dtype=torch.uint8)
    depth = 2.0 + torch.rand(2, 1, 32, 24, device=DEVICE)
    camera = G.Camera.centered(32, 24, 40.0)
    rendered = G.render_views(rgb, depth, camera, [(0.0, 0.0)], crack_tolerance=None, flying_ratio=None)
    assert torch.equal(rendered.rgb[:, 0], rgb)
    assert torch.allclose(rendered.depth[:, 0], depth)
    assert rendered.visible.min().item() == 1.0


def test_pixel_lands_where_the_pinhole_model_says():
    height, width = 48, 64
    rgb = column_coded_image(height, width)
    depth = torch.full((1, 1, height, width), 2.0, device=DEVICE)
    camera = G.Camera.centered(height, width, 60.0)
    yaw = 12.0
    rendered = G.render_views(rgb, depth, camera, [(yaw, 0.0)])

    source_col, source_row = 40, 20
    z = 2.0
    x = (source_col - camera.cx) / camera.fx * z
    y = (source_row - camera.cy) / camera.fy * z
    pivot = G.person_pivot(G.lift_points(depth[:, 0], camera), G.split_near_far(depth[:, 0])).cpu().numpy().ravel()
    rotation = G.rotation_matrix(yaw, 0.0).numpy()
    moved = rotation @ (np.array([x, y, z]) - pivot) + pivot
    target_col = round(camera.fx * moved[0] / moved[2] + camera.cx)
    target_row = round(camera.fy * moved[1] / moved[2] + camera.cy)

    expected = source_col * 255 // (width - 1)
    assert abs(int(rendered.rgb[0, 0, 0, target_row, target_col]) - expected) <= 5


@pytest.mark.parametrize("seed", range(3))
def test_z_buffer_matches_brute_force(seed):
    generator = torch.Generator().manual_seed(seed)
    height, width = 18, 14
    rgb = torch.randint(0, 256, (1, 3, height, width), generator=generator, dtype=torch.uint8)
    depth = 1.0 + 3.0 * torch.rand(1, 1, height, width, generator=generator)
    camera = G.Camera.centered(height, width, 20.0)
    yaw, pitch = 17.0, -9.0
    rendered = G.render_views(rgb, depth, camera, [(yaw, pitch)], crack_tolerance=None, flying_ratio=None, close_iterations=0)

    z = depth[0, 0].numpy()
    pivot = G.person_pivot(G.lift_points(depth[:, 0], camera), G.split_near_far(depth[:, 0])).numpy().ravel()
    rotation = G.rotation_matrix(yaw, pitch).numpy()
    best = {}
    for row in range(height):
        for col in range(width):
            point = np.array([(col - camera.cx) / camera.fx * z[row, col], (row - camera.cy) / camera.fy * z[row, col], z[row, col]])
            moved = rotation @ (point - pivot) + pivot
            u = int(torch.tensor(camera.fx * moved[0] / moved[2] + camera.cx, dtype=torch.float32).round())
            v = int(torch.tensor(camera.fy * moved[1] / moved[2] + camera.cy, dtype=torch.float32).round())
            if 0 <= u < width and 0 <= v < height:
                best[(v, u)] = min(best.get((v, u), math.inf), moved[2])

    output = rendered.depth[0, 0, 0].numpy()
    assert (output > 0).sum() == len(best)
    for (v, u), expected in best.items():
        assert output[v, u] == pytest.approx(expected, rel=1e-3)


def test_nearer_surface_hides_farther_surface():
    height, width = 40, 40
    rgb = torch.zeros(1, 3, height, width, dtype=torch.uint8, device=DEVICE)
    rgb[:, 2] = 255  # far background is blue
    depth = torch.full((1, 1, height, width), 4.0, device=DEVICE)
    rgb[:, :, 10:30, 10:30] = torch.tensor([255, 0, 0], device=DEVICE, dtype=torch.uint8).view(3, 1, 1)  # near red box
    depth[:, :, 10:30, 10:30] = 2.0
    camera = G.Camera.centered(height, width, 50.0)
    rendered = G.render_views(rgb, depth, camera, [(0.0, 0.0), (20.0, 0.0)])
    view = rendered.rgb[0, 1]
    # The red box must remain a solid block, foreshortened by the rotation, with no blue showing through it.
    red = (view[0] > 200) & (view[2] < 50)
    row = red[20]
    assert row.sum().item() >= 17
    columns = row.nonzero().flatten()
    assert columns.max().item() - columns.min().item() + 1 == row.sum().item()


def test_cracks_are_found_but_object_edges_are_not():
    z = torch.tensor([[1.0, 1.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0]]).unsqueeze(0).repeat(1, 3, 1)
    cracks = G.find_cracks(z, 0.1)
    assert cracks[0, 1].tolist() == [False, False, True, False, False, False, False, False]


def test_fill_missing_keeps_known_values_and_can_be_limited():
    values = torch.zeros(1, 1, 9, 9)
    values[0, 0, 4, 4] = 7.0
    known = values > 0
    full, filled = G.fill_missing(values, known)
    assert filled.all() and full[0, 0, 4, 4] == 7.0 and (full == 7.0).all()
    partial, partly = G.fill_missing(values, known, iterations=1)
    assert partly.sum().item() == 9
    assert partial[0, 0, 0, 0] == 0.0


def test_invalid_sensor_pixels_are_not_counted_or_invented():
    rgb = torch.randint(0, 256, (1, 3, 30, 30), device=DEVICE, dtype=torch.uint8)
    depth = torch.full((1, 1, 30, 30), 2.0, device=DEVICE)
    depth[:, :, 0:10, 0:10] = 0.0  # a sensor dropout
    rendered = G.render_views(rgb, depth, G.Camera.centered(30, 30, 40.0), [(0.0, 0.0)])
    assert rendered.visible[0, 0].item() == pytest.approx(1 - 100 / 900, abs=1e-6)
    assert (rendered.depth[0, 0, 0, :7, :7] == 0).all()
    assert (rendered.depth[0, 0, 0, 12:, 12:] > 0).all()


def test_pivot_follows_an_off_centre_person():
    z = torch.full((1, 60, 60), 3.0, device=DEVICE)
    z[:, 40:58, 40:58] = 1.5  # a person near the corner of the crop, 9% of the image
    z[:, 5:15, 5:15] = 2.9  # some other low object, not the nearest
    camera = G.Camera.centered(60, 60, 70.0)
    points = G.lift_points(z, camera)
    near = G.split_near_far(z)
    assert near.sum().item() == 18 * 18  # only the person, not the low object
    pivot = G.person_pivot(points, near)
    expected = points[0, 49, 49].cpu()
    assert torch.allclose(pivot.cpu().ravel(), expected, atol=0.1)


@pytest.mark.parametrize("side", [8, 20, 40])
def test_near_far_split_works_for_any_person_size(side):
    z = torch.full((1, 60, 60), 3.0, device=DEVICE)
    z[:, 10:10 + side, 10:10 + side] = 1.5
    assert G.split_near_far(z).sum().item() == side * side


def test_near_far_split_of_a_flat_scene_keeps_everything():
    z = torch.full((1, 30, 30), 2.0, device=DEVICE)
    assert G.split_near_far(z).all()


def test_flying_pixels_move_with_their_surface():
    height, width = 40, 40
    rgb = torch.zeros(1, 3, height, width, dtype=torch.uint8, device=DEVICE)
    rgb[:, :, 10, 30] = 255  # one white pixel with a bad depth value
    depth = torch.full((1, 1, height, width), 2.0, device=DEVICE)
    depth[:, :, 10, 30] = 0.8
    camera = G.Camera.centered(height, width, 50.0)
    plane = torch.full_like(depth, 2.0)
    expected = G.render_views(rgb, plane, camera, [(20.0, 0.0)]).rgb[0, 0, 0] > 100
    cleaned = G.render_views(rgb, depth, camera, [(20.0, 0.0)]).rgb[0, 0, 0] > 100
    assert cleaned.any() and (cleaned & expected).any()
    unfiltered = G.render_views(rgb, depth, camera, [(20.0, 0.0)], flying_ratio=None).rgb[0, 0, 0] > 100
    assert not (unfiltered & expected).any()  # without the filter the pixel really does fly away


def test_holes_show_source_background_not_streaks():
    height, width = 48, 48
    rgb = column_coded_image(height, width)
    depth = torch.full((1, 1, height, width), 3.0, device=DEVICE)
    depth[:, :, 18:30, 18:30] = 1.0  # the person; everything else is clearly background
    camera = G.Camera.centered(height, width, 60.0)
    rendered = G.render_views(rgb, depth, camera, [(25.0, 0.0)], flying_ratio=None, close_iterations=0, fill_rings=0)
    holes = rendered.depth[0, 0, 0] == 0
    person = depth[0, 0] == 1.0
    assert holes.sum() > 0
    usable = holes & ~person
    assert torch.equal(rendered.rgb[0, 0, 0][usable], rgb[0, 0][usable])


def test_narrow_gaps_are_filled_from_nearby_pixels_not_the_source():
    values = torch.zeros(1, 3, 20, 20)
    values[:, 0] = 200.0
    known = torch.ones(1, 1, 20, 20, dtype=torch.bool)
    known[:, :, 8:12, 8:12] = False  # a gap four pixels wide closes within two rings
    filled, done = G.fill_missing(values, known, iterations=2)
    assert done.all() and torch.allclose(filled[:, 0], torch.full((1, 20, 20), 200.0))


def test_person_is_never_copied_into_a_hole():
    height, width = 48, 48
    rgb = torch.zeros(1, 3, height, width, dtype=torch.uint8, device=DEVICE)
    rgb[:, 2] = 200  # blue background
    rgb[:, 0, 16:40, 16:32] = 255
    rgb[:, 2, 16:40, 16:32] = 0  # red person
    depth = torch.full((1, 1, height, width), 3.0, device=DEVICE)
    depth[:, :, 16:40, 16:32] = 1.0
    camera = G.Camera.centered(height, width, 60.0)
    rendered = G.render_views(rgb, depth, camera, [(25.0, 0.0)], close_iterations=0, fill_rings=0)
    red = rendered.rgb[0, 0, 0] > 128
    # Source pixels are used only for background, so red can only appear where the person really projects.
    assert red.sum().item() <= ((depth[0, 0] == 1.0).sum().item()) * 1.25


def test_depth_from_disparity_range_and_order():
    disparity = torch.linspace(0.0, 1.0, 100).view(1, 1, 10, 10)
    depth = G.depth_from_disparity(disparity, near=1.0, far=2.0, low=0.0, high=1.0)
    assert depth.min().item() == pytest.approx(1.0, abs=1e-4)
    assert depth.max().item() == pytest.approx(2.0, abs=1e-4)
    assert (depth.flatten().diff() < 0).all()  # more disparity means closer


def test_pseudo_depth_is_read_like_sensor_depth(tmp_path):
    disparity = torch.linspace(0.05, 1.0, 64 * 64).view(1, 1, 64, 64)
    stored = G.disparity_to_depth16(disparity).numpy()[0, 0]
    path = tmp_path / "frame_estdepth.png"
    Image.fromarray(stored).save(path)
    encoded = D.read_depth(path, (64, 64))
    assert encoded[0, 63, 63] > encoded[0, 32, 32] > encoded[0, 0, 0]  # closest pixels are brightest


def test_alignment_metrics_recover_a_known_scale_and_shift():
    depth = 1000.0 + 2000.0 * torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(0))
    disparity = 3.0 * (1.0 / depth) + 0.5
    scores = G.alignment_metrics(disparity, depth)
    assert max(scores["absrel"]) < 1e-3
    assert min(scores["pearson"]) > 0.9999


def test_alignment_metrics_detect_unrelated_prediction():
    generator = torch.Generator().manual_seed(1)
    depth = 1000.0 + 2000.0 * torch.rand(1, 1, 64, 64, generator=generator)
    scores = G.alignment_metrics(torch.rand(1, 1, 64, 64, generator=generator), depth)
    assert abs(scores["pearson"][0]) < 0.1
