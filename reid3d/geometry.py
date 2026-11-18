"""New camera views from one image plus its geometry (depth-image-based rendering).

Each pixel is lifted to 3D using its depth, the points are rotated about the
person, and they are projected back with a z-buffer. Unlike a full mesh, this
uses only surfaces the camera really saw. Regions it never saw are marked as not
visible and are excluded from the depth views. In colour views they are filled
with plausible background so that the images stay natural.

The same code serves measured depth (a real sensor) and depth estimated from RGB.
Everything is batched tensor code, so it runs on the GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Camera:
    """Pinhole camera, in pixels of the image being rendered."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def centered(cls, height: int, width: int, focal: float) -> "Camera":
        return cls(focal, focal, (width - 1) / 2, (height - 1) / 2)


@dataclass
class RenderedViews:
    rgb: torch.Tensor  # (N, V, 3, H, W), uint8, holes filled from nearby pixels
    depth: torch.Tensor  # (N, V, 1, H, W), same units as the input, 0 where nothing was observed
    visible: torch.Tensor  # (N, V), fraction of each view that comes from valid source pixels


def rotation_matrix(yaw_deg: float, pitch_deg: float, device: torch.device | str | None = None) -> torch.Tensor:
    """Rotation about the image's vertical axis (yaw), then its horizontal axis (pitch)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    about_vertical = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device=device)
    about_horizontal = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], device=device)
    return about_vertical @ about_horizontal


def fill_missing(values: torch.Tensor, known: torch.Tensor, iterations: int | None = None):
    """Grow known pixels into missing ones, one ring at a time.

    `values` is (N, C, H, W) and `known` is a boolean (N, 1, H, W). Returns the
    filled values and the mask of pixels that now have a value. Known pixels are
    never changed.
    """
    values = values * known
    filled = known.clone()
    limit = max(values.shape[-2:]) if iterations is None else iterations
    for _ in range(limit):
        if filled.all():
            break
        weight = filled.float()
        total = F.avg_pool2d(values, 3, stride=1, padding=1) * 9
        count = F.avg_pool2d(weight, 3, stride=1, padding=1) * 9
        grow = ~filled & (count > 0.5)
        if not grow.any():
            break
        values = torch.where(grow, total / count.clamp_min(1.0), values)
        filled = filled | grow
    return values, filled


def find_cracks(z: torch.Tensor, tolerance: torch.Tensor | float) -> torch.Tensor:
    """Pixels sitting behind both neighbours on one axis, in a (N, H, W) z-buffer.

    Forward warping leaves one-pixel gaps in stretched surfaces, and far geometry
    can leak through them. Such a pixel is farther than the pixels on both of its
    sides. Ordinary object boundaries have a near pixel on one side only, so they
    are left alone. Empty pixels hold infinity and count as cracks when enclosed.
    """
    left, right = z.roll(1, dims=2), z.roll(-1, dims=2)
    up, down = z.roll(1, dims=1), z.roll(-1, dims=1)
    horizontal = (left < z - tolerance) & (right < z - tolerance)
    vertical = (up < z - tolerance) & (down < z - tolerance)
    horizontal[:, :, 0] = horizontal[:, :, -1] = False
    vertical[:, 0, :] = vertical[:, -1, :] = False
    return horizontal | vertical


def median_filter(values: torch.Tensor, size: int = 5) -> torch.Tensor:
    """Median of each pixel's neighbourhood, for a (N, 1, H, W) tensor."""
    radius = size // 2
    padded = F.pad(values, (radius, radius, radius, radius), mode="replicate")
    windows = padded.unfold(2, size, 1).unfold(3, size, 1).flatten(-2)
    return windows.median(dim=-1).values


def lift_points(z: torch.Tensor, camera: Camera) -> torch.Tensor:
    """3D point (N, H, W, 3) of every pixel of a (N, H, W) depth map."""
    height, width = z.shape[-2:]
    rows, cols = torch.meshgrid(torch.arange(height, device=z.device), torch.arange(width, device=z.device), indexing="ij")
    return torch.stack([(cols - camera.cx) / camera.fx * z, (rows - camera.cy) / camera.fy * z, z], dim=-1)


def split_near_far(z: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Boolean (N, H, W) mask of the near part of each depth map, by Otsu's threshold.

    In a re-ID crop the near part is the person, whatever fraction of the crop
    they fill. The depth range is taken from the 1st to 99th percentile so that a
    few extreme values cannot compress the histogram.
    """
    flat = z.flatten(1)
    bounds = torch.quantile(flat, torch.tensor([0.01, 0.99], device=z.device, dtype=flat.dtype), dim=1)
    low, span = bounds[0].unsqueeze(1), (bounds[1] - bounds[0]).clamp_min(1e-6).unsqueeze(1)
    index = ((flat - low) / span * (bins - 1)).round().long().clamp(0, bins - 1)
    counts = torch.zeros(flat.size(0), bins, device=z.device, dtype=flat.dtype).scatter_add_(1, index, torch.ones_like(flat))
    centres = low + span * torch.linspace(0, 1, bins, device=z.device, dtype=flat.dtype).unsqueeze(0)
    near_weight = counts.cumsum(dim=1)
    far_weight = near_weight[:, -1:] - near_weight
    near_sum = (counts * centres).cumsum(dim=1)
    far_sum = near_sum[:, -1:] - near_sum
    separation = (near_sum / near_weight.clamp_min(1.0) - far_sum / far_weight.clamp_min(1.0)).pow(2) * near_weight * far_weight
    threshold = centres.gather(1, separation.argmax(dim=1, keepdim=True)).view(-1, 1, 1)
    return z <= threshold


def person_pivot(points: torch.Tensor, near: torch.Tensor) -> torch.Tensor:
    """Rotation centre (N, 1, 1, 3): the median point of the near part of each image.

    Rotating about the person keeps them in view, including in entry and exit
    frames where they are not at the centre of the crop.
    """
    return torch.stack([points[i][near[i]].median(dim=0).values for i in range(points.size(0))]).view(-1, 1, 1, 3)


def prepare_depth(depth: torch.Tensor, flying_ratio: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill invalid pixels and replace flying pixels; returns the complete depth and the trusted mask."""
    valid = depth > 0
    complete, known = fill_missing(depth, valid)
    complete = torch.where(known, complete, torch.ones_like(complete))  # an image with no valid depth at all
    if flying_ratio is not None:
        smooth = median_filter(complete)
        flying = (complete - smooth).abs() > flying_ratio * smooth
        complete = torch.where(flying, smooth, complete)
        valid = valid & ~flying
    return complete, valid


@torch.no_grad()
def render_views(rgb: torch.Tensor, depth: torch.Tensor, camera: Camera, angles: list[tuple[float, float]],
                 crack_tolerance: float | None = 0.02, flying_ratio: float | None = 0.05,
                 close_iterations: int = 2, fill_rings: int = 6,
                 layout: torch.Tensor | None = None) -> RenderedViews:
    """Render (yaw, pitch) views of a batch of images, rotating about the person.

    `layout` is an optional (N, 1, H, W) depth map that decides where the person
    is: the rotation centre and which source pixels count as background when
    holes are filled. By default it is `depth` itself. Controls pass the
    image's own estimated depth here, so that a flat plane or a borrowed depth
    map differs from the estimated geometry only in the depth used for warping.

    `rgb` is uint8 (N, 3, H, W) and `depth` is float (N, 1, H, W) with 0 for
    invalid pixels. `flying_ratio` removes the stray depth values depth sensors
    produce along object edges, and `close_iterations` closes the small gaps that
    forward warping leaves in stretched surfaces.

    In colour views, gaps up to `2 * fill_rings` pixels wide are filled from nearby
    pixels, which avoids a speckled mix of warped and copied pixels in stretched
    regions. Wider holes take the source image where it showed background. Person
    pixels are never copied into holes, because that would leave a ghost.
    """
    batch, _, height, width = rgb.shape
    device = rgb.device
    pixels = batch * height * width
    colours = rgb.float().permute(0, 2, 3, 1)
    complete, valid = prepare_depth(depth, flying_ratio)
    z = complete[:, 0]
    points = lift_points(z, camera)
    # The person mask comes from the layout depth, prepared exactly as the warping depth would be.
    near = split_near_far(z if layout is None else prepare_depth(layout, flying_ratio)[0][:, 0])
    pivot = person_pivot(points, near)
    pivot_depth = pivot[:, 0, 0, 2]
    background = ~near.unsqueeze(1)
    image_index = torch.arange(batch, device=device).view(-1, 1, 1).expand_as(z)

    rgb_views, depth_views, visible = [], [], []
    for yaw, pitch in angles:
        rotation = rotation_matrix(yaw, pitch, device)
        moved = (points - pivot) @ rotation.T + pivot
        new_z = moved[..., 2].clamp_min(1e-6)
        target_col = (camera.fx * moved[..., 0] / new_z + camera.cx).round().long()
        target_row = (camera.fy * moved[..., 1] / new_z + camera.cy).round().long()
        inside = (moved[..., 2] > 1e-6) & (target_col >= 0) & (target_col < width) & (target_row >= 0) & (target_row < height)

        target = (image_index * height * width + target_row * width + target_col)[inside]
        depth_inside = new_z[inside]
        nearest = torch.full((pixels,), float("inf"), device=device, dtype=depth_inside.dtype)
        nearest.scatter_reduce_(0, target, depth_inside, reduce="amin", include_self=True)
        winner = depth_inside <= nearest[target]

        canvas = torch.zeros(pixels, 3, device=device)
        canvas[target[winner]] = colours[inside][winner]
        trusted = torch.zeros(pixels, dtype=torch.bool, device=device)
        trusted[target[winner]] = valid[:, 0][inside][winner]
        z_buffer = nearest.view(batch, height, width)
        covered = torch.isfinite(z_buffer)
        trusted = trusted.view(batch, height, width)
        if crack_tolerance is not None:
            cracks = find_cracks(z_buffer, crack_tolerance * pivot_depth.view(-1, 1, 1))
            covered, trusted = covered & ~cracks, trusted & ~cracks

        canvas = canvas.view(batch, height, width, 3).permute(0, 3, 1, 2)
        closed_rgb, closed = fill_missing(canvas, covered.unsqueeze(1), iterations=fill_rings)
        use_source = ~closed & background
        closed_rgb = torch.where(use_source, rgb.float(), closed_rgb)
        filled_rgb, _ = fill_missing(closed_rgb, closed | use_source)
        rgb_views.append(filled_rgb.round().clamp(0, 255).byte())

        # Depth stays honest: regions never observed remain 0, as on a real sensor.
        depth_view = torch.where(trusted, z_buffer, torch.zeros_like(z_buffer)).unsqueeze(1)
        closed_depth, _ = fill_missing(depth_view, trusted.unsqueeze(1), iterations=close_iterations)
        depth_views.append(closed_depth)
        visible.append(trusted.float().mean(dim=(1, 2)))

    return RenderedViews(torch.stack(rgb_views, dim=1), torch.stack(depth_views, dim=1), torch.stack(visible, dim=1))


def depth_from_disparity(disparity: torch.Tensor, near: float = 1.0, far: float = 2.0,
                         low: float = 0.02, high: float = 0.98) -> torch.Tensor:
    """Turn relative disparity into a plausible depth map for rendering.

    Monocular models give disparity only up to an unknown scale and shift, so
    each image is mapped so its 2nd to 98th percentile span depths `far` to `near`.
    """
    flat = disparity.flatten(1)
    bounds = torch.quantile(flat, torch.tensor([low, high], device=disparity.device, dtype=flat.dtype), dim=1)
    lowest, highest = bounds[0].view(-1, 1, 1, 1), bounds[1].view(-1, 1, 1, 1)
    scaled = ((disparity - lowest) / (highest - lowest).clamp_min(1e-6)).clamp(0, 1)
    return 1.0 / (scaled * (1.0 / near - 1.0 / far) + 1.0 / far)


def disparity_to_depth16(disparity: torch.Tensor) -> torch.Tensor:
    """Store relative disparity as 16-bit pseudo-depth, readable exactly like sensor depth."""
    top = torch.quantile(disparity.flatten(1), 0.98, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (1000.0 * top / disparity.clamp_min(1e-6)).round().clamp(1, 65535).to(torch.uint16)


def alignment_metrics(disparity: torch.Tensor, measured_depth: torch.Tensor) -> dict[str, list[float]]:
    """How well estimated disparity matches a sensor, per image.

    Disparity is aligned to measured inverse depth with the least-squares scale
    and shift, the usual way to score relative depth. Returns the Pearson
    correlation and the absolute relative depth error on valid pixels.
    """
    mask = (measured_depth > 0).float()
    count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    target = (1.0 / measured_depth.clamp_min(1e-6)) * mask
    prediction = disparity * mask
    mean_x = prediction.sum(dim=(1, 2, 3)) / count
    mean_y = target.sum(dim=(1, 2, 3)) / count
    view = (-1, 1, 1, 1)
    dx = (disparity - mean_x.view(view)) * mask
    dy = (target - mean_y.view(view)) * mask
    var_x = dx.pow(2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
    var_y = dy.pow(2).sum(dim=(1, 2, 3)).clamp_min(1e-12)
    covariance = (dx * dy).sum(dim=(1, 2, 3))
    scale = covariance / var_x
    shift = mean_y - scale * mean_x
    aligned = scale.view(view) * disparity + shift.view(view)
    floor = (1.0 / (measured_depth.flatten(1).max(dim=1).values.clamp_min(1.0) * 1.5)).view(view)
    predicted_depth = 1.0 / torch.maximum(aligned, floor)
    error = ((predicted_depth - measured_depth).abs() / measured_depth.clamp_min(1e-6)) * mask
    return {"absrel": (error.sum(dim=(1, 2, 3)) / count).tolist(),
            "pearson": (covariance / (var_x * var_y).sqrt()).tolist()}
