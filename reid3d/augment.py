"""Batched augmentation that runs on the GPU, so data loading is never the bottleneck."""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_float(batch: torch.Tensor) -> torch.Tensor:
    return batch.float().div(255.0)


def normalize(batch: torch.Tensor) -> torch.Tensor:
    mean, std = IMAGENET_MEAN.to(batch.device), IMAGENET_STD.to(batch.device)
    return (batch - mean) / std


def random_flip(batch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    flip = torch.rand(batch.size(0), device=batch.device) < p
    batch = batch.clone()
    batch[flip] = batch[flip].flip(-1)
    return batch


def random_shift(batch: torch.Tensor, fraction: float = 0.05) -> torch.Tensor:
    """Random translation by up to `fraction` of the image size."""
    n = batch.size(0)
    theta = torch.zeros(n, 2, 3, device=batch.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, :2, 2] = (torch.rand(n, 2, device=batch.device) * 2 - 1) * (2 * fraction)
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    return F.grid_sample(batch, grid, padding_mode="border", align_corners=False)


def color_jitter(batch: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    """Brightness and contrast jitter, per image."""
    n = batch.size(0)
    brightness = 1 + (torch.rand(n, 1, 1, 1, device=batch.device) * 2 - 1) * strength
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=batch.device) * 2 - 1) * strength
    mean = batch.mean(dim=(2, 3), keepdim=True)
    return ((batch - mean) * contrast + mean * brightness).clamp(0, 1)


def random_erase(batch: torch.Tensor, p: float = 0.5, area: tuple[float, float] = (0.02, 0.33)) -> torch.Tensor:
    """Replace one random rectangle per image with a random colour.

    Fully batched: a per-image Python loop would wait on the GPU dozens of times
    per step, which is a large share of the step time on a fast GPU.
    """
    n, _, height, width = batch.shape
    device = batch.device
    target = height * width * (area[0] + (area[1] - area[0]) * torch.rand(n, device=device))
    ratio = 0.3 + 3.0 * torch.rand(n, device=device)
    h = (target * ratio).sqrt().round().clamp(1, height)
    w = (target / ratio).sqrt().round().clamp(1, width)
    top = (torch.rand(n, device=device) * (height - h + 1)).floor().view(-1, 1, 1)
    left = (torch.rand(n, device=device) * (width - w + 1)).floor().view(-1, 1, 1)
    rows = torch.arange(height, device=device).view(1, height, 1)
    cols = torch.arange(width, device=device).view(1, 1, width)
    inside = (rows >= top) & (rows < top + h.view(-1, 1, 1)) & (cols >= left) & (cols < left + w.view(-1, 1, 1))
    selected = (torch.rand(n, device=device) < p).view(-1, 1, 1)
    return torch.where((inside & selected).unsqueeze(1), torch.rand(n, 3, 1, 1, device=device), batch)


def random_perspective(batch: torch.Tensor, p: float = 0.5, distortion: float = 0.15) -> torch.Tensor:
    """A random projective warp, the 2D baseline for rendered views.

    Each image corner moves by up to `distortion` of the half-size in x and y, and
    the image is resampled through the homography that maps the moved corners back.
    With 0.15 the largest possible corner shift equals that of a plane rotated by about
    23 degrees of yaw under the rendering camera (30 degrees would need about 0.21). Each
    corner moves by a uniform random amount up to that limit, so the typical warp is weaker.
    Edges are padded by repeating border pixels, as holes in flat views are filled.
    """
    n = batch.size(0)
    device = batch.device
    corners = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], device=device).expand(n, 4, 2)
    selected = (torch.rand(n, 1, 1, device=device) < p).float()
    moved = corners + (torch.rand(n, 4, 2, device=device) * 2 - 1) * distortion * selected
    # Homography H with H @ (moved, 1) ~ (corner, 1): output pixels look up their source.
    x, y, u, v = moved[..., 0], moved[..., 1], corners[..., 0], corners[..., 1]
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    rows_u = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    rows_v = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    system = torch.cat([rows_u, rows_v], dim=1)
    target = torch.cat([u, v], dim=1)
    h = torch.linalg.solve(system, target.unsqueeze(-1)).squeeze(-1)
    homography = torch.cat([h, torch.ones(n, 1, device=device)], dim=1).view(n, 3, 3)
    height, width = batch.shape[-2:]
    ys = (torch.arange(height, device=device) + 0.5) / height * 2 - 1
    xs = (torch.arange(width, device=device) + 0.5) / width * 2 - 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).view(1, -1, 3)
    mapped = points @ homography.transpose(1, 2)
    grid = (mapped[..., :2] / mapped[..., 2:].clamp_min(1e-6)).view(n, height, width, 2)
    return F.grid_sample(batch, grid, padding_mode="border", align_corners=False)


def weak(batch: torch.Tensor) -> torch.Tensor:
    """Flip and small shift: the minimum baseline used in re-ID."""
    return normalize(random_shift(random_flip(to_float(batch))))


def strong(batch: torch.Tensor) -> torch.Tensor:
    """Weak augmentation plus colour jitter and random erasing."""
    return normalize(random_erase(color_jitter(random_shift(random_flip(to_float(batch))))))


def weak_perspective(batch: torch.Tensor) -> torch.Tensor:
    """Weak augmentation plus a random projective warp (the 2D baseline for Light+V)."""
    return normalize(random_shift(random_perspective(random_flip(to_float(batch)))))


def strong_perspective(batch: torch.Tensor) -> torch.Tensor:
    """Strong augmentation plus a random projective warp (the 2D baseline for Strong+V)."""
    return normalize(random_erase(color_jitter(random_shift(random_perspective(random_flip(to_float(batch)))))))
