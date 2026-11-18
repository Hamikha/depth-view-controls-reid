"""Depth estimation from a single RGB image, using Depth Anything V2 (Small)."""
from __future__ import annotations

import torch

from . import augment as aug

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def processing_size(height: int, width: int, long_side: int = 518, patch: int = 14) -> tuple[int, int]:
    """Input size for the model: longest side `long_side`, both sides multiples of the patch size."""
    scale = long_side / max(height, width)
    return (max(patch, round(height * scale / patch) * patch), max(patch, round(width * scale / patch) * patch))


class MonocularDepth:
    """Callable mapping uint8 RGB (N, 3, H, W) to relative disparity (N, 1, H, W)."""

    def __init__(self, device: torch.device | str, model_name: str = DEFAULT_MODEL):
        from transformers import AutoModelForDepthEstimation

        self.device = torch.device(device)
        self.model_name = model_name
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        batch = aug.normalize(aug.to_float(images.to(self.device, non_blocking=True)))
        with torch.autocast(self.device.type):
            disparity = self.model(pixel_values=batch).predicted_depth
        return disparity.float().unsqueeze(1).clamp_min(0)
