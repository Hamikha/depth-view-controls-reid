"""Re-ID network: ResNet-50 baseline with the standard re-ID modifications.

Last-stride 1, generalised-mean pooling and a batch-norm neck follow the widely
used "bag of tricks" baseline. A single class covers RGB and depth inputs. The
ArcFace head is kept so the original paper's loss can be compared directly.

A second backbone, DINOv2 ViT-S/14 (self-supervised, similar size to ResNet-50),
checks that the findings do not depend on one architecture.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class GeM(nn.Module):
    """Generalised-mean pooling."""

    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(1e-6).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)


class ArcFaceHead(nn.Module):
    """Additive angular margin classifier (Deng et al., CVPR 2019)."""

    def __init__(self, dim: int, classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.cos_m, self.sin_m = scale, math.cos(margin), math.sin(margin)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.normalize(features, dim=1) @ F.normalize(self.weight, dim=1).T
        if labels is None:
            return self.scale * cosine
        target = cosine.gather(1, labels.unsqueeze(1))
        sine = (1 - target.pow(2)).clamp_min(1e-7).sqrt()
        with_margin = target * self.cos_m - sine * self.sin_m
        return self.scale * cosine.scatter(1, labels.unsqueeze(1), with_margin)


DINOV2_SMALL = "facebook/dinov2-small"
BACKBONES = ("resnet50", "dinov2_s")


class ResNetEncoder(nn.Module):
    """ResNet-50 with last stride 1 and GeM pooling; 2048-d output."""

    dim = 2048

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # Last stride 1 keeps twice the spatial detail in the final feature map.
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layers = nn.Sequential(backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4)
        self.pool = GeM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.layers(self.stem(x)))


class Dinov2Encoder(nn.Module):
    """DINOv2 ViT-S/14; the class token after the final layer norm, 384-d output.

    Inputs of any size work: the position embeddings are interpolated, and the
    patch embedding ignores the last few rows or columns that do not fill a patch.
    """

    dim = 384

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model  # only needed for this backbone
        if pretrained:
            self.model = Dinov2Model.from_pretrained(DINOV2_SMALL)
        else:  # same architecture with random weights, for tests
            self.model = Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                                                  intermediate_size=1536, patch_size=14, image_size=518))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).pooler_output


class ReIDNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, id_loss: str = "ce", backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet50":
            self.encoder: nn.Module = ResNetEncoder(pretrained)
        elif backbone == "dinov2_s":
            self.encoder = Dinov2Encoder(pretrained)
        else:
            raise ValueError(f"unknown backbone '{backbone}', expected one of {BACKBONES}")
        dim = self.encoder.dim
        self.neck = nn.BatchNorm1d(dim)
        self.neck.bias.requires_grad_(False)
        if id_loss == "arcface":
            self.head: nn.Module = ArcFaceHead(dim, num_classes)
        elif id_loss == "ce":
            self.head = nn.Linear(dim, num_classes, bias=False)
            nn.init.normal_(self.head.weight, std=0.001)
        else:
            raise ValueError(f"unknown id_loss: {id_loss}")
        self.id_loss = id_loss

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        global_feature = self.encoder(x)
        neck_feature = self.neck(global_feature)
        if not self.training:
            return neck_feature
        if self.id_loss == "arcface":
            return self.head(neck_feature, labels), global_feature
        return self.head(neck_feature), global_feature

    def parameter_groups(self, lr: float) -> list[dict]:
        """Pretrained layers train ten times slower than the newly added ones."""
        fresh = list(self.neck.parameters()) + list(self.head.parameters())
        fresh_ids = {id(p) for p in fresh}
        pretrained = [p for p in self.parameters() if id(p) not in fresh_ids]
        return [{"params": pretrained, "lr": lr * 0.1}, {"params": fresh, "lr": lr}]


def batch_hard_triplet(features: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Batch-hard triplet loss on L2-normalised features."""
    features = F.normalize(features, dim=1)
    distances = torch.cdist(features, features).clamp_min(1e-12)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positives = same & ~eye
    has_positive = positives.any(dim=1)
    has_negative = (~same).any(dim=1)
    usable = has_positive & has_negative
    if not usable.any():
        return features.sum() * 0.0
    hardest_positive = (distances * positives).max(dim=1).values
    hardest_negative = (distances + same.float() * 1e6).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin)[usable].mean()
