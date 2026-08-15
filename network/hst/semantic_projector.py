"""Hierarchy-specific projection into a shared latent semantic space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchySemanticProjector(nn.Module):
    """GAP -> Linear(C_i, d_c) -> LayerNorm semantic descriptor."""

    def __init__(self, in_channels: int, latent_dim: int = 256):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.projection = nn.Linear(in_channels, latent_dim)
        self.normalization = nn.LayerNorm(latent_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(
                "HierarchySemanticProjector expects [B,C,H,W], "
                f"got shape {tuple(feature.shape)}"
            )
        if feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {feature.shape[1]}"
            )

        descriptor = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        return self.normalization(self.projection(descriptor))
