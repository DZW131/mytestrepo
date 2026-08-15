"""Lightweight interaction over the four HST hierarchy tokens."""

from typing import Tuple, Union

import torch
import torch.nn as nn


class HierarchicalLatentInteraction(nn.Module):
    """Preserve identity or mix semantic information across hierarchy tokens.

    ``mode="identity"`` is the exact A1/A2 control. ``mode="mlp"`` implements
    the A3 residual form ``Z + Mixer(LN(Z))``. The mixer first operates across
    the four-token hierarchy axis, then applies the specified lightweight
    channel MLP independently at each retained stage position.
    """

    num_hierarchy_tokens = 4

    def __init__(self, latent_dim: int = 256, mode: str = "identity"):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        normalized_mode = mode.lower()
        if normalized_mode not in {"identity", "mlp"}:
            raise ValueError(
                "HierarchicalLatentInteraction mode must be 'identity' or "
                f"'mlp', got {mode!r}"
            )
        self.latent_dim = latent_dim
        self.mode = normalized_mode

        if self.mode == "mlp":
            self.normalization = nn.LayerNorm(latent_dim)
            self.token_mixer = nn.Sequential(
                nn.Linear(
                    self.num_hierarchy_tokens,
                    self.num_hierarchy_tokens,
                ),
                nn.GELU(),
                nn.Linear(
                    self.num_hierarchy_tokens,
                    self.num_hierarchy_tokens,
                ),
            )
            self.channel_mixer = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )

    def forward(
        self,
        descriptors: torch.Tensor,
        return_residual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if descriptors.ndim != 3:
            raise ValueError(
                "HierarchicalLatentInteraction expects [B,4,d_c], "
                f"got shape {tuple(descriptors.shape)}"
            )
        if (
            descriptors.shape[1] != self.num_hierarchy_tokens
            or descriptors.shape[2] != self.latent_dim
        ):
            raise ValueError(
                f"expected [B,4,{self.latent_dim}], got {tuple(descriptors.shape)}"
            )

        if self.mode == "identity":
            if return_residual:
                return descriptors, torch.zeros_like(descriptors)
            return descriptors

        normalized = self.normalization(descriptors)
        token_mixed = self.token_mixer(normalized.transpose(1, 2)).transpose(1, 2)
        residual = self.channel_mixer(token_mixed)
        interacted = descriptors + residual
        if return_residual:
            return interacted, residual
        return interacted
