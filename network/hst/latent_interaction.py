"""Latent interaction interface for later HST ablations."""

import torch
import torch.nn as nn


class HierarchicalLatentInteraction(nn.Module):
    """Identity-only A1 interface over four hierarchy tokens.

    A1 is explicitly progressive-only and therefore must not mix hierarchy
    tokens.  The class establishes the stable API needed by A3 while rejecting
    accidental activation of an unvalidated mixer.
    """

    def __init__(self, latent_dim: int = 256, mode: str = "identity"):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if mode != "identity":
            raise ValueError(
                "The A1 milestone supports only mode='identity'; "
                "MLP/attention interaction belongs to the later A3 milestone."
            )
        self.latent_dim = latent_dim
        self.mode = mode

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 3:
            raise ValueError(
                "HierarchicalLatentInteraction expects [B,4,d_c], "
                f"got shape {tuple(descriptors.shape)}"
            )
        if descriptors.shape[1] != 4 or descriptors.shape[2] != self.latent_dim:
            raise ValueError(
                f"expected [B,4,{self.latent_dim}], got {tuple(descriptors.shape)}"
            )
        return descriptors
