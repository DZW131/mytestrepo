"""Target-conditioned semantic transition used by HST A2."""

from typing import Tuple, Union

import torch
import torch.nn as nn


class StageSemanticTransition(nn.Module):
    """Update a parent correction state using the target hierarchy descriptor.

    The residual scale ``rho`` is initialized to zero. Therefore an enabled A2
    block is exactly equivalent to A1 at initialization while retaining a
    learnable path toward stage-specific correction states.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.latent_dim = latent_dim
        self.transition_mlp = nn.Sequential(
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.rho = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        parent_c: torch.Tensor,
        target_z: torch.Tensor,
        return_delta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._validate_inputs(parent_c, target_z)
        transition_input = torch.cat(
            (
                parent_c,
                target_z,
                parent_c - target_z,
                parent_c * target_z,
            ),
            dim=-1,
        )
        delta = self.transition_mlp(transition_input)
        current_c = parent_c + self.rho * delta
        if return_delta:
            return current_c, delta
        return current_c

    def _validate_inputs(
        self, parent_c: torch.Tensor, target_z: torch.Tensor
    ) -> None:
        if parent_c.ndim != 2 or target_z.ndim != 2:
            raise ValueError(
                "StageSemanticTransition expects [B,d_c] tensors, got "
                f"{tuple(parent_c.shape)} and {tuple(target_z.shape)}"
            )
        if parent_c.shape != target_z.shape:
            raise ValueError(
                "parent_c and target_z must have identical shapes, got "
                f"{tuple(parent_c.shape)} and {tuple(target_z.shape)}"
            )
        if parent_c.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected latent dimension {self.latent_dim}, "
                f"got {parent_c.shape[1]}"
            )
