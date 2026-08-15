"""A1 progressive-only Hierarchical Semantic Transition rectifier."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from .context import apply_contextual_homogenization, build_context_conv
from .semantic_projector import HierarchySemanticProjector


@dataclass(frozen=True)
class HSTConfig:
    """Single source of truth for the currently validated HST milestone."""

    variant: str = "a1"
    latent_dim: int = 256
    context_kernel: int = 15

    def __post_init__(self) -> None:
        if self.variant.lower() != "a1":
            raise ValueError(
                "Only the progressive-only A1 variant is implemented in this milestone; "
                f"got {self.variant!r}."
            )
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.context_kernel <= 0 or self.context_kernel % 2 == 0:
            raise ValueError(
                "context_kernel must be a positive odd integer, "
                f"got {self.context_kernel}"
            )

    @classmethod
    def from_value(cls, value: Optional[Any]) -> "HSTConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError(
            "hst_config must be None, HSTConfig, or a mapping, "
            f"got {type(value).__name__}"
        )


class HSTRectifier(nn.Module):
    """Progress correction states from deep to stage3, stage2, and stage1.

    A1 deliberately copies the parent correction state at each step.  It does
    not use a transition MLP or latent token mixer.  Stage-specific gate heads
    still allow each hierarchy to decode the shared correction state into its
    own channel space.
    """

    stage_channels = {
        "stage1": 256,
        "stage2": 512,
        "stage3": 1024,
        "deep": 4096,
    }
    top_down_stages = ("stage3", "stage2", "stage1")

    def __init__(self, config: Optional[Any] = None):
        super().__init__()
        self.config = HSTConfig.from_value(config)
        latent_dim = self.config.latent_dim

        self.semantic_projectors = nn.ModuleDict(
            {
                name: HierarchySemanticProjector(channels, latent_dim)
                for name, channels in self.stage_channels.items()
            }
        )

        # phi_D is learnable but starts as the identity, so C_D initially has
        # the same semantics and scale as the normalized deep descriptor.
        self.deep_state_initializer = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.eye_(self.deep_state_initializer.weight)

        self.semantic_gates = nn.ModuleDict(
            {
                stage: nn.Linear(latent_dim, self.stage_channels[stage])
                for stage in self.top_down_stages
            }
        )
        self.context_convs = nn.ModuleDict(
            {
                stage: build_context_conv(
                    self.stage_channels[stage], self.config.context_kernel
                )
                for stage in self.top_down_stages
            }
        )
        self.gamma_sem = nn.ParameterDict(
            {stage: nn.Parameter(torch.zeros(1)) for stage in self.top_down_stages}
        )
        self.gamma_ctx = nn.ParameterDict(
            {stage: nn.Parameter(torch.zeros(1)) for stage in self.top_down_stages}
        )

    def residual_scale_parameters(self):
        """Scalars not owned by Conv/Linear/Norm modules for optimizer grouping."""
        yield from self.gamma_sem.values()
        yield from self.gamma_ctx.values()

    def forward(
        self,
        feat_stage1: torch.Tensor,
        feat_stage2: torch.Tensor,
        feat_stage3: torch.Tensor,
        feat_deep: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        base_features = {
            "stage1": feat_stage1,
            "stage2": feat_stage2,
            "stage3": feat_stage3,
            "deep": feat_deep,
        }
        self._validate_features(base_features)

        descriptors = {
            stage: self.semantic_projectors[stage](feature)
            for stage, feature in base_features.items()
        }

        correction_states = {
            "deep": self.deep_state_initializer(descriptors["deep"])
        }
        semantic_gates: Dict[str, torch.Tensor] = {}
        semantic_features: Dict[str, torch.Tensor] = {}
        context_features: Dict[str, torch.Tensor] = {}
        rectified_features: Dict[str, torch.Tensor] = {"deep": feat_deep}

        parent_state = correction_states["deep"]
        for stage in self.top_down_stages:
            # A1 progressive-only: propagate the correction state itself, with
            # no target-conditioned transition and no raw feature cascade.
            current_state = parent_state
            correction_states[stage] = current_state

            gate = torch.sigmoid(self.semantic_gates[stage](current_state))
            gate_4d = gate.unsqueeze(-1).unsqueeze(-1)
            semantic_feature = base_features[stage] * gate_4d
            context_feature = apply_contextual_homogenization(
                self.context_convs[stage], base_features[stage]
            )
            rectified_feature = (
                base_features[stage]
                + self.gamma_sem[stage] * semantic_feature
                + self.gamma_ctx[stage] * context_feature
            )

            semantic_gates[stage] = gate
            semantic_features[stage] = semantic_feature
            context_features[stage] = context_feature
            rectified_features[stage] = rectified_feature
            parent_state = current_state

        return {
            "base_features": base_features,
            "semantic_descriptors": descriptors,
            "correction_states": correction_states,
            "semantic_gates": semantic_gates,
            "semantic_features": semantic_features,
            "context_features": context_features,
            "rectified_features": rectified_features,
        }

    def _validate_features(self, features: Mapping[str, torch.Tensor]) -> None:
        batch_size = None
        for stage, expected_channels in self.stage_channels.items():
            feature = features[stage]
            if feature.ndim != 4:
                raise ValueError(
                    f"{stage} must have shape [B,C,H,W], got {tuple(feature.shape)}"
                )
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"{stage} expected {expected_channels} channels, "
                    f"got {feature.shape[1]}"
                )
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError("all hierarchy features must share the same batch size")
