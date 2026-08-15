"""A1/A2/A3 Hierarchical Semantic Transition rectifier."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from .context import apply_contextual_homogenization, build_context_conv
from .latent_interaction import HierarchicalLatentInteraction
from .semantic_projector import HierarchySemanticProjector
from .transition_block import StageSemanticTransition


@dataclass(frozen=True)
class HSTConfig:
    """Single source of truth for the currently validated HST milestones."""

    variant: str = "a1"
    latent_dim: int = 256
    context_kernel: int = 15
    transition_enabled: Optional[bool] = None
    hli_mode: Optional[str] = None

    def __post_init__(self) -> None:
        normalized_variant = self.variant.lower()
        if normalized_variant not in {"a1", "a2", "a3"}:
            raise ValueError(
                "HST variant must be 'a1', 'a2', or 'a3'; "
                f"got {self.variant!r}."
            )
        object.__setattr__(self, "variant", normalized_variant)

        transition_enabled = self.transition_enabled
        if transition_enabled is None:
            transition_enabled = normalized_variant in {"a2", "a3"}
        if normalized_variant == "a1" and transition_enabled:
            raise ValueError("A1 cannot enable stage-specific transitions")
        if normalized_variant == "a3" and not transition_enabled:
            raise ValueError("A3 requires stage-specific transitions")
        object.__setattr__(self, "transition_enabled", transition_enabled)

        hli_mode = self.hli_mode
        if hli_mode is None:
            hli_mode = "mlp" if normalized_variant == "a3" else "identity"
        hli_mode = hli_mode.lower()
        if hli_mode not in {"identity", "mlp"}:
            raise ValueError(
                "hli_mode must be 'identity' or 'mlp', "
                f"got {self.hli_mode!r}"
            )
        if normalized_variant in {"a1", "a2"} and hli_mode != "identity":
            raise ValueError(
                "A1/A2 require hli_mode='identity'; use A3 for MLP interaction"
            )
        object.__setattr__(self, "hli_mode", hli_mode)
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

    A1 copies the parent correction state at each step. A2 adds target-
    conditioned residual transitions while retaining identity latent
    interaction. A3 keeps the A2 transitions and first mixes the four hierarchy
    descriptors in latent space. Stage-specific gate heads decode each
    correction state into its own channel space.
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

        if self.config.variant in {"a2", "a3"}:
            self.latent_interaction = HierarchicalLatentInteraction(
                latent_dim=latent_dim,
                mode=self.config.hli_mode,
            )
            self.transitions = nn.ModuleDict(
                {
                    stage: StageSemanticTransition(latent_dim)
                    for stage in self.top_down_stages
                }
            )

    def residual_scale_parameters(self):
        """Scalars not owned by Conv/Linear/Norm modules for optimizer grouping."""
        yield from self.gamma_sem.values()
        yield from self.gamma_ctx.values()
        if hasattr(self, "transitions"):
            for transition in self.transitions.values():
                yield transition.rho

    def forward(
        self,
        feat_stage1: torch.Tensor,
        feat_stage2: torch.Tensor,
        feat_stage3: torch.Tensor,
        feat_deep: torch.Tensor,
    ) -> Dict[str, Any]:
        base_features = {
            "stage1": feat_stage1,
            "stage2": feat_stage2,
            "stage3": feat_stage3,
            "deep": feat_deep,
        }
        self._validate_features(base_features)

        raw_descriptors = {
            stage: self.semantic_projectors[stage](feature)
            for stage, feature in base_features.items()
        }

        raw_latent_tokens = torch.stack(
            [
                raw_descriptors["deep"],
                raw_descriptors["stage3"],
                raw_descriptors["stage2"],
                raw_descriptors["stage1"],
            ],
            dim=1,
        )
        if self.config.variant in {"a2", "a3"}:
            latent_tokens, hli_residual = self.latent_interaction(
                raw_latent_tokens,
                return_residual=True,
            )
            descriptors = {
                "deep": latent_tokens[:, 0],
                "stage3": latent_tokens[:, 1],
                "stage2": latent_tokens[:, 2],
                "stage1": latent_tokens[:, 3],
            }
        else:
            # Preserve the A1 computational graph: target descriptors remain
            # diagnostics-only until A2 transitions consume them.
            latent_tokens = raw_latent_tokens
            hli_residual = None
            descriptors = raw_descriptors

        correction_states = {
            "deep": self.deep_state_initializer(descriptors["deep"])
        }
        semantic_gates: Dict[str, torch.Tensor] = {}
        semantic_features: Dict[str, torch.Tensor] = {}
        context_features: Dict[str, torch.Tensor] = {}
        rectified_features: Dict[str, torch.Tensor] = {"deep": feat_deep}
        transition_deltas: Dict[str, torch.Tensor] = {}

        parent_state = correction_states["deep"]
        for stage in self.top_down_stages:
            if self.config.transition_enabled:
                current_state, transition_delta = self.transitions[stage](
                    parent_state,
                    descriptors[stage],
                    return_delta=True,
                )
                transition_deltas[stage] = transition_delta
            else:
                # A1/control path: propagate the correction state itself, with
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
            "raw_semantic_descriptors": raw_descriptors,
            "raw_latent_tokens": raw_latent_tokens,
            "semantic_descriptors": descriptors,
            "latent_tokens": latent_tokens,
            "hli_residual": hli_residual,
            "correction_states": correction_states,
            "transition_deltas": transition_deltas,
            "transition_scales": (
                {
                    stage: transition.rho
                    for stage, transition in self.transitions.items()
                }
                if hasattr(self, "transitions")
                else {}
            ),
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
