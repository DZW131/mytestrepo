"""Shared Contextual Homogenization primitives.

The factory intentionally mirrors the public SSHR implementation exactly:
depthwise convolution, odd kernel, no bias, and a uniform averaging-kernel
initialization.  Keeping this in one place lets HFRM and HST call the same CH
implementation without changing the baseline state-dict layout.
"""

import torch
import torch.nn as nn


def build_context_conv(in_channels: int, context_kernel: int = 15) -> nn.Conv2d:
    """Build the original SSHR Contextual Homogenization convolution."""
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if context_kernel <= 0 or context_kernel % 2 == 0:
        raise ValueError(
            f"context_kernel must be a positive odd integer, got {context_kernel}"
        )

    context_conv = nn.Conv2d(
        in_channels,
        in_channels,
        kernel_size=context_kernel,
        padding=context_kernel // 2,
        groups=in_channels,
        bias=False,
    )
    nn.init.constant_(context_conv.weight, 1.0 / (context_kernel**2))
    return context_conv


def apply_contextual_homogenization(
    context_conv: nn.Conv2d, feature: torch.Tensor
) -> torch.Tensor:
    """Apply CH through the supplied stage-specific depthwise convolution."""
    return context_conv(feature)
