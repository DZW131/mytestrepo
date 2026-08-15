"""Compare HFRM and an HST variant's parameters, FLOPs, and latency."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net


class OperationCounter:
    """Forward-hook counter for Conv2d/Linear multiply-add operations."""

    def __init__(self):
        self.macs = 0
        self.handles = []

    def register(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _conv_hook(self, module, inputs, output):
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        self.macs += output.numel() * kernel_ops

    def _linear_hook(self, module, inputs, output):
        self.macs += output.numel() * module.in_features


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile(
    rectifier,
    device,
    batch_size,
    image_size,
    warmup,
    iterations,
    hst_variant="a1",
):
    torch.manual_seed(42)
    model_kwargs = {"rectifier_type": rectifier}
    if rectifier == "hst":
        model_kwargs["hst_config"] = {"variant": hst_variant}
    model = Net(n_class=4, **model_kwargs).to(device)
    model.eval()
    sample = torch.randn(batch_size, 3, image_size, image_size, device=device)

    counter = OperationCounter()
    counter.register(model)
    with torch.no_grad():
        model(sample)
    counter.close()

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    peak_cuda_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )

    return {
        "rectifier": rectifier,
        "hst_variant": hst_variant if rectifier == "hst" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "rectifier_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "hfrm_" in name or "hst_rectifier" in name
        ),
        "conv_linear_macs_per_image": counter.macs / batch_size,
        "conv_linear_flops_per_image": 2.0 * counter.macs / batch_size,
        "latency_ms_batch_median": statistics.median(timings_ms),
        "latency_ms_batch_mean": statistics.mean(timings_ms),
        "peak_cuda_memory_bytes": peak_cuda_memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iterations", default=30, type=int)
    parser.add_argument(
        "--hst_variant", default="a1", choices=["a1", "a2", "a3"]
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = [
        profile(
            rectifier,
            device,
            args.batch_size,
            args.image_size,
            args.warmup,
            args.iterations,
            args.hst_variant,
        )
        for rectifier in ("hfrm", "hst")
    ]
    baseline, hst = results
    comparison = {
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "hst_variant": args.hst_variant,
        "flop_scope": "Conv2d and Linear only; one multiply-add counts as 2 FLOPs",
        "results": results,
        "hst_vs_hfrm_percent": {
            "parameters": 100.0 * (hst["parameters"] / baseline["parameters"] - 1.0),
            "conv_linear_flops": 100.0
            * (
                hst["conv_linear_flops_per_image"]
                / baseline["conv_linear_flops_per_image"]
                - 1.0
            ),
            "median_latency": 100.0
            * (
                hst["latency_ms_batch_median"]
                / baseline["latency_ms_batch_median"]
                - 1.0
            ),
        },
    }
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
