"""Dataset-free two-step HST-A1 CUDA/CPU training smoke test."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net
from tool.torchutils import PolyOptimizer


LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)


def make_optimizer(model):
    groups = model.get_parameter_groups()
    base_lr = 0.01
    weight_decay = 5e-4
    optimizer_groups = [
        {"params": groups[0], "lr": base_lr, "weight_decay": weight_decay},
        {"params": groups[1], "lr": 2 * base_lr, "weight_decay": 0},
        {"params": groups[2], "lr": 10 * base_lr, "weight_decay": weight_decay},
        {"params": groups[3], "lr": 20 * base_lr, "weight_decay": 0},
    ]
    return PolyOptimizer(
        optimizer_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        max_step=2,
    )


def classification_loss(outputs, labels):
    logits = outputs[:4]
    return sum(
        weight * F.multilabel_soft_margin_loss(logit, labels)
        for weight, logit in zip(LOSS_WEIGHTS, logits)
    )


def gradient_summary(parameter):
    if parameter.grad is None:
        return {"present": False, "finite": False, "norm": None, "nonzero": False}
    gradient = parameter.grad
    norm = gradient.norm().item()
    return {
        "present": True,
        "finite": torch.isfinite(gradient).all().item(),
        "norm": norm,
        "nonzero": norm > 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)

    model = Net(n_class=4, rectifier_type="hst").to(device)
    model.train()
    optimizer = make_optimizer(model)
    images = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    labels = torch.randint(0, 2, (args.batch_size, 4), device=device).float()

    step_records = []
    for step in range(2):
        optimizer.zero_grad()
        outputs = model(images)
        loss = classification_loss(outputs, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()

        rectifier = model.hst_rectifier
        step_records.append(
            {
                "step": step + 1,
                "loss": loss.item(),
                "gamma_sem": {
                    stage: gamma.item()
                    for stage, gamma in rectifier.gamma_sem.items()
                },
                "gamma_sem_grad": {
                    stage: gradient_summary(gamma)
                    for stage, gamma in rectifier.gamma_sem.items()
                },
                "deep_projector_grad": gradient_summary(
                    rectifier.semantic_projectors["deep"].projection.weight
                ),
                "stage_gate_grad": {
                    stage: gradient_summary(gate.weight)
                    for stage, gate in rectifier.semantic_gates.items()
                },
                "all_outputs_finite": all(
                    torch.isfinite(output).all().item() for output in outputs
                ),
            }
        )
        optimizer.step()

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "steps": step_records,
        "final_gamma_sem": {
            stage: gamma.item()
            for stage, gamma in model.hst_rectifier.gamma_sem.items()
        },
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "expected_zero_init_behavior": (
            "step 1 updates gamma; inner semantic branch gradients become connected "
            "on step 2"
        ),
    }

    if not all(record["all_outputs_finite"] for record in step_records):
        raise RuntimeError("non-finite model output detected")
    if not all(
        summary["finite"] and summary["nonzero"]
        for summary in step_records[0]["gamma_sem_grad"].values()
    ):
        raise RuntimeError("semantic residual scales did not receive step-1 gradients")
    if not (
        step_records[1]["deep_projector_grad"]["finite"]
        and step_records[1]["deep_projector_grad"]["nonzero"]
    ):
        raise RuntimeError("deep semantic projector was not connected on step 2")
    if not all(
        summary["finite"] and summary["nonzero"]
        for summary in step_records[1]["stage_gate_grad"].values()
    ):
        raise RuntimeError("stage semantic gates were not connected on step 2")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
