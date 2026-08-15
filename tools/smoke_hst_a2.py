"""Dataset-free three-step HST-A2 CUDA/CPU training smoke test."""

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
        max_step=3,
    )


def classification_loss(outputs, labels):
    return sum(
        weight * F.multilabel_soft_margin_loss(logit, labels)
        for weight, logit in zip(LOSS_WEIGHTS, outputs[:4])
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


def all_nonzero_finite(summaries):
    return all(item["finite"] and item["nonzero"] for item in summaries.values())


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

    model = Net(
        n_class=4,
        rectifier_type="hst",
        hst_config={"variant": "a2"},
    ).to(device)
    model.train()
    optimizer = make_optimizer(model)
    images = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    labels = torch.randint(0, 2, (args.batch_size, 4), device=device).float()

    records = []
    for step in range(3):
        optimizer.zero_grad()
        outputs = model(images)
        loss = classification_loss(outputs, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()

        rectifier = model.hst_rectifier
        records.append(
            {
                "step": step + 1,
                "loss": loss.item(),
                "all_outputs_finite": all(
                    torch.isfinite(output).all().item() for output in outputs
                ),
                "gamma_sem": {
                    stage: gamma.item()
                    for stage, gamma in rectifier.gamma_sem.items()
                },
                "rho": {
                    stage: transition.rho.item()
                    for stage, transition in rectifier.transitions.items()
                },
                "gamma_grad": {
                    stage: gradient_summary(gamma)
                    for stage, gamma in rectifier.gamma_sem.items()
                },
                "rho_grad": {
                    stage: gradient_summary(transition.rho)
                    for stage, transition in rectifier.transitions.items()
                },
                "gate_grad": {
                    stage: gradient_summary(rectifier.semantic_gates[stage].weight)
                    for stage in rectifier.top_down_stages
                },
                "target_projector_grad": {
                    stage: gradient_summary(
                        rectifier.semantic_projectors[stage].projection.weight
                    )
                    for stage in rectifier.top_down_stages
                },
                "transition_mlp_grad": {
                    stage: gradient_summary(
                        rectifier.transitions[stage].transition_mlp[0].weight
                    )
                    for stage in rectifier.top_down_stages
                },
            }
        )
        optimizer.step()

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "activation_sequence": [
            "step 1: gamma receives gradients",
            "step 2: gates and rho receive gradients after gamma opens",
            "step 3: target projectors and transition MLP receive gradients after rho opens",
        ],
        "steps": records,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }

    if not all(record["all_outputs_finite"] for record in records):
        raise RuntimeError("non-finite model output detected")
    if not all_nonzero_finite(records[0]["gamma_grad"]):
        raise RuntimeError("gamma did not receive the expected step-1 gradients")
    if not all_nonzero_finite(records[1]["rho_grad"]):
        raise RuntimeError("rho did not receive the expected step-2 gradients")
    if not all_nonzero_finite(records[1]["gate_grad"]):
        raise RuntimeError("semantic gates did not open on step 2")
    if not all_nonzero_finite(records[2]["target_projector_grad"]):
        raise RuntimeError("target projectors did not receive step-3 gradients")
    if not all_nonzero_finite(records[2]["transition_mlp_grad"]):
        raise RuntimeError("transition MLP did not receive step-3 gradients")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
