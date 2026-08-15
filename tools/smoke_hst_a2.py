"""Dataset-free ten-step HST-A2 CUDA/CPU optimization-readiness smoke test."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.hst.transition_block import OFFICIAL_A2_RHO_INIT
from network.resnet38_cls import Net
from tool.torchutils import PolyOptimizer


LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)


def make_optimizer(model, max_step):
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
        max_step=max_step,
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


def module_gradient_summary(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    present = bool(gradients) and all(
        gradient is not None for gradient in gradients
    )
    available = [gradient for gradient in gradients if gradient is not None]
    finite = present and all(
        torch.isfinite(gradient).all().item() for gradient in available
    )
    norm = (
        math.sqrt(
            sum(
                gradient.detach().float().pow(2).sum().item()
                for gradient in available
            )
        )
        if available
        else None
    )
    return {
        "present": present,
        "finite": finite,
        "norm": norm,
        "nonzero": norm is not None and norm > 0.0,
    }


def transition_update_summary(rectifier, diagnostics):
    parent_stages = {
        "stage3": "deep",
        "stage2": "stage3",
        "stage1": "stage2",
    }
    summaries = {}
    for stage in rectifier.top_down_stages:
        parent = diagnostics["correction_states"][parent_stages[stage]]
        delta = diagnostics["transition_deltas"][stage]
        update = rectifier.transitions[stage].rho * delta
        parent_norm = parent.detach().float().norm().item()
        update_norm = update.detach().float().norm().item()
        ratio = update_norm / parent_norm if parent_norm > 0.0 else None
        summaries[stage] = {
            "update_norm": update_norm,
            "parent_norm": parent_norm,
            "ratio": ratio,
            "finite": (
                torch.isfinite(parent).all().item()
                and torch.isfinite(delta).all().item()
                and torch.isfinite(update).all().item()
                and ratio is not None
                and math.isfinite(ratio)
            ),
        }
    return summaries


def all_nonzero_finite(summaries):
    return all(item["finite"] and item["nonzero"] for item in summaries.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--steps", default=10, type=int)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()
    if args.steps < 3:
        raise ValueError("--steps must be at least 3 for the readiness check")

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
    optimizer = make_optimizer(model, max_step=args.steps)
    images = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    labels = torch.randint(0, 2, (args.batch_size, 4), device=device).float()

    records = []
    for step in range(args.steps):
        optimizer.zero_grad()
        outputs, diagnostics = model.forward_with_diagnostics(images)
        loss = classification_loss(outputs, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.item()}")
        loss.backward()

        rectifier = model.hst_rectifier
        gamma_grad = {
            stage: gradient_summary(gamma)
            for stage, gamma in rectifier.gamma_sem.items()
        }
        rho_grad = {
            stage: gradient_summary(transition.rho)
            for stage, transition in rectifier.transitions.items()
        }
        target_projector_grad = {
            stage: module_gradient_summary(rectifier.semantic_projectors[stage])
            for stage in rectifier.top_down_stages
        }
        transition_mlp_grad = {
            stage: module_gradient_summary(
                rectifier.transitions[stage].transition_mlp
            )
            for stage in rectifier.top_down_stages
        }
        relative_transition_update = transition_update_summary(
            rectifier, diagnostics
        )
        all_outputs_finite = all(
            torch.isfinite(output).all().item() for output in outputs
        )
        finite_checks = {
            "loss": torch.isfinite(loss).item(),
            "outputs": all_outputs_finite,
            "correction_states": all(
                torch.isfinite(state).all().item()
                for state in diagnostics["correction_states"].values()
            ),
            "transition_deltas": all(
                torch.isfinite(delta).all().item()
                for delta in diagnostics["transition_deltas"].values()
            ),
            "relative_transition_updates": all(
                summary["finite"]
                for summary in relative_transition_update.values()
            ),
            "gamma_gradients": all(
                summary["finite"] for summary in gamma_grad.values()
            ),
            "rho_gradients": all(
                summary["finite"] for summary in rho_grad.values()
            ),
            "transition_mlp_gradients": all(
                summary["finite"] for summary in transition_mlp_grad.values()
            ),
            "target_projector_gradients": all(
                summary["finite"] for summary in target_projector_grad.values()
            ),
        }
        finite_checks["all"] = all(finite_checks.values())
        record = {
            "step": step + 1,
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "gamma_sem": {
                stage: gamma.item()
                for stage, gamma in rectifier.gamma_sem.items()
            },
            "gamma_grad": gamma_grad,
            "rho": {
                stage: transition.rho.item()
                for stage, transition in rectifier.transitions.items()
            },
            "rho_grad": rho_grad,
            "transition_mlp_grad": transition_mlp_grad,
            "target_projector_grad": target_projector_grad,
            "relative_transition_update": relative_transition_update,
            "finite_checks": finite_checks,
        }
        optimizer.step()
        record["gamma_sem_after_update"] = {
            stage: gamma.item()
            for stage, gamma in rectifier.gamma_sem.items()
        }
        record["rho_after_update"] = {
            stage: transition.rho.item()
            for stage, transition in rectifier.transitions.items()
        }
        records.append(record)

    path_active_step = next(
        (
            record["step"]
            for record in records[1:3]
            if all_nonzero_finite(record["target_projector_grad"])
            and all_nonzero_finite(record["transition_mlp_grad"])
        ),
        None,
    )
    readiness_pass = path_active_step is not None

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "requested_steps": args.steps,
        "official_rho_init": OFFICIAL_A2_RHO_INIT,
        "activation_sequence": [
            "step 1: gamma receives gradients while rho is already 0.01",
            "step 2: transition MLPs and target projectors should be active after gamma opens",
        ],
        "optimization_readiness": {
            "pass": readiness_pass,
            "path_active_step": path_active_step,
            "criterion": (
                "all stage transition MLP and target projector gradients are "
                "finite and nonzero by step 2 or 3"
            ),
        },
        "steps": records,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }

    if not all(record["finite_checks"]["all"] for record in records):
        raise RuntimeError("non-finite value or gradient detected")
    if not all_nonzero_finite(records[0]["gamma_grad"]):
        raise RuntimeError("gamma did not receive the expected step-1 gradients")
    if not all_nonzero_finite(records[1]["rho_grad"]):
        raise RuntimeError("rho did not receive the expected step-2 gradients")
    if not readiness_pass:
        raise RuntimeError(
            "transition/projector paths were not active by step 2 or 3"
        )

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
