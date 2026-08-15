"""Dataset-free ten-step HST-A3 CUDA/CPU optimization-readiness smoke test."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net
from tools.smoke_hst_a2 import (
    all_nonzero_finite,
    classification_loss,
    gradient_summary,
    make_optimizer,
    module_gradient_summary,
    transition_update_summary,
)


TOKEN_ORDER = ("deep", "stage3", "stage2", "stage1")


def hli_residual_summary(diagnostics):
    raw_tokens = diagnostics["raw_latent_tokens"]
    residual = diagnostics["hli_residual"]
    raw_norm = raw_tokens.detach().float().norm().item()
    residual_norm = residual.detach().float().norm().item()
    ratio = residual_norm / raw_norm if raw_norm > 0.0 else None
    per_token = {}
    for index, stage in enumerate(TOKEN_ORDER):
        stage_raw_norm = raw_tokens[:, index].detach().float().norm().item()
        stage_residual_norm = residual[:, index].detach().float().norm().item()
        stage_ratio = (
            stage_residual_norm / stage_raw_norm
            if stage_raw_norm > 0.0
            else None
        )
        per_token[stage] = {
            "raw_norm": stage_raw_norm,
            "residual_norm": stage_residual_norm,
            "ratio": stage_ratio,
            "finite": (
                stage_ratio is not None
                and math.isfinite(stage_ratio)
                and torch.isfinite(raw_tokens[:, index]).all().item()
                and torch.isfinite(residual[:, index]).all().item()
            ),
        }
    return {
        "raw_norm": raw_norm,
        "residual_norm": residual_norm,
        "ratio": ratio,
        "per_token": per_token,
        "finite": (
            ratio is not None
            and math.isfinite(ratio)
            and all(item["finite"] for item in per_token.values())
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
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
        hst_config={"variant": "a3"},
    ).to(device)
    model.train()
    optimizer = make_optimizer(model, max_step=args.steps)
    images = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )
    labels = torch.randint(
        0,
        2,
        (args.batch_size, 4),
        device=device,
    ).float()

    records = []
    for step in range(args.steps):
        optimizer.zero_grad()
        outputs, diagnostics = model.forward_with_diagnostics(images)
        loss = classification_loss(outputs, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()

        rectifier = model.hst_rectifier
        hli = rectifier.latent_interaction
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
        hli_grad = {
            "all": module_gradient_summary(hli),
            "normalization": module_gradient_summary(hli.normalization),
            "token_mixer": module_gradient_summary(hli.token_mixer),
            "channel_mixer": module_gradient_summary(hli.channel_mixer),
        }
        hli_residual = hli_residual_summary(diagnostics)
        relative_transition_update = transition_update_summary(
            rectifier,
            diagnostics,
        )
        finite_checks = {
            "loss": torch.isfinite(loss).item(),
            "outputs": all(
                torch.isfinite(output).all().item() for output in outputs
            ),
            "raw_latent_tokens": torch.isfinite(
                diagnostics["raw_latent_tokens"]
            ).all().item(),
            "latent_tokens": torch.isfinite(
                diagnostics["latent_tokens"]
            ).all().item(),
            "hli_residual": hli_residual["finite"],
            "correction_states": all(
                torch.isfinite(state).all().item()
                for state in diagnostics["correction_states"].values()
            ),
            "transition_deltas": all(
                torch.isfinite(delta).all().item()
                for delta in diagnostics["transition_deltas"].values()
            ),
            "relative_transition_updates": all(
                item["finite"]
                for item in relative_transition_update.values()
            ),
            "gamma_gradients": all(
                item["finite"] for item in gamma_grad.values()
            ),
            "rho_gradients": all(
                item["finite"] for item in rho_grad.values()
            ),
            "hli_gradients": all(item["finite"] for item in hli_grad.values()),
            "transition_mlp_gradients": all(
                item["finite"] for item in transition_mlp_grad.values()
            ),
            "target_projector_gradients": all(
                item["finite"] for item in target_projector_grad.values()
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
            "hli_grad": hli_grad,
            "hli_residual": hli_residual,
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
            if record["hli_grad"]["all"]["finite"]
            and record["hli_grad"]["all"]["nonzero"]
            and all_nonzero_finite(record["target_projector_grad"])
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
        "variant": "a3",
        "hli_mode": "mlp",
        "optimization_readiness": {
            "pass": readiness_pass,
            "path_active_step": path_active_step,
            "criterion": (
                "HLI, transition MLP, and target projector gradients are "
                "finite and nonzero by step 2 or 3"
            ),
        },
        "steps": records,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }

    if not all(record["finite_checks"]["all"] for record in records):
        raise RuntimeError("non-finite A3 value or gradient detected")
    if not all_nonzero_finite(records[0]["gamma_grad"]):
        raise RuntimeError("gamma did not receive expected step-1 gradients")
    if not readiness_pass:
        raise RuntimeError("A3 paths were not active by step 2 or 3")

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
