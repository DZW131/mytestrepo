import unittest

import torch

from network.hst.hst_rectifier import HSTConfig, HSTRectifier
from network.hst.transition_block import (
    OFFICIAL_A2_RHO_INIT,
    StageSemanticTransition,
)
from network.resnet38_cls import Net
from tests.test_hst_a1 import make_features


A1_PARAMETER_COUNT = 107_537_234


def gradient_is_finite_and_nonzero(parameter):
    return (
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and parameter.grad.norm().item() > 0.0
    )


class StageSemanticTransitionTest(unittest.TestCase):
    def test_zero_rho_is_exact_parent_identity(self):
        torch.manual_seed(1)
        transition = StageSemanticTransition(latent_dim=8)
        parent = torch.randn(3, 8)
        target = torch.randn(3, 8)
        with torch.no_grad():
            transition.rho.zero_()

        current, delta = transition(parent, target, return_delta=True)

        self.assertEqual(delta.shape, parent.shape)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertEqual(transition.rho.item(), 0.0)
        self.assertTrue(torch.equal(current, parent))

    def test_official_a2_rho_initialization_is_fixed(self):
        transition = StageSemanticTransition(latent_dim=8)
        self.assertAlmostEqual(
            transition.rho.item(), OFFICIAL_A2_RHO_INIT, places=7
        )

    def test_residual_update_matches_formula(self):
        torch.manual_seed(2)
        transition = StageSemanticTransition(latent_dim=8)
        parent = torch.randn(2, 8)
        target = torch.randn(2, 8)
        with torch.no_grad():
            transition.rho.fill_(0.25)

        current, delta = transition(parent, target, return_delta=True)
        expected = parent + 0.25 * delta
        self.assertTrue(torch.equal(current, expected))


class HSTA2IntegrationTest(unittest.TestCase):
    def test_variant_configuration_is_explicit(self):
        a1 = HSTRectifier(HSTConfig(variant="a1"))
        a2 = HSTRectifier(HSTConfig(variant="a2"))
        a2_control = HSTRectifier(
            HSTConfig(variant="a2", transition_enabled=False)
        )

        self.assertFalse(hasattr(a1, "transitions"))
        self.assertTrue(hasattr(a2, "transitions"))
        self.assertTrue(a2.config.transition_enabled)
        self.assertFalse(a2_control.config.transition_enabled)
        with self.assertRaises(ValueError):
            HSTConfig(variant="a1", transition_enabled=True)

    def test_a1_parameter_count_is_unchanged(self):
        model = Net(
            n_class=4,
            rectifier_type="hst",
            hst_config={"variant": "a1"},
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            A1_PARAMETER_COUNT,
        )

    def test_zero_rho_a2_is_exact_a1_progression(self):
        torch.manual_seed(42)
        a1 = HSTRectifier(HSTConfig(variant="a1"))
        torch.manual_seed(42)
        a2 = HSTRectifier(HSTConfig(variant="a2"))
        incompatible = a2.load_state_dict(a1.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            all(
                key.startswith("transitions.")
                for key in incompatible.missing_keys
            )
        )
        with torch.no_grad():
            for transition in a2.transitions.values():
                transition.rho.zero_()

        features = make_features(batch_size=2)
        with torch.no_grad():
            output_a1 = a1(
                features["stage1"],
                features["stage2"],
                features["stage3"],
                features["deep"],
            )
            output_a2 = a2(
                features["stage1"],
                features["stage2"],
                features["stage3"],
                features["deep"],
            )

        for stage in ("deep", "stage3", "stage2", "stage1"):
            self.assertTrue(
                torch.equal(
                    output_a2["correction_states"][stage],
                    output_a1["correction_states"][stage],
                )
            )
            self.assertTrue(
                torch.equal(
                    output_a2["rectified_features"][stage],
                    output_a1["rectified_features"][stage],
                )
            )
        for stage in ("stage3", "stage2", "stage1"):
            self.assertTrue(
                torch.equal(
                    output_a2["semantic_gates"][stage],
                    output_a1["semantic_gates"][stage],
                )
            )
            self.assertEqual(a2.transitions[stage].rho.item(), 0.0)

    def test_official_a2_initialization_keeps_gamma_zero_and_rho_nonzero(self):
        rectifier = HSTRectifier(HSTConfig(variant="a2"))
        for stage in rectifier.top_down_stages:
            self.assertEqual(rectifier.gamma_sem[stage].item(), 0.0)
            self.assertAlmostEqual(
                rectifier.transitions[stage].rho.item(),
                OFFICIAL_A2_RHO_INIT,
                places=7,
            )

    def test_disabled_a2_transition_is_strict_a1_control(self):
        torch.manual_seed(9)
        a1 = HSTRectifier(HSTConfig(variant="a1"))
        torch.manual_seed(9)
        a2_control = HSTRectifier(
            HSTConfig(variant="a2", transition_enabled=False)
        )
        a2_control.load_state_dict(a1.state_dict(), strict=False)
        with torch.no_grad():
            for transition in a2_control.transitions.values():
                transition.rho.fill_(1.0)

        features = make_features(batch_size=1)
        with torch.no_grad():
            output_a1 = a1(
                features["stage1"],
                features["stage2"],
                features["stage3"],
                features["deep"],
            )
            output_control = a2_control(
                features["stage1"],
                features["stage2"],
                features["stage3"],
                features["deep"],
            )

        self.assertEqual(output_control["transition_deltas"], {})
        for stage in ("deep", "stage3", "stage2", "stage1"):
            self.assertTrue(
                torch.equal(
                    output_control["correction_states"][stage],
                    output_a1["correction_states"][stage],
                )
            )

    def test_nonzero_rho_creates_stage_specific_correction_states(self):
        torch.manual_seed(13)
        rectifier = HSTRectifier(HSTConfig(variant="a2"))
        with torch.no_grad():
            for transition in rectifier.transitions.values():
                transition.rho.fill_(0.1)

        features = make_features(batch_size=2)
        output = rectifier(
            features["stage1"],
            features["stage2"],
            features["stage3"],
            features["deep"],
        )

        parent_stage = "deep"
        for stage in ("stage3", "stage2", "stage1"):
            parent = output["correction_states"][parent_stage]
            current = output["correction_states"][stage]
            delta = output["transition_deltas"][stage]
            expected = parent + rectifier.transitions[stage].rho * delta
            self.assertTrue(torch.equal(current, expected))
            self.assertFalse(torch.equal(current, parent))
            self.assertTrue(torch.isfinite(current).all())
            parent_stage = stage

    def test_a2_gradient_connectivity_when_residual_paths_are_open(self):
        torch.manual_seed(21)
        rectifier = HSTRectifier(HSTConfig(variant="a2"))
        with torch.no_grad():
            for gamma in rectifier.gamma_sem.values():
                gamma.fill_(0.1)
            for gamma in rectifier.gamma_ctx.values():
                gamma.fill_(0.1)
            for transition in rectifier.transitions.values():
                transition.rho.fill_(0.1)

        features = make_features(batch_size=1)
        output = rectifier(
            features["stage1"],
            features["stage2"],
            features["stage3"],
            features["deep"],
        )
        loss = sum(
            output["rectified_features"][stage].mean()
            for stage in ("stage1", "stage2", "stage3")
        )
        loss.backward()

        self.assertTrue(
            gradient_is_finite_and_nonzero(
                rectifier.semantic_projectors["deep"].projection.weight
            )
        )
        for stage in ("stage3", "stage2", "stage1"):
            self.assertTrue(
                gradient_is_finite_and_nonzero(
                    rectifier.semantic_projectors[stage].projection.weight
                )
            )
            self.assertTrue(
                gradient_is_finite_and_nonzero(
                    rectifier.transitions[stage].transition_mlp[0].weight
                )
            )
            self.assertTrue(
                gradient_is_finite_and_nonzero(rectifier.transitions[stage].rho)
            )
            self.assertTrue(
                gradient_is_finite_and_nonzero(
                    rectifier.semantic_gates[stage].weight
                )
            )

    def test_a2_optimizer_groups_cover_parameters_once(self):
        model = Net(
            n_class=4,
            rectifier_type="hst",
            hst_config={"variant": "a2"},
        )
        groups = model.get_parameter_groups()
        grouped_ids = [id(parameter) for group in groups for parameter in group]
        trainable_ids = [
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(trainable_ids))

    def test_full_a2_forward_diagnostics_are_finite(self):
        torch.manual_seed(34)
        model = Net(
            n_class=4,
            rectifier_type="hst",
            hst_config={"variant": "a2"},
        )
        model.eval()
        with torch.no_grad():
            outputs, diagnostics = model.forward_with_diagnostics(
                torch.randn(1, 3, 64, 64)
            )

        self.assertEqual(len(outputs), 10)
        self.assertEqual(diagnostics["latent_tokens"].shape, (1, 4, 256))
        self.assertEqual(
            set(diagnostics["transition_deltas"]),
            {"stage3", "stage2", "stage1"},
        )
        self.assertEqual(
            set(diagnostics["transition_scales"]),
            {"stage3", "stage2", "stage1"},
        )
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
