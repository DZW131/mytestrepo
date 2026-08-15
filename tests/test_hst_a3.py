import unittest
from types import SimpleNamespace

import torch

from network.hst.hst_rectifier import HSTConfig, HSTRectifier
from network.hst.latent_interaction import HierarchicalLatentInteraction
from network.resnet38_cls import Net
from tests.test_hst_a1 import make_features
from tests.test_hst_a2 import gradient_is_finite_and_nonzero
from train_sshr import get_model_kwargs


A2_PARAMETER_COUNT = 109_505_621
A3_HLI_PARAMETER_COUNT = 132_136


class HierarchicalLatentInteractionMLPTest(unittest.TestCase):
    def test_mlp_residual_shape_and_cross_token_dependency(self):
        torch.manual_seed(5)
        module = HierarchicalLatentInteraction(latent_dim=8, mode="mlp")
        descriptors = torch.randn(2, 4, 8, requires_grad=True)

        interacted, residual = module(descriptors, return_residual=True)

        self.assertEqual(interacted.shape, descriptors.shape)
        self.assertEqual(residual.shape, descriptors.shape)
        self.assertTrue(torch.equal(interacted, descriptors + residual))
        self.assertTrue(torch.isfinite(interacted).all())
        self.assertFalse(torch.equal(interacted, descriptors))

        gradient = torch.autograd.grad(
            interacted[:, 0].sum(), descriptors, retain_graph=False
        )[0]
        self.assertGreater(gradient[:, 1:].norm().item(), 0.0)

    def test_mlp_is_lightweight_and_identity_has_no_parameters(self):
        identity = HierarchicalLatentInteraction(256, mode="identity")
        mlp = HierarchicalLatentInteraction(256, mode="mlp")

        self.assertEqual(sum(p.numel() for p in identity.parameters()), 0)
        self.assertEqual(
            sum(p.numel() for p in mlp.parameters()),
            A3_HLI_PARAMETER_COUNT,
        )


class HSTA3IntegrationTest(unittest.TestCase):
    def test_variant_defaults_and_controls_are_explicit(self):
        a2 = HSTConfig(variant="a2")
        a3 = HSTConfig(variant="a3")
        a3_identity = HSTConfig(variant="a3", hli_mode="identity")

        self.assertEqual(a2.hli_mode, "identity")
        self.assertEqual(a3.hli_mode, "mlp")
        self.assertTrue(a3.transition_enabled)
        self.assertEqual(a3_identity.hli_mode, "identity")
        with self.assertRaises(ValueError):
            HSTConfig(variant="a2", hli_mode="mlp")
        with self.assertRaises(ValueError):
            HSTConfig(variant="a3", transition_enabled=False)

    def test_training_manifest_resolves_official_a3_defaults(self):
        args = SimpleNamespace(
            rectifier="hst",
            hst_variant="a3",
            hst_latent_dim=256,
            hst_context_kernel=15,
            hst_transition_enabled=None,
            hst_hli_mode=None,
        )
        config = get_model_kwargs(args)["hst_config"]
        self.assertEqual(config["variant"], "a3")
        self.assertTrue(config["transition_enabled"])
        self.assertEqual(config["hli_mode"], "mlp")

    def test_a2_parameter_count_is_unchanged(self):
        model = Net(
            n_class=4,
            rectifier_type="hst",
            hst_config={"variant": "a2"},
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            A2_PARAMETER_COUNT,
        )

    def test_a3_identity_control_is_exact_a2(self):
        torch.manual_seed(17)
        a2 = HSTRectifier(HSTConfig(variant="a2"))
        torch.manual_seed(17)
        a3_identity = HSTRectifier(
            HSTConfig(variant="a3", hli_mode="identity")
        )
        a3_identity.load_state_dict(a2.state_dict(), strict=True)

        features = make_features(batch_size=1)
        with torch.no_grad():
            output_a2 = a2(
                features["stage1"],
                features["stage2"],
                features["stage3"],
                features["deep"],
            )
            output_control = a3_identity(
                features["stage1"],
                features["stage2"],
                features["stage3"],
                features["deep"],
            )

        self.assertTrue(
            torch.equal(
                output_control["hli_residual"],
                torch.zeros_like(output_control["hli_residual"]),
            )
        )
        for key in ("latent_tokens",):
            self.assertTrue(torch.equal(output_control[key], output_a2[key]))
        for collection in (
            "correction_states",
            "semantic_gates",
            "rectified_features",
        ):
            for stage in output_a2[collection]:
                self.assertTrue(
                    torch.equal(
                        output_control[collection][stage],
                        output_a2[collection][stage],
                    )
                )

    def test_a3_mlp_interacts_before_transition_without_breaking_zero_gamma(self):
        torch.manual_seed(23)
        rectifier = HSTRectifier(HSTConfig(variant="a3"))
        features = make_features(batch_size=2)

        output = rectifier(
            features["stage1"],
            features["stage2"],
            features["stage3"],
            features["deep"],
        )

        self.assertFalse(
            torch.equal(output["latent_tokens"], output["raw_latent_tokens"])
        )
        self.assertTrue(torch.isfinite(output["hli_residual"]).all())
        for stage in ("stage1", "stage2", "stage3"):
            self.assertTrue(
                torch.equal(output["rectified_features"][stage], features[stage])
            )

    def test_a3_gradient_connectivity_when_semantic_path_is_open(self):
        torch.manual_seed(29)
        rectifier = HSTRectifier(HSTConfig(variant="a3"))
        with torch.no_grad():
            for gamma in rectifier.gamma_sem.values():
                gamma.fill_(0.1)

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

        hli = rectifier.latent_interaction
        for parameter in (
            hli.normalization.weight,
            hli.token_mixer[0].weight,
            hli.token_mixer[2].weight,
            hli.channel_mixer[0].weight,
            hli.channel_mixer[2].weight,
        ):
            self.assertTrue(gradient_is_finite_and_nonzero(parameter))
        for stage in rectifier.top_down_stages:
            self.assertTrue(
                gradient_is_finite_and_nonzero(
                    rectifier.transitions[stage].transition_mlp[0].weight
                )
            )

    def test_a3_optimizer_groups_cover_every_parameter_once(self):
        model = Net(
            n_class=4,
            rectifier_type="hst",
            hst_config={"variant": "a3"},
        )
        groups = model.get_parameter_groups()
        grouped_ids = [id(parameter) for group in groups for parameter in group]
        trainable_ids = [
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(trainable_ids))

    def test_full_a3_forward_diagnostics_are_finite(self):
        torch.manual_seed(31)
        model = Net(
            n_class=4,
            rectifier_type="hst",
            hst_config={"variant": "a3"},
        )
        model.eval()
        with torch.no_grad():
            outputs, diagnostics = model.forward_with_diagnostics(
                torch.randn(1, 3, 64, 64)
            )

        self.assertEqual(len(outputs), 10)
        self.assertEqual(diagnostics["raw_latent_tokens"].shape, (1, 4, 256))
        self.assertEqual(diagnostics["latent_tokens"].shape, (1, 4, 256))
        self.assertEqual(diagnostics["hli_residual"].shape, (1, 4, 256))
        self.assertGreater(diagnostics["hli_residual"].norm().item(), 0.0)
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())
        for key in (
            "raw_latent_tokens",
            "latent_tokens",
            "hli_residual",
        ):
            self.assertTrue(torch.isfinite(diagnostics[key]).all())


if __name__ == "__main__":
    unittest.main()
