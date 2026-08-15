import hashlib
import unittest

import torch
import torch.nn.functional as F

from network.hst.hst_rectifier import HSTConfig, HSTRectifier
from network.hst.latent_interaction import HierarchicalLatentInteraction
from network.resnet38_cls import Net


BASELINE_PARAMETER_COUNT = 112_709_714
BASELINE_STATE_KEY_SHA256 = (
    "23038075b660d3f97ada855a9c138cda6f82711902214c2aa70e6a394e45b796"
)


def legacy_hfrm_forward(model, x):
    """Verbatim computational path from upstream resnet38_cls.py@7346cc5."""
    x = model.conv1a(x)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)

    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    feat_56 = x

    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x)
    x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    feat_28_1 = F.relu(model.bn45(x))

    x, _ = model.b5(x, get_x_bn_relu=True)
    x = model.b5_1(x); x = model.b5_2(x)
    feat_28_2 = F.relu(model.bn52(x))

    x, _ = model.b6(x, get_x_bn_relu=True); x = model.b7(x)
    feat_deep = F.relu(model.bn7(x))

    feat_56_rectified = model.hfrm_56(feat_56, feat_deep)
    feat_28_1_rectified = model.hfrm_28_1(feat_28_1, feat_deep)
    feat_28_2_rectified = model.hfrm_28_2(feat_28_2, feat_deep)

    cam_56 = model.ic_56(feat_56_rectified)
    cam_28_1 = model.ic1(feat_28_1_rectified)
    cam_28_2 = model.ic2(feat_28_2_rectified)
    feat_deep_drop = model.dropout7(feat_deep)
    cam_deep = model.fc8(feat_deep_drop)

    out_56 = F.avg_pool2d(
        cam_56, kernel_size=(cam_56.size(2), cam_56.size(3)), padding=0
    ).view(x.size(0), -1)
    out_28_1 = F.avg_pool2d(
        cam_28_1, kernel_size=(cam_28_1.size(2), cam_28_1.size(3)), padding=0
    ).view(x.size(0), -1)
    out_28_2 = F.avg_pool2d(
        cam_28_2, kernel_size=(cam_28_2.size(2), cam_28_2.size(3)), padding=0
    ).view(x.size(0), -1)
    out_deep = F.avg_pool2d(
        cam_deep, kernel_size=(cam_deep.size(2), cam_deep.size(3)), padding=0
    ).view(x.size(0), -1)
    y_deep = torch.sigmoid(out_deep)

    return (
        out_56,
        out_28_1,
        out_28_2,
        out_deep,
        y_deep,
        cam_56,
        cam_28_1,
        cam_28_2,
        cam_deep,
        feat_56_rectified,
    )


def make_features(batch_size=2):
    generator = torch.Generator().manual_seed(42)
    return {
        "stage1": torch.randn(batch_size, 256, 8, 8, generator=generator),
        "stage2": torch.randn(batch_size, 512, 4, 4, generator=generator),
        "stage3": torch.randn(batch_size, 1024, 4, 4, generator=generator),
        "deep": torch.randn(batch_size, 4096, 4, 4, generator=generator),
    }


class BaselineCompatibilityTest(unittest.TestCase):
    def test_default_hfrm_state_layout_and_outputs_match_upstream(self):
        torch.manual_seed(20260815)
        model = Net(n_class=4)
        model.eval()

        self.assertEqual(model.rectifier_type, "hfrm")
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            BASELINE_PARAMETER_COUNT,
        )
        state_keys = "\n".join(model.state_dict().keys()).encode()
        self.assertEqual(
            hashlib.sha256(state_keys).hexdigest(), BASELINE_STATE_KEY_SHA256
        )

        x = torch.linspace(-1, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
        with torch.no_grad():
            current = model(x)
            reference = legacy_hfrm_forward(model, x)

        self.assertEqual(len(current), len(reference))
        for current_tensor, reference_tensor in zip(current, reference):
            self.assertTrue(torch.equal(current_tensor, reference_tensor))


class HSTA1ComponentTest(unittest.TestCase):
    def test_identity_hli_is_exact(self):
        descriptors = torch.randn(3, 4, 256)
        module = HierarchicalLatentInteraction(256, mode="identity")
        output = module(descriptors)
        self.assertIs(output, descriptors)
        self.assertTrue(torch.equal(output, descriptors))

    def test_zero_gamma_a1_shapes_states_and_finiteness(self):
        rectifier = HSTRectifier(HSTConfig())
        features = make_features()
        output = rectifier(
            features["stage1"],
            features["stage2"],
            features["stage3"],
            features["deep"],
        )

        for stage, feature in features.items():
            self.assertEqual(output["base_features"][stage].shape, feature.shape)
            self.assertEqual(output["rectified_features"][stage].shape, feature.shape)
            self.assertTrue(
                torch.equal(output["rectified_features"][stage], feature)
            )
            self.assertTrue(torch.isfinite(output["rectified_features"][stage]).all())

        for stage in ("deep", "stage3", "stage2", "stage1"):
            self.assertEqual(output["correction_states"][stage].shape, (2, 256))
        self.assertIs(
            output["correction_states"]["stage3"],
            output["correction_states"]["deep"],
        )
        self.assertIs(
            output["correction_states"]["stage2"],
            output["correction_states"]["stage3"],
        )
        self.assertIs(
            output["correction_states"]["stage1"],
            output["correction_states"]["stage2"],
        )
        self.assertEqual(output["semantic_gates"]["stage1"].shape, (2, 256))
        self.assertEqual(output["semantic_gates"]["stage2"].shape, (2, 512))
        self.assertEqual(output["semantic_gates"]["stage3"].shape, (2, 1024))

    def test_ch_matches_original_hfrm_definition(self):
        torch.manual_seed(7)
        hfrm_model = Net(n_class=4, rectifier_type="hfrm")
        torch.manual_seed(7)
        hst_model = Net(n_class=4, rectifier_type="hst")

        pairs = (
            (hfrm_model.hfrm_56.context_conv, hst_model.hst_rectifier.context_convs["stage1"]),
            (hfrm_model.hfrm_28_1.context_conv, hst_model.hst_rectifier.context_convs["stage2"]),
            (hfrm_model.hfrm_28_2.context_conv, hst_model.hst_rectifier.context_convs["stage3"]),
        )
        features = make_features(batch_size=1)
        for stage, (hfrm_ch, hst_ch) in zip(
            ("stage1", "stage2", "stage3"), pairs
        ):
            self.assertTrue(torch.equal(hfrm_ch.weight, hst_ch.weight))
            self.assertTrue(
                torch.equal(hfrm_ch(features[stage]), hst_ch(features[stage]))
            )

    def test_gradient_connectivity_after_residual_scales_open(self):
        rectifier = HSTRectifier(HSTConfig())
        with torch.no_grad():
            for gamma in rectifier.gamma_sem.values():
                gamma.fill_(0.1)
            for gamma in rectifier.gamma_ctx.values():
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

        connected_parameters = (
            rectifier.semantic_projectors["deep"].projection.weight,
            rectifier.deep_state_initializer.weight,
            rectifier.semantic_gates["stage1"].weight,
            rectifier.semantic_gates["stage2"].weight,
            rectifier.semantic_gates["stage3"].weight,
            rectifier.context_convs["stage1"].weight,
            rectifier.context_convs["stage2"].weight,
            rectifier.context_convs["stage3"].weight,
        )
        for parameter in connected_parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

        # Target projectors are intentionally dormant in A1; A2 will consume
        # z3/z2/z1 in stage-specific transitions and make these trainable paths.
        for stage in ("stage1", "stage2", "stage3"):
            self.assertIsNone(
                rectifier.semantic_projectors[stage].projection.weight.grad
            )

    def test_optimizer_groups_cover_each_trainable_parameter_once(self):
        model = Net(n_class=4, rectifier_type="hst")
        groups = model.get_parameter_groups()
        grouped_ids = [id(parameter) for group in groups for parameter in group]
        trainable_ids = [
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(trainable_ids))

    def test_full_hst_forward_and_diagnostics(self):
        torch.manual_seed(11)
        model = Net(n_class=4, rectifier_type="hst")
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            outputs, diagnostics = model.forward_with_diagnostics(x)

        self.assertEqual(len(outputs), 10)
        self.assertEqual(outputs[0].shape, (1, 4))
        self.assertEqual(outputs[5].shape, (1, 4, 16, 16))
        self.assertEqual(outputs[6].shape, (1, 4, 8, 8))
        self.assertEqual(outputs[7].shape, (1, 4, 8, 8))
        self.assertEqual(outputs[8].shape, (1, 4, 8, 8))
        self.assertEqual(
            set(diagnostics["cam_logits"]),
            {"stage1", "stage2", "stage3", "deep"},
        )
        for tensor in outputs:
            self.assertTrue(torch.isfinite(tensor).all())

    def test_backbone_freezing_does_not_freeze_hst(self):
        model = Net(n_class=4, rectifier_type="hst")
        model.train()
        self.assertFalse(model.conv1a.weight.requires_grad)
        self.assertFalse(model.bn7.weight.requires_grad)
        self.assertFalse(model.bn7.training)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.hst_rectifier.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
