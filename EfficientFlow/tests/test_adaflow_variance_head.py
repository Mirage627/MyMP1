"""CPU tests of the invariant variance head; no escnn or simulator required."""

import ast
import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "invariant_variance_head", ROOT / "model/equi/invariant_variance_head.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
InvariantVarianceHead = MODULE.InvariantVarianceHead


class InvariantVarianceHeadTests(unittest.TestCase):
    def test_initial_variance_one_and_one_scalar_per_trajectory(self):
        head = InvariantVarianceHead(3, 16)
        actual = head(torch.randn(4, 8, 3, 8))
        self.assertEqual(actual.shape, (4,))
        torch.testing.assert_close(actual, torch.zeros(4), rtol=0, atol=0)

    def test_group_permutation_invariance_after_learning(self):
        head = InvariantVarianceHead(3, 16).double()
        # A trained output must be invariant too; a zero-initialized head alone
        # would make an invariance check vacuous.
        with torch.no_grad():
            head.net[-1].weight.normal_()
            head.net[-1].bias.normal_()
        features = torch.randn(4, 8, 3, 8, dtype=torch.float64)
        expected = head(features)
        for shift in range(8):
            torch.testing.assert_close(head(features.roll(shift, dims=-1)), expected)
        torch.testing.assert_close(head(features[..., torch.randperm(8)]), expected)

    def test_second_moment_retains_zero_mean_group_signal(self):
        head = InvariantVarianceHead(1, 1).double()
        with torch.no_grad():
            head.net[0].weight.copy_(torch.tensor([[0., 1.]], dtype=torch.float64))
            head.net[0].bias.zero_()
            head.net[-1].weight.fill_(1.)
        features = torch.tensor([[[[1., -1.]]], [[[2., -2.]]]], dtype=torch.float64)
        values = head(features)
        self.assertGreater(values[1].item(), values[0].item())

    def test_default_detach_isolates_backbone_but_head_trains(self):
        head = InvariantVarianceHead(3, 16)
        features = torch.randn(4, 8, 3, 8, requires_grad=True)
        loss = (head(features) - 2.).square().mean()
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertGreater(head.net[-1].bias.grad.abs().item(), 0.)

    def test_optional_joint_gradient_flows_to_features(self):
        head = InvariantVarianceHead(3, 16)
        with torch.no_grad():
            head.net[-1].weight.fill_(1.)
        features = torch.randn(4, 8, 3, 8, requires_grad=True)
        head(features, detach_features=False).sum().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.)

    def test_dynamic_attachment_preserves_existing_parameters_and_mode(self):
        # Exercise the real attachment method independently of escnn's optional
        # import. Full equivariant forward coverage requires escnn/robomimic.
        source = (ROOT / "model/equi/equi_conditional_unet1d.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        backbone = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "EquiDiffusionUNet")
        method = next(node for node in backbone.body if isinstance(node, ast.FunctionDef)
                      and node.name == "enable_variance_prediction")
        namespace = {"InvariantVarianceHead": InvariantVarianceHead}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "attachment", "exec"), namespace)

        class Backbone(torch.nn.Module):
            enable_variance_prediction = namespace["enable_variance_prediction"]

            def __init__(self):
                super().__init__()
                self.unet = torch.nn.Linear(3, 3).double()
                self.act_emb_dim = 3
                self.variance_head = None

        model = Backbone().eval()
        before = {key: value.clone() for key, value in model.unet.state_dict().items()}
        self.assertIs(model.enable_variance_prediction(16), model)
        first_head = model.variance_head
        self.assertIs(model.enable_variance_prediction(16).variance_head, first_head)
        self.assertFalse(first_head.training)
        self.assertEqual(next(first_head.parameters()).dtype, torch.float64)
        for key, value in model.unet.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_invalid_inputs(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                InvariantVarianceHead(value)
            with self.assertRaises(ValueError):
                InvariantVarianceHead(3, value)
        head = InvariantVarianceHead(3)
        for features in (torch.zeros(2, 3), torch.zeros(2, 4, 2, 8),
                         torch.zeros(0, 4, 3, 8), torch.zeros(2, 4, 3, 8, dtype=torch.long)):
            with self.subTest(shape=features.shape), self.assertRaises(ValueError):
                head(features)


if __name__ == "__main__":
    unittest.main()
