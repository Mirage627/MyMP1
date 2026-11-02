"""Numerical tests for the empirical controller, independent of robotics stacks."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adaptive_flow_test_module", ROOT / "common/adaptive_flow.py")
flow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = flow
SPEC.loader.exec_module(flow)


class AdaptiveFlowTests(unittest.TestCase):
    def test_zero_residual_constant_velocity_is_exact_in_one_step(self):
        initial = torch.randn(5, 4, 2, dtype=torch.float64)
        def field(z, t, ids):
            return torch.ones_like(z) * 3, torch.full_like(t, -100.)
        result = flow.adaptive_euler(field, initial, eta=.1, max_steps=8)
        torch.testing.assert_close(result.samples, initial + 3, rtol=0, atol=0)
        self.assertEqual(result.nfe.tolist(), [1] * 5)
        self.assertEqual(result.network_calls, 1)

    def test_active_compaction_keeps_original_condition_indices(self):
        seen = []
        initial = torch.zeros(3, 2, 1, dtype=torch.float64)
        def field(z, t, ids):
            seen.append(ids.tolist())
            variance = torch.tensor([1e-10, 1., 4.], dtype=z.dtype)[ids]
            velocity = (ids + 1).to(z.dtype)[:, None, None].expand_as(z)
            return velocity, variance.log()
        result = flow.adaptive_euler(field, initial, eta=.25, max_steps=8)
        torch.testing.assert_close(result.samples[:, 0, 0], torch.tensor([1., 2., 3.], dtype=initial.dtype))
        self.assertEqual(result.nfe.tolist(), [1, 4, 8])
        self.assertEqual(seen[0], [0, 1, 2])
        self.assertEqual(seen[1], [1, 2])
        self.assertEqual(seen[-1], [2])

    def test_endpoint_roundoff_does_not_add_a_network_evaluation(self):
        # Change of units from summed to mean residuals can put a mathematically
        # exact endpoint a few ULPs below 1, even with a float64 clock.
        cases = ((1.6 / 160, .1, 1), (1. / 8, .25 / (8 ** .5), 4))
        for variance, eta, expected_steps in cases:
            def field(z, t, ids):
                log_variance = torch.tensor(variance, dtype=z.dtype).log()
                return torch.ones_like(z), log_variance.expand_as(t)
            with self.subTest(variance=variance, eta=eta):
                result = flow.adaptive_euler(
                    field, torch.zeros(1, 2, 1, dtype=torch.float64),
                    eta=eta, max_steps=8)
                self.assertEqual(result.nfe.tolist(), [expected_steps])
                self.assertEqual(result.network_calls, expected_steps)
                torch.testing.assert_close(result.samples, torch.ones_like(result.samples),
                                           rtol=0, atol=1e-15)

    def test_every_candidate_reaches_endpoint_and_budget_is_bounded(self):
        for budget in (1, 3, 7, 8, 17):
            def field(z, t, ids):
                return torch.ones_like(z), torch.full_like(t, 8.)
            result = flow.adaptive_euler(field, torch.zeros(2, 1), eta=1e-6, max_steps=budget)
            torch.testing.assert_close(result.samples, torch.ones_like(result.samples))
            self.assertEqual(result.nfe.tolist(), [budget, budget])
            if budget > 1:
                self.assertGreater(result.min_step_clamps, 0)

    def test_bad_variance_falls_back_but_bad_velocity_fails(self):
        def field(z, t, ids):
            return torch.ones_like(z), torch.full_like(t, float("nan"))
        result = flow.adaptive_euler(field, torch.zeros(2, 1), max_steps=4)
        self.assertEqual(result.variance_fallbacks, 8)
        self.assertEqual(result.nfe.tolist(), [4, 4])
        torch.testing.assert_close(result.samples, torch.ones_like(result.samples))
        def broken(z, t, ids):
            return z * float("nan"), torch.zeros_like(t)
        with self.assertRaises(FloatingPointError):
            flow.adaptive_euler(broken, torch.zeros(1, 2))

    def test_paired_rotation_preserves_schedule_and_samples(self):
        initial = torch.randn(9, 3, 2, dtype=torch.float64)
        angle = .7
        rotation = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
                                 [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]], dtype=initial.dtype)
        # Make exactly orthogonal in float64 for the paired comparison.
        rotation = torch.linalg.qr(rotation).Q
        def field(z, t, ids):
            return -.7 * z, z.square().mean((1, 2)).clamp_min(1e-8).log()
        expected = flow.adaptive_euler(field, initial, eta=.2, max_steps=8)
        actual = flow.adaptive_euler(field, initial @ rotation, eta=.2, max_steps=8)
        torch.testing.assert_close(actual.samples, expected.samples @ rotation)
        torch.testing.assert_close(actual.nfe, expected.nfe)

    def test_fixed_euler_matches_known_recurrence(self):
        noise = torch.tensor([[2.], [3.]])
        result = flow.fixed_euler(lambda z, t, ids: z, noise, steps=4)
        torch.testing.assert_close(result.samples, noise * 1.25 ** 4)
        self.assertEqual(result.nfe.tolist(), [4, 4])

    def test_variance_loss_calibrates_and_does_not_train_velocity(self):
        velocity = torch.tensor([[[1., 3.]]], requires_grad=True)
        log_var = torch.tensor([0.], requires_grad=True)
        loss, residual, _ = flow.residual_variance_loss(velocity, torch.zeros_like(velocity), log_var)
        loss.backward()
        self.assertIsNone(velocity.grad)
        self.assertLess(log_var.grad.item(), 0.)  # variance should rise towards 5
        torch.testing.assert_close(residual, torch.tensor([5.]))
        optimum = torch.tensor([5.]).log().requires_grad_()
        loss, _, _ = flow.residual_variance_loss(velocity, torch.zeros_like(velocity), optimum)
        loss.backward()
        self.assertAlmostEqual(optimum.grad.item(), 0., places=6)

    def test_invalid_inputs(self):
        field = lambda z, t, ids: (z, t)
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                flow.adaptive_euler(field, torch.zeros(1, 2), max_steps=budget)
        for eta in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                flow.adaptive_euler(field, torch.zeros(1, 2), eta=eta)
        with self.assertRaises(ValueError):
            flow.adaptive_euler(lambda z, t, ids: (z, t[:, None]), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            flow.fixed_euler(lambda z, t, ids: z, torch.empty(0, 2))


if __name__ == "__main__":
    unittest.main()
