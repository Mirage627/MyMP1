"""CPU checks of the H second-moment estimator and separated losses."""

import importlib.util
from pathlib import Path
import unittest

import torch


SPEC = importlib.util.spec_from_file_location(
    "meanflow_statistics", Path(__file__).resolve().parents[1] / "common" / "meanflow_statistics.py")
statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(statistics)


class SecondMomentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(57)
        torch.set_num_threads(1)

    def head(self, mode, **kwargs):
        return statistics.ResidualSecondMomentHead(4, 3, 2, mode=mode, hidden_dim=12, **kwargs).double()

    def test_initial_spd_shapes_and_nonzero_full_factor(self):
        features = torch.randn(5, 3, 4, dtype=torch.float64)
        for mode in ("diag", "full"):
            head = self.head(mode)
            covariance = head(features)
            self.assertEqual(covariance.shape, (5, 6, 6))
            self.assertEqual(covariance.dtype, torch.float64)
            self.assertEqual(head.D, 6)
            self.assertEqual(head.mode, mode)
            torch.testing.assert_close(covariance, torch.eye(6, dtype=torch.float64).expand(5, -1, -1),
                                       atol=1e-7, rtol=1e-7)
            self.assertTrue(bool((torch.linalg.eigvalsh(covariance) > 0).all()))
            if mode == "full":
                diagonal = head.tril_rows == head.tril_cols
                self.assertTrue(bool((head.network[-1].bias[diagonal] != 0).all()))

    def test_full_head_models_cross_time_correlations(self):
        head = self.head("full")
        with torch.no_grad():
            index = ((head.tril_rows == 4) & (head.tril_cols == 0)).nonzero().item()
            head.network[-1].bias[index] = 0.6
        covariance = head(torch.randn(2, 3, 4, dtype=torch.float64))
        self.assertGreater(covariance[0, 0, 4].item(), 0.5)
        self.assertTrue(bool((torch.linalg.eigvalsh(covariance) > head.min_variance).all()))

    def test_diagonal_and_full_match_given_same_statistics(self):
        q = torch.tensor([.2, .5, 1., 2., 3., 7.], dtype=torch.float64)
        diagonal, full = self.head("diag"), self.head("full")
        with torch.no_grad():
            remaining = q - diagonal.min_variance
            diagonal.network[-1].bias.copy_(remaining + torch.log(-torch.expm1(-remaining)))
            full.network[-1].bias.zero_()
            full.network[-1].bias[full.tril_rows == full.tril_cols] = remaining.sqrt()
        features = torch.randn(4, 3, 4, dtype=torch.float64)
        covariance_diagonal, covariance_full = diagonal(features), full(features)
        torch.testing.assert_close(covariance_diagonal, covariance_full, atol=1e-12, rtol=1e-12)
        error = torch.randn(4, 3, 2, dtype=torch.float64)
        torch.testing.assert_close(statistics.gaussian_second_moment_nll(error, covariance_diagonal),
                                   statistics.gaussian_second_moment_nll(error, covariance_full))
        torch.testing.assert_close(statistics.bounded_metric(covariance_diagonal),
                                   statistics.bounded_metric(covariance_full))

    def test_spectral_formula_bounds_trace_and_rotation(self):
        q = torch.tensor([[.01, .3, 4., 20.], [.2, 2., 9., 10.]], dtype=torch.float64)
        orthogonal = torch.linalg.qr(torch.randn(2, 4, 4, dtype=torch.float64)).Q
        covariance = orthogonal @ torch.diag_embed(q) @ orthogonal.transpose(-1, -2)
        rho, ridge = .4, .03
        actual = statistics.bounded_metric(covariance, ridge=ridge, rho=rho)
        ratios = (ridge + q[:, :1]) / (ridge + q)
        expected_eigenvalues = 1 + rho * (ratios - ratios.mean(-1, keepdim=True))
        expected = orthogonal @ torch.diag_embed(expected_eigenvalues) @ orthogonal.transpose(-1, -2)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertFalse(actual.requires_grad)
        spectrum = torch.linalg.eigvalsh(actual)
        self.assertTrue(bool((spectrum >= 1 - rho).all()))
        self.assertTrue(bool((spectrum <= 1 + rho).all()))
        torch.testing.assert_close(actual.diagonal(dim1=-2, dim2=-1).sum(-1),
                                   torch.full((2,), 4., dtype=torch.float64))

    def test_isotropic_and_single_dimension_reduce_to_identity(self):
        for dimension in (1, 6):
            for scale in (0., .001, 3., 100.):
                covariance = (scale * torch.eye(dimension, dtype=torch.float64)).unsqueeze(0)
                weight = statistics.bounded_metric(covariance, rho=.7)
                torch.testing.assert_close(weight, torch.eye(dimension, dtype=torch.float64).unsqueeze(0),
                                           rtol=0, atol=0)

    def test_two_losses_have_disjoint_parameter_gradients(self):
        for mode in ("diag", "full"):
            head = self.head(mode)
            features = torch.randn(4, 3, 4, dtype=torch.float64, requires_grad=True)
            error = torch.randn(4, 3, 2, dtype=torch.float64, requires_grad=True)
            covariance = head(features)
            weight = statistics.bounded_metric(covariance)
            velocity_loss = statistics.weighted_meanflow_loss(error, weight)
            velocity_gradients = torch.autograd.grad(
                velocity_loss, (error, features, *head.parameters()), allow_unused=True)
            self.assertGreater(velocity_gradients[0].abs().sum().item(), 0)
            self.assertTrue(all(g is None for g in velocity_gradients[1:]))
            head_loss = statistics.gaussian_second_moment_nll(error, covariance)
            head_gradients = torch.autograd.grad(
                head_loss, (error, features, *head.parameters()), allow_unused=True)
            self.assertIsNone(head_gradients[0])
            self.assertIsNone(head_gradients[1])
            self.assertGreater(sum(g.abs().sum().item() for g in head_gradients[2:] if g is not None), 0)

    def test_losses_defensively_detach_direct_error_and_weight_graphs(self):
        error = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
        weight = torch.eye(6, dtype=torch.float64).expand(2, -1, -1).clone().requires_grad_()
        velocity = statistics.weighted_meanflow_loss(error, weight)
        error_grad, weight_grad = torch.autograd.grad(velocity, (error, weight), allow_unused=True)
        self.assertIsNotNone(error_grad)
        self.assertIsNone(weight_grad)
        covariance = weight * 2
        nll = statistics.gaussian_second_moment_nll(error, covariance)
        error_grad, covariance_grad = torch.autograd.grad(nll, (error, weight), allow_unused=True)
        self.assertIsNone(error_grad)
        self.assertIsNotNone(covariance_grad)

    def test_rho_zero_exact_mse_while_head_still_learns(self):
        head = self.head("full")
        features = torch.randn(3, 3, 4, dtype=torch.float64)
        error = torch.randn(3, 3, 2, dtype=torch.float64, requires_grad=True)
        covariance = head(features)
        loss = statistics.weighted_meanflow_loss(error, statistics.bounded_metric(covariance, rho=0))
        mse = error.square().mean()
        torch.testing.assert_close(loss, mse, atol=0, rtol=0)
        actual_grad = torch.autograd.grad(loss, error)[0]
        expected_grad = torch.autograd.grad(mse, error)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        optimizer = torch.optim.SGD(head.parameters(), lr=.02)
        before = head(features).detach().clone()
        statistics.gaussian_second_moment_nll(error, covariance).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, head(features)))

    def test_conditional_nll_recovers_second_moment_including_bias(self):
        covariance = torch.tensor([[2., .6], [.6, 1.]], dtype=torch.float64)
        centered = math_sqrt_two() * torch.linalg.cholesky(covariance).T
        residuals = torch.cat((centered, -centered), dim=0) + torch.tensor([1., -.4], dtype=torch.float64)
        second_moment = residuals.T @ residuals / residuals.shape[0]
        expected = covariance + torch.outer(torch.tensor([1., -.4], dtype=torch.float64),
                                             torch.tensor([1., -.4], dtype=torch.float64))
        torch.testing.assert_close(second_moment, expected)
        estimate = second_moment.clone().requires_grad_()
        optimum = statistics.gaussian_second_moment_nll(residuals, estimate.expand(4, -1, -1))
        gradient = torch.autograd.grad(optimum, estimate)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
        for alternative in (covariance, .6 * second_moment, 1.6 * second_moment,
                            torch.diag(torch.diagonal(second_moment))):
            loss = statistics.gaussian_second_moment_nll(residuals, alternative.expand(4, -1, -1))
            self.assertGreater(loss.item(), optimum.item())

    def test_time_mask_does_not_remove_head_supervision(self):
        t = torch.tensor([.7, .7, 1., .6, .999, .5], dtype=torch.float64)
        r = torch.tensor([.2, .7, .2, 0., .2, .001], dtype=torch.float64)
        mask = statistics.metric_time_mask(t, r, endpoint_margin=.001)
        self.assertEqual(mask.tolist(), [True, False, False, False, False, False])
        covariance = torch.diag(torch.tensor([.1, 3.], dtype=torch.float64)).expand(6, -1, -1)
        weight = statistics.bounded_metric(covariance, active_mask=mask)
        torch.testing.assert_close(weight[~mask], torch.eye(2, dtype=torch.float64).expand(5, -1, -1),
                                   atol=0, rtol=0)
        self.assertFalse(torch.equal(weight[0], torch.eye(2, dtype=torch.float64)))

    def test_half_precision_inputs_use_float32_matrix_math(self):
        covariance = torch.diag(torch.tensor([.2, 2.], dtype=torch.float16)).unsqueeze(0)
        error = torch.tensor([[.5, 1.]], dtype=torch.float16)
        self.assertEqual(statistics.bounded_metric(covariance).dtype, torch.float32)
        self.assertEqual(statistics.gaussian_second_moment_nll(error, covariance).dtype, torch.float32)

    def test_invalid_parameters_shapes_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):
            self.head("unknown")
        with self.assertRaises(ValueError):
            self.head("full", initial_variance=1e-5)
        with self.assertRaises(ValueError):
            self.head("diag")(torch.randn(2, 4, 3, dtype=torch.float64))
        for rho in (-.1, 1., float("nan")):
            with self.assertRaises(ValueError):
                statistics.bounded_metric(torch.eye(2).unsqueeze(0), rho=rho)
        with self.assertRaises(FloatingPointError):
            statistics.bounded_metric(torch.full((1, 2, 2), float("nan")))
        with self.assertRaises(ValueError):
            statistics.bounded_metric(torch.tensor([[[1., .2], [0., 1.]]]))
        with self.assertRaises(ValueError):
            statistics.bounded_metric(torch.diag(torch.tensor([-1., 1.])).unsqueeze(0))
        with self.assertRaises(ValueError):
            statistics.metric_time_mask(torch.tensor([.2]), torch.tensor([.5]))
        with self.assertRaises(ValueError):
            statistics.weighted_meanflow_loss(torch.zeros(2, 3), torch.eye(2).expand(2, -1, -1))
        with self.assertRaises(FloatingPointError):
            statistics.gaussian_second_moment_nll(torch.tensor([[float("inf"), 0.]]), torch.eye(2).unsqueeze(0))


def math_sqrt_two():
    return 2.0 ** 0.5


if __name__ == "__main__":
    unittest.main()
