"""Pure PyTorch checks for the standalone variance proxy; no simulator needed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "meanflow_variance", ROOT / "experiments" / "meanflow_eval" / "variance.py")
variance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(variance)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class VarianceProbeTests(unittest.TestCase):
    def test_constant_mse_has_expected_standard_deviation_and_reloads(self):
        features = torch.arange(80, dtype=torch.float32).reshape(20, 4)
        targets = torch.full((20,), 4.)
        rng = torch.random.get_rng_state().clone()
        payload = variance.fit_probe_from_arrays(features, targets, steps=20, seed=3)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.pt"
            torch.save(payload, path)
            probe = variance.VarianceProbe(path)
            scores = probe.predict_features(features)
            self.assertTrue(torch.allclose(scores, torch.full((20,), 2.), atol=.08))
            self.assertEqual(probe.metadata["kind"], "variance_proxy")

    def test_rejects_bad_data_and_feature_dimensions(self):
        x = torch.zeros(8, 4)
        for target in (torch.full((8,), -1.), torch.full((8,), float("nan")), torch.zeros(7)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                variance.fit_probe_from_arrays(x, target, steps=1)
        with self.assertRaises(ValueError):
            variance.fit_probe_from_arrays(torch.zeros(0, 4), torch.zeros(0), steps=1)
        with self.assertRaises(ValueError):
            variance.fit_probe_from_arrays(x, torch.ones(8), steps=0)
        payload = variance.fit_probe_from_arrays(x, torch.ones(8), steps=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.pt"
            torch.save(payload, path)
            probe = variance.VarianceProbe(path)
            with self.assertRaises(ValueError):
                probe.predict_features(torch.zeros(1, 3))
            with self.assertRaises(ValueError):
                probe.predict_features(torch.full((1, 4), float("inf")))

    def test_offline_training_freezes_policy_and_score_skips_velocity(self):
        try:
            import hydra.utils
        except ImportError:
            self.skipTest("hydra is not installed")

        class IdentityNormalizer:
            def normalize(self, value):
                return value

            def __getitem__(self, key):
                return self

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 3)

            def forward(self, obs):
                return self.linear(obs["x"].reshape(obs["x"].shape[0], 2))

        class Velocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.2))
                self.calls = 0

            def forward(self, *, z, t, r, global_cond):
                self.calls += 1
                return z * self.weight

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc, self.diff = Encoder(), Velocity()
                self.normalizer = IdentityNormalizer()
                self.horizon, self.action_dim, self.n_obs_steps = 2, 1, 2
                self.obs_as_global_cond = True

        data = [{"obs": {"x": torch.tensor([[i / 10.], [i / 10.]])},
                 "action": torch.tensor([[i / 10.], [i / 10.]])} for i in range(12)]
        policy = Policy().train()
        policy.enc.eval()  # Mixed modes must survive temporary policy.eval().
        before = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
        grad_flags = [parameter.requires_grad for parameter in policy.parameters()]
        modes = [module.training for module in policy.modules()]
        rng = torch.random.get_rng_state().clone()
        cfg = {"task": {"dataset": {"_target_": "dummy.train_split", "dataset_path": "train_only"}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.pt"
            with patch("hydra.utils.instantiate", return_value=data) as instantiate:
                metadata = variance.train_variance_probe(
                    policy, cfg, path, device="cpu", batches=4, batch_size=4, seed=4)
                instantiate.assert_called_once_with(cfg["task"]["dataset"])
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(metadata["n_training_samples"], 16)
            self.assertEqual(modes, [module.training for module in policy.modules()])
            self.assertEqual(grad_flags, [parameter.requires_grad for parameter in policy.parameters()])
            for key, tensor in policy.state_dict().items():
                self.assertTrue(torch.equal(before[key], tensor), key)
            probe = variance.VarianceProbe(path)
            calls = policy.diff.calls
            score = probe.score(policy, {"x": torch.zeros(1, 2, 1)}, torch.zeros(1, 2, 1))
            self.assertGreater(score, 0)
            self.assertEqual(calls, policy.diff.calls)
            self.assertEqual(modes, [module.training for module in policy.modules()])
            with self.assertRaises(FileExistsError):
                variance.train_variance_probe(policy, cfg, path, device="cpu", batches=1)
            with self.assertRaises(ValueError):
                probe.score(policy, {"x": torch.zeros(1, 2, 1)}, torch.zeros(2, 2))


if __name__ == "__main__":
    unittest.main()
