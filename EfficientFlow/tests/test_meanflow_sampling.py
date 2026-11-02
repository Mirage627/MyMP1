"""CPU unit tests; schedule tests also run when torch is not installed."""

import ast
import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "meanflow_sampling", ROOT / "common" / "meanflow_sampling.py")
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)

try:
    import torch
except ImportError:
    torch = None


class TimeGridTests(unittest.TestCase):
    def test_default_and_uniform(self):
        self.assertEqual(sampling.resolve_time_grid(), (1.0, 0.0))
        self.assertEqual(sampling.make_time_grid(4), (1.0, .75, .5, .25, 0.0))
        self.assertEqual(sampling.resolve_time_grid(num_inference_steps=2), (1.0, .5, 0.0))

    def test_density_conventions_and_aliases(self):
        noise = sampling.make_time_grid(4, "noise_first", power=2)
        data = sampling.make_time_grid(4, "data_first", power=2)
        self.assertLess(noise[0] - noise[1], noise[-2] - noise[-1])
        self.assertGreater(data[0] - data[1], data[-2] - data[-1])
        self.assertEqual(noise, sampling.make_time_grid(4, "noise_dense"))
        self.assertEqual(data, sampling.make_time_grid(4, "data_dense"))
        for family in ("uniform", "noise_first", "data_first"):
            self.assertEqual(sampling.make_time_grid(1, family), (1.0, 0.0))

    def test_explicit_grid(self):
        self.assertEqual(sampling.validate_time_grid([1, .8, .1, 0]), (1., .8, .1, 0.))
        self.assertEqual(sampling.resolve_time_grid([1, .8, 0], 2), (1., .8, 0.))
        with self.assertRaises(ValueError):
            sampling.resolve_time_grid([1, .8, 0], 3)

    def test_invalid_grids(self):
        for grid in (None, [], [1], [0, 1], [1, .5], [.9, 0],
                     [1, .5, .5, 0], [1, .4, .6, 0], [1, -1, 0],
                     [1, 1.1, 0], [1, math.nan, 0], [1, math.inf, 0],
                     [True, 0], "1,0", [[1], [0]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                sampling.validate_time_grid(grid)

    def test_invalid_generation_arguments(self):
        for steps in (0, -1, True, 1.5, "2"):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                sampling.make_time_grid(steps)
        for power in (0, -1, math.nan, math.inf, "bad"):
            with self.subTest(power=power), self.assertRaises(ValueError):
                sampling.make_time_grid(2, power=power)
        with self.assertRaises(ValueError):
            sampling.make_time_grid(2, "unknown")


@unittest.skipIf(torch is None, "torch is not installed")
class MeanFlowSamplingTests(unittest.TestCase):
    def setUp(self):
        self.data = torch.zeros(2, 4, 3)
        self.noise = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    def test_default_matches_original_one_step(self):
        calls = []

        def velocity(z, t, r):
            calls.append((t.clone(), r.clone()))
            return 2 * z + 3

        actual = sampling.sample_meanflow(velocity, self.data, initial_noise=self.noise)
        torch.testing.assert_close(actual, self.noise - (2 * self.noise + 3), rtol=0, atol=0)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0][0], torch.ones(2))
        torch.testing.assert_close(calls[0][1], torch.zeros(2))

    def test_each_interval_recomputes_average_velocity(self):
        seen = []

        def velocity(z, t, r):
            seen.append((z.clone(), float(t[0]), float(r[0])))
            return z + (t - r).reshape(-1, 1, 1)

        actual = sampling.sample_meanflow(
            velocity, self.data, initial_noise=self.noise, time_grid=[1, .75, 0])
        first = self.noise - .25 * (self.noise + .25)
        expected = first - .75 * (first + .75)
        torch.testing.assert_close(actual, expected)
        self.assertEqual([(item[1], item[2]) for item in seen], [(1., .75), (.75, 0.)])
        torch.testing.assert_close(seen[1][0], first)

    def test_inpainting_before_every_call_and_after_final_update(self):
        mask = torch.zeros_like(self.data, dtype=torch.bool)
        mask[:, :2, 1:] = True
        self.data[mask] = 7

        def velocity(z, t, r):
            torch.testing.assert_close(z[mask], self.data[mask])
            return torch.ones_like(z)

        original = self.noise.clone()
        actual = sampling.sample_meanflow(
            velocity, self.data, mask, time_grid=[1, .5, 0], initial_noise=self.noise)
        torch.testing.assert_close(actual[mask], self.data[mask])
        torch.testing.assert_close(actual[~mask], (original - 1)[~mask])
        torch.testing.assert_close(self.noise, original, rtol=0, atol=0)

    def test_generator_reproducibility(self):
        first = torch.Generator().manual_seed(37)
        second = torch.Generator().manual_seed(37)
        velocity = lambda z, t, r: torch.zeros_like(z)
        a = sampling.sample_meanflow(velocity, self.data, generator=first)
        b = sampling.sample_meanflow(velocity, self.data, generator=second)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_bad_noise_mask_and_velocity(self):
        velocity = lambda z, t, r: torch.zeros_like(z)
        with self.assertRaises(ValueError):
            sampling.sample_meanflow(velocity, self.data, initial_noise=torch.zeros(1))
        with self.assertRaises(ValueError):
            sampling.sample_meanflow(velocity, self.data, condition_mask=self.data)
        with self.assertRaises(ValueError):
            sampling.sample_meanflow(lambda z, t, r: (z, None), self.data)

    def test_policy_adapters_without_optional_vision_dependencies(self):
        # Compile the inference method and its preprocessing helper, then supply a tiny
        # encoder/model. This exercises the real adapters without importing
        # robomimic, point-cloud extensions, or equivariant CUDA dependencies.
        from typing import Dict

        class IdentityNormalizer:
            def normalize(self, obs):
                return dict(obs)

            def __getitem__(self, key):
                return self

            def unnormalize(self, action):
                return action

        class StubModel:
            def __init__(self, kind):
                self.kind = kind
                self.calls = []

            def eval(self):
                return self

            def __call__(self, **kwargs):
                if self.kind == "mp1":
                    self.assert_inference(kwargs)
                    z, t, r = kwargs["sample"], kwargs["timestep"], kwargs["r"]
                else:
                    z, t, r = kwargs["z"], kwargs["t"], kwargs["r"]
                self.calls.append((float(t[0]), float(r[0])))
                return torch.ones_like(z) * (t - r).reshape(-1, 1, 1)

            @staticmethod
            def assert_inference(kwargs):
                if kwargs.get("training") is not False:
                    raise AssertionError("MP1 must request its tensor-only inference output")

        obs = {"point_cloud": torch.zeros(2, 2, 5, 3)}
        for filename, class_name, kind in (
                ("mp1.py", "Meanpolicy", "mp1"),
                ("EquiMeanFlow.py", "EquiMeanFlowPolicy", "equi")):
            with self.subTest(policy=filename):
                tree = ast.parse((ROOT / "policy" / filename).read_text(encoding="utf-8"))
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == class_name)
                methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                           and node.name in {"predict_action", "_normalize_observations"}]
                namespace = {"Dict": Dict, "torch": torch,
                             "resolve_time_grid": sampling.resolve_time_grid,
                             "sample_meanflow": sampling.sample_meanflow,
                             "dict_apply": lambda d, fn: {k: fn(v) for k, v in d.items()}}
                exec(compile(ast.Module(body=methods, type_ignores=[]), filename, "exec"), namespace)
                policy = type("StubPolicy", (), {
                    method.name: namespace[method.name] for method in methods})()
                policy.normalizer = IdentityNormalizer()
                policy.horizon, policy.action_dim, policy.obs_feature_dim = 4, 3, 2
                policy.n_obs_steps, policy.n_action_steps = 2, 2
                policy.compared_action_steps = 2
                policy.device, policy.dtype = torch.device("cpu"), torch.float32
                policy.use_pc_color = False
                policy.obs_as_global_cond, policy.condition_type = True, "film"
                model = StubModel(kind)
                policy.model = policy.diff = model
                policy.obs_encoder = lambda d: torch.zeros(4, 2)
                policy.enc = lambda d: torch.zeros(2, 2, 2)
                noise = torch.zeros(2, 4, 3)
                # Existing second positional arguments remain accepted.
                default = policy.predict_action(obs, None, initial_noise=noise)
                torch.testing.assert_close(default["action_pred"], noise - 1)
                self.assertEqual(policy.last_sampling_nfe, 1)
                result = policy.predict_action(obs, None, initial_noise=noise, num_inference_steps=2)
                torch.testing.assert_close(result["action_pred"], noise - .5)
                self.assertEqual(policy.last_sampling_nfe, 2)
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
