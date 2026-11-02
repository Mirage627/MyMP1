"""Real policy integration with small substitute robotics dependencies.

Only the old policy/encoder are substituted. The new policy, flow solvers,
residual loss, and invariant variance head execute their production code.
"""

import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def _file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules):
    HEAD = _file_module("_adaflow_integration_head", ROOT / "model/equi/invariant_variance_head.py")


class ActionNormalizer:
    def normalize(self, value):
        return (value - 5.) / 2.

    def unnormalize(self, value):
        return value * 2. + 5.


class Normalizer:
    def normalize(self, value):
        return dict(value)

    def __getitem__(self, key):
        if key != "action":
            raise KeyError(key)
        return ActionNormalizer()


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.seen_obs_steps = None

    def forward(self, obs):
        self.seen_obs_steps = obs["x"].shape[1]
        return obs["x"] * self.weight


class TinyDiff(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.variance_head = None
        self.calls = []
        self.constant_velocity = False
        self.heterogeneous_sampling_variance = False
        self.bad_velocity = False
        self.bad_variance = False

    def enable_variance_prediction(self, hidden_dim=128):
        if self.variance_head is None:
            self.variance_head = HEAD.InvariantVarianceHead(2, hidden_dim)
        return self

    def forward(self, z, timestep, *, global_cond, return_variance=False,
                detach_variance_features=True):
        marker = global_cond[:, 0]
        self.calls.append({"condition": global_cond.detach().clone(),
                           "times": timestep.detach().clone(),
                           "batch": z.shape[0], "return_variance": return_variance})
        velocity = marker[:, None, None].expand_as(z)
        if not self.constant_velocity:
            velocity = velocity + z * self.weight
        if self.bad_velocity:
            velocity = velocity * float("nan")
        if not return_variance:
            return velocity
        features = torch.stack((z.mean((1, 2)), global_cond.mean(1)), dim=1)
        log_variance = self.variance_head(features[:, None, :, None],
                                        detach_features=detach_variance_features)
        if self.heterogeneous_sampling_variance:
            log_variance = torch.where(marker < 1.5,
                torch.full_like(marker, math.log(.01 ** 2)),
                torch.full_like(marker, math.log(4. ** 2)))
        if self.bad_variance:
            log_variance = log_variance * float("nan")
        return velocity, log_variance


class TinyOldPolicy(torch.nn.Module):
    def __init__(self, *, shape_meta=None, horizon=6, n_obs_steps=2,
                 n_action_steps=2, num_of_trajectories=3, rot_aug=False, **kwargs):
        super().__init__()
        self.action_dim = (shape_meta or {"action": {"shape": [10]}})["action"]["shape"][0]
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_of_trajectories = num_of_trajectories
        self.rot_aug = rot_aug
        self.enc = TinyEncoder()
        self.diff = TinyDiff()
        self.normalizer = Normalizer()
        self.predict_action_times_tlag = 0
        self.rot_randomizer = lambda obs, action: (obs, action)


def _load_policy():
    names = ("EfficientFlow", "EfficientFlow.common", "EfficientFlow.policy")
    stubs = {}
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module
    old_policy = types.ModuleType("EfficientFlow.policy.EfficientFlow")
    old_policy.EfficientFlowPolicy = TinyOldPolicy
    stubs[old_policy.__name__] = old_policy
    # patch.dict restores every prior module and removes newly imported stubs,
    # even if loading raises. No fake EfficientFlow package survives this call.
    with patch.dict(sys.modules, stubs):
        _file_module("EfficientFlow.common.adaptive_flow", ROOT / "common/adaptive_flow.py")
        _file_module("EfficientFlow.common.pytorch_util", ROOT / "common/pytorch_util.py")
        return _file_module("_adaflow_policy_integration", ROOT / "policy/EfficientAdaFlow.py")


POLICY_MODULE = _load_policy()
Policy = POLICY_MODULE.EfficientAdaFlowPolicy


def observations(batch=2):
    # Extra observation steps must be truncated by the production policy.
    return {"x": torch.arange(1, batch + 1, dtype=torch.float32)[:, None, None].expand(-1, 4, 1)}


def training_batch(policy):
    count = 2 * policy.horizon * policy.action_dim
    normalized = torch.linspace(-.8, .8, count).reshape(2, policy.horizon, policy.action_dim)
    return {"obs": observations(), "action": policy.normalizer["action"].unnormalize(normalized)}


def candidate_noise(policy, batch=2):
    rows = policy.num_of_trajectories * batch
    return (torch.arange(rows)[:, None, None] * .5
            + torch.arange(policy.horizon)[None, :, None] * .01
            + torch.arange(policy.action_dim)[None, None, :] * .001)


class EfficientAdaFlowIntegrationTests(unittest.TestCase):
    def test_stub_imports_are_restored_without_global_package_pollution(self):
        names = ("EfficientFlow", "EfficientFlow.common", "EfficientFlow.policy",
                 "EfficientFlow.common.adaptive_flow", "EfficientFlow.common.pytorch_util",
                 "EfficientFlow.policy.EfficientFlow", "_adaflow_policy_integration")
        missing = object()
        before = {name: sys.modules.get(name, missing) for name in names}
        loaded = _load_policy()
        self.assertTrue(issubclass(loaded.EfficientAdaFlowPolicy, TinyOldPolicy))
        for name in names:
            self.assertIs(sys.modules.get(name, missing), before[name], name)

    def test_joint_loss_has_gradients_and_variance_does_not_modify_velocity_gradient(self):
        without_nll = Policy(variance_warmup_steps=0, variance_loss_weight=0.)
        with_nll = Policy(variance_warmup_steps=0, variance_loss_weight=1.)
        with_nll.load_state_dict(without_nll.state_dict())
        for policy in (without_nll, with_nll):
            torch.manual_seed(29)
            loss = policy.compute_loss(training_batch(policy))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertEqual(policy.enc.seen_obs_steps, 2)
            self.assertGreater(policy.enc.weight.grad.abs().item(), 0.)
            self.assertGreater(policy.diff.weight.grad.abs().item(), 0.)
        torch.testing.assert_close(with_nll.enc.weight.grad, without_nll.enc.weight.grad)
        torch.testing.assert_close(with_nll.diff.weight.grad, without_nll.diff.weight.grad)
        self.assertGreater(with_nll.diff.variance_head.net[-1].bias.grad.abs().item(), 0.)
        self.assertEqual(without_nll.diff.variance_head.net[-1].bias.grad.item(), 0.)

    def test_training_counter_warmup_and_validation_do_not_change_training_state(self):
        policy = Policy(variance_warmup_steps=1)
        batch = training_batch(policy)
        policy.compute_loss(batch)
        self.assertEqual(policy._variance_updates.item(), 1)
        self.assertEqual(policy.last_loss_metrics["variance_ready"], 0.)
        policy.compute_loss(batch)
        self.assertEqual(policy._variance_updates.item(), 2)
        self.assertEqual(policy.last_loss_metrics["variance_ready"], 1.)
        policy.eval()
        policy.compute_loss(batch)
        self.assertEqual(policy._variance_updates.item(), 2)
        policy.train()
        with torch.no_grad():
            policy.compute_loss(batch)
        self.assertEqual(policy._variance_updates.item(), 2)

    def test_head_only_freezes_backbone_and_preserves_its_eval_mode(self):
        policy = Policy(variance_training="head_only", variance_warmup_steps=999)
        self.assertEqual(policy.variance_warmup_steps, 0)
        for mode in (False, True):
            policy.train(mode)
            self.assertFalse(policy.enc.training)
            self.assertFalse(policy.diff.training)
            self.assertEqual(policy.diff.variance_head.training, mode)
        before = {key: value.clone() for key, value in policy.state_dict().items()
                  if "variance_head" not in key and key != "_variance_updates"}
        policy.compute_loss(training_batch(policy)).backward()
        self.assertIsNone(policy.enc.weight.grad)
        self.assertIsNone(policy.diff.weight.grad)
        self.assertFalse(policy.enc.weight.requires_grad)
        self.assertFalse(policy.diff.weight.requires_grad)
        self.assertGreater(policy.diff.variance_head.net[-1].bias.grad.abs().item(), 0.)
        # An explicit SGD update avoids importing optional compile/export
        # integrations from the machine's installed optimizer environment.
        with torch.no_grad():
            for parameter in policy.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-.01)
        for key, value in before.items():
            torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)

    def test_fixed_and_adaptive_candidates_keep_batch_conditions_and_count_all_work(self):
        for sampler in ("fixed", "adaptive"):
            with self.subTest(sampler=sampler):
                policy = Policy(sampler=sampler, fixed_steps=3, adaptive_eta=1.,
                                adaptive_max_steps=4).eval()
                policy.diff.constant_velocity = True
                policy.diff.heterogeneous_sampling_variance = True
                noise = candidate_noise(policy)
                selected = torch.tensor([2, 1])
                with patch.object(torch, "randn", return_value=noise), \
                        patch.object(torch, "randint", return_value=selected):
                    result = policy.predict_action(observations())
                expected_candidates = noise.reshape(3, 2, 6, 10) + torch.tensor([1., 2.])[None, :, None, None]
                expected = policy.normalizer["action"].unnormalize(expected_candidates[selected, torch.arange(2)])
                torch.testing.assert_close(result["action_pred"], expected)
                torch.testing.assert_close(result["action"], expected[:, 1:3])
                self.assertEqual(policy.enc.seen_obs_steps, 2)
                torch.testing.assert_close(policy.diff.calls[0]["condition"][:, 0],
                                           torch.tensor([1., 2., 1., 2., 1., 2.]))
                if sampler == "adaptive":
                    self.assertEqual(len(policy.diff.calls), 4)
                    for call in policy.diff.calls[1:]:
                        torch.testing.assert_close(call["condition"][:, 0], torch.full((3,), 2.))
                    expected_mean, expected_evaluations = 2.5, 7.5
                else:
                    self.assertEqual(len(policy.diff.calls), 3)
                    self.assertTrue(all(not call["return_variance"] for call in policy.diff.calls))
                    self.assertTrue(all(call["batch"] == 6 for call in policy.diff.calls))
                    expected_mean, expected_evaluations = 3., 9.
                metrics = policy.get_sampling_metrics()
                self.assertEqual(metrics["sampled_candidates"], 6.)
                self.assertEqual(metrics["nfe_mean"], expected_mean)
                self.assertEqual(metrics["candidate_evaluations_per_prediction"], expected_evaluations)
                policy.reset()
                self.assertEqual(policy.get_sampling_metrics()["sampled_candidates"], 6.)
                with patch.object(torch, "randn", return_value=noise), \
                        patch.object(torch, "randint", return_value=selected):
                    policy.predict_action(observations())
                combined = policy.get_sampling_metrics()
                self.assertEqual(combined["sampled_candidates"], 12.)
                self.assertEqual(combined["nfe_mean"], expected_mean)
                self.assertEqual(combined["candidate_evaluations_per_prediction"], expected_evaluations)
                self.assertEqual(policy.get_sampling_metrics(reset=True), combined)
                self.assertEqual(policy.get_sampling_metrics(), {})

    def test_history_selection_overlap_periodic_random_reset_and_explicit_reset(self):
        policy = Policy(sampler="fixed", candidate_reset_interval=2).eval()
        policy.diff.constant_velocity = True
        noise = candidate_noise(policy)
        candidates = noise.reshape(3, 2, 6, 10) + torch.tensor([1., 2.])[None, :, None, None]
        wanted = torch.tensor([1, 2])
        previous = policy.normalizer["action"].unnormalize(candidates[wanted, torch.arange(2), :2])
        policy.predict_action_times_tlag = 1
        with patch.object(torch, "randn", return_value=noise), \
                patch.object(torch, "randint", side_effect=AssertionError("history must select candidates")):
            result = policy.predict_action(observations(), previous)
        expected = policy.normalizer["action"].unnormalize(candidates[wanted, torch.arange(2)])
        torch.testing.assert_close(result["action_pred"], expected)
        torch.testing.assert_close(result["action_to_compare"], expected[:, 2:4])
        self.assertEqual(policy.predict_action_times_tlag, 2)
        # The next periodic reset and an explicit reset both restore random
        # selection even when a previous trajectory is supplied.
        for explicit in (False, True):
            if explicit:
                policy.reset()
                self.assertEqual(policy.predict_action_times_tlag, 0)
            with patch.object(torch, "randn", return_value=noise), \
                    patch.object(torch, "randint", return_value=torch.zeros(2, dtype=torch.long)) as random:
                result = policy.predict_action(observations(), previous)
            random.assert_called_once()
            torch.testing.assert_close(result["action_pred"],
                policy.normalizer["action"].unnormalize(candidates[0]))

    def test_invalid_configuration_training_shape_and_history_shape(self):
        invalid = ({"sampler": "unknown"}, {"variance_training": "unknown"},
                   {"obs_as_global_cond": False}, {"adaptive_max_steps": 0},
                   {"fixed_steps": True}, {"num_of_trajectories": 0},
                   {"candidate_reset_interval": 0}, {"variance_warmup_steps": -1},
                   {"adaptive_eta": 0.}, {"adaptive_eta": True}, {"adaptive_eta": "0.1"},
                   {"variance_scale": float("nan")},
                   {"fabo_delta": 1.}, {"flow_loss_weight": -1.},
                   {"variance_loss_weight": -1.}, {"fabo_weight": -1.},
                   {"boundary_time_probability": 1.1},
                   {"log_variance_min": 3., "log_variance_max": 2.},
                   {"n_obs_steps": 0}, {"n_obs_steps": 1.5}, {"n_action_steps": 0},
                   {"horizon": True}, {"horizon": 6.5}, {"fabo_weight": True},
                   {"horizon": 2, "n_obs_steps": 2, "n_action_steps": 2},
                   {"variance_training": "head_only", "variance_loss_weight": 0.},
                   {"variance_hidden_dim": 0}, {"shape_meta": {"action": {"shape": [20]}}})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Policy(**kwargs)
        policy = Policy(sampler="fixed").eval()
        for shape in ((2, 5, 10), (2, 6, 9), (2, 6)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                policy.compute_loss({"obs": observations(), "action": torch.zeros(shape)})
        policy.predict_action_times_tlag = 1
        for history in (torch.zeros(2, 2), torch.zeros(1, 2, 10), torch.zeros(2, 2, 9),
                        torch.zeros(2, 0, 10)):
            with self.subTest(shape=history.shape), self.assertRaises(ValueError):
                policy.predict_action(observations(), history)

    def test_nonfinite_training_outputs_fail_loudly(self):
        for attr in ("bad_velocity", "bad_variance"):
            policy = Policy()
            setattr(policy.diff, attr, True)
            with self.subTest(attr=attr), self.assertRaises(FloatingPointError):
                policy.compute_loss(training_batch(policy))


HAS_ESCNN = importlib.util.find_spec("escnn") is not None


@unittest.skipUnless(HAS_ESCNN, "escnn is absent; real equivariant U-Net forward remains unverified")
class OptionalEquivariantBackboneIntegrationTests(unittest.TestCase):
    def test_enabling_head_preserves_velocity_and_returns_scalar_log_variance(self):
        with patch.object(sys, "path", [str(ROOT.parent)] + sys.path):
            module = importlib.import_module("EfficientFlow.model.equi.equi_conditional_unet1d")
        model = module.EquiDiffusionUNet(act_emb_dim=8, local_cond_dim=None,
            global_cond_dim=4, diffusion_step_embed_dim=16, down_dims=(16, 32),
            kernel_size=3, n_groups=8, cond_predict_scale=True, N=8).eval()
        action, times, condition = torch.randn(2, 8, 10), torch.tensor([0., 33.]), torch.randn(2, 32)
        with torch.no_grad():
            expected = model(action, times, global_cond=condition)
            with self.assertRaises(RuntimeError):
                model(action, times, global_cond=condition, return_variance=True)
            model.enable_variance_prediction(16)
            actual, log_variance = model(action, times, global_cond=condition, return_variance=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(log_variance.shape, (2,))
        torch.testing.assert_close(log_variance, torch.zeros(2), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
