"""Real CPU policy/workspace integration, without robot simulator or datasets."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if "EfficientFlow" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "EfficientFlow", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["EfficientFlow"] = package
    spec.loader.exec_module(package)

from EfficientFlow.policy.mp1 import Meanpolicy
from EfficientFlow.policy.mp1_ablation import MP1AblationPolicy
from EfficientFlow.model.common.normalizer import LinearNormalizer
from EfficientFlow.workspace.train_mp1_ablation_workspace import TrainMP1AblationWorkspace
from EfficientFlow.workspace import train_mp1_ablation_workspace as workspace_module


class SyntheticRolloutRunner:
    instances = []

    def __init__(self, output_dir, fail=False, **kwargs):
        self.output_dir, self.fail = output_dir, fail
        self.video_backend, self.closed = "wandb", False
        self.instances.append(self)

    def run(self, policy):
        from PIL import Image
        if self.fail:
            raise RuntimeError("synthetic rollout failure")
        path = Path(self.output_dir)/"rollout.gif"
        Image.new("RGB", (16, 16), "red").save(path)
        return {"test/mean_score": np.float64(.75), "test/sim_video_1": {
            "type": "video", "path": str(path), "fps": 10.}}

    def close(self):
        self.closed = True


def options():
    return dict(shape_meta=OmegaConf.create({"action": {"shape": [3]}, "obs": {
        "point_cloud": {"shape": [8, 3]}, "state": {"shape": [2]}}}),
        horizon=4, n_obs_steps=2, n_action_steps=3, diffusion_step_embed_dim=8,
        down_dims=[8, 16], kernel_size=3, n_groups=2, encoder_output_dim=8,
        pointnet_type="mlp", pointcloud_encoder_cfg=OmegaConf.create({
            "in_channels": 3, "out_channels": 8, "use_layernorm": True,
            "final_norm": "layernorm", "normal_channel": False}))


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(71)
        self.batch = {"action": torch.randn(6, 4, 3, generator=generator)*2 + 1,
                      "obs": {"point_cloud": torch.randn(6, 2, 8, 3, generator=generator)*3,
                              "state": torch.randn(6, 2, 2, generator=generator) + 2}}

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"action": self.batch["action"][index],
                "obs": {k: v[index] for k, v in self.batch["obs"].items()}}

    def get_validation_dataset(self):
        return self

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        normalizer.fit({"action": self.batch["action"], **self.batch["obs"]})
        return normalizer


def workspace_config():
    return OmegaConf.create({
        "policy": {"_target_": "EfficientFlow.policy.mp1_mgad_full.Policy",
                   "profile": "theory", "statistics_hidden_dim": 12, **options()},
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": .001},
        "head_optimizer": {"_target_": "torch.optim.AdamW", "lr": .002},
        "ema": {"_target_": "EfficientFlow.model.diffusion.ema_model.EMAModel",
                "update_after_step": 0, "inv_gamma": 1., "power": .75,
                "min_value": 0., "max_value": .9999},
        "task": {"dataset": {"_target_": __name__ + ".SyntheticDataset"}},
        "dataloader": {"batch_size": 2, "num_workers": 0, "shuffle": True},
        "val_dataloader": {"batch_size": 4, "num_workers": 0, "shuffle": False},
        "training": {"seed": 11, "use_ema": True, "max_grad_norm": 1.,
                     "device": "cpu", "gradient_accumulate_every": 2, "num_epochs": 2,
                     "max_train_steps": None, "max_val_steps": None,
                     "lr_scheduler": "cosine", "lr_warmup_steps": 0, "resume": False,
                     "enable_validation": True, "enable_rollouts": False,
                     "checkpoint_every": 1, "val_every": 1, "rollout_every": 1},
        "checkpoint": {"save_ckpt": True}})


class MP1IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.dataset = SyntheticDataset()
        self.batch = {"action": self.dataset.batch["action"][:3],
                      "obs": {k: v[:3] for k, v in self.dataset.batch["obs"].items()}}

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def make_policy(self, cls=MP1AblationPolicy, **kwargs):
        torch.manual_seed(19)
        policy = cls(**options(), **kwargs)
        policy.set_normalizer(self.dataset.get_normalizer())
        return policy

    @staticmethod
    def reset_rng():
        np.random.seed(23)
        torch.manual_seed(29)

    def compare_gradients(self, original, adapted, tolerance=2e-5):
        actual = dict(adapted.named_parameters())
        for name, expected in original.named_parameters():
            value = actual[name]
            if expected.grad is None:
                self.assertIsNone(value.grad, name)
            else:
                self.assertIsNotNone(value.grad, name)
                torch.testing.assert_close(value.grad, expected.grad, rtol=tolerance,
                                           atol=tolerance*0.01, msg=name)

    def test_original_mp1_loss_and_all_parameter_gradients(self):
        original = self.make_policy(Meanpolicy)
        adapted = self.make_policy(profile="mp1")
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[name], rtol=0, atol=0)
        self.reset_rng()
        expected, expected_logs = original.compute_loss(self.batch)
        expected.backward()
        self.reset_rng()
        actual, actual_logs = adapted.compute_loss(self.batch)
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        for key in ("meanflow_loss", "mse_val", "dis_loss"):
            self.assertAlmostEqual(actual_logs[key], expected_logs[key], places=6)
        self.compare_gradients(original, adapted)

    def test_render_cameras_do_not_change_normalized_state_or_inference(self):
        # Real MimicGen rollouts include cameras absent from get_normalizer().
        # Compare with field-wise normalized input, independently of dict order.
        for mgad in (False, True):
            for mode in ("none", "diag", "full"):
                with self.subTest(mgad=mgad, mode=mode):
                    policy = self.make_policy(profile="mp1", use_mgad=mgad,
                                              statistics_mode=mode, statistics_hidden_dim=12)
                    policy.eval()
                    observations = self.batch["obs"]
                    cameras = {"robot0_eye_in_hand_image": torch.rand(3, 2, 3, 8, 8),
                               "agentview_image": torch.rand(3, 2, 3, 8, 8)}
                    expected = {k: policy.normalizer.normalize({k: v})[k]
                                for k, v in observations.items()}
                    noise = torch.zeros(3, 4, 3)
                    reference = policy.predict_action(observations, initial_noise=noise)
                    for inputs in ({**cameras, **observations}, {**observations, **cameras}):
                        normalized = policy.normalizer.normalize(inputs)
                        for key, value in expected.items():
                            torch.testing.assert_close(normalized[key], value, rtol=0, atol=0)
                        for key, value in cameras.items():
                            self.assertIs(normalized[key], value)
                        captured = []
                        hook = policy.obs_encoder.register_forward_pre_hook(
                            lambda module, args: captured.append(args[0]))
                        try:
                            actual = policy.predict_action(inputs, initial_noise=noise)
                        finally:
                            hook.remove()
                        for key, value in expected.items():
                            torch.testing.assert_close(captured[0][key], value.reshape(-1, *value.shape[2:]),
                                                       rtol=0, atol=0)
                        torch.testing.assert_close(actual["action"], reference["action"], rtol=0, atol=0)
                        training_inputs = {"action": self.batch["action"], "obs": inputs}
                        _, condition = policy._training_inputs(training_inputs)
                        expected_condition = policy.obs_encoder({
                            k: v.reshape(-1, *v.shape[2:]) for k, v in expected.items()}).reshape(3, -1)
                        torch.testing.assert_close(condition, expected_condition, rtol=0, atol=0)

    def test_checkpoint_can_be_reevaluated_without_training_or_overwriting_logs(self):
        from test_mp1_swanlab import FakeSDK
        spec = importlib.util.spec_from_file_location("mp1_reeval", ROOT/"scripts"/"eval_mp1_checkpoint.py")
        evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation)
        with tempfile.TemporaryDirectory() as directory:
            cfg = workspace_config()
            cfg.training.num_epochs = 1
            cfg.task.env_runner = {"_target_": __name__+".SyntheticRolloutRunner",
                                   "n_train": 1, "n_train_vis": 1, "n_test": 0, "n_test_vis": 0}
            cfg.logging = {"mode": "disabled", "id": None, "name": "training"}
            training_dir, eval_dir = Path(directory)/"training", Path(directory)/"evaluation"
            workspace = TrainMP1AblationWorkspace(cfg, str(training_dir))
            workspace.run()
            before = (training_dir/"metrics.jsonl").read_bytes()
            sdk = FakeSDK()
            result = evaluation.evaluate_checkpoint(workspace.get_checkpoint_path(), eval_dir,
                                                     device="cpu", swanlab_module=sdk)
            self.assertEqual(result["state_key"], "ema_model")
            self.assertEqual(result["metrics"]["rollout/video_count"], 1)
            self.assertEqual(result["metrics"]["test/mean_score"], .75)
            self.assertEqual((training_dir/"metrics.jsonl").read_bytes(), before)
            self.assertTrue((eval_dir/"eval_results.json").is_file())
            self.assertEqual(len(sdk.videos), 1)
            self.assertEqual(sdk.finishes, 1)
            self.assertTrue(SyntheticRolloutRunner.instances[-1].closed)
            with self.assertRaisesRegex(ValueError, "separate evaluation output"):
                evaluation.evaluate_checkpoint(workspace.get_checkpoint_path(), training_dir, device="cpu")

    def test_rho_zero_matches_mse_and_keeps_head_gradient(self):
        baseline = self.make_policy(profile="theory")
        self.reset_rng()
        expected, _, _ = baseline.loss_terms(self.batch)
        expected.backward()
        for mode in ("diag", "full"):
            adapted = self.make_policy(profile="theory", statistics_mode=mode,
                                       metric_rho=0., statistics_hidden_dim=12)
            self.reset_rng()
            velocity, head, _ = adapted.loss_terms(self.batch)
            (velocity+head).backward()
            torch.testing.assert_close(velocity, expected, rtol=1e-6, atol=1e-7)
            self.compare_gradients(baseline, adapted)
            self.assertGreater(float(adapted.statistics_head.network[-1].weight.grad.abs().sum()), 0.)

    def test_six_variants_gradients_updates_and_one_nfe_inference(self):
        for mgad in (False, True):
            for mode in ("none", "diag", "full"):
                with self.subTest(mgad=mgad, mode=mode):
                    policy = self.make_policy(profile="mp1", use_mgad=mgad,
                                              statistics_mode=mode, statistics_hidden_dim=12)
                    theta, phi = policy.parameter_groups()
                    theta_opt = torch.optim.AdamW(theta, lr=.001)
                    phi_opt = torch.optim.AdamW(phi, lr=.001) if phi else None
                    velocity, head, _ = policy.loss_terms(self.batch)
                    velocity.backward(retain_graph=True)
                    self.assertTrue(all(p.grad is None for p in phi))
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.obs_encoder.parameters()
                                           if p.grad is not None), 0.)
                    theta_grads = [None if p.grad is None else p.grad.clone() for p in theta]
                    if phi:
                        theta_opt.zero_grad(set_to_none=True)
                        head.backward()
                        self.assertTrue(all(p.grad is None for p in theta))
                        self.assertGreater(sum(float(p.grad.abs().sum()) for p in phi if p.grad is not None), 0.)
                        for p, grad in zip(theta, theta_grads):
                            p.grad = grad
                    before_theta = [p.detach().clone() for p in theta]
                    before_phi = [p.detach().clone() for p in phi]
                    theta_opt.step()
                    if phi_opt:
                        phi_opt.step()
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(theta, before_theta)))
                    if phi:
                        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(phi, before_phi)))
                    if mgad:
                        self.assertGreater(float(policy.model.mgad.cross_attn.key.weight.grad.abs().sum()), 0.)
                    calls = []
                    handle = policy.model.register_forward_hook(lambda *args: calls.append(1))
                    guard = (patch.object(policy.statistics_head, "forward", side_effect=AssertionError("head in inference"))
                             if phi else contextlib.nullcontext())
                    with guard, torch.no_grad():
                        result = policy.predict_action(self.batch["obs"], initial_noise=torch.zeros(3, 4, 3))
                    handle.remove()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(result["action"].shape, (3, 3, 3))
                    self.assertTrue(torch.isfinite(result["action"]).all())

    def test_compute_loss_sum_matches_separate_gradients_with_anisotropic_head(self):
        from EfficientFlow.policy import mp1_ablation as policy_module

        for profile in ("theory", "mp1"):
            for mgad in (False, True):
                for mode in ("diag", "full"):
                    with self.subTest(profile=profile, mgad=mgad, mode=mode):
                        policy = self.make_policy(profile=profile, use_mgad=mgad,
                                                  statistics_mode=mode, statistics_hidden_dim=12)
                        # Exercise a trained, feature-dependent head rather than its
                        # isotropic initialization, where the metric is exactly I.
                        with torch.no_grad():
                            final = policy.statistics_head.network[-1]
                            final.weight.normal_(std=.02)
                            final.bias.add_(torch.linspace(-.2, .5, final.bias.numel()))
                        policy.sample_t_r = lambda size, device: (
                            torch.full((size,), .7, device=device),
                            torch.full((size,), .2, device=device))
                        theta, phi = policy.parameter_groups()
                        weights = []
                        bounded_metric = policy_module.bounded_metric

                        def capture_metric(*args, **kwargs):
                            weight = bounded_metric(*args, **kwargs)
                            weights.append(weight.detach().clone())
                            return weight

                        with patch.object(policy_module, "bounded_metric", side_effect=capture_metric):
                            self.reset_rng()
                            velocity, head, _ = policy.loss_terms(self.batch)
                            identity = torch.eye(weights[0].shape[-1]).expand_as(weights[0])
                            self.assertGreater(float((weights[0]-identity).abs().max()), 1e-4)

                            # The two graphs must support independent backward calls
                            # without retaining either graph or updating parameters.
                            velocity.backward()
                            self.assertTrue(all(p.grad is None for p in phi))
                            theta_grads = [None if p.grad is None else p.grad.clone() for p in theta]
                            self.assertGreater(sum(float(g.abs().sum()) for g in theta_grads
                                                   if g is not None), 0.)
                            policy.zero_grad(set_to_none=True)
                            head.backward()
                            self.assertTrue(all(p.grad is None for p in theta))
                            phi_grads = [None if p.grad is None else p.grad.clone() for p in phi]
                            self.assertGreater(sum(float(g.abs().sum()) for g in phi_grads
                                                   if g is not None), 0.)

                            policy.zero_grad(set_to_none=True)
                            self.reset_rng()
                            total, _ = policy.compute_loss(self.batch)
                            total.backward()

                        torch.testing.assert_close(total.detach(), velocity.detach()+head.detach(),
                                                   rtol=0, atol=0)
                        torch.testing.assert_close(weights[1], weights[0], rtol=0, atol=0)
                        for parameter, expected in zip(theta+phi, theta_grads+phi_grads):
                            if expected is None:
                                self.assertIsNone(parameter.grad)
                            else:
                                torch.testing.assert_close(parameter.grad, expected, rtol=1e-6, atol=1e-8)

    def test_rho_zero_workspace_updates_match_mse_with_separate_clipping(self):
        # Compare actual optimizer trajectories, not only a single backward.
        for profile in ("theory", "mp1"):
            for mgad in (False, True):
                for mode in ("diag", "full"):
                    with self.subTest(profile=profile, mgad=mgad, mode=mode), tempfile.TemporaryDirectory() as directory:
                        cfg = workspace_config()
                        cfg.training.use_ema = False
                        cfg.training.max_grad_norm = .01
                        cfg.policy = {"_target_": "EfficientFlow.policy.mp1_ablation.MP1AblationPolicy",
                                      "profile": profile, "baseline_loss": "mse", "use_mgad": mgad,
                                      "statistics_mode": "none", "metric_rho": 0.,
                                      "statistics_hidden_dim": 12, **options()}
                        baseline = TrainMP1AblationWorkspace(cfg, directory)
                        headed_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
                        headed_cfg.policy.statistics_mode = mode
                        headed = TrainMP1AblationWorkspace(headed_cfg, directory)
                        for workspace in (baseline, headed):
                            workspace.model.set_normalizer(self.dataset.get_normalizer())
                        phi_before = [p.detach().clone() for p in headed.model.statistics_head.parameters()]
                        original_clip = torch.nn.utils.clip_grad_norm_
                        clipped_groups = []

                        def capture_clip(parameters, *args, **kwargs):
                            parameters = list(parameters)
                            norm = original_clip(parameters, *args, **kwargs)
                            clipped_groups.append(({id(p) for p in parameters}, float(norm)))
                            return norm

                        for step in range(2):
                            np.random.seed(23 + step)
                            torch.manual_seed(29 + step)
                            baseline_logs = baseline._step_batch([self.batch])
                            np.random.seed(23 + step)
                            torch.manual_seed(29 + step)
                            clipped_groups.clear()
                            with patch("torch.nn.utils.clip_grad_norm_", side_effect=capture_clip):
                                headed_logs = headed._step_batch([self.batch])
                            theta, phi = headed.model.parameter_groups()
                            self.assertEqual([group for group, _ in clipped_groups],
                                             [{id(p) for p in theta}, {id(p) for p in phi}])
                            self.assertTrue(all(norm > cfg.training.max_grad_norm for _, norm in clipped_groups))
                            self.assertAlmostEqual(headed_logs["velocity_loss"], baseline_logs["velocity_loss"], places=6)
                            actual = dict(headed.model.named_parameters())
                            for name, expected in baseline.model.named_parameters():
                                torch.testing.assert_close(actual[name], expected, rtol=1e-6, atol=1e-7, msg=name)
                            self.compare_gradients(baseline.model, headed.model)
                        self.assertTrue(any(not torch.equal(old, new) for old, new in
                                            zip(phi_before, headed.model.statistics_head.parameters())))

    def test_affine_teacher_has_fixed_r_tangent_and_stopped_target(self):
        class Affine(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, sample, timestep, r, global_cond, training=True):
                value = self.alpha*sample + 2*timestep[:, None, None] + 3*r[:, None, None]
                return (value, sample.flatten(1).unsqueeze(0)) if training else value
        t, r = torch.tensor([.7, .6, .4]), torch.tensor([.2, .6, .1])
        noise = torch.full_like(self.batch["action"], .3)
        for profile, w in (("theory", 1.), ("mp1", 2.)):
            policy = self.make_policy(profile=profile, baseline_loss="mse")
            policy.model = Affine()
            policy.dispersive_weight = 0.
            policy.sample_t_r = lambda *args: (t, r)
            action = policy.normalizer["action"].normalize(self.batch["action"])
            x = (1-t[:, None, None])*action+t[:, None, None]*noise
            direction = w*(noise-action)+(1-w)*(.2*x+5*t[:, None, None])
            target = direction-(t-r)[:, None, None]*(.2*direction+2)
            error = .2*x+2*t[:, None, None]+3*r[:, None, None]-target
            with patch("torch.randn_like", return_value=noise):
                velocity, _, _ = policy.loss_terms(self.batch)
            torch.testing.assert_close(velocity, error.square().mean())
            velocity.backward()
            torch.testing.assert_close(policy.model.alpha.grad, (2*error*x).mean())

    def assert_nested_close(self, a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_nested_close(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for first, second in zip(a, b):
                self.assert_nested_close(first, second)
        else:
            self.assertEqual(a, b)

    def test_checkpoint_resume_matches_continuous_training_and_rejects_profile_change(self):
        class Interrupted(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            continuous_dir, interrupted_dir, relocated_dir = [str(Path(directory)/x) for x in ("continuous", "interrupted", "relocated")]
            cfg = workspace_config()
            continuous = TrainMP1AblationWorkspace(cfg, continuous_dir)
            continuous.run()
            interrupted = TrainMP1AblationWorkspace(cfg, interrupted_dir)
            real_save = interrupted.save_checkpoint
            def stop_after_save(*args, **kwargs):
                real_save(*args, **kwargs)
                raise Interrupted()
            with patch.object(interrupted, "save_checkpoint", new=stop_after_save):
                with self.assertRaises(Interrupted):
                    interrupted.run()
            resumed = TrainMP1AblationWorkspace(cfg, relocated_dir)
            checkpoint = interrupted.get_checkpoint_path()
            resumed.load_checkpoint(path=checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(resumed.output_dir, relocated_dir)
            self.assertIsNone(resumed.lr_scheduler)
            self.assertEqual(resumed.epoch, 1)
            restored = TrainMP1AblationWorkspace.create_from_checkpoint(checkpoint)
            self.assertEqual(restored.global_step, resumed.global_step)
            self.assert_nested_close(restored.model.state_dict(), resumed.model.state_dict())
            for parameter in resumed.model.normalizer.parameters():
                self.assertFalse(parameter.requires_grad)
            resumed.run()
            self.assertEqual(resumed.global_step, continuous.global_step)
            self.assertEqual(resumed.epoch, continuous.epoch)
            for name in ("model", "ema_model", "optimizer", "head_optimizer", "lr_scheduler", "head_lr_scheduler"):
                self.assert_nested_close(getattr(resumed, name).state_dict(), getattr(continuous, name).state_dict())
            bad = OmegaConf.create(OmegaConf.to_container(cfg))
            bad.policy.profile = "mp1"
            wrong_profile = TrainMP1AblationWorkspace(bad, relocated_dir)
            with self.assertRaisesRegex(ValueError, "Incompatible ablation checkpoint"):
                wrong_profile.load_checkpoint(path=checkpoint, map_location="cpu", weights_only=False)

    def test_rollout_media_logged_and_logger_finished_on_success_and_failure(self):
        from test_mp1_swanlab import FakeSDK
        from EfficientFlow.common.swanlab_logging import SwanLabLogger
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                sdk = FakeSDK()
                cfg = workspace_config()
                cfg.training.num_epochs = 1
                cfg.training.enable_rollouts = True
                cfg.task.env_runner = {"_target_": __name__+".SyntheticRolloutRunner", "fail": fail}
                cfg.logging = {"mode": "offline", "project": "integration"}
                cfg.checkpoint.save_ckpt = False
                workspace = TrainMP1AblationWorkspace(cfg, directory)
                factory = lambda *args, **kwargs: SwanLabLogger(*args, **kwargs, swanlab_module=sdk)
                with patch.object(workspace_module, "SwanLabLogger", side_effect=factory):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "synthetic rollout failure"):
                            workspace.run()
                    else:
                        workspace.run()
                        payload, _ = sdk.logs[0]
                        self.assertEqual(payload["test/mean_score"], .75)
                        self.assertIs(payload["test/sim_video_1"], sdk.videos[0])
                self.assertTrue(SyntheticRolloutRunner.instances[-1].closed)
                self.assertEqual(SyntheticRolloutRunner.instances[-1].video_backend, "path")
                self.assertEqual(sdk.finishes, 1)

    def test_complete_yaml_variants_build_and_update_expected_optimizer_groups(self):
        from hydra import compose, initialize_config_dir
        for name in ("MP1_baseline", "MP1_mgad", "MP1_diag", "MP1_full", "MP1_mgad_diag", "MP1_mgad_full"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                with initialize_config_dir(version_base=None, config_dir=str(ROOT/"config")):
                    cfg = compose(config_name=name)
                # Real modules with small channels; task/data are not constructed.
                for key, value in options().items():
                    cfg.policy[key] = value
                cfg.training.use_ema = False
                workspace = TrainMP1AblationWorkspace(cfg, directory)
                workspace.model.set_normalizer(self.dataset.get_normalizer())
                has_head = any(token in name for token in ("diag", "full"))
                self.assertEqual(workspace.head_optimizer is not None, has_head)
                logs = workspace._step_batch([self.batch])
                self.assertTrue(np.isfinite(logs["train_loss"]))
                self.assertEqual(workspace.model.use_mgad, "mgad" in name)


if __name__ == "__main__":
    unittest.main()
