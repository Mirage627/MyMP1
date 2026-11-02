"""Hydra composition contracts for the six complete MP1 experiment YAMLs."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mp1_config_entry", ROOT/"train_mp1_ablation.py")
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)
VARIANTS = tuple(ENTRY.ABLATIONS)


def configuration(name, overrides=(), config_dir=None):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir or ROOT/"config")):
        return compose(config_name=name, overrides=list(overrides))


class MP1ConfigurationTests(unittest.TestCase):
    def test_six_standalone_files_need_only_the_task_group(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir/"task").mkdir()
            shutil.copy2(ROOT/"config"/"task"/"mimicgen_pc_abs.yaml", config_dir/"task")
            for name in VARIANTS:
                shutil.copy2(ROOT/"config"/(name+".yaml"), config_dir)
            for name in VARIANTS:
                with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                    cfg = configuration(name, config_dir=config_dir)
                    ENTRY._validate_ablation_configuration(cfg)
                    self.assertEqual(cfg._target_, ENTRY.MP1_ABLATION_WORKSPACE)
                    self.assertEqual(cfg.policy._target_, ENTRY.ABLATIONS[name])
                    self.assertEqual(cfg.training.num_epochs, 250)
                    self.assertEqual(cfg.training.rollout_every, 5)
                    self.assertEqual(cfg.training.checkpoint_every, 5)
                    self.assertEqual(cfg.task_name, "kitchen_d1")
                    self.assertEqual(cfg.training.seed, 20)
                    self.assertEqual(cfg.task.shape_meta.action.shape, [10])
                    self.assertEqual(cfg.policy.down_dims, [512, 1024, 2048])
                    self.assertEqual(cfg.policy.horizon, 4)
                    self.assertEqual(cfg.policy.n_obs_steps, 2)
                    self.assertEqual(cfg.policy.n_action_steps, 4)

    def test_demo_budget_and_both_profiles_resolve_for_every_variant(self):
        for name in VARIANTS:
            for demos, epochs, interval in ((100, 500, 10), (200, 250, 5), (500, 100, 2)):
                cfg = configuration(name, (f"n_demo={demos}", "profile=theory"))
                OmegaConf.resolve(cfg)
                self.assertIs(type(cfg.training.num_epochs), int)
                self.assertEqual((cfg.training.num_epochs, cfg.training.rollout_every,
                                  cfg.training.checkpoint_every), (epochs, interval, interval))
                self.assertEqual(cfg.policy.profile, "theory")

    def test_single_dataset_override_reaches_training_and_rollout(self):
        for name in VARIANTS:
            cfg = configuration(name, ("dataset_path=/shared/robot/kitchen_point_cloud.hdf5",))
            OmegaConf.resolve(cfg)
            self.assertEqual(cfg.task.dataset.dataset_path, "/shared/robot/kitchen_point_cloud.hdf5")
            self.assertEqual(cfg.task.env_runner.dataset_path, cfg.task.dataset.dataset_path)

    def test_common_backbone_and_optimizer_budget_match_across_variants(self):
        shared = None
        for name in VARIANTS:
            cfg = configuration(name)
            OmegaConf.resolve(cfg)
            head = any(token in name for token in ("diag", "full"))
            mgad = "mgad" in name
            self.assertEqual("head_optimizer" in cfg, head)
            self.assertEqual("statistics_hidden_dim" in cfg.policy, head)
            self.assertEqual("mgad_num_heads" in cfg.policy, mgad)
            policy = OmegaConf.to_container(cfg.policy, resolve=True)
            for key in ("_target_", "baseline_loss", "metric_rho", "metric_ridge", "metric_endpoint_margin",
                        "statistics_hidden_dim", "statistics_min_variance", "statistics_initial_variance",
                        "mgad_num_heads", "mgad_mlp_ratio"):
                policy.pop(key, None)
            values = {key: OmegaConf.to_container(cfg[key], resolve=True)
                      for key in ("task", "training", "optimizer", "ema", "dataloader", "val_dataloader")}
            values["policy"] = policy
            if shared is None:
                shared = values
            self.assertEqual(values, shared)
            self.assertEqual(set(cfg.checkpoint), {"save_ckpt"})
            self.assertTrue(cfg.checkpoint.save_ckpt)
            self.assertNotIn("debug", cfg.training)
            self.assertNotIn("topk", cfg.checkpoint)
            self.assertNotIn("use_point_crop", cfg.policy)

    def test_incompatible_or_zero_schedule_fails_before_training(self):
        for overrides in (("n_demo=2000",), ("training.rollout_every=0",),
                          ("task.env_runner.dataset_path=/different/data.hdf5",)):
            cfg = configuration("MP1_baseline", overrides)
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                ENTRY._validate_ablation_configuration(cfg)


if __name__ == "__main__":
    unittest.main()
