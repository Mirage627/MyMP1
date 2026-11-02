"""Regression checks for selecting the image training workspace at the entry point."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_TARGET = "EfficientFlow.workspace.train_equi_workspace.TrainEquiWorkspace"
POLICY_TARGET = "EfficientFlow.policy.EfficientAdaFlow.EfficientAdaFlowPolicy"
HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None


@unittest.skipUnless(HYDRA_AVAILABLE, "Hydra is not installed")
class TrainingEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hydra
        from omegaconf import OmegaConf

        spec = importlib.util.spec_from_file_location(
            "adaflow_training_entry_under_test", ROOT / "common/train_entry.py")
        cls.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.entry)
        cls.entry.register_resolvers()
        cls.hydra = hydra
        cls.OmegaConf = OmegaConf

    def _cli(self, entry, *arguments, cwd=None):
        return subprocess.run(
            [sys.executable, str(entry), *arguments], cwd=cwd or ROOT,
            capture_output=True, text=True, timeout=45)

    def _copy_source_package(self, destination):
        destination.mkdir(parents=True)
        for relative in ("__init__.py", "train_efficientadaflow.py"):
            shutil.copy2(ROOT / relative, destination / relative)
        # The real YAML config and Python sources are needed for --verify,
        # but no dataset, simulator, checkpoint, or model construction is.
        for relative in ("common", "config", "policy", "model", "workspace", "dataset", "env_runner"):
            shutil.copytree(
                ROOT / relative, destination / relative,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    def _nested_server_layout(self, directory):
        outer = Path(directory) / "PROJECT_EfficientFlow"
        inner = outer / "EfficientFlow"
        self._copy_source_package(inner)
        shutil.copy2(ROOT / "train_efficientadaflow.py", outer / "train_efficientadaflow.py")
        (outer / "train.py").write_text(
            "raise RuntimeError('LEGACY_TRAIN_MUST_NOT_BE_IMPORTED')\n", encoding="utf-8")
        return outer, inner

    def test_dedicated_verify_checks_real_config_without_loading_the_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._cli(
                ROOT / "train_efficientadaflow.py", "--verify",
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20",
                "dataset_path=__missing_dataset_for_entry_verification__.hdf5",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(WORKSPACE_TARGET, result.stdout)
        self.assertIn(POLICY_TARGET, result.stdout)
        self.assertIn(str(ROOT), result.stdout)

    def test_dedicated_entry_composes_kitchen_configuration(self):
        result = self._cli(
            ROOT / "train_efficientadaflow.py", "--cfg", "job", "--resolve",
            "task_name=kitchen_d1", "n_demo=200", "training.seed=20")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(WORKSPACE_TARGET, result.stdout)
        self.assertIn(POLICY_TARGET, result.stdout)
        self.assertIn("num_epochs: 250", result.stdout)
        self.assertIn("seed: 20", result.stdout)

    def test_dedicated_entry_rejects_mp1_before_constructing_a_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self._cli(
                ROOT / "train_efficientadaflow.py", "--config-name=MP1", "--verify",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TrainMP1Workspace", result.stdout + result.stderr)
        self.assertNotIn("ENTRY_CHECK_OK", result.stdout)

    def test_dedicated_entry_does_not_import_a_legacy_train_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "PROJECT_EfficientFlow"
            self._copy_source_package(checkout)
            (checkout / "train.py").write_text(
                "raise RuntimeError('LEGACY_TRAIN_MUST_NOT_BE_IMPORTED')\n", encoding="utf-8")
            legacy_package = checkout / "EfficientFlow"
            legacy_package.mkdir()
            (legacy_package / "__init__.py").write_text(
                "raise RuntimeError('NESTED_PACKAGE_MUST_NOT_BE_IMPORTED')\n", encoding="utf-8")
            result = self._cli(
                checkout / "train_efficientadaflow.py", "--verify",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=directory)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(str(checkout), result.stdout)
        self.assertNotIn("LEGACY_TRAIN_MUST_NOT_BE_IMPORTED", result.stdout + result.stderr)
        self.assertNotIn("NESTED_PACKAGE_MUST_NOT_BE_IMPORTED", result.stdout + result.stderr)

    def test_outer_launcher_finds_inner_sources_without_outer_package_init(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, inner = self._nested_server_layout(directory)
            self.assertFalse((outer / "__init__.py").exists())
            result = self._cli(
                outer / "train_efficientadaflow.py", "--verify",
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20",
                "dataset_path=__missing_dataset_for_entry_verification__.hdf5",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=outer)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(WORKSPACE_TARGET, result.stdout)
        self.assertIn(POLICY_TARGET, result.stdout)
        self.assertIn(str(inner), result.stdout)
        self.assertNotIn("LEGACY_TRAIN_MUST_NOT_BE_IMPORTED", result.stdout + result.stderr)

    def test_inner_launcher_can_be_run_from_outer_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, inner = self._nested_server_layout(directory)
            result = self._cli(
                inner / "train_efficientadaflow.py", "--verify",
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=outer)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(str(inner), result.stdout)
        self.assertNotIn("LEGACY_TRAIN_MUST_NOT_BE_IMPORTED", result.stdout + result.stderr)

    def test_outer_empty_init_does_not_hide_inner_source_package(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, inner = self._nested_server_layout(directory)
            (outer / "__init__.py").write_text("", encoding="utf-8")
            result = self._cli(
                outer / "train_efficientadaflow.py", "--verify",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=outer)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(f"SOURCE_ROOT={inner}", result.stdout)

    def test_missing_source_package_fails_with_actionable_layout_message(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "PROJECT_EfficientFlow"
            outer.mkdir()
            shutil.copy2(ROOT / "train_efficientadaflow.py", outer / "train_efficientadaflow.py")
            result = self._cli(outer / "train_efficientadaflow.py", "--verify", cwd=outer)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Cannot locate the EfficientFlow source package", result.stdout + result.stderr)
        self.assertIn(str(outer / "EfficientFlow"), result.stdout + result.stderr)
        self.assertIn("config/, policy/ and workspace/", result.stdout + result.stderr)
        self.assertNotIn("FileNotFoundError", result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_missing_shared_entry_helper_reports_the_required_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory) / "PROJECT_EfficientFlow"
            inner = outer / "EfficientFlow"
            inner.mkdir(parents=True)
            (inner / "__init__.py").write_text("", encoding="utf-8")
            for name in ("config", "policy", "workspace"):
                (inner / name).mkdir()
            shutil.copy2(ROOT / "train_efficientadaflow.py", outer / "train_efficientadaflow.py")
            result = self._cli(outer / "train_efficientadaflow.py", "--verify", cwd=outer)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Upload common/train_entry.py", result.stdout + result.stderr)
        self.assertIn(str(inner / "common" / "train_entry.py"), result.stdout + result.stderr)
        self.assertNotIn("FileNotFoundError", result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stdout + result.stderr)

    def test_general_outer_launcher_verifies_adaflow_without_outer_package_init(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, inner = self._nested_server_layout(directory)
            shutil.copy2(ROOT / "train.py", outer / "train.py")
            self.assertFalse((outer / "__init__.py").exists())
            result = self._cli(
                outer / "train.py", "--config-name=EfficientAdaFlow", "--verify",
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20",
                "dataset_path=__missing_dataset_for_entry_verification__.hdf5",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=outer)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(f"ENTRY_FILE={outer / 'train.py'}", result.stdout)
        self.assertIn(f"SOURCE_ROOT={inner}", result.stdout)
        self.assertIn(f"WORKSPACE_TARGET={WORKSPACE_TARGET}", result.stdout)
        self.assertIn(f"POLICY_TARGET={POLICY_TARGET}", result.stdout)

    def test_general_outer_launcher_runs_the_image_workspace_for_adaflow(self):
        # Exercise the normal CLI dispatch, not --verify. A small image workspace
        # stands in for GPU/simulator construction; importing the MP1 workspace
        # is forbidden. This reproduces the server's outer-entry/inner-source
        # layout without needing its dataset or robot dependencies.
        with tempfile.TemporaryDirectory() as directory:
            outer, inner = self._nested_server_layout(directory)
            shutil.copy2(ROOT / "train.py", outer / "train.py")
            (inner / "workspace/train_equi_workspace.py").write_text(
                "class TrainEquiWorkspace:\n"
                "    def __init__(self, cfg):\n"
                f"        assert cfg._target_ == {WORKSPACE_TARGET!r}\n"
                f"        assert cfg.policy._target_ == {POLICY_TARGET!r}\n"
                "        assert cfg.task_name == 'kitchen_d1'\n"
                "        assert cfg.n_demo == 200\n"
                "        assert cfg.training.seed == 20\n"
                "        assert cfg.dataset_path == '__missing_dataset__.hdf5'\n"
                "    def run(self):\n"
                "        print('SENTINEL_IMAGE_WORKSPACE_RUN', flush=True)\n",
                encoding="utf-8")
            (inner / "workspace/train_mp1_workspace.py").write_text(
                "raise RuntimeError('MP1_WORKSPACE_MUST_NOT_BE_IMPORTED')\n", encoding="utf-8")
            result = self._cli(
                outer / "train.py", "--config-name=EfficientAdaFlow",
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20",
                "dataset_path=__missing_dataset__.hdf5",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=outer)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"SOURCE_ROOT={inner}", result.stdout)
        self.assertIn(f"WORKSPACE_TARGET={WORKSPACE_TARGET}", result.stdout)
        self.assertIn(f"WORKSPACE_FILE={inner / 'workspace/train_equi_workspace.py'}", result.stdout)
        self.assertIn("SENTINEL_IMAGE_WORKSPACE_RUN", result.stdout)
        self.assertNotIn("MP1_WORKSPACE_MUST_NOT_BE_IMPORTED", result.stdout + result.stderr)
        self.assertNotIn("ENTRY_CHECK_OK", result.stdout)

    def test_general_outer_launcher_still_verifies_mp1_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            outer, inner = self._nested_server_layout(directory)
            shutil.copy2(ROOT / "train.py", outer / "train.py")
            (inner / "workspace/train_mp1_workspace.py").write_text(
                "raise RuntimeError('VERIFY_MUST_NOT_IMPORT_ROBOTICS')\n", encoding="utf-8")
            result = self._cli(
                outer / "train.py", "--config-name=MP1", "--verify",
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20",
                f"hydra.run.dir='{Path(directory).as_posix()}/run'", cwd=outer)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ENTRY_CHECK_OK", result.stdout)
        self.assertIn(f"SOURCE_ROOT={inner}", result.stdout)
        self.assertIn(
            "WORKSPACE_TARGET=EfficientFlow.workspace.train_mp1_workspace.TrainMP1Workspace",
            result.stdout)
        self.assertIn("POLICY_TARGET=EfficientFlow.policy.mp1.Meanpolicy", result.stdout)
        self.assertNotIn("VERIFY_MUST_NOT_IMPORT_ROBOTICS", result.stdout + result.stderr)

    def test_dispatch_uses_the_composed_workspace_target(self):
        with self.hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
            cfg = self.hydra.compose(config_name="EfficientAdaFlow", overrides=[
                "task_name=kitchen_d1", "n_demo=200", "training.seed=20"])

        events = []

        class SelectedWorkspace:
            def __init__(self, actual_cfg):
                events.append(("construct", actual_cfg))

            def run(self):
                events.append(("run", None))

        with mock.patch.object(self.entry.hydra.utils, "get_class", return_value=SelectedWorkspace) as get_class:
            with contextlib.redirect_stdout(io.StringIO()):
                self.entry.run_workspace(
                    cfg, source_root=ROOT, entry_file=ROOT / "train_efficientadaflow.py",
                    expected_workspace=WORKSPACE_TARGET, expected_policy=POLICY_TARGET)
        get_class.assert_called_once_with(WORKSPACE_TARGET)
        self.assertEqual([event[0] for event in events], ["construct", "run"])
        self.assertIs(events[0][1], cfg)

    def test_general_entry_rejects_an_adaflow_policy_with_mp1_workspace(self):
        with self.hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
            cfg = self.hydra.compose(config_name="EfficientAdaFlow", overrides=[
                "_target_=EfficientFlow.workspace.train_mp1_workspace.TrainMP1Workspace"])
        with mock.patch.object(self.entry.hydra.utils, "get_class") as get_class:
            with self.assertRaisesRegex(ValueError, "Wrong workspace.*TrainMP1Workspace"):
                self.entry.run_workspace(cfg, source_root=ROOT, entry_file=ROOT / "train.py")
        get_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
