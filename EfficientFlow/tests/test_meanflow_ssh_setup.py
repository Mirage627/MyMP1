import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("omegaconf"), "OmegaConf unavailable")
class CheckpointLookupTests(unittest.TestCase):
    def test_lookup_matches_task_seed_demo_and_policy(self):
        from experiments.meanflow_eval.checkpoints import find_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, seed, policy in (("valid", 20, "mp1.Meanpolicy"),
                    ("wrongseed", 21, "mp1.Meanpolicy"), ("wrongpolicy", 20, "EfficientFlow.EfficientFlowPolicy")):
                run = root/name
                (run/".hydra").mkdir(parents=True)
                (run/"checkpoints").mkdir()
                (run/".hydra/config.yaml").write_text(
                    f"task_name: kitchen_d1\nn_demo: 200\ntraining:\n  seed: {seed}\n"
                    f"policy:\n  _target_: EfficientFlow.policy.{policy}\n", encoding="utf-8")
                (run/"checkpoints/latest.ckpt").write_bytes(b"fixture")
            self.assertEqual(find_checkpoint(root, task="kitchen_d1", training_seed=20, n_demo=200),
                             root/"valid/checkpoints/latest.ckpt")

    def test_missing_checkpoint_has_save_flag_guidance(self):
        from experiments.meanflow_eval.checkpoints import find_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint.save_ckpt=True"):
                find_checkpoint(directory, task="kitchen_d1", training_seed=20, n_demo=200)


class RenamedCheckoutTests(unittest.TestCase):
    def test_entry_point_in_arbitrary_directory(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"my_server_MP1_checkout"
            (root/"scripts").mkdir(parents=True)
            shutil.copy2(source/"__init__.py", root/"__init__.py")
            shutil.copy2(source/"scripts/meanflow_experiment.py", root/"scripts/meanflow_experiment.py")
            shutil.copytree(source/"experiments", root/"experiments", ignore=shutil.ignore_patterns("__pycache__"))
            result = subprocess.run([sys.executable, str(root/"scripts/meanflow_experiment.py"), "doctor"],
                cwd=directory, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(Path(json.loads(result.stdout)["checkout"]), root)

    def test_entry_point_always_selects_root_even_with_nested_package_and_env_override(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"PROJECT_EfficientFlow"
            (root/"scripts").mkdir(parents=True)
            (root/"EfficientFlow").mkdir()
            (root/"EfficientFlow/__init__.py").write_text("# training package\n", encoding="utf-8")
            shutil.copy2(source/"__init__.py", root/"__init__.py")
            shutil.copy2(source/"scripts/meanflow_experiment.py", root/"scripts/meanflow_experiment.py")
            shutil.copytree(source/"experiments", root/"experiments", ignore=shutil.ignore_patterns("__pycache__"))
            for choice in ("auto", "nested", "root", "invalid"):
                env = dict(os.environ, MEANFLOW_SOURCE_TREE=choice)
                result = subprocess.run([sys.executable, str(root/"scripts/meanflow_experiment.py"), "doctor"],
                    cwd=directory, env=env, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(Path(json.loads(result.stdout)["source_tree"]), root)


class BundleInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle, self.project = self.root/"bundle", self.root/"server"
        self.bundle.mkdir()
        (self.bundle/"overlay").mkdir()
        (self.project/"policy").mkdir(parents=True)
        (self.project/"config").mkdir()
        self.policy_bytes = {
            "mp1.py": b"# Different server revision\r\nclass ServerOnlyMeanPolicy: pass\r\n",
            "EquiMeanFlow.py": b"# Server equi policy, unrelated version\n",
            "custom_policy.py": b"\x00\xffserver policy bytes\r\n",
        }
        for name, content in self.policy_bytes.items():
            (self.project/"policy"/name).write_bytes(content)
        (self.project/"config/MP1.yaml").write_text("server: untouched\n", encoding="utf-8")
        source = Path(__file__).resolve().parents[1]/"scripts/install_meanflow_bundle.py"
        shutil.copy2(source, self.bundle/"install.py")

    def overlay(self, files):
        for name, content in files.items():
            target = self.bundle/"overlay"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (self.bundle/"bundle_manifest.json").write_text(
            json.dumps({"overlay_files": list(files)}), encoding="utf-8")

    def install(self, *, no_git=False):
        env = os.environ.copy()
        if no_git:
            env["PATH"] = ""
        return subprocess.run([sys.executable, str(self.bundle/"install.py"),
                               "--project", str(self.project)], cwd=self.root,
                              env=env, capture_output=True, text=True, encoding="utf-8")

    def assertPoliciesUnchanged(self):
        self.assertEqual(sorted(path.name for path in (self.project/"policy").iterdir()),
                         sorted(self.policy_bytes))
        for name, content in self.policy_bytes.items():
            self.assertEqual((self.project/"policy"/name).read_bytes(), content)

    def project_bytes(self):
        return {path.relative_to(self.project).as_posix(): path.read_bytes()
                for path in self.project.rglob("*") if path.is_file()}

    def test_different_server_sources_install_without_git_or_patch(self):
        files = {"experiments/__init__.py": b"",
                 "experiments/meanflow_eval/backends.py": b"# independent backend\n",
                 "experiments/meanflow_eval/README.zh-CN.md": b"documentation\n",
                 "common/meanflow_sampling.py": b"# independent sampler\n",
                 "scripts/meanflow_experiment.py": b"# experiment entry point\n",
                 "scripts/install_meanflow_bundle.py": b"# independent installer support\n",
                 "tests/test_meanflow_install.py": b"# independent test\n"}
        self.overlay(files)
        # There is deliberately no .git directory and no policy patch file.
        result = self.install(no_git=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        for name, content in files.items():
            self.assertEqual((self.project/name).read_bytes(), content)
        self.assertPoliciesUnchanged()
        self.assertEqual((self.project/"config/MP1.yaml").read_text(encoding="utf-8"),
                         "server: untouched\n")

    def test_root_policy_replacement_ignores_context_and_preserves_recoverable_crlf_backup(self):
        policy_name = "policy/mp1.py"
        replacement = b"# Bundled root MP1 with multistep inference\nclass Meanpolicy: pass\n"
        self.overlay({policy_name: replacement,
                      "experiments/meanflow_eval/core.py": b"# experiment code\n"})
        # The server's different source and CRLF bytes require no git context.
        result = self.install(no_git=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual((self.project/policy_name).read_bytes(), replacement)
        self.assertIn("Updated root MP1 policy", result.stdout)
        self.assertIn(str(self.project/policy_name), result.stdout)
        self.assertIn("Other policy files", result.stdout)
        backups = list((self.project/"data/meanflow_install_backup").glob("*/"+policy_name))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), self.policy_bytes["mp1.py"])
        for name, content in self.policy_bytes.items():
            if name != "mp1.py":
                self.assertEqual((self.project/"policy"/name).read_bytes(), content)
        shutil.copy2(backups[0], self.project/policy_name)
        self.assertPoliciesUnchanged()

    def test_disallowed_overlay_paths_are_rejected_before_any_copy(self):
        allowed = "experiments/meanflow_eval/core.py"
        for forbidden in ("train.py", "policy/custom_policy.py", "policy/EquiMeanFlow.py",
                          "config/MP1.yaml", "common/other.py", "experiments/other.py",
                          "experiments/meanflow_eval/payload.json", "scripts/other.py",
                          "tests/unrelated.py"):
            with self.subTest(forbidden=forbidden):
                self.overlay({allowed: b"allowed file must not be copied\n",
                              forbidden: b"forbidden overwrite\n"})
                before = self.project_bytes()
                result = self.install()
                self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
                self.assertIn("not allowed", result.stderr)
                self.assertEqual(self.project_bytes(), before)
                self.assertFalse((self.project/"experiments").exists())
                self.assertFalse((self.project/"data").exists())
                self.assertPoliciesUnchanged()

    def test_traversal_and_absolute_manifest_paths_are_rejected(self):
        for name in ("../policy/mp1.py", "/policy/mp1.py", "C:/policy/mp1.py",
                     "C:policy/mp1.py", "experiments/meanflow_eval/../escape.py",
                     "experiments\\meanflow_eval\\core.py",
                     "experiments/meanflow_eval/core.py:stream", "./experiments/__init__.py"):
            with self.subTest(path=name):
                self.overlay({"experiments/meanflow_eval/core.py": b"valid\n"})
                (self.bundle/"bundle_manifest.json").write_text(json.dumps({
                    "overlay_files": ["experiments/meanflow_eval/core.py", name]}), encoding="utf-8")
                before = self.project_bytes()
                result = self.install()
                self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
                self.assertEqual(self.project_bytes(), before)
                self.assertFalse((self.project/"experiments").exists())

    def test_missing_late_source_prevents_all_copies(self):
        self.overlay({"experiments/meanflow_eval/core.py": b"valid\n"})
        (self.bundle/"bundle_manifest.json").write_text(json.dumps({"overlay_files": [
            "experiments/meanflow_eval/core.py", "experiments/meanflow_eval/missing.py"]}), encoding="utf-8")
        before = self.project_bytes()
        result = self.install()
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn("Missing bundle file", result.stderr)
        self.assertEqual(self.project_bytes(), before)
        self.assertFalse((self.project/"data").exists())

    def test_reinstallation_backs_up_each_replaced_experiment_version(self):
        name = "experiments/meanflow_eval/core.py"
        target = self.project/name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"original server experiment\r\n")
        self.overlay({name: b"bundle version one\n"})
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stdout+first.stderr)
        self.assertEqual(target.read_bytes(), b"bundle version one\n")
        self.overlay({name: b"bundle version two\n"})
        second = self.install()
        self.assertEqual(second.returncode, 0, second.stdout+second.stderr)
        self.assertEqual(target.read_bytes(), b"bundle version two\n")
        backups = list((self.project/"data/meanflow_install_backup").glob("*/"+name))
        self.assertEqual(len(backups), 2)
        self.assertEqual({path.read_bytes() for path in backups},
                         {b"original server experiment\r\n", b"bundle version one\n"})
        self.assertPoliciesUnchanged()

    def test_missing_project_marker_refuses_install(self):
        self.overlay({"experiments/meanflow_eval/core.py": b"valid\n"})
        (self.project/"config/MP1.yaml").unlink()
        result = self.install()
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn("Not the expected MP1 checkout", result.stderr)
        self.assertFalse((self.project/"experiments").exists())


if __name__ == "__main__":
    unittest.main()
