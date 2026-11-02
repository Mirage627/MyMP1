"""Standard-library checks for the server launcher; no tensor/simulator imports."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "run_kitchen_adaflow_ablation.py"
spec = importlib.util.spec_from_file_location("adaflow_ablation_launcher", LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source_fixture(self, nested=False):
        source = self.root / "EfficientFlow" if nested else self.root
        for name in launcher.REQUIRED_SOURCE:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture\n", encoding="utf-8")
        evaluator = source / "scripts/efficientadaflow_experiment.py"
        evaluator.parent.mkdir(parents=True, exist_ok=True)
        evaluator.write_text("# fixture\n", encoding="utf-8")
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "latest.ckpt").write_bytes(b"fixture")
        return source, evaluator, checkpoints

    def args(self, *extra):
        return launcher.build_parser().parse_args([
            "--project-root", str(self.root), "--checkpoint", "checkpoints", *extra])

    def test_prefers_latest_and_not_highest_score(self):
        folder = self.root / "checkpoints"
        folder.mkdir()
        for name in ("latest.ckpt", "epoch=100-test_mean_score=0.999.ckpt"):
            (folder / name).write_bytes(b"fixture")
        self.assertEqual(launcher.select_checkpoint(folder)[0].name, "latest.ckpt")
        (folder / "latest.ckpt").unlink()
        (folder / "epoch=101-test_mean_score=0.100.ckpt").write_bytes(b"fixture")
        self.assertEqual(launcher.select_checkpoint(folder)[0].name,
                         "epoch=101-test_mean_score=0.100.ckpt")

    def test_tied_latest_epoch_requires_explicit_file(self):
        for name in ("epoch=10-test_mean_score=0.1.ckpt", "epoch=10-test_mean_score=0.9.ckpt"):
            (self.root / name).write_bytes(b"fixture")
        with self.assertRaisesRegex(ValueError, "Multiple checkpoints at highest filename epoch"):
            launcher.select_checkpoint(self.root)
        explicit = self.root / "epoch=10-test_mean_score=0.1.ckpt"
        self.assertEqual(launcher.select_checkpoint(explicit)[0], explicit.resolve())

    def test_unknown_names_are_not_selected_by_mtime(self):
        for name in ("best.ckpt", "last.ckpt"):
            (self.root / name).write_bytes(b"fixture")
        with self.assertRaisesRegex(ValueError, "explicit file"):
            launcher.select_checkpoint(self.root)

    def test_flat_and_nested_source_layouts(self):
        source, evaluator, _ = self.source_fixture(nested=True)
        plan = launcher.make_plan(self.args())
        self.assertEqual(plan["source_root"], source)
        self.assertEqual(plan["evaluator"], evaluator)
        self.assertEqual(plan["episodes"], 4)
        self.assertEqual(plan["doctor"][3], "doctor")
        self.assertNotIn("--output", plan["doctor"])
        self.assertNotIn("--dataset-path", plan["sweep"])
        self.assertIn("200000", plan["sweep"])

    def test_pilot_and_custom_grid_budget(self):
        self.source_fixture()
        pilot = launcher.make_plan(self.args("--mode", "pilot"))
        self.assertEqual(pilot["episodes"], 400)
        self.assertEqual(pilot["grid"]["fixed_steps"], [1, 2, 4, 8])
        custom = launcher.make_plan(self.args("--mode", "pilot", "--n-test", "7",
                                              "--seeds", "0", "2", "--candidates", "1", "5"))
        self.assertEqual(custom["episodes"], 8 * 7 * 2 * 2)
        self.assertNotEqual(pilot["output"], custom["output"])

    def test_inner_evaluator_wins_over_stale_outer_entry(self):
        source, evaluator, _ = self.source_fixture(nested=True)
        stale_outer = self.root / "scripts/efficientadaflow_experiment.py"
        stale_outer.parent.mkdir()
        stale_outer.write_text("# obsolete entry without doctor\n", encoding="utf-8")
        plan = launcher.make_plan(self.args())
        self.assertEqual(plan["source_root"], source)
        self.assertEqual(plan["evaluator"], evaluator)

    def test_outer_entry_can_use_inner_source_when_inner_entry_missing(self):
        source, evaluator, _ = self.source_fixture(nested=True)
        evaluator.unlink()
        outer_entry = self.root / "scripts/efficientadaflow_experiment.py"
        outer_entry.parent.mkdir()
        outer_entry.write_text("# independent entry\n", encoding="utf-8")
        plan = launcher.make_plan(self.args())
        self.assertEqual(plan["source_root"], source)
        self.assertEqual(plan["evaluator"], outer_entry)

    def test_dry_run_uses_only_standard_library_and_writes_nothing(self):
        self.source_fixture()
        before = set(self.root.rglob("*"))
        result = subprocess.run([
            sys.executable, "-S", str(LAUNCHER), "--project-root", str(self.root),
            "--checkpoint", "checkpoints", "--mode", "pilot", "--dry-run"],
            cwd=str(self.root), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("= 400 episodes", result.stdout)
        self.assertIn("no checkpoint loaded", result.stdout)
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_doctor_then_sweep_use_project_cwd_and_egl(self):
        _, evaluator, _ = self.source_fixture()
        evaluator.write_text(
            "import json, os, pathlib, sys\n"
            "with pathlib.Path('calls.jsonl').open('a') as stream:\n"
            "    stream.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
            "'mujoco': os.environ.get('MUJOCO_GL'), "
            "'pyopengl': os.environ.get('PYOPENGL_PLATFORM')}) + '\\n')\n",
            encoding="utf-8")
        clean_environment = {key: value for key, value in os.environ.items()
                             if key not in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
        with patch.dict(os.environ, clean_environment, clear=True), \
                contextlib.redirect_stdout(io.StringIO()):
            result = launcher.main([
                "--project-root", str(self.root), "--checkpoint", "checkpoints"])
        self.assertEqual(result, 0)
        calls = [json.loads(row) for row in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual([row["argv"][0] for row in calls], ["doctor", "sweep"])
        self.assertTrue(all(Path(row["cwd"]) == self.root for row in calls))
        self.assertTrue(all(row["mujoco"] == "egl" and row["pyopengl"] == "egl" for row in calls))

    def test_doctor_failure_stops_before_rollouts(self):
        _, evaluator, _ = self.source_fixture()
        evaluator.write_text(
            "import pathlib, sys\n"
            "pathlib.Path('called_' + sys.argv[1]).write_text('called')\n"
            "raise SystemExit(9)\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = launcher.main([
                "--project-root", str(self.root), "--checkpoint", "checkpoints"])
        self.assertEqual(result, 9)
        self.assertTrue((self.root / "called_doctor").exists())
        self.assertFalse((self.root / "called_sweep").exists())


if __name__ == "__main__":
    unittest.main()
