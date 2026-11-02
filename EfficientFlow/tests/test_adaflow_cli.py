"""Pure CLI/provenance checks; no simulator or checkpoint is synthesized."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adaflow_experiment_cli", ROOT / "scripts/efficientadaflow_experiment.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class ExperimentCliTests(unittest.TestCase):
    def test_help_does_not_require_robotics_imports(self):
        for suffix in ([], ["doctor"], ["sweep"], ["calibrate"]):
            result = subprocess.run([sys.executable, str(ROOT / "scripts/efficientadaflow_experiment.py"),
                                     *suffix, "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--help", result.stdout)

    def test_calibration_recovers_scale_without_empty_tie_bins(self):
        records = [{"predicted_variance": q, "residual_mse": 3 * q} for q in [1., 1., 2., 2., 4., 4.]]
        result = cli._calibration_summary(records, np, 10)
        self.assertEqual(result["overall"]["ratio_of_means"], 3.)
        self.assertEqual(result["overall"]["mean_residual_to_prediction_ratio"], 3.)
        self.assertEqual(sum(row["count"] for row in result["quantile_bins"]), len(records))
        plateau = cli._calibration_summary(records[:2], np, 10)
        self.assertEqual(len(plateau["quantile_bins"]), 1)
        self.assertEqual(plateau["quantile_bins"][0]["count"], 2)
        with self.assertRaises(ValueError):
            cli._calibration_summary([], np, 10)

    def test_reports_cannot_silently_replace_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "results.jsonl"
            report.write_text("prior run\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                cli._output_directory(root, False)
            self.assertEqual(report.read_text(encoding="utf-8"), "prior run\n")
            self.assertEqual(cli._output_directory(root, True), root.resolve())
            # Explicit overwrite permits report writing, but does not delete a directory.
            self.assertTrue(report.exists())

    def test_saved_numeric_resolvers_are_restricted_to_arithmetic(self):
        self.assertEqual(cli._arithmetic("max(1, int(50000 / 100))"), 500)
        self.assertEqual(cli._arithmetic("2**3 + abs(-2)"), 10)
        for expression in ("__import__('os')", "(1).__class__", "[1,2]", "open('x')"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                cli._arithmetic(expression)

    def test_nonfinite_runner_metrics_are_not_written_as_results(self):
        result, skipped = cli._numeric_metrics({"test/mean_score": .5, "video": object()})
        self.assertEqual(result, {"test/mean_score": .5})
        self.assertEqual(skipped, ["video"])
        with self.assertRaises(FloatingPointError):
            cli._numeric_metrics({"test/mean_score": float("nan")})


if __name__ == "__main__":
    unittest.main()
