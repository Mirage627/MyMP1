"""Check real-result accounting without constructing a model or simulator."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adaflow_eval_reports", ROOT / "scripts/efficientadaflow_experiment.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class ReportTests(unittest.TestCase):
    def metrics(self, rewards=(1., 0., 1.)):
        metrics = {
            "test/mean_score": sum(rewards) / len(rewards),
            "sampling/nfe_mean": 1.02, "sampling/one_step_fraction": .99,
            "sampling/nfe_p95": 1., "sampling/nfe_max": 3.,
            "timing/policy_ms_p50": 10., "timing/policy_ms_p95": 20.,
        }
        metrics.update({f"test/sim_max_reward_{200000 + i}": value for i, value in enumerate(rewards)})
        return metrics

    def row(self, metrics=None):
        return {
            "variant": {"sampler": "adaptive", "eta": .1}, "candidates": 5,
            "seed": 0, "test_start_seed": 200000, "n_test": 3, "status": "ok",
            "metrics": self.metrics() if metrics is None else metrics,
        }

    def test_incomplete_or_inconsistent_episode_results_are_rejected(self):
        metrics = self.metrics()
        cli._validate_rollout_metrics(metrics, 3)
        with self.assertRaisesRegex(RuntimeError, "Expected 4"):
            cli._validate_rollout_metrics(metrics, 4)
        wrong = dict(metrics, **{"test/mean_score": .5})
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            cli._validate_rollout_metrics(wrong, 3)
        del metrics["sampling/nfe_mean"]
        with self.assertRaisesRegex(RuntimeError, "required metrics"):
            cli._validate_rollout_metrics(metrics, 3)

    def test_binary_counts_and_interval_are_from_episode_outcomes(self):
        result = cli._summary_row(self.row())
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["failures"], 1)
        self.assertTrue(result["binary_rewards"])
        self.assertLess(result["episode_wilson95_low"], 2 / 3)
        self.assertGreater(result["episode_wilson95_high"], 2 / 3)
        self.assertEqual(result["nfe_mean"], 1.02)
        self.assertEqual(result["nfe_max"], 3.)
        shaped = cli._summary_row(self.row(self.metrics((.5, .5, 1.))))
        self.assertFalse(shaped["binary_rewards"])
        self.assertEqual(shaped["successes"], "")
        self.assertEqual(shaped["episode_wilson95_low"], "")

    def test_partial_reports_preserve_errors_without_claiming_success(self):
        good = self.row()
        failed = self.row()
        failed.update({"variant": {"sampler": "fixed", "fixed_steps": 8},
                       "status": "error", "metrics": {}, "error": "diagnostic failure"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cli._write_sweep_summary(output, [good, failed])
            with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["variant"], "adaptive_eta_0.1")
            self.assertEqual(rows[1]["status"], "error")
            self.assertEqual(rows[1]["successes"], "")
            text = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("2/3", text)
            self.assertIn("diagnostic failure", text)
            self.assertIn("not independent training seeds", text)


if __name__ == "__main__":
    unittest.main()
