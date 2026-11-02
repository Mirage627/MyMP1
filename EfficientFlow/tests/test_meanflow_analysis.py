import json
import itertools
import tempfile
import unittest
from pathlib import Path

from experiments.meanflow_eval.analysis import summarize, write_report


def branch_rows(base, two, four=None, clusters=None, signals=None):
    four = two if four is None else four
    clusters = clusters or [f"c{i}" for i in range(len(base))]
    result = []
    for k, outcomes in ((1, base), (2, two), (4, four)):
        for i, success in enumerate(outcomes):
            result.append({"state_id": f"s{i}", "cluster_id": clusters[i],
                "variant": f"uniform_{k}", "nfe": k, "success": bool(success),
                "reward": float(success), "latency_ms": float(k),
                "signals": signals[i] if signals is not None else {"composition_defect": float(i)}})
    return result


def fixed_row(episode, cluster, variant, success, nfe=1):
    return {"episode_id": episode, "cluster_id": cluster, "variant": variant,
            "nfe": nfe, "success": bool(success), "reward": float(success),
            "mean_nfe": float(nfe), "latency_ms_mean": 2.0 * nfe,
            "latency_ms_p95": 3.0 * nfe, "action_delta_rms": None,
            "action_jerk_rms": None, "collision": None}


class MeanFlowAnalysisTests(unittest.TestCase):
    def pair(self, rows, budgets=(1, 1.25, 1.5, 2)):
        return summarize([], rows, bootstrap_samples=100, seed=7, budgets=budgets)["branches"]["families"]["uniform"]["pairs"]["1_vs_2"]

    def test_exact_integer_budget_oracle_and_matched_random(self):
        pair = self.pair(branch_rows([0, 0, 1, 1], [1, 1, 0, 1]))
        point = pair["frontier"][1]
        self.assertEqual(point["oracle_n_upgraded"], 1)
        self.assertEqual(point["oracle_mean_nfe"], 1.25)
        self.assertEqual(point["oracle_success"], 0.75)
        self.assertAlmostEqual(point["random_matched_expected_success"], 0.5625)
        self.assertEqual(point["random_matched_n_upgraded"], 1)
        self.assertEqual(pair["overlap"], {"both_success": 1, "upgrade_win": 2,
            "upgrade_hurt": 1, "both_fail": 0, "unchanged": 1})
        full = pair["frontier"][-1]
        self.assertEqual(full["oracle_n_upgraded"], 2)
        self.assertEqual(full["oracle_mean_nfe"], 1.5)
        self.assertEqual(full["oracle_unused_budget"], 0.5)

    def test_constant_benefit_has_no_adaptive_headroom(self):
        pair = self.pair(branch_rows([0] * 4, [1] * 4))
        self.assertTrue(pair["constant_observed_benefit"])
        self.assertEqual(pair["benefit_variance"], 0)
        for point in pair["frontier"]:
            self.assertAlmostEqual(point["oracle_gain_over_matched_random"], 0)
            self.assertAlmostEqual(point["oracle_gain_over_matched_binary_nonadaptive"], 0)

    def test_negative_upgrades_are_rejected(self):
        pair = self.pair(branch_rows([1, 1, 0], [0, 0, 0]))
        for point in pair["frontier"]:
            self.assertEqual(point["oracle_n_upgraded"], 0)
            self.assertEqual(point["oracle_mean_nfe"], 1)
            self.assertAlmostEqual(point["oracle_success"], 2 / 3)

    def test_oracle_matches_exhaustive_binary_choices(self):
        base, upgraded = [0, 1, 0, 1, 0], [1, 0, 1, 1, 0]
        budgets = (1.0, 1.19, 1.2, 1.4, 1.6, 2.0, 2.2, 4.0)
        summary = summarize([], branch_rows(base, upgraded), bootstrap_samples=0, budgets=budgets)
        for k in (2, 4):
            pair = summary["branches"]["families"]["uniform"]["pairs"][f"1_vs_{k}"]
            for point in pair["frontier"]:
                feasible = []
                for choice in itertools.product((0, 1), repeat=len(base)):
                    cost = 1 + sum(choice) / len(base) * (k - 1)
                    if cost <= point["budget_mean_nfe"] + 1e-12:
                        success = sum(upgraded[i] if choice[i] else base[i] for i in range(len(base))) / len(base)
                        feasible.append(success)
                self.assertAlmostEqual(point["oracle_success"], max(feasible))

    def test_nonadaptive_frontier_uses_all_k(self):
        pair = self.pair(branch_rows([0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]), budgets=(2,))
        point = pair["frontier"][0]
        requested = point["best_all_k_nonadaptive_at_requested_budget"]
        self.assertAlmostEqual(requested["success"], 1 / 3)
        self.assertAlmostEqual(requested["weights"]["uniform_4"], 1 / 3)

    def test_better_fixed_grid_removes_apparent_adaptive_headroom(self):
        rows = branch_rows([0]*4, [1, 0, 0, 0], [0]*4)
        other = branch_rows([0]*4, [1]*4, [1]*4)
        for row in other:
            row["variant"] = row["variant"].replace("uniform", "data_first")
        pair = self.pair(rows+other, budgets=(1.5,))
        point = pair["frontier"][0]
        self.assertGreater(point["oracle_gain_over_matched_all_k_nonadaptive"], 0)
        self.assertAlmostEqual(point["oracle_gain_over_matched_all_grid_nonadaptive"], 0)

    def test_cluster_bootstrap_keeps_correlated_trials_together(self):
        rows = []
        # Two clusters; duplicating each observation within its cluster should
        # not narrow the bootstrap interval as an episode bootstrap would.
        for copies in (1, 20):
            current = []
            for cluster, delta in (("a", 1), ("b", 0)):
                for trial in range(copies):
                    episode = f"{cluster}/{trial}"
                    current.append(fixed_row(episode, cluster, "uniform_1", 0))
                    current.append(fixed_row(episode, cluster, "uniform_2", delta, 2))
            result = summarize(current, [], bootstrap_samples=1000, seed=11)
            interval = result["fixed"]["paired"][0]["success_difference"]
            rows.append(interval)
        self.assertEqual(rows[0]["n_clusters"], 2)
        self.assertEqual(rows[0]["mean"], 0.5)
        self.assertEqual(rows[0]["ci95_low"], rows[1]["ci95_low"])
        self.assertEqual(rows[0]["ci95_high"], rows[1]["ci95_high"])
        self.assertEqual(rows[1]["n_observations"], 40)

    def test_missing_pairs_and_family_integrity(self):
        fixed = [fixed_row("a", "a", "uniform_1", 0),
                 fixed_row("b", "b", "uniform_1", 1),
                 fixed_row("a", "a", "uniform_2", 1, 2)]
        result = summarize(fixed, [], bootstrap_samples=10)
        pair = result["fixed"]["paired"][0]
        self.assertEqual(pair["n_pairs"], 1)
        self.assertEqual(pair["only_reference"], 1)
        incomplete = [r for r in branch_rows([0], [1]) if r["nfe"] != 4]
        result = summarize([], incomplete, bootstrap_samples=0)
        self.assertEqual(result["branches"]["families"]["uniform"]["missing_k"], [4])

    def test_signal_holdout_and_nonfinite_json_safety(self):
        n = 40
        base = [i % 2 for i in range(n)]
        upgraded = [1] * n
        signals = [{"composition_defect": float(1 - base[i]),
                    "action_disagreement": float("nan") if i % 3 == 0 else float(i),
                    "infinite": float("inf")} for i in range(n)]
        result = summarize([], branch_rows(base, upgraded, signals=signals), bootstrap_samples=10, seed=4)
        pair = result["branches"]["families"]["uniform"]["pairs"]["1_vs_2"]
        report = pair["signals"]
        self.assertFalse(set(report["train_clusters"]) & set(report["holdout_clusters"]))
        self.assertEqual(report["metrics"]["composition_defect"]["holdout_benefit_auc_raw"], 1)
        self.assertEqual(report["metrics"]["infinite"]["n_holdout"], 0)
        self.assertIsNone(report["metrics"]["infinite"]["holdout_benefit_auc_raw"])
        self.assertGreater(report["metrics"]["action_disagreement"]["n_missing_nonfinite"], 0)
        json.dumps(result, allow_nan=False)

    def test_report_writes_readable_artifacts(self):
        fixed = [fixed_row("a", "a", "uniform_1", 0), fixed_row("a", "a", "uniform_2", 1, 2)]
        result = summarize(fixed, branch_rows([0, 1], [1, 1]), bootstrap_samples=10)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_report(result, directory)
            self.assertEqual(json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))["schema_version"], 1)
            markdown = Path(paths["report_md"]).read_text(encoding="utf-8")
            self.assertIn("counterfactual", markdown)
            self.assertIn("Matched random expectation", markdown)
            self.assertTrue(Path(paths["fixed_csv"]).exists())
            self.assertTrue(Path(paths["frontier_csv"]).exists())

    def test_duplicate_and_mismatched_clusters_fail(self):
        fixed = [fixed_row("a", "a", "uniform_1", 0)] * 2
        with self.assertRaises(ValueError):
            summarize(fixed, [], bootstrap_samples=0)
        fixed = [fixed_row("a", "a", "uniform_1", 0), fixed_row("a", "b", "uniform_2", 1, 2)]
        with self.assertRaises(ValueError):
            summarize(fixed, [], bootstrap_samples=0)


if __name__ == "__main__":
    unittest.main()
