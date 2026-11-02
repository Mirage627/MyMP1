import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.meanflow_eval.router import BenefitRouter, fit_router


def example_rows(n_clusters=6, family="uniform", high_nfe=2):
    rows = []
    for cluster in range(n_clusters):
        for index, signal in enumerate((-4., -3., -2., -1., 1., 2., 3., 4.)):
            state = f"c{cluster}_s{index}"
            for nfe, success in ((1, signal < 0), (high_nfe, signal > 0)):
                rows.append({
                    "state_id": state, "cluster_id": f"c{cluster}",
                    "variant": f"{family}_{nfe}", "nfe": nfe, "success": success,
                    "features": [signal, 8.], "feature_names": ["signal", "constant"],
                })
    return rows


class BenefitRouterTests(unittest.TestCase):
    def test_fit_and_inference_with_cluster_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            meta = fit_router(example_rows(), path, seed=5)
            self.assertTrue(path.exists())
            self.assertEqual(meta["split_unit"], "cluster_id")
            self.assertFalse(set(meta["training_clusters"]) & set(meta["heldout_clusters"]))
            self.assertEqual(set(meta["training_clusters"]) | set(meta["heldout_clusters"]),
                             {f"c{i}" for i in range(6)})
            self.assertEqual(meta["scale"][1], 1.)
            self.assertEqual(meta["heldout"]["achieved_mean_nfe"], 1.5)
            self.assertEqual(meta["heldout"]["mean_observed_counterfactual_success"], 1.)
            self.assertEqual(meta["heldout"]["random_matched_upgrade_rate_expected_success"], .5)
            self.assertEqual(meta["heldout"]["best_fixed_mixture_at_achieved_budget"]["success"], .5)
            self.assertIn("not deployed", meta["interpretation"])
            router = BenefitRouter(path)
            self.assertEqual(router.choose([-4, 8]), 1)
            self.assertEqual(router.choose([4, 8]), 2)
            self.assertGreater(router.score([4, 8]), router.score([-4, 8]))
            self.assertEqual(router.feature_names, ["signal", "constant"])

    def test_heldout_labels_and_features_cannot_tune_model_or_threshold(self):
        rows = example_rows()
        with tempfile.TemporaryDirectory() as directory:
            first = fit_router(rows, Path(directory) / "first.json", seed=11)
            heldout = set(first["heldout_clusters"])
            modified = copy.deepcopy(rows)
            for row in modified:
                if row["cluster_id"] in heldout:
                    if row["nfe"] == 2:
                        row["success"] = False
                    row["features"][0] = row["features"][0] * 1000 + 10000
            second = fit_router(modified, Path(directory) / "second.json", seed=11)
            for key in ("coef", "intercept", "mean", "scale", "threshold",
                        "training_clusters", "heldout_clusters"):
                self.assertEqual(first[key], second[key], key)
            self.assertNotEqual(first["heldout"]["benefit_prediction_mse"],
                                second["heldout"]["benefit_prediction_mse"])

    def test_input_order_does_not_change_split_or_fit(self):
        rows = example_rows()
        with tempfile.TemporaryDirectory() as directory:
            first = fit_router(rows, Path(directory) / "first.json", seed=2)
            second = fit_router(reversed(rows), Path(directory) / "second.json", seed=2)
            self.assertEqual(first, second)

    def test_features_are_read_only_from_baseline(self):
        rows = example_rows()
        for row in rows:
            if row["nfe"] == 2:
                row["features"], row["feature_names"] = [float("nan")], ["future_outcome"]
        with tempfile.TemporaryDirectory() as directory:
            meta = fit_router(rows, Path(directory) / "router.json")
            self.assertEqual(meta["counts"]["invalid_feature_states"], 0)
            self.assertEqual(meta["feature_names"], ["signal", "constant"])

    def test_missing_pairs_and_invalid_rows_are_counted(self):
        rows = example_rows()
        prototype = copy.deepcopy(rows[0])
        rows.append(dict(prototype, state_id="only_low"))
        rows.append(dict(prototype, state_id="only_high", variant="uniform_2", nfe=2))
        for state, change in (("bad_feature", {"features": [float("nan"), 8.]}),
                              ("bad_success", {"success": None}),
                              ("no_cluster", {"cluster_id": None})):
            for nfe in (1, 2):
                row = dict(prototype, state_id=state, nfe=nfe, variant=f"uniform_{nfe}")
                row.update(change)
                rows.append(row)
        rows.append(dict(prototype, state_id=None))
        rows.append(dict(prototype, variant="data_first_1"))
        with tempfile.TemporaryDirectory() as directory:
            meta = fit_router(rows, Path(directory) / "router.json")
            counts = meta["counts"]
            self.assertEqual(counts["complete_pair_states"], 48)
            for key in ("missing_baseline_states", "missing_candidate_states", "invalid_feature_states",
                        "invalid_success_states", "missing_cluster_states", "missing_state_id_rows",
                        "ignored_variant_rows"):
                self.assertEqual(counts[key], 1, key)
            self.assertEqual(counts["dropped_states"], 5)

    def test_duplicates_and_inconsistent_cluster_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = example_rows()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                fit_router(rows + [rows[0]], Path(directory) / "duplicate.json")
            rows[1]["cluster_id"] = "wrong_cluster"
            with self.assertRaisesRegex(ValueError, "inconsistent cluster"):
                fit_router(rows, Path(directory) / "mismatch.json")

    def test_schema_mismatch_and_single_cluster_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = example_rows()
            rows[2]["feature_names"] = ["constant", "signal"]
            with self.assertRaisesRegex(ValueError, "feature_names order"):
                fit_router(rows, Path(directory) / "schema.json")
            with self.assertRaisesRegex(ValueError, "two complete cluster"):
                fit_router(example_rows(1), Path(directory) / "one.json")

    def test_training_threshold_is_a_budget_ceiling_with_ties(self):
        rows = example_rows()
        for row in rows:
            row["features"] = [1., 8.]
            row["success"] = row["nfe"] == 2
        with tempfile.TemporaryDirectory() as directory:
            meta = fit_router(rows, Path(directory) / "router.json", target_mean_nfe=1.5)
            self.assertLessEqual(meta["training"]["achieved_mean_nfe"], 1.5)
            self.assertEqual(meta["training"]["achieved_mean_nfe"], 1.)

    def test_no_positive_benefit_keeps_one_step(self):
        rows = example_rows()
        for row in rows:
            row["success"] = row["nfe"] == 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            meta = fit_router(rows, path)
            self.assertEqual(meta["threshold"], 0.)
            self.assertEqual(meta["heldout"]["achieved_mean_nfe"], 1.)
            router = BenefitRouter(path)
            self.assertEqual(router.choose([1000., 8.]), 1)

    def test_zero_and_full_upgrade_budgets_and_four_nfe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.json"
            fit_router(example_rows(), path, target_mean_nfe=1)
            self.assertEqual(BenefitRouter(path).choose([1000., 8.]), 1)
            rows = example_rows(family="noise_first", high_nfe=4)
            for row in rows:
                row["success"] = row["nfe"] == 4
            path = Path(directory) / "four.json"
            meta = fit_router(rows, path, family="noise_first", high_nfe=4, target_mean_nfe=4)
            self.assertEqual(meta["heldout"]["achieved_mean_nfe"], 4.)
            self.assertEqual(BenefitRouter(path).choose([1., 8.]), 4)

    def test_refuses_to_overwrite_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            path.write_text("existing payload", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fit_router(example_rows(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), "existing payload")

    def test_ridge_zero_and_more_features_than_states(self):
        rows = example_rows(2)
        for row in rows:
            row["features"] = row["features"] + [float(row["features"][0]) ** i for i in range(3, 20)]
            row["feature_names"] += [f"extra_{i}" for i in range(3, 20)]
        with tempfile.TemporaryDirectory() as directory:
            for penalty in (0., 1.):
                meta = fit_router(rows, Path(directory) / f"router_{penalty}.json", ridge=penalty)
                self.assertTrue(np.isfinite(meta["coef"]).all())

    def test_invalid_arguments_and_inference_features(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ({"high_nfe": 1}, {"high_nfe": True}, {"target_mean_nfe": .5},
                            {"target_mean_nfe": 3}, {"train_fraction": 1},
                            {"train_fraction": 0}, {"ridge": -1}, {"family": "other"}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    fit_router(example_rows(), Path(directory) / "bad.json", **options)
            path = Path(directory) / "router.json"
            fit_router(example_rows(), path)
            router = BenefitRouter(path)
            for features in ([1.], [1., float("nan")], [[1., 8.]], None):
                with self.subTest(features=features), self.assertRaises(ValueError):
                    router.choose(features)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["scale"][0] = 0
            bad_path = Path(directory) / "corrupt.json"
            bad_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scales must be positive"):
                BenefitRouter(bad_path)


if __name__ == "__main__":
    unittest.main()
