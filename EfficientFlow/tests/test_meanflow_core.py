import copy
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.meanflow_eval.backends import ToyBackend
from experiments.meanflow_eval.core import (
    collect_branches, fixed_rollout, isolated_policy_rng, read_jsonl, restore_anchor,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.backend = ToyBackend()
        self.kwargs = dict(env_seed=7, trial=0, variant="uniform_1", grid=(1., 0.),
            base_seed=4, max_steps=12, anchor_every=2, max_anchors=3, capture_anchors=True)

    def test_reference_suffix_replay_and_current_chunk_cost(self):
        reference, anchors = fixed_rollout(self.backend, **self.kwargs)
        variants = {"uniform_1": (1., 0.), "uniform_2": (1., .5, 0.),
                    "uniform_4": (1., .75, .5, .25, 0.)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"branches.jsonl"
            collect_branches(self.backend, anchors, variants, path,
                base_seed=4, max_steps=12, verify_repeats=3)
            rows = read_jsonl(path)
        self.assertEqual(len(rows), len(anchors)*3)
        for row in rows:
            self.assertTrue(row["replay_verified"])
            self.assertTrue(row["branch_repeat_verified"])
            self.assertLessEqual(row["actual_branch_steps"], row["remaining_physical_steps"])
            self.assertEqual(row["continuation_grid"], [1., 0.])
            self.assertEqual(row["composition_signal_nfe"], 3)
            if row["nfe"] == 1:
                self.assertEqual(row["success"], reference["success"])
        for anchor in anchors:
            values = [row for row in rows if row["state_id"] == anchor["state_id"]]
            self.assertTrue(all(row["features"] == values[0]["features"] for row in values))

    def test_bad_replay_or_baseline_suffix_fails(self):
        _, anchors = fixed_rollout(self.backend, **self.kwargs)
        bad = copy.deepcopy(anchors[0])
        bad["sim_state"][0] += .1
        with self.assertRaisesRegex(RuntimeError, "Non-deterministic"):
            restore_anchor(self.backend, bad)
        bad = copy.deepcopy(anchors[0])
        bad["reference_suffix_action_hash"] = "wrong"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Baseline continuation mismatch"):
                collect_branches(self.backend, [bad], {"uniform_1": (1., 0.)},
                    Path(directory)/"rows.jsonl", base_seed=4, max_steps=12)

    def test_router_randomness_cannot_change_simulator_rng_or_same_actions(self):
        class StochasticSensor(ToyBackend):
            def _obs(self):
                obs = super()._obs()
                obs["sensor_noise"] = np.array([np.random.random()], np.float32)
                return obs
        backend = StochasticSensor()
        fixed, _ = fixed_rollout(backend, **self.kwargs)
        def choose(obs, noise, features, names):
            np.random.normal(size=100)
            random.random()
            return (1., 0.)
        routed, _ = fixed_rollout(backend, **dict(self.kwargs, chooser=choose))
        self.assertEqual(fixed["action_hash"], routed["action_hash"])
        self.assertEqual(fixed["success"], routed["success"])

    def test_isolated_rng_restores_python_numpy(self):
        random.seed(3)
        np.random.seed(5)
        py, numpy = random.getstate(), np.random.get_state()
        with isolated_policy_rng(99):
            random.random()
            np.random.random(4)
        self.assertEqual(py, random.getstate())
        np.testing.assert_array_equal(numpy[1], np.random.get_state()[1])

    def test_nondefault_continuation_is_explicit_and_reference_still_checked(self):
        _, anchors = fixed_rollout(self.backend, **self.kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"branches.jsonl"
            collect_branches(self.backend, anchors[:1], {"uniform_1": (1., 0.)}, path,
                base_seed=4, max_steps=12, continuation_grid=(1., .5, 0.))
            row = read_jsonl(path)[0]
        self.assertEqual(row["continuation_grid"], [1., .5, 0.])
        self.assertEqual(row["continuation_nfe"] % 2, 0)


class CliTests(unittest.TestCase):
    def test_cli_fixed_branches_router_roundtrip_from_other_cwd(self):
        script = Path(__file__).resolve().parents[1]/"scripts"/"meanflow_experiment.py"
        def run(*arguments):
            result = subprocess.run([sys.executable, str(script), *map(str, arguments)],
                cwd=directory, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            pilot, test = Path(directory)/"pilot", Path(directory)/"test"
            router = Path(directory)/"router.json"
            run("run", "--backend", "toy", "--phase", "fixed", "--output", pilot,
                "--episodes", 4, "--max-steps", 10, "--schedules", "uniform",
                "--bootstrap-samples", 10)
            run("run", "--phase", "branches", "--output", pilot, "--bootstrap-samples", 10)
            run("fit-router", "--input", pilot, "--output", router)
            run("run", "--backend", "toy", "--phase", "router", "--router", router,
                "--output", test, "--episodes", 4, "--seed-start", 200000,
                "--max-steps", 10, "--bootstrap-samples", 10)
            self.assertTrue((test/"report.md").is_file())
            rows = read_jsonl(test/"router_rollouts.jsonl")
            self.assertEqual({row["variant"] for row in rows}, {"benefit_router", "random_budget"})


if __name__ == "__main__":
    unittest.main()
