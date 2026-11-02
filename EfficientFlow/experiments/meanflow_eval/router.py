"""A small held-out benefit router for paired MeanFlow branch experiments.

The response is an observed counterfactual continuation-success difference,
not a best-of-noise label and not deployed adaptive-policy success. Splits are
made over entire cluster_id groups before fitting or threshold selection.
"""

from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np


_FAMILIES = {"uniform", "noise_first", "data_first"}


def _success(value):
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return float(value)
    raise ValueError("success must be boolean or numeric 0/1")


def _feature_vector(features, size=None):
    try:
        values = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be a finite numeric vector") from exc
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or (size is not None and values.size != size)):
        raise ValueError("features must be a finite numeric vector of the expected length")
    return values


def _threshold(scores, upgrade_fraction):
    """Use training scores only; tied cutoffs conservatively leave budget unused."""
    capacity = int(math.floor(len(scores) * upgrade_fraction + 1e-12))
    positive = np.sort(scores[scores > 0])[::-1]
    if capacity >= len(positive):
        return 0.0
    if capacity == 0:
        return max(0.0, float(scores.max()))
    # A strict comparison excludes all ties when splitting them would exceed
    # the training budget; there is no state-dependent random tie breaking.
    upper, lower = float(positive[capacity - 1]), float(positive[capacity])
    return upper if upper == lower else lower + (upper - lower) / 2


def _counterfactual_metrics(base, high, scores, *, threshold, high_nfe, enabled):
    upgrade = (scores > threshold) & (scores > 0) & enabled
    fraction = float(upgrade.mean())
    baseline_success = float(base.mean())
    high_success = float(high.mean())
    random_success = float(baseline_success + fraction * (high_success - baseline_success))
    mean_nfe = float(1 + fraction * (high_nfe - 1))
    if random_success > baseline_success:
        best_mixture = {"success": random_success, "mean_nfe": mean_nfe,
                        "weights": {"low": 1 - fraction, "high": fraction}}
    else:
        best_mixture = {"success": baseline_success, "mean_nfe": 1.0,
                        "weights": {"low": 1.0, "high": 0.0}}
    routed = np.where(upgrade, high, base)
    return {
        "n_states": int(len(base)),
        "n_upgraded": int(upgrade.sum()),
        "upgrade_fraction": fraction,
        "achieved_mean_nfe": mean_nfe,
        "mean_observed_counterfactual_success": float(routed.mean()),
        "baseline_counterfactual_success": baseline_success,
        "high_counterfactual_success": high_success,
        "random_matched_upgrade_rate_expected_success": random_success,
        "best_fixed_mixture_at_achieved_budget": best_mixture,
        "counterfactual_gain_over_random": float(routed.mean() - random_success),
        "benefit_prediction_mse": float(np.mean((scores - (high - base)) ** 2)),
    }


def fit_router(branch_rows, output_path, *, family="uniform", high_nfe=2,
               target_mean_nfe=1.5, seed=0, train_fraction=.7, ridge=1.0):
    """Fit and save a new JSON ridge router, refusing to overwrite any payload.

    Only ``family_1`` and ``family_high_nfe`` rows are paired. Each state_id must
    identify one noise-paired branch comparison; duplicate rows are errors.
    Features are read exclusively from the 1-NFE row. Incomplete pairs and
    missing/nonfinite feature or success records are dropped and counted.
    Inconsistent cluster identities or feature-name orders are errors.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite router payload: {output_path}")
    if family not in _FAMILIES:
        raise ValueError("family must be uniform, noise_first, or data_first")
    if isinstance(high_nfe, bool) or not isinstance(high_nfe, Integral) or high_nfe <= 1:
        raise ValueError("high_nfe must be an integer greater than 1")
    high_nfe = int(high_nfe)
    target_mean_nfe, train_fraction, ridge = map(float, (target_mean_nfe, train_fraction, ridge))
    if not math.isfinite(target_mean_nfe) or not 1 <= target_mean_nfe <= high_nfe:
        raise ValueError("target_mean_nfe must lie between 1 and high_nfe")
    if not math.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    wanted = {f"{family}_1": 1, f"{family}_{high_nfe}": high_nfe}
    by_nfe = {1: {}, high_nfe: {}}
    counts = {"input_rows": 0, "ignored_variant_rows": 0, "missing_state_id_rows": 0,
              "missing_baseline_states": 0, "missing_candidate_states": 0,
              "invalid_feature_states": 0, "invalid_success_states": 0,
              "missing_cluster_states": 0}
    for row in branch_rows:
        counts["input_rows"] += 1
        variant = row.get("variant")
        if variant not in wanted:
            counts["ignored_variant_rows"] += 1
            continue
        nfe = wanted[variant]
        if row.get("nfe") != nfe:
            raise ValueError(f"variant={variant!r} does not match nfe={row.get('nfe')!r}")
        if row.get("state_id") is None:
            counts["missing_state_id_rows"] += 1
            continue
        state_id = str(row["state_id"])
        if state_id in by_nfe[nfe]:
            raise ValueError(f"Duplicate state_id={state_id!r} within {variant}")
        by_nfe[nfe][state_id] = row

    low_rows, high_rows = by_nfe[1], by_nfe[high_nfe]
    counts["missing_baseline_states"] = len(set(high_rows) - set(low_rows))
    counts["missing_candidate_states"] = len(set(low_rows) - set(high_rows))
    features, baseline, upgraded, clusters, states = [], [], [], [], []
    feature_names = None
    for state_id in sorted(set(low_rows) & set(high_rows)):
        low, high = low_rows[state_id], high_rows[state_id]
        if low.get("cluster_id") is None or high.get("cluster_id") is None:
            counts["missing_cluster_states"] += 1
            continue
        cluster = str(low["cluster_id"])
        if cluster != str(high["cluster_id"]):
            raise ValueError(f"state_id={state_id!r} has inconsistent cluster_id")
        try:
            low_success, high_success = _success(low.get("success")), _success(high.get("success"))
        except ValueError:
            counts["invalid_success_states"] += 1
            continue
        names = low.get("feature_names")
        if (not isinstance(names, list) or not names
                or not all(isinstance(name, str) and name for name in names)
                or len(set(names)) != len(names)):
            counts["invalid_feature_states"] += 1
            continue
        try:
            vector = _feature_vector(low.get("features"), len(names))
        except ValueError:
            counts["invalid_feature_states"] += 1
            continue
        if feature_names is not None and names != feature_names:
            raise ValueError(f"state_id={state_id!r} has a different feature_names order")
        feature_names = list(names)
        features.append(vector)
        baseline.append(low_success)
        upgraded.append(high_success)
        clusters.append(cluster)
        states.append(state_id)

    counts["complete_pair_states"] = len(states)
    counts["dropped_states"] = sum(counts[name] for name in (
        "missing_baseline_states", "missing_candidate_states", "invalid_feature_states",
        "invalid_success_states", "missing_cluster_states"))
    unique_clusters = sorted(set(clusters))
    if len(unique_clusters) < 2:
        raise ValueError("At least two complete cluster_id groups are required for train/heldout; "
                         f"counts={counts}")
    shuffled = np.random.default_rng(seed).permutation(unique_clusters)
    n_train = min(len(shuffled) - 1, max(1, int(math.floor(len(shuffled) * train_fraction))))
    training_clusters = sorted(shuffled[:n_train].tolist())
    heldout_clusters = sorted(shuffled[n_train:].tolist())
    training_set = set(training_clusters)
    train = np.array([cluster in training_set for cluster in clusters])
    heldout = ~train
    x, base, high = np.asarray(features), np.asarray(baseline), np.asarray(upgraded)
    response = high - base
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    intercept = float(response[train].mean())
    design, centered_y = standardized[train], response[train] - intercept
    if ridge == 0:
        coef = np.linalg.lstsq(design, centered_y, rcond=None)[0]
    elif design.shape[1] > design.shape[0]:
        dual = np.linalg.solve(design @ design.T + ridge * np.eye(design.shape[0]), centered_y)
        coef = design.T @ dual
    else:
        coef = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]),
                               design.T @ centered_y)
    scores = standardized @ coef + intercept
    target_fraction = (target_mean_nfe - 1) / (high_nfe - 1)
    threshold = _threshold(scores[train], target_fraction)
    enabled = target_fraction > 0
    metrics_kwargs = {"threshold": threshold, "high_nfe": high_nfe, "enabled": enabled}
    payload = {
        "schema": "meanflow_benefit_router_v1",
        "family": family, "low_nfe": 1, "high_nfe": high_nfe,
        "feature_names": feature_names, "coef": coef.tolist(), "intercept": intercept,
        "mean": mean.tolist(), "scale": scale.tolist(), "threshold": threshold,
        "upgrade_enabled": enabled,
        "target_mean_nfe": target_mean_nfe, "target_upgrade_fraction": target_fraction,
        "seed": int(seed), "train_fraction": train_fraction, "ridge": ridge,
        "split_unit": "cluster_id", "training_clusters": training_clusters,
        "heldout_clusters": heldout_clusters, "counts": counts,
        "training": _counterfactual_metrics(base[train], high[train], scores[train], **metrics_kwargs),
        "heldout": _counterfactual_metrics(base[heldout], high[heldout], scores[heldout], **metrics_kwargs),
        "nonadaptive_comparator_scope": "The same low/high pair only; all-K frontiers are in analysis.py.",
        "interpretation": "Observed state-branch counterfactual continuation success under a common "
                          "continuation policy; not deployed closed-loop router-policy success. "
                          "No heldout labels or scores select coefficients, normalization, or threshold. "
                          "The training budget is a ceiling; heldout/deployment NFE may differ. "
                          "Random and best-fixed-mixture comparators use the achieved heldout upgrade rate.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


class BenefitRouter:
    """CPU/numpy linear inference with no simulator or future outcome access."""

    def __init__(self, path):
        with Path(path).open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        payload = self.metadata
        if payload.get("schema") != "meanflow_benefit_router_v1":
            raise ValueError("Unsupported router payload schema")
        self.feature_names = list(payload["feature_names"])
        size = len(self.feature_names)
        self.coef = _feature_vector(payload["coef"], size)
        self.mean = _feature_vector(payload["mean"], size)
        self.scale = _feature_vector(payload["scale"], size)
        if np.any(self.scale <= 0):
            raise ValueError("Router scales must be positive")
        self.intercept, self.threshold = float(payload["intercept"]), float(payload["threshold"])
        if not math.isfinite(self.intercept) or not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Router intercept and nonnegative threshold must be finite")
        self.low_nfe, self.high_nfe = int(payload["low_nfe"]), int(payload["high_nfe"])
        self.family = payload["family"]
        self.upgrade_enabled = bool(payload.get("upgrade_enabled", True))
        if self.low_nfe != 1 or self.high_nfe <= 1 or self.family not in _FAMILIES:
            raise ValueError("Invalid router family or inference budgets")

    def score(self, features):
        values = _feature_vector(features, len(self.feature_names))
        return float(((values - self.mean) / self.scale) @ self.coef + self.intercept)

    def choose(self, features):
        score = self.score(features)
        return self.high_nfe if self.upgrade_enabled and score > self.threshold and score > 0 else self.low_nfe
