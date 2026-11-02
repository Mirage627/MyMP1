"""Descriptive, paired analysis for the MeanFlow go/no-go experiments.

Branch outcomes are counterfactual continuation outcomes from saved states, not
rollouts of a deployed adaptive sampler.  The binary oracle below is the exact
integer optimum for its *observed* 1-vs-K outcomes and a mean-NFE cost ceiling.
It is deliberately not advertised as a general multi-choice oracle or an
unbiased estimate of a policy's expected return.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np


_FAMILY = re.compile(r"^(uniform|noise_first|data_first)_(\d+)$")
_METRICS = ("reward", "latency_ms_mean", "latency_ms_p95", "mean_nfe",
            "action_delta_rms", "action_jerk_rms", "total_nfe", "physical_steps",
            "decision_count", "router_latency_ms_mean")


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _mean(values):
    valid = [v for value in values if (v := _number(value)) is not None]
    return float(np.mean(valid)) if valid else None


def _stable_seed(seed, label):
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _bootstrap(values, clusters, samples, seed):
    """Resample entire clusters, preserving all their paired observations.

    The estimand is the pooled observation mean, not an equally weighted mean
    of cluster means. Unequal cluster sizes are retained in every resample.
    """
    values = np.asarray(values, dtype=float)
    result = {"mean": float(values.mean()) if len(values) else None,
              "ci95_low": None, "ci95_high": None,
              "n_observations": int(len(values)),
              "n_clusters": int(len(set(clusters))),
              "bootstrap_samples": int(samples),
              "method": "paired_cluster_percentile_bootstrap"}
    if not len(values):
        return result
    labels = sorted(set(clusters))
    sums = np.array([values[np.array(clusters) == label].sum() for label in labels])
    counts = np.array([sum(c == label for c in clusters) for label in labels])
    if samples:
        rng = np.random.default_rng(seed)
        estimates = []
        # Bound temporary memory for large experiments.
        for start in range(0, samples, 128):
            weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)),
                                      size=min(128, samples - start))
            estimates.extend(((weights @ sums) / (weights @ counts)).tolist())
        result["ci95_low"], result["ci95_high"] = map(
            float, np.quantile(estimates, [0.025, 0.975]))
    if len(labels) < 2:
        result["caution"] = "One cluster: the bootstrap cannot measure across-cluster uncertainty."
    return result


def _overlap(base, upgraded):
    base, upgraded = np.asarray(base, bool), np.asarray(upgraded, bool)
    return {"both_success": int(np.sum(base & upgraded)),
            "upgrade_win": int(np.sum(~base & upgraded)),
            "upgrade_hurt": int(np.sum(base & ~upgraded)),
            "both_fail": int(np.sum(~base & ~upgraded)),
            "unchanged": int(np.sum(base == upgraded))}


def _index(rows, key):
    indexed = {}
    for row in rows:
        name = str(row[key])
        if name in indexed:
            raise ValueError(f"Duplicate {key}={name!r} within one variant")
        indexed[name] = row
    return indexed


def _paired(base, upgraded, key, samples, seed):
    common = sorted(set(base) & set(upgraded))
    clusters = []
    for item in common:
        left = str(base[item]["cluster_id"])
        right = str(upgraded[item]["cluster_id"])
        if left != right:
            raise ValueError(f"{key}={item!r} has inconsistent cluster_id")
        clusters.append(left)
    b = np.array([bool(base[item]["success"]) for item in common])
    u = np.array([bool(upgraded[item]["success"]) for item in common])
    return common, clusters, b, u, {
        "n_pairs": len(common), "only_reference": len(set(base) - set(upgraded)),
        "only_candidate": len(set(upgraded) - set(base)),
        "success_difference": _bootstrap(u.astype(float) - b.astype(float), clusters,
                                         samples, seed),
        "overlap": _overlap(b, u),
    }


def _ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranked


def _spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = _ranks(x), _ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(scores, positive):
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = float(_ranks(scores)[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _signals(base_rows, benefits, clusters, seed):
    labels = sorted(set(clusters))
    if len(labels) < 2:
        return {"status": "unavailable", "reason": "At least two clusters are needed for holdout."}
    shuffled = np.random.default_rng(seed).permutation(labels)
    n_test = max(1, int(math.ceil(0.3 * len(labels))))
    n_test = min(n_test, len(labels) - 1)
    test_clusters = set(shuffled[:n_test].tolist())
    names = sorted({name for row in base_rows for name in (row.get("signals") or {})})
    metrics = {}
    for name in names:
        train_x, train_y, test_x, test_y = [], [], [], []
        missing = 0
        for row, benefit, cluster in zip(base_rows, benefits, clusters):
            value = _number((row.get("signals") or {}).get(name))
            if value is None:
                missing += 1
                continue
            xs, ys = (test_x, test_y) if cluster in test_clusters else (train_x, train_y)
            xs.append(value)
            ys.append(float(benefit))
        train_auc = _auc(train_x, np.asarray(train_y) > 0)
        direction = -1 if train_auc is not None and train_auc < 0.5 else 1
        metrics[name] = {
            "n_train": len(train_x), "n_holdout": len(test_x), "n_missing_nonfinite": missing,
            "direction_selected_on_train": direction,
            "direction_identifiable_on_train": train_auc is not None,
            "train_benefit_auc_raw": train_auc,
            "holdout_benefit_auc_raw": _auc(test_x, np.asarray(test_y) > 0),
            "holdout_benefit_auc_train_oriented": _auc(
                np.asarray(test_x) * direction, np.asarray(test_y) > 0),
            "holdout_benefit_spearman_raw": _spearman(test_x, test_y),
        }
    return {"status": "ok", "split_unit": "cluster_id", "holdout_fraction_target": 0.3,
            "train_clusters": sorted(set(labels) - test_clusters),
            "holdout_clusters": sorted(test_clusters), "metrics": metrics,
            "note": "Signals come only from the 1-NFE row. No holdout thresholds are tuned. "
                    "AUC predicts upgrade_win; Spearman uses signed success difference. "
                    "Signal computation overhead is not included unless logged separately."}


def _mixture(points, budget):
    """Exact LP optimum over global, state-independent convex mixtures.

    With one scalar cost constraint an optimum needs at most two variants, so
    enumerating all vertices and two-point segments solves this small LP.
    """
    candidates = []
    for name, cost, success in points:
        if cost <= budget + 1e-12:
            candidates.append((success, cost, {name: 1.0}))
    for (left, lc, ls), (right, rc, rs) in combinations(points, 2):
        if lc == rc or not min(lc, rc) <= budget <= max(lc, rc):
            continue
        fraction = (budget - lc) / (rc - lc)
        candidates.append((ls * (1 - fraction) + rs * fraction, budget,
                           {left: float(1 - fraction), right: float(fraction)}))
    if not candidates:
        return None
    success, cost, weights = max(candidates, key=lambda x: (x[0], -x[1]))
    return {"success": float(success), "mean_nfe": float(cost), "weights": weights}


def _binary_frontier(base, upgraded, upgrade_nfe, budgets, points, seed):
    n = len(base)
    benefits = upgraded.astype(float) - base.astype(float)
    base_success = float(base.mean())
    wins = int(np.sum(benefits > 0))
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(n)
    rows = []
    for budget in budgets:
        if budget < 1:
            rows.append({"budget_mean_nfe": float(budget), "status": "below_base_cost"})
            continue
        capacity = min(n, max(0, math.floor(n * (budget - 1) / (upgrade_nfe - 1) + 1e-10)))
        selected = min(wins, capacity)
        fraction = selected / n
        used = 1 + fraction * (upgrade_nfe - 1)
        oracle_success = base_success + selected / n
        random_expected = base_success + fraction * float(benefits.mean())
        random_single = base_success + float(benefits[random_order[:selected]].sum()) / n
        variance_sum = (selected * (n - selected) / (n - 1) * float(np.var(benefits))) if n > 1 else 0.0
        binary_points = [point for point in points if point[1] in (1, upgrade_nfe)]
        matched_binary = _mixture(binary_points, used)
        matched_all = _mixture(points, used)
        rows.append({
            "budget_mean_nfe": float(budget), "status": "ok", "upgrade_capacity": capacity,
            "oracle_n_upgraded": selected, "oracle_upgrade_fraction": float(fraction),
            "oracle_mean_nfe": float(used), "oracle_success": float(oracle_success),
            "oracle_unused_budget": float(max(0, budget - used)),
            "random_matched_n_upgraded": selected,
            "random_matched_expected_success": float(random_expected),
            "random_matched_allocation_std": float(math.sqrt(max(0, variance_sum)) / n),
            "random_matched_single_success": float(random_single),
            "best_binary_nonadaptive_at_oracle_cost": matched_binary,
            "best_all_k_nonadaptive_at_oracle_cost": matched_all,
            "best_all_k_nonadaptive_at_requested_budget": _mixture(points, budget),
            "oracle_gain_over_matched_random": float(oracle_success - random_expected),
            "oracle_gain_over_matched_binary_nonadaptive": float(oracle_success - matched_binary["success"]),
            "oracle_gain_over_matched_all_k_nonadaptive": float(oracle_success - matched_all["success"]),
        })
    return rows


def summarize(fixed_rows, branch_rows, *, bootstrap_samples=2000, seed=0,
              budgets=(1.0, 1.25, 1.5, 2.0, 3.0, 4.0)):
    """Return a JSON-safe analysis without fitting thresholds on evaluation data."""
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    budgets = tuple(float(value) for value in budgets)
    if any(not math.isfinite(value) for value in budgets):
        raise ValueError("budgets must be finite")
    fixed_rows, branch_rows = list(fixed_rows), list(branch_rows)
    by_variant = {}
    for row in fixed_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    indexes = {name: _index(rows, "episode_id") for name, rows in by_variant.items()}
    fixed = {"variants": {}, "paired": []}
    order = sorted(by_variant, key=lambda name: (
        _mean(r.get("mean_nfe", r.get("nfe")) for r in by_variant[name]) or float("inf"), name))
    for name in order:
        rows = by_variant[name]
        success = [float(bool(row["success"])) for row in rows]
        entry = {"n_episodes": len(rows),
                 "n_clusters": len({str(row["cluster_id"]) for row in rows}),
                 "nfe_values": sorted({int(row["nfe"]) for row in rows if row.get("nfe") is not None}
                                      | {int(k) for row in rows for k in row.get("nfe_counts", {})}),
                 "success": _bootstrap(success, [str(r["cluster_id"]) for r in rows],
                                       bootstrap_samples, _stable_seed(seed, "fixed/" + name))}
        entry.update({metric: _mean(row.get(metric) for row in rows) for metric in _METRICS})
        collisions = [row.get("collision") for row in rows if row.get("collision") is not None]
        entry["collision_rate"] = _mean(collisions)
        entry["collision_n_observed"] = len(collisions)
        fixed["variants"][name] = entry
    for reference, candidate in combinations(order, 2):
        ids, clusters, _, _, pair = _paired(indexes[reference], indexes[candidate], "episode_id",
                                           bootstrap_samples, _stable_seed(seed, reference + "/" + candidate))
        pair.update({"reference": reference, "candidate": candidate})
        differences = []
        reward_clusters = []
        for item, cluster in zip(ids, clusters):
            a = _number(indexes[reference][item].get("reward"))
            b = _number(indexes[candidate][item].get("reward"))
            if a is not None and b is not None:
                differences.append(b - a)
                reward_clusters.append(cluster)
        pair["reward_difference"] = _bootstrap(differences, reward_clusters, bootstrap_samples,
                                                _stable_seed(seed, "reward/" + reference + "/" + candidate))
        fixed["paired"].append(pair)

    families, ignored = {}, []
    for row in branch_rows:
        match = _FAMILY.match(str(row["variant"]))
        if not match:
            ignored.append(str(row["variant"]))
            continue
        family, k = match.group(1), int(match.group(2))
        if int(row["nfe"]) != k:
            raise ValueError(f"variant={row['variant']!r} does not match nfe={row['nfe']!r}")
        families.setdefault(family, {}).setdefault(k, []).append(row)
    branch_summary = {"families": {}, "ignored_variants": sorted(set(ignored))}
    all_branch_indexes = {
        str(name): _index([row for row in branch_rows if str(row["variant"]) == name], "state_id")
        for name in sorted({str(row["variant"]) for row in branch_rows})}
    for family in sorted(families):
        indexes_k = {k: _index(rows, "state_id") for k, rows in families[family].items()}
        missing_k = sorted({1, 2, 4} - set(indexes_k))
        if missing_k:
            branch_summary["families"][family] = {"status": "incomplete", "missing_k": missing_k,
                "reason": "Family analysis requires K=1,2,4; no cross-family substitution is made."}
            continue
        common = sorted(set.intersection(*(set(indexes_k[k]) for k in (1, 2, 4))))
        item = {"status": "ok" if common else "no_complete_states", "n_complete_states": len(common),
                "excluded_incomplete_states": len(set.union(*(set(indexes_k[k]) for k in (1, 2, 4)))) - len(common),
                "pairs": {}}
        branch_summary["families"][family] = item
        if not common:
            continue
        complete = {k: {state: indexes_k[k][state] for state in common} for k in (1, 2, 4)}
        points = [(f"{family}_{k}", k, _mean(float(bool(complete[k][s]["success"])) for s in common))
                  for k in (1, 2, 4)]
        item["nonadaptive_points"] = [{"variant": name, "nfe": k, "success": rate}
                                        for name, k, rate in points]
        global_points = []
        for name, index in all_branch_indexes.items():
            if not set(common).issubset(index):
                continue
            costs = {_number(index[state].get("nfe")) for state in common}
            if len(costs) != 1 or None in costs:
                continue
            for state in common:
                if str(index[state]["cluster_id"]) != str(complete[1][state]["cluster_id"]):
                    raise ValueError(f"Cross-grid cluster mismatch at {state}")
            global_points.append((name, next(iter(costs)),
                float(np.mean([bool(index[state]["success"]) for state in common]))))
        item["all_grid_nonadaptive_points"] = [
            {"variant": name, "nfe": k, "success": rate} for name, k, rate in global_points]
        item["variant_stats"] = {}
        for k in (1, 2, 4):
            matched_rows = [complete[k][state] for state in common]
            latencies = [value for row in matched_rows
                         if (value := _number(row.get("latency_ms"))) is not None]
            item["variant_stats"][f"{family}_{k}"] = {
                "success": _mean(float(bool(row["success"])) for row in matched_rows),
                "reward": _mean(row.get("reward") for row in matched_rows),
                "latency_ms_mean": _mean(latencies),
                "latency_ms_p95": float(np.quantile(latencies, 0.95)) if latencies else None,
                "latency_n_observed": len(latencies),
            }
        for k in (2, 4):
            label = f"{family}/1_vs_{k}"
            ids, clusters, b, u, pair = _paired(complete[1], complete[k], "state_id", bootstrap_samples,
                                               _stable_seed(seed, label))
            benefits = u.astype(float) - b.astype(float)
            pair.update({"reference": f"{family}_1", "candidate": f"{family}_{k}",
                "base_success": float(b.mean()), "upgraded_success": float(u.mean()),
                "benefit_variance": float(np.var(benefits)),
                "constant_observed_benefit": bool(np.ptp(benefits) == 0),
                "frontier": _binary_frontier(b, u, k, budgets, points, _stable_seed(seed, label + "/random")),
                "signals": _signals([complete[1][s] for s in ids], benefits, clusters,
                                     _stable_seed(seed, family + "/holdout"))})
            for point in pair["frontier"]:
                if point["status"] == "ok":
                    comparator = _mixture(global_points, point["oracle_mean_nfe"])
                    point["best_all_grid_nonadaptive_at_oracle_cost"] = comparator
                    point["oracle_gain_over_matched_all_grid_nonadaptive"] = point["oracle_success"]-comparator["success"]
            item["pairs"][f"1_vs_{k}"] = pair
    return {"schema_version": 1, "seed": int(seed), "bootstrap_samples": bootstrap_samples,
        "budgets": list(budgets), "fixed": fixed, "branches": branch_summary,
        "interpretation": [
            "Fixed comparisons pair identical episode_id and resample cluster_id (environment seed), not individual noise trials.",
            "Branch success is counterfactual Q under one common continuation policy and horizon, not a complete adaptive-policy rollout.",
            "Oracle is an exact observed-outcome integer optimum ONLY for binary 1-vs-2 or 1-vs-4 allocation within a grid family; it can leave budget unused and does not solve the general 1/2/4 allocation problem.",
            "This hindsight oracle sees evaluation outcomes and is an optimistic diagnostic envelope, not a deployable success rate, unbiased expected-Q oracle, or online guarantee.",
            "Random comparison chooses exactly the oracle's upgrade count; its expectation is analytic and its allocation standard deviation is not a confidence interval.",
            "Best nonadaptive mixtures solve the finite global convex-mixture LP under the same cost ceiling, using the same complete states; they are hindsight descriptive comparators too.",
            "All-grid nonadaptive comparators also include other time-grid families evaluated on those same states; this prevents attributing a better fixed schedule to adaptive routing.",
            "Mean NFE is not equal wall-clock cost. Fixed latency_ms_p95 is averaged across episodes, not a pooled global p95. Branch generation latencies are descriptive only.",
            "Signal direction is selected on training clusters only; AUC/Spearman are evaluated on disjoint holdout clusters. Missing/nonfinite scores are dropped per signal.",
        ]}


def _fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def _interval(value):
    return f"{_fmt(value['mean'])} [{_fmt(value['ci95_low'])}, {_fmt(value['ci95_high'])}]"


def write_report(summary, output_dir):
    """Write summary.json, report.md and compact CSV tables; return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, report_path = output_dir / "summary.json", output_dir / "report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# MeanFlow go/no-go diagnostics", "", "## Interpretation", ""]
    lines.extend(f"- {note}" for note in summary["interpretation"])
    lines.extend(["", "## Fixed complete rollouts", "",
                  "| Variant | Episodes | Success [cluster-bootstrap 95% CI] | Mean NFE | Mean latency ms | Mean episode p95 ms |",
                  "|---|---:|---|---:|---:|---:|"])
    fixed_csv = output_dir / "fixed_summary.csv"
    with fixed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "n_episodes", "success", "ci95_low", "ci95_high", "mean_nfe", "latency_ms_mean", "latency_ms_p95"])
        writer.writeheader()
        for name, row in summary["fixed"]["variants"].items():
            lines.append(f"| {name} | {row['n_episodes']} | {_interval(row['success'])} | {_fmt(row['mean_nfe'])} | {_fmt(row['latency_ms_mean'])} | {_fmt(row['latency_ms_p95'])} |")
            writer.writerow({"variant": name, "n_episodes": row["n_episodes"], "success": row["success"]["mean"],
                             "ci95_low": row["success"]["ci95_low"], "ci95_high": row["success"]["ci95_high"],
                             **{key: row[key] for key in ("mean_nfe", "latency_ms_mean", "latency_ms_p95")}})
    lines.extend(["", "## Paired success differences", "",
                  "| Candidate minus reference | Paired episodes | Difference [95% CI] | Wins | Hurts | Both success | Both fail |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    for row in summary["fixed"]["paired"]:
        overlap = row["overlap"]
        lines.append(f"| {row['candidate']} - {row['reference']} | {row['n_pairs']} | {_interval(row['success_difference'])} | {overlap['upgrade_win']} | {overlap['upgrade_hurt']} | {overlap['both_success']} | {overlap['both_fail']} |")
    frontier_rows = []
    lines.extend(["", "## Saved-state branch diagnostics", "",
                  "These are observed counterfactual continuation outcomes, not adaptive rollout success.", ""])
    for family, family_row in summary["branches"]["families"].items():
        lines.extend([f"### {family}", ""])
        if family_row["status"] != "ok":
            lines.extend([family_row.get("reason", family_row["status"]), ""])
            continue
        lines.extend([f"Complete matched states: {family_row['n_complete_states']}; excluded incomplete states: {family_row['excluded_incomplete_states']}.", ""])
        for pair_name, pair in family_row["pairs"].items():
            overlap = pair["overlap"]
            lines.extend([f"#### {pair_name}", "",
                f"Upgrade wins / hurts / unchanged: {overlap['upgrade_win']} / {overlap['upgrade_hurt']} / {overlap['unchanged']}. "
                f"Mean benefit [cluster-bootstrap 95% CI]: {_interval(pair['success_difference'])}.", "",
                "| Budget | Oracle actual mean NFE | Observed binary oracle | Matched random expectation | Best same-family all-K | Best ALL-grid nonadaptive | Oracle minus ALL-grid nonadaptive |",
                "|---:|---:|---:|---:|---:|---:|---:|"])
            for point in pair["frontier"]:
                if point["status"] != "ok":
                    continue
                lines.append(f"| {_fmt(point['budget_mean_nfe'])} | {_fmt(point['oracle_mean_nfe'])} | {_fmt(point['oracle_success'])} | {_fmt(point['random_matched_expected_success'])} | {_fmt(point['best_all_k_nonadaptive_at_oracle_cost']['success'])} | {_fmt(point['best_all_grid_nonadaptive_at_oracle_cost']['success'])} | {_fmt(point['oracle_gain_over_matched_all_grid_nonadaptive'])} |")
                frontier_rows.append({"family": family, "pair": pair_name,
                    **{key: point[key] for key in ("budget_mean_nfe", "oracle_mean_nfe", "oracle_n_upgraded", "oracle_success", "random_matched_expected_success", "oracle_gain_over_matched_all_k_nonadaptive", "oracle_gain_over_matched_all_grid_nonadaptive")}})
            lines.extend(["", "Holdout signals (baseline 1-NFE only; direction chosen on train clusters):", "",
                          "| Signal | Train / holdout | Missing | Holdout AUC (train orientation) | Holdout Spearman (raw) |",
                          "|---|---:|---:|---:|---:|"])
            signals = pair["signals"]
            if signals["status"] == "ok":
                for name, metric in signals["metrics"].items():
                    lines.append(f"| {name} | {metric['n_train']} / {metric['n_holdout']} | {metric['n_missing_nonfinite']} | {_fmt(metric['holdout_benefit_auc_train_oriented'])} | {_fmt(metric['holdout_benefit_spearman_raw'])} |")
            else:
                lines.append(signals["reason"])
            lines.append("")
    frontier_csv = output_dir / "oracle_frontiers.csv"
    with frontier_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["family", "pair", "budget_mean_nfe", "oracle_mean_nfe", "oracle_n_upgraded", "oracle_success", "random_matched_expected_success", "oracle_gain_over_matched_all_k_nonadaptive", "oracle_gain_over_matched_all_grid_nonadaptive"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frontier_rows)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"summary_json": str(json_path), "report_md": str(report_path),
            "fixed_csv": str(fixed_csv), "frontier_csv": str(frontier_csv)}
