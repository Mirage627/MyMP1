"""Evaluate and diagnose real EfficientAdaFlow checkpoints.

Only the Python standard library is needed for --help. Robotics and tensor
dependencies are loaded after a subcommand is selected. No synthetic backend
or fabricated policy result is provided by this entry point.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import math
import numbers
import operator
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import traceback


def _source_root(entry_root):
    """Keep outer server launchers and the nested source package compatible."""
    for candidate in (entry_root, entry_root / "EfficientFlow"):
        if ((candidate / "__init__.py").is_file()
                and (candidate / "policy/EfficientAdaFlow.py").is_file()
                and (candidate / "common/adaptive_flow.py").is_file()):
            return candidate
    raise FileNotFoundError(
        f"Cannot find the EfficientFlow source package at {entry_root} or "
        f"{entry_root / 'EfficientFlow'}. Keep policy/, common/ and __init__.py together.")


ROOT = _source_root(Path(__file__).resolve().parents[1])
POLICY_TARGET = "EfficientFlow.policy.EfficientAdaFlow.EfficientAdaFlowPolicy"


def _register_package():
    # Spawned rollout workers inherit sys.path, not the parent's sys.modules.
    # The parent directory makes a normally named EfficientFlow package visible.
    for path in (ROOT.parent, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if getattr(sys.modules.get("EfficientFlow"), "_efficientadaflow_source_root", None) == str(ROOT):
        return
    spec = importlib.util.spec_from_file_location(
        "EfficientFlow", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["EfficientFlow"] = package
    spec.loader.exec_module(package)
    package._efficientadaflow_source_root = str(ROOT)


# Multiprocessing spawn re-executes the main module before unpickling rollout
# workers. Register the package in that process too, even when the checkout's
# directory is named PROJECT_EfficientFlow rather than EfficientFlow. This uses
# only the standard library; package __init__.py has no runtime dependencies.
_register_package()


def _arithmetic(expression):
    """Support saved numerical Hydra expressions without executing Python code."""
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
              ast.Pow: operator.pow, ast.Mod: operator.mod}
    unary = {ast.USub: operator.neg, ast.UAdd: operator.pos}
    functions = {"int": int, "float": float, "max": max, "min": min,
                 "round": round, "abs": abs}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            return unary[type(node.op)](visit(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and not node.keywords):
            return functions[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError(f"Unsupported checkpoint arithmetic expression: {expression}")

    return visit(ast.parse(str(expression), mode="eval").body)


def _dependencies():
    _register_package()
    try:
        import dill
        import hydra
        import numpy as np
        import torch
        from omegaconf import OmegaConf, open_dict
        # Legacy servers can retain an unrelated outer/inner train.py. Reuse
        # only the lightweight resolver module used by the independent entry.
        from EfficientFlow.common.train_entry import register_resolvers
    except ImportError as exc:
        raise RuntimeError(
            f"Missing runtime dependency ({exc}). Activate the EfficientFlow training "
            "environment; --help works without robotics dependencies.") from exc
    register_resolvers()
    OmegaConf.register_new_resolver("eval", _arithmetic, replace=True)
    if not OmegaConf.has_resolver("now"):
        # Older saved configs can retain timestamps in unused logging metadata;
        # a plain CLI has no active Hydra application to register this resolver.
        OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern))
    return dill, hydra, np, torch, OmegaConf, open_dict


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes():
    names = (
        "scripts/efficientadaflow_experiment.py", "common/train_entry.py",
        "policy/EfficientAdaFlow.py", "policy/EfficientFlow.py",
        "common/adaptive_flow.py", "common/equivariant_checkpoint.py",
        "model/equi/invariant_variance_head.py", "model/equi/equi_conditional_unet1d.py",
        "model/equi/equi_obs_encoder.py", "model/common/normalizer.py",
        "env_runner/robomimic_image_runner.py", "config/EfficientAdaFlow.yaml",
        "config/task/mimicgen_abs.yaml", "dataset/robomimic_replay_image_dataset.py",
        "dataset/robomimic_replay_image_sym_dataset.py",
    )
    return {name: _sha256(ROOT / name) for name in names if (ROOT / name).is_file()}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  default=_json_default, allow_nan=False)
        stream.write("\n")


def _append_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, default=_json_default,
                            allow_nan=False) + "\n")
    stream.flush()


def _output_directory(path, overwrite):
    directory = Path(path).expanduser().resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        if not overwrite:
            raise FileExistsError(
                f"Output is not an empty directory: {directory}. Choose a new output "
                "or pass --overwrite to replace this command's JSON reports.")
        if not directory.is_dir():
            raise ValueError(f"Output must be a directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _seed_everything(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _load(args):
    dill, hydra, np, torch, OmegaConf, open_dict = _dependencies()
    from EfficientFlow.common.equivariant_checkpoint import load_equivariant_state_dict
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    kwargs = {"map_location": "cpu", "pickle_module": dill}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    # Hash and load the same open file version. A running trainer can atomically
    # replace latest.ckpt; the manifest must identify the weights actually read.
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        payload = torch.load(stream, **kwargs)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(
                "Checkpoint changed while loading. Select an immutable epoch checkpoint "
                "with --checkpoint /absolute/path/epoch=....ckpt and rerun.")
    if not isinstance(payload, dict) or not {"cfg", "state_dicts"}.issubset(payload):
        raise ValueError("Expected a workspace checkpoint with cfg and state_dicts")
    cfg = OmegaConf.create(OmegaConf.to_container(payload["cfg"], resolve=False)
                           if OmegaConf.is_config(payload["cfg"]) else payload["cfg"])
    if OmegaConf.select(cfg, "policy._target_") != POLICY_TARGET:
        raise ValueError(
            f"This CLI only accepts {POLICY_TARGET}. First initialize and train the "
            "variance head with training.init_checkpoint and variance_training=head_only "
            "when starting from a published EfficientFlow checkpoint.")
    if args.dataset_path:
        dataset_path = str(Path(args.dataset_path).expanduser().resolve())
        with open_dict(cfg):
            cfg.dataset_path = dataset_path
            for node in ("task.dataset", "task.env_runner"):
                if OmegaConf.select(cfg, node) is not None:
                    OmegaConf.update(cfg, f"{node}.dataset_path", dataset_path)
    OmegaConf.resolve(cfg)
    if args.weights not in payload["state_dicts"]:
        raise KeyError(
            f"Checkpoint lacks {args.weights}; available: {list(payload['state_dicts'])}. "
            "Select the intended state explicitly with --weights model or --weights ema_model.")
    _seed_everything(0, np, torch)
    try:
        policy = hydra.utils.instantiate(cfg.policy)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing policy dependency ({exc}). Use the complete EfficientFlow environment.") from exc
    # escnn saves mode-derived matrix/filter buffers in eval checkpoints. The
    # shared loader discards only recognized derived caches, strictly checks all
    # real parameters/buffers, and refreshes caches from the loaded weights.
    load_equivariant_state_dict(policy, payload["state_dicts"][args.weights])
    policy.to(device).eval()
    if args.variance_scale is not None:
        policy.variance_scale = args.variance_scale
    if not len(policy.normalizer.params_dict):
        raise ValueError("The checkpoint has no saved normalizer; evaluation cannot refit it")
    policy._evaluation_checkpoint_sha256 = digest.hexdigest()
    return policy, cfg, checkpoint, (hydra, np, torch, OmegaConf, open_dict)


def _manifest(args, cfg, checkpoint, policy, runtime):
    _, np, torch, OmegaConf, _ = runtime
    device = torch.device(args.device)
    versions = {}
    for distribution in ("hydra-core", "omegaconf", "dill", "torchvision", "escnn",
                         "robomimic", "robosuite", "mujoco", "mimicgen", "h5py"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "schema_version": 1, "command": args.command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": policy._evaluation_checkpoint_sha256,
        "weights": args.weights, "policy_target": POLICY_TARGET,
        "variance_updates": int(policy._variance_updates.item()),
        "variance_training": policy.variance_training,
        "variance_scale": policy.variance_scale,
        "checkpoint_variance_scale": OmegaConf.select(cfg, "policy.variance_scale", default=1.),
        "residual_convention": "mean squared velocity residual over horizon and action coordinates",
        "args": vars(args), "resolved_cfg": OmegaConf.to_container(cfg, resolve=True),
        "resolved_runner_max_steps": OmegaConf.select(cfg, "task.env_runner.max_steps"),
        "runtime": {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "dependency_versions": versions,
        },
        "source_root": str(ROOT), "source_sha256": _source_hashes(),
        "evaluation_entry": str(Path(__file__).resolve()),
        "evaluation_entry_sha256": _sha256(__file__),
        "notes": [
            "Real checkpoint evaluation; no synthetic or substituted results.",
            "Choose eta and variance calibration on validation data; reserve independent test episodes.",
            "A newly initialized or uncalibrated variance head does not yield meaningful adaptive results. "
            "The update counter is provenance, not proof of calibration; inspect validation residual diagnostics.",
            "Seeded execution does not certify bitwise reproducibility across hardware or dependency versions.",
        ],
    }


def _numeric_metrics(log):
    metrics, skipped = {}, []
    for key, value in log.items():
        if isinstance(value, numbers.Real):
            number = float(value)
            if not math.isfinite(number):
                raise FloatingPointError(f"Runner returned nonfinite metric {key}={number}")
            metrics[str(key)] = number
        else:
            skipped.append(str(key))
    return metrics, skipped


def _close_runner(runner):
    if runner is None:
        return
    environment = getattr(runner, "env", None)
    if environment is not None and callable(getattr(environment, "close", None)):
        environment.close()
    elif callable(getattr(runner, "close", None)):
        runner.close()


def doctor(args):
    """Validate real weights and metadata without starting a simulator rollout."""
    policy, cfg, checkpoint, runtime = _load(args)
    hydra, _, _, OmegaConf, _ = runtime
    dataset = Path(str(OmegaConf.select(cfg, "task.env_runner.dataset_path"))).expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(
            f"Saved rollout dataset does not exist: {dataset}. Run from the project root "
            "or provide --dataset-path /absolute/path/kitchen_d1_abs.hdf5.")
    runner_class = hydra.utils.get_class(cfg.task.env_runner._target_)
    import robomimic.utils.file_utils as FileUtils
    environment = FileUtils.get_env_metadata_from_dataset(str(dataset))
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": policy._evaluation_checkpoint_sha256,
        "source_root": str(ROOT), "weights": args.weights,
        "task_name": OmegaConf.select(cfg, "task_name"),
        "training_seed": OmegaConf.select(cfg, "training.seed"),
        "dataset": str(dataset), "environment": environment.get("env_name"),
        "environment_reward_shaping": environment.get("env_kwargs", {}).get("reward_shaping"),
        "runner_file": inspect.getsourcefile(runner_class),
        "saved_candidates": policy.num_of_trajectories,
        "saved_sampler": policy.sampler, "variance_updates": int(policy._variance_updates.item()),
        "variance_scale": policy.variance_scale,
        "normalizer": "loaded from checkpoint, never refitted",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("DOCTOR_OK (weights, normalizer, imports and dataset metadata; rollout not started)", flush=True)


def _validate_rollout_metrics(metrics, n_test):
    required = ("test/mean_score", "sampling/nfe_mean", "sampling/one_step_fraction",
                "timing/policy_ms_p50", "timing/policy_ms_p95")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(
            f"Evaluation runner did not return required metrics: {missing}. "
            "Check that env_runner/robomimic_image_runner.py is the instrumented version "
            "used for EfficientAdaFlow training.")
    rewards = [value for key, value in metrics.items() if key.startswith("test/sim_max_reward_")]
    if len(rewards) != n_test:
        raise RuntimeError(f"Expected {n_test} test episode rewards, received {len(rewards)}")
    if not math.isclose(statistics.mean(rewards), metrics["test/mean_score"], abs_tol=1e-6):
        raise RuntimeError("test/mean_score disagrees with individual test episode rewards")


def _wilson_interval(successes, total):
    """Descriptive interval for binary episode outcomes, not training-seed variation."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1. + z * z / total
    center = (p + z * z / (2. * total)) / denominator
    radius = z * math.sqrt(p * (1. - p) / total + z * z / (4. * total * total)) / denominator
    return max(0., center - radius), min(1., center + radius)


def _summary_row(row):
    variant = row["variant"]
    metrics = row.get("metrics", {})
    result = {
        "variant": (f"fixed_{variant['fixed_steps']}" if variant["sampler"] == "fixed"
                    else f"adaptive_eta_{variant['eta']:g}"),
        "candidates": row["candidates"], "evaluation_seed": row["seed"],
        "test_start_seed": row["test_start_seed"], "n_test": row["n_test"],
        "status": row["status"], "test_mean_score": metrics.get("test/mean_score", ""),
    }
    for key in ("nfe_mean", "nfe_p95", "nfe_max", "one_step_fraction",
                "model_calls_per_prediction", "candidate_evaluations_per_prediction"):
        result[key] = metrics.get(f"sampling/{key}", "")
    for key in ("policy_ms_mean", "policy_ms_p50", "policy_ms_p95", "policy_batch_size"):
        result[key] = metrics.get(f"timing/{key}", "")
    rewards = [value for key, value in metrics.items() if key.startswith("test/sim_max_reward_")]
    binary = row["status"] == "ok" and len(rewards) == row["n_test"] and all(x in (0., 1.) for x in rewards)
    result.update({"binary_rewards": binary, "successes": "", "failures": "",
                   "episode_wilson95_low": "", "episode_wilson95_high": ""})
    if binary:
        successes = int(sum(rewards))
        lower, upper = _wilson_interval(successes, len(rewards))
        result.update({"successes": successes, "failures": len(rewards) - successes,
                       "episode_wilson95_low": lower, "episode_wilson95_high": upper})
    result["elapsed_seconds_including_environment_setup"] = row.get("elapsed_seconds_including_environment_setup", "")
    result["error"] = row.get("error", "")
    return result


def _write_sweep_summary(output, rows):
    """Keep readable, partial results after each real evaluated variant."""
    if not rows:
        return
    summaries = [_summary_row(row) for row in rows]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    def number(value, scale=1., suffix=""):
        return "—" if value == "" else f"{value * scale:.3f}{suffix}"

    lines = [
        "# EfficientAdaFlow same-checkpoint sampling comparison", "",
        "Each row uses the same saved policy/normalizer and changes sampling settings only. "
        "Evaluation seeds are rollout repeats of one trained model, not independent training seeds.", "",
        "| Sampler | Candidates | Eval seed | Episodes | Status | Test score | Successes | NFE mean | One-step % | P50 ms | P95 ms |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        count = "—" if row["successes"] == "" else f"{row['successes']}/{row['n_test']}"
        lines.append(
            f"| {row['variant']} | {row['candidates']} | {row['evaluation_seed']} | {row['n_test']} "
            f"| {row['status']} | {number(row['test_mean_score'])} | {count} "
            f"| {number(row['nfe_mean'])} | {number(row['one_step_fraction'], 100.)} "
            f"| {number(row['policy_ms_p50'])} | {number(row['policy_ms_p95'])} |")
    lines.extend([
        "", "Test score is the mean episode maximum reward. It is a success rate only when rewards "
        "are binary 0/1; the CSV records that check and per-run descriptive Wilson intervals.", "",
        "Compare fixed-1 against fixed-2/4/8 first: if extra fixed steps do not improve outcomes, "
        "this checkpoint may have little room for adaptive integration. If fixed multi-step improves "
        "but adaptive stays at one step, inspect eta and residual calibration. To support adaptive "
        "allocation, compare against fixed sampling at similar actual latency/computation.", "",
        "P50/P95 are per predict_action call at the recorded environment batch size. Timing includes "
        "the encoder, controller and candidate selection, excludes environment simulation and observation "
        "CPU-to-device transfer, and retains cold-start predictions. Candidate NFE is not batch network calls.", "",
        "Treat this parameter sweep as a diagnostic/validation experiment. Lock selected settings before "
        "running an independent final test set. Read manifest.json for checkpoint hash, source hashes, "
        "configuration and seed protocol. results.jsonl contains the individual episode rewards.", "",
    ])
    errors = [row for row in summaries if row["error"]]
    for row in errors:
        lines.extend([f"Failure in {row['variant']}: `{row['error']}`", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def sweep(args):
    policy, cfg, checkpoint, runtime = _load(args)
    hydra, np, torch, OmegaConf, open_dict = runtime
    output = _output_directory(args.output, args.overwrite)
    manifest = _manifest(args, cfg, checkpoint, policy, runtime)
    manifest["latency_protocol"] = (
        "Runner synchronizes CUDA around predict_action. Includes encoder, controller and "
        "selection; excludes observation CPU-to-device transfer and environment simulation. "
        "No prediction warmup is excluded; inspect cold-start influence before publication.")
    manifest["episode_seed_protocol"] = (
        "test_start_seed + evaluation_seed * n_test, followed by n_test consecutive seeds; "
        "each variant reseeds Python, NumPy and Torch to evaluation_seed.")
    manifest["status"] = "running"
    _write_json(output / "manifest.json", manifest)
    variants = ([{"sampler": "fixed", "fixed_steps": step} for step in args.fixed_steps]
                + [{"sampler": "adaptive", "eta": eta} for eta in args.etas])
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    successes = 0
    rows = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for candidate_count in args.candidates:
            for variant in variants:
                for seed in args.seeds:
                    row = {
                        "variant": variant, "candidates": candidate_count, "seed": seed,
                        "n_test": args.n_test, "weights": args.weights,
                        "test_start_seed": args.test_start_seed + seed * args.n_test,
                        "adaptive_max_steps": args.max_steps,
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    descriptor = (f"fixed{variant['fixed_steps']}" if variant["sampler"] == "fixed"
                                  else f"eta{variant['eta']:g}")
                    run_dir = output / "rollouts" / run_stamp / (
                        f"c{candidate_count}_{descriptor}_seed{seed}")
                    run_dir.mkdir(parents=True, exist_ok=False)
                    runner = None
                    started = time.perf_counter()
                    try:
                        _seed_everything(seed, np, torch)
                        policy.sampler = variant["sampler"]
                        policy.num_of_trajectories = candidate_count
                        policy.adaptive_max_steps = args.max_steps
                        if policy.sampler == "fixed":
                            policy.fixed_steps = variant["fixed_steps"]
                        else:
                            policy.adaptive_eta = variant["eta"]
                        policy.eval()
                        policy.reset()
                        policy.get_sampling_metrics(reset=True)
                        runner_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.env_runner))
                        with open_dict(runner_cfg):
                            runner_cfg.n_train = 0
                            runner_cfg.n_train_vis = 0
                            runner_cfg.n_test = args.n_test
                            runner_cfg.n_test_vis = min(args.n_test, args.n_videos) if args.video else 0
                            runner_cfg.test_start_seed = row["test_start_seed"]
                            runner_cfg.n_envs = min(args.n_envs, args.n_test)
                        row["resolved_runner"] = OmegaConf.to_container(runner_cfg, resolve=True)
                        row["padded_envs_in_final_batch"] = (
                            -args.n_test) % int(runner_cfg.n_envs)
                        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(run_dir))
                        # Constructor work may consume RNG; start policy noise from the
                        # same seed after constructing each environment batch.
                        _seed_everything(seed, np, torch)
                        with torch.no_grad():
                            runner_log = runner.run(policy)
                        row["metrics"], row["skipped_nonnumeric_metrics"] = _numeric_metrics(runner_log)
                        _validate_rollout_metrics(row["metrics"], args.n_test)
                        row["videos"] = [str(path) for path in sorted(run_dir.rglob("*.mp4"))]
                        row["status"] = "ok"
                        successes += 1
                    except BaseException as exc:
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["traceback"] = traceback.format_exc()
                        raise
                    finally:
                        try:
                            _close_runner(runner)
                        except Exception as exc:
                            row["close_error"] = f"{type(exc).__name__}: {exc}"
                        row["elapsed_seconds_including_environment_setup"] = time.perf_counter() - started
                        _append_jsonl(stream, row)
                        rows.append(row)
                        _write_sweep_summary(output, rows)
                        manifest["completed_variants"] = successes
                        manifest["last_status"] = row["status"]
                        manifest["status"] = "running" if row["status"] == "ok" else "failed"
                        _write_json(output / "manifest.json", manifest)
                    print(json.dumps({"variant": descriptor, "candidates": candidate_count,
                                      "seed": seed, "metrics": row["metrics"]}, ensure_ascii=False))
    manifest["status"] = "complete"
    _write_json(output / "manifest.json", manifest)
    print(f"Saved {successes} evaluated variants to {output / 'results.jsonl'}", flush=True)
    print(f"Readable report: {output / 'report.md'}", flush=True)


def _calibration_summary(records, np, bins):
    prediction = np.asarray([row["predicted_variance"] for row in records], dtype=np.float64)
    residual = np.asarray([row["residual_mse"] for row in records], dtype=np.float64)
    if not len(records) or not np.isfinite(prediction).all() or not np.isfinite(residual).all():
        raise ValueError("Calibration requires finite, nonempty predictions and residuals")

    def statistics(mask):
        predicted, observed = prediction[mask], residual[mask]
        return {"count": int(mask.sum()), "predicted_variance_mean": float(predicted.mean()),
                "residual_mse_mean": float(observed.mean()),
                "ratio_of_means": float(observed.mean() / predicted.mean()),
                "mean_residual_to_prediction_ratio": float((observed / predicted).mean()),
                "gaussian_nll_mean": float((.5 * (observed / predicted + np.log(predicted))).mean())}

    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, bins + 1)))
    grouped = []
    if len(edges) == 1:
        grouped.append({"lower_inclusive": float(edges[0]), "upper_inclusive": float(edges[0]),
                        **statistics(np.ones(len(records), dtype=bool))})
    else:
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (prediction >= low) & ((prediction <= high) if index == len(edges)-2
                                          else (prediction < high))
            if mask.any():
                grouped.append({"lower_inclusive": float(low), "upper": float(high),
                                "upper_inclusive": index == len(edges)-2, **statistics(mask)})
    return {"overall": statistics(np.ones(len(records), dtype=bool)),
            "quantile_bin_edges": edges.tolist(), "quantile_bins": grouped}


def calibrate(args):
    policy, cfg, checkpoint, runtime = _load(args)
    hydra, np, torch, OmegaConf, open_dict = runtime
    from itertools import islice
    from torch.utils.data import DataLoader
    from EfficientFlow.common.pytorch_util import dict_apply

    output = _output_directory(args.output, args.overwrite)
    manifest = _manifest(args, cfg, checkpoint, policy, runtime)
    manifest["status"] = "running"
    manifest["diagnostic"] = (
        "Conditional straight-path residual calibration on the saved validation split. "
        "The saved checkpoint normalizer is used without refitting. This measures residual "
        "calibration, not a certified ODE error bound, action confidence, or test success.")
    _write_json(output / "manifest.json", manifest)
    _seed_everything(args.seed, np, torch)
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
    with open_dict(dataset_cfg):
        # Loading diagnostics must not create caches beside the source dataset.
        dataset_cfg.use_cache = False
    records = []
    try:
        dataset = hydra.utils.instantiate(dataset_cfg)
        validation = dataset.get_validation_dataset()
        if len(validation) == 0:
            raise ValueError("Saved validation split is empty; cannot calibrate on training data implicitly")
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(validation, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=False, generator=generator)
        with (output / "calibration_samples.jsonl").open("w", encoding="utf-8") as stream:
            with torch.no_grad():
                for batch_index, batch in enumerate(islice(loader, args.max_batches)):
                    batch = dict_apply(batch, lambda value: value.to(args.device))
                    nobs = policy.normalizer.normalize(policy._observations(batch["obs"]))
                    target = policy.normalizer["action"].normalize(batch["action"])
                    if target.shape[1:] != (policy.horizon, policy.action_dim):
                        raise ValueError("Validation action shape disagrees with saved policy")
                    condition = policy.enc(nobs).reshape(target.shape[0], -1)
                    noise = torch.randn_like(target)
                    target_velocity = target - noise
                    queries = [None] if args.random_times else args.times
                    for query in queries:
                        times = (torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
                                 if query is None else torch.full((target.shape[0],), query,
                                                                 device=target.device, dtype=target.dtype))
                        xt = times[:, None, None] * target + (1-times[:, None, None]) * noise
                        velocity, log_variance = policy.diff(
                            xt, times * 99, global_cond=condition, return_variance=True,
                            detach_variance_features=True)
                        residual = (velocity - target_velocity).square().flatten(1).mean(1)
                        effective_variance = log_variance.clamp(*policy.log_variance_bounds).exp()
                        effective_variance = effective_variance * policy.variance_scale
                        if (not torch.isfinite(residual).all()
                                or not torch.isfinite(effective_variance).all()
                                or not torch.isfinite(log_variance).all()):
                            raise FloatingPointError("Nonfinite calibration prediction or target")
                        for position, (flow_time, pred, actual, raw_log) in enumerate(zip(
                                times.cpu().tolist(), effective_variance.cpu().tolist(),
                                residual.cpu().tolist(), log_variance.cpu().tolist())):
                            row = {"batch": batch_index, "batch_position": position,
                                   "time": flow_time, "predicted_variance": pred,
                                   "residual_mse": actual, "raw_log_variance": raw_log}
                            records.append(row)
                            _append_jsonl(stream, row)
        report = _calibration_summary(records, np, args.bins)
        report["by_time"] = [] if args.random_times else [
            {"time": query, **_calibration_summary(
                [row for row in records if abs(row["time"]-query) < 1e-6], np, args.bins)}
            for query in args.times]
        report["validation_sequences_available"] = len(validation)
        report["validation_split_seed"] = OmegaConf.select(cfg, "task.dataset.seed")
        report["validation_split_ratio"] = OmegaConf.select(cfg, "task.dataset.val_ratio")
        report["normalizer_source"] = "saved checkpoint; never refitted"
        report["prediction_includes_variance_scale"] = policy.variance_scale
        report["interpretation"] = (
            "ratio_of_means > 1 indicates underpredicted residual energy on these validation "
            "queries. It can motivate a held-out calibration experiment, not a guaranteed "
            "inference correction. Select eta on validation rollouts; do not tune on test results.")
        _write_json(output / "calibration.json", report)
        manifest["status"] = "complete"
        manifest["calibration_queries"] = len(records)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json(output / "manifest.json", manifest)
    print(f"Saved validation calibration for {len(records)} queries to {output / 'calibration.json'}")


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError("must be an integer in [0, 2**32)")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _flow_time(value):
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return parsed


def _checkpoint_arguments(parser):
    parser.add_argument("--checkpoint", required=True, help="Real EfficientAdaFlow workspace .ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema_model", "model"), default="ema_model",
                        help="Explicit state_dict selection; no silent EMA fallback (default: ema_model)")
    parser.add_argument("--dataset-path", help="Optional local dataset relocation, without normalizer refitting")
    parser.add_argument("--variance-scale", type=_positive_float, default=None,
                        help="Override the saved variance multiplier for held-out calibration experiments")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checker = commands.add_parser("doctor", help="Validate real weights, imports, normalizer and dataset metadata")
    _checkpoint_arguments(checker)
    checker.set_defaults(handler=doctor)
    shared = argparse.ArgumentParser(add_help=False)
    _checkpoint_arguments(shared)
    shared.add_argument("--output", required=True, help="New or empty report directory")
    shared.add_argument("--overwrite", action="store_true", help="Replace JSON reports in an existing output")
    evaluator = commands.add_parser("sweep", parents=[shared], help="Run fixed/adaptive evaluation rollouts")
    evaluator.add_argument("--etas", nargs="+", type=_positive_float, default=[.05, .1, .2])
    evaluator.add_argument("--fixed-steps", nargs="+", type=_positive_int, default=[1, 2, 3, 5, 8])
    evaluator.add_argument("--seeds", nargs="+", type=_nonnegative_int, default=[0, 1, 2])
    evaluator.add_argument("--candidates", nargs="+", type=_positive_int, default=[1, 5])
    evaluator.add_argument("--max-steps", type=_positive_int, default=8, help="Adaptive NFE cap (not episode horizon)")
    evaluator.add_argument("--n-test", type=_positive_int, default=50)
    evaluator.add_argument("--n-envs", type=_positive_int, default=1)
    evaluator.add_argument("--test-start-seed", type=_nonnegative_int, default=100000)
    evaluator.add_argument("--video", action="store_true", help="Record videos (default: off)")
    evaluator.add_argument("--n-videos", type=_positive_int, default=1)
    evaluator.set_defaults(handler=sweep)
    diagnostic = commands.add_parser("calibrate", parents=[shared], help="Diagnose variance on the saved validation split")
    diagnostic.add_argument("--max-batches", type=_positive_int, default=20)
    diagnostic.add_argument("--batch-size", type=_positive_int, default=32)
    diagnostic.add_argument("--num-workers", type=_nonnegative_int, default=0)
    diagnostic.add_argument("--seed", type=_nonnegative_int, default=42)
    diagnostic.add_argument("--times", nargs="+", type=_flow_time, default=[0., .25, .5, .75, .99])
    diagnostic.add_argument("--random-times", action="store_true", help="Uniform random time per sequence instead of --times")
    diagnostic.add_argument("--bins", type=_positive_int, default=10)
    diagnostic.set_defaults(handler=calibrate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # A function is not a useful provenance value and cannot be JSON encoded.
    handler = args.handler
    del args.handler
    try:
        if args.command == "sweep":
            for name in ("etas", "fixed_steps", "seeds", "candidates"):
                values = getattr(args, name)
                if len(values) != len(set(values)):
                    raise ValueError(f"Duplicate values in --{name.replace('_', '-')}")
            if args.test_start_seed + max(args.seeds) * args.n_test + args.n_test > 2**32:
                raise ValueError("The requested episode seeds exceed the NumPy seed range")
        elif args.command == "calibrate" and len(args.times) != len(set(args.times)):
            raise ValueError("Duplicate values in --times")
        handler(args)
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
