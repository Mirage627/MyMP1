"""Command-line entry point; all heavy simulation imports are lazy."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .analysis import summarize, write_report
from .backends import ToyBackend, RobomimicBackend
from .core import (collect_branches, isolated_policy_rng, make_noise, predict,
                   read_jsonl, run_fixed, stack_history, stable_seed)


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=str)+"\n", encoding="utf-8")


def add_backend_options(parser):
    parser.add_argument("--backend", choices=("toy", "robomimic"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", choices=("auto", "model", "ema_model"), default=None)
    parser.add_argument("--policy-target", help="Explicit Hydra policy target override")
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--register-module", action="append", default=None)
    parser.add_argument("--override", action="append", default=None, help="Checkpoint OmegaConf key=value")


def build_parser():
    parser = argparse.ArgumentParser(description="Paired MeanFlow go/no-go experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check dependencies and checkpoint compatibility")
    add_backend_options(doctor)
    locate = sub.add_parser("find-checkpoint", help="Print newest matching MP1 latest.ckpt from Hydra run metadata")
    locate.add_argument("--root", default="data/outputs")
    locate.add_argument("--task", required=True)
    locate.add_argument("--training-seed", type=int, required=True)
    locate.add_argument("--n-demo", type=int, required=True)
    run = sub.add_parser("run", help="Fixed rollouts, replay-verified branches, or deployed router")
    add_backend_options(run)
    run.add_argument("--output", required=True)
    run.add_argument("--phase", choices=("all", "fixed", "branches", "router"), default="all")
    run.add_argument("--episodes", type=int, default=None)
    run.add_argument("--seed-start", type=int, default=None)
    run.add_argument("--trials", type=int, default=None, help="Latent-noise trials per environment seed")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--steps", nargs="+", type=int, default=None)
    run.add_argument("--schedules", nargs="+", choices=("uniform", "noise_first", "data_first"))
    run.add_argument("--power", type=float, default=None)
    run.add_argument("--grid", action="append", default=None, help="Custom family=1,.7,0; adds family_1 too")
    run.add_argument("--max-steps", type=int, default=None, help="Primitive physical steps, same for all variants")
    run.add_argument("--anchor-every", type=int, default=None, help="Reference-policy decision interval")
    run.add_argument("--max-anchors", type=int, default=None)
    run.add_argument("--verify-repeats", type=int, default=None)
    run.add_argument("--continuation-nfe", type=int, choices=(1, 2, 4), default=None,
                     help="Common FUTURE policy for every current-chunk branch; default 1")
    run.add_argument("--warmup", type=int, default=None)
    run.add_argument("--variance-probe")
    run.add_argument("--router")
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze = sub.add_parser("analyze", help="Rebuild reports from existing JSONL")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=0)
    router = sub.add_parser("fit-router", help="Fit a grouped-heldout cheap benefit predictor")
    router.add_argument("--input", required=True)
    router.add_argument("--output", required=True)
    router.add_argument("--family", choices=("uniform", "noise_first", "data_first"), default="uniform")
    router.add_argument("--high-nfe", type=int, default=2)
    router.add_argument("--target-mean-nfe", type=float, default=1.5)
    router.add_argument("--train-fraction", type=float, default=.7)
    router.add_argument("--ridge", type=float, default=1.)
    router.add_argument("--seed", type=int, default=0)
    variance = sub.add_parser("fit-variance", help="Train a separate AdaFlow-style diagonal residual proxy")
    add_backend_options(variance)
    variance.add_argument("--output", required=True)
    variance.add_argument("--batches", type=int, default=200)
    variance.add_argument("--batch-size", type=int, default=64)
    variance.add_argument("--lr", type=float, default=1e-3)
    variance.add_argument("--seed", type=int, default=0)
    return parser


def create_backend(args):
    backend = args.backend or "robomimic"
    if backend == "toy":
        if args.checkpoint:
            raise ValueError("--backend toy does not load a real checkpoint")
        return ToyBackend()
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --backend robomimic")
    return RobomimicBackend(args.checkpoint, device=args.device or "cuda:0",
        dataset_path=args.dataset_path, policy_target=args.policy_target,
        weights=args.weights or "auto", execute_steps=args.execute_steps,
        register_modules=args.register_module or (), overrides=args.override or ())


def doctor(args):
    deps = ("numpy", "torch", "hydra", "omegaconf", "dill", "robomimic", "robosuite", "gym", "pytorch3d")
    result = {"python": sys.executable, "python_version": platform.python_version(),
              "checkout": str(ROOT), "dependencies": {name: importlib.util.find_spec(name) is not None for name in deps},
              "source_tree": getattr(sys.modules.get("EfficientFlow"), "_meanflow_source_root", str(ROOT))}
    result["existing_conflict_markers"] = [name for name in ("train.py", "conda_environment.yaml")
        if (ROOT/name).exists() and "<<<<<<<" in (ROOT/name).read_text(encoding="utf-8")]
    if args.checkpoint:
        backend = create_backend(args)
        try:
            result["checkpoint"] = backend.describe()
        finally:
            backend.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def _variants(args):
    from EfficientFlow.common.meanflow_sampling import make_time_grid, validate_time_grid
    if 1 not in args.steps:
        raise ValueError("Include --steps 1 for the paired reference")
    if len(args.steps) != len(set(args.steps)) or len(args.schedules) != len(set(args.schedules)):
        raise ValueError("Duplicate steps or schedules")
    variants = {f"{family}_{k}": make_time_grid(k, family, args.power)
                for family in args.schedules for k in args.steps}
    variants.setdefault("uniform_1", (1.0, 0.0))
    for spec in args.grid or ():
        name, numbers = spec.split("=", 1)
        if not name or not all(char.isalnum() or char in "_-" for char in name):
            raise ValueError("Custom grid family must use letters, digits, underscore or hyphen")
        grid = validate_time_grid([float(number) for number in numbers.split(",")])
        key = f"{name}_{len(grid)-1}"
        if key in variants:
            raise ValueError(f"Duplicate custom grid: {key}")
        variants[key] = grid
        variants.setdefault(f"{name}_1", (1.0, 0.0))
    return variants


def _defaults(args, manifest=None):
    previous = (manifest or {}).get("arguments", {})
    defaults = {"backend": "robomimic", "device": "cuda:0", "weights": "auto",
                "episodes": 50, "seed_start": 100000, "trials": 1, "seed": 0,
                "steps": [1, 2, 4], "schedules": ["uniform", "noise_first", "data_first"],
                "power": 2., "max_steps": None, "anchor_every": 5, "max_anchors": 3,
                "verify_repeats": 1, "warmup": 3, "register_module": [], "override": [], "grid": []}
    defaults["continuation_nfe"] = 1
    for name in ("checkpoint", "dataset_path", "policy_target", "execute_steps", "variance_probe"):
        if getattr(args, name, None) is None and previous.get(name) is not None:
            setattr(args, name, previous[name])
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, previous.get(name, value))
    for name in ("episodes", "trials", "max_steps", "anchor_every", "max_anchors", "verify_repeats"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")


def analyze_folder(folder, samples=2000, seed=0):
    folder = Path(folder)
    fixed = read_jsonl(folder/"fixed_rollouts.jsonl")
    branches = read_jsonl(folder/"branches.jsonl")
    router = read_jsonl(folder/"router_rollouts.jsonl")
    if not fixed and not branches and not router:
        raise ValueError(f"No experiment rows in {folder}")
    # Deployed router rollouts are separate from its training-branch diagnostics.
    summary = summarize(fixed + router, branches, bootstrap_samples=samples, seed=seed)
    paths = write_report(summary, folder)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return summary


def _bind_artifact(path, source_manifest):
    manifest = {"source_manifest": str(source_manifest),
                "checkpoint_sha256": None, "source_episode_seed_start": None,
                "source_episode_count": None}
    if Path(source_manifest).is_file():
        original = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
        manifest.update({"checkpoint_sha256": original.get("checkpoint_sha256"),
            "source_episode_seed_start": original.get("arguments", {}).get("seed_start"),
            "source_episode_count": original.get("arguments", {}).get("episodes"),
            "source_sampling": {"power": original.get("arguments", {}).get("power"),
                "variants": original.get("variants"),
                "execute_steps": original.get("backend", {}).get("execute_steps"),
                "horizon": original.get("backend", {}).get("horizon"),
                "n_obs_steps": original.get("backend", {}).get("n_obs_steps"),
                "weights": original.get("backend", {}).get("weights"),
                "continuation_nfe": original.get("branch_configuration", {}).get("continuation_nfe", 1)}})
    save_json(str(path)+".manifest.json", manifest)


def run_experiment(args):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    _defaults(args, manifest)
    for name in ("checkpoint", "dataset_path", "variance_probe", "router"):
        if getattr(args, name, None):
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    args.output = str(output)
    variants = _variants(args)
    if args.phase in ("all", "fixed", "router") and any((output/name).exists() for name in
            ("fixed_rollouts.jsonl", "branches.jsonl", "anchors.jsonl", "router_rollouts.jsonl")):
        raise FileExistsError("Use a new --output directory for new rollouts; existing rows are never overwritten/appended twice")
    if args.phase == "branches":
        if manifest is None or not (output/"anchors.jsonl").is_file():
            raise ValueError("Run --phase fixed first in this output directory")
        if (output/"branches.jsonl").exists():
            raise FileExistsError("branches.jsonl already exists; use a new output for a repeat experiment")
        old = manifest["arguments"]
        invariants = ("checkpoint", "dataset_path", "backend", "weights", "execute_steps", "max_steps",
                      "seed", "steps", "schedules", "power", "grid", "override", "policy_target")
        for key in invariants:
            if old.get(key) != getattr(args, key, None):
                raise ValueError(f"--phase branches must preserve {key}: {old.get(key)!r}")
    backend = create_backend(args)
    try:
        if args.max_steps is None:
            args.max_steps = 30 if args.backend == "toy" else int(backend.cfg.task.env_runner.max_steps)
        checksum = file_hash(args.checkpoint) if args.checkpoint else None
        if manifest and manifest.get("checkpoint_sha256") != checksum:
            raise ValueError("Checkpoint contents changed since fixed rollouts")
        if manifest and args.backend != "toy":
            current = backend.describe()
            for key in ("policy_source", "policy_source_sha256"):
                previous = manifest.get("backend", {}).get(key)
                if previous is not None and previous != current.get(key):
                    raise ValueError(f"Policy source changed since fixed rollouts: {key}")
        if manifest is None:
            manifest = {"schema": "meanflow_go_nogo_v1", "arguments": vars(args).copy(),
                "checkpoint_sha256": checksum, "variants": variants,
                "backend": backend.describe(), "python": sys.executable,
                "notes": ["Serial batch=1 synchronized end-to-end predict_action latency; GPU contention is uncontrolled.",
                    "Physical control frequency/horizon is fixed; measured compute latency is NOT injected into simulator time.",
                    "Success means task success reached at any executed primitive step; stop on success.",
                    "Anchors come only from uniform_1; branch current-chunk changes use the same configured future policy."]}
            save_json(manifest_path, manifest)
        # Warm each interval budget before measuring (no environment transitions).
        observation = backend.reset(args.seed_start)
        from collections import deque
        warm_obs = stack_history(deque([observation]), backend.n_obs_steps)
        warm_noise = make_noise(backend, "warmup", 0, args.seed)
        for grid in variants.values():
            for iteration in range(args.warmup):
                predict(backend, warm_obs, warm_noise, grid, stable_seed(args.seed, "warmup", iteration))
        probe = None
        if args.variance_probe:
            if args.backend == "toy":
                raise ValueError("variance_proxy requires a real supported MeanFlow policy")
            from .variance import VarianceProbe
            binding_path = Path(str(args.variance_probe)+".manifest.json")
            if not binding_path.is_file():
                raise FileNotFoundError("variance_proxy requires its CLI-created .manifest.json binding")
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            if binding.get("checkpoint_sha256") != checksum or binding.get("weights") != backend.weight_key:
                raise ValueError("variance_proxy was bound to different checkpoint weights")
            probe = VarianceProbe(args.variance_probe, device=args.device)
        seeds = list(range(args.seed_start, args.seed_start+args.episodes))
        trials = list(range(args.trials))
        if args.phase in ("all", "fixed"):
            run_fixed(backend, variants, output, env_seeds=seeds, trials=trials,
                base_seed=args.seed, max_steps=args.max_steps,
                anchor_every=args.anchor_every, max_anchors=args.max_anchors)
        if args.phase in ("all", "branches"):
            anchors = read_jsonl(output/"anchors.jsonl")
            collect_branches(backend, anchors, variants, output/"branches.jsonl",
                base_seed=args.seed, max_steps=args.max_steps,
                verify_repeats=args.verify_repeats, variance_probe=probe,
                continuation_grid=tuple(np.linspace(1., 0., args.continuation_nfe+1)))
            manifest["branch_configuration"] = {
                "continuation_nfe": args.continuation_nfe,
                "variance_probe": args.variance_probe,
                "verify_repeats": args.verify_repeats}
        if args.phase == "router":
            if not args.router:
                raise ValueError("--router is required for --phase router")
            from .router import BenefitRouter
            from EfficientFlow.common.meanflow_sampling import make_time_grid
            router = BenefitRouter(args.router)
            binding_path = Path(str(args.router)+".manifest.json")
            if not binding_path.is_file():
                raise FileNotFoundError("Router requires its CLI-created .manifest.json binding")
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            if binding.get("checkpoint_sha256") != checksum:
                raise ValueError("Router was trained with a different checkpoint")
            source_sampling = binding.get("source_sampling", {})
            for field in ("execute_steps", "horizon", "n_obs_steps"):
                if source_sampling.get(field) != getattr(backend, field):
                    raise ValueError(f"Router deployment must preserve source {field}")
            if source_sampling.get("weights") != backend.describe().get("weights"):
                raise ValueError("Router deployment uses different model/EMA weights")
            if source_sampling.get("power") != args.power:
                raise ValueError("Router deployment must preserve --power from the source experiment")
            start, count = binding.get("source_episode_seed_start"), binding.get("source_episode_count")
            if start is not None and count is not None and set(seeds) & set(range(start, start+count)):
                raise ValueError("Deployed-router evaluation seeds overlap training/diagnostic seeds; choose --seed-start 200000")
            source_grids = source_sampling["variants"]
            controls = {f"{router.family}_{k}": tuple(source_grids[f"{router.family}_{k}"])
                        for k in (1, router.high_nfe)}
            def choose(obs, noise, features, feature_names):
                if feature_names != router.feature_names:
                    raise ValueError("Router feature schema differs from current policy observations")
                return controls[f"{router.family}_{router.choose(features)}"]
            # Collect fixed baselines on the exact NEW environment/noise pairs.
            run_fixed(backend, controls, output, env_seeds=seeds, trials=trials,
                base_seed=args.seed, max_steps=args.max_steps, anchor_every=args.anchor_every,
                max_anchors=args.max_anchors, capture_anchors=False)
            run_fixed(backend, {"benefit_router": (1., 0.)}, output, env_seeds=seeds,
                trials=trials, base_seed=args.seed, max_steps=args.max_steps,
                anchor_every=args.anchor_every, max_anchors=args.max_anchors,
                capture_anchors=False, chooser=choose)
            fraction = (router.metadata["target_mean_nfe"]-1)/(router.high_nfe-1)
            def random_choose(obs, noise, features, feature_names):
                k = router.high_nfe if np.random.random() < fraction else 1
                return controls[f"{router.family}_{k}"]
            run_fixed(backend, {"random_budget": (1., 0.)}, output, env_seeds=seeds,
                trials=trials, base_seed=args.seed, max_steps=args.max_steps,
                anchor_every=args.anchor_every, max_anchors=args.max_anchors,
                capture_anchors=False, chooser=random_choose)
            manifest["random_router_upgrade_probability"] = fraction
            manifest["notes"].append("Random router uses the same TRAINING target budget; check achieved costs on the new distribution.")
        manifest["backend"] = backend.describe()
        save_json(manifest_path, manifest)
        analyze_folder(output, args.bootstrap_samples, args.seed)
    finally:
        backend.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            doctor(args)
        elif args.command == "find-checkpoint":
            from .checkpoints import find_checkpoint
            print(find_checkpoint(args.root, task=args.task,
                training_seed=args.training_seed, n_demo=args.n_demo))
        elif args.command == "run":
            run_experiment(args)
        elif args.command == "analyze":
            analyze_folder(args.input, args.bootstrap_samples, args.seed)
        elif args.command == "fit-router":
            from .router import fit_router
            payload = fit_router(read_jsonl(Path(args.input)/"branches.jsonl"), args.output,
                family=args.family, high_nfe=args.high_nfe, target_mean_nfe=args.target_mean_nfe,
                seed=args.seed, train_fraction=args.train_fraction, ridge=args.ridge)
            _bind_artifact(args.output, Path(args.input)/"manifest.json")
            print(json.dumps(payload["heldout"], indent=2))
        elif args.command == "fit-variance":
            from .variance import train_variance_probe
            backend = create_backend(args)
            try:
                if not isinstance(backend, RobomimicBackend):
                    raise ValueError("fit-variance requires --backend robomimic and a checkpoint")
                metadata = train_variance_probe(backend.policy, backend.cfg, args.output,
                    device=args.device or "cuda:0", batches=args.batches,
                    batch_size=args.batch_size, lr=args.lr, seed=args.seed)
                save_json(str(args.output)+".manifest.json", {
                    "checkpoint_sha256": file_hash(args.checkpoint),
                    "weights": backend.weight_key, "metadata": metadata})
                print(json.dumps(metadata, ensure_ascii=False, indent=2))
            finally:
                backend.close()
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, KeyError, RuntimeError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
