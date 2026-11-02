"""Run a same-checkpoint Kitchen sampling ablation on the training server.

Upload this launcher to /home/YOUR_USER/PROJECT_EfficientFlow/. It accepts both
an outer project containing EfficientFlow/ and a flat EfficientFlow checkout.
Use --dry-run to inspect the plan without importing PyTorch or robotics.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


DEFAULT_CHECKPOINT = (
    "/home/YOUR_USER/PROJECT_EfficientFlow/data/outputs/2026.10.23/"
    "00.36.16_EfficientAdaFlow_demo200_kitchen_d1/checkpoints"
)
PRESETS = {
    "smoke": {"n_test": 2, "seeds": [0], "fixed_steps": [1], "etas": [0.1]},
    "pilot": {"n_test": 50, "seeds": [0], "fixed_steps": [1, 2, 4, 8],
              "etas": [0.1, 0.05, 0.02, 0.01]},
}
REQUIRED_SOURCE = (
    "__init__.py", "policy/EfficientAdaFlow.py", "common/adaptive_flow.py",
    "common/equivariant_checkpoint.py", "common/train_entry.py",
)


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _seed(value):
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError("must be an integer in [0, 2**32)")
    return parsed


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(PRESETS), default="smoke",
                        help="smoke: 4 episodes; pilot: 400 episodes with default grids")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint file or directory; directory selection never uses score")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent),
                        help="Outer training project; subprocess working directory (default: launcher folder)")
    parser.add_argument("--dataset-path", help="Optional relocation; otherwise keep saved dataset path")
    parser.add_argument("--output", help="Parent for a new timestamped result folder (default: <project>/data/diagnostics)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema_model", "model"), default="ema_model",
                        help="Explicit checkpoint state; no fallback when absent")
    parser.add_argument("--n-envs", type=_positive_int, default=1)
    parser.add_argument("--candidates", type=_positive_int, nargs="+", default=[5])
    parser.add_argument("--n-test", type=_positive_int)
    parser.add_argument("--seeds", type=_seed, nargs="+")
    parser.add_argument("--fixed-steps", type=_positive_int, nargs="+")
    parser.add_argument("--etas", type=_positive_float, nargs="+")
    parser.add_argument("--test-start-seed", type=_seed, default=200000,
                        help="New diagnostic episode seed range, separate from training's 100000")
    parser.add_argument("--max-steps", type=_positive_int, default=8,
                        help="Adaptive integration cap; does not change environment episode horizon")
    parser.add_argument("--variance-scale", type=_positive_float,
                        help="Optional explicit override; default keeps checkpoint scale, without calibration")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate file layout and print plan; no model load, rollouts or output writes")
    return parser


def _relative_to_project(value, project_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def select_checkpoint(path):
    """Prefer latest, otherwise latest named epoch; reject ambiguous selections."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".ckpt":
            raise ValueError(f"Checkpoint must be a .ckpt file: {path}")
        return path.resolve(), "explicit file"
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint file/directory not found: {path}")
    latest = path / "latest.ckpt"
    if latest.is_file():
        return latest.resolve(), "latest.ckpt (preferred; no score selection)"
    files = sorted(item for item in path.iterdir()
                   if item.is_file() and item.suffix.lower() == ".ckpt")
    if not files:
        raise FileNotFoundError(f"No .ckpt files found in {path}")
    if len(files) == 1:
        return files[0].resolve(), "only .ckpt file"
    by_epoch = []
    for item in files:
        match = re.search(r"(?:^|[-_])epoch[=_-]?(\d+)(?=$|[-_.])", item.stem,
                          flags=re.IGNORECASE)
        if match:
            by_epoch.append((int(match.group(1)), item))
    if not by_epoch:
        raise ValueError(
            f"Multiple checkpoints without recognizable epoch numbers in {path}; "
            "pass --checkpoint with an explicit file path.")
    highest_epoch = max(epoch for epoch, _ in by_epoch)
    matches = [item for epoch, item in by_epoch if epoch == highest_epoch]
    if len(matches) != 1:
        names = ", ".join(item.name for item in matches)
        raise ValueError(
            f"Multiple checkpoints at highest filename epoch {highest_epoch}: {names}. "
            "Pass --checkpoint with an explicit file path; scores are not used to choose.")
    return matches[0].resolve(), f"highest filename epoch {highest_epoch} (no score selection)"


def find_evaluator(project_root):
    # Match the evaluator's own source-root precedence before looking for its
    # entry point. An old outer scripts/ entry must not shadow a new inner one.
    source_root = next((root for root in (project_root, project_root / "EfficientFlow")
                        if all((root / name).is_file() for name in REQUIRED_SOURCE[:3])), None)
    if source_root is None:
        raise FileNotFoundError(
            f"No complete EfficientFlow source package at {project_root} or "
            f"{project_root / 'EfficientFlow'}. Upload the diagnostic bundle "
            "without replacing trained model source.")
    missing = [name for name in REQUIRED_SOURCE if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required source files missing from {source_root}: {', '.join(missing)}. "
            "Upload the matching diagnostic dependencies before running.")
    candidates = tuple(dict.fromkeys((
        source_root / "scripts/efficientadaflow_experiment.py",
        project_root / "scripts/efficientadaflow_experiment.py",
    )))
    for evaluator in candidates:
        if evaluator.is_file():
            return evaluator, source_root
    raise FileNotFoundError(
        "Evaluator not found. Expected " + " or ".join(str(item) for item in candidates))


def make_plan(args):
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    evaluator, source_root = find_evaluator(project_root)
    checkpoint, selection = select_checkpoint(_relative_to_project(args.checkpoint, project_root))
    preset = PRESETS[args.mode]
    grid = {name: getattr(args, name) if getattr(args, name) is not None else preset[name]
            for name in ("n_test", "seeds", "fixed_steps", "etas")}
    for name in ("seeds", "fixed_steps", "etas"):
        if len(grid[name]) != len(set(grid[name])):
            raise ValueError(f"Duplicate values in --{name.replace('_', '-')}")
    if len(args.candidates) != len(set(args.candidates)):
        raise ValueError("Duplicate values in --candidates")
    if args.test_start_seed + (max(grid["seeds"]) + 1) * grid["n_test"] > 2**32:
        raise ValueError("The requested episode seeds exceed the NumPy seed range")
    output_parent = (_relative_to_project(args.output, project_root) if args.output
                     else project_root / "data/diagnostics")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output_parent / f"kitchen_adaflow_{args.mode}_{timestamp}"
    shared = ["--checkpoint", str(checkpoint), "--device", args.device,
              "--weights", args.weights]
    if args.dataset_path:
        shared += ["--dataset-path", str(_relative_to_project(args.dataset_path, project_root))]
    if args.variance_scale is not None:
        shared += ["--variance-scale", str(args.variance_scale)]
    prefix = [sys.executable, "-u", str(evaluator)]
    doctor = prefix + ["doctor"] + shared
    sweep = prefix + ["sweep"] + shared + [
        "--output", str(output), "--n-test", str(grid["n_test"]),
        "--n-envs", str(args.n_envs), "--test-start-seed", str(args.test_start_seed),
        "--max-steps", str(args.max_steps),
    ]
    for option, values in (("seeds", grid["seeds"]), ("fixed-steps", grid["fixed_steps"]),
                           ("etas", grid["etas"]), ("candidates", args.candidates)):
        sweep += ["--" + option] + [str(value) for value in values]
    if args.video:
        sweep += ["--video"]
    variants = (len(grid["fixed_steps"]) + len(grid["etas"])) * len(args.candidates)
    return {
        "project_root": project_root, "source_root": source_root, "evaluator": evaluator,
        "checkpoint": checkpoint, "selection": selection, "output": output,
        "grid": grid, "variants": variants,
        "episodes": variants * grid["n_test"] * len(grid["seeds"]),
        "doctor": doctor, "sweep": sweep,
    }


def print_plan(args, plan):
    print(f"Mode: {args.mode}")
    print(f"Project root / subprocess cwd: {plan['project_root']}")
    print(f"Evaluator: {plan['evaluator']}")
    print(f"Source root: {plan['source_root']}")
    print(f"Checkpoint: {plan['checkpoint']}")
    print(f"Checkpoint selection: {plan['selection']}")
    print(f"Weights: {args.weights} (strict; no fallback)")
    print(f"Output: {plan['output']}")
    print(f"Dataset: {args.dataset_path or 'saved checkpoint path (no override)'}")
    print(f"Candidates per prediction: {args.candidates}")
    print(f"Fixed steps: {plan['grid']['fixed_steps']}; adaptive etas: {plan['grid']['etas']}")
    print(f"RNG seeds: {plan['grid']['seeds']}; test-start-seed: {args.test_start_seed}")
    print(f"Rollout budget: {plan['variants']} sampling/candidate variants x "
          f"{len(plan['grid']['seeds'])} RNG seeds x {plan['grid']['n_test']} episodes "
          f"= {plan['episodes']} episodes")
    print("No variance calibration is run; saved normalizer and scale are preserved unless overridden.")
    print("Doctor command: " + shlex.join(plan["doctor"]))
    print("Sweep command: " + shlex.join(plan["sweep"]))
    sys.stdout.flush()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = make_plan(args)
        print_plan(args, plan)
        if args.dry_run:
            print("Dry run complete; no checkpoint loaded and no output written.")
            return 0
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        print("Loading and checking the checkpoint before rollouts...", flush=True)
        subprocess.run(plan["doctor"], cwd=str(plan["project_root"]), check=True)
        print("Checkpoint check passed. Starting sampling ablation...", flush=True)
        subprocess.run(plan["sweep"], cwd=str(plan["project_root"]), check=True)
        print(f"Finished. Results: {plan['output']}")
    except subprocess.CalledProcessError as exc:
        print(f"error: evaluation subprocess failed (exit {exc.returncode}); "
              "see the error above. No other checkpoint or weights were substituted.", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed evaluator records remain in the printed output folder.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
