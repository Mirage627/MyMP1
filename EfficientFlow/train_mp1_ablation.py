"""Independent entry for the six controlled MP1 / MGAD / H ablations.

``--verify`` resolves configuration and checks source files without constructing
the model, demonstration dataset, or simulator. Flat and nested source layouts
are supported without importing the legacy train.py or MP1 workspace.
"""

import ast
import importlib.util
from pathlib import Path
import sys


ENTRY_DIR = Path(__file__).resolve().parent
MP1_ABLATION_WORKSPACE = (
    "EfficientFlow.workspace.train_mp1_ablation_workspace.TrainMP1AblationWorkspace")
ABLATIONS = {
    "MP1_baseline": "EfficientFlow.policy.mp1_baseline.Policy",
    "MP1_mgad": "EfficientFlow.policy.mp1_mgad.Policy",
    "MP1_diag": "EfficientFlow.policy.mp1_diag.Policy",
    "MP1_full": "EfficientFlow.policy.mp1_full.Policy",
    "MP1_mgad_diag": "EfficientFlow.policy.mp1_mgad_diag.Policy",
    "MP1_mgad_full": "EfficientFlow.policy.mp1_mgad_full.Policy",
}


def _source_root(entry_dir):
    """Find one complete package; retain cwd for relative dataset paths."""
    candidates = (entry_dir, entry_dir / "EfficientFlow")
    for candidate in candidates:
        if ((candidate / "__init__.py").is_file()
                and all((candidate / name).is_dir()
                        for name in ("config", "policy", "workspace"))):
            return candidate
    checked = "\n".join(f"  {candidate}" for candidate in candidates)
    raise SystemExit(
        "Cannot locate the EfficientFlow source package. Checked:\n"
        f"{checked}\nExpected __init__.py plus config/, policy/ and workspace/. "
        "Synchronize the complete source package instead of adding an empty launcher package.")


ROOT = _source_root(ENTRY_DIR)
if not (ROOT / "common" / "train_entry.py").is_file():
    raise SystemExit(f"Missing {ROOT / 'common' / 'train_entry.py'}; synchronize common/ first.")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "EfficientFlow", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["EfficientFlow"] = package
spec.loader.exec_module(package)

import hydra
from omegaconf import OmegaConf
from EfficientFlow.common.train_entry import register_resolvers, run_workspace


def _target_source(target):
    if not isinstance(target, str) or not target.startswith("EfficientFlow."):
        raise ValueError(f"Expected a source-package target, received {target!r}")
    module, symbol = target.rsplit(".", 1)
    relative = module[len("EfficientFlow."):].replace(".", "/") + ".py"
    return relative, symbol


def _validate_ablation_configuration(cfg):
    """Check targets statically so verification needs no robot dependencies."""
    n_demo = OmegaConf.select(cfg, "n_demo")
    if isinstance(n_demo, bool) or not isinstance(n_demo, int) or n_demo < 1:
        raise ValueError("n_demo must be a positive integer")
    OmegaConf.resolve(cfg)
    for key in ("num_epochs", "gradient_accumulate_every", "rollout_every", "checkpoint_every", "val_every"):
        value = OmegaConf.select(cfg, "training."+key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must resolve to a positive integer; check n_demo or override the schedule")
    workspace = OmegaConf.select(cfg, "_target_")
    policy = OmegaConf.select(cfg, "policy._target_")
    if workspace != MP1_ABLATION_WORKSPACE:
        raise ValueError(
            f"Wrong workspace {workspace!r}; this entry requires {MP1_ABLATION_WORKSPACE}.")
    if policy not in ABLATIONS.values():
        raise ValueError(f"Unsupported MP1 ablation policy {policy!r}; choose one of {tuple(ABLATIONS)}.")
    profile = OmegaConf.select(cfg, "policy.profile")
    if profile not in ("mp1", "theory"):
        raise ValueError("policy.profile must be mp1 or theory")
    if OmegaConf.select(cfg, "profile") != profile:
        raise ValueError("Set profile=... consistently; policy.profile must match the top-level profile")
    if (profile == "theory"
            and OmegaConf.select(cfg, "policy.baseline_loss") not in (None, "mse")):
        raise ValueError("profile=theory requires MSE for the no-head controls")

    required = {
        "__init__.py", "common/train_entry.py", "common/pytorch_util.py",
        "common/meanflow_sampling.py", "common/meanflow_statistics.py",
        "common/swanlab_logging.py",
        "model/meanflow_mgad.py", "model/diffusion/meanflow_conditional_unet1d.py",
        "model/vision/pointnet_extractor.py", "model/common/normalizer.py",
        "policy/mp1.py", "policy/mp1_ablation.py", "workspace/base_workspace.py",
    }
    class_targets = (MP1_ABLATION_WORKSPACE, *ABLATIONS.values())
    required.update(_target_source(target)[0] for target in class_targets)
    required.update(f"config/{name}.yaml" for name in ABLATIONS)
    dataset_target = OmegaConf.select(cfg, "task.dataset._target_")
    required.add(_target_source(dataset_target)[0])
    if OmegaConf.select(cfg, "training.enable_rollouts"):
        required.add(_target_source(OmegaConf.select(cfg, "task.env_runner._target_"))[0])
        if OmegaConf.select(cfg, "task.env_runner.dataset_path") != OmegaConf.select(cfg, "task.dataset.dataset_path"):
            raise ValueError("Training and rollout dataset paths must match in these controlled ablations")
    missing = sorted(name for name in required if not (ROOT / name).is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete MP1 ablation source at {ROOT}: {missing}")
    for target in class_targets:
        relative, symbol = _target_source(target)
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"), filename=relative)
        if not any(isinstance(node, ast.ClassDef) and node.name == symbol for node in tree.body):
            raise ValueError(f"Missing class {symbol} in {ROOT / relative}")
    print(f"ABLATION_PROFILE={profile}", flush=True)
    print(f"DATASET_PATH={OmegaConf.select(cfg, 'task.dataset.dataset_path')}", flush=True)
    print(f"TRAINING_SCHEDULE=epochs:{cfg.training.num_epochs},rollout_every:{cfg.training.rollout_every},checkpoint_every:{cfg.training.checkpoint_every}", flush=True)
    print("ABLATION_SOURCE_CHECK_OK (six policy modules and shared sources)", flush=True)
    return policy


register_resolvers()
VERIFY_ONLY = "--verify" in sys.argv
if VERIFY_ONLY:
    sys.argv.remove("--verify")


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="MP1_baseline")
def main(cfg):
    selected_policy = _validate_ablation_configuration(cfg)
    run_workspace(
        cfg, source_root=ROOT, entry_file=__file__,
        expected_workspace=MP1_ABLATION_WORKSPACE, expected_policy=selected_policy,
        verify_only=VERIFY_ONLY)


if __name__ == "__main__":
    main()
