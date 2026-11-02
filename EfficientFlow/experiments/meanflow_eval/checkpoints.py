"""Locate a trained MP1 run by its saved Hydra configuration, not its name."""
from pathlib import Path


def find_checkpoint(root, *, task, training_seed, n_demo):
    from omegaconf import OmegaConf
    root = Path(root).expanduser().resolve()
    candidates = []
    for path in root.rglob("latest.ckpt"):
        if path.parent.name != "checkpoints":
            continue
        config_path = path.parent.parent / ".hydra" / "config.yaml"
        if not config_path.is_file():
            continue
        cfg = OmegaConf.load(config_path)
        if (str(cfg.get("task_name")) != task or int(cfg.get("n_demo", -1)) != n_demo
                or int(OmegaConf.select(cfg, "training.seed", default=-1)) != training_seed):
            continue
        target = str(OmegaConf.select(cfg, "policy._target_", default=""))
        if target != "EfficientFlow.policy.mp1.Meanpolicy":
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No MP1 latest.ckpt with task={task}, n_demo={n_demo}, training.seed={training_seed} "
            f"under {root}. MP1.yaml currently disables checkpoint saving by default; "
            "enable checkpoint.save_ckpt=True for training, or use --checkpoint with an existing workspace checkpoint. "
            "Automatic lookup requires the run's .hydra/config.yaml."
        )
    # This chooses the most recently written matching run, not the best test score.
    return max(candidates, key=lambda item: (item.stat().st_mtime_ns, str(item)))
