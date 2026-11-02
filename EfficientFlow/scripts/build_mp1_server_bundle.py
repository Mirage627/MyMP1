"""Build a clean source archive for /home/YOUR_USER/MyMP1/EfficientFlow.

Exclude data, checkpoints, credentials, Git history and local test dependencies.
Keep MimicGen's runtime models/assets; no inference/training weights are bundled.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("common", "config", "dataset", "env", "env_runner", "gym_util", "model/common",
               "model/diffusion", "model/vision", "models", "shared_memory", "codecs", "mimicgen_environments")
EXTRA_FILES = ("__init__.py", "sde_lib.py", "losses.py", "LICENSE", "README.md", "setup.py",
               "train_mp1_ablation.py", "scripts/build_mp1_server_bundle.py",
               "scripts/run_mp1_ablation_server.sh", "scripts/check_mp1_swanlab.py",
               "scripts/eval_mp1_checkpoint.py", "docs/MP1_zero_score_fix_20261101_zh.md",
               "docs/MP1_server_commands_zh.md", "docs/MP1_ablation_runbook_zh.md",
               "docs/mp1_theory_audit_zh.md", "docs/MP1_yaml_audit_zh.md", "tests/test_meanflow_sampling.py",
               "model/meanflow_mgad.py",
               "policy/base_policy.py", "policy/base_image_policy.py", "policy/mp1.py",
               "workspace/base_workspace.py",
               "workspace/train_mp1_workspace.py", "workspace/train_mp1_ablation_workspace.py")
EXCLUDED_DIRS = {".git", ".github", ".codex", ".agents", ".ssh", ".aws", "__pycache__",
                 ".pytest_cache", ".mypy_cache", "wandb", "swanlog", "checkpoints", "outputs"}
EXCLUDED_EXTENSIONS = {".ckpt", ".pt", ".pth", ".hdf5", ".h5", ".pyc", ".pyo", ".zip", ".tar", ".gz"}


def source_files():
    names = set(EXTRA_FILES)
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT/"tests").glob("test_mp1_*.py"))
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT/"policy").glob("mp1_*.py"))
    for top in DIRECTORIES:
        for directory, subdirs, files in os.walk(ROOT/top, followlinks=False):
            subdirs[:] = sorted(name for name in subdirs if name not in EXCLUDED_DIRS
                                and not name.endswith(".egg-info")
                                and not (top == "mimicgen_environments" and name == "docs"))
            for name in files:
                path = Path(directory)/name
                if (name.startswith(".env") or path.suffix.lower() in EXCLUDED_EXTENSIONS
                        or name in {".git", "credentials", "id_rsa", "id_ed25519"}):
                    continue
                if path.is_symlink():
                    raise ValueError(f"Review symlink before packaging: {path}")
                names.add(path.relative_to(ROOT).as_posix())
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT/"artifacts"/"mp1_server_source_20261031.tar.gz"))
    args = parser.parse_args()
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = source_files()
    manifest = {"layout": "EfficientFlow/", "target": "YOUR_USER@YOUR_SERVER:/home/YOUR_USER/MyMP1",
                "default_task": "kitchen_d1", "n_demo": 200, "training_seed": 20,
                "yaml_layout": "six complete configs sharing only the task group",
                "default_schedule": {"epochs": 250, "rollout_every": 5, "checkpoint_every": 5},
                "dataset_included": False, "checkpoints_included": False,
                "files": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}}
    with tarfile.open(destination, "w:gz") as archive:
        for name in names:
            path = ROOT/name
            data = path.read_bytes()
            if path.suffix == ".py":
                compile(data, name, "exec")
            if path.suffix == ".sh":
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo("EfficientFlow/"+name)
            info.size = len(data)
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            archive.addfile(info, io.BytesIO(data))
        content = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("EfficientFlow/MP1_SERVER_MANIFEST.json")
        info.size, info.mode = len(content), 0o644
        archive.addfile(info, io.BytesIO(content))
    with tarfile.open(destination, "r:gz") as archive:
        assert len(archive.getmembers()) == len(names)+1
        assert all(member.name.startswith("EfficientFlow/") and ".." not in Path(member.name).parts
                   for member in archive.getmembers())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+".sha256").write_text(digest+"  "+destination.name+"\n", encoding="ascii")
    print(json.dumps({"archive": str(destination), "files": len(names),
                      "bytes": destination.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
