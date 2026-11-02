"""Install the root MP1 policy and experiment files with recoverable backups."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
from datetime import datetime
import json
import shutil
import uuid


def _allowed_relative(name):
    """Accept the root MP1 policy and experiment overlay, on both platforms."""
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PureWindowsPath(name).drive or PurePosixPath(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Invalid bundle path: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    allowed = (
        name in {"policy/mp1.py", "experiments/__init__.py", "common/meanflow_sampling.py",
                 "scripts/meanflow_experiment.py", "scripts/install_meanflow_bundle.py"}
        or (len(parts) >= 3 and parts[:2] == ("experiments", "meanflow_eval")
            and path.suffix in {".py", ".md"})
        or (len(parts) == 2 and parts[0] == "tests"
            and path.name.startswith("test_meanflow_") and path.suffix == ".py")
    )
    if not allowed:
        raise ValueError(f"Overlay path is not allowed: {name!r}")
    return Path(*parts)


def _check_plain_path(path, root, *, directory=False):
    """Reject traversal and links/junctions, including links inside the project.

    A contained symlink can still redirect an allowed experiment path onto a
    policy file, so containment alone is deliberately insufficient.
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes its root: {path}") from error
    if resolved != path or path.is_symlink():
        raise ValueError(f"Linked or redirected bundle path is not allowed: {path}")
    for parent in path.parents:
        if parent == root:
            break
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Parent path is not a directory: {parent}")
    if path.exists() and (not path.is_dir() if directory else not path.is_file()):
        raise ValueError(f"Unexpected destination type: {path}")


def install_bundle(bundle, project):
    """Preflight the complete manifest before creating backups or copying files."""
    bundle, project = Path(bundle).resolve(), Path(project).expanduser().resolve()
    if not (project / "policy/mp1.py").is_file() or not (project / "config/MP1.yaml").is_file():
        raise ValueError(f"Not the expected MP1 checkout: {project}")
    overlay, manifest = bundle / "overlay", bundle / "bundle_manifest.json"
    if not overlay.is_dir() or not manifest.is_file():
        raise ValueError("Run install.py from the extracted meanflow_eval_bundle directory.")
    _check_plain_path(overlay, bundle, directory=True)
    _check_plain_path(manifest, bundle)
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    files = contents.get("overlay_files") if isinstance(contents, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("bundle_manifest.json must contain a nonempty overlay_files list.")
    planned, seen = [], set()
    for name in files:
        relative = _allowed_relative(name)
        canonical = name.casefold()
        if canonical in seen:
            raise ValueError(f"Duplicate bundle path: {name!r}")
        seen.add(canonical)
        source, target = overlay / relative, project / relative
        _check_plain_path(source, overlay)
        if not source.is_file():
            raise ValueError(f"Missing bundle file: {name}")
        _check_plain_path(target, project)
        planned.append((relative, source, target))

    backup = project / "data/meanflow_install_backup" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8])
    _check_plain_path(backup, project, directory=True)
    # Nothing above this point creates directories, backs up, or copies files.
    replaced = [(relative, target) for relative, _, target in planned if target.is_file()]
    for relative, target in replaced:
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, saved)
    for _, source, target in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return backup if replaced else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Existing server checkout containing policy/mp1.py")
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    try:
        bundle = Path(__file__).resolve().parent
        backup = install_bundle(bundle, project)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Experiment bundle installation failed: {error}\n")
    files = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))["overlay_files"]
    policy_updated = "policy/mp1.py" in files
    if policy_updated:
        print(f"Updated root MP1 policy: {project / 'policy/mp1.py'}")
    print(f"Installed experiment code: {project}")
    if backup is not None:
        print(f"Original replaced files backed up under: {backup}")
    else:
        print("No existing experiment files needed replacement.")
    if policy_updated:
        print("Other policy files, training entry point, environment YAML, existing datasets and checkpoints were not changed.")
    else:
        print("Policy source files, training entry point, environment YAML, existing datasets and checkpoints were not changed.")
    print("Next: python scripts/meanflow_experiment.py doctor")


if __name__ == "__main__":
    main()
