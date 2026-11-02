"""Independent EfficientAdaFlow entry, including for servers with legacy train.py.

Use --verify to check the selected configuration and source files before loading
images. This file never imports the old train.py or an embedded MP1 workspace.
"""
from pathlib import Path
import importlib.util
import sys


ENTRY_DIR = Path(__file__).resolve().parent


def _source_root(entry_dir):
    """Support a flat checkout and a launcher beside an EfficientFlow package.

    The server keeps entry scripts in PROJECT_EfficientFlow/ and source files
    in PROJECT_EfficientFlow/EfficientFlow/. Do not assume the launcher's own
    directory is a Python package, or change cwd away from the data directory.
    """
    candidates = (entry_dir, entry_dir / "EfficientFlow")
    for candidate in candidates:
        if ((candidate / "__init__.py").is_file()
                and all((candidate / name).is_dir() for name in ("config", "policy", "workspace"))):
            return candidate
    checked = "\n".join(f"  {candidate}" for candidate in candidates)
    raise SystemExit(
        "Cannot locate the EfficientFlow source package. Checked:\n"
        f"{checked}\n"
        "Expected __init__.py plus config/, policy/ and workspace/ in one directory. "
        "Keep the source package together; an empty __init__.py beside the launcher is not a fix.")


ROOT = _source_root(ENTRY_DIR)
if not (ROOT / "common" / "train_entry.py").is_file():
    raise SystemExit(
        f"Missing {ROOT / 'common' / 'train_entry.py'}. "
        "Upload common/train_entry.py into the same source package before starting.")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "EfficientFlow", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["EfficientFlow"] = package
spec.loader.exec_module(package)

import hydra
from EfficientFlow.common.train_entry import (
    EFFICIENT_ADAFLOW_POLICY, EFFICIENT_ADAFLOW_WORKSPACE,
    register_resolvers, run_workspace,
)

register_resolvers()
VERIFY_ONLY = "--verify" in sys.argv
if VERIFY_ONLY:
    sys.argv.remove("--verify")


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="EfficientAdaFlow")
def main(cfg):
    run_workspace(
        cfg, source_root=ROOT, entry_file=__file__,
        expected_workspace=EFFICIENT_ADAFLOW_WORKSPACE,
        expected_policy=EFFICIENT_ADAFLOW_POLICY, verify_only=VERIFY_ONLY)


if __name__ == "__main__":
    main()
