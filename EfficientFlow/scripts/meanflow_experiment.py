"""Run experiments against this checkout's root policy/mp1.py."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The user explicitly selected root policy/mp1.py. Never search the nested copy
# or let the checkout directory name change the historical package import.
spec = importlib.util.spec_from_file_location(
    "EfficientFlow", ROOT/"__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["EfficientFlow"] = package
spec.loader.exec_module(package)
package._meanflow_source_root = str(ROOT)

from EfficientFlow.experiments.meanflow_eval.cli import main

if __name__ == "__main__":
    main()
