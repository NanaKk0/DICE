"""Paths for the public operator tests; DICE is resolved from this interpreter."""
import importlib.util
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SKILL_ROOT = Path(__file__).resolve().parents[3] / "skills" / "dice-operator"
spec = importlib.util.find_spec("workflow_core")
if spec is None or not spec.submodule_search_locations:
    raise ImportError("Run these tests in the Python environment containing DICE.")
CODE_ROOT = Path(next(iter(spec.submodule_search_locations))).resolve().parent
