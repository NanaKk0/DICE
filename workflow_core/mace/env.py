from __future__ import annotations

import importlib
import sys
from typing import Any

from workflow_core.mace.atomic_number_patch import apply_atomic_number_table_patch
from workflow_core.mace.backend import normalize_mace_backend
from workflow_core.mace.noise import suppress_known_mace_noise


def _safe_import(module_name: str):
    try:
        with suppress_known_mace_noise():
            module = importlib.import_module(module_name)
        return True, module, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def probe_mace_environment() -> dict[str, Any]:
    mace_ok, mace_module, mace_error = _safe_import("mace")
    les_ok, les_module, les_error = _safe_import("les")

    calculator_ok = False
    calculator_error = None
    try:
        with suppress_known_mace_noise():
            from mace.calculators import MACECalculator  # noqa: F401
        calculator_ok = True
    except Exception as exc:
        calculator_error = f"{type(exc).__name__}: {exc}"

    has_mace_les_class = False
    if mace_module is not None:
        try:
            models_mod = getattr(getattr(mace_module, "modules"), "models")
            has_mace_les_class = hasattr(models_mod, "MACELES")
        except Exception:
            has_mace_les_class = False

    if has_mace_les_class and les_ok:
        repo_mode_guess = "les"
    elif has_mace_les_class:
        repo_mode_guess = "hybrid"
    elif mace_ok:
        repo_mode_guess = "standard"
    else:
        repo_mode_guess = "missing"

    return {
        "python_executable": sys.executable,
        "import_mace_ok": mace_ok,
        "import_mace_error": mace_error,
        "import_les_ok": les_ok,
        "import_les_error": les_error,
        "import_mace_calculator_ok": calculator_ok,
        "import_mace_calculator_error": calculator_error,
        "has_mace_les_class": has_mace_les_class,
        "repo_mode_guess": repo_mode_guess,
    }


def ensure_backend_environment(backend: str, env_report: dict[str, Any] | None = None) -> dict[str, Any]:
    backend = normalize_mace_backend(backend)
    env_report = dict(env_report or probe_mace_environment())

    if not env_report.get("import_mace_ok"):
        raise RuntimeError(
            "MACE environment check failed: unable to import 'mace'. "
            f"Details: {env_report.get('import_mace_error')}"
        )

    if not env_report.get("import_mace_calculator_ok"):
        raise RuntimeError(
            "MACE environment check failed: unable to import 'from mace.calculators import MACECalculator'. "
            f"Details: {env_report.get('import_mace_calculator_error')}"
        )

    if backend == "les":
        if not env_report.get("import_les_ok"):
            raise RuntimeError(
                "MACE-LES backend requested, but the current environment cannot import 'les'. "
                f"Details: {env_report.get('import_les_error')}"
            )
        if not env_report.get("has_mace_les_class"):
            raise RuntimeError(
                "MACE-LES backend requested, but this environment does not expose mace.modules.models.MACELES. "
                "Please confirm that the installed MACE/MACE-LES package version is correct."
            )

    apply_atomic_number_table_patch()
    return env_report

