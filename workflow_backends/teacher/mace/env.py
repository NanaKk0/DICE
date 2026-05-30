from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from typing import Any

from workflow_core.mace.atomic_number_patch import apply_atomic_number_table_patch
from workflow_backends.teacher.mace.backend import normalize_mace_backend
from workflow_core.mace.noise import suppress_known_mace_noise


def _safe_import(module_name: str):
    try:
        with suppress_known_mace_noise():
            module = importlib.import_module(module_name)
        return True, module, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _find_mace_les_class_module(mace_module) -> str | None:
    if mace_module is not None:
        try:
            modules_mod = getattr(mace_module, "modules")
            attr_candidates = (
                ("mace.modules.models", getattr(modules_mod, "models", None)),
                ("mace.modules.extensions", getattr(modules_mod, "extensions", None)),
            )
            for module_label, module in attr_candidates:
                if module is not None and hasattr(module, "MACELES"):
                    return module_label
        except Exception:
            pass

    for module_label in ("mace.modules.models", "mace.modules.extensions"):
        module_ok, module, _ = _safe_import(module_label)
        if module_ok and module is not None and hasattr(module, "MACELES"):
            return module_label
    return None


def probe_mace_environment(execution_context=None) -> dict[str, Any]:
    if execution_context is not None and getattr(execution_context, "source_list", ()):
        return _probe_mace_environment_with_python(execution_context)
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

    mace_les_class_module = _find_mace_les_class_module(mace_module) if mace_ok else None
    has_mace_les_class = mace_les_class_module is not None

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
        "mace_les_class_module": mace_les_class_module,
        "repo_mode_guess": repo_mode_guess,
    }


def _probe_mace_environment_with_python(execution_context) -> dict[str, Any]:
    script = (
        "import json; "
        "from workflow_backends.teacher.mace.env import probe_mace_environment; "
        "print(json.dumps(probe_mace_environment(), ensure_ascii=False))"
    )
    command = [str(sys.executable), "-c", script]
    if hasattr(execution_context, "source_list") and execution_context.source_list:
        from workflow_backends.execution_context import BackendExecution

        command = BackendExecution(execution_context).shell_command(command)
    completed = subprocess.run(
        command,
        env=execution_context.merge_env(os.environ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = [line.strip() for line in str(completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("MACE environment probe produced no JSON output")
    return json.loads(lines[-1])


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
                "MACE-LES backend requested, but this environment does not expose MACELES from "
                "mace.modules.models or mace.modules.extensions. "
                "Please confirm that the installed MACE/MACE-LES package version is correct."
            )

    apply_atomic_number_table_patch()
    return env_report

