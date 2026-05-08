from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


VALID_MACE_BACKENDS = {"standard", "les"}
BACKEND_MODEL_TYPES = {
    "standard": "MACE",
    "les": "MACELES",
}


def normalize_mace_backend(value: str | None) -> str:
    backend = str(value or "standard").strip().lower()
    if backend not in VALID_MACE_BACKENDS:
        raise ValueError(f"Unsupported mace_backend={value!r}. Expected one of: {sorted(VALID_MACE_BACKENDS)}")
    return backend


def resolve_mace_backend(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return normalize_mace_backend(value)
    return "standard"


def get_model_type_for_backend(backend: str) -> str:
    backend = normalize_mace_backend(backend)
    return BACKEND_MODEL_TYPES[backend]


def apply_backend_to_train_config(train_cfg: dict[str, Any], backend: str) -> dict[str, Any]:
    backend = normalize_mace_backend(backend)
    payload = dict(train_cfg)
    payload["mace_backend"] = backend
    payload["model"] = get_model_type_for_backend(backend)
    return payload


def build_backend_log_payload(env_report: dict[str, Any], backend: str) -> dict[str, Any]:
    backend = normalize_mace_backend(backend)
    return {
        "backend": backend,
        "model_type": get_model_type_for_backend(backend),
        "python_executable": env_report.get("python_executable", sys.executable),
        "repo_mode_guess": env_report.get("repo_mode_guess", "unknown"),
        "has_mace_les_class": bool(env_report.get("has_mace_les_class", False)),
    }


def print_backend_banner(env_report: dict[str, Any], backend: str) -> None:
    payload = build_backend_log_payload(env_report, backend)
    print("\nMACE backend")
    print("----------------")
    for key, value in payload.items():
        print(f"{key}: {value}")


def build_mace_calculator(model_path: str | Path, backend: str, *, device: str = "cpu", default_dtype: str = "float32"):
    backend = normalize_mace_backend(backend)
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_path=str(Path(model_path).resolve()),
        device=device,
        default_dtype=default_dtype,
    )
