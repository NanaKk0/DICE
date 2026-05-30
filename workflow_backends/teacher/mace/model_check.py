from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_backends.teacher.mace.backend import get_model_type_for_backend, normalize_mace_backend
from workflow_core.common.io import read_json


def _metadata_path(model_root: str | Path) -> Path:
    return Path(model_root).resolve() / "mace_model_metadata.json"


def _model_root_from_path(model_path: str | Path) -> Path:
    model_path = Path(model_path).resolve()
    return model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent


def write_mace_model_metadata(
    model_root: str | Path,
    *,
    backend: str,
    model_role: str,
    env_report: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    backend = normalize_mace_backend(backend)
    model_root = Path(model_root).resolve()
    model_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "mace_backend": backend,
        "model_type": get_model_type_for_backend(backend),
        "model_role": str(model_role),
    }
    if env_report is not None:
        payload.update(
            {
                "python_executable": env_report.get("python_executable"),
                "repo_mode_guess": env_report.get("repo_mode_guess"),
                "has_mace_les_class": bool(env_report.get("has_mace_les_class", False)),
            }
        )
    if extra:
        payload.update(extra)

    metadata_path = _metadata_path(model_root)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return metadata_path


def load_mace_model_metadata(model_root: str | Path) -> dict[str, Any] | None:
    metadata_path = _metadata_path(model_root)
    if not metadata_path.exists():
        return None
    return read_json(metadata_path)


def infer_mace_backend(model_root: str | Path, fallback: str = "standard") -> str:
    metadata = load_mace_model_metadata(model_root)
    if metadata is None:
        return normalize_mace_backend(fallback)
    return normalize_mace_backend(metadata.get("mace_backend", fallback))


def ensure_foundation_backend_compatible(model_path: str | Path, backend: str) -> str:
    backend = normalize_mace_backend(backend)
    model_root = _model_root_from_path(model_path)
    # User-provided foundation models are trusted to match the configured
    # backend once they are placed under user_assets/foundation_models.
    # Persist metadata so later steps can reuse the same backend label.
    write_mace_model_metadata(model_root, backend=backend, model_role="foundation")
    return backend


def ensure_committee_backend_compatible(teacher_dir: str | Path, backend: str, teacher_num: int | None = None) -> str:
    backend = normalize_mace_backend(backend)
    teacher_dir = Path(teacher_dir).resolve()
    root_backend = infer_mace_backend(teacher_dir, fallback="standard")

    member_backends: list[str] = []
    teacher_dirs = sorted(path for path in teacher_dir.glob("teacher_*") if path.is_dir())
    if teacher_num is not None and teacher_dirs and len(teacher_dirs) < int(teacher_num):
        raise RuntimeError(
            f"Teacher committee appears incomplete in {teacher_dir}: expected {teacher_num} teacher directories, found {len(teacher_dirs)}"
        )

    for member_dir in teacher_dirs:
        member_backends.append(infer_mace_backend(member_dir, fallback=root_backend))

    all_backends = {root_backend, *member_backends}
    if len(all_backends) > 1:
        raise RuntimeError(
            f"Mixed MACE backends detected inside teacher committee {teacher_dir}: {sorted(all_backends)}. "
            "Standard MACE and MACE-LES teachers cannot be mixed."
        )

    resolved = next(iter(all_backends)) if all_backends else root_backend
    if resolved != backend:
        raise RuntimeError(
            f"Teacher committee backend mismatch: requested backend={backend}, but teacher committee is {resolved}. "
            f"Teacher dir: {teacher_dir}"
        )
    return resolved
