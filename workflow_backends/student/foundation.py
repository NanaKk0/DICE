from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from workflow_backends.student.base import StudentFoundationBootstrapSpec
from workflow_core.student_pipeline.config.train_regime import (
    INITIALIZATION_MODE_FOUNDATION,
    ResolvedTrainDecision,
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _backend_aliases(backend_name: str) -> tuple[str, ...]:
    normalized = str(backend_name or "").strip().lower()
    if normalized == "deepmd":
        return ("deepmd", "dp")
    return (normalized,)


def _normalize_type_map(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def resolve_foundation_bootstrap_spec(
    cfg: dict[str, Any],
    backend_name: str,
) -> StudentFoundationBootstrapSpec:
    raw = _as_dict(cfg.get("student_foundation_asset"))
    aliases = _backend_aliases(backend_name)
    backend_payload: dict[str, Any] = {}
    for alias in aliases:
        if isinstance(raw.get(alias), dict):
            backend_payload = _as_dict(raw.get(alias))
            break

    merged = deepcopy(raw)
    for alias in ("deepmd", "dp", "nep"):
        merged.pop(alias, None)
    merged.update(backend_payload)

    type_map = _normalize_type_map(merged.get("type_map") or raw.get("type_map"))
    if not type_map:
        type_map = _normalize_type_map(_as_dict(cfg.get(str(backend_name).strip().lower())).get("type_map"))
    backend_options = dict(merged)
    for key in ("model_path", "restart_path", "type_map"):
        backend_options.pop(key, None)
    return StudentFoundationBootstrapSpec(
        backend_name=str(backend_name).strip().lower(),
        model_path=str(merged["model_path"]) if merged.get("model_path") else None,
        restart_path=str(merged["restart_path"]) if merged.get("restart_path") else None,
        type_map=type_map,
        backend_options=backend_options,
    )


def foundation_bootstrap_applies(
    cfg: dict[str, Any],
    backend_name: str,
    *,
    outer_round: int,
    student_round: int,
) -> StudentFoundationBootstrapSpec | None:
    raw = _as_dict(cfg.get("student_foundation_asset"))
    if not raw:
        return None
    spec = resolve_foundation_bootstrap_spec(cfg, backend_name)
    if int(outer_round) == 0 and int(student_round) == 0:
        return spec
    return None


def resolve_train_regime_foundation_asset(
    cfg: dict[str, Any],
    backend_name: str,
    decision: ResolvedTrainDecision,
) -> StudentFoundationBootstrapSpec | None:
    if decision.initialization_mode != INITIALIZATION_MODE_FOUNDATION or not decision.uses_foundation_model:
        return None
    raw = _as_dict(cfg.get("student_foundation_asset"))
    if not raw:
        return None
    return resolve_foundation_bootstrap_spec(cfg, backend_name)


def validate_foundation_paths(*paths: str | Path | None) -> None:
    for raw_path in paths:
        if raw_path is None:
            raise ValueError("Foundation bootstrap path is required")
        path = Path(str(raw_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Foundation bootstrap file not found: {path}")


def validate_type_map_coverage(foundation_type_map: list[str], required_type_map: list[str], *, label: str) -> None:
    foundation = {str(item) for item in foundation_type_map}
    missing = [str(item) for item in required_type_map if str(item) not in foundation]
    if missing:
        raise ValueError(
            f"{label} foundation model type_map does not cover training elements. "
            f"missing={missing}, foundation_type_map={foundation_type_map}, required_type_map={required_type_map}"
        )
