from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_backends.student.registry import get_student_config_backend, get_student_model_catalog_backend
from workflow_core.common.io import read_json


def validate_bootstrap_committee_summary(cfg: dict[str, Any], summary_path: str | Path) -> None:
    summary_path = Path(summary_path).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"bootstrap_committee_summary not found: {summary_path}")
    configured_backend_name = str(cfg.get("student_backend", "deepmd")).strip().lower()
    try:
        payload = read_json(summary_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read bootstrap_committee_summary: {summary_path}") from exc
    summary_backend_name = str(payload.get("backend", configured_backend_name)).strip().lower()
    if summary_backend_name != configured_backend_name:
        raise ValueError(
            f"bootstrap_committee_summary backend mismatch: config expects {configured_backend_name}, "
            f"but summary declares {summary_backend_name}: {summary_path}"
        )
    config_backend = get_student_config_backend(configured_backend_name)
    catalog_backend = get_student_model_catalog_backend(configured_backend_name)
    contract = config_backend.continuation_contract(dict(cfg))
    summary = catalog_backend.load_committee_summary(summary_path)
    if summary.usable_member_count < 2:
        raise RuntimeError(
            f"bootstrap_committee_summary has insufficient usable committee members for backend {configured_backend_name}: "
            f"{summary_path}"
        )
    if contract.requires_restart_assets and not summary.continuation_assets_complete:
        raise RuntimeError(
            f"bootstrap_committee_summary for backend {configured_backend_name} is not continuation-ready. "
            f"Required member assets: {', '.join(contract.required_member_assets) or 'none'}. "
            f"summary={summary_path}"
        )


__all__ = ["validate_bootstrap_committee_summary"]
