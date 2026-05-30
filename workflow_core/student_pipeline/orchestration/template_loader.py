from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json


def _resolve_relative_path(value: str | Path, *, anchor_path: Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    return (anchor_path.parent / raw).resolve()


def resolve_template_registry(
    registry_payload: dict[str, Any],
    *,
    config_path: str | Path | None,
) -> dict[str, str]:
    if not isinstance(registry_payload, dict) or not registry_payload:
        raise ValueError("student_pipeline_templates must be a non-empty object")
    if config_path is None:
        raise ValueError("pipeline-template mode requires a config_path for template resolution")
    anchor_path = Path(config_path).resolve()
    resolved: dict[str, str] = {}
    for template_name, template_path in registry_payload.items():
        if not str(template_name).strip():
            raise ValueError("student_pipeline_templates contains an empty template name")
        resolved_path = _resolve_relative_path(str(template_path), anchor_path=anchor_path)
        if not resolved_path.exists():
            raise ValueError(
                f"student_pipeline_template '{template_name}' points to a missing file: {resolved_path}"
            )
        resolved[str(template_name)] = str(resolved_path)
    return resolved


def load_resolved_templates(template_registry: dict[str, str]) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for template_name, template_path_text in template_registry.items():
        template_path = Path(template_path_text).resolve()
        payload = read_json(template_path)
        extra_top_level_keys = sorted(set(payload) - {"stages"})
        if extra_top_level_keys:
            raise ValueError(
                f"student_pipeline_template '{template_name}' only supports the top-level key stages; got {extra_top_level_keys}"
            )
        stages_payload = payload.get("stages", [])
        stages: list[dict[str, str]] = []
        for index, item in enumerate(stages_payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"student_pipeline_template '{template_name}' stage #{index} must be an object")
            extra_stage_keys = sorted(set(item) - {"stage", "config"})
            if extra_stage_keys:
                raise ValueError(
                    f"student_pipeline_template '{template_name}' stage #{index} only supports stage and config; got {extra_stage_keys}"
                )
            config_path = _resolve_relative_path(str(item.get("config", "")), anchor_path=template_path)
            stages.append(
                {
                    "stage": str(item.get("stage", "")).strip(),
                    "config": str(config_path),
                }
            )
        resolved[template_name] = {
            "template_path": str(template_path),
            "stages": stages,
        }
    return resolved
