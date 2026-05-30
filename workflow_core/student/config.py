from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_core.student_pipeline.config import resolve_student_pipeline_mode_config
from workflow_core.student_pipeline.config.validator import detect_student_subworkflow_config_mode


def resolve_student_active_learning_config(
    raw_cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    detect_student_subworkflow_config_mode(raw_cfg)
    return resolve_student_pipeline_mode_config(raw_cfg, config_path=config_path)


def load_student_active_learning_config(config_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    return resolve_student_active_learning_config(read_json(resolved_path), config_path=resolved_path)


def load_active_learning_config(config_path: str | Path) -> dict[str, Any]:
    return load_student_active_learning_config(config_path)
