from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_core.student.config import load_student_active_learning_config as _load_student_active_learning_config
from workflow_core.student_pipeline.config.validator import detect_student_subworkflow_config_mode


def load_student_active_learning_config(config_path: str | Path) -> dict[str, Any]:
    return _load_student_active_learning_config(config_path)


def run_student_active_learning_subworkflow(
    cfg_path: str | Path,
    paths,
    outer_round: int,
    metrics_logger=None,
    *,
    extra_cfg: dict[str, Any] | None = None,
) -> str:
    if detect_student_subworkflow_config_mode(read_json(Path(cfg_path).resolve())) != "pipeline_templates":
        raise ValueError("student_subworkflow requires student pipeline template mode")
    return _run_student_pipeline_template_subworkflow(
        cfg_path,
        paths,
        outer_round,
        metrics_logger=metrics_logger,
        extra_cfg=extra_cfg,
    )


def _run_student_pipeline_template_subworkflow(
    cfg_path: str | Path,
    paths,
    outer_round: int,
    metrics_logger=None,
    *,
    extra_cfg: dict[str, Any] | None = None,
) -> str:
    from workflow_core.student_pipeline.orchestration.orchestrator import run_student_pipeline_template_subworkflow

    return run_student_pipeline_template_subworkflow(
        cfg_path,
        paths,
        outer_round,
        metrics_logger=metrics_logger,
        extra_cfg=extra_cfg,
    )
