from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def resolve_stage_context(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for invocation in list(runtime_state.get("invocations", []) or []):
        if int(invocation["student_round"]) != int(student_round):
            continue
        for stage in list(invocation.get("stages", []) or []):
            if int(stage["stage_run_index"]) == int(stage_run_index):
                stage_dir = Path(str(stage["stage_dir"])).resolve()
                stage_results_dir = Path(str(stage["stage_results_dir"])).resolve()
                stage_dir.mkdir(parents=True, exist_ok=True)
                stage_results_dir.mkdir(parents=True, exist_ok=True)
                (stage_dir / "input").mkdir(parents=True, exist_ok=True)
                context = {
                    "teacher_round": int(invocation["teacher_round"]),
                    "student_round": int(invocation["student_round"]),
                    "template_name": str(invocation["template_name"]),
                    "stage_run_index": int(stage["stage_run_index"]),
                    "stage": str(stage["stage"]),
                    "stage_dir": str(stage_dir),
                    "stage_results_dir": str(stage_results_dir),
                    "config": str(stage["config"]),
                }
                if "stage_config" in stage:
                    context["stage_config"] = deepcopy(dict(stage.get("stage_config", {}) or {}))
                if "stage_config_fingerprint" in stage:
                    context["stage_config_fingerprint"] = str(stage["stage_config_fingerprint"])
                return invocation, context
    raise ValueError(f"unknown stage context for student_round={student_round}, stage_run_index={stage_run_index}")


def resolve_stage_config(stage_context: dict[str, Any], expected_stage_type: str | None = None) -> dict[str, Any]:
    if "stage_config" in stage_context:
        return deepcopy(dict(stage_context.get("stage_config", {}) or {}))
    from workflow_core.student_pipeline.config.stage_config import load_pipeline_stage_config

    stage_type = str(expected_stage_type or stage_context.get("stage", "")).strip()
    return load_pipeline_stage_config(stage_type, Path(str(stage_context["config"])).resolve())
