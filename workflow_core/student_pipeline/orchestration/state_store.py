from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from workflow_core.student_pipeline.request_context import build_backend_identity_context

def build_student_pipeline_runtime_state(
    cfg: dict[str, Any],
    *,
    teacher_round: int,
    paths,
) -> dict[str, Any]:
    if str(cfg.get("config_mode", "")).strip().lower() != "pipeline_templates":
        raise ValueError("build_student_pipeline_runtime_state requires pipeline-template mode config")

    invocations: list[dict[str, Any]] = []
    student_round = 1
    schedule = list(cfg.get("student_pipeline_schedule", []) or [])
    resolved_templates = dict(cfg.get("resolved_student_pipeline_templates", {}) or {})

    for schedule_entry_index, schedule_entry in enumerate(schedule, start=1):
        template_name = str(schedule_entry["template"])
        template_payload = dict(resolved_templates[template_name] or {})
        stages_payload = list(template_payload.get("stages", []) or [])
        repeat = int(schedule_entry.get("repeat", 1))

        for template_repeat_index in range(1, repeat + 1):
            stages: list[dict[str, Any]] = []
            for stage_run_index, stage_payload in enumerate(stages_payload, start=1):
                stage_type = str(stage_payload["stage"])
                stage_dir = _student_stage_artifacts_dir(paths, teacher_round, student_round, stage_run_index, stage_type)
                stage_results_dir = stage_dir
                stage_record = {
                    "teacher_round": int(teacher_round),
                    "student_round": int(student_round),
                    "stage_run_index": int(stage_run_index),
                    "stage": stage_type,
                    "config": str(stage_payload["config"]),
                    "stage_dir": str(stage_dir),
                    "stage_results_dir": str(stage_results_dir),
                    "log_paths": {
                        "stdout_path": str(
                            _student_stage_stdout_log(
                                paths,
                                teacher_round,
                                student_round,
                                stage_run_index,
                                stage_type,
                            )
                        ),
                        "stderr_path": str(
                            _student_stage_stderr_log(
                                paths,
                                teacher_round,
                                student_round,
                                stage_run_index,
                                stage_type,
                            )
                        ),
                    },
                    "completed": False,
                }
                if "stage_config" in stage_payload:
                    stage_config = deepcopy(dict(stage_payload.get("stage_config", {}) or {}))
                    stage_record["stage_config"] = stage_config
                    stage_record["stage_config_fingerprint"] = _fingerprint(stage_config)
                stages.append(stage_record)

            invocations.append(
                {
                    "teacher_round": int(teacher_round),
                    "student_round": int(student_round),
                    "template_name": template_name,
                    "schedule_entry_index": int(schedule_entry_index),
                    "template_repeat_index": int(template_repeat_index),
                    "round_dir": str(_student_round_artifacts_dir(paths, teacher_round, student_round)),
                    "round_results_dir": str(_student_round_artifacts_dir(paths, teacher_round, student_round)),
                    "stage_count": len(stages),
                    "current_stage_run_index": 0,
                    "completed_stage_run_indexes": [],
                    "completed": False,
                    "stages": stages,
                }
            )
            student_round += 1

    planned_student_rounds = [int(item["student_round"]) for item in invocations]
    runtime_state = {
        "teacher_round": int(teacher_round),
        "config_mode": "pipeline_templates",
        "cfg": cfg,
        "paths": paths,
        "invocations": invocations,
        "status": {
            "teacher_round": int(teacher_round),
            "config_mode": "pipeline_templates",
            "current_student_round": 0,
            "finished_rounds": [],
            "completed_invocations": 0,
            "planned_student_rounds": planned_student_rounds,
            "next_student_round": planned_student_rounds[0] if planned_student_rounds else None,
            "schedule_exhausted": False,
            "latest_student_committee_path": None,
            "final_student_model_path": None,
        },
    }
    runtime_state["global_invariant_fingerprint"] = _fingerprint(_global_invariant_payload(cfg))
    runtime_state["stage_sequence_fingerprint"] = _fingerprint(
        _stage_sequence_payload(runtime_state["invocations"])
    )
    return runtime_state


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _global_invariant_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return build_backend_identity_context(cfg).global_invariant_payload()


def _stage_sequence_payload(invocations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for invocation in invocations:
        for stage in list(invocation.get("stages", []) or []):
            sequence.append(
                {
                    "teacher_round": int(stage["teacher_round"]),
                    "student_round": int(stage["student_round"]),
                    "stage_run_index": int(stage["stage_run_index"]),
                    "stage": str(stage["stage"]),
                }
            )
    return sequence


def _student_subworkflow_artifacts_dir(paths, teacher_round: int) -> Path:
    return paths.student_subworkflow_artifacts_dir(teacher_round)


def _student_round_artifacts_dir(paths, teacher_round: int, student_round: int) -> Path:
    return paths.student_round_artifacts_dir(teacher_round, student_round)


def _student_stage_artifacts_dir(paths, teacher_round: int, student_round: int, stage_run_index: int, stage_type: str) -> Path:
    return paths.student_stage_artifacts_dir(teacher_round, student_round, stage_run_index, stage_type)


def _student_stage_stdout_log(paths, teacher_round: int, student_round: int, stage_run_index: int, stage_type: str) -> Path:
    return paths.semantic_stdout_log(
        teacher_round,
        "committee_cycle",
        "student_subworkflow",
        paths.student_round_key(student_round),
        paths.stage_key(stage_run_index, stage_type),
    )


def _student_stage_stderr_log(paths, teacher_round: int, student_round: int, stage_run_index: int, stage_type: str) -> Path:
    return paths.semantic_stderr_log(
        teacher_round,
        "committee_cycle",
        "student_subworkflow",
        paths.student_round_key(student_round),
        paths.stage_key(stage_run_index, stage_type),
    )
