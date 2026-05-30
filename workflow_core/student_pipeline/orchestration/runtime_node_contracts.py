from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from workflow_core.student_pipeline.artifacts.stage_contracts import student_stage_contract_descriptor
from workflow_runtime.artifacts.registry import artifact_check_for
from workflow_runtime.node_contracts import MARKER_SUFFIXES, RuntimeNodeContract


def student_stage_runtime_node_contract(
    runtime_state: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> RuntimeNodeContract:
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("student pipeline runtime_state is missing paths")

    teacher_round = int(stage["teacher_round"])
    student_round = int(stage["student_round"])
    stage_run_index = int(stage["stage_run_index"])
    stage_type = str(stage["stage"])
    stage_dir = Path(str(stage["stage_dir"])).resolve()
    try:
        descriptor = student_stage_contract_descriptor(stage_type)
        artifact_contract_id = descriptor.artifact_contract_id
        artifact_root = descriptor.artifact_root(stage_dir)
    except ValueError:
        artifact_contract_id = f"student_stage:{stage_type}"
        artifact_root = stage_dir
    node_id = student_stage_node_id(
        paths,
        teacher_round=teacher_round,
        student_round=student_round,
        stage_run_index=stage_run_index,
        stage_type=stage_type,
    )
    return RuntimeNodeContract(
        node_id=node_id,
        teacher_round=teacher_round,
        marker_paths_by_suffix={
            suffix: paths.student_stage_marker_path(
                teacher_round,
                student_round,
                stage_run_index,
                stage_type,
                suffix,
            )
            for suffix in MARKER_SUFFIXES
        },
        artifact_check=artifact_check_for(artifact_contract_id, stage_dir),
        identity={
            "teacher_round": teacher_round,
            "student_round": student_round,
            "stage_run_index": stage_run_index,
            "stage": stage_type,
        },
        artifact_contract_id=artifact_contract_id,
        global_invariant_fingerprint=str(runtime_state.get("global_invariant_fingerprint", "")),
        stage_sequence_fingerprint=str(runtime_state.get("stage_sequence_fingerprint", "")),
        stage_local_config_fingerprint=str(
            stage.get("stage_local_config_fingerprint", stage.get("stage_config_fingerprint", ""))
        ),
        artifact_roots=(artifact_root,),
        semantic_log_roots=_student_stage_semantic_log_roots(stage),
        backend_log_roots=(
            Path(
                paths.student_stage_backend_logs_dir(
                    teacher_round,
                    student_round,
                    stage_run_index,
                    stage_type,
                )
            ).resolve(),
        ),
    )


def student_stage_node_id(
    paths: Any,
    *,
    teacher_round: int,
    student_round: int,
    stage_run_index: int,
    stage_type: str,
) -> str:
    return (
        f"r{int(teacher_round)}.committee_cycle.student_subworkflow."
        f"{paths.student_round_key(student_round)}.{paths.stage_key(stage_run_index, stage_type)}"
    )


def _student_stage_semantic_log_roots(stage: Mapping[str, Any]) -> tuple[Path, ...]:
    log_paths = dict(stage.get("log_paths", {}) or {})
    for key in ("stdout_path", "stderr_path"):
        if log_paths.get(key):
            return (Path(str(log_paths[key])).resolve().parent,)
    return ()
