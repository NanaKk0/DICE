from __future__ import annotations

from pathlib import Path
from typing import Iterable

from workflow_runtime.artifacts.registry import register_artifact_contract

_REGISTERED = False


def ensure_runtime_artifact_contracts_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    register_artifact_contract("model_artifacts", model_artifacts_complete, replace=True)
    for stage_type in (
        "relabel",
        "train",
        "sampling",
        "student_relabel_input_filtering",
        "seed_pool_refresh",
        "teacher_dft_candidate_filtering",
    ):
        register_artifact_contract(
            f"student_stage:{stage_type}",
            lambda root, stage_type=stage_type: student_stage_complete(stage_type, root),
            replace=True,
        )
    register_artifact_contract("student_round", student_round_complete, replace=True)
    register_artifact_contract("student_subworkflow", student_subworkflow_complete, replace=True)
    register_artifact_contract("teacher_foundation_train", teacher_foundation_train_complete, replace=True)
    register_artifact_contract("teacher_committee_finetune", teacher_committee_finetune_complete, replace=True)
    register_artifact_contract("first_principles_prepare", first_principles_prepare_complete, replace=True)
    register_artifact_contract("first_principles_run", first_principles_run_complete, replace=True)
    register_artifact_contract("first_principles_collect", first_principles_collect_complete, replace=True)
    register_artifact_contract("first_principles", first_principles_complete, replace=True)
    register_artifact_contract("dft_submit", dft_submit_complete, replace=True)


def model_artifacts_complete(root: str | Path, *, minimum_count: int = 1) -> bool:
    from workflow_artifacts.teacher.contracts import model_artifacts_complete as check

    return check(root, minimum_count=minimum_count)


def student_stage_complete(stage_type: str, stage_dir: str | Path) -> bool:
    from workflow_core.student_pipeline.artifacts.contracts import student_stage_complete as check

    return check(stage_type, stage_dir)


def student_round_complete(
    student_round_dir: str | Path,
    planned_stage_types: Iterable[str] | None = None,
) -> bool:
    from workflow_core.student_pipeline.artifacts.contracts import student_round_complete as check

    return check(student_round_dir, planned_stage_types=planned_stage_types)


def student_subworkflow_complete(root: str | Path) -> bool:
    from workflow_core.student_pipeline.artifacts.contracts import student_subworkflow_complete as check

    return check(root)


def teacher_foundation_train_complete(root: str | Path) -> bool:
    from workflow_artifacts.teacher.contracts import teacher_foundation_train_complete as check

    return check(root)


def teacher_committee_finetune_complete(root: str | Path, *, teacher_count: int = 1) -> bool:
    from workflow_artifacts.teacher.contracts import teacher_committee_finetune_complete as check

    return check(root, teacher_count=teacher_count)


def first_principles_prepare_complete(prepare_dir: str | Path) -> bool:
    from workflow_artifacts.first_principles.contracts import prepare_complete

    return prepare_complete(prepare_dir)


def first_principles_run_complete(run_dir: str | Path) -> bool:
    from workflow_artifacts.first_principles.contracts import run_complete

    return run_complete(run_dir)


def first_principles_collect_complete(collect_dir: str | Path) -> bool:
    from workflow_artifacts.first_principles.contracts import collect_complete

    return collect_complete(collect_dir)


def first_principles_complete(root: str | Path) -> bool:
    from workflow_artifacts.first_principles.contracts import first_principles_complete as check

    return check(root)


def dft_submit_complete(root: str | Path) -> bool:
    from workflow_artifacts.first_principles.contracts import dft_submit_complete as check

    return check(root)
