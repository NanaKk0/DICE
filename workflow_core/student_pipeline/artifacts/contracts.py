from __future__ import annotations

from pathlib import Path
from typing import Iterable

from workflow_artifacts.teacher.contracts import model_artifacts_complete
from workflow_core.student_pipeline.artifacts.stage_contracts import student_stage_contract_descriptor
from workflow_core.student_pipeline.config.stage_config import canonical_stage_type
from workflow_runtime.artifacts.checks import xyz_file_complete


def student_stage_complete(stage_type: str, stage_dir: str | Path) -> bool:
    root = Path(stage_dir)
    stage = canonical_stage_type(stage_type)
    descriptor = student_stage_contract_descriptor(stage)
    if stage == "train":
        return model_artifacts_complete(root)
    try:
        return all(
            xyz_file_complete(
                descriptor.published_output_path(role, root),
                allow_empty=role in descriptor.allow_empty_extxyz_roles,
            )
            for role in descriptor.completion_evidence_roles
        )
    except ValueError:
        return False


def student_round_complete(
    student_round_dir: str | Path,
    planned_stage_types: Iterable[str] | None = None,
) -> bool:
    root = Path(student_round_dir)
    if planned_stage_types is None:
        stage_dirs = sorted(path for path in root.glob("stage_*_*") if path.is_dir())
        return bool(stage_dirs) and all(_student_stage_dir_complete(path) for path in stage_dirs)
    for stage_type in planned_stage_types:
        matches = sorted(root.glob(f"stage_*_{stage_type}"))
        if not matches or not any(student_stage_complete(stage_type, path) for path in matches):
            return False
    return True


def student_subworkflow_complete(root: str | Path) -> bool:
    base = Path(root)
    round_dirs = sorted(path for path in base.glob("student_round_*") if path.is_dir())
    return bool(round_dirs) and all(student_round_complete(path) for path in round_dirs)


def _student_stage_dir_complete(stage_dir: Path) -> bool:
    parts = stage_dir.name.split("_", 2)
    if len(parts) != 3:
        return False
    return student_stage_complete(parts[2], stage_dir)
