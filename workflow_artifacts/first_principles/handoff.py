from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.structures.formats import validate_extxyz_file


FIRST_PRINCIPLES_HANDOFF_FILENAME = "input_for_first_principles.xyz"


def first_principles_handoff_input_path(paths: Any, teacher_round: int) -> Path:
    """Return the canonical external first-principles handoff input path."""
    return first_principles_handoff_input_path_from_teacher_round_dir(
        paths.teacher_round_artifacts_dir(int(teacher_round))
    )


def first_principles_handoff_input_path_from_teacher_round_dir(teacher_round_dir: str | Path) -> Path:
    return (
        Path(teacher_round_dir).resolve()
        / "first_principles"
        / "prepare"
        / "source"
        / FIRST_PRINCIPLES_HANDOFF_FILENAME
    ).resolve()


def first_principles_handoff_input_path_from_prepare_dir(prepare_dir: str | Path) -> Path:
    return (Path(prepare_dir).resolve() / "source" / FIRST_PRINCIPLES_HANDOFF_FILENAME).resolve()


def first_principles_handoff_input_path_for_stage(stage_dir: str | Path) -> Path | None:
    teacher_round_dir = teacher_round_artifact_dir_for_stage(stage_dir)
    if teacher_round_dir is None:
        return None
    return first_principles_handoff_input_path_from_teacher_round_dir(teacher_round_dir)


def teacher_round_artifact_dir_for_stage(stage_dir: str | Path) -> Path | None:
    root = Path(stage_dir).resolve()
    for path in (root, *root.parents):
        if path.name.startswith("teacher_round_"):
            return path
    return None


def first_principles_handoff_input_complete(path: str | Path, *, allow_empty: bool = True) -> bool:
    return validate_extxyz_file(path, allow_empty=allow_empty)
