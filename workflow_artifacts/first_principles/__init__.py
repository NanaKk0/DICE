"""First-principles artifact compatibility facades."""
from workflow_artifacts.first_principles.handoff import (
    FIRST_PRINCIPLES_HANDOFF_FILENAME,
    first_principles_handoff_input_complete,
    first_principles_handoff_input_path,
    first_principles_handoff_input_path_for_stage,
    first_principles_handoff_input_path_from_prepare_dir,
    first_principles_handoff_input_path_from_teacher_round_dir,
    teacher_round_artifact_dir_for_stage,
)


__all__ = [
    "FIRST_PRINCIPLES_HANDOFF_FILENAME",
    "first_principles_handoff_input_complete",
    "first_principles_handoff_input_path",
    "first_principles_handoff_input_path_for_stage",
    "first_principles_handoff_input_path_from_prepare_dir",
    "first_principles_handoff_input_path_from_teacher_round_dir",
    "teacher_round_artifact_dir_for_stage",
]
