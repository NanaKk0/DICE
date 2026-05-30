from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.training_data_assets import validate_path_only_xyz_list


def resolve_previous_outer_round_committee_root(paths, outer_round: int) -> Path | None:
    if int(outer_round) <= 0:
        return None
    root = paths.student_subworkflow_artifacts_dir(int(outer_round) - 1)
    train_stages = sorted(root.glob("student_round_*/stage_*_train"))
    return train_stages[-1].resolve() if train_stages else None


def resolve_round0_source_mode(outer_round: int) -> str:
    if int(outer_round) <= 0:
        return "bootstrap"
    return "previous_outer_all_dft_candidates"


def resolve_previous_outer_all_dft_candidate_xyzs(paths, outer_round: int) -> list[Path]:
    if int(outer_round) <= 0:
        return []
    root = paths.student_subworkflow_artifacts_dir(int(outer_round) - 1)
    return [
        path.resolve()
        for path in sorted(root.glob("student_round_*/stage_*_relabel/active_learning/dft.xyz"))
        if path.is_file() and path.stat().st_size > 0
    ]


def build_round0_bootstrap_source(paths, outer_round: int, cfg: dict[str, Any]) -> Path:
    from workflow_core.structures.datasets import merge_xyz_files

    bootstrap_inputs = validate_path_only_xyz_list(
        cfg.get("initial_student_inputs"),
        field_name="initial_student_inputs",
        bare_filename_root=getattr(paths, "user_student_initial", paths.user_assets / "student_initial"),
    )

    if not bootstrap_inputs:
        raise RuntimeError(
            "No dataset available for student round0 bootstrap. "
            "Please provide initial_student_inputs under user_assets/student_initial/ or as absolute .xyz paths."
        )

    merged_xyz = (
        paths.student_subworkflow_artifacts_dir(outer_round)
        / "source"
        / "initial_student_inputs.xyz"
    ).resolve()
    merge_xyz_files(bootstrap_inputs, merged_xyz)
    return merged_xyz


def build_cross_outer_round0_source(paths, outer_round: int, cfg: dict[str, Any], status: dict[str, Any], resume: bool, metrics_logger=None) -> Path:
    del cfg, status, metrics_logger
    from workflow_core.structures.datasets import merge_xyz_files

    del resume
    merged_xyz = (
        paths.student_subworkflow_artifacts_dir(outer_round)
        / "source"
        / "cross_teacher_round_relabel_input.xyz"
    ).resolve()
    if merged_xyz.exists() and merged_xyz.stat().st_size > 0:
        return merged_xyz

    source_mode = resolve_round0_source_mode(outer_round)
    if source_mode != "previous_outer_all_dft_candidates":
        raise RuntimeError(f"Unsupported cross-outer round0 source_mode '{source_mode}' for teacher_round={outer_round}")
    previous_candidates = resolve_previous_outer_all_dft_candidate_xyzs(paths, outer_round)
    merge_inputs: list[Path] = [path.resolve() for path in previous_candidates]
    if not merge_inputs:
        raise RuntimeError(f"No previous teacher-round candidate xyz files found for teacher_round={outer_round - 1}")

    merge_xyz_files(merge_inputs, merged_xyz)
    return merged_xyz

