from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from workflow_core.student_pipeline.artifacts.successful_stage_resolver import SuccessfulStageArtifactResolver
from workflow_core.structures.formats import concat_extxyz_files, copy_extxyz_file
from workflow_core.training_data_assets import validate_path_only_xyz_list


def materialize_initial_seed_pool_snapshot(
    cfg: dict[str, Any],
    *,
    teacher_round: int,
    paths,
) -> Path:
    target = Path(paths.student_subworkflow_initial_seed_pool_xyz(teacher_round)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if int(teacher_round) == 0:
        sources = _initial_seed_pool_sources(cfg, paths=paths)
        if not sources:
            raise ValueError("source.initial_seed_pool must contain at least one seed-pool extxyz for teacher_round=0")
        _concatenate_files(sources, target)
        return target

    source = _previous_teacher_round_seed_successor(paths, int(teacher_round) - 1)
    if source is None:
        raise ValueError(f"cannot derive initial seed pool for teacher_round={teacher_round}: previous source is missing")
    copy_extxyz_file(source, target, allow_empty=False)
    return target


def _initial_seed_pool_sources(cfg: dict[str, Any], *, paths) -> tuple[Path, ...]:
    source_cfg = dict(cfg.get("source", {}) or {})
    raw_values = source_cfg.get("initial_seed_pool", ())
    bare_root = Path(getattr(paths, "user_seed_pool", Path.cwd() / "user_assets" / "seed_pool")).resolve()
    return tuple(
        validate_path_only_xyz_list(
            raw_values,
            field_name="source.initial_seed_pool",
            bare_filename_root=bare_root,
        )
    )


def _concatenate_files(sources: Iterable[Path], target: Path) -> None:
    concat_extxyz_files(sources, target, allow_empty_sources=False, allow_empty_output=False)


def _previous_teacher_round_seed_successor(paths, teacher_round: int) -> Path | None:
    source = SuccessfulStageArtifactResolver({"paths": paths}).previous_teacher_seed_successor(teacher_round)
    return None if source is None else source.path
