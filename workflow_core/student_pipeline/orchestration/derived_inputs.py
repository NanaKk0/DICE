from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from workflow_core.student_pipeline.artifacts.successful_stage_resolver import SuccessfulStageArtifactResolver
from workflow_core.structures.formats import concat_extxyz_files, copy_extxyz_file
from workflow_core.training_data_assets import validate_path_only_xyz_list


def materialize_stage_inputs(runtime_state: dict[str, Any], stage: dict[str, Any]) -> dict[str, Path]:
    stage_type = str(stage["stage"])
    if stage_type == "sampling":
        return _materialize_sampling_inputs(runtime_state, stage)
    if stage_type == "seed_pool_refresh":
        return _materialize_seed_pool_refresh_inputs(runtime_state, stage)
    if stage_type == "student_relabel_input_filtering":
        return _materialize_student_relabel_input_filtering_inputs(runtime_state, stage)
    if stage_type == "train":
        return _materialize_train_inputs(runtime_state, stage)
    if stage_type == "teacher_dft_candidate_filtering":
        return _materialize_teacher_dft_candidate_filtering_inputs(runtime_state, stage)
    if stage_type == "relabel":
        return _materialize_relabel_inputs(runtime_state, stage)
    return {}


def _materialize_sampling_inputs(runtime_state: dict[str, Any], stage: dict[str, Any]) -> dict[str, Path]:
    seed_pool = _input_dir(stage) / "seed_pool.xyz"
    _copy_file(_current_effective_seed_pool(runtime_state, stage), seed_pool)
    return {"seed_pool": seed_pool}


def _materialize_seed_pool_refresh_inputs(runtime_state: dict[str, Any], stage: dict[str, Any]) -> dict[str, Path]:
    input_dir = _input_dir(stage)
    seed_pool = input_dir / "seed_pool.xyz"
    candidate_increment = input_dir / "candidate_increment.xyz"
    failed_increment = input_dir / "failed_increment.xyz"
    _copy_file(_current_effective_seed_pool(runtime_state, stage), seed_pool)
    window = _successful_sampling_window_after(runtime_state, stage, previous_stage_type="seed_pool_refresh")
    _concat((Path(item["stage_dir"]) / "candidate.xyz" for item in window), candidate_increment)
    _concat((Path(item["stage_dir"]) / "failed.xyz" for item in window), failed_increment)
    return {
        "seed_pool": seed_pool,
        "candidate_increment": candidate_increment,
        "failed_increment": failed_increment,
    }


def _materialize_student_relabel_input_filtering_inputs(
    runtime_state: dict[str, Any],
    stage: dict[str, Any],
) -> dict[str, Path]:
    input_dir = _input_dir(stage)
    candidate = input_dir / "candidate.xyz"
    failed = input_dir / "failed.xyz"
    window = _successful_sampling_window_after(runtime_state, stage, previous_stage_type="relabel")
    if not window:
        raise ValueError("student_relabel_input_filtering has no successful sampling outputs to consume")
    _concat((Path(item["stage_dir"]) / "candidate.xyz" for item in window), candidate)
    _concat((Path(item["stage_dir"]) / "failed.xyz" for item in window), failed)
    if candidate.stat().st_size == 0 and failed.stat().st_size == 0:
        raise ValueError("student_relabel_input_filtering input window is empty")
    return {"candidate": candidate, "failed": failed}


def _materialize_train_inputs(runtime_state: dict[str, Any], stage: dict[str, Any]) -> dict[str, Path]:
    train = _input_dir(stage) / "train.xyz"
    cfg = dict(runtime_state.get("cfg", {}) or {})
    train_regime = dict(cfg.get("train_regime", {}) or {})
    continue_training = bool(train_regime.get("continue_training", False))
    relabel_stages = _successful_stages_before(runtime_state, stage, "relabel")
    if not relabel_stages:
        raise ValueError("train stage has no successful relabel training artifacts to consume")
    selected = relabel_stages[-1:] if continue_training else relabel_stages
    _concat((Path(item["stage_dir"]) / "label" / "train.xyz" for item in selected), train)
    return {"train": train}


def _materialize_teacher_dft_candidate_filtering_inputs(
    runtime_state: dict[str, Any],
    stage: dict[str, Any],
) -> dict[str, Path]:
    candidates = _input_dir(stage) / "teacher_dft_candidates.xyz"
    relabel_stages = [
        item
        for item in _successful_stages_before(runtime_state, stage, "relabel")
        if int(item["teacher_round"]) == int(stage["teacher_round"])
    ]
    if not relabel_stages:
        raise ValueError("teacher_dft_candidate_filtering has no current-round relabel DFT artifacts")
    _concat((Path(item["stage_dir"]) / "active_learning" / "dft.xyz" for item in relabel_stages), candidates)
    return {"teacher_dft_candidates": candidates}


def _materialize_relabel_inputs(runtime_state: dict[str, Any], stage: dict[str, Any]) -> dict[str, Path]:
    relabel_input = _input_dir(stage) / "relabel_input.xyz"
    filtering_stages = _successful_stages_before(runtime_state, stage, "student_relabel_input_filtering")
    if filtering_stages:
        latest = filtering_stages[-1]
        failed_backflow_input = _input_dir(stage) / "failed_backflow_input.xyz"
        _copy_file(Path(latest["stage_dir"]) / "filtered_candidate.xyz", relabel_input)
        failed_source = Path(latest["stage_dir"]) / "filtered_failed.xyz"
        if failed_source.exists():
            _copy_file(failed_source, failed_backflow_input)
        else:
            failed_backflow_input.write_text("", encoding="utf-8")
        return {"relabel_input": relabel_input, "failed_backflow_input": failed_backflow_input}

    cfg = dict(runtime_state.get("cfg", {}) or {})
    if int(stage["teacher_round"]) == 0:
        paths = runtime_state.get("paths")
        if paths is None:
            raise ValueError("runtime_state is missing paths")
        sources = validate_path_only_xyz_list(
            cfg.get("initial_student_inputs"),
            field_name="initial_student_inputs",
            bare_filename_root=getattr(paths, "user_student_initial", paths.user_assets / "student_initial"),
        )
        if not sources:
            raise ValueError("entry relabel has no initial_student_inputs")
        _concat(sources, relabel_input)
        return {"relabel_input": relabel_input}

    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("runtime_state is missing paths")
    sources = SuccessfulStageArtifactResolver(runtime_state).teacher_round_dft_candidate_sources(
        int(stage["teacher_round"]) - 1
    )
    if not sources:
        raise ValueError("cross-teacher-round relabel has no previous DFT candidate artifacts")
    _concat(sources, relabel_input)
    return {"relabel_input": relabel_input}


def _current_effective_seed_pool(runtime_state: dict[str, Any], stage: dict[str, Any]) -> Path:
    return SuccessfulStageArtifactResolver(runtime_state).current_effective_seed_pool_before(stage).path


def _successful_sampling_window_after(
    runtime_state: dict[str, Any],
    stage: dict[str, Any],
    *,
    previous_stage_type: str,
) -> list[dict[str, Any]]:
    return [
        item.stage.stage_record
        for item in SuccessfulStageArtifactResolver(runtime_state).sampling_outputs_after_boundary(
            stage,
            boundary_stage_type=previous_stage_type,
        )
    ]


def _successful_stages_before(runtime_state: dict[str, Any], stage: dict[str, Any], stage_type: str) -> list[dict[str, Any]]:
    return [
        item.stage_record
        for item in SuccessfulStageArtifactResolver(runtime_state).successful_stages_before(stage, stage_type=stage_type)
    ]


def _input_dir(stage: dict[str, Any]) -> Path:
    path = Path(str(stage["stage_dir"])).resolve() / "input"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_file(source: Path, target: Path) -> None:
    copy_extxyz_file(source, target, allow_empty=True)


def _concat(sources: Iterable[Path], target: Path) -> None:
    concat_extxyz_files([Path(source) for source in sources if Path(source).exists()], target, allow_empty_sources=True)
