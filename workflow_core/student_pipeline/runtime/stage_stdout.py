from __future__ import annotations

from pathlib import Path
from typing import Any


_MISSING = object()


def write_student_stage_stdout_log(stage: dict[str, Any], stage_payload: dict[str, Any]) -> Path:
    stdout_path = _stdout_path(stage)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(render_student_stage_stdout(str(stage["stage"]), stage_payload), encoding="utf-8")
    return stdout_path


def render_student_stage_stdout(stage_type: str, stage_payload: dict[str, Any]) -> str:
    stage = str(stage_type).strip()
    payload = dict(stage_payload or {})
    renderers = {
        "relabel": _render_relabel,
        "train": _render_train,
        "sampling": _render_sampling,
        "student_relabel_input_filtering": _render_filtering,
        "seed_pool_refresh": _render_seed_pool_refresh,
        "teacher_dft_candidate_filtering": _render_teacher_dft_candidate_filtering,
    }
    renderer = renderers.get(stage)
    if renderer is None:
        raise ValueError(f"unsupported student stage stdout contract: {stage_type!r}")
    return _without_path_lines(renderer(payload)).rstrip() + "\n"


def _render_relabel(payload: dict[str, Any]) -> list[str]:
    label_counts = dict(payload.get("label_counts", {}) or {})
    input_count = _first_int(payload, "input_structures", "input_count", default=_int(label_counts.get("input", 0)))
    train_count = _first_int(payload, "train_structures", "train_count", default=_int(label_counts.get("train", 0)))
    dft_count = _first_int(payload, "dft_structures", "dft_count", default=_int(label_counts.get("dft", 0)))
    discard_count = _first_int(payload, "discard_structures", "discard_count", default=_int(label_counts.get("discard", 0)))
    lines = [
        "stage: relabel",
        "routing:",
        f"  input_total: {input_count}",
        f"  train_total: {train_count}",
        f"  active_learning_dft_total: {dft_count}",
        f"  discard_total: {discard_count}",
    ]
    if bool(payload.get("failed_backflow_enabled", False)):
        failed = dict(payload.get("failed_backflow_summary", {}) or {})
        failed_counts = dict(failed.get("label_counts", {}) or failed)
        lines.extend(
            [
                "failed_backflow:",
                f"  input_total: {_int(failed_counts.get('input', 0))}",
                f"  train_total: {_int(failed_counts.get('train', 0))}",
                f"  active_learning_dft_total: {_int(failed_counts.get('dft', 0))}",
                f"  discard_total: {_int(failed_counts.get('discard', 0))}",
            ]
        )
    return lines


def _render_train(payload: dict[str, Any]) -> list[str]:
    member_total = _first_int(payload, "committee_member_count", "member_count", "total_member_count")
    successful = _first_int(payload, "successful_member_count", "success_member_count", "usable_member_count")
    failed = _first_int(payload, "failed_member_count", default=max(0, member_total - successful) if member_total else 0)
    return [
        "stage: train",
        "training:",
        f"  training_mode: {_first_str(payload, 'training_mode', 'effective_train_recipe_branch')}",
        f"  initialization_mode: {_first_str(payload, 'initialization_mode')}",
        f"  dataset_mode: {_first_str(payload, 'dataset_mode')}",
        f"  committee_member_total: {member_total}",
        f"  successful_member_count: {successful}",
        f"  failed_member_count: {failed}",
        f"  completed: {_bool_text(payload.get('completed', successful >= member_total if member_total else False))}",
    ]


def _render_sampling(payload: dict[str, Any]) -> list[str]:
    return [
        "stage: sampling",
        "seed_selection:",
        f"  selected_seed_total: {_first_int(payload, 'selected_seed_count')}",
        f"  selected_seed_by_system: {_format_counts(payload.get('selected_seed_count_per_system'))}",
        "structures:",
        f"  accurate_total: {_first_int(payload, 'accurate_count', 'accurate_total')}",
        f"  accurate_by_system: {_format_counts(_first_mapping(payload, 'accurate_count_per_system', 'accurate_by_system'))}",
        f"  candidate_total: {_first_int(payload, 'candidate_count', 'candidate_total')}",
        f"  candidate_by_system: {_format_counts(_first_mapping(payload, 'candidate_count_per_system', 'candidate_by_system'))}",
        f"  failed_total: {_first_int(payload, 'failed_count', 'failed_total')}",
        f"  failed_by_system: {_format_counts(_first_mapping(payload, 'failed_count_per_system', 'failed_by_system'))}",
        (
            f"tasks: total={_first_int(payload, 'task_count', 'task_total')}, "
            f"succeeded={_first_int(payload, 'succeeded_task_count', 'successful_task_count')}, "
            f"failed={_first_int(payload, 'failed_task_count')}"
        ),
    ]


def _render_filtering(payload: dict[str, Any]) -> list[str]:
    return [
        "stage: student_relabel_input_filtering",
        f"selection_mode: {_first_str(payload, 'selection_mode')}",
        "candidate:",
        f"  input_total: {_first_int(payload, 'candidate_input_count')}",
        f"  selected_total: {_first_int(payload, 'candidate_selected_count')}",
        f"  selected_by_system: {_format_counts(payload.get('candidate_selected_count_per_system'))}",
        "failed:",
        f"  input_total: {_first_int(payload, 'failed_input_count')}",
        f"  selected_total: {_first_int(payload, 'failed_selected_count')}",
        f"  selected_by_system: {_format_counts(payload.get('failed_selected_count_per_system'))}",
    ]


def _render_seed_pool_refresh(payload: dict[str, Any]) -> list[str]:
    return [
        "stage: seed_pool_refresh",
        "inputs:",
        f"  current_seed_pool_total: {_first_int(payload, 'current_seed_pool_count')}",
        f"  candidate_increment_total: {_first_int(payload, 'candidate_increment_count')}",
        f"  failed_increment_total: {_first_int(payload, 'failed_increment_count')}",
        "rescored:",
        f"  candidate_increment_total: {_first_int(payload, 'rescored_candidate_increment_count')}",
        f"  failed_increment_total: {_first_int(payload, 'rescored_failed_increment_count')}",
        f"  seed_pool_total: {_first_int(payload, 'rescored_seed_pool_count')}",
        "result:",
        f"  final_seed_pool_total: {_first_int(payload, 'seed_pool_size_after')}",
        f"  final_seed_pool_by_system: {_format_counts(payload.get('selected_count_per_system'))}",
    ]


def _render_teacher_dft_candidate_filtering(payload: dict[str, Any]) -> list[str]:
    return [
        "stage: teacher_dft_candidate_filtering",
        f"selection_mode: {_first_str(payload, 'selection_mode')}",
        "candidate:",
        f"  input_total: {_first_int(payload, 'input_candidate_count', 'input_count')}",
        f"  input_by_system: {_format_counts(_first_mapping(payload, 'input_count_per_system', 'input_candidate_count_per_system'))}",
        "screened:",
        f"  screened_total: {_first_int(payload, 'screened_count')}",
        f"  screened_by_system: {_format_counts(payload.get('selected_count_per_system'))}",
    ]


def _stdout_path(stage: dict[str, Any]) -> Path:
    log_paths = dict(stage.get("log_paths", {}) or {})
    if not log_paths.get("stdout_path"):
        raise ValueError("stage is missing log_paths.stdout_path")
    return Path(str(log_paths["stdout_path"])).resolve()


def _without_path_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if not _looks_like_path_line(line))


def _looks_like_path_line(line: str) -> bool:
    lowered = line.lower()
    if "/" in line or "\\" in line:
        return True
    path_words = ("_path", " path:", "artifact:", "config:", "input_file:", "output_file:")
    return any(word in lowered for word in path_words)


def _first_int(payload: dict[str, Any], *keys: str, default: Any = _MISSING) -> int:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return _int(payload.get(key))
    if default is _MISSING:
        raise ValueError(f"missing required payload field: {' or '.join(keys)}")
    return int(default)


def _first_str(payload: dict[str, Any], *keys: str, default: Any = _MISSING) -> str:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            return str(payload.get(key))
    if default is _MISSING:
        raise ValueError(f"missing required payload field: {' or '.join(keys)}")
    return default


def _first_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _format_counts(raw: Any) -> str:
    if not isinstance(raw, dict) or not raw:
        return "none"
    return ", ".join(f"{key}={_int(raw[key])}" for key in sorted(raw))


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"
