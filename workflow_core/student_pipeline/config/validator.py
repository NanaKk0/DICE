from __future__ import annotations

from typing import Any


NEW_MODE_KEYS = {
    "student_pipeline_templates",
    "student_pipeline_schedule",
}

LEGACY_MODE_KEYS = {
    "bootstrap_data",
    "round0",
    "dp_domain_strategy",
    "training_policy",
    "within_outer_round_training_mode",
    "sampling_schedule",
    "sampling_profiles",
    "sampling_seed_policy",
    "seed_pool_policy",
    "failed_rescue_labeling_policy",
    "bootstrap_committee_summary",
    "max_student_rounds",
    "stop_ratio_threshold",
    "label_config",
    "student_train_committee_config",
    "frame_selection_policy",
    "uncertainty_policy",
    "candidate_train_selection",
    "seed_pool_refresh",
    "dp_replay",
    "accumulate_dp_train_across_outer_rounds",
    "incremental_data",
}

VALID_STAGE_TYPES = {
    "relabel",
    "train",
    "sampling",
    "seed_pool_refresh",
    "student_relabel_input_filtering",
    "teacher_dft_candidate_filtering",
}


def canonical_stage_type(stage_type: str) -> str:
    return str(stage_type or "").strip()


def detect_student_subworkflow_config_mode(raw_cfg: dict[str, Any]) -> str:
    present_keys = set(raw_cfg)
    has_new_mode = bool(present_keys & NEW_MODE_KEYS)
    has_legacy_mode = bool(present_keys & LEGACY_MODE_KEYS)
    if has_new_mode and has_legacy_mode:
        retired_keys = sorted(present_keys & LEGACY_MODE_KEYS)
        raise ValueError(
            "legacy student_subworkflow config keys and pipeline-template mode keys are mutually exclusive: "
            f"{retired_keys}"
        )
    if has_new_mode:
        missing = sorted(NEW_MODE_KEYS - present_keys)
        if missing:
            raise ValueError(
                "pipeline-template mode requires both student_pipeline_templates and student_pipeline_schedule"
            )
        return "pipeline_templates"
    retired_keys = sorted(present_keys & LEGACY_MODE_KEYS)
    if retired_keys:
        raise ValueError(
            "retired legacy student_subworkflow config keys are no longer supported; "
            f"use pipeline-template mode instead: {retired_keys}"
        )
    raise ValueError(
        "student_subworkflow config requires pipeline-template mode with both "
        "student_pipeline_templates and student_pipeline_schedule"
    )


def normalize_pipeline_schedule(schedule_payload: Any) -> list[dict[str, int | str]]:
    if not isinstance(schedule_payload, list) or not schedule_payload:
        raise ValueError("student_pipeline_schedule must be a non-empty list")
    normalized: list[dict[str, int | str]] = []
    for index, item in enumerate(schedule_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"student_pipeline_schedule entry #{index} must be an object")
        extra_keys = sorted(set(item) - {"template", "repeat"})
        if extra_keys:
            raise ValueError(
                f"student_pipeline_schedule entry #{index} only supports template and repeat; got {extra_keys}"
            )
        template_name = str(item.get("template", "")).strip()
        if not template_name:
            raise ValueError(f"student_pipeline_schedule entry #{index} must define a non-empty template")
        repeat = require_positive_int_not_bool(
            item.get("repeat", 1),
            field_name=f"student_pipeline_schedule entry #{index} repeat",
        )
        if repeat <= 0:
            raise ValueError(f"student_pipeline_schedule entry #{index} repeat must be > 0")
        normalized.append({"template": template_name, "repeat": repeat})
    return normalized


def require_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def require_positive_int_not_bool(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return int(value)


def reject_unknown_keys(payload: dict[str, Any], allowed_keys: set[str], *, context: str) -> None:
    unknown = sorted(set(payload) - set(allowed_keys))
    if unknown:
        raise ValueError(f"unknown {context} keys: {unknown}")


def validate_resolved_pipeline_templates(
    template_registry: dict[str, str],
    resolved_templates: dict[str, dict[str, Any]],
) -> None:
    for template_name, resolved in resolved_templates.items():
        stages = resolved.get("stages")
        if template_name not in template_registry:
            raise ValueError(f"resolved template '{template_name}' is missing from student_pipeline_templates")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"student_pipeline_template '{template_name}' must define a non-empty stages list")
        template_path = resolved.get("template_path", template_registry[template_name])
        template_keys = {"stages"}
        # The loader only keeps normalized fields, so this validator assumes the original
        # shape was screened by template_loader through read_json + stages extraction.
        for index, stage_entry in enumerate(stages, start=1):
            if not isinstance(stage_entry, dict):
                raise ValueError(f"student_pipeline_template '{template_name}' stage #{index} must be an object")
            entry_keys = set(stage_entry)
            if entry_keys - {"stage", "config"}:
                extra_keys = sorted(entry_keys - {"stage", "config"})
                raise ValueError(
                    f"student_pipeline_template '{template_name}' stage #{index} only supports stage and config; got {extra_keys}"
                )
            stage_type = str(stage_entry.get("stage", "")).strip()
            if stage_type not in VALID_STAGE_TYPES:
                raise ValueError(
                    f"student_pipeline_template '{template_name}' stage #{index} has unsupported stage type '{stage_type}'"
                )
            config_path = str(stage_entry.get("config", "")).strip()
            if not config_path:
                raise ValueError(
                    f"student_pipeline_template '{template_name}' stage #{index} must define a non-empty config path"
                )
        _validate_stage_predecessors(template_name, stages)


def _validate_stage_predecessors(template_name: str, stages: list[dict[str, Any]]) -> None:
    for index, stage_entry in enumerate(stages, start=1):
        stage_type = canonical_stage_type(str(stage_entry.get("stage", "")).strip())
        previous_type = (
            canonical_stage_type(str(stages[index - 2].get("stage", "")).strip()) if index > 1 else None
        )
        if stage_type == "student_relabel_input_filtering" and previous_type != "sampling":
            raise ValueError(
                f"student_pipeline_template '{template_name}' stage #{index} requires sampling as its predecessor"
            )
        if stage_type == "relabel" and index > 1 and previous_type != "student_relabel_input_filtering":
            raise ValueError(
                f"student_pipeline_template '{template_name}' stage #{index} requires student_relabel_input_filtering as its predecessor"
            )
        if stage_type == "train":
            train_follows_relabel = previous_type == "relabel"
            train_follows_teacher_filtering = previous_type == "teacher_dft_candidate_filtering" and index > 2 and canonical_stage_type(
                str(stages[index - 3].get("stage", "")).strip()
            ) == "relabel"
            if not (train_follows_relabel or train_follows_teacher_filtering):
                raise ValueError(
                    f"student_pipeline_template '{template_name}' stage #{index} requires relabel or "
                    "teacher_dft_candidate_filtering after relabel as its predecessor"
                )
        is_standalone_teacher_filtering = (
            stage_type == "teacher_dft_candidate_filtering" and index == 1 and len(stages) == 1
        )
        if (
            stage_type == "teacher_dft_candidate_filtering"
            and previous_type != "relabel"
            and not is_standalone_teacher_filtering
        ):
            raise ValueError(
                f"student_pipeline_template '{template_name}' stage #{index} requires relabel as its predecessor"
            )


def validate_schedule_against_templates(
    schedule: list[dict[str, int | str]],
    resolved_templates: dict[str, dict[str, Any]],
) -> None:
    for index, item in enumerate(schedule, start=1):
        template_name = str(item["template"])
        if template_name not in resolved_templates:
            raise ValueError(
                f"student_pipeline_schedule entry #{index} references unknown template '{template_name}'"
            )
        stages = list(resolved_templates[template_name].get("stages", []) or [])
        first_stage = canonical_stage_type(str((stages[0] if stages else {}).get("stage", "")).strip())
        if index == 1:
            if first_stage != "relabel":
                raise ValueError(
                    f"the first student_pipeline invocation must begin with relabel, got '{first_stage}'"
                )
            continue
        if first_stage == "teacher_dft_candidate_filtering":
            if index != len(schedule):
                raise ValueError(
                    "teacher_dft_candidate_filtering aggregate template must be the final "
                    "student_pipeline_schedule invocation"
                )
            if int(item["repeat"]) != 1:
                raise ValueError(
                    "teacher_dft_candidate_filtering aggregate template must use repeat 1"
                )
            continue
        if first_stage not in {"sampling", "relabel"}:
            raise ValueError(
                f"non-first student_pipeline invocations must begin with sampling or relabel, got '{first_stage}'"
            )
