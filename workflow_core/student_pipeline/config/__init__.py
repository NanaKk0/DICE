from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from workflow_backends.student.config import resolve_student_backend_config
from workflow_core.common.io import read_json
from workflow_core.student_pipeline.config.stage_config import load_pipeline_stage_config
from workflow_core.student_pipeline.orchestration.template_loader import load_resolved_templates, resolve_template_registry
from workflow_core.training_data_assets import infer_workspace_root_from_config, validate_path_only_xyz_list
from workflow_core.student_pipeline.config.validator import (
    detect_student_subworkflow_config_mode,
    normalize_pipeline_schedule,
    reject_unknown_keys,
    require_bool,
    validate_resolved_pipeline_templates,
    validate_schedule_against_templates,
)

_PIPELINE_TEMPLATE_TOP_LEVEL_KEYS = {
    "student_backend",
    "sampling_backend",
    "train_regime",
    "initial_student_inputs",
    "source",
    "student_foundation_asset",
    "student_model",
    "student_model_identity",
    "sampling_backend_config",
    "sampling_backend_identity",
    "deepmd",
    "dp",
    "nep",
    "lammps",
    "gpumd",
    "uncertainty",
    "first_principles_handoff",
    "student_pipeline_templates",
    "student_pipeline_schedule",
}

def resolve_student_pipeline_mode_config(
    raw_cfg: dict[str, Any],
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    if detect_student_subworkflow_config_mode(raw_cfg) != "pipeline_templates":
        raise ValueError("resolve_student_pipeline_mode_config requires pipeline-template mode config")
    payload = deepcopy(raw_cfg)
    payload["config_mode"] = "pipeline_templates"
    _validate_removed_formal_fields(payload)
    reject_unknown_keys(
        payload,
        _PIPELINE_TEMPLATE_TOP_LEVEL_KEYS | {"config_mode"},
        context="top-level student_subworkflow config",
    )
    _validate_initial_student_inputs(payload, config_path=config_path)
    payload["sampling_backend"] = str(payload.get("sampling_backend", "")).strip().lower()
    if not payload["sampling_backend"]:
        raise ValueError("pipeline-template mode requires a non-empty sampling_backend")

    payload["student_pipeline_schedule"] = normalize_pipeline_schedule(payload.get("student_pipeline_schedule"))
    payload["student_pipeline_templates"] = resolve_template_registry(
        dict(payload.get("student_pipeline_templates", {}) or {}),
        config_path=config_path,
    )
    payload["resolved_student_pipeline_templates"] = load_resolved_templates(payload["student_pipeline_templates"])
    validate_resolved_pipeline_templates(
        payload["student_pipeline_templates"],
        payload["resolved_student_pipeline_templates"],
    )
    _load_referenced_stage_configs(
        payload["resolved_student_pipeline_templates"],
        sampling_backend=payload["sampling_backend"],
    )
    scheduled_stage_types = _scheduled_stage_types(
        payload["student_pipeline_schedule"],
        payload["resolved_student_pipeline_templates"],
    )
    _validate_initial_seed_pool_source(payload, config_path=config_path, scheduled_stage_types=scheduled_stage_types)
    _validate_first_principles_handoff_readiness(payload, scheduled_stage_types=scheduled_stage_types)
    _validate_train_regime(
        payload,
        has_train_stage=_pipeline_has_stage_type(payload["resolved_student_pipeline_templates"], "train"),
    )
    _validate_student_model_identity(payload)
    _validate_sampling_backend_config(payload)
    _validate_backend_recipe_boundaries(payload)
    validate_schedule_against_templates(
        payload["student_pipeline_schedule"],
        payload["resolved_student_pipeline_templates"],
    )
    return resolve_student_backend_config(payload, config_path=config_path)


def _validate_student_model_identity(payload: dict[str, Any]) -> None:
    student_model = payload.get("student_model")
    if not isinstance(student_model, dict):
        raise ValueError("pipeline-template mode requires student_model with committee_size and type_map")
    type_map = [str(item) for item in student_model.get("type_map", []) or []]
    if not type_map:
        raise ValueError("student_model.type_map must be a non-empty list")
    committee_size = int(student_model.get("committee_size", 0) or 0)
    if committee_size <= 0:
        raise ValueError("student_model.committee_size must be a positive integer")
    payload["student_model"] = {
        **dict(student_model),
        "type_map": type_map,
        "committee_size": committee_size,
    }
    for block_name in ("deepmd", "nep"):
        block = payload.get(block_name)
        if not isinstance(block, dict):
            continue
        for key in ("type_map", "committee_size"):
            if key in block:
                raise ValueError(f"{block_name}.{key} is workflow-owned; move it to student_model.{key}")
    for block_name in ("lammps", "gpumd"):
        block = payload.get(block_name)
        if isinstance(block, dict) and "type_map" in block:
            raise ValueError(f"{block_name}.type_map is workflow-owned; move it to student_model.type_map")


def _validate_sampling_backend_config(payload: dict[str, Any]) -> None:
    raw = payload.get("sampling_backend_config", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("sampling_backend_config must be an object when declared")
    for backend_name, forbidden in {
        "lammps": {"type_map", "thermostat_damp_fs", "thermostat_damp", "steps", "ensemble", "dump_freq", "thermo_freq", "model_devi_freq"},
        "gpumd": {"type_map", "steps", "steps_long", "steps_short", "ensemble", "timestep_fs", "active_interval", "dump_interval", "temperature_schedule"},
    }.items():
        block = raw.get(backend_name)
        if not isinstance(block, dict):
            continue
        misplaced = sorted(set(block) & forbidden)
        if misplaced:
            raise ValueError(
                f"sampling_backend_config.{backend_name}.{misplaced[0]} is MD or student-model owned; "
                "move MD behavior to sampling stage md/execution and type_map to student_model"
            )


def _validate_removed_formal_fields(payload: dict[str, Any]) -> None:
    if "incremental_data" in payload:
        raise ValueError("pipeline-template mode rejects incremental_data; use train_regime.continue_training")
    present_trust_aliases = sorted(set(payload) & {"student_trust_lo", "student_trust_hi"})
    if present_trust_aliases:
        raise ValueError(
            "pipeline-template mode rejects top-level student trust aliases "
            f"{present_trust_aliases}; use uncertainty.trust_lo and uncertainty.trust_hi"
        )
    uncertainty = payload.get("uncertainty")
    if isinstance(uncertainty, dict) and "source" in uncertainty:
        raise ValueError("pipeline-template uncertainty config rejects source; source is backend-implied")
    gpumd = payload.get("gpumd")
    if isinstance(gpumd, dict) and "committee_model_pattern" in gpumd:
        raise ValueError("pipeline-template gpumd config rejects committee_model_pattern; NEP model names are fixed by backend output")
    foundation = payload.get("student_foundation_asset")
    if isinstance(foundation, dict):
        present = sorted(set(foundation) & {"enabled", "strategy"})
        if present:
            raise ValueError(f"student_foundation_asset rejects redundant fields: {present}")


def _validate_initial_student_inputs(payload: dict[str, Any], *, config_path: str | Path | None) -> None:
    if "bootstrap_data" in payload:
        raise ValueError("pipeline-template mode rejects bootstrap_data; declare initial_student_inputs instead")
    bootstrap_data = dict(payload.get("bootstrap_data", {}) or {})
    if "include_rgps_in_initial_dataset" in bootstrap_data:
        raise ValueError(
            "pipeline-template mode rejects bootstrap_data.include_rgps_in_initial_dataset; "
            "declare initial_student_inputs instead"
        )
    workspace_root = infer_workspace_root_from_config(config_path)
    validate_path_only_xyz_list(
        payload.get("initial_student_inputs"),
        field_name="initial_student_inputs",
        bare_filename_root=workspace_root / "user_assets" / "student_initial",
    )


def _validate_initial_seed_pool_source(
    payload: dict[str, Any],
    *,
    config_path: str | Path | None,
    scheduled_stage_types: list[str],
) -> None:
    source_payload = payload.get("source", {}) or {}
    if not isinstance(source_payload, dict):
        raise ValueError("source must be an object when declared in pipeline-template mode")
    if "initial_seed_pool" not in source_payload and "sampling" not in scheduled_stage_types:
        return
    workspace_root = infer_workspace_root_from_config(config_path)
    validate_path_only_xyz_list(
        source_payload.get("initial_seed_pool"),
        field_name="source.initial_seed_pool",
        bare_filename_root=workspace_root / "user_assets" / "seed_pool",
    )


def _validate_first_principles_handoff_readiness(
    payload: dict[str, Any],
    *,
    scheduled_stage_types: list[str],
) -> None:
    handoff_payload = payload.get("first_principles_handoff", {}) or {}
    if not isinstance(handoff_payload, dict):
        raise ValueError("first_principles_handoff must be an object when declared in pipeline-template mode")
    if "required" in handoff_payload:
        required = require_bool(
            handoff_payload["required"],
            field_name="first_principles_handoff.required",
        )
    else:
        required = False
    if not required:
        return
    if "teacher_dft_candidate_filtering" not in scheduled_stage_types:
        raise ValueError(
            "first_principles_handoff.required needs an active student_pipeline_schedule path "
            "with teacher_dft_candidate_filtering"
        )


def _load_referenced_stage_configs(
    resolved_templates: dict[str, dict[str, Any]],
    *,
    sampling_backend: str | None,
) -> None:
    for template_payload in resolved_templates.values():
        for stage in list(template_payload.get("stages", []) or []):
            stage["stage_config"] = load_pipeline_stage_config(
                str(stage["stage"]),
                str(stage["config"]),
                mode="strict",
                sampling_backend=sampling_backend,
            )


def _pipeline_has_stage_type(
    resolved_templates: dict[str, dict[str, Any]],
    stage_type: str,
) -> bool:
    return any(
        str(stage.get("stage", "")).strip() == stage_type
        for template_payload in resolved_templates.values()
        for stage in list(template_payload.get("stages", []) or [])
    )


def _scheduled_stage_types(
    schedule: list[dict[str, int | str]],
    resolved_templates: dict[str, dict[str, Any]],
) -> list[str]:
    stage_types: list[str] = []
    for item in schedule:
        template_name = str(item["template"])
        template_payload = resolved_templates.get(template_name, {})
        for stage in list(template_payload.get("stages", []) or []):
            stage_types.append(str(stage.get("stage", "")).strip())
    return stage_types


def _validate_train_regime(payload: dict[str, Any], *, has_train_stage: bool) -> None:
    if not has_train_stage:
        return
    raw = dict(payload.get("train_regime", {}) or {})
    required = {"use_foundation_model", "continue_training"}
    if set(raw) != required:
        raise ValueError(
            "student pipeline with train stages requires train_regime.use_foundation_model "
            "and train_regime.continue_training"
        )
    require_bool(raw["use_foundation_model"], field_name="train_regime.use_foundation_model")
    require_bool(raw["continue_training"], field_name="train_regime.continue_training")


def _validate_backend_recipe_boundaries(payload: dict[str, Any]) -> None:
    backend = str(payload.get("student_backend", "")).strip().lower()
    if backend not in {"nep", "deepmd", "dp"}:
        return
    checked: set[str] = set()
    for template_payload in dict(payload.get("resolved_student_pipeline_templates", {}) or {}).values():
        for stage in list(template_payload.get("stages", []) or []):
            if str(stage.get("stage")) != "train":
                continue
            stage_cfg = dict(stage.get("stage_config", {}) or {})
            selector_path = stage_cfg.get("train_recipe_ref")
            if not selector_path:
                continue
            selector = Path(str(selector_path)).resolve()
            if str(selector) in checked:
                continue
            checked.add(str(selector))
            if backend == "nep":
                _validate_nep_selector_recipe_boundaries(selector)
            else:
                _validate_deepmd_selector_recipe_boundaries(selector)


def _validate_nep_selector_recipe_boundaries(selector_path: Path) -> None:
    if selector_path.suffix.lower() in {".in", ".nep"}:
        if not selector_path.is_file():
            raise FileNotFoundError(f"NEP native recipe not found: {selector_path}")
        return
    selector_payload = dict(read_json(selector_path) or {})
    _reject_legacy_train_recipe_selector_keys(selector_path, selector_payload)
    _reject_nep_json_native_recipe_fields(selector_path, selector_payload)
    if isinstance(selector_payload.get("nep"), dict) and not any(
        selector_payload.get(key) for key in ("scratch_entry", "foundation_entry", "continuation")
    ):
        _validate_nep_recipe_identity_boundary(selector_path, selector_payload)
        return
    for key in ("scratch_entry", "foundation_entry", "continuation"):
        recipe_ref = selector_payload.get(key)
        if not recipe_ref:
            continue
        recipe_path = _resolve_config_reference(recipe_ref, anchor=selector_path)
        if recipe_path.suffix.lower() in {".in", ".nep"}:
            if not recipe_path.is_file():
                raise FileNotFoundError(f"NEP native recipe not found: {recipe_path}")
            continue
        recipe_payload = dict(read_json(recipe_path) or {})
        _reject_nep_json_native_recipe_fields(recipe_path, recipe_payload)
        _validate_nep_recipe_identity_boundary(recipe_path, recipe_payload)


def _validate_deepmd_selector_recipe_boundaries(selector_path: Path) -> None:
    selector_payload = dict(read_json(selector_path) or {})
    _reject_legacy_train_recipe_selector_keys(selector_path, selector_payload)
    selector_identity = sorted(set(selector_payload) & {"committee_size", "type_map"})
    if selector_identity:
        raise ValueError(
            "DeepMD backend selector must not define student identity fields "
            f"{selector_identity}; move them to subworkflow deepmd identity: {selector_path}"
        )
    if isinstance(selector_payload.get("dp_config"), dict) and not any(
        selector_payload.get(key) for key in ("scratch_entry", "foundation_entry", "continuation")
    ):
        _validate_deepmd_recipe_identity_boundary(selector_path, selector_payload)
        return
    for key in ("scratch_entry", "foundation_entry", "continuation"):
        recipe_ref = selector_payload.get(key)
        if not recipe_ref:
            continue
        recipe_path = _resolve_config_reference(recipe_ref, anchor=selector_path)
        recipe_payload = dict(read_json(recipe_path) or {})
        _validate_deepmd_recipe_identity_boundary(recipe_path, recipe_payload)


def _reject_legacy_train_recipe_selector_keys(selector_path: Path, selector_payload: dict[str, Any]) -> None:
    present = sorted(set(selector_payload) & {"bootstrap_train_config", "finetune_train_config"})
    if present:
        raise ValueError(
            "pipeline-template train recipe selector rejects legacy selector fields "
            f"{present}; use scratch_entry, foundation_entry, and continuation: {selector_path}"
        )


def _validate_nep_recipe_identity_boundary(recipe_path: Path, recipe_payload: dict[str, Any]) -> None:
    nep_payload = dict(recipe_payload.get("nep", recipe_payload) or {})
    misplaced = sorted(set(nep_payload) & {"type_map", "committee_size"})
    if misplaced:
        raise ValueError(
            "backend train recipe must not define student identity fields "
            f"{misplaced}; move them to subworkflow student identity: {recipe_path}"
        )


def _reject_nep_json_native_recipe_fields(recipe_path: Path, recipe_payload: dict[str, Any]) -> None:
    nep_payload = dict(recipe_payload.get("nep", recipe_payload) or {})
    native_recipe_keys = {
        "arch",
        "batch",
        "basis_size",
        "cutoff",
        "generation",
        "l_max",
        "lambda_1",
        "lambda_2",
        "lambda_e",
        "lambda_f",
        "lambda_v",
        "loss",
        "loss_stage_1",
        "loss_stage_2",
        "n_max",
        "neuron",
        "population",
        "save_potential",
        "training",
        "training_stage_1",
        "training_stage_2",
        "version",
        "zbl",
    }
    present = sorted(set(nep_payload) & native_recipe_keys)
    if present:
        raise ValueError(
            "NEP JSON train recipe fields that describe native nep.in content are retired "
            f"{present}; create a user-maintained nep.in file and point train_recipe_ref at it: {recipe_path}"
        )


def _validate_deepmd_recipe_identity_boundary(recipe_path: Path, recipe_payload: dict[str, Any]) -> None:
    model_payload = dict(dict(recipe_payload.get("dp_config", {}) or {}).get("model", {}) or {})
    misplaced = sorted(set(model_payload) & {"type_map"})
    if misplaced:
        raise ValueError(
            "DeepMD backend train recipe must not define student identity fields "
            f"{misplaced}; move them to subworkflow deepmd identity: {recipe_path}"
        )


def _resolve_config_reference(value: str | Path, *, anchor: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (anchor.parent / path).resolve()
