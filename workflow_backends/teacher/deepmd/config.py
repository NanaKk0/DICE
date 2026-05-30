from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_backends.deepmd_common.execution import normalize_nan_watchdog_config
from workflow_backends.teacher.deepmd.family import get_deepmd_family_adapter
from workflow_core.common.io import read_json
from workflow_core.teacher.finetune_config import flatten_teacher_finetune_config, load_teacher_finetune_config_boundary


SUPPORTED_DPA4_INIT_MODES = ("scratch", "foundation_finetune")
WORKFLOW_OWNED_NATIVE_FIELDS = {"init_mode", "foundation_model"}
FOUNDATION_WORKFLOW_OWNED_NATIVE_FIELDS = {"model_family", "type_map", "training_data", "foundation_model", "init_mode"}
RETIRED_DEEPMD_EXECUTION_FIELDS = frozenset({"train_command", "freeze_command", "dp_command", "env"})


def _resolve_relative_to_config(value: str | Path, *, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _validate_teacher_seeds(train_cfg: dict[str, Any], *, config_path: Path) -> list[int]:
    teacher_num = int(train_cfg.get("teacher_num", 0) or 0)
    if teacher_num <= 0:
        raise ValueError(f"Invalid DeePMD teacher config {config_path}: teacher_num must be positive")
    raw = train_cfg.get("teacher_seeds")
    if not isinstance(raw, list):
        raise ValueError(
            f"Invalid DeePMD teacher config {config_path}: teacher_seeds must be an explicit list with "
            f"exactly teacher_num={teacher_num} entries"
        )
    seeds = [int(item) for item in raw]
    if len(seeds) != teacher_num:
        raise ValueError(
            f"Invalid DeePMD teacher config {config_path}: teacher_seeds has {len(seeds)} entries, "
            f"but teacher_num={teacher_num}"
        )
    return seeds


def _native_template_payload(payload: dict[str, Any]) -> dict[str, Any]:
    native = dict(payload)
    native.pop("execution", None)
    return native


def _normalize_teacher_execution(payload: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    raw_execution = payload.get("execution", {})
    if raw_execution is None:
        raw_execution = {}
    if not isinstance(raw_execution, dict):
        raise TypeError(f"{field_name} must be an object")
    present = sorted(set(raw_execution) & RETIRED_DEEPMD_EXECUTION_FIELDS)
    if present:
        raise ValueError(
            f"{field_name} contains retired environment/command field(s): {', '.join(present)}; "
            "declare executable 'dp' in environment_profile.software_profiles instead"
        )
    allowed = {"max_retries_per_member", "retry_seed_stride", "nan_watchdog"}
    unknown = sorted(str(key) for key in raw_execution if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported field(s): {', '.join(unknown)}")
    watchdog_payload = {"nan_watchdog": raw_execution.get("nan_watchdog", {})}
    watchdog = normalize_nan_watchdog_config(watchdog_payload, field_name=f"{field_name}.nan_watchdog")
    return {
        "max_retries_per_member": max(0, int(raw_execution.get("max_retries_per_member", 2))),
        "retry_seed_stride": max(1, int(raw_execution.get("retry_seed_stride", 1000))),
        "nan_watchdog": {
            "enabled": watchdog.enabled,
            "terminate_timeout_seconds": watchdog.terminate_timeout_seconds,
            "watch_lcurve_fallback": watchdog.watch_lcurve_fallback,
            "lcurve_poll_interval_seconds": watchdog.lcurve_poll_interval_seconds,
        },
    }


def load_deepmd_committee_finetune_config(config_path: str | Path) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    bounded = load_teacher_finetune_config_boundary(resolved, backend_name="deepmd")
    train_cfg = flatten_teacher_finetune_config(bounded)
    model_family = str(train_cfg.get("model_family", "")).strip().lower()
    if not model_family:
        raise ValueError(f"Invalid DeePMD teacher config {resolved}: model_family is required")
    adapter = get_deepmd_family_adapter(model_family)

    native_config = _native_template_payload(dict(train_cfg.get("backend_native", {}) or {}))
    misplaced = sorted(set(native_config) & WORKFLOW_OWNED_NATIVE_FIELDS)
    if misplaced:
        raise ValueError(
            f"Invalid DeePMD teacher config {resolved}: workflow-owned field(s) {misplaced} must be under "
            "'workflow', not the native 'deepmd' block"
        )

    init_mode = str(train_cfg.get("init_mode", "scratch")).strip().lower()
    if init_mode not in SUPPORTED_DPA4_INIT_MODES:
        supported = ", ".join(SUPPORTED_DPA4_INIT_MODES)
        raise ValueError(f"Invalid DeePMD teacher config {resolved}: unsupported init_mode={init_mode!r}; expected one of: {supported}")

    type_map = [str(item) for item in train_cfg.get("type_map", []) or []]
    if not type_map:
        raise ValueError(f"Invalid DeePMD teacher config {resolved}: type_map must be a non-empty list")

    teacher_seeds = _validate_teacher_seeds(train_cfg, config_path=resolved)
    adapter.validate_template(native_config, type_map=type_map, template_path=resolved)

    execution = _normalize_teacher_execution(train_cfg, field_name="deepmd_teacher.execution")

    normalized = dict(train_cfg)
    normalized["model_family"] = adapter.name
    normalized["init_mode"] = init_mode
    normalized["teacher_num"] = int(train_cfg["teacher_num"])
    normalized["teacher_seeds"] = teacher_seeds
    normalized["type_map"] = type_map
    normalized["valid_fraction"] = float(train_cfg.get("valid_fraction", 0.1))
    if not 0.0 < normalized["valid_fraction"] < 1.0:
        raise ValueError(f"Invalid DeePMD teacher config {resolved}: valid_fraction must be between 0 and 1")
    normalized["split_seed"] = int(train_cfg.get("split_seed", 42))
    normalized["native_config"] = native_config
    normalized["execution"] = execution
    return normalized


def load_deepmd_foundation_train_config(config_path: str | Path) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"DeePMD teacher foundation train config must be an object: {resolved}")
    workflow = payload.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError(f"DeePMD teacher foundation train config requires a 'workflow' block: {resolved}")
    raw_native_config = payload.get("deepmd")
    if not isinstance(raw_native_config, dict):
        raise ValueError(f"DeePMD teacher foundation train config requires a 'deepmd' block: {resolved}")
    misplaced = sorted(set(raw_native_config) & FOUNDATION_WORKFLOW_OWNED_NATIVE_FIELDS)
    if misplaced:
        raise ValueError(
            f"Invalid DeePMD teacher foundation config {resolved}: workflow-owned field(s) {misplaced} must be "
            "under 'workflow', not the native 'deepmd' block"
        )

    model_family = str(workflow.get("model_family", "")).strip().lower()
    if not model_family:
        raise ValueError(f"DeePMD teacher foundation workflow.model_family is required: {resolved}")
    adapter = get_deepmd_family_adapter(model_family)
    type_map = [str(item) for item in workflow.get("type_map", []) or []]
    if not type_map:
        raise ValueError(f"DeePMD teacher foundation workflow.type_map must be a non-empty list: {resolved}")
    training_data = workflow.get("training_data")
    if not isinstance(training_data, dict):
        raise ValueError(f"DeePMD teacher foundation workflow.training_data must be an object: {resolved}")
    base_datasets = training_data.get("base_datasets")
    if not isinstance(base_datasets, list) or not base_datasets:
        raise ValueError(
            f"DeePMD teacher foundation workflow.training_data.base_datasets must be a non-empty list: {resolved}"
        )

    native_config = _native_template_payload(raw_native_config)
    adapter.validate_template(native_config, type_map=type_map, template_path=resolved)
    execution = _normalize_teacher_execution(raw_native_config, field_name="deepmd_foundation.execution")
    return {
        "backend_name": "deepmd",
        "workflow": dict(workflow),
        "backend_native": dict(native_config),
        "native_config": dict(native_config),
        "source_path": str(resolved),
        "model_family": adapter.name,
        "type_map": type_map,
        "training_data": dict(training_data),
        "valid_fraction": float(workflow.get("valid_fraction", 0.1)),
        "split_seed": int(workflow.get("split_seed", 42)),
        "seed": int(workflow.get("seed", 0)),
        "execution": execution,
    }
