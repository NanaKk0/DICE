from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_backends.deepmd_common.execution import (
    normalize_nan_watchdog_config,
)
from workflow_backends.student.config import load_train_recipe_selector_paths


@dataclass(slots=True)
class DPCommitteeTrainConfig:
    committee_size: int
    type_map: list[str]
    model_family: str
    valid_fraction: float
    split_seed: int
    split_mode: str
    model_seeds: list[int]
    dp_command: list[str]
    freeze_command: list[str]
    reuse_existing_split: bool
    reuse_existing_model: bool
    max_retries_per_member: int
    retry_seed_stride: int
    nan_watchdog_enabled: bool
    nan_terminate_timeout_seconds: float
    nan_watch_lcurve_fallback: bool
    nan_lcurve_poll_interval_seconds: float
    recommend_disp_freq: int | None
    recommend_save_freq_mode: str | None
    scratch_entry_train_config: Path
    foundation_entry_train_config: Path
    continuation_train_config: Path
    env: dict[str, str]

    def select_train_config(self, training_stage: str | None = None) -> Path:
        stage = str(training_stage or "scratch_entry").strip().lower()
        if stage in {"scratch_entry", "bootstrap"}:
            return self.scratch_entry_train_config
        if stage == "foundation_entry":
            return self.foundation_entry_train_config
        if stage in {"continuation", "finetune"}:
            return self.continuation_train_config
        return self.scratch_entry_train_config


def load_committee_train_config(
    config_path: str | Path,
    *,
    identity: dict[str, Any] | None = None,
) -> DPCommitteeTrainConfig:
    config_path = Path(config_path).resolve()

    def _discover_workspace_root(path: Path) -> Path:
        for candidate in [path.parent, *path.parent.parents]:
            if (candidate / "workflow.py").exists() and (candidate / "configs").is_dir():
                return candidate
        return path.parent.parent

    workspace_root = _discover_workspace_root(config_path)

    with open(config_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    discovered_identity = _discover_subworkflow_identity(config_path)
    identity_payload = _deep_update(discovered_identity, dict(identity or {}))
    payload = _deep_update(_deepmd_training_defaults(identity_payload), payload)
    _reject_retired_student_execution_fields(payload, config_path=config_path)
    model_family = _normalize_student_model_family(payload.get("model_family", identity_payload.get("model_family")))
    dp_command, freeze_command = _student_deepmd_commands(model_family)
    execution = _student_execution_controls(payload)

    committee_size = int(identity_payload.get("committee_size", payload.get("committee_size", 4)))
    if committee_size <= 0:
        raise ValueError("committee_size must be positive")

    valid_fraction = float(payload.get("valid_fraction", 0.1))
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")

    split_seed = int(payload.get("split_seed", 42))
    split_mode = "system"

    raw_model_seeds = payload.get("model_seeds")
    if raw_model_seeds is None:
        model_seeds = [split_seed + 10000 + idx for idx in range(committee_size)]
    else:
        model_seeds = [int(item) for item in raw_model_seeds]
        if len(model_seeds) < committee_size:
            raise ValueError(
                f"model_seeds has {len(model_seeds)} entries, but committee_size={committee_size}"
            )
        model_seeds = model_seeds[:committee_size]

    def _resolve_config_path(value: str | Path | None, default: str | Path) -> Path:
        resolved = Path(value if value is not None else default)
        if not resolved.is_absolute():
            candidate_from_root = (workspace_root / resolved).resolve()
            candidate_from_config = (config_path.parent / resolved).resolve()
            resolved = candidate_from_root if candidate_from_root.exists() else candidate_from_config
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Base DP train config not found: {resolved}")
        return resolved

    selector_paths = load_train_recipe_selector_paths(config_path)
    if not selector_paths and _extract_dp_config_payload(payload):
        selector_paths = {
            "scratch_entry": str(config_path),
            "foundation_entry": str(config_path),
            "continuation": str(config_path),
        }
    scratch_entry_train_config = _resolve_config_path(
        selector_paths.get("scratch_entry", Path("configs") / "student" / "deepmd" / "committee_base.json"),
        Path("configs") / "student" / "deepmd" / "committee_base.json",
    )
    foundation_entry_train_config = _resolve_config_path(
        selector_paths.get("foundation_entry", scratch_entry_train_config),
        scratch_entry_train_config,
    )
    continuation_train_config = _resolve_config_path(
        selector_paths.get("continuation", foundation_entry_train_config),
        foundation_entry_train_config,
    )
    type_map = [str(item) for item in identity_payload.get("type_map", []) or []]
    if not type_map:
        type_map = _load_legacy_recipe_type_map(scratch_entry_train_config)
    if not type_map:
        raise ValueError("DeepMD committee config requires student_model.type_map student identity")
    return DPCommitteeTrainConfig(
        committee_size=committee_size,
        type_map=type_map,
        model_family=model_family,
        valid_fraction=valid_fraction,
        split_seed=split_seed,
        split_mode=split_mode,
        model_seeds=model_seeds,
        dp_command=dp_command,
        freeze_command=freeze_command,
        reuse_existing_split=bool(payload.get("reuse_existing_split", True)),
        reuse_existing_model=bool(payload.get("reuse_existing_model", True)),
        max_retries_per_member=execution.max_retries_per_member,
        retry_seed_stride=execution.retry_seed_stride,
        nan_watchdog_enabled=execution.nan_watchdog.enabled,
        nan_terminate_timeout_seconds=execution.nan_watchdog.terminate_timeout_seconds,
        nan_watch_lcurve_fallback=execution.nan_watchdog.watch_lcurve_fallback,
        nan_lcurve_poll_interval_seconds=execution.nan_watchdog.lcurve_poll_interval_seconds,
        recommend_disp_freq=100,
        recommend_save_freq_mode="keep_existing" if execution.nan_watchdog.enabled else "same_as_disp",
        scratch_entry_train_config=scratch_entry_train_config,
        foundation_entry_train_config=foundation_entry_train_config,
        continuation_train_config=continuation_train_config,
        env=execution.env,
    )


@dataclass(frozen=True, slots=True)
class _StudentExecutionControls:
    env: dict[str, str]
    max_retries_per_member: int
    retry_seed_stride: int
    nan_watchdog: Any


def _student_execution_controls(payload: dict[str, Any]) -> _StudentExecutionControls:
    execution = payload.get("execution", {})
    if execution is None:
        execution = {}
    if not isinstance(execution, dict):
        raise TypeError("DeepMD student training execution must be an object")
    retries_value = execution.get("max_retries_per_member", payload.get("max_retries_per_member", 2))
    seed_stride_value = execution.get("retry_seed_stride", payload.get("retry_seed_stride", 1000))
    watchdog_payload = dict(payload)
    if "nan_watchdog" in execution:
        watchdog_payload["nan_watchdog"] = execution["nan_watchdog"]
    return _StudentExecutionControls(
        env={},
        max_retries_per_member=max(0, int(retries_value)),
        retry_seed_stride=max(1, int(seed_stride_value)),
        nan_watchdog=normalize_nan_watchdog_config(
            watchdog_payload,
            field_name="deepmd.training.nan_watchdog",
        ),
    )


def _normalize_student_model_family(value: Any) -> str:
    family = str(value or "deepmd").strip().lower()
    if family in {"dp", "standard"}:
        return "deepmd"
    if family in {"deepmd", "dpa4"}:
        return family
    raise ValueError(f"Unsupported DeepMD student model_family={value!r}; expected 'deepmd' or 'dpa4'")


def _student_deepmd_commands(model_family: str) -> tuple[list[str], list[str]]:
    family = _normalize_student_model_family(model_family)
    if family == "dpa4":
        return ["dp", "--pt", "train"], ["dp", "--pt", "freeze", "-o", "graph.pb"]
    return ["dp", "train"], ["dp", "freeze", "-o", "graph.pb"]


def _reject_retired_student_execution_fields(payload: dict[str, Any], *, config_path: Path) -> None:
    retired: list[str] = []
    for key in ("train_command", "freeze_command", "dp_command", "env"):
        if key in payload:
            retired.append(key)
    execution = payload.get("execution")
    if isinstance(execution, dict):
        for key in ("train_command", "freeze_command", "dp_command", "env"):
            if key in execution:
                retired.append(f"execution.{key}")
    if retired:
        raise ValueError(
            f"DeepMD student training config {config_path} contains retired command/env field(s): {sorted(retired)}; "
            "move executable selection and shell environment to environment_profile executables/source scripts"
        )


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in dict(overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _deepmd_training_defaults(identity_payload: dict[str, Any]) -> dict[str, Any]:
    training = dict(dict(identity_payload or {}).get("training", {}) or {})
    training.pop("split_mode", None)
    return training


def _load_legacy_recipe_type_map(train_config_path: str | Path) -> list[str]:
    try:
        template = load_dp_train_template(train_config_path)
    except Exception:
        return []
    type_map = dict(template.get("model", {}) or {}).get("type_map")
    if isinstance(type_map, list) and type_map:
        return [str(item) for item in type_map]
    return []


def _discover_subworkflow_identity(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    for candidate_dir in [path.parent, *path.parent.parents]:
        candidate = candidate_dir / "subworkflow.json"
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        student_model = payload.get("student_model")
        deepmd_payload = payload.get("deepmd")
        discovered: dict[str, Any] = {}
        if isinstance(student_model, dict) and student_model:
            discovered = dict(student_model)
        if isinstance(deepmd_payload, dict) and isinstance(deepmd_payload.get("training"), dict):
            discovered["training"] = dict(deepmd_payload.get("training", {}) or {})
        if discovered:
            return discovered
    return {}


def load_dp_train_template(
    train_config_path: str | Path,
    *,
    type_map: list[str] | None = None,
) -> dict[str, Any]:
    train_config_path = Path(train_config_path).resolve()

    with open(train_config_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    dp_config = _extract_dp_config_payload(payload)
    if dp_config is None:
        raise KeyError(f"DP train config missing 'dp_config': {train_config_path}")

    if type_map is not None:
        model = dp_config.setdefault("model", {})
        if not isinstance(model, dict):
            raise TypeError(f"DP train config model must be a mapping: {train_config_path}")
        model["type_map"] = [str(item) for item in type_map]
    return dp_config


def _extract_dp_config_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(payload.get("deepmd"), dict) and isinstance(payload["deepmd"].get("dp_config"), dict):
        return payload["deepmd"]["dp_config"]
    if isinstance(payload.get("dp_config"), dict):
        return payload["dp_config"]
    return None

