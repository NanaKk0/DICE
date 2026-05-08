from __future__ import annotations

from typing import Any


def _has_path(payload: dict[str, Any], *keys: str) -> bool:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _get_path(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_bootstrap_data(raw_cfg: dict[str, Any], deprecated_fields: list[str]) -> dict[str, Any]:
    if _has_path(raw_cfg, "bootstrap_data", "include_rgps_in_initial_dataset"):
        include_rgps = bool(_get_path(raw_cfg, "bootstrap_data", "include_rgps_in_initial_dataset", default=False))
    else:
        include_rgps = False
        if _has_path(raw_cfg, "round0", "include_base_rgps_in_dp_bootstrap"):
            deprecated_fields.append("round0.include_base_rgps_in_dp_bootstrap")
            include_rgps = bool(_get_path(raw_cfg, "round0", "include_base_rgps_in_dp_bootstrap", default=False))
        elif _has_path(raw_cfg, "dp_domain_strategy", "bootstrap_use_rgps"):
            deprecated_fields.append("dp_domain_strategy.bootstrap_use_rgps")
            include_rgps = bool(_get_path(raw_cfg, "dp_domain_strategy", "bootstrap_use_rgps", default=False))

    if _has_path(raw_cfg, "round0", "use_current_mace_training_dataset"):
        deprecated_fields.append("round0.use_current_mace_training_dataset")
    if _has_path(raw_cfg, "round0", "include_historical_dp_increment_train"):
        deprecated_fields.append("round0.include_historical_dp_increment_train")

    return {
        "include_rgps_in_initial_dataset": bool(include_rgps),
        "physical_dataset_always_enabled": True,
    }


def _resolve_incremental_data(raw_cfg: dict[str, Any], deprecated_fields: list[str]) -> dict[str, Any]:
    if "dp_replay" in raw_cfg:
        deprecated_fields.append("dp_replay")
    if "dp_domain_strategy" in raw_cfg:
        deprecated_fields.append("dp_domain_strategy")
    if "accumulate_dp_train_across_outer_rounds" in raw_cfg:
        deprecated_fields.append("accumulate_dp_train_across_outer_rounds")

    resolved = dict(_get_path(raw_cfg, "incremental_data", default={}) or {})
    return {
        "use_incremental_data_only": bool(resolved.get("use_incremental_data_only", True)),
        "train_on_current_round_increment_only": True,
    }


def _resolve_training_policy(raw_cfg: dict[str, Any], deprecated_fields: list[str]) -> dict[str, Any]:
    if (
        _has_path(raw_cfg, "training_policy", "round0_stage")
        or _has_path(raw_cfg, "training_policy", "later_round_stage")
    ):
        training_policy = dict(_get_path(raw_cfg, "training_policy", default={}) or {})
        round0_stage = str(training_policy.get("round0_stage", "bootstrap")).strip().lower()
        later_round_stage = str(training_policy.get("later_round_stage", "finetune")).strip().lower()
    else:
        legacy_mode = str(raw_cfg.get("within_outer_round_training_mode", "incremental_finetune_from_previous_model")).strip().lower()
        if "within_outer_round_training_mode" in raw_cfg:
            deprecated_fields.append("within_outer_round_training_mode")
        round0_stage = "bootstrap"
        later_round_stage = "finetune" if legacy_mode == "incremental_finetune_from_previous_model" else "bootstrap"

    return {
        "round0_stage": round0_stage,
        "later_round_stage": later_round_stage,
    }


def resolve_dp_active_learning_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    deprecated_fields: list[str] = []
    resolved = dict(raw_cfg)
    resolved["bootstrap_data"] = _resolve_bootstrap_data(raw_cfg, deprecated_fields)
    resolved["incremental_data"] = _resolve_incremental_data(raw_cfg, deprecated_fields)
    resolved["training_policy"] = _resolve_training_policy(raw_cfg, deprecated_fields)
    resolved["deprecated_config_fields"] = sorted(set(deprecated_fields))
    return resolved
