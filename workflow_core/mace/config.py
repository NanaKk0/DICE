from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json


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


def _resolve_mace_finetune(raw_cfg: dict[str, Any], deprecated_fields: list[str]) -> dict[str, Any]:
    if _has_path(raw_cfg, "mace_finetune"):
        payload = dict(_get_path(raw_cfg, "mace_finetune", default={}) or {})
    else:
        payload = {}
        if "finetune_config" in raw_cfg:
            deprecated_fields.append("finetune_config")
        if "reuse_existing_committee" in raw_cfg:
            deprecated_fields.append("reuse_existing_committee")
        if "reuse_previous_labeled_dft" in raw_cfg:
            deprecated_fields.append("reuse_previous_labeled_dft")
        if "previous_active_learning_round_offset" in raw_cfg:
            deprecated_fields.append("previous_active_learning_round_offset")
        if "reuse_previous_active_learning_pool" in raw_cfg:
            deprecated_fields.append("reuse_previous_active_learning_pool")
        payload["config"] = raw_cfg.get("finetune_config", "configs/mace/committee_finetune.json")
        payload["reuse_existing_committee"] = bool(raw_cfg.get("reuse_existing_committee", True))
        payload["training_data"] = {
            "base_datasets": {
                "include_physical": True,
                "include_rgps": False,
            },
        }

    payload.setdefault("config", "configs/mace/committee_finetune.json")
    payload["reuse_existing_committee"] = bool(payload.get("reuse_existing_committee", True))

    training_data = dict(payload.get("training_data", {}) or {})
    base_datasets = dict(training_data.get("base_datasets", {}) or {})
    training_data["base_datasets"] = {
        "include_physical": bool(base_datasets.get("include_physical", True)),
        "include_rgps": bool(base_datasets.get("include_rgps", False)),
    }
    payload["training_data"] = training_data
    return payload


def _resolve_dp_subworkflow(raw_cfg: dict[str, Any], deprecated_fields: list[str]) -> dict[str, Any]:
    if _has_path(raw_cfg, "dp_subworkflow"):
        payload = dict(_get_path(raw_cfg, "dp_subworkflow", default={}) or {})
    else:
        if "dp_subworkflow_config" in raw_cfg:
            deprecated_fields.append("dp_subworkflow_config")
        payload = {"config": raw_cfg.get("dp_subworkflow_config", "configs/dp/active_learning_subworkflow.json")}
    payload.setdefault("config", "configs/dp/active_learning_subworkflow.json")
    return payload


def _resolve_include_rgps_from_train_config(
    mace_finetune_cfg: dict[str, Any],
    deprecated_fields: list[str],
) -> dict[str, Any]:
    config_path = Path(str(mace_finetune_cfg.get("config", "configs/mace/committee_finetune.json"))).resolve()
    train_cfg = read_json(config_path)
    base_datasets = dict(mace_finetune_cfg.get("training_data", {}).get("base_datasets", {}) or {})
    if "include_base_rgps_in_finetune" in train_cfg:
        deprecated_fields.append("mace_committee_finetune.include_base_rgps_in_finetune")
        if not _has_path(mace_finetune_cfg, "training_data", "base_datasets", "include_rgps"):
            base_datasets["include_rgps"] = bool(train_cfg.get("include_base_rgps_in_finetune", True))
    mace_finetune_cfg["training_data"]["base_datasets"] = {
        "include_physical": bool(base_datasets.get("include_physical", True)),
        "include_rgps": bool(base_datasets.get("include_rgps", False)),
    }
    return train_cfg


def resolve_committee_cycle_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    deprecated_fields: list[str] = []
    resolved = deepcopy(raw_cfg)
    resolved["mace_finetune"] = _resolve_mace_finetune(raw_cfg, deprecated_fields)
    resolved["dp_subworkflow"] = _resolve_dp_subworkflow(raw_cfg, deprecated_fields)
    train_cfg = _resolve_include_rgps_from_train_config(resolved["mace_finetune"], deprecated_fields)

    if "reuse_previous_active_learning_pool" in raw_cfg:
        resolved["mace_finetune"]["training_data"]["al_pool_reuse_removed"] = True

    resolved["deprecated_config_fields"] = sorted(set(deprecated_fields))
    resolved["mace_finetune_train_config_snapshot"] = train_cfg
    return resolved

