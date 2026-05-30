from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json
from workflow_core.training_data_assets import reject_nested_source_toggle, validate_path_only_xyz_list


CANONICAL_TEACHER_COMMITTEE_FINETUNE_CONFIG = (
    "configs/pipelines/committee_cycle/teacher_committee_finetune/mace/committee_finetune.json"
)
DEFAULT_STUDENT_SUBWORKFLOW_CONFIG = (
    "configs/pipelines/committee_cycle/student_subworkflow/deepmd/active_learning/subworkflow.json"
)


def load_committee_cycle_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = read_json(config_path)
    return resolve_committee_cycle_config(payload, config_path=config_path)


def resolve_committee_cycle_config(
    raw_cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    deprecated_fields: list[str] = []
    resolved_workspace_root = Path(workspace_root).resolve() if workspace_root is not None else _infer_workspace_root(config_path)
    config_anchor = Path(config_path).resolve().parent if config_path is not None else None
    resolved = deepcopy(raw_cfg)
    resolved["teacher_committee_finetune"] = _resolve_teacher_committee_finetune(raw_cfg, config_anchor=config_anchor)
    resolved["student_subworkflow"] = _resolve_student_subworkflow(raw_cfg, config_anchor=config_anchor)

    train_cfg = _load_and_validate_teacher_finetune_config(
        resolved["teacher_committee_finetune"],
        deprecated_fields,
        workspace_root=resolved_workspace_root,
    )

    resolved["deprecated_config_fields"] = sorted(set(deprecated_fields))
    resolved["teacher_committee_finetune_train_config_snapshot"] = train_cfg
    return resolved


def _get_teacher_backend(name: str | None):
    from workflow_backends.teacher.registry import get_teacher_backend

    return get_teacher_backend(name)


def _resolve_teacher_committee_finetune(raw_cfg: dict[str, Any], *, config_anchor: Path | None = None) -> dict[str, Any]:
    retired_keys = sorted(
        set(raw_cfg)
        & {
            "finetune_config",
            "reuse_previous_labeled_dft",
            "previous_active_learning_round_offset",
            "reuse_previous_active_learning_pool",
        }
    )
    if retired_keys:
        raise ValueError(
            "committee_cycle config rejects retired teacher finetune aliases; "
            f"use teacher_committee_finetune.config and committee_finetune training_data.base_datasets instead: {retired_keys}"
        )
    has_canonical = "teacher_committee_finetune" in raw_cfg
    has_legacy = "mace_finetune" in raw_cfg
    if has_legacy:
        raise ValueError("committee_cycle config rejects legacy mace_finetune; use teacher_committee_finetune")
    if has_canonical:
        payload = dict(raw_cfg.get("teacher_committee_finetune") or {})
    else:
        payload = {
            "teacher_backend": "mace",
            "config": CANONICAL_TEACHER_COMMITTEE_FINETUNE_CONFIG,
            "reuse_existing_committee": True,
        }
    reject_nested_source_toggle(
        dict(payload.get("training_data", {}) or {}),
        field_name="teacher_committee_finetune.training_data",
        forbidden_keys={"include_physical", "include_rgps"},
    )
    payload.pop("training_data", None)
    payload.setdefault("teacher_backend", "mace")
    payload.setdefault("config", CANONICAL_TEACHER_COMMITTEE_FINETUNE_CONFIG)
    payload["teacher_backend"] = str(payload["teacher_backend"]).strip().lower()
    payload["config"] = str(_resolve_config_reference(payload["config"], config_anchor=config_anchor))
    _get_teacher_backend(payload["teacher_backend"])
    payload["reuse_existing_committee"] = bool(payload.get("reuse_existing_committee", True))
    return payload


def _resolve_student_subworkflow(raw_cfg: dict[str, Any], *, config_anchor: Path | None = None) -> dict[str, Any]:
    retired_keys = sorted(set(raw_cfg) & {"dp_subworkflow", "student_subworkflow_config", "dp_subworkflow_config"})
    if retired_keys:
        raise ValueError(
            "committee_cycle config rejects retired student subworkflow aliases; "
            f"use student_subworkflow.config instead: {retired_keys}"
        )
    if "student_subworkflow" in raw_cfg:
        payload = dict(raw_cfg.get("student_subworkflow") or {})
    else:
        payload = {"config": DEFAULT_STUDENT_SUBWORKFLOW_CONFIG}
    payload.setdefault("config", DEFAULT_STUDENT_SUBWORKFLOW_CONFIG)
    payload["config"] = str(_resolve_config_reference(payload["config"], config_anchor=config_anchor))
    return payload


def _load_and_validate_teacher_finetune_config(
    teacher_finetune_cfg: dict[str, Any],
    deprecated_fields: list[str],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    config_path = Path(str(teacher_finetune_cfg.get("config", CANONICAL_TEACHER_COMMITTEE_FINETUNE_CONFIG))).resolve()
    backend = _get_teacher_backend(str(teacher_finetune_cfg.get("teacher_backend", "mace")))
    train_cfg = backend.load_committee_finetune_config(config_path)
    if "include_base_rgps_in_finetune" in train_cfg:
        deprecated_fields.append("mace_committee_finetune.include_base_rgps_in_finetune")
    training_data = dict(train_cfg.get("training_data", {}) or {})
    validate_path_only_xyz_list(
        training_data.get("base_datasets"),
        field_name="training_data.base_datasets",
        bare_filename_root=Path(workspace_root).resolve() / "user_assets" / "teacher_finetune",
    )
    _validate_teacher_init_mode_assets(train_cfg)
    _resolve_teacher_foundation_model(train_cfg, workspace_root=workspace_root)
    return train_cfg


def _validate_teacher_init_mode_assets(train_cfg: dict[str, Any]) -> None:
    init_mode = str(train_cfg.get("init_mode", "scratch")).strip().lower()
    if init_mode == "foundation_finetune" and not train_cfg.get("foundation_model"):
        raise ValueError("foundation_finetune teacher initialization requires workflow.foundation_model")


def _resolve_teacher_foundation_model(train_cfg: dict[str, Any], *, workspace_root: str | Path) -> None:
    raw_value = train_cfg.get("foundation_model")
    if raw_value is None:
        return
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("foundation_model must be a non-empty path string")
    raw_path = Path(raw_value.strip())
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        if raw_path.name != str(raw_path):
            raise ValueError("foundation_model must be an absolute path or a bare filename")
        resolved = (Path(workspace_root).resolve() / "user_assets" / "foundation_models" / raw_path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"foundation_model not found: {resolved}")
    train_cfg["foundation_model"] = str(resolved)


def _resolve_config_reference(value: str | Path, *, config_anchor: Path | None) -> Path:
    raw_path = Path(str(value))
    if raw_path.is_absolute():
        return raw_path.resolve()
    cwd_candidate = raw_path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if config_anchor is not None:
        return (Path(config_anchor).resolve() / raw_path).resolve()
    return raw_path.resolve()


def _infer_workspace_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd().resolve()
    resolved = Path(config_path).resolve()
    if "configs" in resolved.parts:
        parts = list(resolved.parts)
        configs_index = parts.index("configs")
        if configs_index > 0:
            return Path(*parts[:configs_index]).resolve()
    return resolved.parent.resolve()
