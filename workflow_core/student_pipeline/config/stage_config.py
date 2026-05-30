from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json


STRICT = "strict"

def canonical_stage_type(stage_type: str) -> str:
    normalized = str(stage_type or "").strip()
    return normalized


def load_pipeline_stage_config(
    stage_type: str,
    config_path: str | Path,
    *,
    mode: str = STRICT,
    sampling_backend: str | None = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    payload = dict(read_json(path) or {})
    return normalize_pipeline_stage_config(
        stage_type,
        payload,
        source_path=path,
        mode=mode,
        sampling_backend=sampling_backend,
    )


def normalize_pipeline_stage_config(
    stage_type: str,
    payload: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    mode: str = STRICT,
    sampling_backend: str | None = None,
) -> dict[str, Any]:
    mode = _normalize_mode(mode)
    stage = canonical_stage_type(stage_type)
    data = deepcopy(dict(payload or {}))
    if stage == "sampling":
        return _normalize_sampling(data, source_path=source_path, mode=mode, sampling_backend=sampling_backend)
    if stage == "relabel":
        return _normalize_relabel(data, source_path=source_path, mode=mode)
    if stage == "train":
        return _normalize_train(data, source_path=source_path, mode=mode)
    if stage == "student_relabel_input_filtering":
        return _normalize_student_relabel_input_filtering(data, mode=mode)
    if stage == "seed_pool_refresh":
        return _normalize_seed_pool_refresh(data, mode=mode)
    if stage == "teacher_dft_candidate_filtering":
        return _normalize_teacher_dft_candidate_filtering(data, mode=mode)
    raise ValueError(f"unsupported student pipeline stage type for config normalization: {stage_type!r}")


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or STRICT).strip().lower()
    if normalized != STRICT:
        raise ValueError(f"unsupported stage config mode: {mode!r}")
    return normalized


def _normalize_sampling(
    payload: dict[str, Any],
    *,
    source_path: str | Path | None,
    mode: str,
    sampling_backend: str | None,
) -> dict[str, Any]:
    del mode
    _reject_keys(payload, {"seed_selection", "md", "execution", "native_recipe_ref"}, "sampling stage config")
    md_payload = dict(payload.get("md", {}) or {})
    _normalize_sampling_md_fields(md_payload, sampling_backend=sampling_backend)
    normalized = {
        "seed_selection": dict(payload.get("seed_selection", {}) or {}),
        "md": md_payload,
        "execution": dict(payload.get("execution", {}) or {}),
    }
    if payload.get("native_recipe_ref"):
        normalized["native_recipe_ref"] = str(payload["native_recipe_ref"])
    return _resolve_stage_reference_fields(normalized, source_path, {"native_recipe_ref"})


def _normalize_sampling_md_fields(md_payload: dict[str, Any], *, sampling_backend: str | None) -> None:
    backend = str(sampling_backend or "").strip().lower()
    if backend == "dp":
        backend = "lammps"
    if "eval_every" in md_payload:
        raise ValueError(
            "sampling stage md.eval_every is retired; use model_devi_interval for LAMMPS, "
            "active_interval for GPUMD, or dump_interval for workflow-managed parser dumps"
        )
    allowed = {
        "temperature_schedule",
        "steps",
        "model_devi_interval",
        "active_interval",
        "dump_interval",
        "atomic_model_devi",
        "active_outputs",
        "seed",
    }
    native_fields = {
        "ensemble",
        "timestep_fs",
        "thermostat_damp_fs",
        "thermostat_damp",
        "log_every",
        "velocity_seed",
        "zero_rotation",
        "save_forces",
        "save_velocities",
    }
    present_native = sorted(set(md_payload) & native_fields)
    if present_native:
        raise ValueError(
            "sampling stage md contains backend-native physical MD fields "
            f"{present_native}; move them to the native run.in/input.lammps template"
        )
    extra = sorted(set(md_payload) - allowed)
    if extra:
        raise ValueError(f"sampling stage md contains unsupported workflow fields {extra}")
    if backend == "lammps":
        _reject_backend_interval_fields(md_payload, {"active_interval", "active_outputs"}, "gpumd")
        _normalize_lammps_intervals(md_payload)
    elif backend == "gpumd":
        _reject_backend_interval_fields(md_payload, {"model_devi_interval", "atomic_model_devi"}, "lammps")
        _normalize_gpumd_intervals(md_payload)
    else:
        _normalize_active_outputs(md_payload)


def _reject_backend_interval_fields(md_payload: dict[str, Any], keys: set[str], backend: str) -> None:
    present = sorted(set(md_payload) & keys)
    if present:
        raise ValueError(f"sampling stage md.{present[0]} only applies to sampling_backend {backend}")


def _normalize_lammps_intervals(md_payload: dict[str, Any]) -> None:
    model_interval = md_payload.get("model_devi_interval")
    dump_interval = md_payload.get("dump_interval")
    if model_interval is None and dump_interval is None:
        return
    if model_interval is None:
        md_payload["model_devi_interval"] = _positive_int(dump_interval, "model_devi_interval")
        md_payload["dump_interval"] = _positive_int(dump_interval, "dump_interval")
        return
    if dump_interval is None:
        md_payload["model_devi_interval"] = _positive_int(model_interval, "model_devi_interval")
        md_payload["dump_interval"] = _positive_int(model_interval, "dump_interval")
        return
    md_payload["model_devi_interval"] = _positive_int(model_interval, "model_devi_interval")
    md_payload["dump_interval"] = _positive_int(dump_interval, "dump_interval")
    if md_payload["model_devi_interval"] != md_payload["dump_interval"]:
        raise ValueError("sampling stage md.model_devi_interval and md.dump_interval must match for LAMMPS")


def _normalize_gpumd_intervals(md_payload: dict[str, Any]) -> None:
    active_interval = md_payload.get("active_interval")
    dump_interval = md_payload.get("dump_interval")
    if active_interval is None and dump_interval is None:
        _normalize_active_outputs(md_payload)
        return
    if active_interval is None:
        md_payload["active_interval"] = _positive_int(dump_interval, "active_interval")
        md_payload["dump_interval"] = _positive_int(dump_interval, "dump_interval")
    elif dump_interval is None:
        md_payload["active_interval"] = _positive_int(active_interval, "active_interval")
        md_payload["dump_interval"] = _positive_int(active_interval, "dump_interval")
    else:
        md_payload["active_interval"] = _positive_int(active_interval, "active_interval")
        md_payload["dump_interval"] = _positive_int(dump_interval, "dump_interval")
        if md_payload["active_interval"] != md_payload["dump_interval"]:
            raise ValueError("sampling stage md.active_interval and md.dump_interval must match for GPUMD")
    _normalize_active_outputs(md_payload)


def _normalize_active_outputs(md_payload: dict[str, Any]) -> None:
    if "active_outputs" not in md_payload:
        return
    outputs = dict(md_payload.get("active_outputs") or {})
    allowed = {"velocity", "force", "uncertainty"}
    extra = sorted(set(outputs) - allowed)
    if extra:
        raise ValueError(f"sampling stage md.active_outputs contains unsupported fields {extra}")
    if outputs.get("uncertainty") is False:
        raise ValueError("sampling stage md.active_outputs.uncertainty must remain enabled for workflow routing")
    normalized = {
        "velocity": bool(outputs.get("velocity", True)),
        "force": bool(outputs.get("force", True)),
        "uncertainty": True,
    }
    md_payload["active_outputs"] = normalized


def _positive_int(value: Any, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"sampling stage md.{field_name} must be a positive integer")
    return normalized


def _normalize_relabel(
    payload: dict[str, Any],
    *,
    source_path: str | Path | None,
    mode: str,
) -> dict[str, Any]:
    del mode
    allowed = {
        "teacher_backend",
        "teacher_backend_options",
        "split_seed",
        "min_val",
        "teacher_trust_lo",
        "teacher_trust_hi",
        "max_force",
        "min_distances",
        "include_user_student_train",
        "failed_backflow",
        "valid_fraction",
        "element_order",
        "train_committee_config",
        "training_stage",
    }
    _reject_keys(payload, allowed, "relabel stage config")
    if "candidate_sigma_threshold" in payload:
        raise ValueError("relabel stage config uses legacy candidate_sigma_threshold; use teacher_trust_lo")
    return _resolve_stage_reference_fields(payload, source_path, {"train_committee_config"})


def _normalize_train(
    payload: dict[str, Any],
    *,
    source_path: str | Path | None,
    mode: str,
) -> dict[str, Any]:
    del mode
    selector_keys = {"scratch_entry", "foundation_entry", "continuation"}
    present_selector_keys = sorted(set(payload) & selector_keys)
    if present_selector_keys:
        raise ValueError(
            "train stage config rejects selector-shaped train recipe fields "
            f"{present_selector_keys}; point train_recipe_ref directly at a backend-native recipe"
        )
    if "train_recipe_ref" not in payload and _looks_like_direct_train_recipe(payload):
        if source_path is None:
            raise ValueError("direct train recipe stage config requires a source path")
        return {"train_recipe_ref": str(Path(source_path).resolve())}
    forbidden = {"type_map", "committee_size", "student_trust_lo", "student_trust_hi"}
    _reject_forbidden_keys(payload, forbidden, "train stage config")
    allowed = {"train_recipe_ref"}
    _reject_keys(payload, allowed, "train stage config")
    return _resolve_stage_reference_fields(payload, source_path, {"train_recipe_ref"})


def _looks_like_direct_train_recipe(payload: dict[str, Any]) -> bool:
    recipe_keys = {
        "nep",
        "deepmd",
        "dp_config",
        "model",
        "learning_rate",
        "loss",
        "training",
        "arch",
        "loss_stage_1",
        "training_stage_1",
    }
    return bool(set(payload) & recipe_keys)


def _normalize_student_relabel_input_filtering(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    del mode
    forbidden = {
        "teacher_trust_lo",
        "teacher_trust_hi",
        "candidate_sigma_threshold",
        "teacher_backend",
        "mace_backend",
        "teacher_num",
        "valid_fraction",
        "training_stage",
        "seed_pool_refresh",
    }
    _reject_forbidden_keys(payload, forbidden, "student_relabel_input_filtering stage config")
    allowed = {
        "source_window",
        "selection_mode",
        "random_seed",
        "direct",
        "candidate",
        "failed",
        "max_structures_per_system",
        "candidate_max_structures_per_system",
        "failed_max_structures_per_system",
        "dedup",
    }
    _reject_keys(payload, allowed, "student_relabel_input_filtering stage config")
    return _normalize_direct_filtering_stage_config(
        payload,
        stage_label="student_relabel_input_filtering stage config",
        default_selection_mode="random_per_system",
    )


def _normalize_seed_pool_refresh(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    del mode
    forbidden = {
        "seed_selection",
        "n_seed_per_system",
        "md",
        "execution",
        "temperature_schedule",
        "steps",
        "max_parallel_tasks",
    }
    _reject_forbidden_keys(payload, forbidden, "seed_pool_refresh stage config")
    allowed = {
        "source_merge",
        "rescore",
        "refresh_selection",
        "keep_if_ge_low",
        "keep_if_lt_high",
        "max_pool_size_per_system",
        "selection_mode",
        "dedup_enabled",
        "reset_times_used",
        "random_seed",
    }
    _reject_keys(payload, allowed, "seed_pool_refresh stage config")
    return payload


def _normalize_teacher_dft_candidate_filtering(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    del mode
    forbidden = {"dft_submit", "submit", "import", "external_dft_import", "label_config"}
    _reject_forbidden_keys(payload, forbidden, "teacher_dft_candidate_filtering stage config")
    allowed = {
        "source",
        "selection",
        "handoff",
        "selection_mode",
        "budget_per_system",
        "max_structures_per_system",
        "random_seed",
        "dedup",
        "bucket",
        "representative",
        "direct",
    }
    _reject_keys(payload, allowed, "teacher_dft_candidate_filtering stage config")
    return _normalize_direct_filtering_stage_config(
        payload,
        stage_label="teacher_dft_candidate_filtering stage config",
        default_selection_mode="top_uncertainty_per_system",
    )


def _normalize_direct_filtering_stage_config(
    payload: dict[str, Any],
    *,
    stage_label: str,
    default_selection_mode: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    selection_mode = str(normalized.get("selection_mode", default_selection_mode) or default_selection_mode).strip().lower()
    allowed_modes = {"random_per_system", "top_uncertainty_per_system", "direct"}
    if "teacher_dft_candidate_filtering" in stage_label:
        allowed_modes = {*allowed_modes, "bucket_random_per_system"}
    if selection_mode not in allowed_modes:
        raise ValueError(f"{stage_label} selection_mode must be one of {sorted(allowed_modes)}; got {selection_mode!r}")
    normalized["selection_mode"] = selection_mode

    direct_block = normalized.get("direct")
    if selection_mode != "direct":
        if direct_block not in (None, {}, []):
            raise ValueError(f"{stage_label} only allows direct config when selection_mode='direct'")
        normalized.pop("direct", None)
        return normalized

    if not isinstance(direct_block, dict) or not direct_block:
        raise ValueError(f"{stage_label} selection_mode='direct' requires a direct config block")
    allowed_direct_keys = {"soap", "direct"}
    extra_direct = sorted(set(direct_block) - allowed_direct_keys)
    if extra_direct:
        raise ValueError(f"{stage_label} direct config only supports {sorted(allowed_direct_keys)}; got {extra_direct}")

    from workflow_core.structure_descriptors.soap import normalize_soap_encoder_config
    from workflow_policies.direct.schema import normalize_direct_config

    soap_cfg = normalize_soap_encoder_config(dict(direct_block.get("soap", {}) or {}))
    direct_cfg = normalize_direct_config(dict(direct_block.get("direct", {}) or {}))
    normalized["direct"] = {
        "soap": {
            "species": list(soap_cfg.species),
            "r_cut": float(soap_cfg.r_cut),
            "n_max": int(soap_cfg.n_max),
            "l_max": int(soap_cfg.l_max),
            "sigma": float(soap_cfg.sigma),
            "periodic": bool(soap_cfg.periodic),
            "sparse": bool(soap_cfg.sparse),
            "dtype": str(soap_cfg.dtype),
        },
        "direct": asdict(direct_cfg),
    }
    return normalized


def _resolve_stage_reference_fields(
    payload: dict[str, Any],
    source_path: str | Path | None,
    fields: set[str],
) -> dict[str, Any]:
    if source_path is None:
        return payload
    source = Path(source_path).resolve()
    resolved = dict(payload)
    for field in fields:
        value = resolved.get(field)
        if value in (None, ""):
            continue
        path = Path(str(value))
        resolved[field] = str(path.resolve() if path.is_absolute() else (source.parent / path).resolve())
    return resolved


def _reject_keys(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ValueError(f"{label} only supports {sorted(allowed)}; got {extra}")


def _reject_forbidden_keys(payload: dict[str, Any], forbidden: set[str], label: str) -> None:
    present = sorted(set(payload) & forbidden)
    if present:
        raise ValueError(f"{label} contains fields owned by another config layer: {present}")
