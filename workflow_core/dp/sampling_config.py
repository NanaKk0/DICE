from __future__ import annotations

from copy import deepcopy
from typing import Any


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged.get(key, {}) or {}), dict(value or {}))
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_seed_pool_policy(raw_cfg: dict[str, Any], deprecated_fields: list[str]) -> dict[str, Any]:
    if "seed_pool_policy" in raw_cfg:
        payload = dict(raw_cfg.get("seed_pool_policy", {}) or {})
    else:
        payload = dict(raw_cfg.get("seed_pool", {}) or {})
        if "seed_pool" in raw_cfg:
            deprecated_fields.append("seed_pool")

    if "source" in payload:
        deprecated_fields.append("seed_pool.source")
    if "selection_method" in payload:
        deprecated_fields.append("seed_pool.selection_method")
    if "physical_filter" in payload:
        deprecated_fields.append("seed_pool.physical_filter")
    if "gate" in payload:
        deprecated_fields.append("seed_pool_policy.gate")
    if "selector_config" in payload:
        deprecated_fields.append("seed_pool_policy.selector_config")
    if "max_size" in payload:
        deprecated_fields.append("seed_pool_policy.max_size")
    if "update_every_round" in payload:
        deprecated_fields.append("seed_pool_policy.update_every_round")
    if "build_seed_pool_config" in raw_cfg:
        deprecated_fields.append("build_seed_pool_config")

    return {
        "enabled": bool(payload.get("enabled", True)),
        "initialize_at_subworkflow_start": bool(payload.get("initialize_at_subworkflow_start", True)),
        "max_pool_size_per_system": int(payload.get("max_pool_size_per_system", payload.get("max_size", 2000))),
        "refresh_after_training": bool(payload.get("refresh_after_training", payload.get("update_every_round", True))),
    }


def _resolve_sampling_seed_policy(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw_cfg.get("sampling_seed_policy", {}) or {})
    return {
        "backend": str(payload.get("backend", "lammps")),
        "driver_policy": str(payload.get("driver_policy", "round_robin")),
        "md": dict(
            payload.get(
                "md",
                {
                    "timestep_fs": 0.5,
                    "eval_every": 20,
                    "log_every": 10,
                    "seed": 20260319,
                    "friction": 0.05,
                    "force_temp": True,
                    "zero_rotation": False,
                },
            )
            or {}
        ),
        "lammps": dict(
            payload.get(
                "lammps",
                {
                    "command": "auto",
                    "plugin_path": "auto",
                    "type_map": ["H", "O", "Pt"],
                    "ensemble": "nvt",
                    "thermo_freq": 10,
                    "dump_freq": 10,
                    "model_devi_freq": 10,
                    "f_trust_lo": 0.05,
                    "f_trust_hi": 0.15,
                    "rotate_driver_model": True,
                    "atomic_model_devi": True,
                    "thermostat_damp_fs": 100.0,
                    "device_policy": "cpu",
                    "execution_backend": "inline",
                    "group_fail_fast": True,
                    "task_layout_version": "v2",
                },
            )
            or {}
        ),
    }


def _resolve_sampling_schedule(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw_cfg.get("sampling_schedule", {}) or {})
    return {
        "mode": str(payload.get("mode", "long_first_then_short")),
        "long_round_selector": str(payload.get("long_round_selector", "first_sampling_round_only")),
        "long_profile_name": str(payload.get("long_profile_name", "long_exploration")),
        "short_profile_name": str(payload.get("short_profile_name", "short_refinement")),
    }


def _resolve_sampling_profiles(
    raw_cfg: dict[str, Any],
    sampling_seed_policy: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(raw_cfg.get("sampling_profiles", {}) or {})
    base_profile = {
        "seed_selection": {"n_seed_per_system": 8, "selection_mode": "top_uncertainty_per_system"},
        "sampling_seed_policy": {
            "backend": str(sampling_seed_policy.get("backend", "lammps")),
            "driver_policy": str(sampling_seed_policy.get("driver_policy", "round_robin")),
            "md": dict(sampling_seed_policy.get("md", {}) or {}),
            "lammps": dict(sampling_seed_policy.get("lammps", {}) or {}),
        },
        "candidate_train_selection": {
            "enabled": True,
            "selection_mode": "random_per_system",
            "max_train_candidates_per_system": 200,
            "random_seed": 20260502,
        },
        "seed_pool": {
            "refresh_mode": "merge_old_pool_with_current_candidate_failed",
            "random_seed": 20260502,
        },
    }

    def _profile_overlay(name: str, refresh_mode: str) -> dict[str, Any]:
        raw_profile = dict(payload.get(name, {}) or {})
        normalized_overlay = {
            "seed_selection": dict(raw_profile.get("seed_selection", {}) or {}),
            "sampling_seed_policy": {
                **(
                    {"backend": raw_profile["backend"]}
                    if "backend" in raw_profile
                    else {}
                ),
                **(
                    {"driver_policy": raw_profile["driver_policy"]}
                    if "driver_policy" in raw_profile
                    else {}
                ),
                **({"md": dict(raw_profile.get("sampling", {}) or {})} if "sampling" in raw_profile else {}),
                **({"lammps": dict(raw_profile.get("lammps", {}) or {})} if "lammps" in raw_profile else {}),
            },
            "candidate_train_selection": dict(raw_profile.get("candidate_train_selection", {}) or {}),
            "seed_pool": dict(raw_profile.get("seed_pool", {}) or {}),
        }
        profile = _deep_update(base_profile, normalized_overlay)
        profile["seed_pool"]["refresh_mode"] = str(profile["seed_pool"].get("refresh_mode", refresh_mode))
        if "selection_mode" not in (raw_profile.get("seed_selection", {}) or {}):
            profile["seed_selection"]["selection_mode"] = (
                "top_uncertainty_per_system"
                if str(profile["seed_pool"]["refresh_mode"]).strip().lower() == "merge_old_pool_with_current_candidate_failed"
                else "usage_then_uncertainty_per_system"
            )
        return profile

    return {
        "long_exploration": _profile_overlay("long_exploration", "merge_old_pool_with_current_candidate_failed"),
        "short_refinement": _profile_overlay("short_refinement", "rescore_existing_pool_only"),
    }


def resolve_dp_sampling_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(raw_cfg)
    deprecated_fields = list(resolved.get("deprecated_config_fields", []) or [])

    if "frame_selection_policy" in raw_cfg:
        deprecated_fields.append("frame_selection_policy")
    if "uncertainty_policy" in raw_cfg:
        deprecated_fields.append("uncertainty_policy")
    if "candidate_train_selection" in raw_cfg:
        deprecated_fields.append("candidate_train_selection")
    if "seed_pool_refresh" in raw_cfg:
        deprecated_fields.append("seed_pool_refresh")

    resolved["seed_pool_policy"] = _resolve_seed_pool_policy(raw_cfg, deprecated_fields)
    resolved["sampling_seed_policy"] = _resolve_sampling_seed_policy(raw_cfg)
    resolved["sampling_schedule"] = _resolve_sampling_schedule(raw_cfg)
    resolved["sampling_profiles"] = _resolve_sampling_profiles(
        raw_cfg,
        resolved["sampling_seed_policy"],
    )
    resolved["deprecated_config_fields"] = sorted(set(deprecated_fields))
    return resolved
