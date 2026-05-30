from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

TRAIN_RECIPE_BRANCHES = ("scratch_entry", "foundation_entry", "continuation")
RETIRED_FIXED_LOOP_BACKEND_FIELDS = {
    "bootstrap_train_config",
    "finetune_train_config",
    "foundation_bootstrap",
    "max_dp_rounds",
    "sampling_profiles",
    "sampling_schedule",
    "sampling_seed_policy",
    "student_train_committee_config",
    "train_dp_committee_config",
}
RETIRED_DFT_SCREENING_FIELDS = {
    "stop_dp_round_when_current_screened_count_gte",
    "stop_dp_round_when_cumulative_screened_count_gte",
}


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged.get(key, {}) or {}), dict(value or {}))
        else:
            merged[key] = deepcopy(value)
    return merged


def _default_nep_config() -> dict[str, Any]:
    return {
        "command": "auto",
        "split_seed": 42,
        "type_map": [],
        "committee_size": 3,
        "version": 4,
        "cutoff": {"radial": 7.0, "angular": 5.0},
        "n_max": {"radial": 8, "angular": 4},
        "basis_size": {"radial": 8, "angular": 8},
        "l_max": "4 2 0",
        "neuron": 80,
        "zbl": None,
        "lambda_1": None,
        "lambda_e": 1.0,
        "lambda_f": 1.0,
        "lambda_v": 0.1,
        "batch": 1000,
        "population": 50,
        "generation": 100000,
        "train_test_split": 0.95,
        "save_potential_every": None,
        "save_potential_format": 0,
        "save_potential_save_restart": 1,
    }


def _default_gpumd_config() -> dict[str, Any]:
    return {
        "command": "auto",
        "sampling_backend": "gpumd",
        "ensemble": "nvt",
        "temperatures": [323, 473, 673],
        "steps_long": 20000,
        "steps_short": 500,
        "timestep_fs": 0.5,
        "active_interval": 20,
        "active_threshold": 0.10,
        "dump_interval": 20,
        "max_parallel_tasks": 1,
        "thermostat_damp": 100.0,
        "save_forces": True,
        "save_velocities": True,
    }


def _default_uncertainty_config() -> dict[str, Any]:
    return {
        "metric": "force_std_max",
        "trust_lo": 0.10,
        "trust_hi": 0.40,
    }


def _normalize_student_backend_name(value: Any) -> str:
    text = str(value or "deepmd").strip().lower()
    if text == "dp":
        return "deepmd"
    return text


def _infer_student_backend_name(raw_cfg: dict[str, Any]) -> str:
    explicit = raw_cfg.get("student_backend")
    if explicit not in (None, ""):
        return _normalize_student_backend_name(explicit)
    if isinstance(raw_cfg.get("nep"), dict):
        return "nep"
    if any(raw_cfg.get(key) for key in TRAIN_RECIPE_BRANCHES):
        return "nep"
    return "deepmd"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _normalize_l_max(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return " ".join(str(int(item)) for item in value)
    return value


def _normalize_zbl(value: Any) -> Any:
    if isinstance(value, dict):
        if not bool(value.get("enabled", True)):
            return None
        if value.get("outer_cutoff") is not None:
            return float(value["outer_cutoff"])
        if value.get("cutoff") is not None:
            return float(value["cutoff"])
        return None
    return value


def _normalize_nep_config(nep_cfg: dict[str, Any], *, run_stage: str = "stage_1") -> dict[str, Any]:
    cfg = deepcopy(nep_cfg)
    if "split_seed" in cfg and cfg.get("split_seed") is not None:
        cfg["split_seed"] = int(cfg["split_seed"])

    arch = _as_dict(cfg.get("arch"))
    for key in ("version", "zbl", "cutoff", "n_max", "basis_size", "l_max", "neuron"):
        if key in arch:
            cfg[key] = deepcopy(arch[key])
    if "zbl" in cfg:
        cfg["zbl"] = _normalize_zbl(cfg["zbl"])
    if "l_max" in cfg:
        cfg["l_max"] = _normalize_l_max(cfg["l_max"])

    stage_suffix = "2" if str(run_stage).strip().lower() in {"2", "stage_2", "second"} else "1"
    loss = _as_dict(cfg.get("loss"))
    if not loss and isinstance(cfg.get(f"loss_stage_{stage_suffix}"), dict):
        loss = _as_dict(cfg.get(f"loss_stage_{stage_suffix}"))
    for key in ("lambda_1", "lambda_2", "lambda_e", "lambda_f", "lambda_v"):
        if key in loss:
            cfg[key] = deepcopy(loss[key])

    training = _as_dict(cfg.get("training"))
    if not training and isinstance(cfg.get(f"training_stage_{stage_suffix}"), dict):
        training = _as_dict(cfg.get(f"training_stage_{stage_suffix}"))
    for key in ("batch", "population", "generation", "train_test_split", "shuffle"):
        if key in training:
            cfg[key] = deepcopy(training[key])
    save_potential = training.get("save_potential", cfg.get("save_potential"))
    if isinstance(save_potential, dict):
        if "interval" in save_potential:
            cfg["save_potential_every"] = save_potential["interval"]
        if "format" in save_potential:
            cfg["save_potential_format"] = save_potential["format"]
        if "save_restart" in save_potential:
            cfg["save_potential_save_restart"] = save_potential["save_restart"]

    return cfg


def _discover_workspace_root(path: Path) -> Path:
    for candidate in [path.parent, *path.parent.parents]:
        if (candidate / "workflow.py").exists() and (candidate / "configs").is_dir():
            return candidate
    return Path.cwd().resolve()


def _read_json_file(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return dict(payload or {}) if isinstance(payload, dict) else {}


def _is_native_nep_recipe_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".in", ".nep"}


def _native_nep_training_config(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_referenced_config_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"NEP native recipe not found: {resolved}")
    return {"native_recipe_path": str(resolved)}


def _reject_nep_selector_payload(payload: dict[str, Any], *, config_path: str | Path) -> None:
    selector_keys = sorted(set(payload) & set(TRAIN_RECIPE_BRANCHES))
    if selector_keys:
        raise ValueError(
            "NEP train recipe no longer accepts selector fields "
            f"{selector_keys}; point the train stage directly at a backend-native recipe: {config_path}"
        )


def _reject_json_nep_native_recipe_payload(payload: dict[str, Any], *, config_path: str | Path) -> None:
    nep_payload = _unwrap_nep_config_payload(payload)
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
        "save_potential_every",
        "save_potential_format",
        "save_potential_save_restart",
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
            f"{present}; create a user-maintained nep.in file and point train_recipe_ref at it: {config_path}"
        )


def _resolve_referenced_config_path(value: str | Path, *, base_config_path: str | Path | None = None) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    base_path = Path(base_config_path).resolve() if base_config_path is not None else None
    workspace_root = _discover_workspace_root(base_path) if base_path is not None else Path.cwd().resolve()
    root_candidate = (workspace_root / raw).resolve()
    if root_candidate.exists():
        return root_candidate
    if base_path is not None:
        sibling_candidate = (base_path.parent / raw).resolve()
        if sibling_candidate.exists():
            return sibling_candidate
    return root_candidate


def _unwrap_nep_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("nep", payload) or {})


def _selector_paths_from_payload(
    selector_payload: dict[str, Any],
    *,
    base_config_path: str | Path | None = None,
) -> dict[str, str]:
    payload = dict(selector_payload or {})
    selector_paths: dict[str, str] = {}
    for branch in TRAIN_RECIPE_BRANCHES:
        value = payload.get(branch)
        if not value:
            continue
        selector_paths[branch] = str(_resolve_referenced_config_path(value, base_config_path=base_config_path))
    return selector_paths


def load_train_recipe_selector_paths(config_path: str | Path) -> dict[str, str]:
    resolved_path = _resolve_referenced_config_path(config_path)
    payload = _read_json_file(resolved_path)
    return _selector_paths_from_payload(payload, base_config_path=resolved_path)


def _load_nep_training_config_from_payload(
    payload: dict[str, Any],
    *,
    base_config_path: str | Path | None = None,
    training_stage: str | None = "bootstrap",
    recipe_branch: str | None = None,
) -> dict[str, Any]:
    del training_stage
    _reject_nep_selector_payload(payload, config_path=base_config_path or "<inline>")
    selector_paths = _selector_paths_from_payload(payload, base_config_path=base_config_path)
    selected = selector_paths.get(recipe_branch or "scratch_entry")
    if selected:
        selected_path = _resolve_referenced_config_path(selected, base_config_path=base_config_path)
        selected_payload = _read_json_file(selected_path)
        selected_nep = _unwrap_nep_config_payload(selected_payload)

        merged: dict[str, Any] = {}
        scratch_path = selector_paths.get("scratch_entry")
        if recipe_branch == "continuation" and scratch_path:
            bootstrap_path = _resolve_referenced_config_path(scratch_path, base_config_path=base_config_path)
            bootstrap_payload = _read_json_file(bootstrap_path)
            merged = _unwrap_nep_config_payload(bootstrap_payload)

        merged = _deep_update(merged, selected_nep)
        parent_nep = _unwrap_nep_config_payload(payload)
        if parent_nep:
            merged = _deep_update(merged, parent_nep)
        return _normalize_nep_config(merged)

    if "nep" in payload:
        return _normalize_nep_config(_unwrap_nep_config_payload(payload))
    return _normalize_nep_config(_unwrap_nep_config_payload(payload))


def load_nep_training_config(
    config_path: str | Path,
    *,
    training_stage: str | None = "bootstrap",
    recipe_branch: str | None = None,
) -> dict[str, Any]:
    """Load the NEP train config selected by a thin student committee config."""
    if _is_native_nep_recipe_path(config_path):
        return _native_nep_training_config(config_path)
    resolved_path = _resolve_referenced_config_path(config_path)
    if _is_native_nep_recipe_path(resolved_path):
        return _native_nep_training_config(resolved_path)
    payload = _read_json_file(resolved_path)
    _reject_json_nep_native_recipe_payload(payload, config_path=resolved_path)
    return _load_nep_training_config_from_payload(
        payload,
        base_config_path=resolved_path,
        training_stage=training_stage,
        recipe_branch=recipe_branch,
    )


def _nep_train_config_paths(config_path: str | Path) -> dict[str, str]:
    resolved_path = _resolve_referenced_config_path(config_path)
    payload = _read_json_file(resolved_path)
    return _selector_paths_from_payload(payload, base_config_path=resolved_path)


def _normalize_gpumd_config(gpumd_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(gpumd_cfg)
    md = _as_dict(cfg.get("md"))
    if "ensemble" in md:
        cfg["ensemble"] = md["ensemble"]
    if "timestep_fs" in md:
        cfg["timestep_fs"] = md["timestep_fs"]
    if "thermostat_damp_fs" in md:
        cfg["thermostat_damp"] = md["thermostat_damp_fs"]
    if "thermostat_damp" in md:
        cfg["thermostat_damp"] = md["thermostat_damp"]
    if "velocity_seed" in md:
        cfg["velocity_seed"] = md["velocity_seed"]
    if "zero_rotation" in md:
        cfg["zero_rotation"] = md["zero_rotation"]

    uncertainty = _as_dict(cfg.get("uncertainty"))
    if "active_interval" in uncertainty:
        cfg["active_interval"] = uncertainty["active_interval"]
    if "active_threshold" in uncertainty:
        cfg["active_threshold"] = uncertainty["active_threshold"]
    if "active_has_force" in uncertainty:
        cfg["save_forces"] = bool(int(uncertainty["active_has_force"]))
    if "active_has_velocity" in uncertainty:
        cfg["save_velocities"] = bool(int(uncertainty["active_has_velocity"]))

    return cfg


def _validate_nep_settings(
    nep_cfg: dict[str, Any],
    gpumd_cfg: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    student_model_cfg: dict[str, Any] | None = None,
    *,
    require_gpumd_temperatures: bool = True,
) -> None:
    identity_cfg = _as_dict(student_model_cfg)
    type_map = [str(item) for item in identity_cfg.get("type_map", nep_cfg.get("type_map", [])) or []]
    if not type_map:
        raise ValueError("NEP backend requires student_model.type_map to be a non-empty list")
    committee_size = int(identity_cfg.get("committee_size", nep_cfg.get("committee_size", 0)) or 0)
    if committee_size <= 0:
        raise ValueError("NEP backend requires student_model.committee_size > 0")
    split_seed = int(nep_cfg.get("split_seed", 42))
    if split_seed < 0:
        raise ValueError("NEP backend requires nep.split_seed >= 0")
    split = float(nep_cfg.get("train_test_split", 0.95))
    if not 0.0 < split <= 1.0:
        raise ValueError("nep.train_test_split must be in (0, 1]")
    trust_lo = float(uncertainty_cfg.get("trust_lo", 0.10))
    trust_hi = float(uncertainty_cfg.get("trust_hi", 0.40))
    if trust_lo < 0.0 or trust_hi < trust_lo:
        raise ValueError("uncertainty trust thresholds must satisfy 0 <= trust_lo <= trust_hi")
    temperatures = list(gpumd_cfg.get("temperatures", []) or [])
    if require_gpumd_temperatures and not temperatures:
        raise ValueError("NEP backend requires gpumd.temperatures to be non-empty")


def _normalize_student_model_identity(student_model_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(student_model_cfg)
    type_map = [str(item) for item in cfg.get("type_map", []) or []]
    if not type_map:
        raise ValueError("student_model.type_map must be a non-empty list")
    committee_size = int(cfg.get("committee_size", 0) or 0)
    if committee_size <= 0:
        raise ValueError("student_model.committee_size must be positive")
    cfg["type_map"] = type_map
    cfg["committee_size"] = committee_size
    return cfg


def _normalize_deepmd_identity(deepmd_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(deepmd_cfg)
    type_map = [str(item) for item in cfg.get("type_map", []) or []]
    if not type_map:
        raise ValueError("DeepMD backend requires deepmd.type_map to be a non-empty list")
    committee_size = int(cfg.get("committee_size", 0) or 0)
    if committee_size <= 0:
        raise ValueError("DeepMD backend requires deepmd.committee_size > 0")
    cfg["type_map"] = type_map
    cfg["committee_size"] = committee_size
    return cfg


def resolve_student_backend_config(raw_cfg: dict[str, Any], *, config_path: str | Path | None = None) -> dict[str, Any]:
    retired_keys = sorted(set(raw_cfg) & RETIRED_FIXED_LOOP_BACKEND_FIELDS)
    if retired_keys:
        raise ValueError(f"student backend config contains retired fixed-loop fields: {retired_keys}")
    dft_policy = _as_dict(raw_cfg.get("dft_screening_policy"))
    retired_dft_keys = sorted(set(dft_policy) & RETIRED_DFT_SCREENING_FIELDS)
    if retired_dft_keys:
        raise ValueError(f"dft_screening_policy contains retired fixed-loop fields: {retired_dft_keys}")

    resolved = dict(raw_cfg)
    student_backend = _infer_student_backend_name(raw_cfg)
    resolved["student_backend"] = student_backend
    student_model_cfg = _normalize_student_model_identity(_as_dict(raw_cfg.get("student_model")))
    resolved["student_model"] = student_model_cfg

    uncertainty_cfg = _default_uncertainty_config()
    gpumd_raw = _as_dict(raw_cfg.get("gpumd"))
    if isinstance(gpumd_raw.get("uncertainty"), dict):
        uncertainty_cfg = _deep_update(uncertainty_cfg, dict(gpumd_raw.get("uncertainty", {}) or {}))
    if isinstance(raw_cfg.get("uncertainty"), dict):
        uncertainty_cfg = _deep_update(uncertainty_cfg, dict(raw_cfg.get("uncertainty", {}) or {}))
    if raw_cfg.get("student_trust_lo") is not None:
        uncertainty_cfg["trust_lo"] = float(raw_cfg["student_trust_lo"])
    if raw_cfg.get("student_trust_hi") is not None:
        uncertainty_cfg["trust_hi"] = float(raw_cfg["student_trust_hi"])

    uncertainty_cfg.setdefault("metric", "force_std_max")
    uncertainty_cfg["trust_lo"] = float(uncertainty_cfg.get("trust_lo", 0.10))
    uncertainty_cfg["trust_hi"] = float(uncertainty_cfg.get("trust_hi", 0.40))
    resolved["uncertainty"] = uncertainty_cfg

    if student_backend == "nep":
        nep_raw = dict(raw_cfg.get("nep", {}) or {})
        nep_cfg = _normalize_nep_config(_deep_update(_default_nep_config(), nep_raw))
        gpumd_raw = dict(raw_cfg.get("gpumd", {}) or {})
        gpumd_cfg = _normalize_gpumd_config(_deep_update(_default_gpumd_config(), gpumd_raw))
        _validate_nep_settings(
            nep_cfg,
            gpumd_cfg,
            uncertainty_cfg,
            student_model_cfg,
            require_gpumd_temperatures=str(raw_cfg.get("config_mode", "")).strip().lower() != "pipeline_templates",
        )
        resolved["nep"] = nep_cfg
        resolved["gpumd"] = gpumd_cfg
    else:
        resolved.setdefault("nep", deepcopy(_default_nep_config()))
        resolved.setdefault("gpumd", deepcopy(_default_gpumd_config()))
        if str(raw_cfg.get("config_mode", "")).strip().lower() == "pipeline_templates":
            resolved["deepmd"] = dict(raw_cfg.get("deepmd", {}) or {})
        elif isinstance(raw_cfg.get("deepmd"), dict) and raw_cfg.get("deepmd"):
            resolved["deepmd"] = _normalize_deepmd_identity(dict(raw_cfg.get("deepmd", {}) or {}))

    return resolved
