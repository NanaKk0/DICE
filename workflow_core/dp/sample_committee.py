from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.io import read, write

from workflow_backends.lammps.model_devi_driver import run_lammps_model_devi_sampling
from workflow_core.common.io import read_json, write_json


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged.get(key, {}) or {}), dict(value or {}))
        else:
            merged[key] = value
    return merged


def load_sampling_committee_config(config_path: str | Path | None) -> dict[str, Any]:
    payload = {}
    if config_path:
        config_path = Path(config_path).resolve()
        payload = read_json(config_path)

    payload.setdefault("backend", "lammps")
    payload.setdefault("driver_policy", "round_robin")
    payload.setdefault(
        "md",
        {
            "temperature_schedule": [273, 323, 373],
            "steps": 1000,
            "timestep_fs": 0.5,
            "eval_every": 20,
            "log_every": 10,
            "seed": 20260319,
            "friction": 0.05,
            "force_temp": True,
            "zero_rotation": False,
        },
    )
    payload.setdefault(
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
            "max_parallel_tasks": 1,
            "device_policy": "cpu",
            "execution_backend": "inline",
            "task_group_size": 1,
            "max_parallel_groups": 0,
            "group_parallel_degree": 1,
            "group_fail_fast": True,
            "task_layout_version": "v2",
            "postprocess": {
                "enabled": True,
                "compute_novelty": True,
                "compute_rmsd": True,
                "classify_suspicious_candidate": True,
                "build_high_tail_buffer": True,
            },
        },
    )
    payload.setdefault(
        "uncertainty",
        {
            "mode": "fixed",
            "level_f_lo": 0.10,
            "level_f_hi": 0.40,
            "use_metrics": ["force_std_max", "force_std_avg", "force_std_p90"],
            "decision_metric": "force_std_max",
        },
    )
    payload.setdefault(
        "hard_stop",
        {
            "hard_min_pair_distance": 0.65,
            "max_force_hard": 20.0,
            "max_eatom_jump": 10.0,
        },
    )
    payload.setdefault(
        "save_policy",
        {
            "save_interval": 10,
            "max_saved_per_traj": 20,
            "rmsd_save_threshold": 0.1,
            "bond_change_save": 0.08,
            "height_change_save": 0.1,
            "dynamic_reference": True,
        },
    )
    payload.setdefault(
        "novelty",
        {
            "feature_type": "geometric",
            "novelty_quantile": 0.8,
            "novelty_threshold_floor": 0.1,
        },
    )
    payload.setdefault(
        "early_stop",
        {
            "min_prod_steps": 150,
            "window_struct": 20,
            "mean_rmsd_stop": 0.08,
            "max_rmsd_stop": 0.15,
            "window_novelty": 20,
            "novelty_low_frac": 0.8,
        },
    )
    payload.setdefault(
        "geometry_filter",
        {
            "enabled": True,
            "hard_min_pair_distance": 0.65,
            "soft_min_pair_distance": 0.85,
        },
    )
    payload.setdefault(
        "selection",
        {
            "min_frame_gap": 5,
            "max_candidates_total": 200,
            "max_candidates_per_traj": 12,
            "bin_count": 5,
            "rmsd_threshold": 0.15,
        },
    )
    payload.setdefault(
        "export",
        {
            "write_xyz": True,
            "write_jsonl": True,
        },
    )
    payload["selection"]["decision_metric"] = payload["uncertainty"].get("decision_metric", "force_std_max")
    return payload


def load_committee_pool(summary_path: str | Path):
    summary_path = Path(summary_path).resolve()
    summary = read_json(summary_path)

    members = []
    for member in summary.get("members", []):
        graph_pb = Path(member["graph_pb"]).resolve()
        if member.get("status") == "success" and graph_pb.exists():
            updated = dict(member)
            updated["graph_pb"] = str(graph_pb)
            members.append(updated)

    if len(members) < 2:
        raise RuntimeError(
            f"Usable DP committee size is {len(members)}; at least 2 successful graph.pb files are required"
        )

    return summary, members


def collect_initial_structures(paths, round_id: int, source: str, work_root: Path, context: dict[str, Any] | None = None) -> list[str]:
    context = context or {}
    initial_dir = (work_root / "initial_structures").resolve()
    initial_dir.mkdir(parents=True, exist_ok=True)

    atoms_list = []
    if source == "seed_pool":
        seed_pool_default = paths.seed_pool_xyz() if context.get("seed_pool_override") is None else context["seed_pool_override"]
        seed_pool = Path(seed_pool_default).resolve()
        if not seed_pool.exists():
            raise FileNotFoundError(f"Seed pool xyz not found: {seed_pool}")
        atoms_list = list(read(str(seed_pool.resolve()), index=":"))
    elif source == "manual_seed_dir":
        manual_dir = paths.manual_seed_dir(round_id).resolve()
        xyz_files = sorted(manual_dir.glob("*.xyz"))
        if not xyz_files:
            raise RuntimeError(f"No xyz files found in {manual_dir}")
        for xyz_file in xyz_files:
            atoms_list.extend(list(read(str(xyz_file), index=":")))
    else:
        raise ValueError(f"Unsupported initial_structure_source: {source}")

    output_paths = []
    for index, atoms in enumerate(atoms_list):
        path = (initial_dir / f"seed_{index:06d}.xyz").resolve()
        write(path, atoms, format="extxyz")
        output_paths.append(str(path))

    if not output_paths:
        raise RuntimeError("No initial structures prepared for committee sampling")

    return output_paths


def _apply_sampling_overrides(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    payload = dict(cfg)
    sampling_seed_policy = dict(context.get("sampling_seed_policy", {}) or {})
    if sampling_seed_policy:
        payload["backend"] = str(sampling_seed_policy.get("backend", payload.get("backend", "lammps")))
        payload["driver_policy"] = str(sampling_seed_policy.get("driver_policy", payload.get("driver_policy", "round_robin")))
        payload["md"] = _deep_update(dict(payload.get("md", {}) or {}), dict(sampling_seed_policy.get("md", {}) or {}))
        if "lammps" in sampling_seed_policy:
            payload["lammps"] = _deep_update(
                dict(payload.get("lammps", {}) or {}),
                dict(sampling_seed_policy.get("lammps", {}) or {}),
            )
    profile_cfg = dict(context.get("sampling_profile_config", {}) or {})
    if profile_cfg:
        payload["backend"] = str(profile_cfg.get("backend", payload.get("backend", "lammps")))
        payload["driver_policy"] = str(profile_cfg.get("driver_policy", payload.get("driver_policy", "round_robin")))
        if "md" in profile_cfg:
            payload["md"] = _deep_update(dict(payload.get("md", {}) or {}), dict(profile_cfg.get("md", {}) or {}))
        if "lammps" in profile_cfg:
            payload["lammps"] = _deep_update(dict(payload.get("lammps", {}) or {}), dict(profile_cfg.get("lammps", {}) or {}))
    if "candidate_train_selection" in context:
        payload["candidate_train_selection"] = dict(context.get("candidate_train_selection", {}) or {})
    if "sampling_profile_name" in context:
        payload["sampling_profile_name"] = str(context.get("sampling_profile_name"))
    return payload


def run_dp_sample_committee(cfg_path: str | Path | None, paths, round_id: int, context: dict[str, Any] | None = None) -> str:
    context = context or {}
    cfg = _apply_sampling_overrides(load_sampling_committee_config(cfg_path), context)
    summary_override = context.get("committee_summary_override")
    if summary_override is not None:
        summary_path = Path(summary_override).resolve()
    else:
        summary_path = paths.dp_committee_summary_json(round_id)
    summary, committee_pool = load_committee_pool(summary_path)

    work_root = Path(context.get("work_root_override", paths.dp_sampling_committee_dir(round_id))).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    initial_structures = collect_initial_structures(
        paths=paths,
        round_id=round_id,
        source=str(context.get("initial_structure_source_override", "seed_pool")),
        work_root=work_root,
        context=context,
    )

    backend = str(cfg.get("backend", "lammps")).strip().lower()
    if backend != "lammps":
        raise ValueError(f"Unsupported sampling backend: {backend}. Only backend='lammps' is supported.")
    return run_lammps_model_devi_sampling(
        initial_structure_paths=initial_structures,
        committee_pool=committee_pool,
        cfg=cfg,
        work_root=work_root,
        summary=summary,
        context=context,
    )


def run_task(cfg, paths, round_id: int) -> None:
    cfg_path = cfg.get("config", "configs/dp/sample_committee.json")
    run_dp_sample_committee(cfg_path, paths, round_id)


