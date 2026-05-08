from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from workflow_artifacts import seed_pool_meta as _seed_pool_meta
from workflow_core.common.io import (
    append_candidate_xyz,
    ensure_dir,
    load_atoms_list,
    read_json,
    write_candidate_xyz,
    write_json,
)
from workflow_core.seed_pool.uncertainty import (
    combine_grouped_atoms,
    deduplicate_atoms_by_structure_id,
    ensure_minimal_seed_metadata,
    group_atoms_by_system,
    infer_seed_system_id,
    load_system_extxyz_dir,
    rescore_atoms_with_committee,
    sanitize_system_id,
    sort_atoms_by_uncertainty_desc,
    write_system_extxyz_dir,
)
from workflow_policies import seed_selection as _seed_selection
ROLE_ANCHOR = "anchor"


def _load_seed_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_cfg = dict(cfg.get("seed_pool_policy", cfg.get("seed_pool", {})) or {})
    seed_cfg.setdefault("enabled", True)
    seed_cfg.setdefault("initialize_at_subworkflow_start", True)
    seed_cfg.setdefault("max_pool_size_per_system", 2000)
    seed_cfg.setdefault("refresh_after_training", True)
    return seed_cfg


def _committee_graphs_from_summary(summary_path: str | Path) -> list[str]:
    payload = read_json(summary_path)
    members = []
    for member in payload.get("members", []):
        graph_pb = Path(member.get("graph_pb", "")).resolve()
        if member.get("status") == "success" and graph_pb.exists():
            members.append(str(graph_pb))
    if len(members) < 2:
        raise RuntimeError(f"Usable DP committee size is {len(members)}; at least 2 successful graph.pb files are required")
    return members


def _seed_refresh_cfg(
    seed_cfg: dict[str, Any],
    *,
    keep_if_ge_low: float | None = None,
    keep_if_lt_high: float | None = None,
) -> dict[str, Any]:
    return {
        "metric": "force_std_max",
        "keep_if_ge_low": keep_if_ge_low,
        "keep_if_lt_high": float(0.40 if keep_if_lt_high is None else keep_if_lt_high),
        "max_pool_size_per_system": int(seed_cfg.get("max_pool_size_per_system", 2000)),
        "dedup_enabled": True,
    }


def _stable_random_seed(*parts: object, base_seed: int) -> int:
    return _seed_selection.stable_random_seed(*parts, base_seed=base_seed)


def _times_used(atoms: Atoms) -> int:
    return _seed_selection.times_used(atoms)


def _set_times_used(atoms: Atoms, value: int) -> None:
    _seed_selection.set_times_used(atoms, value)


def _sort_atoms_by_usage_then_uncertainty_desc(atoms_list: list[Atoms]) -> list[Atoms]:
    return _seed_selection.sort_atoms_by_usage_then_uncertainty_desc(atoms_list)


def _resolve_seed_selection_mode(profile_cfg: dict[str, Any], *, dp_round: int) -> str:
    return _seed_selection.resolve_seed_selection_mode(profile_cfg, dp_round=dp_round)


def _fill_seed_pool_by_system(
    atoms_list: list[Atoms],
    *,
    selection_mode: str,
    max_pool_size: int,
    random_seed: int,
    outer_round: int,
    dp_round: int,
    system_id: str,
) -> list[Atoms]:
    return _seed_selection.fill_seed_pool_by_system(
        atoms_list,
        selection_mode=selection_mode,
        max_pool_size=max_pool_size,
        random_seed=random_seed,
        outer_round=outer_round,
        dp_round=dp_round,
        system_id=system_id,
    )


def _write_formal_seed_pool(paths, grouped_atoms: dict[str, list[Atoms]]) -> Path:
    write_system_extxyz_dir(paths.seed_pool_systems_dir(), grouped_atoms)
    write_candidate_xyz(paths.seed_pool_xyz(), combine_grouped_atoms(grouped_atoms))
    return paths.seed_pool_xyz().resolve()


def _load_formal_seed_pool_grouped(paths) -> dict[str, list[Atoms]]:
    grouped = load_system_extxyz_dir(paths.seed_pool_systems_dir())
    if grouped:
        return grouped
    seed_pool_xyz = paths.seed_pool_xyz()
    if not seed_pool_xyz.exists() or seed_pool_xyz.stat().st_size == 0:
        return {}
    atoms_list = [ensure_minimal_seed_metadata(atoms) for atoms in load_atoms_list(seed_pool_xyz)]
    return group_atoms_by_system(atoms_list)


def _collect_seed_source_xyz(paths) -> list[Path]:
    source_dir = paths.user_initial_seed_pool_dir()
    xyz_files = sorted(source_dir.glob("*.xyz"))
    if not xyz_files:
        raise RuntimeError(
            "No initial physical seed structures found.\n"
            "Please provide xyz files in:\n"
            "user_assets/seed_pool/"
        )
    return [path.resolve() for path in xyz_files]


def _merge_atoms_from_xyz_files(xyz_files: list[Path]) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for xyz_file in xyz_files:
        atoms_list.extend(load_atoms_list(xyz_file))
    return atoms_list


def _update_seed_pool_meta(paths, seed_cfg: dict[str, Any], seed_pool_atoms: list[Atoms], update_info: dict[str, Any]) -> Path:
    return Path(_seed_pool_meta.update_seed_pool_meta(paths, seed_cfg, seed_pool_atoms, update_info)).resolve()


def init_seed_pool(cfg, paths) -> str:
    seed_cfg = _load_seed_cfg(cfg)
    ensure_dir(paths.seed_pool_root_dir())
    ensure_dir(paths.seed_pool_archive_dir())
    ensure_dir(paths.seed_pool_selected_dir())
    ensure_dir(paths.seed_pool_results_dir())
    ensure_dir(paths.seed_pool_systems_dir())
    ensure_dir(paths.seed_candidate_pool_systems_dir())

    source_xyz = _collect_seed_source_xyz(paths)
    atoms_list = _merge_atoms_from_xyz_files(source_xyz)
    if not atoms_list:
        raise RuntimeError("Initial physical seed pool is empty")

    for idx, atoms in enumerate(atoms_list):
        system_id = infer_seed_system_id(atoms)
        structure_id = str(atoms.info.get("structure_id") or atoms.info.get("seed_structure_id") or f"initial_{idx:06d}")
        ensure_minimal_seed_metadata(atoms, system_id=system_id, structure_id=structure_id)
        atoms.info["seed_pool_source"] = "initial_physical_seed"
        atoms.info["seed_source_type"] = "initial_physical_seed"
        atoms.info["seed_pool_role"] = ROLE_ANCHOR
        atoms.info["seed_formula"] = atoms.get_chemical_formula(mode="hill")
        atoms.info["seed_system_id"] = system_id
        atoms.info["seed_pool_init_index"] = idx
        atoms.info["source_round"] = "initial"
        atoms.info["seed_round_added"] = -1
        atoms.info["seed_last_used_round"] = -1
        atoms.info["times_used"] = 0
        atoms.info["seed_use_count"] = 0
        atoms.info["history_novelty_score"] = 0.0
        atoms.info["frontier_score"] = 0.0
        atoms.info["cooldown_until_round"] = -1

    grouped_atoms = group_atoms_by_system(atoms_list)
    _write_formal_seed_pool(paths, grouped_atoms)
    write_candidate_xyz(paths.seed_candidate_pool_xyz(), [])
    write_system_extxyz_dir(paths.seed_candidate_pool_systems_dir(), {})
    _update_seed_pool_meta(
        paths,
        seed_cfg,
        combine_grouped_atoms(grouped_atoms),
        {
            "initialized": True,
            "updated_outer_round": None,
            "updated_dp_round": None,
            "source_files": [str(path) for path in source_xyz],
        },
    )
    return str(paths.seed_pool_xyz().resolve())


def collect_seed_candidates_from_md(cfg, paths, outer_round, dp_round, sampling_dir) -> str:
    sampling_dir = Path(sampling_dir).resolve()
    candidate_by_system_dir = sampling_dir / "candidate_by_system"
    archive_file = paths.seed_pool_archive_candidate_xyz(outer_round, dp_round)
    archive_summary = paths.seed_pool_archive_candidate_summary_json(outer_round, dp_round)
    grouped_candidates = load_system_extxyz_dir(candidate_by_system_dir)
    candidate_atoms = combine_grouped_atoms(grouped_candidates)
    write_candidate_xyz(archive_file, candidate_atoms)
    append_candidate_xyz(paths.seed_candidate_pool_xyz(), candidate_atoms)
    write_system_extxyz_dir(paths.seed_candidate_pool_systems_dir(), grouped_candidates)
    write_json(
        archive_summary,
        {
            "outer_round": int(outer_round),
            "dp_round": int(dp_round),
            "candidate_count_raw": len(candidate_atoms),
            "candidate_count_physical": len(candidate_atoms),
            "archive_candidate_file": str(archive_file.resolve()),
            "candidate_by_system_dir": str(candidate_by_system_dir.resolve()),
        },
    )
    return str(candidate_by_system_dir.resolve())


def update_seed_pool(cfg, paths, outer_round, dp_round, candidate_file) -> str:
    seed_cfg = _load_seed_cfg(cfg)
    candidate_file = Path(candidate_file).resolve()
    candidate_summary_path = paths.seed_pool_archive_candidate_summary_json(outer_round, dp_round)
    candidate_summary = read_json(candidate_summary_path) if candidate_summary_path.exists() else {}
    current_grouped = _load_formal_seed_pool_grouped(paths)
    if candidate_file.is_dir():
        candidate_grouped = load_system_extxyz_dir(candidate_file)
    else:
        candidate_grouped = group_atoms_by_system(load_atoms_list(candidate_file) if candidate_file.exists() and candidate_file.stat().st_size > 0 else [])
    refresh_cfg = _seed_refresh_cfg(seed_cfg)
    threshold_hi = float(refresh_cfg["keep_if_lt_high"])
    max_size = int(refresh_cfg["max_pool_size_per_system"])

    merged_grouped: dict[str, list[Atoms]] = {}
    for system_id in sorted(set(current_grouped) | set(candidate_grouped)):
        merged_atoms = [atoms.copy() for atoms in current_grouped.get(system_id, [])] + [
            atoms.copy() for atoms in candidate_grouped.get(system_id, [])
        ]
        merged_atoms = deduplicate_atoms_by_structure_id(merged_atoms)
        merged_atoms = [atoms for atoms in merged_atoms if float(atoms.info.get("force_std_max", 0.0) or 0.0) < threshold_hi]
        merged_grouped[system_id] = sort_atoms_by_uncertainty_desc(merged_atoms)[:max_size]

    current_atoms = combine_grouped_atoms(current_grouped)
    next_atoms = combine_grouped_atoms(merged_grouped)
    size_before = len(current_atoms)
    candidate_count_physical = len(combine_grouped_atoms(candidate_grouped))
    merged_count = len(next_atoms)

    _write_formal_seed_pool(paths, merged_grouped)
    write_candidate_xyz(paths.seed_candidate_pool_xyz(), [])
    write_system_extxyz_dir(paths.seed_candidate_pool_systems_dir(), {})

    summary = {
        "outer_round": int(outer_round),
        "dp_round": int(dp_round),
        "seed_pool_size_before": size_before,
        "candidate_count_raw": int(candidate_summary.get("candidate_count_raw", candidate_count_physical)),
        "candidate_count_physical": candidate_count_physical,
        "merged_count": merged_count,
        "seed_pool_size_after": len(next_atoms),
        "max_pool_size_per_system": max_size,
        "compression_applied": False,
        "compression_method": "uncertainty_only_existing_scores",
        "candidate_file": str(candidate_file.resolve()),
        "seed_pool_file": str(paths.seed_pool_xyz().resolve()),
    }
    write_json(paths.seed_pool_update_summary_json(outer_round, dp_round), summary)
    _update_seed_pool_meta(
        paths,
        seed_cfg,
        next_atoms,
        {
            "initialized": True,
            "updated_outer_round": int(outer_round),
            "updated_dp_round": int(dp_round),
            "last_update_summary": str(paths.seed_pool_update_summary_json(outer_round, dp_round).resolve()),
        },
    )
    return str(paths.seed_pool_xyz().resolve())


def select_md_seeds_from_pool(cfg, paths, outer_round, dp_round, profile_cfg: dict[str, Any] | None = None) -> str:
    seed_cfg = _load_seed_cfg(cfg)
    profile_cfg = dict(profile_cfg or {})
    n_seed_per_system = int(((profile_cfg.get("seed_selection", {}) or {}).get("n_seed_per_system", 8)) or 0)
    selection_mode = _resolve_seed_selection_mode(profile_cfg, dp_round=dp_round)
    grouped_pool = _load_formal_seed_pool_grouped(paths)
    selected_seed_file = paths.seed_pool_selected_xyz(outer_round, dp_round)
    selected_grouped: dict[str, list[Atoms]] = {}
    for system_id, atoms_list in grouped_pool.items():
        copied = [atoms.copy() for atoms in atoms_list]
        if selection_mode == "usage_then_uncertainty_per_system":
            ranked = _sort_atoms_by_usage_then_uncertainty_desc(copied)
        else:
            ranked = sort_atoms_by_uncertainty_desc(copied)
        selected_grouped[system_id] = ranked[: max(0, n_seed_per_system)]
    write_system_extxyz_dir(paths.seed_pool_selected_systems_dir(outer_round, dp_round), selected_grouped)
    write_candidate_xyz(selected_seed_file, combine_grouped_atoms(selected_grouped))
    return str(selected_seed_file.resolve())


def refresh_seed_pool_after_training(
    cfg,
    paths,
    outer_round: int,
    dp_round: int,
    committee_summary_path: str | Path,
    *,
    refresh_mode: str,
    keep_if_ge_low: float | None = None,
    keep_if_lt_high: float | None = None,
    reset_times_used: bool = False,
) -> str:
    seed_cfg = _load_seed_cfg(cfg)
    refresh_cfg = _seed_refresh_cfg(seed_cfg, keep_if_ge_low=keep_if_ge_low, keep_if_lt_high=keep_if_lt_high)
    committee_graphs = _committee_graphs_from_summary(committee_summary_path)
    current_grouped = _load_formal_seed_pool_grouped(paths)
    source_grouped: dict[str, list[Atoms]] = {
        system_id: [atoms.copy() for atoms in atoms_list]
        for system_id, atoms_list in current_grouped.items()
    }

    refresh_mode = str(refresh_mode).strip().lower()
    if refresh_mode == "merge_old_pool_with_current_candidate_failed":
        candidate_grouped = load_system_extxyz_dir(paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round) / "candidate_by_system")
        failed_grouped = load_system_extxyz_dir(paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round) / "failed_by_system")
        for grouped_source in (candidate_grouped, failed_grouped):
            for system_id, atoms_list in grouped_source.items():
                source_grouped.setdefault(system_id, [])
                source_grouped[system_id].extend(atoms.copy() for atoms in atoms_list)
    elif refresh_mode != "rescore_existing_pool_only":
        raise ValueError(f"Unsupported seed pool refresh mode: {refresh_mode}")

    refreshed_grouped: dict[str, list[Atoms]] = {}
    threshold_low = refresh_cfg.get("keep_if_ge_low")
    threshold_hi = float(refresh_cfg["keep_if_lt_high"])
    max_pool_size = int(refresh_cfg["max_pool_size_per_system"])
    dedup_enabled = bool(refresh_cfg.get("dedup_enabled", True))
    refresh_mode_norm = str(refresh_mode).strip().lower()
    selection_mode = "random_per_system" if refresh_mode_norm == "merge_old_pool_with_current_candidate_failed" else "top_uncertainty_per_system"
    refresh_seed_cfg = dict(
        (
            cfg.get("sampling_profiles", {})
            .get("long_exploration" if refresh_mode_norm == "merge_old_pool_with_current_candidate_failed" else "short_refinement", {})
            .get("seed_pool", {})
        )
        or {}
    )
    random_seed = int(refresh_seed_cfg.get("random_seed", 20260502))
    temp_root = ensure_dir(paths.seed_pool_root_dir() / "_refresh_tmp" / f"r{outer_round}_dp{dp_round}")
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for system_id in sorted(source_grouped):
        atoms_list = [ensure_minimal_seed_metadata(atoms.copy(), system_id=system_id) for atoms in source_grouped.get(system_id, [])]
        if dedup_enabled:
            atoms_list = deduplicate_atoms_by_structure_id(atoms_list)
        if not atoms_list:
            refreshed_grouped[system_id] = []
            continue
        rescored = rescore_atoms_with_committee(
            atoms_list,
            committee_graphs=committee_graphs,
            temp_root=temp_root,
            prefix=f"{sanitize_system_id(system_id)}_seed_pool",
        )
        filtered = []
        for atoms in rescored:
            value = float(atoms.info.get("force_std_max", 0.0) or 0.0)
            if value >= threshold_hi:
                continue
            if threshold_low is not None and value < float(threshold_low):
                continue
            filtered.append(atoms)
        eligible_counts[system_id] = len(filtered)
        selected = _fill_seed_pool_by_system(
            filtered,
            selection_mode=selection_mode,
            max_pool_size=max_pool_size,
            random_seed=random_seed,
            outer_round=outer_round,
            dp_round=dp_round,
            system_id=system_id,
        )
        if reset_times_used:
            for atoms in selected:
                _set_times_used(atoms, 0)
                atoms.info["seed_last_used_round"] = -1
        refreshed_grouped[system_id] = selected
        selected_counts[system_id] = len(selected)

    pool_path = _write_formal_seed_pool(paths, refreshed_grouped)
    summary_path = paths.seed_pool_update_summary_json(outer_round, dp_round)
    summary = {
        "outer_round": int(outer_round),
        "dp_round": int(dp_round),
        "refresh_mode": refresh_mode,
        "committee_summary_path": str(Path(committee_summary_path).resolve()),
        "keep_if_ge_low": None if threshold_low is None else float(threshold_low),
        "keep_if_lt_high": threshold_hi,
        "max_pool_size_per_system": max_pool_size,
        "pool_fill_selection_mode": selection_mode,
        "pool_random_seed": random_seed,
        "pool_size_after": len(combine_grouped_atoms(refreshed_grouped)),
        "eligible_count_per_system": {system_id: eligible_counts.get(system_id, 0) for system_id in sorted(eligible_counts)},
        "selected_count_per_system": {system_id: selected_counts.get(system_id, 0) for system_id in sorted(selected_counts)},
        "system_counts_after": {system_id: len(items) for system_id, items in sorted(refreshed_grouped.items())},
        "seed_pool_file": str(pool_path),
        "seed_pool_systems_dir": str(paths.seed_pool_systems_dir().resolve()),
        "reset_times_used": bool(reset_times_used),
    }
    write_json(summary_path, summary)
    _update_seed_pool_meta(
        paths,
        seed_cfg,
        combine_grouped_atoms(refreshed_grouped),
        {
            "initialized": True,
            "updated_outer_round": int(outer_round),
            "updated_dp_round": int(dp_round),
            "last_update_summary": str(summary_path.resolve()),
            "refresh_mode": refresh_mode,
        },
    )
    return str(summary_path.resolve())


def record_seed_usage_after_sampling(paths, outer_round: int, dp_round: int) -> str:
    grouped_pool = _load_formal_seed_pool_grouped(paths)
    selected_seed_file = paths.dp_subworkflow_seed_pool_xyz(outer_round, dp_round).resolve()
    if not selected_seed_file.exists() or selected_seed_file.stat().st_size == 0:
        return str(paths.seed_pool_xyz().resolve())

    selected_atoms = [ensure_minimal_seed_metadata(atoms) for atoms in load_atoms_list(selected_seed_file)]
    selected_ids = {str(atoms.info["structure_id"]) for atoms in selected_atoms}
    if not selected_ids:
        return str(paths.seed_pool_xyz().resolve())

    for system_id, atoms_list in grouped_pool.items():
        for atoms in atoms_list:
            if str(atoms.info.get("structure_id", "")) in selected_ids:
                _set_times_used(atoms, _times_used(atoms) + 1)
                atoms.info["seed_last_used_round"] = int(dp_round)

    _write_formal_seed_pool(paths, grouped_pool)
    return str(paths.seed_pool_xyz().resolve())

