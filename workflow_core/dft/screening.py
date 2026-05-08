from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from workflow_core.common.io import ensure_dir, write_candidate_xyz, write_json


def resolve_dft_screening_policy(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    policy = dict(raw_cfg.get("dft_screening_policy", {}) or {})
    budget_per_system = int(policy.get("budget_per_system", 20))
    total_budget = max(1, budget_per_system) * 8
    return {
        "enabled": bool(policy.get("enabled", False)),
        "selection_mode": str(policy.get("selection_mode", "bucket_random_per_system")),
        "budget_per_system": budget_per_system,
        "random_seed": int(policy.get("random_seed", 20260503)),
        "signature_neighbor_order": max(1, int(policy.get("signature_neighbor_order", 3))),
        "signature_quantiles": [float(item) for item in policy.get("signature_quantiles", [0.10, 0.50, 0.90])],
        "bucket_width": float(policy.get("bucket_width", 4.0)),
        "keep_top_per_bucket": max(1, int(policy.get("keep_top_per_bucket", 1))),
        "stop_dp_round_when_current_screened_count_gte": int(
            policy.get("stop_dp_round_when_current_screened_count_gte", total_budget)
        ),
        "stop_dp_round_when_cumulative_screened_count_gte": int(
            policy.get("stop_dp_round_when_cumulative_screened_count_gte", total_budget)
        ),
    }


def _system_key(atoms) -> str:
    return str(atoms.info.get("formula") or atoms.get_chemical_formula(mode="hill"))


def _uncertainty(atoms) -> float:
    value = atoms.info.get("force_uncertainty", atoms.info.get("force_std_max", 0.0))
    try:
        return float(value)
    except Exception:
        return 0.0


def _stable_int(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _exact_structure_hash(atoms) -> str:
    numbers = np.asarray(atoms.numbers, dtype=np.int16)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    payload = [
        numbers.tobytes(),
        np.round(positions, 8).tobytes(),
        np.round(cell, 8).tobytes(),
        np.asarray(atoms.pbc, dtype=np.int8).tobytes(),
    ]
    return hashlib.sha256(b"".join(payload)).hexdigest()


def _dedup_exact_by_hash(atoms_list) -> list:
    grouped: dict[str, list] = defaultdict(list)
    for atoms in atoms_list:
        grouped[_exact_structure_hash(atoms)].append(atoms)
    deduped = []
    for bucket in grouped.values():
        bucket.sort(key=_uncertainty, reverse=True)
        deduped.append(bucket[0])
    return deduped


def _build_neighbor_signature(atoms, neighbor_order: int, quantiles: list[float]) -> np.ndarray:
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    n_atoms = distances.shape[0]
    if n_atoms <= 1:
        return np.zeros(neighbor_order * len(quantiles) + 1, dtype=float)
    np.fill_diagonal(distances, np.inf)
    usable_order = min(max(1, neighbor_order), n_atoms - 1)
    nearest = np.partition(distances, kth=usable_order - 1, axis=1)[:, :usable_order]
    parts: list[float] = []
    for rank in range(usable_order):
        rank_dist = nearest[:, rank]
        parts.extend(float(np.quantile(rank_dist, q)) for q in quantiles)
    if usable_order < neighbor_order:
        missing = (neighbor_order - usable_order) * len(quantiles)
        parts.extend([parts[-1] if parts else 0.0] * missing)
    finite_distances = distances[np.isfinite(distances)]
    min_pair_distance = float(np.min(finite_distances)) if finite_distances.size else 0.0
    parts.append(min_pair_distance)
    return np.asarray(parts, dtype=float)


def _robust_normalize(signatures: np.ndarray) -> np.ndarray:
    if signatures.size == 0:
        return signatures
    median = np.median(signatures, axis=0)
    mad = np.median(np.abs(signatures - median), axis=0)
    scale = np.where(mad > 1e-8, mad, 1.0)
    return (signatures - median) / scale


def _bucket_key(normalized_signature: np.ndarray, bucket_width: float) -> tuple[int, ...]:
    clipped = np.clip(normalized_signature, -8.0, 8.0)
    return tuple(int(x) for x in np.rint(clipped / max(bucket_width, 1e-6)))


def _choose_screened_atoms_for_system(
    atoms_list: list,
    *,
    budget_per_system: int,
    random_seed: int,
    neighbor_order: int,
    quantiles: list[float],
    bucket_width: float,
    keep_top_per_bucket: int,
    system_key: str,
    outer_round: int,
    dp_round: int,
) -> tuple[list, dict[str, Any]]:
    deduped = _dedup_exact_by_hash(atoms_list)
    if not deduped:
        return [], {
            "raw_count": 0,
            "exact_dedup_count": 0,
            "bucket_count": 0,
            "screened_count": 0,
            "uncertainty_min": None,
            "uncertainty_median": None,
            "uncertainty_max": None,
        }
    signatures = np.vstack([
        _build_neighbor_signature(atoms, neighbor_order=neighbor_order, quantiles=quantiles)
        for atoms in deduped
    ])
    normalized = _robust_normalize(signatures)
    grouped: dict[tuple[int, ...], list] = defaultdict(list)
    for atoms, normalized_signature in zip(deduped, normalized):
        grouped[_bucket_key(normalized_signature, bucket_width)].append(atoms)
    for bucket in grouped.values():
        bucket.sort(key=_uncertainty, reverse=True)
    layers: list[list] = []
    max_bucket_depth = max((min(len(bucket), keep_top_per_bucket) for bucket in grouped.values()), default=0)
    if max_bucket_depth <= 0:
        max_bucket_depth = max((len(bucket) for bucket in grouped.values()), default=0)
    for depth in range(max_bucket_depth):
        layer = []
        for bucket in grouped.values():
            if depth < len(bucket):
                layer.append(bucket[depth])
        layers.append(layer)
    selected: list = []
    selected_hashes: set[str] = set()
    system_seed = _stable_int(f"{random_seed}:{outer_round}:{dp_round}:{system_key}")
    rng = np.random.default_rng(system_seed)
    for layer in layers:
        remaining = budget_per_system - len(selected)
        if remaining <= 0:
            break
        if not layer:
            continue
        indices = np.arange(len(layer))
        rng.shuffle(indices)
        for index in indices:
            atoms = layer[int(index)]
            atom_hash = _exact_structure_hash(atoms)
            if atom_hash in selected_hashes:
                continue
            selected.append(atoms)
            selected_hashes.add(atom_hash)
            if len(selected) >= budget_per_system:
                break
    if len(selected) < budget_per_system:
        leftovers = []
        for bucket in grouped.values():
            for atoms in bucket:
                atom_hash = _exact_structure_hash(atoms)
                if atom_hash not in selected_hashes:
                    leftovers.append(atoms)
        if leftovers:
            indices = np.arange(len(leftovers))
            rng.shuffle(indices)
            for index in indices:
                atoms = leftovers[int(index)]
                selected.append(atoms)
                if len(selected) >= budget_per_system:
                    break
    uncertainties = [_uncertainty(atoms) for atoms in selected]
    return selected, {
        "raw_count": int(len(atoms_list)),
        "exact_dedup_count": int(len(deduped)),
        "bucket_count": int(len(grouped)),
        "screened_count": int(len(selected)),
        "uncertainty_min": float(min(uncertainties)) if uncertainties else None,
        "uncertainty_median": float(np.median(uncertainties)) if uncertainties else None,
        "uncertainty_max": float(max(uncertainties)) if uncertainties else None,
    }


def append_screened_round_into_pool(paths, outer_round: int, screened_xyz: str | Path) -> Path:
    from ase.io import read

    pool_path = paths.dp_subworkflow_screened_dft_pool_xyz(outer_round)
    ensure_dir(pool_path.parent)
    screened_path = Path(screened_xyz).resolve()
    atoms_list = []
    if screened_path.exists() and screened_path.stat().st_size > 0:
        atoms_list = list(read(str(screened_path), index=":"))
    write_candidate_xyz(pool_path, atoms_list)
    return pool_path.resolve()


def rebuild_screened_dft_pool(paths, outer_round: int, selected_xyz_files: list[str | Path]) -> Path:
    from ase.io import read

    pool_path = paths.dp_subworkflow_screened_dft_pool_xyz(outer_round)
    ensure_dir(pool_path.parent)
    chosen_path: Path | None = None
    for item in selected_xyz_files:
        xyz_path = Path(item).resolve()
        if xyz_path.exists():
            chosen_path = xyz_path
    atoms_list = []
    if chosen_path is not None and chosen_path.exists() and chosen_path.stat().st_size > 0:
        atoms_list = list(read(str(chosen_path), index=":"))
    write_candidate_xyz(pool_path, atoms_list)
    return pool_path.resolve()


def screen_current_round_dft_candidates(paths, outer_round: int, dp_round: int, policy: dict[str, Any]) -> dict[str, Any]:
    from ase.io import read

    output_dir = ensure_dir(paths.dp_subworkflow_round_dft_screening_dir(outer_round, dp_round))
    current_round_xyz = paths.dp_subworkflow_round_active_learning_dir(outer_round, dp_round) / "dft.xyz"
    if not current_round_xyz.exists():
        raise FileNotFoundError(f"DFT candidate xyz not found for screening: {current_round_xyz}")

    cumulative_candidates_xyz = paths.dp_subworkflow_mace_al_pool_xyz(outer_round)
    candidates_xyz = cumulative_candidates_xyz if cumulative_candidates_xyz.exists() else current_round_xyz
    atoms_list = list(read(str(candidates_xyz), index=":")) if candidates_xyz.exists() and candidates_xyz.stat().st_size > 0 else []

    grouped: dict[str, list] = defaultdict(list)
    for atoms in atoms_list:
        grouped[_system_key(atoms)].append(atoms)

    screened_by_system: dict[str, list] = {}
    system_stats: dict[str, Any] = {}
    all_selected = []
    for key in sorted(grouped):
        selected, stats = _choose_screened_atoms_for_system(
            grouped[key],
            budget_per_system=int(policy["budget_per_system"]),
            random_seed=int(policy["random_seed"]),
            neighbor_order=int(policy["signature_neighbor_order"]),
            quantiles=list(policy["signature_quantiles"]),
            bucket_width=float(policy["bucket_width"]),
            keep_top_per_bucket=int(policy["keep_top_per_bucket"]),
            system_key=key,
            outer_round=int(outer_round),
            dp_round=int(dp_round),
        )
        screened_by_system[key] = selected
        system_stats[key] = stats
        all_selected.extend(selected)

    canonical_selected_xyz = paths.dp_subworkflow_round_dft_screened_xyz(outer_round, dp_round)
    ensure_dir(canonical_selected_xyz.parent)
    write_candidate_xyz(canonical_selected_xyz, all_selected)

    by_system_dir = ensure_dir(output_dir / "by_system")
    for key, selected in screened_by_system.items():
        write_candidate_xyz(by_system_dir / f"{key}.extxyz", selected)

    pool_path = append_screened_round_into_pool(paths, outer_round, canonical_selected_xyz)
    screening_summary = {
        "outer_round": int(outer_round),
        "dp_round": int(dp_round),
        "policy": policy,
        "current_round_candidates_xyz": str(current_round_xyz.resolve()),
        "cumulative_candidates_xyz": str(candidates_xyz.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selection_mode": str(policy.get("selection_mode", "bucket_random_per_system")),
        "screened_xyz": str(canonical_selected_xyz.resolve()),
        "screened_count": int(len(all_selected)),
        "current_outer_cumulative_screened_count": int(len(all_selected)),
        "raw_dft_candidate_count": int(len(atoms_list)),
        "system_stats": system_stats,
        "current_outer_screened_pool_xyz": str(pool_path.resolve()),
    }
    write_json(paths.dp_subworkflow_round_dft_screening_summary_json(outer_round, dp_round), screening_summary)
    return screening_summary

