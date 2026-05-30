from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from workflow_core.seed_pool.metadata import ensure_minimal_seed_metadata, group_atoms_by_system


@dataclass(frozen=True, slots=True)
class FirstPrinciplesCandidateSelectionResult:
    selected_atoms: list[Any]
    input_count: int
    selected_count: int
    input_count_per_system: dict[str, int]
    selected_count_per_system: dict[str, int]
    system_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


def select_first_principles_candidates(
    atoms_list: list[Any],
    *,
    cfg: dict[str, Any],
    teacher_round: int,
    student_round: int,
    stage_run_index: int = 0,
    direct_selector=None,
) -> FirstPrinciplesCandidateSelectionResult:
    policy = _normalize_policy(cfg)
    grouped = group_atoms_by_system(atoms_list)
    selected: list[Any] = []
    input_counts = {system_id: len(items) for system_id, items in sorted(grouped.items())}
    selected_counts: dict[str, int] = {}
    system_stats: dict[str, dict[str, Any]] = {}
    for system_id in sorted(grouped):
        items = [ensure_minimal_seed_metadata(item, system_id=system_id) for item in grouped[system_id]]
        chosen, stats = _select_for_system(
            items,
            system_id=system_id,
            policy=policy,
            teacher_round=teacher_round,
            student_round=student_round,
            stage_run_index=stage_run_index,
            direct_selector=direct_selector,
        )
        selected.extend(chosen)
        selected_counts[system_id] = len(chosen)
        system_stats[system_id] = stats
    return FirstPrinciplesCandidateSelectionResult(
        selected_atoms=selected,
        input_count=len(atoms_list),
        selected_count=len(selected),
        input_count_per_system=input_counts,
        selected_count_per_system=selected_counts,
        system_stats=system_stats,
    )


def _normalize_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = dict(cfg or {})
    selection_block = dict(payload.get("selection", {}) or {})
    bucket_block = dict(payload.get("bucket", {}) or {})
    representative_block = dict(payload.get("representative", {}) or {})
    dedup_block = dict(payload.get("dedup", {}) or {})
    return {
        **payload,
        "selection_mode": str(payload.get("selection_mode", "top_uncertainty_per_system")).strip().lower(),
        "budget_per_system": max(0, int(payload.get("budget_per_system", payload.get("max_structures_per_system", 20)))),
        "random_seed": int(payload.get("random_seed", 20260517)),
        "dedup_exact": bool(dedup_block.get("exact", payload.get("dedup_exact", True))),
        "bucket_enabled": bool(bucket_block.get("enabled", payload.get("bucket_enabled", False))),
        "signature_neighbor_order": max(
            1,
            int(
                bucket_block.get(
                    "signature_neighbor_order",
                    payload.get("signature_neighbor_order", 3),
                )
            ),
        ),
        "signature_quantiles": [
            float(item)
            for item in bucket_block.get(
                "signature_quantiles",
                payload.get("signature_quantiles", [0.10, 0.50, 0.90]),
            )
        ],
        "bucket_width": float(bucket_block.get("bucket_width", payload.get("bucket_width", 4.0))),
        "keep_top_per_bucket": max(
            1,
            int(bucket_block.get("keep_top_per_bucket", payload.get("keep_top_per_bucket", 1))),
        ),
        "representative_metric": str(
            representative_block.get("metric", payload.get("representative_metric", "force_std_max"))
        ),
        "direct": dict(payload.get("direct", {}) or {}),
        "selection": selection_block,
    }


def _select_for_system(
    items: list[Any],
    *,
    system_id: str,
    policy: dict[str, Any],
    teacher_round: int,
    student_round: int,
    stage_run_index: int,
    direct_selector,
) -> tuple[list[Any], dict[str, Any]]:
    mode = str(policy["selection_mode"])
    budget = int(policy["budget_per_system"])
    prepared = _dedup_exact_by_hash(items, metric=str(policy["representative_metric"])) if policy["dedup_exact"] else list(items)
    stats: dict[str, Any] = {
        "raw_count": len(items),
        "exact_dedup_count": len(prepared),
        "bucket_count": 0,
        "screened_count": 0,
    }
    if not prepared or budget <= 0:
        return [], stats
    if mode == "direct":
        if direct_selector is None:
            raise ValueError("first-principles candidate direct selection requires a direct_selector")
        chosen = list(direct_selector(prepared, budget_per_system=budget, direct_cfg=dict(policy["direct"])))
    elif mode == "random_per_system":
        chosen = _random_per_system(
            prepared,
            budget=budget,
            seed=_stable_random_seed(
                "first_principles_candidate_selection",
                teacher_round,
                student_round,
                stage_run_index,
                system_id,
                base_seed=int(policy["random_seed"]),
            ),
        )
    elif mode == "bucket_random_per_system":
        chosen, bucket_stats = _bucket_random_per_system(
            prepared,
            budget=budget,
            system_id=system_id,
            teacher_round=teacher_round,
            student_round=student_round,
            random_seed=int(policy["random_seed"]),
            neighbor_order=int(policy["signature_neighbor_order"]),
            quantiles=list(policy["signature_quantiles"]),
            bucket_width=float(policy["bucket_width"]),
            keep_top_per_bucket=int(policy["keep_top_per_bucket"]),
            metric=str(policy["representative_metric"]),
        )
        stats.update(bucket_stats)
    elif mode == "top_uncertainty_per_system":
        chosen = sorted(
            prepared,
            key=lambda item: (
                -_uncertainty(item, metric=str(policy["representative_metric"])),
                str(item.info.get("structure_id", "")),
            ),
        )[:budget]
    else:
        raise ValueError(f"unsupported first-principles candidate selection_mode: {mode!r}")
    stats["screened_count"] = len(chosen)
    uncertainties = [_uncertainty(item, metric=str(policy["representative_metric"])) for item in chosen]
    stats["uncertainty_min"] = float(min(uncertainties)) if uncertainties else None
    stats["uncertainty_median"] = float(np.median(uncertainties)) if uncertainties else None
    stats["uncertainty_max"] = float(max(uncertainties)) if uncertainties else None
    return chosen, stats


def _system_key(atoms) -> str:
    return str(atoms.info.get("formula") or atoms.get_chemical_formula(mode="hill"))


def _uncertainty(atoms, *, metric: str = "force_std_max") -> float:
    for key in (metric, "force_uncertainty", "force_std_max", "decision_metric"):
        value = atoms.info.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _stable_int(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _stable_random_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) ^ int(base_seed)) % (2**32)


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


def _dedup_exact_by_hash(atoms_list: list[Any], *, metric: str) -> list[Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for atoms in atoms_list:
        grouped[_exact_structure_hash(atoms)].append(atoms)
    deduped = []
    for bucket in grouped.values():
        bucket.sort(key=lambda item: _uncertainty(item, metric=metric), reverse=True)
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
    parts.append(float(np.min(finite_distances)) if finite_distances.size else 0.0)
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


def _bucket_random_per_system(
    atoms_list: list[Any],
    *,
    budget: int,
    system_id: str,
    teacher_round: int,
    student_round: int,
    random_seed: int,
    neighbor_order: int,
    quantiles: list[float],
    bucket_width: float,
    keep_top_per_bucket: int,
    metric: str,
) -> tuple[list[Any], dict[str, Any]]:
    if not atoms_list:
        return [], {"bucket_count": 0}
    signatures = np.vstack([
        _build_neighbor_signature(atoms, neighbor_order=neighbor_order, quantiles=quantiles)
        for atoms in atoms_list
    ])
    normalized = _robust_normalize(signatures)
    grouped: dict[tuple[int, ...], list[Any]] = defaultdict(list)
    for atoms, normalized_signature in zip(atoms_list, normalized):
        grouped[_bucket_key(normalized_signature, bucket_width)].append(atoms)
    for bucket in grouped.values():
        bucket.sort(key=lambda item: _uncertainty(item, metric=metric), reverse=True)
    layers: list[list[Any]] = []
    max_bucket_depth = max((min(len(bucket), keep_top_per_bucket) for bucket in grouped.values()), default=0)
    if max_bucket_depth <= 0:
        max_bucket_depth = max((len(bucket) for bucket in grouped.values()), default=0)
    for depth in range(max_bucket_depth):
        layer = []
        for bucket in grouped.values():
            if depth < len(bucket):
                layer.append(bucket[depth])
        layers.append(layer)
    selected: list[Any] = []
    selected_hashes: set[str] = set()
    rng = np.random.default_rng(_stable_int(f"{random_seed}:{teacher_round}:{student_round}:{system_id}"))
    for layer in layers:
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        indices = np.arange(len(layer))
        rng.shuffle(indices)
        for index in indices:
            atoms = layer[int(index)]
            atom_hash = _exact_structure_hash(atoms)
            if atom_hash in selected_hashes:
                continue
            selected.append(atoms)
            selected_hashes.add(atom_hash)
            if len(selected) >= budget:
                break
    if len(selected) < budget:
        leftovers = []
        for bucket in grouped.values():
            for atoms in bucket:
                atom_hash = _exact_structure_hash(atoms)
                if atom_hash not in selected_hashes:
                    leftovers.append(atoms)
        indices = np.arange(len(leftovers))
        rng.shuffle(indices)
        for index in indices:
            selected.append(leftovers[int(index)])
            if len(selected) >= budget:
                break
    return selected, {"bucket_count": int(len(grouped))}


def _random_per_system(items: list[Any], *, budget: int, seed: int) -> list[Any]:
    chosen = list(items)
    if len(chosen) > 1:
        rng = np.random.default_rng(seed)
        rng.shuffle(chosen)
    return chosen[:budget]
