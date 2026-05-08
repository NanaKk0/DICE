from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from ase import Atoms

from workflow_core.seed_pool.uncertainty import ensure_minimal_seed_metadata, sort_atoms_by_uncertainty_desc


def stable_random_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) ^ int(base_seed)) % (2**32)


def times_used(atoms: Atoms) -> int:
    return max(0, int(atoms.info.get("times_used", atoms.info.get("seed_use_count", 0)) or 0))


def set_times_used(atoms: Atoms, value: int) -> None:
    safe_value = max(0, int(value))
    atoms.info["times_used"] = safe_value
    atoms.info["seed_use_count"] = safe_value


def sort_atoms_by_usage_then_uncertainty_desc(atoms_list: list[Atoms]) -> list[Atoms]:
    return sorted(
        (ensure_minimal_seed_metadata(item) for item in atoms_list),
        key=lambda atoms: (
            times_used(atoms),
            -float(atoms.info.get("force_std_max", 0.0)),
            str(atoms.info.get("structure_id", "")),
        ),
    )


def resolve_seed_selection_mode(profile_cfg: dict[str, Any], *, dp_round: int) -> str:
    selection_cfg = dict((profile_cfg.get("seed_selection", {}) or {}))
    explicit = str(selection_cfg.get("selection_mode", "") or "").strip().lower()
    if explicit:
        return explicit
    refresh_mode = str((profile_cfg.get("seed_pool", {}) or {}).get("refresh_mode", "") or "").strip().lower()
    if refresh_mode == "merge_old_pool_with_current_candidate_failed" or int(dp_round) == 1:
        return "top_uncertainty_per_system"
    return "usage_then_uncertainty_per_system"


def fill_seed_pool_by_system(
    atoms_list: list[Atoms],
    *,
    selection_mode: str,
    max_pool_size: int,
    random_seed: int,
    outer_round: int,
    dp_round: int,
    system_id: str,
) -> list[Atoms]:
    mode = str(selection_mode).strip().lower()
    if mode == "random_per_system":
        selected = [atoms.copy() for atoms in atoms_list]
        if len(selected) > 1:
            rng = np.random.default_rng(
                stable_random_seed("seed_pool_refresh", outer_round, dp_round, system_id, base_seed=random_seed)
            )
            rng.shuffle(selected)
        return selected[:max_pool_size]
    return sort_atoms_by_uncertainty_desc(atoms_list)[:max_pool_size]
