from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ase import Atoms

from workflow_core.round_context import StudentRoundContext
from workflow_core.seed_pool.metadata import (
    ensure_minimal_seed_metadata,
    set_times_used,
    sort_atoms_by_uncertainty_desc,
    sort_atoms_by_usage_then_uncertainty_desc,
    times_used,
)


@dataclass(slots=True)
class SeedSelectionResult:
    selected_grouped: dict[str, list[Atoms]]
    selection_mode: str
    selected_count_per_system: dict[str, int] = field(default_factory=dict)


def stable_random_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) ^ int(base_seed)) % (2**32)


def resolve_seed_selection_mode(profile_cfg: dict[str, Any], *, student_round: int) -> str:
    selection_cfg = dict((profile_cfg.get("seed_selection", {}) or {}))
    explicit = str(selection_cfg.get("selection_mode", "") or "").strip().lower()
    if explicit:
        return explicit
    refresh_mode = str((profile_cfg.get("seed_pool", {}) or {}).get("refresh_mode", "") or "").strip().lower()
    if refresh_mode == "merge_old_pool_with_current_candidate_failed" or int(student_round) == 1:
        return "top_uncertainty_per_system"
    return "usage_then_uncertainty_per_system"


def fill_seed_pool_by_system(
    atoms_list: list[Atoms],
    *,
    selection_mode: str,
    max_pool_size: int,
    random_seed: int,
    teacher_round: int,
    student_round: int,
    system_id: str,
) -> list[Atoms]:
    mode = str(selection_mode).strip().lower()
    if mode == "random_per_system":
        selected = [atoms.copy() for atoms in atoms_list]
        if len(selected) > 1:
            rng = np.random.default_rng(
                stable_random_seed("seed_pool_refresh", teacher_round, student_round, system_id, base_seed=random_seed)
            )
            rng.shuffle(selected)
        return selected[:max_pool_size]
    if mode == "usage_then_uncertainty_per_system":
        return sort_atoms_by_usage_then_uncertainty_desc(atoms_list)[:max_pool_size]
    return sort_atoms_by_uncertainty_desc(atoms_list)[:max_pool_size]


def select_seed_atoms(
    grouped_pool: dict[str, list[Atoms]],
    *,
    profile_cfg: dict[str, Any],
    context: StudentRoundContext,
) -> SeedSelectionResult:
    selection_cfg = dict((profile_cfg.get("seed_selection", {}) or {}))
    n_seed_per_system = int(selection_cfg.get("n_seed_per_system", 8) or 0)
    selection_mode = resolve_seed_selection_mode(profile_cfg, student_round=context.student_round)
    selected_grouped: dict[str, list[Atoms]] = {}

    for system_id, atoms_list in sorted(grouped_pool.items()):
        selected_grouped[system_id] = fill_seed_pool_by_system(
            [atoms.copy() for atoms in atoms_list],
            selection_mode=selection_mode,
            max_pool_size=max(0, n_seed_per_system),
            random_seed=int(selection_cfg.get("random_seed", 20260502)),
            teacher_round=context.teacher_round,
            student_round=context.student_round,
            system_id=system_id,
        )

    return SeedSelectionResult(
        selected_grouped=selected_grouped,
        selection_mode=selection_mode,
        selected_count_per_system={system_id: len(items) for system_id, items in selected_grouped.items()},
    )
