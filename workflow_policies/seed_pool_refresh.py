from __future__ import annotations

from dataclasses import dataclass, field

from ase import Atoms

from workflow_core.round_context import StudentRoundContext
from workflow_core.seed_pool.metadata import (
    deduplicate_atoms_by_structure_id,
    ensure_minimal_seed_metadata,
    set_times_used,
)
from workflow_policies.seed_selection import fill_seed_pool_by_system


@dataclass(slots=True)
class SeedPoolRefreshPolicy:
    keep_if_ge_low: float | None = None
    keep_if_lt_high: float = 0.40
    max_pool_size_per_system: int = 2000
    selection_mode: str = "top_uncertainty_per_system"
    random_seed: int = 20260502
    dedup_enabled: bool = True
    reset_times_used: bool = False


@dataclass(slots=True)
class SeedPoolRefreshResult:
    refreshed_grouped: dict[str, list[Atoms]]
    eligible_count_per_system: dict[str, int] = field(default_factory=dict)
    selected_count_per_system: dict[str, int] = field(default_factory=dict)


def merge_refresh_sources(
    current_grouped: dict[str, list[Atoms]],
    *grouped_sources: dict[str, list[Atoms]],
) -> dict[str, list[Atoms]]:
    merged: dict[str, list[Atoms]] = {
        system_id: [atoms.copy() for atoms in atoms_list]
        for system_id, atoms_list in current_grouped.items()
    }
    for grouped_source in grouped_sources:
        for system_id, atoms_list in grouped_source.items():
            merged.setdefault(system_id, [])
            merged[system_id].extend(atoms.copy() for atoms in atoms_list)
    return merged


def apply_seed_pool_refresh_policy(
    rescored_grouped: dict[str, list[Atoms]],
    *,
    policy: SeedPoolRefreshPolicy,
    context: StudentRoundContext,
) -> SeedPoolRefreshResult:
    refreshed_grouped: dict[str, list[Atoms]] = {}
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for system_id in sorted(rescored_grouped):
        atoms_list = [
            ensure_minimal_seed_metadata(atoms.copy(), system_id=system_id)
            for atoms in rescored_grouped.get(system_id, [])
        ]
        if policy.dedup_enabled:
            atoms_list = deduplicate_atoms_by_structure_id(atoms_list)
        filtered = []
        for atoms in atoms_list:
            value = float(atoms.info.get("force_std_max", 0.0) or 0.0)
            if value >= float(policy.keep_if_lt_high):
                continue
            if policy.keep_if_ge_low is not None and value < float(policy.keep_if_ge_low):
                continue
            filtered.append(atoms)
        eligible_counts[system_id] = len(filtered)
        selected = fill_seed_pool_by_system(
            filtered,
            selection_mode=policy.selection_mode,
            max_pool_size=max(0, int(policy.max_pool_size_per_system)),
            random_seed=int(policy.random_seed),
            teacher_round=context.teacher_round,
            student_round=context.student_round,
            system_id=system_id,
        )
        if policy.reset_times_used:
            for atoms in selected:
                set_times_used(atoms, 0)
                atoms.info["seed_last_used_round"] = -1
        refreshed_grouped[system_id] = selected
        selected_counts[system_id] = len(selected)

    return SeedPoolRefreshResult(
        refreshed_grouped=refreshed_grouped,
        eligible_count_per_system=eligible_counts,
        selected_count_per_system=selected_counts,
    )
