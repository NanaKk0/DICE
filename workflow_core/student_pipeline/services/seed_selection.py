from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from workflow_core.common.io import load_atoms_list, write_candidate_xyz
from workflow_core.seed_pool.metadata import (
    combine_grouped_atoms,
    ensure_minimal_seed_metadata,
    group_atoms_by_system,
    set_times_used,
    times_used,
)
from workflow_policies.seed_selection import fill_seed_pool_by_system

if TYPE_CHECKING:
    from ase import Atoms


_SUPPORTED_SELECTION_MODES = {
    "top_uncertainty_per_system",
    "usage_then_uncertainty_per_system",
    "random_per_system",
}


@dataclass(slots=True)
class SeedSelectionResult:
    selected_grouped: dict[str, list[Atoms]]
    selection_mode: str
    selected_count_per_system: dict[str, int] = field(default_factory=dict)
    selected_structure_ids: list[str] = field(default_factory=list)


def select_sampling_seeds(
    seed_pool_path: str | Path,
    stage_cfg: dict,
    *,
    teacher_round: int,
    student_round: int,
) -> SeedSelectionResult:
    selection_cfg = dict((stage_cfg.get("seed_selection", {}) or {}))
    selection_mode = str(selection_cfg.get("selection_mode", "") or "").strip().lower()
    if not selection_mode:
        raise ValueError("sampling stage seed_selection.selection_mode is required for seed selection")
    if selection_mode not in _SUPPORTED_SELECTION_MODES:
        raise ValueError(
            f"sampling stage uses unsupported selection_mode {selection_mode!r}; "
            f"supported modes: {sorted(_SUPPORTED_SELECTION_MODES)}"
        )

    seed_pool = Path(seed_pool_path).resolve()
    if not seed_pool.exists() or seed_pool.stat().st_size == 0:
        return SeedSelectionResult(selected_grouped={}, selection_mode=selection_mode)

    seed_atoms = [ensure_minimal_seed_metadata(atoms) for atoms in load_atoms_list(seed_pool)]
    _assert_unique_structure_ids(seed_atoms, label="seed_pool")
    grouped_pool = group_atoms_by_system(seed_atoms)
    n_seed_per_system = int(selection_cfg.get("n_seed_per_system", 8) or 0)
    random_seed = int(selection_cfg.get("random_seed", 20260502))

    selected_grouped: dict[str, list[Atoms]] = {}
    selected_structure_ids: list[str] = []
    for system_id, atoms_list in sorted(grouped_pool.items()):
        selected_items = fill_seed_pool_by_system(
            [atoms.copy() for atoms in atoms_list],
            selection_mode=selection_mode,
            max_pool_size=max(0, n_seed_per_system),
            random_seed=random_seed,
            teacher_round=int(teacher_round),
            student_round=int(student_round),
            system_id=system_id,
        )
        selected_grouped[system_id] = [ensure_minimal_seed_metadata(item, system_id=system_id) for item in selected_items]
        selected_structure_ids.extend(str(item.info["structure_id"]) for item in selected_grouped[system_id])

    return SeedSelectionResult(
        selected_grouped=selected_grouped,
        selection_mode=selection_mode,
        selected_count_per_system={system_id: len(items) for system_id, items in selected_grouped.items()},
        selected_structure_ids=selected_structure_ids,
    )


def write_selected_seed_artifact(result: SeedSelectionResult, output_path: str | Path) -> Path:
    return write_candidate_xyz(output_path, combine_grouped_atoms(result.selected_grouped)).resolve()


def stage_seed_usage_update(
    seed_pool_path: str | Path,
    selected_seed_path: str | Path,
    staged_output_path: str | Path,
) -> Path:
    seed_pool = Path(seed_pool_path).resolve()
    selected_seed_file = Path(selected_seed_path).resolve()
    staged_output = Path(staged_output_path).resolve()
    seed_atoms = (
        [ensure_minimal_seed_metadata(atoms) for atoms in load_atoms_list(seed_pool)]
        if seed_pool.exists() and seed_pool.stat().st_size > 0
        else []
    )
    selected_atoms = (
        [ensure_minimal_seed_metadata(atoms) for atoms in load_atoms_list(selected_seed_file)]
        if selected_seed_file.exists() and selected_seed_file.stat().st_size > 0
        else []
    )
    _assert_unique_structure_ids(seed_atoms, label="seed_pool")
    _assert_unique_structure_ids(selected_atoms, label="selected seed artifact")
    seed_by_id = {str(item.info["structure_id"]): item for item in seed_atoms}

    for selected in selected_atoms:
        structure_id = str(selected.info["structure_id"])
        seed_item = seed_by_id.get(structure_id)
        if seed_item is None:
            raise ValueError(f"selected seed structure_id is absent from seed_pool: {structure_id}")
        set_times_used(seed_item, times_used(seed_item) + 1)

    return write_candidate_xyz(staged_output, seed_atoms).resolve()


def _assert_unique_structure_ids(atoms_list: list[Atoms], *, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in atoms_list:
        structure_id = str(ensure_minimal_seed_metadata(item).info["structure_id"])
        if structure_id in seen:
            duplicates.add(structure_id)
        seen.add(structure_id)
    if duplicates:
        raise ValueError(f"{label} contains duplicate structure_id values: {sorted(duplicates)}")
