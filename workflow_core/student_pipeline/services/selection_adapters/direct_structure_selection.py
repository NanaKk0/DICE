from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ase import Atoms

from workflow_core.seed_pool.metadata import ensure_minimal_seed_metadata
from workflow_policies.direct import direct_select


@dataclass(frozen=True, slots=True)
class DirectStructureSelectionConfig:
    samples_per_cluster: int = 1
    selection_criteria: str = "center"
    use_explained_variance_weighting: bool = False
    birch_threshold: float = 0.5
    birch_branching_factor: int = 50
    birch_n_clusters: int | None = None
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class DirectStructureSelectionResult:
    selected_atoms: list[Atoms]
    selected_indices: list[int]
    input_count_per_system: dict[str, int]
    selected_count_per_system: dict[str, int]


def normalize_direct_structure_selection_config(payload: dict[str, Any] | DirectStructureSelectionConfig | None) -> DirectStructureSelectionConfig:
    if payload is None:
        return DirectStructureSelectionConfig()
    if isinstance(payload, DirectStructureSelectionConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"direct structure selection config must be a dict or DirectStructureSelectionConfig, got {type(payload)!r}")
    return DirectStructureSelectionConfig(
        samples_per_cluster=max(1, int(payload.get("samples_per_cluster", 1))),
        selection_criteria=str(payload.get("selection_criteria", "center")).strip().lower(),
        use_explained_variance_weighting=bool(payload.get("use_explained_variance_weighting", False)),
        birch_threshold=float(payload.get("birch_threshold", 0.5)),
        birch_branching_factor=int(payload.get("birch_branching_factor", 50)),
        birch_n_clusters=None if payload.get("birch_n_clusters") in {None, ""} else int(payload.get("birch_n_clusters")),
        random_state=int(payload.get("random_state", 42)),
    )


def select_atoms_by_system_with_direct(
    atoms_list: Sequence[Atoms],
    *,
    budget_per_system: int,
    structure_encoder: Callable[[Sequence[Atoms]], Any],
    direct_cfg: dict[str, Any] | DirectStructureSelectionConfig | None,
) -> DirectStructureSelectionResult:
    cfg = normalize_direct_structure_selection_config(direct_cfg)
    grouped: dict[str, list[tuple[int, Atoms]]] = defaultdict(list)
    for index, atoms in enumerate(atoms_list):
        item = ensure_minimal_seed_metadata(atoms)
        grouped[str(item.info["system_id"])].append((int(index), item))
    selected_atoms: list[Atoms] = []
    selected_indices: list[int] = []
    input_count_per_system: dict[str, int] = {}
    selected_count_per_system: dict[str, int] = {}

    for system_id in sorted(grouped):
        indexed_items = grouped[system_id]
        items = [ensure_minimal_seed_metadata(item, system_id=system_id) for _, item in indexed_items]
        input_count_per_system[system_id] = len(items)
        if not items or budget_per_system <= 0:
            selected_count_per_system[system_id] = 0
            continue
        if len(items) <= budget_per_system:
            selected_atoms.extend(items)
            selected_indices.extend(index for index, _ in indexed_items)
            selected_count_per_system[system_id] = len(items)
            continue
        X = structure_encoder(items)
        result = direct_select(
            X,
            samples_per_cluster=int(budget_per_system),
            selection_criteria=cfg.selection_criteria,
            use_explained_variance_weighting=cfg.use_explained_variance_weighting,
            birch_threshold=cfg.birch_threshold,
            birch_branching_factor=cfg.birch_branching_factor,
            birch_n_clusters=cfg.birch_n_clusters,
            random_state=cfg.random_state,
        )
        chosen_local = sorted(result.selected_indices)
        if len(chosen_local) > budget_per_system:
            chosen_local = chosen_local[:budget_per_system]
        selected_atoms.extend(items[index] for index in chosen_local)
        selected_indices.extend(indexed_items[index][0] for index in chosen_local)
        selected_count_per_system[system_id] = len(chosen_local)

    ordered = sorted(zip(selected_indices, selected_atoms), key=lambda item: item[0])
    if ordered:
        selected_indices = [index for index, _ in ordered]
        selected_atoms = [atoms for _, atoms in ordered]
    return DirectStructureSelectionResult(
        selected_atoms=selected_atoms,
        selected_indices=selected_indices,
        input_count_per_system=input_count_per_system,
        selected_count_per_system=selected_count_per_system,
    )
