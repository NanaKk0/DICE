from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from workflow_core.common.io import write_candidate_xyz
from workflow_core.structure_descriptors import build_soap_structure_encoder
from workflow_core.seed_pool.metadata import ensure_minimal_seed_metadata, group_atoms_by_system


@dataclass(frozen=True, slots=True)
class StudentRelabelInputFilteringResult:
    filtered_candidate_path: Path
    filtered_failed_path: Path
    candidate_input_count: int
    failed_input_count: int
    candidate_selected_count: int
    failed_selected_count: int
    selection_mode: str
    direct_selection_mode: str
    candidate_input_count_per_system: dict[str, int]
    failed_input_count_per_system: dict[str, int]
    candidate_selected_count_per_system: dict[str, int]
    failed_selected_count_per_system: dict[str, int]

    def to_stage_payload(self) -> dict[str, Any]:
        return {
            "filtered_candidate_path": str(self.filtered_candidate_path.resolve()),
            "filtered_failed_path": str(self.filtered_failed_path.resolve()),
            "selection_mode": str(self.selection_mode),
            "direct_selection_mode": str(self.direct_selection_mode),
            "candidate_input_count": int(self.candidate_input_count),
            "failed_input_count": int(self.failed_input_count),
            "candidate_selected_count": int(self.candidate_selected_count),
            "failed_selected_count": int(self.failed_selected_count),
            "candidate_input_count_per_system": dict(self.candidate_input_count_per_system),
            "failed_input_count_per_system": dict(self.failed_input_count_per_system),
            "candidate_selected_count_per_system": dict(self.candidate_selected_count_per_system),
            "failed_selected_count_per_system": dict(self.failed_selected_count_per_system),
        }


def filter_student_relabel_inputs(
    *,
    candidate_atoms: list[Any],
    failed_atoms: list[Any],
    output_dir: str | Path,
    cfg: dict[str, Any],
    teacher_round: int,
    student_round: int,
    stage_run_index: int,
) -> StudentRelabelInputFilteringResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selection_mode = str(cfg.get("selection_mode", "random_per_system")).strip().lower()
    random_seed = int(cfg.get("random_seed", 20260517))
    direct_cfg = dict(cfg.get("direct", {}) or {})

    selected_candidate, candidate_input_counts, candidate_selected_counts = _select_by_system(
        candidate_atoms,
        max_structures_per_system=_path_cap(cfg, "candidate"),
        selection_mode=selection_mode,
        random_seed=random_seed,
        direct_cfg=direct_cfg,
        teacher_round=teacher_round,
        student_round=student_round,
        stage_run_index=stage_run_index,
        route="candidate",
    )
    selected_failed, failed_input_counts, failed_selected_counts = _select_by_system(
        failed_atoms,
        max_structures_per_system=_path_cap(cfg, "failed"),
        selection_mode=selection_mode,
        random_seed=random_seed,
        direct_cfg=direct_cfg,
        teacher_round=teacher_round,
        student_round=student_round,
        stage_run_index=stage_run_index,
        route="failed",
    )

    candidate_path = write_candidate_xyz(output_root / "filtered_candidate.xyz", selected_candidate).resolve()
    failed_path = write_candidate_xyz(output_root / "filtered_failed.xyz", selected_failed).resolve()
    return StudentRelabelInputFilteringResult(
        filtered_candidate_path=candidate_path,
        filtered_failed_path=failed_path,
        candidate_input_count=len(candidate_atoms),
        failed_input_count=len(failed_atoms),
        candidate_selected_count=len(selected_candidate),
        failed_selected_count=len(selected_failed),
        selection_mode=selection_mode,
        direct_selection_mode=str(direct_cfg.get("selection_criteria", "")) if selection_mode == "direct" else "",
        candidate_input_count_per_system=candidate_input_counts,
        failed_input_count_per_system=failed_input_counts,
        candidate_selected_count_per_system=candidate_selected_counts,
        failed_selected_count_per_system=failed_selected_counts,
    )


def _path_cap(cfg: dict[str, Any], route: str) -> int:
    route_cfg = dict(cfg.get(route, {}) or {})
    explicit = route_cfg.get("max_structures_per_system")
    if explicit is None:
        explicit = cfg.get(f"{route}_max_structures_per_system", cfg.get("max_structures_per_system", 200))
    return max(0, int(explicit))


def _select_by_system(
    atoms_list: list[Any],
    *,
    max_structures_per_system: int,
    selection_mode: str,
    random_seed: int,
    direct_cfg: dict[str, Any],
    teacher_round: int,
    student_round: int,
    stage_run_index: int,
    route: str,
) -> tuple[list[Any], dict[str, int], dict[str, int]]:
    grouped = group_atoms_by_system(atoms_list)
    selected: list[Any] = []
    input_counts = {system_id: len(items) for system_id, items in sorted(grouped.items())}
    selected_counts: dict[str, int] = {}

    for system_id in sorted(grouped):
        items = [ensure_minimal_seed_metadata(item, system_id=system_id) for item in grouped[system_id]]
        if selection_mode == "direct":
            chosen = _select_direct_by_cluster(
                items,
                max_structures_per_system=max_structures_per_system,
                direct_cfg=direct_cfg,
            )
            selected.extend(chosen)
            selected_counts[system_id] = len(chosen)
            continue
        if selection_mode == "random_per_system":
            seed = _stable_random_seed(
                "student_relabel_input_filtering",
                teacher_round,
                student_round,
                stage_run_index,
                route,
                system_id,
                base_seed=random_seed,
            )
            rng = np.random.default_rng(seed)
            if len(items) > 1:
                rng.shuffle(items)
            chosen = items[:max_structures_per_system]
        else:
            chosen = sorted(
                items,
                key=lambda item: (
                    -float(item.info.get("force_std_max", item.info.get("decision_metric", 0.0)) or 0.0),
                    str(item.info.get("structure_id", "")),
                ),
            )[:max_structures_per_system]
        selected.extend(chosen)
        selected_counts[system_id] = len(chosen)
    return selected, input_counts, selected_counts


def _select_direct_by_cluster(
    items: list[Any],
    *,
    max_structures_per_system: int,
    direct_cfg: dict[str, Any],
) -> list[Any]:
    if not items or max_structures_per_system <= 0:
        return []

    soap_cfg = dict(direct_cfg.get("soap", {}) or {})
    direct_cfg = dict(direct_cfg.get("direct", {}) or {})
    if not soap_cfg or not direct_cfg:
        raise ValueError("student_relabel_input_filtering direct mode requires direct.soap and direct.direct config blocks")
    from workflow_core.student_pipeline.services.selection_adapters.direct_structure_selection import (
        select_atoms_by_system_with_direct,
    )

    result = select_atoms_by_system_with_direct(
        items,
        budget_per_system=max_structures_per_system,
        structure_encoder=build_soap_structure_encoder(_with_default_soap_species(soap_cfg)),
        direct_cfg=direct_cfg,
    )
    return result.selected_atoms


def _with_default_soap_species(soap_cfg: dict[str, Any]) -> dict[str, Any]:
    if "soap_species" in soap_cfg or "species" in soap_cfg:
        return soap_cfg
    merged = dict(soap_cfg)
    merged["soap_species"] = ("Pt", "O", "H")
    return merged


def _stable_random_seed(*parts: object, base_seed: int) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return (int(digest, 16) ^ int(base_seed)) % (2**32)
