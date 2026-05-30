from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_core.common.io import write_candidate_xyz
from workflow_core.structure_descriptors import build_soap_structure_encoder
from workflow_policies.first_principles_screening.candidate_selection import select_first_principles_candidates


@dataclass(frozen=True, slots=True)
class TeacherDftCandidateFilteringResult:
    screened_dft_candidate_path: Path
    input_count: int
    screened_count: int
    input_count_per_system: dict[str, int]
    selected_count_per_system: dict[str, int]

    def to_stage_payload(self) -> dict[str, Any]:
        return {
            "screened_dft_candidate_path": str(self.screened_dft_candidate_path.resolve()),
            "input_count": int(self.input_count),
            "screened_count": int(self.screened_count),
            "input_count_per_system": dict(self.input_count_per_system),
            "selected_count_per_system": dict(self.selected_count_per_system),
        }


def filter_teacher_dft_candidates(
    *,
    atoms_list: list[Any],
    output_dir: str | Path,
    cfg: dict[str, Any],
    teacher_round: int,
    student_round: int,
    stage_run_index: int,
) -> TeacherDftCandidateFilteringResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selection = select_first_principles_candidates(
        atoms_list,
        cfg=cfg,
        teacher_round=teacher_round,
        student_round=student_round,
        stage_run_index=stage_run_index,
        direct_selector=_select_direct_by_cluster,
    )
    screened_path = write_candidate_xyz(output_root / "screened_dft_candidates.xyz", selection.selected_atoms).resolve()
    return TeacherDftCandidateFilteringResult(
        screened_dft_candidate_path=screened_path,
        input_count=selection.input_count,
        screened_count=selection.selected_count,
        input_count_per_system=selection.input_count_per_system,
        selected_count_per_system=selection.selected_count_per_system,
    )


def _select_direct_by_cluster(
    items: list[Any],
    *,
    budget_per_system: int,
    direct_cfg: dict[str, Any],
) -> list[Any]:
    if not items or budget_per_system <= 0:
        return []
    soap_cfg = dict(direct_cfg.get("soap", {}) or {})
    direct_cfg = dict(direct_cfg.get("direct", {}) or {})
    if not soap_cfg or not direct_cfg:
        raise ValueError("teacher_dft_candidate_filtering direct mode requires direct.soap and direct.direct config blocks")
    from workflow_core.student_pipeline.services.selection_adapters.direct_structure_selection import (
        select_atoms_by_system_with_direct,
    )

    result = select_atoms_by_system_with_direct(
        items,
        budget_per_system=budget_per_system,
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
