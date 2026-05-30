from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import ensure_dir, write_candidate_xyz
from workflow_policies.first_principles_screening.screening_policy import resolve_dft_screening_policy
from workflow_core.round_context import StudentRoundContext, coerce_student_round_context
from workflow_policies.first_principles_screening.candidate_selection import select_first_principles_candidates


def append_screened_round_into_pool(paths, outer_round: int, screened_xyz: str | Path) -> Path:
    from ase.io import read

    pool_path = _screened_dft_pool_xyz(paths, outer_round)
    ensure_dir(pool_path.parent)
    screened_path = Path(screened_xyz).resolve()
    atoms_list = []
    if screened_path.exists() and screened_path.stat().st_size > 0:
        atoms_list = list(read(str(screened_path), index=":"))
    write_candidate_xyz(pool_path, atoms_list)
    return pool_path.resolve()


def rebuild_screened_dft_pool(paths, outer_round: int, selected_xyz_files: list[str | Path]) -> Path:
    from ase.io import read

    pool_path = _screened_dft_pool_xyz(paths, outer_round)
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


def screen_current_round_dft_candidates(
    paths,
    teacher_round: int | StudentRoundContext,
    student_round: int | dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ase.io import read

    if isinstance(student_round, dict) and policy is None:
        policy = student_round
        student_round = None
    if policy is None:
        raise ValueError("DFT screening policy is required")
    context = coerce_student_round_context(teacher_round, student_round)
    output_dir = ensure_dir(_dft_screening_dir(paths, context.teacher_round, context.student_round))
    current_round_xyz = _current_round_dft_candidates_xyz(paths, context.teacher_round, context.student_round)
    if not current_round_xyz.exists():
        raise FileNotFoundError(f"DFT candidate xyz not found for screening: {current_round_xyz}")

    cumulative_candidates_xyz = _cumulative_dft_candidates_xyz(paths, context.teacher_round)
    candidates_xyz = cumulative_candidates_xyz if cumulative_candidates_xyz.exists() else current_round_xyz
    atoms_list = list(read(str(candidates_xyz), index=":")) if candidates_xyz.exists() and candidates_xyz.stat().st_size > 0 else []

    selection_cfg = {
        **dict(policy or {}),
        "selection_mode": str(policy.get("selection_mode", "bucket_random_per_system")),
        "bucket": {
            "enabled": True,
            "signature_neighbor_order": int(policy["signature_neighbor_order"]),
            "signature_quantiles": list(policy["signature_quantiles"]),
            "bucket_width": float(policy["bucket_width"]),
            "keep_top_per_bucket": int(policy["keep_top_per_bucket"]),
        },
    }
    selection = select_first_principles_candidates(
        atoms_list,
        cfg=selection_cfg,
        teacher_round=int(context.teacher_round),
        student_round=int(context.student_round),
    )

    canonical_selected_xyz = output_dir / "screened_dft_candidates.xyz"
    ensure_dir(canonical_selected_xyz.parent)
    write_candidate_xyz(canonical_selected_xyz, selection.selected_atoms)

    by_system_dir = ensure_dir(output_dir / "by_system")
    grouped_selected = {}
    for atoms in selection.selected_atoms:
        grouped_selected.setdefault(str(atoms.info.get("system_id", atoms.info.get("formula", "unknown"))), []).append(atoms)
    for key, selected in sorted(grouped_selected.items()):
        write_candidate_xyz(by_system_dir / f"{key}.extxyz", selected)

    pool_path = append_screened_round_into_pool(paths, context.teacher_round, canonical_selected_xyz)
    screening_summary = {
        "teacher_round": int(context.teacher_round),
        "student_round": int(context.student_round),
        "policy": policy,
        "current_round_candidates_xyz": str(current_round_xyz.resolve()),
        "cumulative_candidates_xyz": str(candidates_xyz.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selection_mode": str(policy.get("selection_mode", "bucket_random_per_system")),
        "screened_xyz": str(canonical_selected_xyz.resolve()),
        "screened_count": int(selection.selected_count),
        "current_outer_cumulative_screened_count": int(selection.selected_count),
        "raw_dft_candidate_count": int(selection.input_count),
        "system_stats": dict(selection.system_stats),
        "current_outer_screened_pool_xyz": str(pool_path.resolve()),
    }
    return screening_summary


def _dft_screening_dir(paths, teacher_round: int, student_round: int) -> Path:
    return (
        paths.first_principles_substep_artifacts_dir(teacher_round, "screening")
        / paths.student_round_key(student_round)
    )


def _current_round_dft_candidates_xyz(paths, teacher_round: int, student_round: int) -> Path:
    round_dir = paths.student_round_artifacts_dir(teacher_round, student_round)
    candidates = sorted(round_dir.glob("stage_*_relabel/active_learning/dft.xyz"))
    if candidates:
        return candidates[-1].resolve()
    return round_dir / "stage_01_relabel" / "active_learning" / "dft.xyz"


def _cumulative_dft_candidates_xyz(paths, teacher_round: int) -> Path:
    return _screened_dft_pool_xyz(paths, teacher_round)


def _screened_dft_pool_xyz(paths, teacher_round: int) -> Path:
    return paths.first_principles_substep_artifacts_dir(teacher_round, "screening") / "screened_dft_pool.xyz"

