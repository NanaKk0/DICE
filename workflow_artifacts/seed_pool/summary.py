from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import write_json
from workflow_core.round_context import StudentRoundContext


def round_context_payload(context: StudentRoundContext) -> dict[str, int]:
    return {
        "teacher_round": int(context.teacher_round),
        "student_round": int(context.student_round),
    }


def write_candidate_archive_summary(
    summary_path: str | Path,
    *,
    context: StudentRoundContext,
    candidate_count_raw: int,
    candidate_count_physical: int,
    archive_candidate_file: str | Path,
    candidate_by_system_dir: str | Path,
) -> str:
    return write_json(
        summary_path,
        {
            **round_context_payload(context),
            "candidate_count_raw": int(candidate_count_raw),
            "candidate_count_physical": int(candidate_count_physical),
            "archive_candidate_file": str(Path(archive_candidate_file).resolve()),
            "candidate_by_system_dir": str(Path(candidate_by_system_dir).resolve()),
        },
    )


def build_seed_pool_update_summary(
    *,
    context: StudentRoundContext,
    seed_pool_size_before: int,
    candidate_count_raw: int,
    candidate_count_physical: int,
    merged_count: int,
    seed_pool_size_after: int,
    max_pool_size_per_system: int,
    candidate_file: str | Path,
    seed_pool_file: str | Path,
) -> dict[str, Any]:
    return {
        **round_context_payload(context),
        "seed_pool_size_before": int(seed_pool_size_before),
        "candidate_count_raw": int(candidate_count_raw),
        "candidate_count_physical": int(candidate_count_physical),
        "merged_count": int(merged_count),
        "seed_pool_size_after": int(seed_pool_size_after),
        "max_pool_size_per_system": int(max_pool_size_per_system),
        "compression_applied": False,
        "compression_method": "uncertainty_only_existing_scores",
        "candidate_file": str(Path(candidate_file).resolve()),
        "seed_pool_file": str(Path(seed_pool_file).resolve()),
    }


def build_seed_pool_refresh_summary(
    *,
    context: StudentRoundContext,
    refresh_mode: str,
    committee_summary_path: str | Path,
    keep_if_ge_low: float | None,
    keep_if_lt_high: float,
    max_pool_size_per_system: int,
    pool_fill_selection_mode: str,
    pool_random_seed: int,
    pool_size_after: int,
    eligible_count_per_system: dict[str, int],
    selected_count_per_system: dict[str, int],
    system_counts_after: dict[str, int],
    seed_pool_file: str | Path,
    seed_pool_systems_dir: str | Path,
    reset_times_used: bool,
) -> dict[str, Any]:
    return {
        **round_context_payload(context),
        "refresh_mode": str(refresh_mode),
        "committee_summary_path": str(Path(committee_summary_path).resolve()),
        "keep_if_ge_low": None if keep_if_ge_low is None else float(keep_if_ge_low),
        "keep_if_lt_high": float(keep_if_lt_high),
        "max_pool_size_per_system": int(max_pool_size_per_system),
        "pool_fill_selection_mode": str(pool_fill_selection_mode),
        "pool_random_seed": int(pool_random_seed),
        "pool_size_after": int(pool_size_after),
        "eligible_count_per_system": {key: int(eligible_count_per_system[key]) for key in sorted(eligible_count_per_system)},
        "selected_count_per_system": {key: int(selected_count_per_system[key]) for key in sorted(selected_count_per_system)},
        "system_counts_after": {key: int(system_counts_after[key]) for key in sorted(system_counts_after)},
        "seed_pool_file": str(Path(seed_pool_file).resolve()),
        "seed_pool_systems_dir": str(Path(seed_pool_systems_dir).resolve()),
        "reset_times_used": bool(reset_times_used),
    }
