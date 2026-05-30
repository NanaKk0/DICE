from __future__ import annotations

from typing import Any

from workflow_core.round_context import StudentRoundContext, coerce_student_round_context


def mark_student_round_completed(
    manifest: dict[str, Any],
    status: dict[str, Any],
    context: StudentRoundContext,
    *,
    round_result: dict[str, Any] | None = None,
) -> None:
    round_context = coerce_student_round_context(context)
    student_round = int(round_context.student_round)
    if round_result is not None:
        manifest["round_result"] = round_result
    manifest["completed"] = True
    finished_rounds = status.setdefault("finished_rounds", [])
    if student_round not in finished_rounds:
        finished_rounds.append(student_round)
    status["current_student_round"] = student_round


def mark_student_subworkflow_stopped(
    status: dict[str, Any],
    context: StudentRoundContext,
    *,
    stop_reason: str,
    final_seed_pool_snapshot: dict[str, Any] | None = None,
) -> None:
    round_context = coerce_student_round_context(context)
    status["stopped"] = True
    status["stop_reason"] = stop_reason
    status["current_student_round"] = int(round_context.student_round)
    if final_seed_pool_snapshot is not None:
        status["final_seed_pool_snapshot"] = final_seed_pool_snapshot
