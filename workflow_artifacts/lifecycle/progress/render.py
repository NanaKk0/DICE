from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
import re

from workflow_runtime.lifecycle.progress.types import (
    AttemptStarted,
    BasicBlockSummary,
    BlockFailed,
    BlockInterrupted,
    BlockStarted,
    BlockSucceeded,
    CountChainSnapshot,
    CountFlowSnapshot,
    DiagnosticArchived,
    DiagnosticPruned,
    GenericWorkSnapshot,
    DatasetSnapshot,
    DescriptorSnapshot,
    FPBlockSummary,
    HeartbeatReported,
    LabelingSnapshot,
    PhaseEntered,
    ProgressEnvelope,
    SnapshotReported,
    ScreenBlockSummary,
    ScoreObservationSnapshot,
    ScorecardSnapshot,
    RoutingSnapshot,
    SamplingBlockSummary,
    SamplingSnapshot,
    TrainingSnapshot,
    TrainBlockSummary,
)


def render_history_event(event: ProgressEnvelope) -> str | None:
    payload = event.payload
    prefix = f"{_utc_timestamp(event.received_at)} "
    if isinstance(payload, AttemptStarted):
        origin = f"resume:{payload.resume_target}" if payload.resume_target else "reconciled"
        return (
            f"{prefix}iter{event.iter_index} [STARTED] attempt={event.attempt_id} "
            f"origin={origin} format=2\n"
        )
    if isinstance(payload, BlockStarted):
        return f"{prefix}{event.block} [RUNNING]\n"
    if isinstance(payload, PhaseEntered):
        operation = (
            f" operation={_compact_token(payload.operation_id)}"
            if payload.operation_id
            else ""
        )
        return (
            f"{prefix}{event.block} [RUNNING] "
            f"phase={_compact_text(payload.phase)}{operation}\n"
        )
    if isinstance(payload, BlockSucceeded):
        detail = _render_history_summary(payload.summary)
        return f"{prefix}{event.block} [SUCCESS]{(' ' + detail) if detail else ''}\n"
    if isinstance(payload, BlockFailed):
        detail = f" message={_failure_message(payload.message)}" if payload.message else ""
        diagnostic_ref = _relative_diagnostic_ref(payload.diagnostic_ref)
        see = f" see: {diagnostic_ref}" if diagnostic_ref else ""
        return f"{prefix}{event.block} [FAILED]{detail}{see}\n"
    if isinstance(payload, BlockInterrupted):
        detail = f" message={_failure_message(payload.message)}" if payload.message else ""
        diagnostic_ref = _relative_diagnostic_ref(payload.diagnostic_ref)
        see = f" see: {diagnostic_ref}" if diagnostic_ref else ""
        return f"{prefix}{event.block} [INTERRUPTED]{detail}{see}\n"
    if isinstance(payload, DiagnosticArchived):
        return (
            f"{prefix}{event.block} [DIAGNOSTIC_ARCHIVED] "
            f"attempt={_compact_text(payload.archived_attempt_id)} "
            f"see: {_compact_text(payload.diagnostic_ref)}\n"
        )
    if isinstance(payload, DiagnosticPruned):
        return (
            f"{prefix}{event.block} [DIAGNOSTIC_PRUNED] "
            f"attempt={_compact_text(payload.pruned_attempt_id)}\n"
        )
    if isinstance(payload, SnapshotReported):
        detail = _render_history_snapshot(payload.snapshot)
        if detail is None:
            return None
        return f"{prefix}{event.block} [PROGRESS] {detail}\n"
    if isinstance(payload, HeartbeatReported):
        return None
    return None


def _render_snapshot(snapshot: object) -> list[str]:
    if isinstance(snapshot, GenericWorkSnapshot):
        position = ""
        if snapshot.completed is not None:
            position = str(snapshot.completed)
            if snapshot.total is not None:
                position += f"/{snapshot.total}"
            if snapshot.unit:
                position = f"{snapshot.unit} {position}"
        detail = f" · {_bounded_compact_text(snapshot.detail)}" if snapshot.detail else ""
        return [f"  {_bounded_label(snapshot.activity)}{(' ' + position) if position else ''}{detail}"]
    if isinstance(snapshot, DatasetSnapshot):
        parts = [f"{_bounded_label(snapshot.label)} {snapshot.unique} unique"]
        parts.extend(f"{item.label} {item.count}" for item in snapshot.contributions)
        if snapshot.deduplicated:
            parts.append(f"dedup {snapshot.deduplicated}")
        if snapshot.train is not None:
            parts.append(f"train {snapshot.train}")
        if snapshot.valid is not None:
            parts.append(f"valid {snapshot.valid}")
        if snapshot.systems is not None:
            parts.append(f"{snapshot.systems} systems")
        return ["  " + " · ".join(parts)]
    if isinstance(snapshot, CountChainSnapshot):
        values = " → ".join(
            f"{step.label} {_count(step.value)}" for step in snapshot.steps
        )
        return [f"  {snapshot.label} {values}"]
    if isinstance(snapshot, CountFlowSnapshot):
        system_ids = tuple(_bounded_label(row.system_id) for row in snapshot.rows)
        widths = [
            max(len(column), *(len(_count(row.values[index])) for row in snapshot.rows))
            for index, column in enumerate(snapshot.columns)
        ]
        system_width = max(_display_width("system"), *(_display_width(value) for value in system_ids))
        lines = [f"  {snapshot.title}"]
        lines.append(
            "  " + "system".ljust(system_width) + "  " + "  ".join(
                column.rjust(width) for column, width in zip(snapshot.columns, widths)
            )
        )
        lines.extend(
            "  " + _ljust_display(system_id, system_width) + "  " + "  ".join(
                _count(value).rjust(width) for value, width in zip(row.values, widths)
            )
            for row, system_id in zip(snapshot.rows, system_ids)
        )
        return lines
    if isinstance(snapshot, TrainingSnapshot):
        parts = [
            f"{snapshot.member_id} {snapshot.member_index}/{snapshot.total_members}",
            f"{snapshot.unit} {snapshot.position}/{snapshot.total}",
            snapshot.stage,
        ]
        if snapshot.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(snapshot.elapsed_seconds)}")
        if snapshot.eta_seconds is not None:
            parts.append(f"ETA {_duration(snapshot.eta_seconds)}")
        if snapshot.learning_rate is not None:
            parts.append(f"lr {_metric_value(snapshot.learning_rate)}")
        lines = ["  " + " · ".join(parts)]
        lines.extend(
            "  " + trend.label + " "
            + "→".join(_metric_value(value) for value in trend.values)
            + f" {trend.unit}"
            for trend in snapshot.trends
        )
        return lines
    if isinstance(snapshot, LabelingSnapshot):
        line = f"  labeling {snapshot.completed}/{snapshot.total}"
        if snapshot.trusted_below is not None:
            trusted = next((row.trusted for row in snapshot.rows if row.system_id == "ALL"), None)
            line += f" · trusted {_count(trusted)}"
        if not snapshot.rows:
            return [line]
        rows, omitted = _bounded_system_rows(snapshot.rows)
        system_ids = tuple(_bounded_label(row.system_id) for row in rows)
        system_width = max(_display_width("system"), *(_display_width(value) for value in system_ids))
        value_width = max(
            len("labeled"),
            *(len(_progress_count(row.completed, row.total)) for row in rows),
        )
        lines = [line, "  " + _ljust_display("system", system_width) + "  " + "labeled".rjust(value_width)]
        lines.extend(
            "  " + _ljust_display(system_id, system_width) + "  "
            + _progress_count(row.completed, row.total).rjust(value_width)
            for row, system_id in zip(rows, system_ids)
        )
        if omitted:
            lines.append(f"  … {omitted} systems omitted")
        return lines
    if isinstance(snapshot, ScoreObservationSnapshot):
        return [f"  {snapshot.role} score {snapshot.completed}/{snapshot.total}"]
    if isinstance(snapshot, ScorecardSnapshot):
        rows, omitted = _bounded_system_rows(snapshot.rows)
        system_ids = tuple(_bounded_label(row.system_id) for row in rows)
        rendered_rows = [
            (*(_count(value) for value in row.values), "—" if row.p_mean is None else _metric_value(row.p_mean))
            for row in rows
        ]
        columns = (*snapshot.columns, "p avg")
        widths = [
            max(len(column), *(len(row[index]) for row in rendered_rows))
            for index, column in enumerate(columns)
        ]
        system_width = max(
            _display_width("system"), *(_display_width(value) for value in system_ids)
        )
        lines = [f"  {snapshot.role} score"]
        lines.append(
            "  " + _ljust_display("system", system_width) + "  "
            + "  ".join(column.rjust(width) for column, width in zip(columns, widths))
        )
        lines.extend(
            "  " + _ljust_display(system_id, system_width) + "  "
            + "  ".join(value.rjust(width) for value, width in zip(values, widths))
            for row, system_id, values in zip(rows, system_ids, rendered_rows)
        )
        if omitted:
            lines.append(f"  … {omitted} systems omitted")
        return lines
    if isinstance(snapshot, RoutingSnapshot):
        rows, omitted = _bounded_system_rows(snapshot.rows)
        system_ids = tuple(_bounded_label(row.system_id) for row in rows)
        teacher_values = [" → ".join(str(value) for value in row.teacher) for row in rows]
        student_values = [" → ".join(str(value) for value in row.student) for row in rows]
        system_width = max(_display_width("system"), *(_display_width(value) for value in system_ids))
        teacher_width = max(len("teacher"), *(len(value) for value in teacher_values))
        student_width = max(len("student"), *(len(value) for value in student_values))
        lines = ["  routing"]
        lines.append(
            "  " + _ljust_display("system", system_width) + "  "
            + "teacher".rjust(teacher_width) + "  " + "student".rjust(student_width)
        )
        lines.extend(
            "  " + _ljust_display(system_id, system_width) + "  "
            + teacher.rjust(teacher_width) + "  " + student.rjust(student_width)
            for system_id, teacher, student in zip(system_ids, teacher_values, student_values)
        )
        if omitted:
            lines.append(f"  … {omitted} systems omitted")
        return lines
    if isinstance(snapshot, SamplingSnapshot):
        parts = [
            snapshot.backend,
            snapshot.model_shape,
            f"{snapshot.completed_tasks}/{snapshot.total_tasks} tasks",
        ]
        if snapshot.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(snapshot.elapsed_seconds)}")
        parts.append(
            "ETA calculating" if snapshot.eta_seconds is None else f"ETA {_duration(snapshot.eta_seconds)}"
        )
        lines = ["  " + " · ".join(parts)]
        matrix = (
            f"matrix {snapshot.systems} systems × {snapshot.temperatures} temperatures · "
            f"{snapshot.ensemble.upper()}"
        )
        if snapshot.pressure_bar is not None:
            matrix += f" · {snapshot.pressure_bar:g} bar"
        if snapshot.task_step_totals:
            low = min(snapshot.task_step_totals)
            high = max(snapshot.task_step_totals)
            step_text = f"{low} steps/task" if low == high else f"{low}–{high} steps/task"
            matrix += f" · {step_text} · {sum(snapshot.task_step_totals)} total steps"
        else:
            matrix += f" · {snapshot.steps_per_task} steps"
        if snapshot.plumed:
            matrix += " · PLUMED"
        if snapshot.mps_phase:
            matrix += f" · MPS {snapshot.mps_phase}"
        lines.append("  " + matrix)
        details = snapshot.task_details[:4]
        for detail in details:
            temperature = "—" if detail.temperature_k is None else f"{detail.temperature_k:g} K"
            step = "—" if detail.step is None else str(detail.step)
            task_total = snapshot.steps_per_task
            if snapshot.task_step_totals and detail.task_id.startswith("task_"):
                try:
                    task_total = snapshot.task_step_totals[int(detail.task_id[5:])]
                except (IndexError, ValueError):
                    task_total = snapshot.steps_per_task
            line = (
                f"  {detail.state} {_bounded_label(detail.task_id)} · "
                f"{_bounded_label(detail.system_id)} · {temperature} · "
                f"step {step}/{task_total}"
            )
            if detail.speed is not None and detail.speed_unit:
                line += f" · {detail.speed:g} {detail.speed_unit}"
            if detail.diagnostic_locator:
                diagnostic_ref = _relative_diagnostic_ref(detail.diagnostic_locator)
                if diagnostic_ref:
                    line += f" · see {_bounded_compact_text(diagnostic_ref, limit=96)}"
            lines.append(line)
        omitted_details = snapshot.omitted_task_details + max(0, len(snapshot.task_details) - 4)
        if omitted_details:
            lines.append(f"  … {omitted_details} active/warning/failed tasks omitted")
        if not details and snapshot.active_task_id is not None:
            temperature = "—" if snapshot.active_temperature_k is None else f"{snapshot.active_temperature_k:g} K"
            step = "—" if snapshot.active_step is None else str(snapshot.active_step)
            lines.append(
                f"  active {_bounded_label(snapshot.active_task_id)} · "
                f"{_bounded_label(snapshot.active_system_id) if snapshot.active_system_id else '—'} · "
                f"{temperature} · step {step}/{snapshot.steps_per_task}"
            )
        if snapshot.speed is not None and snapshot.speed_unit:
            lines.append(f"  speed {snapshot.speed:g} {snapshot.speed_unit}")
        rows, omitted = _bounded_system_rows(snapshot.rows)
        system_ids = tuple(_bounded_label(row.system_id) for row in rows)
        system_width = max(_display_width("system"), *(_display_width(value) for value in system_ids))
        progress_values = [f"{row.completed}/{row.total}" for row in rows]
        progress_width = max(len("tasks"), *(len(value) for value in progress_values))
        lines.append("  " + _ljust_display("system", system_width) + "  " + "tasks".rjust(progress_width))
        lines.extend(
            "  " + _ljust_display(system_id, system_width) + "  " + value.rjust(progress_width)
            for system_id, value in zip(system_ids, progress_values)
        )
        if omitted:
            lines.append(f"  … {omitted} systems omitted")
        for group_id, tasks, total_steps in snapshot.condition_group_totals:
            lines.append(
                f"  condition {_bounded_label(group_id)} · {tasks} tasks · {total_steps} steps"
            )
        return lines
    if isinstance(snapshot, DescriptorSnapshot):
        width = "initializing" if snapshot.width is None else f"{snapshot.width} dims"
        line = f"  {snapshot.label} {snapshot.completed}/{snapshot.total} · descriptor {width}"
        if snapshot.elements_total is not None:
            line += f" · elements {_count(snapshot.elements_completed)}/{snapshot.elements_total}"
        return [line]
    raise TypeError(f"unsupported progress snapshot {type(snapshot)!r}")


def _render_history_snapshot(snapshot: object) -> str | None:
    if isinstance(snapshot, GenericWorkSnapshot):
        parts = [f"activity={_compact_token(snapshot.activity)}"]
        if snapshot.completed is not None:
            if snapshot.total is None:
                parts.append(f"completed={snapshot.completed}")
            else:
                key = _compact_token(snapshot.unit or "completed")
                parts.extend(
                    (
                        f"{key}={snapshot.completed}/{snapshot.total}",
                        f"pct={_percentage(snapshot.completed, snapshot.total)}",
                    )
                )
        if snapshot.detail:
            parts.append(f"detail={_bounded_compact_text(snapshot.detail)}")
        return " ".join(parts)
    if isinstance(snapshot, DatasetSnapshot):
        parts = [
            f"dataset={_compact_token(snapshot.label)}",
            f"unique={snapshot.unique}",
        ]
        if snapshot.train is not None:
            parts.append(f"train={snapshot.train}")
        if snapshot.valid is not None:
            parts.append(f"valid={snapshot.valid}")
        if snapshot.systems is not None:
            parts.append(f"systems={snapshot.systems}")
        if snapshot.deduplicated:
            parts.append(f"dedup={snapshot.deduplicated}")
        return " ".join(parts)
    if isinstance(snapshot, CountChainSnapshot):
        steps = snapshot.steps[:8]
        parts = [f"flow={_compact_token(snapshot.label)}"]
        parts.extend(f"{_compact_token(step.label)}={_count(step.value)}" for step in steps)
        if len(snapshot.steps) > len(steps):
            parts.append(f"omitted={len(snapshot.steps) - len(steps)}")
        return " ".join(parts)
    if isinstance(snapshot, CountFlowSnapshot):
        aggregate = next(
            (row for row in snapshot.rows if row.system_id == "ALL"), snapshot.rows[0]
        )
        columns = tuple(zip(snapshot.columns, aggregate.values))[:8]
        parts = [f"flow={_compact_token(snapshot.title)}"]
        parts.extend(f"{_compact_token(label)}={_count(value)}" for label, value in columns)
        if len(snapshot.columns) > len(columns):
            parts.append(f"omitted={len(snapshot.columns) - len(columns)}")
        return " ".join(parts)
    if isinstance(snapshot, TrainingSnapshot):
        parts = [
            f"member={_compact_token(snapshot.member_id)}",
            f"member_index={snapshot.member_index}/{snapshot.total_members}",
            f"stage={_compact_token(snapshot.stage)}",
            f"{_compact_token(snapshot.unit)}={snapshot.position}/{snapshot.total}",
            f"pct={_percentage(snapshot.position, snapshot.total)}",
        ]
        if snapshot.series_generation:
            parts.append(f"series={snapshot.series_generation}")
        if snapshot.eta_seconds is not None:
            parts.append(f"ETA={_duration(snapshot.eta_seconds)}")
        if snapshot.trends:
            trend = snapshot.trends[0]
            parts.append(
                f"{_compact_token(trend.label)}={_metric_value(trend.values[-1])}_"
                f"{_compact_token(trend.unit)}"
            )
        return " ".join(parts)
    if isinstance(snapshot, LabelingSnapshot):
        parts = [
            f"labeling={snapshot.completed}/{snapshot.total}",
            f"pct={_percentage(snapshot.completed, snapshot.total)}",
        ]
        aggregate = next((row for row in snapshot.rows if row.system_id == "ALL"), None)
        if aggregate is not None and aggregate.trusted is not None:
            parts.append(f"trusted={aggregate.trusted}")
        return " ".join(parts)
    if isinstance(snapshot, ScoreObservationSnapshot):
        parts = [
            f"role={_compact_token(snapshot.role)}",
            f"scored={snapshot.completed}/{snapshot.total}",
            f"pct={_percentage(snapshot.completed, snapshot.total)}",
        ]
        aggregate = next((row for row in snapshot.rows if row.system_id == "ALL"), None)
        if aggregate is not None and aggregate.p_mean is not None:
            parts.append(f"p_mean={_metric_value(aggregate.p_mean)}")
        return " ".join(parts)
    if isinstance(snapshot, ScorecardSnapshot):
        aggregate = next(
            (row for row in snapshot.rows if row.system_id == "ALL"), snapshot.rows[0]
        )
        columns = tuple(zip(snapshot.columns, aggregate.values))[:8]
        parts = [f"role={_compact_token(snapshot.role)}"]
        parts.extend(f"{_compact_token(label)}={_count(value)}" for label, value in columns)
        if aggregate.p_mean is not None:
            parts.append(f"p_mean={_metric_value(aggregate.p_mean)}")
        if len(snapshot.columns) > len(columns):
            parts.append(f"omitted={len(snapshot.columns) - len(columns)}")
        return " ".join(parts)
    if isinstance(snapshot, RoutingSnapshot):
        aggregate = next(
            (row for row in snapshot.rows if row.system_id == "ALL"), snapshot.rows[0]
        )
        return (
            f"teacher={'/'.join(str(value) for value in aggregate.teacher)} "
            f"student={'/'.join(str(value) for value in aggregate.student)}"
        )
    if isinstance(snapshot, DescriptorSnapshot):
        parts = [
            f"descriptor={_compact_token(snapshot.label)}",
            f"completed={snapshot.completed}/{snapshot.total}",
            f"pct={_percentage(snapshot.completed, snapshot.total)}",
        ]
        if snapshot.width is not None:
            parts.append(f"width={snapshot.width}")
        if snapshot.elements_total is not None:
            parts.append(
                f"elements={_count(snapshot.elements_completed)}/{snapshot.elements_total}"
            )
        return " ".join(parts)
    if isinstance(snapshot, SamplingSnapshot):
        if snapshot.task_step_totals:
            position = snapshot.completed_tasks
            total = snapshot.total_tasks
        else:
            position = snapshot.completed_tasks * snapshot.steps_per_task
            if snapshot.completed_tasks < snapshot.total_tasks:
                position += snapshot.active_step or 0
            total = snapshot.total_tasks * snapshot.steps_per_task
        parts = [
            f"backend={_compact_token(snapshot.backend)}",
            f"shape={snapshot.model_shape}",
            f"tasks={snapshot.completed_tasks}/{snapshot.total_tasks}",
            f"pct={_percentage(position, total)}",
        ]
        if snapshot.active_task_id:
            parts.append(f"active={_compact_token(snapshot.active_task_id)}")
        if snapshot.active_step is not None:
            parts.append(f"step={snapshot.active_step}/{snapshot.steps_per_task}")
        if snapshot.eta_seconds is not None:
            parts.append(f"ETA={_duration(snapshot.eta_seconds)}")
        if snapshot.mps_phase:
            parts.append(f"mps={_compact_token(snapshot.mps_phase)}")
        alert = next(
            (
                detail
                for detail in snapshot.task_details
                if detail.state in {"warning", "failed"}
            ),
            None,
        )
        if alert is not None:
            parts.append(f"alert={alert.state}:{_compact_token(alert.task_id)}")
            diagnostic_ref = _relative_diagnostic_ref(alert.diagnostic_locator)
            if diagnostic_ref:
                parts.append(f"see={diagnostic_ref}")
        return " ".join(parts)
    return None


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _elapsed_between(started_at: str, updated_at: str) -> str:
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    return _duration(max(0.0, (updated - started).total_seconds()))


def _overall_status(projection: ProgressProjection) -> str:
    statuses = {block.status.value for block in projection.blocks}
    for terminal in ("FAILED", "INTERRUPTED"):
        if terminal in statuses:
            return terminal
    if "RUNNING" in statuses:
        return "RUNNING"
    if statuses == {"SUCCESS"}:
        return "SUCCESS"
    return "READY"


def _utc_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _render_summary(summary: object | None) -> str:
    if isinstance(summary, BasicBlockSummary):
        parts = [summary.text] if summary.text else []
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(summary.elapsed_seconds)}")
        return " · ".join(parts)
    if isinstance(summary, TrainBlockSummary):
        parts = [
            summary.backend,
            f"{summary.models} {'model' if summary.models == 1 else 'models'}",
            f"dataset {summary.dataset_unique}",
        ]
        if summary.train is not None:
            parts.append(f"train {summary.train}")
        if summary.valid is not None:
            parts.append(f"valid {summary.valid}")
        if summary.evaluation_same_data:
            parts.append("evaluation same dataset")
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(summary.elapsed_seconds)}")
        return " · ".join(parts)
    if isinstance(summary, ScreenBlockSummary):
        parts = [
            f"{summary.observations} observations",
            f"train {summary.train}",
            f"FP candidates {summary.fp_candidates}",
        ]
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(summary.elapsed_seconds)}")
        return " · ".join(parts)
    if isinstance(summary, FPBlockSummary):
        parts = [
            f"{summary.candidates} candidates",
            f"{summary.unique} unique",
            f"{summary.selected} selected",
            "no FP handoff required" if summary.legal_empty else f"handoff {summary.handoff_path}",
        ]
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(summary.elapsed_seconds)}")
        return " · ".join(parts)
    if isinstance(summary, SamplingBlockSummary):
        parts = [
            summary.backend,
            summary.model_shape,
            f"{summary.tasks} tasks",
            f"{summary.systems} systems × {summary.temperatures} temperatures",
            f"{summary.total_steps} {'planned' if summary.failed_tasks else 'total'} steps",
        ]
        if summary.mc_attempts:
            parts.append(f"{summary.mc_attempts} {'planned ' if summary.failed_tasks else ''}MC attempts")
        if summary.failed_tasks:
            parts.append(f"{summary.failed_tasks} failed tasks")
        if summary.kind_counts:
            parts.append(
                "kinds "
                + ", ".join(
                    f"{kind} {count}" for kind, count in summary.kind_counts
                )
            )
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed {_duration(summary.elapsed_seconds)}")
        return " · ".join(parts)
    if summary is None:
        return ""
    raise TypeError(f"unsupported progress summary {type(summary)!r}")


def _render_history_summary(summary: object | None) -> str:
    if isinstance(summary, BasicBlockSummary):
        parts = [f"detail={_compact_text(summary.text)}"] if summary.text else []
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed={_duration(summary.elapsed_seconds)}")
        return " ".join(parts)
    if isinstance(summary, TrainBlockSummary):
        parts = [
            f"backend={_compact_text(summary.backend)}",
            f"models={summary.models}",
            f"dataset={summary.dataset_unique}",
        ]
        if summary.train is not None:
            parts.append(f"train={summary.train}")
        if summary.valid is not None:
            parts.append(f"valid={summary.valid}")
        if summary.evaluation_same_data:
            parts.append("evaluation=same-dataset")
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed={_duration(summary.elapsed_seconds)}")
        return " ".join(parts)
    if isinstance(summary, ScreenBlockSummary):
        parts = [
            f"observations={summary.observations}",
            f"train={summary.train}",
            f"fp_candidates={summary.fp_candidates}",
        ]
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed={_duration(summary.elapsed_seconds)}")
        return " ".join(parts)
    if isinstance(summary, FPBlockSummary):
        parts = [
            f"candidates={summary.candidates}",
            f"unique={summary.unique}",
            f"selected={summary.selected}",
            (
                "handoff=none"
                if summary.legal_empty
                else f"handoff={_compact_text(summary.handoff_path)}"
            ),
        ]
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed={_duration(summary.elapsed_seconds)}")
        return " ".join(parts)
    if isinstance(summary, SamplingBlockSummary):
        parts = [
            f"backend={_compact_text(summary.backend)}",
            f"shape={summary.model_shape}",
            f"tasks={summary.tasks}",
            f"systems={summary.systems}",
            f"temperatures={summary.temperatures}",
            f"{'planned_steps' if summary.failed_tasks else 'steps'}={summary.total_steps}",
        ]
        if summary.mc_attempts:
            parts.append(f"{'planned_mc_attempts' if summary.failed_tasks else 'mc_attempts'}={summary.mc_attempts}")
        if summary.failed_tasks:
            parts.append(f"failed_tasks={summary.failed_tasks}")
        if summary.kind_counts:
            parts.append(
                "kinds="
                + ",".join(
                    f"{_compact_text(kind)}:{count}"
                    for kind, count in summary.kind_counts
                )
            )
        if summary.elapsed_seconds is not None:
            parts.append(f"elapsed={_duration(summary.elapsed_seconds)}")
        return " ".join(parts)
    if summary is None:
        return ""
    raise TypeError(f"unsupported progress summary {type(summary)!r}")


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _compact_token(value: str) -> str:
    return _bounded_label(_compact_text(value).replace(" ", "_"))


def _bounded_compact_text(value: str, *, limit: int = 120) -> str:
    text = _compact_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _percentage(completed: int, total: int) -> int:
    if total <= 0:
        return 100 if completed == total else 0
    return min(100, max(0, int(completed * 100 / total)))


def _failure_message(value: str, *, limit: int = 200) -> str:
    text = re.sub(r"(?:(?:[A-Za-z]:[\\/])|/)\S+", "<path>", _compact_text(value))
    text = re.sub(r"\bsee\s*:", "see", text, flags=re.IGNORECASE)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _relative_diagnostic_ref(value: str | None) -> str | None:
    if not value:
        return None
    compact = _compact_text(value).replace("\\", "/")
    path = PurePosixPath(compact)
    if (
        path.is_absolute()
        or re.match(r"^[A-Za-z]:/", compact)
        or any(part in {"", ".."} for part in path.parts)
    ):
        return None
    return str(path)


def _count(value: int | None) -> str:
    return "—" if value is None else str(value)


def _progress_count(completed: int, total: int | None) -> str:
    return str(completed) if total is None else f"{completed}/{total}"


def _display_width(value: str) -> int:
    import unicodedata

    return sum(
        0 if unicodedata.combining(character) else 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def _bounded_label(value: str, *, max_width: int = 48) -> str:
    if _display_width(value) <= max_width:
        return value
    retained: list[str] = []
    width = 0
    for character in value:
        character_width = _display_width(character)
        if width + character_width > max_width - 1:
            break
        retained.append(character)
        width += character_width
    return "".join(retained).rstrip() + "…"


def _ljust_display(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def _bounded_system_rows(rows, *, system_limit: int = 12):
    rows = tuple(rows)
    aggregate = next((row for row in rows if row.system_id == "ALL"), None)
    systems = tuple(row for row in rows if row.system_id != "ALL")
    bounded = systems[:system_limit]
    if aggregate is not None:
        bounded = (aggregate, *bounded)
    return bounded, max(0, len(systems) - system_limit)


def _metric_value(value: float) -> str:
    return f"{value:.6g}"
