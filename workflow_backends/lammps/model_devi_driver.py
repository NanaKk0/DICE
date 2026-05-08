from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from workflow_artifacts.sampling_export import (
    export_selected_candidates_xyz_only,
    write_sampling_summary_fast,
)
from workflow_artifacts.sampling_summary import (
    append_sampling_summary_to_progress_log as _append_sampling_summary_to_progress_log,
    trim_uncertainty_only_sampling_summary as _trim_uncertainty_only_sampling_summary,
)
from workflow_backends.lammps.model_devi_parser import (
    build_fast_pool_atoms as _build_fast_pool_atoms,
    build_fast_selected_record as _build_fast_selected_record,
    parse_model_devi_out as _parse_model_devi_out,
    route_frame_class_uncertainty_only as _route_frame_class_uncertainty_only,
)
from workflow_backends.lammps.task_sampler import LammpsTaskSampler, SamplingResult, resolve_execution_plan
from workflow_policies.sampling_jobs import build_sampling_jobs
from workflow_policies.sampling_training_selection import (
    select_training_candidates_by_system as _select_training_candidates_by_system,
)
from workflow_core.common.io import read_json, write_json
from workflow_core.seed_pool.uncertainty import group_atoms_by_system, write_system_extxyz_dir
from workflow_core.structures.formats import extract_lammpstrj_selected_frames


def _run_lammps_model_devi_sampling_fast(
    *,
    initial_structure_paths: list[str],
    committee_pool: list[dict[str, Any]],
    cfg: dict[str, Any],
    work_root: Path,
    summary: dict[str, Any],
    context: dict[str, Any],
) -> str:
    jobs = build_sampling_jobs(
        initial_structure_paths=initial_structure_paths,
        committee_pool=committee_pool,
        cfg=cfg,
    )
    task_sampler = LammpsTaskSampler(
        jobs=jobs,
        committee_pool=committee_pool,
        cfg=cfg,
        work_root=work_root,
        seed_source=str(context.get("initial_structure_source_override", "seed_pool")),
        progress_log_path=context.get("sampling_progress_log_path"),
        progress_poll_interval_sec=float(context.get("sampling_progress_poll_interval_sec", 2.0) or 2.0),
    )
    tasks = task_sampler.prepare_tasks()
    write_json(work_root / "jobs.json", [dict(item) for item in jobs])

    lammps_cfg = dict(cfg.get("lammps", {}) or {})
    execution_plan = resolve_execution_plan(lammps_cfg)
    uncertainty_cfg = dict(cfg.get("uncertainty", {}) or {})
    level_f_lo = float(lammps_cfg.get("f_trust_lo", uncertainty_cfg.get("level_f_lo", 0.10)))
    level_f_hi = float(lammps_cfg.get("f_trust_hi", uncertainty_cfg.get("level_f_hi", 0.40)))
    type_map = list(lammps_cfg.get("type_map", ["Pt", "O", "H"]))
    dump_freq = int(lammps_cfg.get("dump_freq", 10))
    high_unc_threshold = context.get("stop_high_unc_threshold")
    high_unc_threshold = None if high_unc_threshold is None else float(high_unc_threshold)

    md_phase_started_at = time.perf_counter()
    task_results = task_sampler.run_tasks(tasks)
    md_phase_wall_time_sec = float(time.perf_counter() - md_phase_started_at)
    sampling_result: SamplingResult = task_sampler.collect_results(tasks, task_results)

    postprocess_started_at = time.perf_counter()
    updated_jobs: list[dict[str, Any]] = list(sampling_result.jobs_payload)
    total_frames = 0
    accurate_count = 0
    candidate_count = 0
    failed_uncertainty_count = 0
    high_unc_count = 0
    system_route_counts: dict[str, Counter[str]] = {}
    candidate_records: list[dict[str, Any]] = []
    candidate_atoms_by_system: list[Atoms] = []
    failed_atoms_by_system: list[Atoms] = []
    selected_decision_values: list[float] = []
    selected_per_job_counts: Counter[str] = Counter()
    candidate_train_cfg = dict(cfg.get("candidate_train_selection", {}) or {})
    max_train_candidates_per_system = int(candidate_train_cfg.get("max_train_candidates_per_system", 200))
    train_selection_enabled = bool(candidate_train_cfg.get("enabled", True))
    candidate_train_selection_mode = str(candidate_train_cfg.get("selection_mode", "random_per_system")).strip().lower()
    candidate_train_random_seed = int(candidate_train_cfg.get("random_seed", 20260502))
    summary_outer_round = int((context.get("summary_extra", {}) or {}).get("outer_round", -1))
    summary_dp_round = int((context.get("summary_extra", {}) or {}).get("dp_round", -1))

    for task_payload, task_result in zip(updated_jobs, sampling_result.task_results):
        if task_result.status != "finished":
            continue
        system_id = str(task_payload.get("source_system_id") or task_payload.get("seed_system_id") or "UNKNOWN")
        route_counts = system_route_counts.setdefault(system_id, Counter())

        devi_rows = _parse_model_devi_out(
            task_result.model_devi_path,
            natoms=None,
            atomic_enabled=False,
        )
        total_frames += len(devi_rows)

        candidate_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        target_steps: set[int] = set()
        for local_index, devi_row in enumerate(devi_rows):
            decision_metric = float(devi_row["max_devi_f"])
            if high_unc_threshold is not None and decision_metric >= high_unc_threshold:
                high_unc_count += 1
            frame_class = _route_frame_class_uncertainty_only(
                decision_metric,
                f_trust_lo=level_f_lo,
                f_trust_hi=level_f_hi,
            )
            if frame_class == "accurate":
                accurate_count += 1
                route_counts["accurate"] += 1
                continue
            if frame_class == "failed_uncertainty":
                failed_uncertainty_count += 1
                route_counts["failed"] += 1
                failed_row = dict(devi_row)
                failed_row["frame_index"] = int(local_index)
                failed_row["fallback_step"] = int(local_index * dump_freq)
                failed_rows.append(failed_row)
                target_steps.add(int(failed_row["step"]))
                target_steps.add(int(failed_row["fallback_step"]))
                continue
            candidate_count += 1
            route_counts["candidate"] += 1
            candidate_row = dict(devi_row)
            candidate_row["frame_index"] = int(local_index)
            candidate_row["fallback_step"] = int(local_index * dump_freq)
            candidate_rows.append(candidate_row)
            target_steps.add(int(candidate_row["step"]))
            target_steps.add(int(candidate_row["fallback_step"]))

        if not candidate_rows and not failed_rows:
            continue

        frame_map = extract_lammpstrj_selected_frames(task_result.traj_path, type_map=type_map, target_steps=target_steps)
        for failed_row in failed_rows:
            failed_atoms = frame_map.get(int(failed_row["step"])) or frame_map.get(int(failed_row["fallback_step"]))
            if failed_atoms is None:
                continue
            failed_atoms_by_system.append(
                _build_fast_pool_atoms(
                    atoms=failed_atoms,
                    job=task_payload,
                    model_devi_row=failed_row,
                    route="failed_uncertainty",
                )
            )
        for candidate_row in candidate_rows:
            frame_atoms = frame_map.get(int(candidate_row["step"]))
            if frame_atoms is None:
                frame_atoms = frame_map.get(int(candidate_row["fallback_step"]))
            if frame_atoms is None:
                continue
            record = _build_fast_selected_record(
                atoms=frame_atoms,
                job=task_payload,
                frame_index=int(candidate_row["frame_index"]),
                model_devi_row=candidate_row,
            )
            candidate_records.append(record)
            candidate_atoms_by_system.append(
                _build_fast_pool_atoms(
                    atoms=record["atoms"],
                    job=task_payload,
                    model_devi_row=candidate_row,
                    route="candidate",
                )
            )

    candidate_by_system_dir = work_root / "candidate_by_system"
    failed_by_system_dir = work_root / "failed_by_system"
    write_system_extxyz_dir(candidate_by_system_dir, group_atoms_by_system(candidate_atoms_by_system))
    write_system_extxyz_dir(failed_by_system_dir, group_atoms_by_system(failed_atoms_by_system))

    selected_records = (
        _select_training_candidates_by_system(
            candidate_records,
            max_train_candidates_per_system=max_train_candidates_per_system,
            selection_mode=candidate_train_selection_mode,
            random_seed=candidate_train_random_seed,
            outer_round=summary_outer_round,
            dp_round=summary_dp_round,
        )
        if train_selection_enabled
        else list(candidate_records)
    )
    selected_count_per_system: Counter[str] = Counter()
    for record in selected_records:
        selected_decision_values.append(float(record["decision_metric"]))
        selected_per_job_counts[str(record["source_job_id"])] += 1
        selected_system_id = str(
            record["atoms"].info.get("system_id")
            or record["atoms"].info.get("seed_system_id")
            or record["atoms"].get_chemical_formula(mode="hill")
        )
        selected_count_per_system[selected_system_id] += 1

    compact_sampling_system_stats = {"group_by": "formula", "systems": {}, "totals": {}}
    totals = Counter()
    for system_id in sorted(system_route_counts):
        counts = system_route_counts[system_id]
        total = int(counts.get("accurate", 0) + counts.get("candidate", 0) + counts.get("failed", 0))
        denom = max(1, total)
        compact_sampling_system_stats["systems"][system_id] = {
            "accurate": int(counts.get("accurate", 0)),
            "candidate": int(counts.get("candidate", 0)),
            "failed": int(counts.get("failed", 0)),
            "accurate_ratio": float(counts.get("accurate", 0) / denom),
            "candidate_ratio": float(counts.get("candidate", 0) / denom),
            "failed_ratio": float(counts.get("failed", 0) / denom),
        }
        totals.update(
            {
                "accurate": int(counts.get("accurate", 0)),
                "candidate": int(counts.get("candidate", 0)),
                "failed": int(counts.get("failed", 0)),
            }
        )
    total_frames_by_system = int(totals.get("accurate", 0) + totals.get("candidate", 0) + totals.get("failed", 0))
    total_denom = max(1, total_frames_by_system)
    compact_sampling_system_stats["totals"] = {
        "accurate": int(totals.get("accurate", 0)),
        "candidate": int(totals.get("candidate", 0)),
        "failed": int(totals.get("failed", 0)),
        "accurate_ratio": float(totals.get("accurate", 0) / total_denom),
        "candidate_ratio": float(totals.get("candidate", 0) / total_denom),
        "failed_ratio": float(totals.get("failed", 0) / total_denom),
    }

    write_json(work_root / "jobs.json", updated_jobs)

    selected_xyz = work_root / "selected_candidates.xyz"
    export_selected_candidates_xyz_only(selected_records, selected_xyz)

    summary_output = write_sampling_summary_fast(
        summary_path=work_root / "sampling_summary.json",
        committee_size=int(summary.get("committee_size", len(summary.get("members", [])))),
        usable_committee_size=len(committee_pool),
        total_jobs=len(updated_jobs),
        total_frames=int(total_frames),
        accurate_count=int(accurate_count),
        candidate_count=int(candidate_count),
        failed_uncertainty_count=int(failed_uncertainty_count),
        selected_count=int(len(selected_records)),
        decision_metric="max_devi_f",
        level_f_lo=level_f_lo,
        level_f_hi=level_f_hi,
        selected_xyz=selected_xyz,
        selected_jsonl=None,
        stop_high_unc_threshold=high_unc_threshold,
        high_unc_count=int(high_unc_count),
        selected_decision_values=selected_decision_values,
        selected_per_job_counts=selected_per_job_counts,
        sampling_system_stats=compact_sampling_system_stats,
        extra_summary={
            **dict(context.get("summary_extra") or {}),
            "high_tail_buffer_count": 0,
            "sampling_policy_mode": "uncertainty_only",
            "sampling_profile_name": str(cfg.get("sampling_profile_name", "")),
            "candidate_train_selected_count_per_system": dict(sorted(selected_count_per_system.items())),
            "md_phase_wall_time_sec": md_phase_wall_time_sec,
            "postprocess_phase_wall_time_sec": float(time.perf_counter() - postprocess_started_at),
        },
    )
    _trim_uncertainty_only_sampling_summary(Path(summary_output).resolve())
    try:
        summary_payload = read_json(summary_output)
        if isinstance(summary_payload, dict):
            _append_sampling_summary_to_progress_log(context.get("sampling_progress_log_path"), summary_payload)
    except Exception:
        pass
    return str(Path(summary_output).resolve())


def run_lammps_model_devi_sampling(
    *,
    initial_structure_paths: list[str],
    committee_pool: list[dict[str, Any]],
    cfg: dict[str, Any],
    work_root: str | Path,
    summary: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> str:
    context = context or {}
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    return _run_lammps_model_devi_sampling_fast(
        initial_structure_paths=initial_structure_paths,
        committee_pool=committee_pool,
        cfg=cfg,
        work_root=work_root,
        summary=summary,
        context=context,
    )


