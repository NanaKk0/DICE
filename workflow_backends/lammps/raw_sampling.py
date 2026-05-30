from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from workflow_backends.lammps.model_devi_parser import parse_model_devi_out
from workflow_backends.lammps.task_sampler import LammpsTaskSampler, SamplingResult, resolve_execution_plan
from workflow_backends.sampling.base import SamplingBackendResult, SamplingObservation, TrajectoryFrameRef
from workflow_core.common.io import write_json
from workflow_policies.sampling_jobs import build_sampling_jobs


def _observation_from_model_devi_row(
    *,
    task_payload: dict[str, Any],
    task_result,
    row: dict[str, Any],
    frame_index: int,
    dump_interval: int,
) -> SamplingObservation:
    job_id = str(task_payload.get("job_id", ""))
    source_system_id = str(task_payload.get("source_system_id") or task_payload.get("seed_system_id") or "UNKNOWN")
    fallback_step = int(frame_index * dump_interval)
    step = int(row.get("step", fallback_step))
    return SamplingObservation(
        task_id=job_id,
        frame_index=int(frame_index),
        step=step,
        force_std_max=float(row["max_devi_f"]),
        force_std_avg=float(row.get("avg_devi_f", row["max_devi_f"])),
        force_std_p90=float(row.get("max_devi_f", row["max_devi_f"])),
        trajectory_ref=TrajectoryFrameRef(
            trajectory_path=str(task_result.traj_path.resolve()),
            trajectory_format="lammpstrj",
            step=step,
            frame_index=int(frame_index),
            fallback_step=fallback_step,
            metadata={
                "model_devi_out": str(task_result.model_devi_path.resolve()),
            },
        ),
        metadata={
            "source_system_id": source_system_id,
            "seed_system_id": task_payload.get("seed_system_id"),
            "source_structure_id": (
                task_payload.get("source_structure_id")
                or task_payload.get("seed_structure_id")
                or Path(str(task_payload.get("initial_structure_path", ""))).stem
            ),
            "initial_structure_path": task_payload.get("initial_structure_path"),
            "driver_model_index": int(task_payload.get("driver_model_index", 0) or 0),
            "model_devi_out": str(task_result.model_devi_path.resolve()),
            "traj_path": str(task_result.traj_path.resolve()),
            "uncertainty_source": "model_devi",
        },
    )


def run_lammps_model_devi_sampling_raw(
    *,
    initial_structure_paths: list[str],
    committee_pool: list[dict[str, Any]],
    cfg: dict[str, Any],
    work_root: str | Path,
    summary: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> SamplingBackendResult:
    context = context or {}
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

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
    write_json(work_root / "jobs.raw.json", [dict(item) for item in jobs])

    lammps_cfg = dict(cfg.get("lammps", {}) or {})
    execution_plan = resolve_execution_plan(lammps_cfg)
    dump_interval = int(lammps_cfg.get("dump_interval", lammps_cfg.get("dump_freq", 10)))

    md_phase_started_at = time.perf_counter()
    task_results = task_sampler.run_tasks(tasks)
    md_phase_wall_time_sec = float(time.perf_counter() - md_phase_started_at)
    sampling_result: SamplingResult = task_sampler.collect_results(tasks, task_results)

    observations: list[SamplingObservation] = []
    task_summaries: list[dict[str, Any]] = []
    for task_payload, task_result in zip(sampling_result.jobs_payload, sampling_result.task_results):
        if task_result.status != "finished":
            task_summaries.append(
                {
                    "task_id": str(task_payload.get("job_id", "")),
                    "status": str(task_result.status),
                    "record_count": 0,
                    "model_devi_out": str(task_result.model_devi_path.resolve()),
                    "traj_path": str(task_result.traj_path.resolve()),
                }
            )
            continue
        rows = parse_model_devi_out(
            task_result.model_devi_path,
            natoms=None,
            atomic_enabled=False,
        )
        for frame_index, row in enumerate(rows):
            observations.append(
                _observation_from_model_devi_row(
                    task_payload=task_payload,
                    task_result=task_result,
                    row=row,
                    frame_index=frame_index,
                    dump_interval=dump_interval,
                )
            )
        task_summaries.append(
            {
                "task_id": str(task_payload.get("job_id", "")),
                "status": str(task_result.status),
                "record_count": int(len(rows)),
                "model_devi_out": str(task_result.model_devi_path.resolve()),
                "traj_path": str(task_result.traj_path.resolve()),
            }
        )

    return SamplingBackendResult(
        backend=str(cfg.get("student_backend", "deepmd")),
        sampling_backend="lammps",
        work_root=work_root,
        jobs=[dict(item) for item in sampling_result.jobs_payload],
        observations=observations,
        task_summaries=task_summaries,
        runtime_artifacts={
            "usable_committee_size": int(len(committee_pool)),
            "execution_backend": execution_plan.backend,
            "md_phase_wall_time_sec": md_phase_wall_time_sec,
        },
    )
