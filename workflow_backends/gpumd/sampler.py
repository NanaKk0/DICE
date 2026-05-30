from __future__ import annotations

import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from workflow_backends.gpumd.active_parser import pair_active_records_with_structures, parse_active_out
from workflow_backends.gpumd.model_writer import link_committee_models, write_task_model_xyz
from workflow_backends.gpumd.run_writer import write_gpumd_run_input
from workflow_backends.student.nep.env_check import resolve_gpumd_command
from workflow_backends.sampling.base import SamplingBackendResult, SamplingObservation
from workflow_core.common.io import load_atoms_list, read_json, write_candidate_xyz, write_json
from workflow_core.common.uncertainty import UncertaintyRecord
from workflow_core.student.sampling_postprocess import postprocess_student_sampling_result


def _default_work_root(paths, outer_round: int, student_round: int) -> Path:
    del paths, outer_round, student_round
    raise RuntimeError("GPUMD target-mode sampling requires an explicit work_root")


def _system_id(atoms) -> str:
    info = atoms.info or {}
    for key in ("system_id", "seed_system_id", "source_system", "formula"):
        value = info.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return atoms.get_chemical_formula(mode="hill")


def _append_progress_log(progress_log_path: str | Path | None, message: str) -> None:
    if progress_log_path is None:
        return
    path = Path(progress_log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def prepare_gpumd_sampling_tasks(
    *,
    seed_xyz_path: str | Path,
    model_paths: list[str],
    gpumd_cfg: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    work_root: str | Path,
    outer_round: int,
    student_round: int,
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    tasks_root = work_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    seeds = load_atoms_list(seed_xyz_path)
    if not seeds:
        raise RuntimeError(f"No seed structures found for GPUMD sampling: {seed_xyz_path}")
    temperatures = [float(item) for item in gpumd_cfg.get("temperatures", [])]
    tasks: list[dict[str, Any]] = []
    for seed_index, atoms in enumerate(seeds):
        seed_file = work_root / "seeds" / f"seed_{seed_index:06d}.xyz"
        write_candidate_xyz(seed_file, [atoms])
        for temp_index, temperature in enumerate(temperatures):
            task_id = f"task_{len(tasks) + 1:06d}"
            task_dir = tasks_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            model_xyz = write_task_model_xyz(seed_file, task_dir / "model.xyz")
            copied_models = link_committee_models(model_paths, task_dir)
            active_outputs = dict(gpumd_cfg.get("active_outputs", {}) or {})
            active_interval = int(gpumd_cfg.get("active_interval", 20))
            dump_interval = int(gpumd_cfg.get("dump_interval", active_interval))
            run_in = write_gpumd_run_input(
                task_dir / "run.in",
                model_paths=[str(item) for item in copied_models],
                temperature=temperature,
                steps=int(gpumd_cfg.get("steps", gpumd_cfg.get("steps_short", 500))),
                timestep_fs=float(gpumd_cfg.get("timestep_fs", 0.5)),
                active_interval=active_interval,
                active_threshold=float(gpumd_cfg.get("active_threshold", uncertainty_cfg.get("trust_lo", 0.10))),
                dump_interval=dump_interval,
                ensemble=str(gpumd_cfg.get("ensemble", "nvt")),
                thermostat_damp=float(gpumd_cfg.get("thermostat_damp", 100.0)),
                save_forces=bool(active_outputs.get("force", gpumd_cfg.get("save_forces", True))),
                save_velocities=bool(active_outputs.get("velocity", gpumd_cfg.get("save_velocities", True))),
                template_path=gpumd_cfg.get("run_template_ref") or gpumd_cfg.get("run_in_template"),
                temperature_override_declared=bool(gpumd_cfg.get("temperature_override_declared", False)),
                steps_override_declared=bool(gpumd_cfg.get("steps_override_declared", False)),
            )
            task_meta = {
                "task_id": task_id,
                "seed_index": seed_index,
                "temperature": temperature,
                "outer_round": int(outer_round),
                "student_round": int(student_round),
                "model_xyz": str(model_xyz),
                "run_in": str(run_in),
                "active_interval": active_interval,
                "dump_interval": dump_interval,
            }
            write_json(task_dir / "task_meta.json", task_meta)
            tasks.append(
                {
                    **task_meta,
                    "task_dir": str(task_dir.resolve()),
                }
            )
    manifest = {
        "backend": "nep",
        "sampling_backend": "gpumd",
        "seed_xyz_path": str(Path(seed_xyz_path).resolve()),
        "model_paths": [str(Path(item).resolve()) for item in model_paths],
        "work_root": str(work_root.resolve()),
        "tasks": tasks,
    }
    write_json(work_root / "tasks_manifest.json", manifest)
    return manifest


def _run_one_gpumd_task(command: list[str], task: dict[str, Any], progress_log_path: str | Path | None = None) -> dict[str, Any]:
    task_dir = Path(task["task_dir"]).resolve()
    stdout_log = task_dir / "gpumd.stdout.log"
    _append_progress_log(progress_log_path, f"START {task['task_id']} T={task.get('temperature')} dir={task_dir}")
    with open(stdout_log, "w", encoding="utf-8") as handle:
        try:
            proc = subprocess.run(
                list(command),
                cwd=task_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"cannot find gpumd executable: {command[0]}") from exc
    _append_progress_log(
        progress_log_path,
        f"END {task['task_id']} returncode={int(proc.returncode)} stdout_log={stdout_log.resolve()}",
    )
    return {
        **task,
        "returncode": int(proc.returncode),
        "stdout_log": str(stdout_log.resolve()),
        "active_out": str((task_dir / "active.out").resolve()),
        "active_xyz": str((task_dir / "active.xyz").resolve()),
        "movie_xyz": str((task_dir / "movie.xyz").resolve()),
    }


def _gpumd_task_failed(task: dict[str, Any]) -> bool:
    return int(task.get("returncode", 1)) != 0


def _sampling_observation_from_active_record(
    *,
    record: UncertaintyRecord,
    atoms,
    outer_round: int,
    student_round: int,
) -> SamplingObservation:
    frame_index = int(str(record.frame_id or "0:0").rsplit(":", 1)[-1])
    metadata = dict(record.metadata or {})
    metadata.update(
        {
            "outer_round": int(outer_round),
            "student_round": int(student_round),
            "uncertainty_source": "gpumd_active",
            "uncertainty_metric": "force_std_max",
        }
    )
    if atoms is not None:
        metadata["source_system_id"] = _system_id(atoms)
    return SamplingObservation(
        task_id=str(record.task_id),
        frame_index=frame_index,
        step=record.step,
        temperature=record.temperature,
        force_std_max=float(record.force_std_max),
        atoms=atoms,
        metadata=metadata,
    )


def run_gpumd_sampling_raw(
    *,
    seed_xyz_path: str | Path,
    model_paths: list[str],
    gpumd_cfg: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    work_root: str | Path,
    outer_round: int,
    student_round: int,
    progress_log_path: str | Path | None = None,
) -> SamplingBackendResult:
    manifest = prepare_gpumd_sampling_tasks(
        seed_xyz_path=seed_xyz_path,
        model_paths=model_paths,
        gpumd_cfg=gpumd_cfg,
        uncertainty_cfg=uncertainty_cfg,
        work_root=work_root,
        outer_round=outer_round,
        student_round=student_round,
    )
    command = resolve_gpumd_command(gpumd_cfg)
    max_parallel_tasks = max(1, int(gpumd_cfg.get("max_parallel_tasks", 1)))
    completed: list[dict[str, Any]] = []
    _append_progress_log(
        progress_log_path,
        f"GPUMD sampling prepared tasks={len(manifest['tasks'])} work_root={Path(work_root).resolve()}",
    )
    if max_parallel_tasks == 1:
        for task in manifest["tasks"]:
            completed.append(_run_one_gpumd_task(command, task, progress_log_path=progress_log_path))
    else:
        with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            future_map = {
                executor.submit(_run_one_gpumd_task, command, task, progress_log_path): task
                for task in manifest["tasks"]
            }
            for future in as_completed(future_map):
                completed.append(future.result())
        completed.sort(key=lambda item: item["task_id"])

    failed_tasks = [item for item in completed if _gpumd_task_failed(item)]
    if failed_tasks:
        examples = ", ".join(
            f"{item['task_id']} rc={item['returncode']} log={item['stdout_log']}"
            for item in failed_tasks[:5]
        )
        raise RuntimeError(
            f"GPUMD sampling failed for {len(failed_tasks)}/{len(completed)} tasks. "
            f"Examples: {examples}"
        )

    trust_lo = float(uncertainty_cfg.get("trust_lo", 0.10))
    trust_hi = float(uncertainty_cfg.get("trust_hi", 0.40))
    all_records = []
    task_summaries: list[dict[str, Any]] = []
    all_routed_pairs: list[tuple[UncertaintyRecord, Any]] = []

    for task in completed:
        task_dir = Path(task["task_dir"]).resolve()
        records = parse_active_out(
            task_dir / "active.out",
            task_id=str(task["task_id"]),
            temperature=float(task["temperature"]),
            trust_lo=trust_lo,
            trust_hi=trust_hi,
        )
        all_records.extend(records)
        routed_pairs = []
        if (task_dir / "active.xyz").exists():
            routed_pairs = pair_active_records_with_structures(
                records,
                task_dir / "active.xyz",
                metadata={
                    "student_backend": "nep",
                    "sampling_backend": "gpumd",
                    "uncertainty_source": "gpumd_active",
                    "uncertainty_metric": "force_std_max",
                    "outer_round": int(outer_round),
                    "student_round": int(student_round),
                    "temperature": float(task["temperature"]),
                    "seed_id": int(task["seed_index"]),
                },
            )
            for record, atoms in routed_pairs:
                atoms.info["source_task"] = str(task["task_id"])
                atoms.info["source_job_id"] = str(task["task_id"])
                atoms.info["source_frame_index"] = int(str(record.frame_id).rsplit(":", 1)[-1])
                atoms.info["md_step"] = int(record.step)
            all_routed_pairs.extend(routed_pairs)
        task_record_count = len(records)
        task_failed_count = sum(1 for item in records if item.route == "failed_uncertainty")
        task_summaries.append(
            {
                "task_id": task["task_id"],
                "returncode": int(task["returncode"]),
                "record_count": int(task_record_count),
                "candidate_count": sum(1 for item in records if item.route == "candidate"),
                "failed_count": int(task_failed_count),
                "accurate_count": sum(1 for item in records if item.route == "accurate"),
                "failed_ratio": float(task_failed_count / task_record_count) if task_record_count else 0.0,
                "stdout_log": task["stdout_log"],
            }
        )

    routed_atoms_by_frame = {str(record.frame_id): atoms for record, atoms in all_routed_pairs}
    observations = [
        _sampling_observation_from_active_record(
            record=record,
            atoms=routed_atoms_by_frame.get(str(record.frame_id)),
            outer_round=outer_round,
            student_round=student_round,
        )
        for record in all_records
    ]
    return SamplingBackendResult(
        backend="nep",
        sampling_backend="gpumd",
        work_root=Path(work_root).resolve(),
        jobs=[dict(item) for item in completed],
        observations=observations,
        task_summaries=task_summaries,
        runtime_artifacts={
            "usable_committee_size": int(len(model_paths)),
            "max_parallel_tasks": int(max_parallel_tasks),
        },
    )


def run_gpumd_sampling(
    *,
    seed_xyz_path: str | Path,
    model_paths: list[str],
    gpumd_cfg: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    work_root: str | Path,
    outer_round: int,
    student_round: int,
    progress_log_path: str | Path | None = None,
) -> str:
    backend_result = run_gpumd_sampling_raw(
        seed_xyz_path=seed_xyz_path,
        model_paths=model_paths,
        gpumd_cfg=gpumd_cfg,
        uncertainty_cfg=uncertainty_cfg,
        work_root=work_root,
        outer_round=outer_round,
        student_round=student_round,
        progress_log_path=progress_log_path,
    )
    trust_lo = float(uncertainty_cfg.get("trust_lo", 0.10))
    trust_hi = float(uncertainty_cfg.get("trust_hi", 0.40))
    selection_cfg = dict(gpumd_cfg.get("candidate_train_selection", {}) or {})
    selection_cfg.setdefault("enabled", False)
    summary_path = postprocess_student_sampling_result(
        backend_result=backend_result,
        cfg={
            "uncertainty": {
                "trust_lo": trust_lo,
                "trust_hi": trust_hi,
                "level_f_lo": trust_lo,
                "level_f_hi": trust_hi,
            },
            "candidate_train_selection": selection_cfg,
        },
        work_root=work_root,
        summary={"committee_size": int(len(model_paths)), "usable_committee_size": int(len(model_paths))},
        context={
            "sampling_progress_log_path": progress_log_path,
            "stop_high_unc_threshold": trust_lo,
            "summary_extra": {
                "outer_round": int(outer_round),
                "student_round": int(student_round),
            },
        },
    )
    summary = read_json(summary_path)
    _append_progress_log(
        progress_log_path,
        "GPUMD sampling summary "
        f"frames={int(summary.get('effective_sample_count', 0) or 0)} "
        f"candidate={int(summary.get('candidate_count', 0) or 0)} "
        f"failed={int(summary.get('failed_uncertainty_count', 0) or 0)} "
        f"accurate={summary['accurate_count']} high_unc_ratio={summary['high_unc_ratio']:.6g} "
        f"summary={summary_path.resolve()}",
    )
    return str(summary_path.resolve())
