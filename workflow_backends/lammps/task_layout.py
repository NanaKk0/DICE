from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SamplingTask:
    task_id: str
    job_id: str
    seed_index: int
    seed_source: str
    initial_structure_path: str
    source_system_id: str
    source_structure_id: str
    temperature: float
    steps: int
    timestep_fs: float
    job_seed: int
    driver_model_index: int
    driver_graph_path: str
    task_dir: Path
    conf_lmp: Path
    input_lammps: Path
    model_paths: list[Path]
    lammps_command: str
    plugin_path: Path | None


@dataclass
class SamplingTaskResult:
    task_id: str
    job_id: str
    task_dir: Path
    status: str
    returncode: int
    stdout_path: Path
    log_path: Path
    model_devi_path: Path
    traj_path: Path
    wall_time_sec: float
    group_id: str | None = None
    executor_backend: str = "inline"


@dataclass
class SamplingResult:
    tasks: list[SamplingTask]
    task_results: list[SamplingTaskResult]
    jobs_payload: list[dict[str, Any]]
    tasks_manifest_path: Path


@dataclass
class TaskGroup:
    group_id: str
    group_index: int
    group_dir: Path
    script_path: Path
    stdout_path: Path
    runtime_path: Path
    manifest_path: Path
    tasks: list[SamplingTask]


def extract_model_paths(committee_pool: list[dict[str, Any]]) -> list[Path]:
    return [Path(member["graph_pb"]).resolve() for member in committee_pool]


def copy_models_into_task(task_dir: Path, model_paths: list[Path]) -> list[Path]:
    copied = []
    for model_index, model_path in enumerate(model_paths):
        suffix = model_path.suffix or ".pb"
        target = task_dir / f"graph_{model_index:03d}{suffix}"
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(model_path)
        except OSError:
            try:
                os.link(model_path, target)
            except OSError:
                shutil.copy2(model_path, target)
        copied.append(target)
    return copied


def serialize_task(task: SamplingTask) -> dict[str, Any]:
    payload = asdict(task)
    payload["task_dir"] = str(task.task_dir.resolve())
    payload["conf_lmp"] = str(task.conf_lmp.resolve())
    payload["input_lammps"] = str(task.input_lammps.resolve())
    payload["model_paths"] = [str(item) for item in task.model_paths]
    payload["plugin_path"] = None if task.plugin_path is None else str(task.plugin_path.resolve())
    return payload


def serialize_job_payload(task: SamplingTask) -> dict[str, Any]:
    return {
        "job_id": task.job_id,
        "task_id": task.task_id,
        "initial_structure_path": str(Path(task.initial_structure_path).resolve()),
        "source_system_id": str(task.source_system_id),
        "source_structure_id": str(task.source_structure_id),
        "temperature": float(task.temperature),
        "steps": int(task.steps),
        "timestep_fs": float(task.timestep_fs),
        "driver_model_index": int(task.driver_model_index),
        "driver_graph_path": str(Path(task.driver_graph_path).resolve()),
        "job_seed": int(task.job_seed),
        "task_dir": str(task.task_dir.resolve()),
        "conf_lmp": str(task.conf_lmp.resolve()),
        "model_paths": [str(item) for item in task.model_paths],
        "lammps_command": str(task.lammps_command),
        "plugin_path": None if task.plugin_path is None else str(task.plugin_path.resolve()),
    }
