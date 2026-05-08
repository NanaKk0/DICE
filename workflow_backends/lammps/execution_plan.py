from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionPlan:
    backend: str
    max_parallel_tasks: int
    device_policy: str
    task_group_size: int
    max_parallel_groups: int
    group_parallel_degree: int
    group_fail_fast: bool
    shell_command: str
    cuda_device_pool: list[str]


def resolve_parallel_runtime(lammps_cfg: dict[str, Any]) -> tuple[int, str]:
    max_parallel_tasks = max(1, int(lammps_cfg.get("max_parallel_tasks", 1) or 1))
    device_policy = str(lammps_cfg.get("device_policy", "cpu")).strip().lower()
    if device_policy == "gpu_single_task" and max_parallel_tasks > 1:
        print("[WARN] gpu_single_task mode forces max_parallel_tasks=1")
        max_parallel_tasks = 1
    if device_policy not in {"cpu", "gpu_single_task", "gpu_shared_experimental"}:
        raise ValueError(f"Unsupported device_policy: {device_policy}")
    return max_parallel_tasks, device_policy


def _parse_device_pool(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    text = str(raw_value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _resolve_shell_command(raw_command: str) -> str:
    command = str(raw_command or "bash").strip() or "bash"
    candidate = os.path.abspath(command)
    if os.path.exists(candidate):
        return candidate
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Unable to find grouped-shell launcher: {command}")


def resolve_execution_plan(lammps_cfg: dict[str, Any]) -> ExecutionPlan:
    max_parallel_tasks, device_policy = resolve_parallel_runtime(lammps_cfg)
    backend = str(lammps_cfg.get("execution_backend", "inline")).strip().lower() or "inline"
    if backend not in {"inline", "grouped_shell"}:
        raise ValueError(f"Unsupported execution_backend: {backend}")

    task_group_size = max(1, int(lammps_cfg.get("task_group_size", 1) or 1))
    max_parallel_groups = int(lammps_cfg.get("max_parallel_groups", 0) or 0)
    if max_parallel_groups < 0:
        raise ValueError("max_parallel_groups must be >= 0")
    group_parallel_degree = max(1, int(lammps_cfg.get("group_parallel_degree", 1) or 1))
    group_fail_fast = bool(lammps_cfg.get("group_fail_fast", True))
    shell_command = "bash"

    cuda_device_pool = []
    if not cuda_device_pool and device_policy != "cpu":
        cuda_device_pool = _parse_device_pool(os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    if backend == "inline":
        task_group_size = 1
        group_parallel_degree = 1
        if max_parallel_groups == 0:
            max_parallel_groups = max_parallel_tasks
    else:
        group_parallel_degree = min(group_parallel_degree, task_group_size)
        if device_policy == "gpu_single_task" and group_parallel_degree > 1:
            print("[WARN] gpu_single_task mode forces group_parallel_degree=1")
            group_parallel_degree = 1
        shell_command = _resolve_shell_command(shell_command)

    return ExecutionPlan(
        backend=backend,
        max_parallel_tasks=max_parallel_tasks,
        device_policy=device_policy,
        task_group_size=task_group_size,
        max_parallel_groups=max_parallel_groups,
        group_parallel_degree=group_parallel_degree,
        group_fail_fast=group_fail_fast,
        shell_command=shell_command,
        cuda_device_pool=cuda_device_pool,
    )


def serialize_execution_plan(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "backend": plan.backend,
        "max_parallel_tasks": int(plan.max_parallel_tasks),
        "device_policy": plan.device_policy,
        "task_group_size": int(plan.task_group_size),
        "max_parallel_groups": int(plan.max_parallel_groups),
        "group_parallel_degree": int(plan.group_parallel_degree),
        "group_fail_fast": bool(plan.group_fail_fast),
        "shell_command": plan.shell_command,
        "cuda_device_pool": list(plan.cuda_device_pool),
    }
