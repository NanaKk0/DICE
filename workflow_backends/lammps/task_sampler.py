from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from workflow_backends.lammps.execution_plan import (
    ExecutionPlan,
    resolve_execution_plan,
    resolve_parallel_runtime,
    serialize_execution_plan as _serialize_execution_plan,
)
from workflow_backends.lammps.input_writer import (
    build_lammps_env,
    find_deepmd_plugin,
    find_lammps_command,
    rotate_models,
    write_lammps_deepmd_input,
)
from workflow_backends.lammps.progress_log import SamplingProgressLogger
from workflow_backends.lammps.task_layout import (
    SamplingResult,
    SamplingTask,
    SamplingTaskResult,
    TaskGroup,
    copy_models_into_task as _copy_models_into_task,
    extract_model_paths as _extract_model_paths,
    serialize_job_payload as _serialize_job_payload,
    serialize_task as _serialize_task,
)
from workflow_core.common.io import write_json
from workflow_core.structures.formats import write_lammps_seed_data_from_paths


class LammpsTaskSampler:
    def __init__(
        self,
        *,
        jobs: list[dict[str, Any]],
        committee_pool: list[dict[str, Any]],
        cfg: dict[str, Any],
        work_root: str | Path,
        seed_source: str = "seed_pool",
        progress_log_path: str | Path | None = None,
        progress_poll_interval_sec: float = 2.0,
    ) -> None:
        self.jobs = list(jobs)
        self.committee_pool = list(committee_pool)
        self.cfg = cfg
        self.work_root = Path(work_root).resolve()
        self.seed_source = str(seed_source)
        self.lammps_cfg = dict(cfg.get("lammps", {}) or {})
        self.execution_plan = resolve_execution_plan(self.lammps_cfg)
        self.type_map = list(self.lammps_cfg.get("type_map", ["Pt", "O", "H"]))
        self.lammps_env = build_lammps_env(self.lammps_cfg)
        self.lammps_command = find_lammps_command(self.lammps_cfg)
        self.plugin_path = find_deepmd_plugin(
            self.lammps_cfg,
            lammps_cmd=self.lammps_command,
            env=self.lammps_env,
        )
        self.full_model_paths = _extract_model_paths(self.committee_pool)
        self.tasks_manifest_path = self.work_root / "tasks_manifest.json"
        self._prepared_tasks: list[SamplingTask] = []
        self.progress_logger = SamplingProgressLogger(
            progress_log_path,
            poll_interval_sec=progress_poll_interval_sec,
        )

    def prepare_tasks(self) -> list[SamplingTask]:
        conf_dir = self.work_root / "confs"
        tasks_dir = self.work_root / "tasks"
        reports_dir = self.work_root / "reports"
        selected_dir = self.work_root / "selected"
        for item in (conf_dir, tasks_dir, reports_dir, selected_dir):
            item.mkdir(parents=True, exist_ok=True)

        unique_initial_paths: list[str] = []
        for job in self.jobs:
            structure_path = str(Path(job["initial_structure_path"]).resolve())
            if structure_path not in unique_initial_paths:
                unique_initial_paths.append(structure_path)
        conf_mapping = write_lammps_seed_data_from_paths(unique_initial_paths, conf_dir, type_map=self.type_map)
        seed_index_map = {path: index for index, path in enumerate(unique_initial_paths)}

        tasks: list[SamplingTask] = []
        for task_index, job in enumerate(self.jobs):
            task_id = f"task.{task_index:06d}"
            task_dir = tasks_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            initial_structure_path = str(Path(job["initial_structure_path"]).resolve())
            conf_path = conf_mapping[initial_structure_path]
            target_conf = task_dir / "conf.lmp"
            if target_conf.exists() or target_conf.is_symlink():
                target_conf.unlink()
            try:
                target_conf.symlink_to(conf_path)
            except OSError:
                shutil.copy2(conf_path, target_conf)

            rotated_models = rotate_models(
                self.full_model_paths,
                task_index=task_index,
                enabled=bool(self.lammps_cfg.get("rotate_driver_model", True)),
            )
            copied_models = _copy_models_into_task(task_dir, rotated_models)
            input_lammps = write_lammps_deepmd_input(
                task_dir / "input.lammps",
                model_paths=copied_models,
                type_map=self.type_map,
                plugin_path=self.plugin_path,
                nsteps=int(job["steps"]),
                timestep_fs=float(job["timestep_fs"]),
                temperature=float(job["temperature"]),
                ensemble=str(self.lammps_cfg.get("ensemble", "nvt")),
                thermo_freq=int(self.lammps_cfg.get("thermo_freq", 10)),
                dump_freq=int(self.lammps_cfg.get("dump_freq", 10)),
                model_devi_freq=int(self.lammps_cfg.get("model_devi_freq", 10)),
                atomic_model_devi=bool(self.lammps_cfg.get("atomic_model_devi", True)),
                neighbor_bin=float(self.lammps_cfg.get("neighbor_bin", 2.0)),
                thermostat_damp_fs=float(self.lammps_cfg.get("thermostat_damp_fs", 100.0)),
                seed=int(job["job_seed"]),
            )
            task = SamplingTask(
                task_id=task_id,
                job_id=str(job["job_id"]),
                seed_index=int(seed_index_map[initial_structure_path]),
                seed_source=self.seed_source,
                initial_structure_path=initial_structure_path,
                source_system_id=str(job.get("source_system_id", "")),
                source_structure_id=str(job.get("source_structure_id", "")),
                temperature=float(job["temperature"]),
                steps=int(job["steps"]),
                timestep_fs=float(job["timestep_fs"]),
                job_seed=int(job["job_seed"]),
                driver_model_index=int(job["driver_model_index"]),
                driver_graph_path=str(Path(job["driver_graph_path"]).resolve()),
                task_dir=task_dir,
                conf_lmp=target_conf,
                input_lammps=input_lammps,
                model_paths=copied_models,
                lammps_command=str(self.lammps_command),
                plugin_path=None if self.plugin_path is None else self.plugin_path.resolve(),
            )
            write_json(task_dir / "job.json", _serialize_job_payload(task))
            write_json(
                task_dir / "task_meta.json",
                {
                    "task_id": task.task_id,
                    "job_id": task.job_id,
                    "seed_index": int(task.seed_index),
                    "seed_source": task.seed_source,
                    "temperature": float(task.temperature),
                    "nsteps": int(task.steps),
                    "timestep_fs": float(task.timestep_fs),
                    "dump_freq": int(self.lammps_cfg.get("dump_freq", 10)),
                    "model_devi_freq": int(self.lammps_cfg.get("model_devi_freq", 10)),
                    "driver_model_index": int(task.driver_model_index),
                    "model_paths": [item.name for item in task.model_paths],
                },
            )
            tasks.append(task)

        write_json(
            self.tasks_manifest_path,
            {
                "task_layout_version": str(self.lammps_cfg.get("task_layout_version", "v2")),
                "lammps_command": str(self.lammps_command),
                "plugin_path": None if self.plugin_path is None else str(self.plugin_path.resolve()),
                "type_map": self.type_map,
                "max_parallel_tasks": int(self.lammps_cfg.get("max_parallel_tasks", 1) or 1),
                "device_policy": str(self.lammps_cfg.get("device_policy", "cpu")),
                "execution": _serialize_execution_plan(self.execution_plan),
                "tasks": [_serialize_task(task) for task in tasks],
            },
        )
        self._prepared_tasks = tasks
        return list(tasks)

    def run_tasks(self, tasks: list[SamplingTask] | None = None) -> list[SamplingTaskResult]:
        effective_tasks = list(self._prepared_tasks if tasks is None else tasks)
        self.progress_logger.start(
            total_tasks=len(effective_tasks),
            backend=self.execution_plan.backend,
            max_parallel_tasks=self.execution_plan.max_parallel_tasks,
            max_parallel_groups=self.execution_plan.max_parallel_groups,
            group_parallel_degree=self.execution_plan.group_parallel_degree,
        )
        if self.execution_plan.backend == "grouped_shell":
            return self._run_tasks_grouped_shell(effective_tasks)
        return self._run_tasks_inline(effective_tasks)

    def collect_results(
        self,
        tasks: list[SamplingTask],
        task_results: list[SamplingTaskResult],
    ) -> SamplingResult:
        jobs_payload = []
        for task, task_result in zip(tasks, task_results):
            payload = _serialize_job_payload(task)
            payload["runtime"] = {
                "status": task_result.status,
                "returncode": int(task_result.returncode),
                "task_dir": str(task_result.task_dir.resolve()),
                "stdout": str(task_result.stdout_path.resolve()),
                "log_lammps": str(task_result.log_path.resolve()),
                "traj_path": str(task_result.traj_path.resolve()),
                "model_devi_out": str(task_result.model_devi_path.resolve()),
                "wall_time_sec": float(task_result.wall_time_sec),
                "group_id": task_result.group_id,
                "executor_backend": task_result.executor_backend,
            }
            jobs_payload.append(payload)
        return SamplingResult(
            tasks=list(tasks),
            task_results=list(task_results),
            jobs_payload=jobs_payload,
            tasks_manifest_path=self.tasks_manifest_path,
        )

    def _run_tasks_inline(self, tasks: list[SamplingTask]) -> list[SamplingTaskResult]:
        max_parallel_tasks = int(self.execution_plan.max_parallel_tasks)
        if max_parallel_tasks <= 1 or len(tasks) <= 1:
            results = []
            for task in tasks:
                result = self._run_one_task(task)
                results.append(result)
                self.progress_logger.record_result(result)
            self.progress_logger.finalize(results)
            return results

        results_by_job_id: dict[str, SamplingTaskResult] = {}
        with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            future_map = {executor.submit(self._run_one_task, task): task.job_id for task in tasks}
            for future in as_completed(future_map):
                job_id = future_map[future]
                result = future.result()
                results_by_job_id[job_id] = result
                self.progress_logger.record_result(result)
        ordered_results = [results_by_job_id[task.job_id] for task in tasks]
        self.progress_logger.finalize(ordered_results)
        return ordered_results

    def _run_tasks_grouped_shell(self, tasks: list[SamplingTask]) -> list[SamplingTaskResult]:
        groups = self._build_task_groups(tasks)
        if not groups:
            self.progress_logger.finalize([])
            return []

        max_parallel_groups = int(self.execution_plan.max_parallel_groups)
        if max_parallel_groups <= 0:
            max_parallel_groups = len(groups)

        group_runtime_by_id: dict[str, dict[str, Any]] = {}
        self.progress_logger.start_task_polling(tasks, self._task_artifact_paths)
        try:
            with ThreadPoolExecutor(max_workers=max_parallel_groups) as executor:
                future_map = {executor.submit(self._run_one_group, group): group.group_id for group in groups}
                for future in as_completed(future_map):
                    group_id = future_map[future]
                    runtime = future.result()
                    group_runtime_by_id[group_id] = runtime
                    if self.execution_plan.group_fail_fast and int(runtime.get("returncode", 0)) != 0:
                        raise RuntimeError(
                            f"grouped_shell execution failed for {group_id}; see {runtime.get('stdout_path')}"
                        )

            results_by_task_id: dict[str, SamplingTaskResult] = {}
            for group in groups:
                group_runtime = group_runtime_by_id.get(group.group_id, {})
                for task in group.tasks:
                    result = self._collect_grouped_task_result(task, group_id=group.group_id, group_runtime=group_runtime)
                    results_by_task_id[task.task_id] = result
            ordered_results = [results_by_task_id[task.task_id] for task in tasks]
            self.progress_logger.finalize(ordered_results)
            return ordered_results
        except Exception:
            self.progress_logger.stop_task_polling()
            raise

    def _build_task_groups(self, tasks: list[SamplingTask]) -> list[TaskGroup]:
        groups_dir = self.work_root / "groups"
        groups_dir.mkdir(parents=True, exist_ok=True)
        task_group_size = int(self.execution_plan.task_group_size)
        groups: list[TaskGroup] = []
        for group_index, start in enumerate(range(0, len(tasks), task_group_size)):
            chunk = list(tasks[start : start + task_group_size])
            group_id = f"group.{group_index:06d}"
            group_dir = groups_dir / group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            group = TaskGroup(
                group_id=group_id,
                group_index=group_index,
                group_dir=group_dir,
                script_path=group_dir / "run_group.sh",
                stdout_path=group_dir / "group.stdout",
                runtime_path=group_dir / "group_runtime.json",
                manifest_path=group_dir / "group_manifest.json",
                tasks=chunk,
            )
            script_text = self._build_group_script(group)
            group.script_path.write_text(script_text, encoding="utf-8", newline="\n")
            try:
                group.script_path.chmod(0o755)
            except OSError:
                pass
            write_json(
                group.manifest_path,
                {
                    "group_id": group.group_id,
                    "group_index": int(group.group_index),
                    "task_ids": [task.task_id for task in group.tasks],
                    "job_ids": [task.job_id for task in group.tasks],
                    "execution": _serialize_execution_plan(self.execution_plan),
                },
            )
            groups.append(group)
        return groups

    def _build_group_script(self, group: TaskGroup) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -u",
            "GROUP_FAIL=0",
            "",
        ]
        batch_size = max(1, int(self.execution_plan.group_parallel_degree))
        for batch_index, start in enumerate(range(0, len(group.tasks), batch_size)):
            batch = list(group.tasks[start : start + batch_size])
            use_background = len(batch) > 1
            if use_background:
                lines.append(f"# batch {batch_index}")
                lines.append("batch_pids=\"\"")
            for local_offset, task in enumerate(batch):
                task_paths = self._task_artifact_paths(task)
                global_task_index = group.group_index * int(self.execution_plan.task_group_size) + start + local_offset
                cuda_device = self._resolve_task_cuda_device(global_task_index)
                lines.extend(
                    self._build_group_task_block(
                        task=task,
                        start_path=task_paths["start_marker"],
                        end_path=task_paths["end_marker"],
                        returncode_path=task_paths["returncode_marker"],
                        cuda_device=cuda_device,
                        background=use_background,
                    )
                )
                if use_background:
                    lines.append("batch_pids=\"$batch_pids $!\"")
                else:
                    lines.append("if [ $? -ne 0 ]; then GROUP_FAIL=1; fi")
                    lines.append("")
            if use_background:
                lines.extend(
                    [
                        "for pid in $batch_pids; do",
                        "  if ! wait \"$pid\"; then",
                        "    GROUP_FAIL=1",
                        "  fi",
                        "done",
                        "",
                    ]
                )
        lines.append("exit \"$GROUP_FAIL\"")
        return "\n".join(lines) + "\n"

    def _build_group_task_block(
        self,
        *,
        task: SamplingTask,
        start_path: Path,
        end_path: Path,
        returncode_path: Path,
        cuda_device: str | None,
        background: bool,
    ) -> list[str]:
        lines = [
            "(",
            "  code=0",
            f"  printf '%s\\n' \"$(date +%s.%N)\" > {shlex.quote(str(start_path))}",
            f"  if cd {shlex.quote(str(task.task_dir.resolve()))}; then",
        ]
        if cuda_device is not None:
            lines.append(f"    export CUDA_VISIBLE_DEVICES={shlex.quote(str(cuda_device))}")
        lines.extend(
            [
                f"    {shlex.quote(str(task.lammps_command))} -in input.lammps > lammps.stdout 2>&1",
                "    code=$?",
                "  else",
                "    code=97",
                "  fi",
                f"  printf '%s\\n' \"$code\" > {shlex.quote(str(returncode_path))}",
                f"  printf '%s\\n' \"$(date +%s.%N)\" > {shlex.quote(str(end_path))}",
                "  exit \"$code\"",
                f"){ ' &' if background else ''}".rstrip(),
            ]
        )
        return lines

    def _run_one_group(self, group: TaskGroup) -> dict[str, Any]:
        env = build_lammps_env(self.lammps_cfg)
        started_at = time.perf_counter()
        with open(group.stdout_path, "w", encoding="utf-8") as out:
            proc = subprocess.run(
                [str(self.execution_plan.shell_command), str(group.script_path.name)],
                cwd=group.group_dir,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        wall_time_sec = float(time.perf_counter() - started_at)
        runtime = {
            "group_id": group.group_id,
            "group_index": int(group.group_index),
            "executor_backend": self.execution_plan.backend,
            "returncode": int(proc.returncode),
            "stdout_path": str(group.stdout_path.resolve()),
            "script_path": str(group.script_path.resolve()),
            "wall_time_sec": float(wall_time_sec),
            "task_ids": [task.task_id for task in group.tasks],
        }
        write_json(group.runtime_path, runtime)
        return runtime

    def _task_artifact_paths(self, task: SamplingTask) -> dict[str, Path]:
        return {
            "stdout": task.task_dir / "lammps.stdout",
            "log": task.task_dir / "log.lammps",
            "model_devi": task.task_dir / "model_devi.out",
            "traj": task.task_dir / "traj.lammpstrj",
            "runtime": task.task_dir / "task_runtime.json",
            "start_marker": task.task_dir / ".task_started_at",
            "end_marker": task.task_dir / ".task_finished_at",
            "returncode_marker": task.task_dir / ".task_returncode",
        }

    def _resolve_task_cuda_device(self, task_index: int) -> str | None:
        if self.execution_plan.device_policy == "cpu":
            return None
        if not self.execution_plan.cuda_device_pool:
            return None
        return str(self.execution_plan.cuda_device_pool[task_index % len(self.execution_plan.cuda_device_pool)])

    def _derive_task_status(self, *, returncode: int, stdout_path: Path, model_devi_path: Path) -> str:
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore") if stdout_path.exists() else ""
        if returncode != 0:
            return "failed"
        if "unrecognized pair style 'deepmd'" in stdout_text.lower():
            return "failed"
        if not model_devi_path.exists() or model_devi_path.stat().st_size == 0:
            return "failed"
        return "finished"

    def _persist_task_result(
        self,
        *,
        task: SamplingTask,
        returncode: int,
        wall_time_sec: float,
        group_id: str | None,
        executor_backend: str,
    ) -> SamplingTaskResult:
        paths = self._task_artifact_paths(task)
        status = self._derive_task_status(
            returncode=int(returncode),
            stdout_path=paths["stdout"],
            model_devi_path=paths["model_devi"],
        )
        result = SamplingTaskResult(
            task_id=task.task_id,
            job_id=task.job_id,
            task_dir=task.task_dir,
            status=status,
            returncode=int(returncode),
            stdout_path=paths["stdout"],
            log_path=paths["log"],
            model_devi_path=paths["model_devi"],
            traj_path=paths["traj"],
            wall_time_sec=float(wall_time_sec),
            group_id=group_id,
            executor_backend=executor_backend,
        )
        write_json(
            paths["runtime"],
            {
                "task_id": task.task_id,
                "job_id": task.job_id,
                "status": status,
                "returncode": int(returncode),
                "stdout": str(paths["stdout"].resolve()),
                "log_lammps": str(paths["log"].resolve()),
                "traj_path": str(paths["traj"].resolve()),
                "model_devi_out": str(paths["model_devi"].resolve()),
                "wall_time_sec": float(wall_time_sec),
                "group_id": group_id,
                "executor_backend": executor_backend,
            },
        )
        return result

    def _run_one_task(self, task: SamplingTask) -> SamplingTaskResult:
        env = build_lammps_env(self.lammps_cfg)
        paths = self._task_artifact_paths(task)
        started_at = time.perf_counter()
        paths["start_marker"].write_text(str(time.time()), encoding="utf-8")
        with open(paths["stdout"], "w", encoding="utf-8") as out:
            proc = subprocess.run(
                [str(task.lammps_command), "-in", "input.lammps"],
                cwd=task.task_dir,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        wall_time_sec = float(time.perf_counter() - started_at)
        paths["end_marker"].write_text(str(time.time()), encoding="utf-8")
        paths["returncode_marker"].write_text(str(int(proc.returncode)), encoding="utf-8")
        return self._persist_task_result(
            task=task,
            returncode=int(proc.returncode),
            wall_time_sec=wall_time_sec,
            group_id=None,
            executor_backend="inline",
        )

    def _collect_grouped_task_result(
        self,
        task: SamplingTask,
        *,
        group_id: str,
        group_runtime: dict[str, Any],
    ) -> SamplingTaskResult:
        paths = self._task_artifact_paths(task)
        returncode = int(group_runtime.get("returncode", 1))
        if paths["returncode_marker"].exists():
            try:
                returncode = int(paths["returncode_marker"].read_text(encoding="utf-8").strip())
            except ValueError:
                returncode = int(group_runtime.get("returncode", 1))

        wall_time_sec = 0.0
        if paths["start_marker"].exists() and paths["end_marker"].exists():
            try:
                started_at = float(paths["start_marker"].read_text(encoding="utf-8").strip())
                ended_at = float(paths["end_marker"].read_text(encoding="utf-8").strip())
                wall_time_sec = max(0.0, ended_at - started_at)
            except ValueError:
                wall_time_sec = 0.0

        return self._persist_task_result(
            task=task,
            returncode=returncode,
            wall_time_sec=wall_time_sec,
            group_id=group_id,
            executor_backend="grouped_shell",
        )


