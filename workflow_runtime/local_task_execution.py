from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

from workflow_runtime.environment.resolution import LocalTaskExecution
from workflow_runtime.environment.shell_rendering import render_local_task_shell_script


@dataclass(frozen=True, slots=True)
class LocalTaskRunResult:
    exit_code: int
    diagnostics_dir: Path
    run_script_path: Path
    stdout_log_path: Path
    stderr_log_path: Path


class LocalTaskRunError(RuntimeError):
    def __init__(self, *, task: LocalTaskExecution, command: str, result: LocalTaskRunResult) -> None:
        self.task = task
        self.command = command
        self.exit_code = int(result.exit_code)
        self.diagnostics_dir = result.diagnostics_dir
        self.run_script_path = result.run_script_path
        self.stdout_log_path = result.stdout_log_path
        self.stderr_log_path = result.stderr_log_path
        super().__init__(
            f"local task failed for role {task.role!r} profile {task.profile_id!r}: "
            f"command {command!r} exited with exit code {self.exit_code}; "
            f"stdout: {self.stdout_log_path}; stderr: {self.stderr_log_path}; "
            f"script: {self.run_script_path}"
        )


def render_local_task_script(task: LocalTaskExecution, *, command: str, workdir: str | Path) -> str:
    return render_local_task_shell_script(
        command=command,
        workdir=workdir,
        source_list=task.environment.source_list,
    )


def run_local_task(
    task: LocalTaskExecution,
    *,
    command: str,
    workdir: str | Path,
    diagnostics_dir: str | Path,
) -> LocalTaskRunResult:
    resolved_diagnostics_dir = Path(diagnostics_dir).resolve()
    resolved_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    run_script_path = resolved_diagnostics_dir / "run.sh"
    stdout_log_path = resolved_diagnostics_dir / "task.stdout.log"
    stderr_log_path = resolved_diagnostics_dir / "task.stderr.log"

    run_script_path.write_text(render_local_task_script(task, command=command, workdir=workdir), encoding="utf-8")
    run_script_path.chmod(0o755)

    with stdout_log_path.open("w", encoding="utf-8") as stdout, stderr_log_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            ["bash", str(run_script_path)],
            cwd=str(workdir),
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
            env=os.environ.copy(),
        )
    result = LocalTaskRunResult(
        exit_code=int(completed.returncode),
        diagnostics_dir=resolved_diagnostics_dir,
        run_script_path=run_script_path,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )
    if completed.returncode != 0:
        raise LocalTaskRunError(task=task, command=command, result=result)
    return result
