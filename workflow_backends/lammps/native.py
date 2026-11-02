from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Mapping

from workflow_backends.execution import run_operation
from workflow_backends.lammps.progress import LammpsSamplingProgressProbe
from workflow_backends.lammps.mps import CudaMpsSession
from workflow_backends.lammps.execution_mode import (
    command_for_mode,
    select_lammps_execution_mode,
)
from workflow_runtime.lifecycle.environment import (
    LammpsExecutionPolicy,
    OperationContext,
)
from workflow_runtime.lifecycle.progress import OperationProgressReporter, SamplingSnapshot


@dataclass(frozen=True, slots=True)
class NativeLammpsExecutor:
    """Run one lifecycle LAMMPS task through its selected ordinary-DeepMD profile."""

    context: OperationContext

    @property
    def execution_policy(self) -> LammpsExecutionPolicy:
        return self.context.lammps_execution_policy or LammpsExecutionPolicy(1, 100)

    def open_mps_session(self, sample_dir: Path) -> AbstractContextManager[Mapping[str, str]]:
        return CudaMpsSession(self.context, self.execution_policy, Path(sample_dir))

    @contextmanager
    def open_mps_device_sessions(
        self,
        sample_dir: Path,
        devices: tuple[str, ...],
    ):
        if (
            not devices
            or len(devices) != self.context.allocation.gpus_per_node
            or len(set(devices)) != len(devices)
        ):
            raise ValueError("LAMMPS MPS device-session inventory is invalid")
        CudaMpsSession.preflight_device_collection(self.context, devices)
        with ExitStack() as stack:
            environments: dict[str, Mapping[str, str]] = {}
            for logical_gpu, device in enumerate(devices):
                session = self._open_mps_device_session(
                    sample_dir,
                    device=device,
                    logical_gpu=logical_gpu,
                    preflight=False,
                )
                environments[device] = stack.enter_context(session)
            yield environments

    def _open_mps_device_session(
        self,
        sample_dir: Path,
        *,
        device: str,
        logical_gpu: int,
        preflight: bool,
    ) -> AbstractContextManager[Mapping[str, str]]:
        return CudaMpsSession(
            self.context,
            self.execution_policy,
            Path(sample_dir),
            device=device,
            diagnostic_namespace=f"gpu_{logical_gpu:03d}",
            preflight=preflight,
        )

    @contextmanager
    def open_recoverable_mps_device_sessions(self, sample_dir, devices):
        """GCMC owns task failure receipts; recovery only admits new tasks."""
        from workflow_backends.lammps.recovery import managed_device_session
        if not devices or len(devices) != self.context.allocation.gpus_per_node or len(set(devices)) != len(devices):
            raise ValueError('LAMMPS MPS device-session inventory is invalid')
        CudaMpsSession.preflight_device_collection(self.context, devices)
        with ExitStack() as stack:
            yield {device: stack.enter_context(managed_device_session(
                self.context, self.execution_policy, Path(sample_dir), device, index))
                for index, device in enumerate(devices)}

    def run(
        self,
        task_dir: Path,
        task: dict[str, Any],
        *,
        progress_reporter: OperationProgressReporter | None = None,
        sampling_snapshot: SamplingSnapshot | None = None,
        execution_environment: Mapping[str, str] | None = None,
    ) -> None:
        root = Path(task_dir)
        type_map = tuple(str(item) for item in task.get("type_map", ()))
        if not type_map:
            raise ValueError("lifecycle LAMMPS native task requires a type_map")
        model_count = int(task.get("model_count", 0))
        if model_count <= 0:
            raise ValueError("lifecycle LAMMPS native task requires a positive model_count")
        recipe_kind = str(task.get("deepmd_recipe_kind", "standard"))
        sampler_role = str(task.get("sampler_role", self.context.role))
        execution_mode = select_lammps_execution_mode(
            sampler_role=sampler_role,
            deepmd_recipe_kind=recipe_kind,
            model_count=model_count,
        )
        declared_mode = task.get("execution_mode")
        if declared_mode is not None and declared_mode != execution_mode:
            raise ValueError("lifecycle LAMMPS task execution mode is incompatible")
        probe = None
        if progress_reporter is not None and sampling_snapshot is not None:
            probe = LammpsSamplingProgressProbe(root / "log.lammps", base=sampling_snapshot)
        run_operation(
            self.context,
            "lmp",
            list(command_for_mode(execution_mode)[1:]),
            cwd=root,
            diagnostic_name="lammps_sample",
            progress_reporter=progress_reporter,
            progress_probe=probe,
            probe_interval_seconds=float(
                self.context.runtime.get("progress_probe_interval_seconds", 1.0)
            ),
            environment_overrides=execution_environment,
            process_registry=getattr(execution_environment, "process_registry", None),
        )
        trajectory = root / "traj.lammpstrj"
        deviation = root / "model_devi.out"
        if not trajectory.is_file() or trajectory.stat().st_size <= 0:
            raise ValueError("lifecycle LAMMPS native task did not produce traj.lammpstrj")
        if model_count > 1 and task.get('kind') != 'md_mc' and (not deviation.is_file() or deviation.stat().st_size <= 0):
            raise ValueError("lifecycle LAMMPS native task did not produce model_devi.out")

    def finalize_md_mc(self, task_dir: Path, *, failure: str | None = None):
        args = ['-m', 'workflow_backends.lammps.md_mc_evaluation', '--task-dir', '.']
        if failure is not None:
            args.extend(['--failure', failure])
        run_operation(self.context, 'python', args, cwd=Path(task_dir),
            diagnostic_name='md_mc_evaluation')
