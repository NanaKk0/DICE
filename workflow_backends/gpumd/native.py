from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_backends.execution import run_operation
from workflow_backends.gpumd.progress import GpumdSamplingProgressProbe
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.progress import OperationProgressReporter, SamplingSnapshot


@dataclass(frozen=True, slots=True)
class NativeGpumdExecutor:
    context: OperationContext

    def run(
        self,
        task_dir: Path,
        task: dict[str, Any],
        *,
        progress_reporter: OperationProgressReporter | None = None,
        sampling_snapshot: SamplingSnapshot | None = None,
    ) -> None:
        root = Path(task_dir)
        cadence = int(task.get("cadence", 0))
        if cadence <= 0:
            raise ValueError("lifecycle GPUMD native task requires a positive cadence")
        probe = None
        if progress_reporter is not None and sampling_snapshot is not None:
            probe = GpumdSamplingProgressProbe(
                root / "gpumd_sample.stdout.log",
                thermo_path=root / "thermo.out",
                thermo_cadence=task.get("thermo_cadence"),
                base=sampling_snapshot,
            )
        run_operation(
            self.context,
            "gpumd",
            (),
            cwd=root,
            diagnostic_name="gpumd_sample",
            progress_reporter=progress_reporter,
            progress_probe=probe,
            probe_interval_seconds=float(
                self.context.runtime.get("progress_probe_interval_seconds", 1.0)
            ),
        )
        active_out = root / "active.out"
        active_xyz = root / "active.xyz"
        if not active_out.is_file() or active_out.stat().st_size <= 0:
            raise ValueError("lifecycle GPUMD native task did not produce active.out")
        if not active_xyz.is_file() or active_xyz.stat().st_size <= 0:
            raise ValueError("lifecycle GPUMD native task did not produce active.xyz")
