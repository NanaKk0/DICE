from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from typing import Protocol

from workflow_runtime.lifecycle.environment import (
    OperationContext,
    activated_operation_environment,
)
from workflow_backends.execution import ProcessGroupRegistry
from workflow_backends.common.device_environment import DeviceSessionEnvironment


_MAX_DIAGNOSTIC_FILES = 8
_MAX_DIAGNOSTIC_BYTES = 1024 * 1024


class MpsPolicy(Protocol):
    mps_enabled: bool
    active_thread_percentage: int


@dataclass(slots=True)
class CudaMpsSession:
    """Own one isolated CUDA MPS daemon for one lifecycle block attempt."""

    context: OperationContext
    policy: MpsPolicy
    sample_dir: Path
    device: str | None = None
    diagnostic_namespace: str | None = None
    preflight: bool = True
    _control: str | None = field(default=None, init=False)
    _environment: dict[str, str] | None = field(default=None, init=False)
    _transient_root: Path | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _process_registry: ProcessGroupRegistry = field(
        default_factory=ProcessGroupRegistry,
        init=False,
    )

    @classmethod
    def preflight_device_collection(
        cls,
        context: OperationContext,
        devices: tuple[str, ...],
    ) -> None:
        if (
            not devices
            or len(set(devices)) != len(devices)
            or any(not device or "," in device for device in devices)
        ):
            raise ValueError("CUDA MPS device collection is invalid")
        activated = activated_operation_environment(context)
        pgrep = cls._resolve_executable(context, activated, "pgrep")
        existing = subprocess.run(
            [pgrep, "-f", "^([^ ]*/)?nvidia-cuda-mps-(control|server)( |$)"],
            env=activated,
            text=True,
            capture_output=True,
            check=False,
        )
        if existing.returncode == 0 and existing.stdout.strip():
            raise ValueError(
                f"lifecycle operation {context.role!r} found a pre-existing CUDA MPS daemon/server; "
                "refusing to adopt it"
            )
        if existing.returncode not in {0, 1}:
            raise ValueError(
                f"lifecycle operation {context.role!r} could not inspect existing CUDA MPS processes"
            )
        nvidia_smi = cls._resolve_executable(context, activated, "nvidia-smi")
        for device in devices:
            device_environment = dict(activated)
            device_environment["CUDA_VISIBLE_DEVICES"] = device
            active = subprocess.run(
                [
                    nvidia_smi,
                    "--id", device,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                env=device_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if active.returncode != 0:
                raise ValueError(
                    f"lifecycle operation {context.role!r} could not inspect active CUDA workloads on device {device}"
                )
            pids = [line.strip() for line in active.stdout.splitlines() if line.strip()]
            if pids:
                raise ValueError(
                    f"lifecycle operation {context.role!r} found incompatible active CUDA compute workload(s) "
                    f"on device {device}: " + ", ".join(pids[:8])
                )

    def __enter__(self) -> DeviceSessionEnvironment:
        if not self.policy.mps_enabled:
            raise ValueError("CUDA MPS session requires parallel_tasks greater than one")
        activated = activated_operation_environment(self.context)
        if self.device is not None:
            if not self.device or "," in self.device:
                raise ValueError("CUDA MPS device session requires one logical GPU")
            activated["CUDA_VISIBLE_DEVICES"] = self.device
        assigned_visible_device = self._preflight_visible_gpu(activated)
        self._control = self._resolve(activated, "nvidia-cuda-mps-control")
        if self.preflight:
            pgrep = self._resolve(activated, "pgrep")
            existing = subprocess.run(
                [pgrep, "-f", "^([^ ]*/)?nvidia-cuda-mps-(control|server)( |$)"],
                env=activated,
                text=True,
                capture_output=True,
                check=False,
            )
            if existing.returncode == 0 and existing.stdout.strip():
                raise ValueError(
                    f"lifecycle operation {self.context.role!r} found a pre-existing CUDA MPS daemon/server; "
                    "refusing to adopt it"
                )
            if existing.returncode not in {0, 1}:
                raise ValueError(f"lifecycle operation {self.context.role!r} could not inspect existing CUDA MPS processes")
            self._reject_active_compute_workload(activated)

        self._transient_root = Path(tempfile.mkdtemp(prefix="dice-mps-"))
        log_dir = self._transient_root / "logs"
        log_dir.mkdir()
        overrides = {
            "CUDA_MPS_PIPE_DIRECTORY": str(self._transient_root),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(self.policy.active_thread_percentage),
        }
        daemon_environment = dict(activated)
        daemon_environment.update(overrides)
        self._environment = daemon_environment
        try:
            started = subprocess.run(
                [self._control, "-d"],
                env=daemon_environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            # The daemon may have started even though its launcher timed out.
            # Resolve only this owned pipe; retain it if quit cannot be proved.
            self._started = True
            self.__exit__(None, None, None)
            raise
        self._write_control_diagnostic("start", started)
        if started.returncode != 0:
            detail = started.stderr.strip()
            try:
                self._retain_logs()
            finally:
                self._cleanup_transient()
            raise ValueError(
                f"lifecycle operation {self.context.role!r} could not start its CUDA MPS daemon"
                + (f": {detail[-1200:]}" if detail else "")
            )
        self._started = True
        return DeviceSessionEnvironment(
            assigned_visible_device=assigned_visible_device,
            client_visible_device="0",
            overrides=overrides,
            process_registry=self._process_registry,
        )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        stop_error: Exception | None = None
        try:
            if exc is not None:
                self._process_registry.cancel_all()
            if self._started and self._control is not None and self._environment is not None:
                stopped = subprocess.run(
                    [self._control],
                    input="quit\n",
                    env=self._environment,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=20,
                )
                self._write_control_diagnostic("stop", stopped)
                if stopped.returncode != 0:
                    stop_error = RuntimeError(
                        f"lifecycle operation {self.context.role!r} could not stop its CUDA MPS daemon"
                    )
        except Exception as error:
            stop_error = error
        finally:
            try:
                self._retain_logs()
            finally:
                # A failed quit leaves ownership uncertain: preserve the pipe
                # and refuse to launch a replacement daemon.
                if stop_error is None:
                    self._cleanup_transient()
                    self._started = False
        if stop_error is not None:
            if exc is None:
                raise stop_error
            raise RuntimeError(f"{exc}; MPS teardown also failed: {stop_error}") from exc
        return False

    def _preflight_visible_gpu(self, activated: Mapping[str, str]) -> str:
        raw = activated.get("CUDA_VISIBLE_DEVICES")
        if raw is not None:
            devices = [item.strip() for item in raw.split(",") if item.strip()]
            if len(devices) != 1 or devices[0] in {"-1", "NoDevFiles", "void"}:
                raise ValueError(
                    f"lifecycle operation {self.context.role!r} CUDA MPS requires exactly one visible GPU"
                )
            return devices[0]
        nvidia_smi = self._resolve(activated, "nvidia-smi")
        visible = subprocess.run(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader,nounits"],
            env=dict(activated),
            text=True,
            capture_output=True,
            check=False,
        )
        devices = [line.strip() for line in visible.stdout.splitlines() if line.strip()]
        if visible.returncode != 0 or len(devices) != 1:
            raise ValueError(
                f"lifecycle operation {self.context.role!r} CUDA MPS requires exactly one visible GPU"
            )
        return devices[0]

    def _reject_active_compute_workload(self, activated: Mapping[str, str]) -> None:
        nvidia_smi = self._resolve(activated, "nvidia-smi")
        active = subprocess.run(
            [nvidia_smi, "--id", self._preflight_visible_gpu(activated),
             "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            env=dict(activated),
            text=True,
            capture_output=True,
            check=False,
        )
        if active.returncode != 0:
            raise ValueError(f"lifecycle operation {self.context.role!r} could not inspect active CUDA workloads")
        pids = [line.strip() for line in active.stdout.splitlines() if line.strip()]
        if pids:
            raise ValueError(
                f"lifecycle operation {self.context.role!r} found incompatible active CUDA compute workload(s): "
                + ", ".join(pids[:8])
            )

    def _resolve(self, activated: Mapping[str, str], executable: str) -> str:
        return self._resolve_executable(self.context, activated, executable)

    @staticmethod
    def _resolve_executable(
        context: OperationContext,
        activated: Mapping[str, str],
        executable: str,
    ) -> str:
        resolved = shutil.which(executable, path=activated.get("PATH"))
        if resolved is None:
            raise ValueError(
                f"lifecycle operation {context.role!r} requires {executable!r} after profile activation"
            )
        return resolved

    def _retain_logs(self) -> None:
        if self._transient_root is None:
            return
        source = self._transient_root / "logs"
        if not source.is_dir():
            return
        target = self.sample_dir / "backend_logs" / "mps"
        if self.diagnostic_namespace is not None:
            target = target / self.diagnostic_namespace
        retained = 0
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            if retained >= _MAX_DIAGNOSTIC_FILES:
                break
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as input_handle, destination.open("wb") as output_handle:
                if path.stat().st_size <= _MAX_DIAGNOSTIC_BYTES:
                    output_handle.write(input_handle.read(_MAX_DIAGNOSTIC_BYTES))
                else:
                    output_handle.write(input_handle.read(_MAX_DIAGNOSTIC_BYTES // 2))
                    output_handle.write(b'\n[DICE: middle truncated; tail follows]\n')
                    input_handle.seek(-_MAX_DIAGNOSTIC_BYTES // 2, 2)
                    output_handle.write(input_handle.read(_MAX_DIAGNOSTIC_BYTES // 2))
            retained += 1

    def _write_control_diagnostic(
        self,
        phase: str,
        result: subprocess.CompletedProcess[str],
    ) -> None:
        if self._transient_root is None:
            return
        log_dir = self._transient_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        text = (
            f"returncode={result.returncode}\n"
            f"stdout:\n{result.stdout[-8192:]}\n"
            f"stderr:\n{result.stderr[-8192:]}\n"
        )
        (log_dir / f"control-{phase}.log").write_text(text, encoding="utf-8")

    def _cleanup_transient(self) -> None:
        if self._transient_root is not None:
            shutil.rmtree(self._transient_root, ignore_errors=True)
            self._transient_root = None
