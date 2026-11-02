"""Explicit backend capability for an allocation-local training pool."""
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
import subprocess
from pathlib import Path
from typing import Callable, Mapping

from workflow_backends.common.device_environment import DeviceSessionEnvironment
from workflow_backends.common.mps import CudaMpsSession
from workflow_runtime.lifecycle.environment import OperationContext, activated_operation_environment


@dataclass(frozen=True, slots=True)
class TrainingPoolCapability:
    """Backend-selected session provider; no model/recipe knowledge in scheduling."""
    open_sessions: Callable[
        [OperationContext, Path, tuple[str, ...]],
        AbstractContextManager[Mapping[str, DeviceSessionEnvironment]],
    ]


@dataclass(frozen=True, slots=True)
class TrainingMpsPolicy:
    active_thread_percentage: int
    mps_enabled: bool = True


@contextmanager
def owned_training_sessions(context: OperationContext, root: Path, devices: tuple[str, ...]):
    environment = activated_operation_environment(context)
    raw = environment.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '')
    if raw and (not raw.isdigit() or not 1 <= int(raw) <= 100):
        raise ValueError('training CUDA_MPS_ACTIVE_THREAD_PERCENTAGE must be an integer from 1 through 100')
    policy = TrainingMpsPolicy(100 if not raw else int(raw))
    # Device aliases (for example 0 and its UUID) must not create two sessions
    # for the same physical GPU. nvidia-smi ignores CUDA_VISIBLE_DEVICES.
    smi = CudaMpsSession._resolve_executable(context, environment, "nvidia-smi")
    physical = []
    for device in devices:
        result = subprocess.run([smi, "--id", device, "--query-gpu=uuid",
            "--format=csv,noheader,nounits"], env=environment, text=True,
            capture_output=True, timeout=20, check=False)
        uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode or len(uuids) != 1 or not uuids[0].startswith("GPU-"):
            raise ValueError(f"training MPS cannot resolve physical GPU {device!r}")
        physical.extend(uuids)
    if len(set(physical)) != len(physical):
        raise ValueError("training MPS requires distinct physical GPUs")
    CudaMpsSession.preflight_device_collection(context, devices)
    with ExitStack() as stack:
        sessions = {}
        for index, device in enumerate(devices):
            sessions[device] = stack.enter_context(CudaMpsSession(
                context, policy, root, device=device,
                diagnostic_namespace=f'gpu_{index:03d}', preflight=False))
        yield sessions
