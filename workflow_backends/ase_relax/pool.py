"""Explicit composition of ASE tasks with common allocation and worker owners."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import subprocess
import json

from workflow_backends.common.mps import CudaMpsSession
from workflow_backends.common.cancellation import handled_sigterm
from workflow_backends.common.persistent_worker import PersistentWorkers, WorkerLost, TaskExecutionError
from workflow_backends.common.sampling_pool import execute_sampling_tasks, sampling_visible_devices
from workflow_runtime.lifecycle.environment import activated_operation_environment


@dataclass(frozen=True)
class RelaxMpsPolicy:
    mps_enabled: bool = True
    active_thread_percentage: int = 100


@contextmanager
def open_relax_session(context, root, device):
    from workflow_backends.common.managed_session import managed_device_session
    from workflow_backends.ase_relax.model_adapters import model_adapter
    families = {'student.deepmd.tf.relax': 'deepmd-tf', 'student.deepmd.pt.relax': 'deepmd-pt'}
    adapter = model_adapter(families[context.role])

    def probe(context, environment):
        environment['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        return subprocess.run([context.executable('python'), '-c', adapter.probe_code()],
            env=environment, text=True, capture_output=True, timeout=30, check=False)

    CudaMpsSession.preflight_device_collection(context, (device,))
    with managed_device_session(context, RelaxMpsPolicy(), root, device, 0,
            session_factory=CudaMpsSession, probe_callback=probe) as session:
        yield session


def execute_relax_pool(context, manifest_path, tasks, *, retain_failure=None, managed_mps=True):
    root = manifest_path.parent
    environment = activated_operation_environment(context)
    repository = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = repository + ':' + environment.get('PYTHONPATH', '')
    threads = max(1, context.allocation.cpus_per_task // context.allocation.per_gpu)
    environment['OMP_NUM_THREADS'] = str(threads)
    environment['OPENBLAS_NUM_THREADS'] = str(threads)
    environment['MKL_NUM_THREADS'] = str(threads)
    device, = sampling_visible_devices(1)
    command = [context.executable('python'), '-m', 'workflow_backends.ase_relax.worker',
               '--manifest', str(manifest_path), '--channel']
    from workflow_backends.common.sampling_pool import DeviceSessionEnvironment
    from workflow_backends.execution import ProcessGroupRegistry
    scope = (open_relax_session(context, root, device) if managed_mps else nullcontext(
        DeviceSessionEnvironment(device, device, {}, ProcessGroupRegistry())))
    with handled_sigterm(), scope as session:
        manifest = json.loads(manifest_path.read_text())
        identity = {'family': manifest['model_family'], 'models': manifest['models']}
        with PersistentWorkers(command, environment=environment, cwd=root, expected_identity=identity) as workers:
            def execute(task):
                from workflow_backends.common.device_fault import report_device_fault
                try:
                    result = workers.execute(task.task_id)
                except (WorkerLost, TaskExecutionError) as error:
                    report_device_fault(str(error))
                    if retain_failure is None:
                        raise
                    return retain_failure(task, error)
                if result.get('status') not in {'completed', 'failed'}:
                    raise ValueError('invalid persistent relax outcome')
                if result['status'] == 'failed':
                    report_device_fault(result.get('diagnostic', ''))
                return result
            return execute_sampling_tasks(tasks, context=context, task_id=lambda task: task.task_id,
                execute=execute, allow_per_gpu=True,
                assignment_environment=lambda assignment: session)
