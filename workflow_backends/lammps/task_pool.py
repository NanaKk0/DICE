"""LAMMPS task dispatch with executor-owned device-session lifetimes."""
from dataclasses import replace
from workflow_backends.common.sampling_pool import execute_sampling_tasks, sampling_visible_devices


def execute_lammps_tasks(tasks, *, executor, sample_dir, task_id, execute):
    context = getattr(executor, 'context', None)
    allocation = getattr(context, 'allocation', None)
    policy = getattr(executor, 'execution_policy', None)
    parallel = getattr(policy, 'parallel_tasks', 1)
    if (allocation is not None and allocation.mode == 'pool'
        and allocation.gpus_per_node == 1 and allocation.per_gpu == 1 and parallel > 1):
        opener = getattr(executor, 'open_mps_session', None)
        if not callable(opener):
            raise ValueError('LAMMPS parallel execution requires a CUDA MPS session')
        pooled = replace(context, allocation=replace(allocation, per_gpu=parallel))
        with opener(sample_dir) as environment:
            return execute_sampling_tasks(tasks, context=pooled, task_id=task_id,
                execute=execute, allow_per_gpu=True,
                assignment_environment=lambda assignment: {} if environment is None else environment)
    if allocation is None or allocation.per_gpu == 1:
        return execute_sampling_tasks(tasks, context=context, task_id=task_id, execute=execute)
    opener = getattr(executor, 'open_mps_device_sessions', None)
    if not callable(opener):
        raise ValueError('LAMMPS pool per_gpu > 1 requires an MPS device-session capability')
    devices = sampling_visible_devices(allocation.gpus_per_node)
    with opener(sample_dir, devices) as sessions:
        if set(sessions) != set(devices):
            raise ValueError('LAMMPS MPS device sessions do not match allocated devices')
        return execute_sampling_tasks(tasks, context=context, task_id=task_id,
            execute=execute, allow_per_gpu=True,
            assignment_environment=lambda assignment: sessions[assignment.visible_device])
