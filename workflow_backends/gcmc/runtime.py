"""Selected-environment worker transport, outside scientific/core intent."""
from dataclasses import asdict, replace
from contextlib import nullcontext
from pathlib import Path
import tempfile

from workflow_backends.execution import run_operation, operation_execution_scope, ProcessGroupRegistry, OperationError
from workflow_backends.common.sampling_pool import execute_sampling_tasks, sampling_visible_devices
from workflow_backends.common.device_fault import report_device_fault
from workflow_backends.gcmc.adapter import GcmcSamplingNativeOps, archive_incomplete_sample_tasks, publish_failed_task, present_task_models
from workflow_backends.gcmc.state import save_state
from workflow_backends.gcmc.stage_scope import single_sample_writer
from workflow_core.structures.canonical_identity import stamp_canonical_identity
from workflow_core.structures.formats import write_extxyz_structures


def execute_gcmc_tasks(tasks, *, context, executor, root, execute):
    """Use the shared pool and the selected executor's MPS ownership boundary."""
    from workflow_runtime.execution.allocation import Allocation
    policy = context.lammps_execution_policy
    if policy is not None and policy.parallel_tasks > 1 and context.allocation == Allocation.local_default():
        context = replace(context, allocation=replace(context.allocation, per_gpu=policy.parallel_tasks))
    if context.allocation.per_gpu != 1:
        open_sessions = getattr(executor, 'open_recoverable_mps_device_sessions', None)
        if not callable(open_sessions):
            open_sessions = getattr(executor, 'open_mps_device_sessions', None)
        if not callable(open_sessions):
            raise ValueError('GCMC per_gpu > 1 requires the selected MPS device-session capability')
        devices = sampling_visible_devices(context.allocation.gpus_per_node)
        with open_sessions(root, devices) as sessions:
            if set(sessions) != set(devices):
                raise ValueError('GCMC MPS session devices differ from the allocation')
            return execute_sampling_tasks(tasks, context=context, task_id=lambda t: t.task_id,
                execute=execute, allow_per_gpu=True,
                assignment_environment=lambda assignment: sessions[assignment.visible_device])
    registry = ProcessGroupRegistry()
    scope = (operation_execution_scope(environment_overrides={
                'CUDA_VISIBLE_DEVICES': sampling_visible_devices(1)[0]}, process_registry=registry)
             if context.allocation.gpus_per_node == 1 else nullcontext())
    try:
        with scope:
            return execute_sampling_tasks(tasks, context=context, task_id=lambda t: t.task_id, execute=execute)
    finally:
        registry.cancel_all()


class SubprocessGcmcSamplingNativeOps(GcmcSamplingNativeOps):
    def __init__(self, context, *, guest_sources, type_map, md_recipes=None, executor=None):
        super().__init__(calculator_provider=None, guest_sources=guest_sources,
                         gcmc_class=None, device='cpu')
        self.context = context
        self.type_map = tuple(type_map)
        self.md_recipes = dict(md_recipes or {})
        self.executor = executor

    @single_sample_writer
    def execute(self, request):
        if self.context is None:
            raise ValueError('read-only GCMC adapter cannot execute')
        if self.context.role not in {'sampler.lammps.tf', 'sampler.lammps.pt'}:
            raise ValueError('GCMC currently requires a selected DeepMD/LAMMPS context')
        self.context.executable('python')
        evidence = request.evidence
        tasks = self._tasks(evidence)
        if not tasks:
            raise ValueError('GCMC worker requires active tasks')
        md_transport = None
        if any(t.gcmc.md is not None for t in tasks):
            from workflow_artifacts.lifecycle.sampling_models import EffectiveSamplingModelPlan
            from workflow_backends.lammps.execution_mode import select_lammps_execution_mode
            if not isinstance(evidence.models, EffectiveSamplingModelPlan):
                raise ValueError('hybrid MD requires the resolved typed DeepMD model plan')
            kinds = {m.deepmd_recipe_kind for m in evidence.models.models}
            if len(kinds) != 1:
                raise ValueError('hybrid MD model recipe kinds must agree')
            md_transport = dict(executable=str(self.context.executable('lmp')),
                mode=select_lammps_execution_mode(sampler_role=self.context.role,
                    deepmd_recipe_kind=kinds.pop(), model_count=len(evidence.model_paths)),
                recipes={key: str(path) for key, path in self.md_recipes.items()})
        if self.validate_current(evidence).complete:
            return self._execution_result(evidence)
        if md_transport is not None:
            from workflow_backends.gcmc.lammps import validate_recipe
            for task in tasks:
                if task.gcmc.md is not None:
                    validate_recipe(self.md_recipes[task.condition_group_id], task.gcmc.md, task.temperature_k)
        evidence.sample_dir.mkdir(parents=True, exist_ok=True)
        archive_incomplete_sample_tasks(evidence.sample_dir)
        write_extxyz_structures(evidence.sample_dir/'seeds.xyz', [stamp_canonical_identity(
            source.frames[0], system_id=source.system_id, structure_source='workflow') for source in evidence.systems])
        work = Path(tempfile.mkdtemp(prefix='.gcmc-worker-', dir=evidence.sample_dir))
        payload = dict(schema='dice.native.gcmc-request', revision=1,
            run_json_path=str(evidence.run_json_path), iter=evidence.iter_index,
            round=evidence.round_index, sample_dir=str(evidence.sample_dir),
            models=[str(p) for p in evidence.model_paths], member_ids=list(evidence.models.member_ids),
            type_map=self.type_map, device='cuda' if self.context.allocation.gpus_per_node else 'cpu',
            operation_role=self.context.role, operation_profile_id=self.context.profile_id)
        if md_transport is not None:
            payload['md_transport'] = md_transport
        def execute_task(task):
            frames = evidence.seed_mapping[task.sampling_role_id]
            if len(frames) != 1:
                raise ValueError('GCMC worker requires one resolved frame per role')
            task_root = work/task.task_id
            task_root.mkdir()
            manifest = task_root/'request.json'
            save_state(manifest, dict(payload, tasks=[dict(task=asdict(task), seed=frames[0],
                                  guest_path=str(self.guest_sources[task.condition_group_id]))]))
            try:
                run_operation(self.context, 'python', ('-m', 'workflow_backends.gcmc.worker', '--manifest', str(manifest)),
                              cwd=task_root, diagnostic_name='gcmc')
            except OperationError as exc:
                report_device_fault(exc.result.stderr)
                publish_failed_task(evidence, task, exc)
                present_task_models(evidence, task, evidence.model_paths)
            finally:
                # Workers may publish FAILED yet exit zero. Native MD stderr is
                # redirected separately, so inspect it as well as worker stderr.
                output = evidence.sample_dir / 'tasks' / task.task_id
                paths = [task_root/'gcmc.stderr.log', output/'gcmc_result.json']
                paths.extend((output/'failed-attempt-diagnostics').glob('md_*/native.stderr.log'))
                for path in paths:
                    if path.is_file():
                        with path.open('rb') as handle:
                            report_device_fault(handle.read(65536).decode('utf-8', errors='replace'))
                            handle.seek(max(0, path.stat().st_size - 65536))
                            report_device_fault(handle.read(65536).decode('utf-8', errors='replace'))
        execute_gcmc_tasks(tasks, context=self.context, executor=self.executor,
                          root=evidence.sample_dir, execute=execute_task)
        return self._execution_result(evidence)
