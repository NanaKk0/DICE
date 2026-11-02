"""Widom selected-environment transport; no scientific model imports."""
from dataclasses import asdict
from pathlib import Path
import os
import tempfile
from workflow_backends.execution import run_operation, OperationError
from workflow_backends.common.device_fault import report_device_fault
from workflow_backends.gcmc.runtime import execute_gcmc_tasks
from workflow_backends.gcmc.state import save_state
from workflow_backends.gcmc.stage_scope import single_sample_writer
from workflow_backends.widom.adapter import WidomSamplingNativeOps
from workflow_core.structures.formats import write_extxyz_structures
from workflow_core.structures.canonical_identity import stamp_canonical_identity


class SubprocessWidomSamplingNativeOps(WidomSamplingNativeOps):
    def __init__(self, context, *, guest_sources, type_map, executor):
        super().__init__(provider=None, guest_sources=guest_sources, type_map=type_map)
        self.context = context; self.executor = executor

    @single_sample_writer
    def execute(self, request):
        evidence = request.evidence
        prepared = self.prepare(evidence)
        if self.context is None: raise ValueError('read-only widom adapter cannot execute')
        if self.context.role not in {'sampler.lammps.tf', 'sampler.lammps.pt'}:
            raise ValueError('widom currently supports DeepMD only')
        self.context.executable('python')
        if self.validate_current(evidence).complete: return self._execution_result(evidence)
        evidence.sample_dir.mkdir(parents=True, exist_ok=True)
        (evidence.sample_dir/'tasks').mkdir(exist_ok=True)
        write_extxyz_structures(evidence.sample_dir/'seeds.xyz', [stamp_canonical_identity(
            s.frames[0], system_id=s.system_id, structure_source='workflow') for s in evidence.systems])
        for task, _, _ in prepared:
            dest = evidence.sample_dir/'tasks'/task.task_id
            if dest.exists():
                old = Path(tempfile.mkdtemp(prefix='.previous-widom-', dir=evidence.sample_dir))
                os.replace(dest, old/task.task_id)
        work = Path(tempfile.mkdtemp(prefix='.widom-worker-', dir=evidence.sample_dir))
        def execute(task):
            directory = work/task.task_id; directory.mkdir()
            path = directory/'request.json'
            save_state(path, dict(schema='dice.native.widom-request', revision=1,
                operation_role=self.context.role, iter=evidence.iter_index, round=evidence.round_index,
                staging_root=str(directory/'result'), sample_dir=str(evidence.sample_dir), models=[str(p) for p in evidence.model_paths],
                member_ids=list(evidence.models.member_ids), type_map=self.type_map,
                task=asdict(task), seed=evidence.seed_mapping[task.sampling_role_id][0],
                guest_path=str(self.guest_sources[task.condition_group_id])))
            try:
                run_operation(self.context, 'python', ('-m', 'workflow_backends.widom.worker', '--manifest', str(path)),
                    cwd=directory, diagnostic_name='widom')
            except OperationError as exc:
                report_device_fault(exc.result.stderr)
                _, seed, guests = next(item for item in prepared if item[0] == task)
                self.publish_process_failure(evidence, task, seed, guests, directory/'result', exc)
            output = evidence.sample_dir/'tasks'/task.task_id/'widom_result.json'
            if output.is_file():
                import json
                failure = json.loads(output.read_text()).get('failure')
                if failure: report_device_fault(failure['message'])
        execute_gcmc_tasks(tuple(t for t, _, _ in prepared), context=self.context, executor=self.executor,
                           root=evidence.sample_dir, execute=execute)
        return self._execution_result(evidence)
