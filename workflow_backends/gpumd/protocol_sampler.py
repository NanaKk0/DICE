"""Clean GPUMD protocols and their native observation transport."""
from dataclasses import asdict
from io import StringIO
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read

from workflow_backends.common.frame_index import index_extxyz_frames
from workflow_backends.common.model_files import link_model_file
from workflow_backends.common.seed_selection import publish_seed_selection_manifest
from workflow_backends.common.recipe_inputs import contained_artifact, stage_recipe_inputs
from workflow_backends.common.sampling_pool import execute_sampling_tasks
from workflow_backends.gpumd.native_protocol import compile_gpumd_protocol
from workflow_backends.common.file_integrity import file_digest as _file_digest
from workflow_backends.observation_source import ObservationLocator, ObservationRecord
from workflow_backends.sample_artifacts import canonical_selected_seed, publish_selected_seeds
from workflow_backends.sampling import SamplingExecutionResult, SamplingObservationResult, SamplingValidationResult
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


def _conditions(row):
    method = row[1] if row is not None else None
    temperature = pressure = None
    if method and method.startswith(('nvt_', 'npt_')):
        targets = [float(v) for v in row[2:4]]
        if len(targets) != 2 or not all(np.isfinite(v) and v > 0 for v in targets):
            raise ValueError('GPUMD native temperature targets must be finite and positive')
        temperature = targets[0] if targets[0] == targets[1] else None
    if method and method.startswith('npt_'):
        axes = {8: 1, 12: 3, 18: 6}.get(len(row))
        if axes is None:
            raise ValueError('GPUMD native pressure control needs axes, elastic moduli and coupling')
        values = [float(v) for v in row[5:5+min(axes, 3)]]
        if not np.isfinite(values).all():
            raise ValueError('GPUMD native pressure targets must be finite')
        pressure = values[0]*1e4 if all(value == values[0] for value in values) else None
    return method, temperature, pressure


def resolve_gpumd_protocol(text, request):
    """Resolve literal native stages; all physical commands retain their order."""
    if any(getattr(request, field) is not None for field in (
        'thermostat', 'barostat', 'group', 'temperature_damping_ps', 'pressure_damping_ps', 'pressure_coupling')):
        raise ValueError('GPUMD controller method and coupling belong in the native ensemble command')
    protocol = compile_gpumd_protocol(text, request)
    rows = [line.split('#', 1)[0].split() for line in protocol.splitlines()]
    rows = [row for row in rows if row]
    timestep, adaptive, cadence, ensemble = 1., False, 20, None
    stages, elapsed, steps, has_time, has_dump = [], 0., 0, False, False
    for row in rows:
        if row[0] == 'active':
            raise ValueError('GPUMD active is the standard DICE observation output; auxiliary native dumps remain available')
        if row[0] == 'time_step':
            if len(row) not in {2, 3}:
                raise ValueError('GPUMD native time_step requires a value and optional displacement limit')
            timestep, adaptive, has_time = float(row[1]), len(row) == 3, True
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError('GPUMD native time_step must be finite and positive')
        elif row[0] == 'dump_position':
            cadence, has_dump = int(row[1]), True
            if cadence <= 0:
                raise ValueError('GPUMD native dump_position interval must be positive')
        elif row[0] == 'ensemble':
            ensemble = row
        elif row[0] == 'run':
            if len(row) != 2 or not row[1].isdigit() or int(row[1]) <= 0:
                raise ValueError('GPUMD native observations require positive literal run budgets')
            method, temperature, pressure = _conditions(ensemble)
            budget = int(row[1])
            stages.append(dict(steps=budget, start_step=steps, start_time_fs=elapsed,
                timestep_fs=timestep, adaptive=adaptive, cadence=request.trajectory_every_steps or cadence,
                native_ensemble=method, temperature_k=temperature, pressure_bar=pressure))
            steps += budget
            elapsed = None if adaptive or elapsed is None else elapsed + budget*timestep
    if not stages:
        raise ValueError('GPUMD native observations require a run command')
    def uniform(field):
        values = [stage[field] for stage in stages]
        return values[0] if all(value == values[0] for value in values) else None
    temperature, pressure, method = (uniform(field) for field in ('temperature_k', 'pressure_bar', 'native_ensemble'))
    timestep_fs, interval = uniform('timestep_fs'), uniform('cadence')
    settings = {
        'temperature_k': dict(value=temperature, source='DICE' if request.temperature_k is not None else 'native'),
        'pressure_bar': dict(value=pressure, source='DICE' if request.pressure_bar is not None else 'native'),
        'ensemble': dict(value=method, source='DICE' if request.ensemble is not None else 'native'),
        'timestep_ps': dict(value=None if timestep_fs is None else timestep_fs/1000,
            source='DICE' if request.timestep_ps is not None else 'native' if has_time else 'default'),
        'steps': dict(value=steps, source='DICE' if request.steps is not None else 'native'),
        'trajectory_every_steps': dict(value=interval, source='DICE' if request.trajectory_every_steps is not None else 'native' if has_dump else 'default')}
    return protocol, dict(steps=steps, timestep_fs=timestep_fs, cadence=interval,
        native_ensemble=method, temperature_k=temperature, pressure_bar=pressure, stages=stages, settings=settings)


def _protocol_active_rows(root, effective):
    expected = []
    for phase, stage in enumerate(effective['stages'], 1):
        for local_step in range(stage['cadence'], stage['steps']+1, stage['cadence']):
            timestamp = (None if stage['adaptive'] or stage['start_time_fs'] is None
                         else stage['start_time_fs']+local_step*stage['timestep_fs'])
            expected.append((stage['start_step']+local_step, timestamp, phase, local_step))
    rows = []
    previous_time = 0.
    with (Path(root)/'active.out').open() as stream:
        for line in stream:
            values = line.split()
            if not values or values[0].startswith('#'):
                continue
            if len(values) < 2 or len(rows) >= len(expected):
                raise ValueError('GPUMD native uncertainty inventory is inconsistent')
            time_fs, deviation = float(values[0]), float(values[1])
            step, timestamp, phase, local_step = expected[len(rows)]
            if (not np.isfinite([time_fs, deviation]).all() or time_fs <= previous_time or deviation < 0
                or (timestamp is not None and not np.isclose(time_fs, timestamp, rtol=1e-10, atol=1e-8))):
                raise ValueError('GPUMD native time or uncertainty evidence is invalid')
            rows.append(dict(md_step=step, md_time_fs=time_fs, uncertainty=deviation,
                             phase=f'run_{phase:04d}', phase_step=local_step))
            previous_time = time_fs
    if len(rows) != len(expected):
        raise ValueError('GPUMD native observation budget is incomplete')
    return rows


def validate_gpumd_protocol_result(root, *, intent=None, seed=None, type_map=None, member_ids=None, digest_cache=None):
    root = Path(root)
    cache = {} if digest_cache is None else digest_cache
    context = json.loads((root/'task_context.json').read_text())
    result = json.loads((root/'gpumd_protocol_result.json').read_text())
    if (context.get('schema') != 'dice.gpumd-protocol-task' or context.get('revision') != 1
        or result.get('schema') != 'dice.gpumd-protocol-result' or result.get('revision') != 1
        or result.get('status') != 'completed' or context.get('task_id') != root.name
        or result.get('task_id') != root.name
        or result.get('context_digest') != _file_digest(root/'task_context.json', cache=cache)):
        raise ValueError('GPUMD native result/context is incompatible')
    if intent is not None:
        for field in ('task_id', 'sampling_role_id', 'condition_group_id', 'selection_group_id', 'source_frame_index'):
            if context[field] != getattr(intent, field):
                raise ValueError('GPUMD native task intent changed')
        if context['request'] != asdict(intent.protocol_request):
            raise ValueError('GPUMD native scientific request changed')
    if member_ids is not None and context['member_ids'] != list(member_ids):
        raise ValueError('GPUMD native model membership changed')
    if type_map is not None and context['type_map'] != list(type_map):
        raise ValueError('GPUMD native element map changed')
    required = {'seed.xyz', 'model.xyz', 'run.in', 'protocol.gpumd', 'active.xyz', 'active.out'}
    required.update(context.get('inputs', ()))
    required |= {f'nep_{i}.txt' for i in range(len(context['member_ids']))}
    if set(result['files']) != required:
        raise ValueError('GPUMD native file inventory is not exact')
    for name, digest in result['files'].items():
        if _file_digest(contained_artifact(root, name), cache=cache) != digest:
            raise ValueError('GPUMD native artifact changed')
    stored, = read_extxyz_structures(root/'seed.xyz', allow_empty=False)
    if seed is not None and exact_structure_hash(seed) != exact_structure_hash(stored):
        raise ValueError('GPUMD native seed changed')
    rows = _protocol_active_rows(root, context['effective'])
    index = index_extxyz_frames(root/'active.xyz', expected_count=len(rows))
    for raw in index.read(range(len(rows))):
        frame = read(StringIO(raw.decode('utf-8')), format='extxyz')
        if (not np.array_equal(frame.numbers, stored.numbers)
            or not np.isfinite(frame.positions).all() or not np.isfinite(frame.cell.array).all()
            or frame.cell.rank != 3 or np.linalg.det(frame.cell.array) <= 0):
            raise ValueError('GPUMD native atom inventory or geometry changed')
    if result['observations'] != rows:
        raise ValueError('GPUMD native observation evidence changed')
    return context, result, index


class GpumdProtocolObservationSource:
    failed_task_count = 0

    def __init__(self, evidence, tasks):
        self._frames, records, cache = {}, [], {}
        namespace = f'{evidence.run_json_path.resolve()}\0{evidence.iter_index}\0{evidence.round_index}\0'
        for root, intent in tasks:
            context, result, index = validate_gpumd_protocol_result(root, intent=intent, digest_cache=cache)
            task_index = int(intent.task_id.removeprefix('task_'))
            for ordinal, row in enumerate(result['observations']):
                step, deviation = row['md_step'], row['uncertainty']
                identity = 'sha256:' + hashlib.sha256((namespace+intent.task_id+'\0'+str(ordinal)).encode()).hexdigest()
                uncertainty = deviation if len(context['member_ids']) > 1 else None
                record = ObservationRecord(identity, intent.selection_group_id, task_index, ordinal, step,
                    uncertainty, ObservationLocator('GPUMD-PROTOCOL', task_index, ordinal, step),
                    sampling_role_id=intent.sampling_role_id, selection_group_id=intent.selection_group_id,
                    condition_group_id=intent.condition_group_id, task_id=intent.task_id,
                    md_time_fs=row['md_time_fs'], phase=row['phase'], phase_step=row['phase_step'])
                records.append(record)
                self._frames[identity] = (index, record)
        self._records = tuple(records)
        if not records:
            raise ValueError('GPUMD native protocol has no observations')

    def catalog(self):
        return self._records

    def replay(self, identities, *, chunk_size):
        identities = tuple(identities)
        if (type(chunk_size) is not int or chunk_size <= 0 or len(set(identities)) != len(identities)
            or any(identity not in self._frames for identity in identities)):
            raise ValueError('GPUMD native replay requires unique known identities and positive chunk size')
        for start in range(0, len(identities), chunk_size):
            wanted, groups, loaded = identities[start:start+chunk_size], {}, {}
            for identity in wanted:
                index, record = self._frames[identity]
                groups.setdefault(index.path, (index, []))[1].append(record)
            for index, records in groups.values():
                for record, raw in zip(records, index.read(row.frame_index for row in records), strict=True):
                    frame = read(StringIO(raw.decode('utf-8')), format='extxyz')
                    frame.info.update(structure_id=record.structure_id, system_id=record.system_id,
                        structure_source='workflow', sampling_role_id=record.sampling_role_id,
                        selection_group_id=record.selection_group_id, condition_group_id=record.condition_group_id,
                        task_id=record.task_id, md_step=record.md_step, md_time_fs=record.md_time_fs,
                        phase=record.phase, phase_step=record.phase_step)
                    if record.student_uncertainty is not None:
                        frame.info['student_uncertainty'] = record.student_uncertainty
                    loaded[record.structure_id] = frame
            yield tuple(loaded[identity] for identity in wanted)


class NativeGpumdProtocolOps:
    def __init__(self, executor, type_map, recipes, inputs=None):
        self.executor, self.type_map, self.recipes = executor, tuple(type_map), recipes
        self.inputs = inputs or {}

    @staticmethod
    def _tasks(evidence):
        return tuple(task for task in evidence.plan.tasks_for_round(evidence.round_index) if task.protocol_request is not None)

    def open_current(self, evidence):
        return SamplingObservationResult(GpumdProtocolObservationSource(evidence,
            tuple((evidence.sample_dir/'tasks'/task.task_id, task) for task in self._tasks(evidence))))

    def validate_current(self, evidence):
        try:
            cache = {}
            for task in self._tasks(evidence):
                validate_gpumd_protocol_result(evidence.sample_dir/'tasks'/task.task_id, intent=task,
                    seed=evidence.seed_mapping[task.sampling_role_id][0], type_map=self.type_map,
                    member_ids=evidence.models.member_ids, digest_cache=cache)
            return SamplingValidationResult(bool(self._tasks(evidence)))
        except (OSError, ValueError, KeyError, TypeError):
            return SamplingValidationResult(False)

    def execute(self, request):
        evidence, prepared, cache = request.evidence, [], {}
        seeds = {role: canonical_selected_seed(frames[0], system_id=role) for role, frames in evidence.seed_mapping.items()}
        publish_selected_seeds(evidence.sample_dir/'seeds.xyz', tuple(seeds.values()))
        if evidence.seed_selections:
            publish_seed_selection_manifest(evidence.sample_dir/'seed_selection.json',
                iter_index=evidence.iter_index, round_index=evidence.round_index,
                selections=evidence.seed_selections)
        for task in self._tasks(evidence):
            source = Path(self.recipes[task.condition_group_id]).read_text()
            protocol, effective = resolve_gpumd_protocol(source, task.protocol_request)
            root = evidence.sample_dir/'tasks'/task.task_id
            root.mkdir(parents=True, exist_ok=True)
            inputs = stage_recipe_inputs(root, self.inputs.get(task.condition_group_id, {}))
            names = []
            for i, model in enumerate(evidence.model_paths):
                name = f'nep_{i}.txt'; link_model_file(model, root/name); names.append(name)
            write_extxyz_structures(root/'seed.xyz', (seeds[task.sampling_role_id],), allow_empty=False)
            write_extxyz_structures(root/'model.xyz', (seeds[task.sampling_role_id],), allow_empty=False)
            lines = [line for line in protocol.splitlines() if line.split('#', 1)[0].split()[:1] != ['potential']]
            (root/'protocol.gpumd').write_text(source)
            rendered, phase = [f'potential {name}' for name in names], 0
            for line in lines:
                if line.split('#', 1)[0].split()[:1] == ['run']:
                    # GPUMD resets its active recorder after each run.
                    rendered.append(f'active {effective["stages"][phase]["cadence"]} 1 1 1 0.0')
                    phase += 1
                rendered.append(line)
            (root/'run.in').write_text('\n'.join(rendered)+'\n')
            context = dict(schema='dice.gpumd-protocol-task', revision=1, request=asdict(task.protocol_request),
                effective=effective, member_ids=list(evidence.models.member_ids), type_map=list(self.type_map), inputs=list(inputs),
                **{name: getattr(task, name) for name in ('task_id', 'sampling_role_id', 'condition_group_id', 'selection_group_id', 'source_frame_index')})
            (root/'task_context.json').write_text(json.dumps(context, sort_keys=True)+'\n')
            prepared.append((task, root, effective, names))
        def execute(item):
            task, root, effective, names = item
            self.executor.run(root, {**effective, 'cadence':effective['stages'][0]['cadence']})
            rows = _protocol_active_rows(root, effective)
            context = json.loads((root/'task_context.json').read_text())
            files = ['seed.xyz', 'model.xyz', 'run.in', 'protocol.gpumd', 'active.xyz', 'active.out', *names, *context['inputs']]
            result = dict(schema='dice.gpumd-protocol-result', revision=1, status='completed', task_id=task.task_id,
                context_digest=_file_digest(root/'task_context.json', cache=cache), observations=rows,
                files={name: _file_digest(root/name, cache=cache) for name in files})
            (root/'gpumd_protocol_result.json').write_text(json.dumps(result, sort_keys=True)+'\n')
            validate_gpumd_protocol_result(root, intent=task, digest_cache=cache)
        execute_sampling_tasks(prepared, context=getattr(self.executor, 'context', None),
            task_id=lambda item: item[0].task_id, execute=execute)
        return SamplingExecutionResult(evidence.sample_dir, len(prepared))
