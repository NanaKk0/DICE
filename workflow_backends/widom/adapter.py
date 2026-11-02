"""Canonical independent insertion results and bounded replay."""
from dataclasses import asdict
from pathlib import Path
from io import StringIO
import hashlib
import json
import os
import tempfile
import numpy as np
from ase.io import iread, read
from ase.io.extxyz import key_val_dict_to_str
from workflow_backends.observation_source import ObservationRecord, ObservationLocator
from workflow_backends.sampling import SamplingExecutionResult, observe_path_backed_sampling_evidence
from workflow_core.lifecycle.sampling import SamplingObservationResult, SamplingValidationResult
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures
from workflow_core.structures.canonical_identity import stamp_canonical_identity, structure_id_for_atoms
from workflow_core.structures.widom import insertion_candidates, validate_inputs
from workflow_backends.widom.evaluation import checked_evaluation
from workflow_backends.gcmc.stage_scope import single_sample_writer
from workflow_backends.common.model_files import link_model_file


def digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('widom canonical evidence must be a regular file')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def validate_task(root, *, task_id=None):
    root = Path(root)
    if root.is_symlink(): raise ValueError('widom task must be regular')
    digest(root/'widom_result.json')
    result = json.loads((root/'widom_result.json').read_text())
    fields = {'schema', 'revision', 'task_status', 'task_id', 'sampling_role_id',
        'selection_group_id', 'condition_group_id', 'member_ids', 'host_atoms',
        'guest_numbers', 'observations', 'counts', 'host', 'guest_library', 'models',
        'parameters', 'entropy', 'source_frame_index', 'frames', 'max_attempts',
        'target_reached', 'termination', 'failure', 'algorithm', 'files'}
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError('widom result fields changed')
    if result['schema'] != 'dice.sample.widom-result' or result['revision'] != 1:
        raise ValueError('unsupported widom result')
    if task_id is not None and result['task_id'] != task_id:
        raise ValueError('widom task identity mismatch')
    members = result['member_ids']
    if not members or len(set(members)) != len(members): raise ValueError('invalid widom members')
    if result['files'] != {'observations.extxyz': digest(root/'observations.extxyz')}:
        raise ValueError('widom damaged canonical observations')
    rows = result['observations']; counts = result['counts']
    if (set(counts) != {'attempted', 'geometry_rejected', 'geometry_accepted', 'emitted'}
        or any(type(v) is not int or v < 0 for v in counts.values())
        or type(result['frames']) is not int or type(result['max_attempts']) is not int
        or not 0 < result['frames'] <= result['max_attempts']
        or counts['emitted'] > result['frames'] or type(result['target_reached']) is not bool
        or result['algorithm'] != 'widom-uniform-v1'):
        raise ValueError('widom invalid budget/algorithm evidence')
    from ase import Atoms
    from workflow_core.common.widom import WidomSamplingParameters
    parameters = WidomSamplingParameters(**result['parameters'])
    if parameters.frames != result['frames'] or parameters.max_attempts != result['max_attempts']:
        raise ValueError('widom parameter budget mismatch')
    host = result['host']
    source = Atoms(numbers=host['numbers'], positions=host['positions'], cell=host['cell'], pbc=True)
    templates = [Atoms(numbers=result['guest_numbers'], positions=positions) for positions in result['guest_library']]
    validate_inputs(source, templates, parameters)
    if len(host['numbers']) != result['host_atoms'] or not result['guest_library']:
        raise ValueError('widom invalid source inventory')
    for index, item in enumerate(result['models']):
        if not item['path'].startswith(f'graph_{index:03d}.'): raise ValueError('widom model order changed')
        if Path(item['path']).name != item['path'] or digest(root/item['path']) != item['sha256']:
            raise ValueError('widom model evidence changed')
    if [item['member_id'] for item in result['models']] != members:
        raise ValueError('widom model member order changed')
    if counts['emitted'] != len(rows) or counts['attempted'] != counts['geometry_rejected']+counts['geometry_accepted']:
        raise ValueError('widom counters disagree')
    if not 0 <= counts['emitted'] <= counts['geometry_accepted'] <= counts['attempted'] <= result['max_attempts']:
        raise ValueError('widom invalid counters')
    if result['task_status'] == 'FAILED':
        if result['termination'] != 'failure' or result['target_reached'] or not result.get('failure'): raise ValueError('missing widom failure')
    elif result['task_status'] == 'SUCCEEDED':
        if result['failure'] is not None: raise ValueError('unexpected widom failure')
        reached = counts['emitted'] == result['frames']
        if counts['emitted'] != counts['geometry_accepted'] or result['target_reached'] != reached:
            raise ValueError('widom target evidence mismatch')
        if result['termination'] != ('target_reached' if reached else 'max_attempts'):
            raise ValueError('widom termination mismatch')
        if not reached and counts['attempted'] != result['max_attempts']: raise ValueError('widom incomplete budget')
    else: raise ValueError('widom invalid status')
    frames = iread(root/'observations.extxyz', ':') if rows else ()
    if not rows and (root/'observations.extxyz').stat().st_size: raise ValueError('unlisted widom frames')
    previous = 0; identities = set()
    for ordinal, (row, frame) in enumerate(zip(rows, frames, strict=True)):
        if (type(row['proposal_index']) is not int or type(row['frame_index']) is not int
            or row['task_index'] != int(result['task_id'].split('_')[-1])
            or row['frame_index'] != ordinal or row['structure_id'] in identities
            or not previous < row['proposal_index'] <= counts['attempted']):
            raise ValueError('widom observation ordering mismatch')
        previous = row['proposal_index']; identities.add(row['structure_id'])
        for key, value in row.items():
            if key != 'force_deviation' and frame.info.get(key) != value:
                raise ValueError(f'widom observation {key} mismatch')
        if (not len(frame) or not np.all(frame.pbc) or not np.isfinite(frame.positions).all()
            or not np.isfinite(frame.cell.array).all() or np.linalg.matrix_rank(frame.cell.array) != 3):
            raise ValueError('widom invalid geometry')
        for key in ('task_id', 'sampling_role_id', 'selection_group_id', 'condition_group_id', 'source_frame_index'):
            if row[key] != result[key]: raise ValueError('widom row task provenance changed')
        if (type(row['guest_index']) is not int or not 0 <= row['guest_index'] < len(result['guest_library'])
            or row['structure_id'] != structure_id_for_atoms(frame)):
            raise ValueError('widom proposal provenance changed')
        n = result['host_atoms']
        if (frame.numbers[:n].tolist() != host['numbers']
            or not np.allclose(frame.positions[:n], host['positions'], rtol=0, atol=1e-8)
            or not np.allclose(frame.cell.array, host['cell'], rtol=0, atol=1e-10)):
            raise ValueError('widom background geometry changed')
        original = np.asarray(result['guest_library'][row['guest_index']])
        actual = frame.positions[n:]
        if not np.allclose(np.linalg.norm(original[:,None]-original, axis=-1),
                           np.linalg.norm(actual[:,None]-actual, axis=-1), rtol=0, atol=3e-8):
            raise ValueError('widom internal guest geometry changed')
        if len(frame) != n+len(result['guest_numbers']) or frame.numbers[n:].tolist() != result['guest_numbers']:
            raise ValueError('widom guest inventory mismatch')
        e = np.asarray(frame.info['member_energies_ev']).reshape(-1)
        f = np.asarray([frame.arrays[f'dice_force_member_{i:03d}'] for i in range(len(members))])
        if e.shape != (len(members),) or f.shape != (len(members), len(frame), 3) or not np.isfinite(e).all() or not np.isfinite(f).all():
            raise ValueError('widom invalid model evidence')
        deviation = None if len(members) == 1 else float(np.linalg.norm(f.std(axis=0), axis=1).max())
        if ((deviation is None and row['force_deviation'] is not None) or
            (deviation is not None and (row['force_deviation'] is None or not np.isclose(
                row['force_deviation'], deviation, rtol=1e-10, atol=1e-8)))):
            raise ValueError('widom force deviation mismatch')
    return result


def serialize_candidate(frame, member_count):
    """Round-trip model coordinates/forces, including high but finite values."""
    fields = ['species:S:1', 'pos:R:3']
    names = [f'dice_force_member_{i:03d}' for i in range(member_count)]
    fields.extend(f'{name}:R:3' for name in names)
    header = key_val_dict_to_str(dict(frame.info, Lattice=frame.cell.array.T,
                                     Properties=':'.join(fields), pbc=frame.pbc))
    values = np.concatenate([frame.positions, *(frame.arrays[n] for n in names)], axis=1)
    lines = [str(len(frame)), header]
    lines.extend(symbol+' '+' '.join(format(float(v), '.17g') for v in row)
                 for symbol, row in zip(frame.get_chemical_symbols(), values, strict=True))
    return ('\n'.join(lines)+'\n').encode()


class WidomObservationSource:
    allows_empty = True
    def __init__(self, tasks):
        self.failed_task_count = sum(r['task_status'] == 'FAILED' for _, r in tasks)
        self._records = []; self._offsets = {}
        for root, result in tasks:
            path = Path(root)/'observations.extxyz'
            with path.open('rb') as stream:
                for row in result['observations']:
                    start = stream.tell(); count = int(stream.readline())
                    for _ in range(count+1):
                        if not stream.readline().endswith(b'\n'): raise ValueError('widom truncated frame')
                    self._offsets[row['structure_id']] = (path, start, stream.tell()-start, path.stat())
                    self._records.append(ObservationRecord(
                        structure_id=row['structure_id'], system_id=row['selection_group_id'],
                        task_index=row['task_index'], frame_index=row['frame_index'], md_step=None,
                        student_uncertainty=row['force_deviation'],
                        locator=ObservationLocator('WIDOM', row['task_index'], row['frame_index'], None),
                        sampling_role_id=row['sampling_role_id'], selection_group_id=row['selection_group_id'],
                        condition_group_id=row['condition_group_id'], task_id=row['task_id'], phase='widom'))
        if len(self._offsets) != len(self._records): raise ValueError('widom duplicate identity')

    def catalog(self): return tuple(self._records)

    def replay(self, ordered_identities, *, chunk_size):
        ids = tuple(ordered_identities)
        if isinstance(chunk_size, bool) or chunk_size <= 0 or len(set(ids)) != len(ids) or any(i not in self._offsets for i in ids):
            raise ValueError('widom invalid replay request')
        for start in range(0, len(ids), chunk_size):
            frames = []
            for identity in ids[start:start+chunk_size]:
                path, offset, length, stamp = self._offsets[identity]
                if path.is_symlink() or path.stat() != stamp: raise ValueError('widom replay evidence changed')
                with path.open('rb') as f:
                    f.seek(offset); frame = read(StringIO(f.read(length).decode()), format='extxyz')
                if frame.info['structure_id'] != identity: raise ValueError('widom replay identity mismatch')
                frames.append(frame)
            yield tuple(frames)


class WidomSamplingNativeOps:
    def __init__(self, *, provider, guest_sources, type_map):
        self.provider = provider
        self.guest_sources = {k: Path(p) for k, p in guest_sources.items()}
        self.type_map = tuple(type_map)

    def _tasks(self, evidence):
        return tuple(t for t in evidence.plan.tasks_for_round(evidence.round_index) if t.kind == 'widom')

    def prepare(self, evidence):
        prepared = []
        libraries = {}
        for task in self._tasks(evidence):
            seeds = evidence.seed_mapping[task.sampling_role_id]
            if len(seeds) != 1: raise ValueError('widom requires one resolved host per task')
            guest_path = self.guest_sources[task.condition_group_id]
            if guest_path not in libraries:
                libraries[guest_path] = read_extxyz_structures(guest_path)
            guests = libraries[guest_path]
            validate_inputs(seeds[0], guests, task.widom, self.type_map)
            prepared.append((task, seeds[0], guests))
        return prepared

    @single_sample_writer
    def execute(self, request):
        evidence = request.evidence
        prepared = self.prepare(evidence)
        self.provider.preflight(evidence.model_paths, self.type_map)
        if self.validate_current(evidence).complete: return self._execution_result(evidence)
        evidence.sample_dir.mkdir(parents=True, exist_ok=True)
        (evidence.sample_dir/'tasks').mkdir(exist_ok=True)
        write_extxyz_structures(evidence.sample_dir/'seeds.xyz', [stamp_canonical_identity(
            s.frames[0], system_id=s.system_id, structure_source='workflow') for s in evidence.systems])
        for task, seed, guests in prepared:
            destination = evidence.sample_dir/'tasks'/task.task_id
            if destination.exists():
                old = Path(tempfile.mkdtemp(prefix='.previous-widom-', dir=evidence.sample_dir))
                os.replace(destination, old/task.task_id)
            self.execute_task(evidence, task, seed, guests)
        return self._execution_result(evidence)

    def execute_task(self, evidence, task, seed, guests, *, staging_root=None):
        root = Path(staging_root) if staging_root is not None else Path(tempfile.mkdtemp(prefix='.'+task.task_id+'.widom-', dir=evidence.sample_dir))
        root.mkdir(parents=True, exist_ok=True)
        path = root/'observations.extxyz'; path.touch()
        rows = []; counts = dict(attempted=0, geometry_rejected=0, geometry_accepted=0, emitted=0)
        error = None; evaluator = None
        entropy = json.dumps([evidence.iter_index, evidence.round_index, task.condition_group_id,
                              task.sampling_role_id, task.source_frame_index])
        def record(row=None):
            with (root/'attempt.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(counts=counts, row=row, end=path.stat().st_size), allow_nan=False)+'\n')
        try:
            evaluator = self.provider.open(evidence.model_paths, self.type_map)
            for attempt, guest_index, frame in insertion_candidates(seed, guests, task.widom, entropy=entropy):
                counts['attempted'] = attempt
                if frame is None:
                    counts['geometry_rejected'] += 1
                    record(); continue
                counts['geometry_accepted'] += 1
                record()
                energies, forces = checked_evaluation(evaluator, frame, len(evidence.models.member_ids))
                frame = stamp_canonical_identity(frame, system_id=task.selection_group_id, structure_source='workflow')
                identity = frame.info['structure_id']
                row = dict(structure_id=identity, task_id=task.task_id, task_index=int(task.task_id.split('_')[-1]),
                    frame_index=len(rows), sampling_role_id=task.sampling_role_id,
                    selection_group_id=task.selection_group_id, condition_group_id=task.condition_group_id,
                    source_frame_index=task.source_frame_index, guest_index=guest_index, proposal_index=attempt)
                frame.info.update(row); frame.info['member_energies_ev'] = energies
                for i, f in enumerate(forces): frame.set_array(f'dice_force_member_{i:03d}', f)
                row['force_deviation'] = None if len(forces) == 1 else float(np.linalg.norm(forces.std(axis=0), axis=1).max())
                # Serialize completely before append; roll back a partial write to the
                # last validated frame boundary rather than publishing a torn tail.
                serialized = serialize_candidate(frame, len(forces))
                boundary = path.stat().st_size
                try:
                    with path.open('ab') as stream:
                        stream.write(serialized)
                except Exception:
                    with path.open('r+b') as stream: stream.truncate(boundary)
                    raise
                rows.append(row); counts['emitted'] += 1
                record(row)
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        finally:
            if evaluator is not None:
                try: evaluator.close()
                except Exception as exc:
                    if error is None: error = dict(type=type(exc).__name__, message=str(exc))
        self._publish_task(evidence, task, seed, guests, root, rows, counts, error)

    def publish_process_failure(self, evidence, task, seed, guests, root, error):
        """Finalize this attempt's validated prefix; never resume a task."""
        destination = evidence.sample_dir/'tasks'/task.task_id
        if destination.exists():
            validate_task(destination, task_id=task.task_id)
            return
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        path = root/'observations.extxyz'; path.touch(exist_ok=True)
        rows = []; counts = dict(attempted=0, geometry_rejected=0, geometry_accepted=0, emitted=0)
        boundary = 0
        journal = root/'attempt.jsonl'
        if journal.exists():
            with journal.open() as stream:
                for line in stream:
                    if not line.endswith('\n'): break
                    event = json.loads(line)
                    counts = event['counts']; boundary = event['end']
                    if event['row'] is not None: rows.append(event['row'])
        with path.open('r+b') as stream: stream.truncate(boundary)
        self._publish_task(evidence, task, seed, guests, root, rows, counts,
                           dict(type=type(error).__name__, message=str(error)))

    def _publish_task(self, evidence, task, seed, guests, root, rows, counts, error):
        path = root/'observations.extxyz'
        entropy = json.dumps([evidence.iter_index, evidence.round_index, task.condition_group_id,
                              task.sampling_role_id, task.source_frame_index])
        model_rows = []
        for i, model in enumerate(evidence.model_paths):
            name = f'graph_{i:03d}{model.suffix}'
            link_model_file(model, root/name)
            model_rows.append(dict(path=name, member_id=evidence.models.member_ids[i], sha256=digest(root/name)))
        reached = counts['emitted'] == task.widom.frames and error is None
        result = dict(schema='dice.sample.widom-result', revision=1, task_status='FAILED' if error else 'SUCCEEDED',
            task_id=task.task_id, sampling_role_id=task.sampling_role_id, selection_group_id=task.selection_group_id,
            condition_group_id=task.condition_group_id, member_ids=list(evidence.models.member_ids),
            host_atoms=len(seed), guest_numbers=guests[0].numbers.tolist(), observations=rows, counts=counts,
            host=dict(numbers=seed.numbers.tolist(), positions=seed.positions.tolist(), cell=seed.cell.array.tolist()),
            guest_library=[g.positions.tolist() for g in guests], models=model_rows,
            parameters=asdict(task.widom), entropy=entropy, source_frame_index=task.source_frame_index,
            frames=task.widom.frames, max_attempts=task.widom.max_attempts, target_reached=reached,
            termination='failure' if error else ('target_reached' if reached else 'max_attempts'), failure=error,
            algorithm='widom-uniform-v1', files={'observations.extxyz': digest(path)})
        (root/'widom_result.json').write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
        validate_task(root, task_id=task.task_id)
        os.replace(root, evidence.sample_dir/'tasks'/task.task_id)

    def _execution_result(self, evidence):
        return SamplingExecutionResult(evidence.sample_dir, len(self._tasks(evidence)), self.open_current(evidence).source.failed_task_count)

    def open_current(self, evidence):
        tasks = []
        for task in self._tasks(evidence):
            root = evidence.sample_dir/'tasks'/task.task_id
            result = validate_task(root, task_id=task.task_id)
            if tuple(result['member_ids']) != evidence.models.member_ids or any(result[k] != getattr(task, k) for k in ('sampling_role_id','selection_group_id','condition_group_id')):
                raise ValueError('widom result membership mismatch')
            if result['parameters'] != json.loads(json.dumps(asdict(task.widom))):
                raise ValueError('widom parameters changed')
            tasks.append((root, result))
        return SamplingObservationResult(WidomObservationSource(tasks))

    def validate_current(self, evidence):
        try:
            if not self._tasks(evidence): return SamplingValidationResult(False)
            self.open_current(evidence)
            return SamplingValidationResult(True)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return SamplingValidationResult(False)

    def observe_evidence(self, evidence): return observe_path_backed_sampling_evidence(evidence)

    def reconstruct_compatible_legacy(self, evidence):
        from workflow_core.lifecycle.sampling import SamplingEvidenceRebuildRequired, SamplingEvidenceFailure
        raise SamplingEvidenceRebuildRequired('Widom has no historical result format; rebuild rN/sample',
            failure=SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE, round_index=evidence.round_index)
