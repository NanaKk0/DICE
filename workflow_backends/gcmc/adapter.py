"""Current-state acquisition behind the ordinary sampling capability."""
from __future__ import annotations

from contextlib import redirect_stdout
import fcntl
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
from ase.data import atomic_numbers, vdw_radii
from ase.io import iread, read, write
from ase.io.extxyz import XYZError

from workflow_backends.gcmc.calculator import SharedCalculator
from workflow_backends.gcmc.phase import McPhase
from workflow_backends.gcmc.eos import reservoir_phi
from workflow_backends.gcmc.lammps import MdSegmentFailure
from workflow_backends.common.model_files import link_model_file
from workflow_backends.gcmc.stage_scope import single_sample_writer
from workflow_backends.observation_source import ObservationLocator, ObservationRecord
from workflow_backends.sampling import SamplingExecutionResult, observe_path_backed_sampling_evidence
from workflow_core.lifecycle.sampling import SamplingObservationResult, SamplingValidationResult
from workflow_core.structures.canonical_identity import stamp_canonical_identity
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


def _digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected regular canonical file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def present_task_models(evidence, task, models):
    """Expose the same ordered driving models to the block-level validator."""
    root = evidence.sample_dir/'tasks'/task.task_id
    for index, path in enumerate(models):
        path = Path(path)
        link_model_file(path, root/f'graph_{index:03d}{path.suffix}')


def archive_incomplete_sample_tasks(sample_dir):
    """Preserve diagnostics, but restart every task of an incomplete stage."""
    tasks = Path(sample_dir)/'tasks'
    if tasks.is_symlink():
        raise ValueError('sampling tasks directory cannot be a symlink')
    if tasks.exists():
        archive = Path(tempfile.mkdtemp(prefix='.previous-sample-', dir=sample_dir))
        os.replace(tasks, archive/'tasks')
    tasks.mkdir(parents=True)


def _validate_canonical_state(frame):
    """Check transport invariants without consulting a producer checkpoint."""
    if (not len(frame) or not np.all(frame.pbc)
        or not np.isfinite(frame.positions).all()
        or not np.isfinite(frame.cell.array).all()
        or np.linalg.matrix_rank(frame.cell.array) != 3):
        raise ValueError('GCMC canonical geometry must be finite and periodic/full-rank')
    masses = frame.get_masses()
    if (not frame.has('momenta') or not np.isfinite(frame.get_momenta()).all()
        or not np.isfinite(masses).all() or np.any(masses <= 0)):
        raise ValueError('GCMC canonical masses/velocities are invalid')
    ids, mids = (frame.arrays[key] for key in ('dice_atom_id', 'dice_molecule_id'))
    if any(a.shape != (len(frame),) or a.dtype.kind not in 'iu' for a in (ids, mids)):
        raise ValueError('GCMC canonical identity/membership must be integer vectors')
    if np.any(ids <= 0) or len(set(ids)) != len(frame) or np.any(mids < 0):
        raise ValueError('GCMC canonical identity/membership is invalid')
    n_host = int(np.count_nonzero(mids == 0))
    if not n_host or np.any(mids[:n_host] != 0):
        raise ValueError('GCMC canonical host must be leading and nonempty')
    # Explicit membership, not inferred bonds, defines molecule boundaries.
    starts = np.r_[n_host, np.flatnonzero(np.diff(mids[n_host:]))+n_host+1]
    guest_ids, guest_species = [], []
    for start, end in zip(starts, np.r_[starts[1:], len(frame)]):
        if start == end:
            continue
        guest_ids.append(int(mids[start]))
        guest_species.append(tuple(frame.numbers[start:end]))
    if len(set(guest_ids)) != len(guest_ids) or len(set(guest_species)) > 1:
        raise ValueError('GCMC canonical guests must be complete contiguous identical templates')
    return tuple(zip(ids[:n_host], frame.numbers[:n_host])), next(iter(guest_species), None)


def validate_task(task_dir, *, task_id=None):
    """Read only intrinsic consumer evidence, never producer-config fingerprints."""
    root = Path(task_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('GCMC task must be a regular directory')
    _digest(root/'gcmc_result.json')
    result = json.loads((root/'gcmc_result.json').read_text())
    if result.get('schema') != 'dice.sample.gcmc-result' or result.get('revision') not in (1, 2):
        raise ValueError('unsupported GCMC result')
    failed = result.get('revision') == 2
    if failed and (result.get('task_status') != 'FAILED' or not isinstance(result.get('failure'), dict)
                   or result.get('retained_observations') != len(result['observations'])):
        raise ValueError('GCMC failed-task evidence is invalid')
    if task_id is not None and result['task_id'] != task_id:
        raise ValueError('GCMC task identity mismatch')
    members = result['member_ids']
    if not members or len(set(members)) != len(members):
        raise ValueError('GCMC ordered member evidence is invalid')
    inventory = {'observations.extxyz'} if failed else {'observations.extxyz', 'final_structure.extxyz'}
    if set(result['files']) != inventory:
        raise ValueError('GCMC canonical inventory is invalid')
    for name, digest in result['files'].items():
        if _digest(root/name) != digest:
            raise ValueError(f'GCMC canonical content damaged: {name}')
    host_inventory, guest_species = None, None
    if not failed:
        final = read_extxyz_structures(root/'final_structure.extxyz')
        if len(final) != 1 or not np.isfinite(final[0].positions).all():
            raise ValueError('GCMC endpoint is invalid')
        host_inventory, guest_species = _validate_canonical_state(final[0])
    rows = result['observations']
    if not rows and (root/'observations.extxyz').stat().st_size:
        raise ValueError('GCMC empty catalog has unlisted canonical frames')
    identities = [row['structure_id'] for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('GCMC observation identities are duplicated')
    frames = iread(root/'observations.extxyz', index=':') if rows else iter(())
    count = 0
    for count, (row, frame) in enumerate(zip(rows, frames, strict=True), 1):
        if row['frame_index'] != count-1 or frame.info.get('structure_id') != row['structure_id']:
            raise ValueError('GCMC observation order/identity is invalid')
        host, guest = _validate_canonical_state(frame)
        if (host_inventory is not None and host != host_inventory) or (guest is not None and guest_species is not None and guest != guest_species):
            raise ValueError('GCMC canonical host/guest inventory changed')
        host_inventory = host
        if guest is not None:
            guest_species = guest
        for key in ('phase', 'cycle', 'phase_step', 'mc_attempts', 'md_steps',
                    'sampling_role_id', 'selection_group_id', 'condition_group_id', 'task_id'):
            if frame.info.get(key) != row[key]:
                raise ValueError(f'GCMC observation {key} mismatch')
        energies = np.asarray(frame.info['member_energies_ev'])
        # ASE extxyz reads a length-one numeric metadata vector as a scalar.
        if len(members) == 1 and energies.ndim == 0:
            energies = energies.reshape(1)
        forces = np.asarray([frame.arrays[f'dice_force_member_{i:03d}'] for i in range(len(members))])
        if energies.shape != (len(members),) or not np.isfinite(energies).all() or forces.shape != (len(members), len(frame), 3) or not np.isfinite(forces).all():
            raise ValueError('GCMC raw ordered model evidence is invalid')
        deviation = None if len(members) == 1 else float(np.linalg.norm(forces.std(axis=0), axis=1).max())
        if (deviation is None and row['force_deviation'] is not None) or (
            deviation is not None and not np.isclose(deviation, row['force_deviation'], atol=1e-7, rtol=1e-7)
        ):
            raise ValueError('GCMC force deviation disagrees with raw evidence')
    if count != len(rows):
        raise ValueError('GCMC observation inventory is incomplete')
    return result


def _saved_frames(path):
    # Read independently: a corrupt trailing count must not prevent ASE from
    # returning the valid prefix while indexing the entire trajectory.
    with path.open() as stream:
        while header := stream.readline():
            count = int(header.strip())
            if count <= 0:
                raise ValueError('invalid saved atom count')
            lines = [stream.readline() for _ in range(count+1)]
            if any(not line for line in lines):
                raise ValueError('truncated saved frame')
            yield read(StringIO(header+''.join(lines)), format='extxyz')


def publish_failed_task(evidence, task, error):
    """Salvage only fully written observations, never resume a failed chain."""
    destination = evidence.sample_dir/'tasks'/task.task_id
    if destination.exists() or destination.is_symlink():
        # Do not overwrite an existing result or mask canonical corruption.
        return validate_task(destination, task_id=task.task_id)
    attempts = tuple(destination.parent.glob('.'+task.task_id+'.gcmc-*'))
    if len(attempts) > 1 or any(p.is_symlink() for p in attempts):
        raise ValueError('ambiguous GCMC failed-task observation source') from error
    output = Path(tempfile.mkdtemp(prefix='.'+task.task_id+'.failed-', dir=destination.parent))
    target = output/'observations.extxyz'
    target.touch()
    rows, discarded = [], None
    host_inventory, guest_species = None, None
    members = evidence.models.member_ids
    source = attempts[0]/'observations.extxyz' if attempts else None
    if source is not None and source.is_file() and not source.is_symlink():
        try:
            for frame in _saved_frames(source):
                host, guest = _validate_canonical_state(frame)
                if ((host_inventory is not None and host != host_inventory)
                    or (guest is not None and guest_species is not None and guest != guest_species)):
                    raise ValueError('saved observation changed host/guest inventory')
                host_inventory = host
                if guest is not None:
                    guest_species = guest
                row = {key: frame.info[key] for key in (
                    'structure_id', 'task_id', 'task_index', 'frame_index', 'sampling_role_id',
                    'selection_group_id', 'condition_group_id', 'phase', 'cycle', 'phase_step',
                    'mc_attempts', 'md_steps', 'md_time_fs')}
                row = {key: value.item() if isinstance(value, np.generic) else value for key, value in row.items()}
                if row['frame_index'] != len(rows) or any(row[key] != getattr(task, key) for key in (
                    'task_id', 'sampling_role_id', 'selection_group_id', 'condition_group_id')):
                    raise ValueError('saved observation task/order mismatch')
                forces = np.asarray([frame.arrays[f'dice_force_member_{i:03d}'] for i in range(len(members))])
                energies = np.asarray(frame.info['member_energies_ev']).reshape(-1)
                if (forces.shape != (len(members), len(frame), 3) or energies.shape != (len(members),)
                    or not np.isfinite(forces).all() or not np.isfinite(energies).all()):
                    raise ValueError('saved observation has invalid model evidence')
                row['force_deviation'] = None if len(members) == 1 else float(np.linalg.norm(forces.std(axis=0), axis=1).max())
                write(target, frame, format='extxyz', append=True)
                rows.append(row)
        except (ValueError, KeyError, TypeError, StopIteration, IndexError, XYZError) as exc:
            discarded = f'{type(exc).__name__}: {exc}'
    result = dict(schema='dice.sample.gcmc-result', revision=2, task_status='FAILED',
        task_id=task.task_id, sampling_role_id=task.sampling_role_id,
        selection_group_id=task.selection_group_id, condition_group_id=task.condition_group_id,
        member_ids=list(members), observations=rows, retained_observations=len(rows),
        failure={'type': type(error).__name__, 'message': str(error), 'discarded_tail': discarded},
        files={'observations.extxyz': _digest(target)})
    _json(output/'gcmc_result.json', result)
    validate_task(output, task_id=task.task_id)
    if attempts:
        os.replace(attempts[0], output/'failed-attempt-diagnostics')
    os.replace(output, destination)
    warnings.warn(f'{task.task_id} FAILED; retained {len(rows)} validated observations: {error}', RuntimeWarning)
    return result


class GcmcObservationSource:
    def __init__(self, tasks):
        self.failed_task_count = sum(result.get('task_status') == 'FAILED' for _, result in tasks)
        self._items = tuple((Path(path), row) for path, result in tasks for row in result['observations'])
        self._records = tuple(ObservationRecord(
            structure_id=row['structure_id'], system_id=row['selection_group_id'],
            task_index=row['task_index'], frame_index=row['frame_index'], md_step=row['md_steps'],
            student_uncertainty=row['force_deviation'],
            locator=ObservationLocator('FLAMES', row['task_index'], row['frame_index'], row['md_steps']),
            sampling_role_id=row['sampling_role_id'], selection_group_id=row['selection_group_id'],
            condition_group_id=row['condition_group_id'], task_id=row['task_id'],
            phase=row['phase'], cycle=row['cycle'], phase_step=row['phase_step'],
            mc_attempts=row['mc_attempts'], md_time_fs=row['md_time_fs'],
        ) for _, row in self._items)
        self._owners = {row['structure_id']: path for path, row in self._items}
        if len(self._owners) != len(self._items):
            raise ValueError('GCMC task observation identities collide')
        self._task_ids = {}
        for root, row in self._items:
            self._task_ids.setdefault(root, []).append(row['structure_id'])
        # Invocation-local byte offsets, never cached Atoms or restart authority.
        self._indices = {}

    def catalog(self):
        return self._records

    @staticmethod
    def _file_stamp(path):
        if path.is_symlink():
            raise ValueError('GCMC replay requires regular canonical evidence')
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _index_task(self, root):
        path = root/'observations.extxyz'
        stamp = self._file_stamp(path)
        offsets = {}
        with path.open('rb') as stream:
            for identity in self._task_ids[root]:
                start = stream.tell()
                header = stream.readline()
                try:
                    count = int(header)
                except ValueError as exc:
                    raise ValueError('GCMC replay truncated/invalid atom count') from exc
                if count <= 0:
                    raise ValueError('GCMC replay invalid atom count')
                # Canonical GCMC extxyz is count + metadata + count atom rows.
                for _ in range(count + 1):
                    if not stream.readline().endswith(b'\n'):
                        raise ValueError('GCMC replay truncated canonical frame')
                offsets[identity] = (start, stream.tell() - start)
            if stream.read(1):
                raise ValueError('GCMC replay canonical evidence has unlisted frames')
        if self._file_stamp(path) != stamp:
            raise ValueError('GCMC replay canonical evidence changed during indexing')
        self._indices[root] = (stamp, offsets)

    def replay(self, ordered_identities, *, chunk_size):
        ids = tuple(ordered_identities)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('GCMC replay chunk_size must be positive')
        if len(set(ids)) != len(ids) or any(i not in self._owners for i in ids):
            raise ValueError('GCMC replay identities must be unique and present')
        for start in range(0, len(ids), chunk_size):
            wanted = ids[start:start+chunk_size]
            loaded = {}
            for root in dict.fromkeys(self._owners[i] for i in wanted):
                if root not in self._indices:
                    self._index_task(root)
                stamp, offsets = self._indices[root]
                path = root/'observations.extxyz'
                if self._file_stamp(path) != stamp:
                    raise ValueError('GCMC replay canonical evidence changed after indexing')
                with path.open('rb') as stream:
                    for identity in wanted:
                        if self._owners[identity] != root:
                            continue
                        offset, size = offsets[identity]
                        stream.seek(offset)
                        frame = read(StringIO(stream.read(size).decode('utf-8')), format='extxyz')
                        if frame.info.get('structure_id') != identity:
                            raise ValueError('GCMC replay canonical identity/order mismatch')
                        loaded[identity] = frame
                if self._file_stamp(path) != stamp:
                    raise ValueError('GCMC replay canonical evidence changed during reading')
            if set(loaded) != set(wanted):
                raise ValueError('GCMC replay canonical evidence is incomplete')
            yield tuple(loaded[i] for i in wanted)


class GcmcSamplingNativeOps:
    def __init__(self, *, calculator_provider, guest_sources, gcmc_class, device,
                 reference_tolerance_ev=0., md_runners=None):
        self.calculator_provider = calculator_provider
        self.guest_sources = {key: Path(path) for key, path in guest_sources.items()}
        self.gcmc_class = gcmc_class
        self.device = device
        self.reference_tolerance_ev = reference_tolerance_ev
        self.md_runners = dict(md_runners or {})

    @staticmethod
    def _tasks(evidence):
        return tuple(t for t in evidence.plan.tasks_for_round(evidence.round_index) if t.kind == 'gcmc_md')

    @single_sample_writer
    def execute(self, request):
        evidence = request.evidence
        tasks = self._tasks(evidence)
        if not tasks:
            raise ValueError('GCMC capability requires active gcmc_md tasks')
        models = tuple(evidence.model_paths)
        self.calculator_provider.preflight(models)
        root = evidence.sample_dir
        if self.validate_current(evidence).complete:
            return self._execution_result(evidence)
        archive_incomplete_sample_tasks(root)
        write_extxyz_structures(root/'seeds.xyz', [stamp_canonical_identity(
            source.frames[0], system_id=source.system_id, structure_source='workflow') for source in evidence.systems])
        for task in tasks:
            seed_frames = evidence.seed_mapping[task.sampling_role_id]
            if len(seed_frames) != 1:
                raise ValueError('GCMC task requires exactly one resolved seed frame')
            guest_path = self.guest_sources[task.condition_group_id]
            _digest(guest_path)
            guests = read_extxyz_structures(guest_path)
            if len(guests) != 1:
                raise ValueError('GCMC requires exactly one guest template')
            self._execute_task(evidence, task, seed_frames[0], guests[0], models)
        return self._execution_result(evidence)

    def _execution_result(self, evidence):
        source = self.open_current(evidence).source
        return SamplingExecutionResult(evidence.sample_dir, len(self._tasks(evidence)), source.failed_task_count)

    def _execute_task(self, evidence, task, seed, guest, models):
        destination = evidence.sample_dir/'tasks'/task.task_id
        # Execution authority still belongs to the outer runtime. This local
        # lock only prevents two workers from publishing the same task.
        with (evidence.sample_dir/('.gcmc-'+task.task_id+'.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('GCMC task already has an active local writer') from exc
            try:
                self._execute_locked(evidence, task, seed, guest, models)
            except Exception as exc:
                publish_failed_task(evidence, task, exc)
            present_task_models(evidence, task, models)

    def _execute_locked(self, evidence, task, seed, guest, models):
        destination = evidence.sample_dir/'tasks'/task.task_id
        if destination.exists() or destination.is_symlink():
            validate_task(destination, task_id=task.task_id)
            return
        parameters = task.gcmc
        phi = reservoir_phi(parameters, temperature_k=task.temperature_k, pressure_bar=task.pressure_bar, guest=guest)
        md_runner = self.md_runners.get(task.condition_group_id)
        if parameters.md is not None and md_runner is None:
            raise ValueError('native MD segment runner is required')
        # Each execution starts from the source seed. Unfinished attempts are
        # diagnostic-only, never a cycle/task recovery authority.
        staging = Path(tempfile.mkdtemp(prefix='.'+task.task_id+'.gcmc-', dir=destination.parent))
        stream_identity = [str(evidence.run_json_path), evidence.iter_index, evidence.round_index,
                           task.sampling_role_id, task.condition_group_id, task.task_id,
                           task.temperature_k, task.pressure_bar, parameters.random_seed]
        salt = hashlib.sha256(json.dumps(stream_identity).encode()).digest()
        mc_seed = int.from_bytes(salt[:8], 'big')
        velocity_seed = int.from_bytes(salt[8:16], 'big')
        calculators = tuple(SharedCalculator(self.calculator_provider.create(path),
                                            context=f'{task.task_id}/{member}')
                            for member, path in zip(evidence.models.member_ids, models, strict=True))
        if parameters.md is not None and parameters.md.ensemble == 'npt':
            # Validate the driving model on the actual seed before any MC
            # attempt. Native LAMMPS still owns integration and its virial;
            # this is a capability check, not a supplied/corrected stress.
            try:
                calculators[0].get_property('stress', seed)
            except Exception as exc:
                raise ValueError('NPT requires valid driving-model stress before sampling') from exc
        radii = np.array(vdw_radii, copy=True)
        if parameters.radii_a:
            radii[:] = np.nan
            for symbol, value in parameters.radii_a:
                radii[atomic_numbers[symbol]] = value
        rows = []
        cycle = 0
        output = staging
        md_steps, md_time_fs = 0, 0.
        active_phase = 'gcmc'
        md_segments = []
        (output/'observations.extxyz').touch()

        def capture(frame):
            # Seed metadata is not evidence for a newly acquired geometry.
            for key in ('student_uncertainty', 'force_uncertainty', 'force_std_max'):
                frame.info.pop(key, None)
            energies, forces = [], []
            for calculator in calculators:
                frame.calc = calculator
                energies.append(float(frame.get_potential_energy()))
                forces.append(frame.get_forces())
            frame.calc = None
            deviation = None if len(forces) == 1 else float(np.linalg.norm(np.std(forces, axis=0), axis=1).max())
            identity = 'sha256:'+hashlib.sha256(json.dumps([stream_identity, len(rows), frame.info['mc_attempts']]).encode()).hexdigest()
            row = dict(structure_id=identity, task_id=task.task_id,
                       task_index=int(task.task_id.removeprefix('task_')), frame_index=len(rows),
                       sampling_role_id=task.sampling_role_id, selection_group_id=task.selection_group_id,
                       condition_group_id=task.condition_group_id, phase=active_phase, cycle=cycle,
                       phase_step=(frame.info['mc_attempts']-(cycle-1)*parameters.moves
                                   if active_phase == 'gcmc' else frame.info['phase_step']),
                       mc_attempts=frame.info['mc_attempts'],
                       md_steps=md_steps if active_phase == 'gcmc' else frame.info['md_steps'],
                       md_time_fs=md_time_fs if active_phase == 'gcmc' else frame.info['md_time_fs'],
                       force_deviation=deviation)
            frame.info.update({k: v for k, v in row.items() if v is not None})
            frame.info.update(system_id=task.selection_group_id, structure_source='workflow',
                              member_energies_ev=np.asarray(energies))
            for i, force in enumerate(forces):
                frame.set_array(f'dice_force_member_{i:03d}', force)
            if deviation is not None:
                frame.info['student_uncertainty'] = deviation
            write(output/'observations.extxyz', frame, format='extxyz', append=True)
            rows.append(row)

        def new_phase(atoms, native):
            return McPhase(self.gcmc_class, atoms, guest, calculators[0], temperature_k=task.temperature_k,
                         pressure_bar=task.pressure_bar, random_seed=mc_seed, velocity_seed=velocity_seed,
                         output_dir=native, device=self.device, radii_a=radii,
                         reference_tolerance_ev=self.reference_tolerance_ev,
                         phi=phi, translate_a=parameters.translate_a, rotate_deg=parameters.rotate_deg,
                         overlap_scale=parameters.overlap_scale, de_max_ev=parameters.de_max_ev,
                         max_guests=parameters.max_guests, weights=dict(zip(
                             ('insertion', 'deletion', 'translation', 'rotation', 'reinsertion'), parameters.weights)))

        with (staging/'flames.log').open('a') as log, redirect_stdout(log):
            atoms, handoff = seed, None
            for cycle in range(1, task.cycles+1):
                active_phase = 'gcmc'
                with new_phase(atoms, staging/f'mc_{cycle:06d}') as phase:
                    if handoff is not None:
                        if parameters.md is None:
                            phase.restore(handoff)
                        else:
                            phase.restore_after_md(handoff)
                    phase.advance(parameters.moves, every=parameters.every, observe=capture)
                    handoff = phase.checkpoint()
                atoms = handoff['atoms']
                if parameters.md is not None:
                    failure = None
                    try:
                        segment = md_runner.run(atoms, parameters=parameters.md, temperature_k=task.temperature_k,
                                                start_step=md_steps, output_dir=staging/f'md_{cycle:06d}')
                    except MdSegmentFailure as exc:
                        segment = failure = exc
                    active_phase = 'md'
                    try:
                        for frame in segment.observations:
                            frame.info.update(mc_attempts=cycle*parameters.moves,
                                md_time_fs=md_time_fs+(frame.info['md_steps']-md_steps)*segment.prepared.timestep_fs)
                            capture(frame)
                    except Exception as exc:
                        if failure is None:
                            raise
                        # Keep the original native failure and the reason why
                        # prefix import stopped. Never skip a damaged frame.
                        failure.args = (f'{failure}; MD prefix stopped: {type(exc).__name__}: {exc}',)
                        raise failure from exc
                    if failure is not None:
                        raise failure
                    atoms = segment.endpoint
                    md_segments.append(dict(cycle=cycle, start_step=md_steps,
                        end_step=md_steps+parameters.md.steps,
                        controls=segment.prepared.controls, thermo=segment.thermo))
                    md_steps += parameters.md.steps
                    md_time_fs += parameters.md.steps*segment.prepared.timestep_fs
            final, controls = atoms, dict(handoff['evidence'], md_steps=md_steps, md_time_fs=md_time_fs)
        write_extxyz_structures(staging/'final_structure.extxyz', [final])
        result = dict(schema='dice.sample.gcmc-result', revision=1, task_id=task.task_id,
                      sampling_role_id=task.sampling_role_id, selection_group_id=task.selection_group_id,
                      condition_group_id=task.condition_group_id,
                      member_ids=list(evidence.models.member_ids), observations=rows, acquisition=controls,
                      md_segments=md_segments,
                      files={name: _digest(staging/name) for name in ('observations.extxyz', 'final_structure.extxyz')})
        _json(staging/'gcmc_result.json', result)
        validate_task(staging, task_id=task.task_id)
        os.replace(staging, destination)

    def open_current(self, evidence):
        tasks = []
        for task in self._tasks(evidence):
            root = evidence.sample_dir/'tasks'/task.task_id
            result = validate_task(root, task_id=task.task_id)
            if tuple(result['member_ids']) != tuple(evidence.models.member_ids) or any(
                result[key] != getattr(task, key)
                for key in ('sampling_role_id', 'selection_group_id', 'condition_group_id')
            ):
                raise ValueError('GCMC consumer task/group/member inventory differs')
            tasks.append((root, result))
        return SamplingObservationResult(GcmcObservationSource(tuple(tasks)))

    def validate_current(self, evidence):
        try:
            self.open_current(evidence)
            return SamplingValidationResult(bool(self._tasks(evidence)))
        except (OSError, ValueError, TypeError, KeyError):
            return SamplingValidationResult(False)

    def observe_evidence(self, evidence):
        return observe_path_backed_sampling_evidence(evidence)

    def reconstruct_compatible_legacy(self, evidence):
        raise ValueError('GCMC has no historical sample interpretation; rebuild rN/sample')
