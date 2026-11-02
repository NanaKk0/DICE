"""Incremental scientific commits and strict failed-task consumer validation."""
import os
import math
import numpy as np

from workflow_backends.common.task_outcomes import TaskOutcome
from workflow_core.structures.formats import read_extxyz_structures
from workflow_core.structures.canonical_identity import structure_id_for_atoms


def commit_step(owner, evidence, task, staging, step, snapshot, models):
    root = staging / 'accepted'
    root.mkdir(exist_ok=True)
    temporary = root / f'.step_{step:08d}'
    target = root / f'step_{step:08d}'
    temporary.mkdir()
    owner._write_frames(temporary / 'frame.extxyz', (snapshot.frame,))
    # Geometry survives even if another committee member fails during evaluation.
    os.replace(temporary, target)
    rows, frames = owner._evaluate_observations(evidence,
        task_index=int(task.task_id.removeprefix('task_')), task=task,
        observations=((step, snapshot),), models=models,
        seed_id=structure_id_for_atoms(snapshot.frame) if step == 0 else
            owner._read_json(staging / 'task_context.json')['seed_structure_id'],
        relaxation_status=(False, 'failed'))
    # Use the raw-seed identity consistently, including step zero with constraints.
    owner._write_frames(target / 'observation.extxyz', frames)
    payload = {'row': rows[0], 'frame_sha256': owner._digest(target / 'frame.extxyz'),
               'observation_sha256': owner._digest(target / 'observation.extxyz')}
    owner._write_json(target / '.evidence.json', payload)
    os.replace(target / '.evidence.json', target / 'evidence.json')


def _accepted(owner, directory, context):
    root = directory / 'accepted'
    steps = sorted(root.glob('step_*')) if root.is_dir() else []
    raw = read_extxyz_structures(directory / 'raw_seed.extxyz', allow_empty=False)
    if len(raw) != 1 or structure_id_for_atoms(raw[0]) != context['seed_structure_id']:
        raise ValueError('failed relax seed identity mismatch')
    rows, frames, files = [], [], {}
    for index, path in enumerate(steps):
        if path.name != f'step_{index:08d}' or path.is_symlink():
            raise ValueError('failed relax accepted-step inventory is not contiguous')
        frame = read_extxyz_structures(path / 'frame.extxyz', allow_empty=False)
        if len(frame) != 1:
            raise ValueError('invalid accepted frame')
        owner._validate_frame_family(raw[0], frame, tuple(context['parameters']['fixed_atoms']))
        files[str((path / 'frame.extxyz').relative_to(directory))] = owner._digest(path / 'frame.extxyz')
        if not (path / 'evidence.json').is_file():
            if index != len(steps) - 1:
                raise ValueError('missing interior committee evidence')
            continue
        record = owner._read_json(path / 'evidence.json')
        if (record.get('frame_sha256') != owner._digest(path / 'frame.extxyz') or
                record.get('observation_sha256') != owner._digest(path / 'observation.extxyz')):
            raise ValueError('failed relax commit digest mismatch')
        row = record['row']
        observed = read_extxyz_structures(path / 'observation.extxyz', allow_empty=False)
        if len(observed) != 1 or not np.array_equal(observed[0].positions, frame[0].positions):
            raise ValueError('failed relax observation geometry mismatch')
        owner._validate_frame_family(raw[0], observed, tuple(context['parameters']['fixed_atoms']))
        for key in ('task_id', 'task_index', 'sampling_role_id', 'selection_group_id',
                    'condition_group_id', 'source_frame_index'):
            if row.get(key) != context[key]:
                raise ValueError('failed relax observation task identity mismatch')
        energies = np.asarray(row.get('member_energies_ev'), dtype=float)
        forces = np.asarray(row.get('member_forces_ev_a'), dtype=float)
        if (energies.shape != (len(context['models']),) or
                forces.shape != (len(context['models']), len(raw[0]), 3) or
                not np.isfinite(energies).all() or not np.isfinite(forces).all() or
                not math.isclose(float(row['force_deviation']), owner._force_deviation(forces), abs_tol=1e-12)):
            raise ValueError('failed relax ordered committee evidence is incomplete')
        from types import SimpleNamespace
        identity = owner._observation_identity(SimpleNamespace(
            run_json_path=__import__('pathlib').Path(context['run_json_path']),
            iter_index=context['iter'], round_index=context['round']),
            task_id=context['task_id'], optimizer_step=index, seed_id=context['seed_structure_id'])
        if (row.get('optimizer_step') != index or row.get('structure_id') != identity or
                observed[0].info.get('structure_id') != identity or
                observed[0].info.get('relax_termination') != 'failed' or
                observed[0].info.get('relax_converged') is not False):
            raise ValueError('failed relax step/observation identity mismatch')
        for name in ('evidence.json', 'observation.extxyz'):
            files[str((path / name).relative_to(directory))] = owner._digest(path / name)
        if index % context['parameters']['history_stride'] == 0:
            rows.append(dict(row, frame_index=len(rows)))
            frames.append(observed[0])
    eligible = context['parameters']['include_history'] and not (
        len(steps) > context['parameters']['max_steps'] and
        not context['parameters'].get('accept_unconverged', False))
    return (rows if eligible else []), (frames if eligible else []), files


def retain_failure(owner, evidence, task, error):
    directory = evidence.sample_dir / 'tasks' / f'.{task.task_id}.staging'
    context = owner._read_json(directory / 'task_context.json')
    rows, frames, files = _accepted(owner, directory, context)
    owner._write_frames(directory / 'observations.extxyz', frames)
    for name in ('task_context.json', 'raw_seed.extxyz', 'observations.extxyz'):
        files[name] = owner._digest(directory / name)
    result = dict(schema='dice.lifecycle.relax-task-result', schema_revision=2,
        task_id=task.task_id, termination='failed', converged=False,
        diagnostic=str(error)[-4096:], observations=rows, files=files)
    owner._write_json(directory / 'relax_result.json', result)
    validate_failure(owner, directory, context, result)
    os.replace(directory, evidence.sample_dir / 'tasks' / task.task_id)
    return TaskOutcome(task.task_id, 'failed', tuple(row['structure_id'] for row in rows), str(error)[-4096:])


def validate_failure(owner, directory, expected, result):
    if (owner._read_json(directory / 'task_context.json') != expected or
            result.get('schema') != 'dice.lifecycle.relax-task-result' or
            result.get('task_id') != expected['task_id'] or
            result.get('converged') is not False or result.get('termination') != 'failed' or
            (directory / 'task_completion_receipt.json').exists()):
        raise ValueError('invalid failed relax terminal evidence')
    rows, frames, files = _accepted(owner, directory, expected)
    for name in ('task_context.json', 'raw_seed.extxyz', 'observations.extxyz'):
        files[name] = owner._digest(directory / name)
    if files != result.get('files') or rows != result.get('observations'):
        raise ValueError('failed relax terminal inventory mismatch')
    actual = read_extxyz_structures(directory / 'observations.extxyz', allow_empty=True)
    if len(actual) != len(frames) or any(
        not np.array_equal(a.positions, b.positions) or a.info != b.info
        for a, b in zip(actual, frames)):
        raise ValueError('failed relax catalog mismatch')
    return rows
