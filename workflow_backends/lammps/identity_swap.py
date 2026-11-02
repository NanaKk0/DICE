"""Native rendering and intrinsic evidence for one continuous identity-swap task.

The shared LAMMPS queue owns placement. This module never schedules work.
"""
from collections import Counter
from dataclasses import asdict
from io import StringIO
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.calculators.lammps import Prism
from ase.io import read

from workflow_core.common.md_mc import IdentitySwapParameters
from workflow_core.structures.canonical_identity import structure_id_for_atoms


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'MDMC requires a regular file: {path.name}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mc_seed(*, iter_index, round_index, native_seed):
    from workflow_backends.sample_artifacts import explicit_task_seed
    return explicit_task_seed(iter_index=iter_index, round_index=round_index,
        sampling_role_id=f'mc-identity-swap:{native_seed}')


def validate_recipe(text, *, ensemble=None):
    """Closed managed input; arbitrary LAMMPS control flow is not accepted."""
    if len(re.findall(r'(?m)^\s*# DICE_CAPABILITY: md_mc\s*$', text)) != 1:
        raise ValueError('md_mc recipe requires # DICE_CAPABILITY: md_mc')
    lines = [line.split('#', 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    allowed = {'units', 'boundary', 'atom_style', 'variable', 'include', 'timestep',
               'neighbor', 'neigh_modify', 'velocity', 'fix', 'thermo', 'thermo_style'}
    if any(line.split()[0] not in allowed or '&' in line or ';' in line for line in lines):
        raise ValueError('md_mc recipe contains unmanaged native commands')
    includes = [line for line in lines if line.startswith('include ')]
    if includes != ['include dice_task_setup.lammps', 'include dice_md_mc.lammps'] or lines[-1] != includes[-1]:
        raise ValueError('md_mc recipe requires ordered setup and terminal managed dynamics includes')
    fixes = [line for line in lines if line.startswith('fix ')]
    prefix = r'fix (?!dice_)\S+ all '
    temperature = r' temp \$\{dice_temperature\} \$\{dice_temperature\} ([0-9.eE+-]+)'
    nvt = re.fullmatch(prefix+'nvt'+temperature, fixes[0]) if len(fixes) == 1 else None
    npt = re.fullmatch(prefix+'npt'+temperature+r' (iso|aniso|tri) \$\{dice_pressure\} \$\{dice_pressure\} ([0-9.eE+-]+) flip no', fixes[0]) if len(fixes) == 1 else None
    actual_ensemble = 'nvt' if nvt else 'npt' if npt else None
    if actual_ensemble is None or (ensemble is not None and ensemble != actual_ensemble):
        raise ValueError('md_mc recipe requires one matching constant-temperature NVT or NPT fix; NPT requires scalar managed pressure and flip no')
    damping = (nvt.group(1),) if nvt else (npt.group(1), npt.group(3))
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in damping):
        raise ValueError('md_mc thermostat/barostat damping must be positive and finite')
    for command in ('units metal', 'boundary p p p', 'atom_style atomic'):
        definitions = [line for line in lines if line.split()[0] == command.split()[0]]
        if definitions != [command] or lines.index(command) > lines.index(includes[0]):
            raise ValueError(f'md_mc recipe requires {command}')
    variables = [line for line in lines if line.startswith('variable ')]
    if len(variables) != 1 or not re.fullmatch(r'variable dice_trajectory_every_steps equal [1-9]\d*', variables[0]):
        raise ValueError('md_mc recipe requires one positive literal trajectory cadence')
    if lines.index(variables[0]) > lines.index(includes[0]):
        raise ValueError('md_mc trajectory cadence must precede task setup')
    timesteps = [line for line in lines if line.startswith('timestep ')]
    if len(timesteps) != 1 or len(timesteps[0].split()) != 2:
        raise ValueError('md_mc recipe requires one literal timestep')
    dt = float(timesteps[0].split()[1])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('md_mc timestep must be positive and finite')
    velocities = [line for line in lines if line.startswith('velocity ')]
    if len(velocities) != 1 or not re.fullmatch(r'velocity all create \$\{dice_temperature\} \$\{dice_seed\}(?: mom (?:yes|no))?(?: rot (?:yes|no))?(?: dist (?:gaussian|uniform))?', velocities[0]):
        raise ValueError('md_mc recipe requires exactly one managed-seed velocity initialization')
    if not (lines.index(includes[0]) < lines.index(timesteps[0]) < lines.index(velocities[0]) < lines.index(fixes[0])):
        raise ValueError('md_mc recipe initialization order is invalid')
    return {'ensemble': actual_ensemble, 'pressure_mode': npt.group(2) if npt else None,
            'native_timestep': dt, 'cadence': int(variables[0].split()[-1])}


def render(parameters, *, steps, cadence, temperature, type_map, mc_seed):
    a, b = (type_map.index(s)+1 for s in parameters.elements)
    prelude = parameters.every_md_steps - 1
    audit = f'1 1 {cadence}'
    lines = ['variable dice_mc_zero equal 0']
    if prelude:
        lines += [f'fix dice_mc_audit all ave/time {audit} v_dice_mc_zero v_dice_mc_zero file mc_pre.out',
                  f'run {prelude}', 'unfix dice_mc_audit']
    lines += [f'fix dice_swap all atom/swap {parameters.every_md_steps} {parameters.attempts_per_event} {mc_seed} {temperature:.17g} types {a} {b} semi-grand no ke yes',
              f'fix dice_mc_audit all ave/time {audit} f_dice_swap[1] f_dice_swap[2] file mc_stats.out',
              f'run {steps-prelude}',
              'print "DICE_MD_MC_COMPLETE $(step:%.0f) $(f_dice_swap[1]:%.0f) $(f_dice_swap[2]:%.0f)" file mc_terminal.out']
    return '\n'.join(lines)+'\n'


def native_blocks(path, *, expected_atom_count=None):
    """Yield complete dump frames in bounded memory; never skip a malformed gap."""
    with Path(path).open() as stream:
        while True:
            first = stream.readline()
            if not first:
                return
            if first.strip() != 'ITEM: TIMESTEP':
                raise ValueError('invalid MDMC dump boundary')
            header = [first] + [stream.readline() for _ in range(8)]
            if any(not line for line in header):
                raise ValueError('truncated MDMC dump header')
            if header[2].strip() != 'ITEM: NUMBER OF ATOMS' or not header[4].startswith('ITEM: BOX BOUNDS'):
                raise ValueError('invalid MDMC dump header')
            count = int(header[3])
            if count <= 0 or (expected_atom_count is not None and count != expected_atom_count):
                raise ValueError('invalid MDMC atom count')
            rows = [stream.readline() for _ in range(count)]
            if any(not row for row in rows):
                raise ValueError('truncated MDMC dump frame')
            yield ''.join(header+rows)


def frame_from_block(block, context):
    config = context.md_mc
    lines = block.splitlines()
    count = int(lines[3])
    if count != len(config['seed_numbers']):
        raise ValueError('MDMC changed atom count')
    columns = lines[8].split()[2:]
    required = ('id', 'type', 'x', 'y', 'z')
    if not set(required) <= set(columns) or len(columns) != len(set(columns)):
        raise ValueError('MDMC dump missing atom identity/coordinates')
    ids = [int(line.split()[columns.index('id')]) for line in lines[9:]]
    if ids != list(range(1, count+1)):
        raise ValueError('MDMC atom IDs are missing, duplicated or reordered')
    types = [int(line.split()[columns.index('type')]) for line in lines[9:]]
    if any(t < 1 or t > len(config['type_map']) for t in types):
        raise ValueError('MDMC atom type outside active map')
    native = read(StringIO(block), format='lammps-dump-text', specorder=config['type_map'])
    if not np.isfinite(native.cell.array).all() or np.linalg.det(native.cell.array) <= 0:
        raise ValueError('MDMC native cell must be finite and right-handed')
    seed_cell = np.asarray(config['seed_cell'])
    prism = Prism(seed_cell, reduce_cell=False)
    frame = Atoms(numbers=native.numbers, cell=prism.update_cell(native.cell.array), pbc=native.pbc)
    frame.set_positions(prism.vector_to_ase(native.positions, wrap=False))
    seed_numbers = np.asarray(config['seed_numbers'])
    swap_numbers = set(Atoms(config['parameters']['elements']).numbers)
    fixed = np.array([n not in swap_numbers for n in seed_numbers])
    if (Counter(frame.numbers) != Counter(seed_numbers) or np.any(frame.numbers[fixed] != seed_numbers[fixed])
        or any(n not in swap_numbers for n in frame.numbers[~fixed])
        or not np.all(frame.pbc) or not np.isfinite(frame.positions).all()
        or not np.isfinite(frame.cell.array).all() or np.linalg.det(frame.cell.array) <= 0
        or (context.ensemble == 'nvt' and not np.allclose(frame.cell.array, seed_cell, rtol=1e-8, atol=1e-10))):
        raise ValueError('MDMC identity, cell or geometry invariant changed')
    frame.info['md_step'] = int(lines[1])
    frame.info['timestep'] = int(lines[1])
    if {'fx', 'fy', 'fz'} <= set(columns):
        forces = prism.vector_to_ase(native.get_forces(), wrap=False)
        if not np.isfinite(forces).all():
            raise ValueError('MDMC native forces are nonfinite')
        frame.set_array('dice_native_force', forces)
    return frame


def read_audit(root):
    audit = {}
    for name in ('mc_pre.out', 'mc_stats.out'):
        path = root/name
        if not path.is_file():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                try:
                    raw = [float(v) for v in line.split()]
                    if len(raw) != 3 or any(not math.isfinite(v) or int(v) != v for v in raw):
                        raise ValueError('invalid MC audit row')
                    step, attempts, accepts = map(int, raw)
                    if step in audit or min(step, attempts, accepts) < 0 or accepts > attempts:
                        raise ValueError('invalid MC audit counters')
                    audit[step] = (attempts, accepts)
                except ValueError:
                    return audit
    return audit


def finalize(root, *, failure=None, evaluator=None):
    """Commit only the same-attempt contiguous geometry/audit/evaluation prefix."""
    from workflow_backends.sample_artifacts import load_task_context
    root = Path(root)
    context = load_task_context(root/'task_context.json')
    parameters = IdentitySwapParameters(**context.md_mc['parameters'])
    audit = read_audit(root)
    rows = []
    expected_step = context.cadence
    previous_accepts = 0
    error = failure
    try:
        with (root/'md_mc_frames.lammpstrj').open('w') as frames, (root/'md_mc_members.jsonl').open('w') as members:
            for block in native_blocks(root/'traj.lammpstrj', expected_atom_count=len(context.md_mc['seed_numbers'])):
                frame = frame_from_block(block, context)
                step = frame.info['md_step']
                if step == 0:
                    continue
                if step != expected_step or step > context.steps:
                    raise ValueError('MDMC cadence prefix is incomplete')
                attempts, accepts = audit[step]
                if attempts != parameters.planned_attempts(step) or accepts < previous_accepts or accepts-previous_accepts > attempts-parameters.planned_attempts(step-context.cadence):
                    raise ValueError('MDMC audit does not match event schedule')
                geometry = structure_id_for_atoms(frame)
                uncertainty = None
                if len(context.ordered_models) > 1:
                    if evaluator is None:
                        raise ValueError('committee MDMC requires formal-frame evaluation')
                    energies, forces = tuple(evaluator.evaluate((frame,)))[0]
                    energies, forces = np.asarray(energies), np.asarray(forces)
                    if (energies.shape != (len(context.ordered_models),) or forces.shape != (len(context.ordered_models), len(frame), 3)
                        or not np.isfinite(energies).all() or not np.isfinite(forces).all()):
                        raise ValueError('invalid MDMC committee member evidence')
                    uncertainty = float(np.linalg.norm(np.std(forces, axis=0), axis=1).max())
                    members.write(json.dumps({'step': step, 'geometry_id': geometry, 'energies': energies.tolist(), 'forces': forces.tolist()}, allow_nan=False)+'\n')
                    members.flush()
                frames.write(block)
                frames.flush()
                rows.append({'step': step, 'attempts': attempts, 'accepts': accepts, 'geometry_id': geometry, 'uncertainty': uncertainty})
                expected_step += context.cadence
                previous_accepts = accepts
        if failure is None:
            terminal = (root/'mc_terminal.out').read_text().split()
            if (len(terminal) != 4 or terminal[0] != 'DICE_MD_MC_COMPLETE'
                or int(terminal[1]) != context.steps or int(terminal[2]) != parameters.planned_attempts(context.steps)
                or not 0 <= int(terminal[3]) <= int(terminal[2]) or int(terminal[3]) < previous_accepts
                or int(terminal[3])-previous_accepts > parameters.planned_attempts(context.steps)-(rows[-1]['attempts'] if rows else 0)
                or expected_step <= context.steps):
                raise ValueError('MDMC did not complete its MD/MC budget')
    except (OSError, ValueError, KeyError, IndexError, EOFError, RuntimeError) as exc:
        error = str(exc) if error is None else f'{error}; prefix: {exc}'
    files = {name: digest(root/name) for name in (
        'md_mc_frames.lammpstrj', 'seed.xyz', 'conf.lmp', 'input.lammps',
        'dice_task_setup.lammps', 'dice_md_mc.lammps')}
    if len(context.ordered_models) > 1:
        files['md_mc_members.jsonl'] = digest(root/'md_mc_members.jsonl')
    result = {'schema': 'dice.sample.md-mc-result', 'revision': 1, 'task_id': context.task_id,
              'status': 'failed' if error is not None else 'completed', 'failure': error,
              'context_sha256': digest(root/'task_context.json'), 'files': files, 'observations': rows,
              'planned_steps': context.steps,
              'planned_attempts': parameters.planned_attempts(context.steps),
              'completed_steps': context.steps if error is None else None,
              'actual_attempts': parameters.planned_attempts(context.steps) if error is None else None,
              'actual_accepts': int(terminal[3]) if error is None else None,
              'acceptance_ratio': int(terminal[3])/parameters.planned_attempts(context.steps) if error is None else None,
              'integrated_time_fs': context.steps*context.native_timestep*1000 if error is None else None,
              'last_retained_step': rows[-1]['step'] if rows else 0}
    temporary = root/'.md_mc_result.tmp'
    temporary.write_text(json.dumps(result, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(root/'md_mc_result.json')
    return result


def validate_result(root):
    """Consumer validation is bounded, read-only and never evaluates a model."""
    from workflow_backends.sample_artifacts import load_task_context, MdMcSampleTaskContext
    root = Path(root)
    context = load_task_context(root/'task_context.json')
    if not isinstance(context, MdMcSampleTaskContext):
        raise ValueError('MDMC result requires MDMC context')
    digest(root/'md_mc_result.json')
    result = json.loads((root/'md_mc_result.json').read_text())
    if (result.get('schema') != 'dice.sample.md-mc-result' or result.get('revision') != 1
        or result.get('task_id') != context.task_id or result.get('status') not in {'completed', 'failed'}
        or result.get('context_sha256') != digest(root/'task_context.json')):
        raise ValueError('invalid MDMC result identity')
    expected_files = {'md_mc_frames.lammpstrj', 'seed.xyz', 'conf.lmp', 'input.lammps',
                      'dice_task_setup.lammps', 'dice_md_mc.lammps'} | ({'md_mc_members.jsonl'} if len(context.ordered_models) > 1 else set())
    if set(result['files']) != expected_files or any(digest(root/name) != d for name, d in result['files'].items()):
        raise ValueError('MDMC canonical evidence is damaged')
    if (context.rendered_input_sha256 != result['files']['input.lammps']
        or context.setup_sha256 != result['files']['dice_task_setup.lammps']
        or context.recipe_sha256 != result['files']['input.lammps']):
        raise ValueError('MDMC context/input bindings disagree')
    recipe = (root/'input.lammps').read_text()
    protocol = validate_recipe(recipe, ensemble=context.ensemble)
    setup_lines = [line.split('#', 1)[0].strip() for line in (root/'dice_task_setup.lammps').read_text().splitlines()]
    pair = 'deepmd/kk' if context.md_mc['execution_mode'] == 'kokkos' else 'deepmd'
    if ([line for line in setup_lines if line.startswith('pair_style ')] != [f'pair_style {pair} {context.ordered_models[0]}']
        or [line for line in setup_lines if line.startswith('pair_coeff ')] != ['pair_coeff * * '+' '.join(context.md_mc['type_map'])]
        or (pair == 'deepmd/kk' and len(context.ordered_models) != 1)):
        raise ValueError('MDMC driving model/mode binding disagrees with context')
    variables = {parts[1]: parts[3] for line in setup_lines if (parts := line.split()) and parts[0] == 'variable' and len(parts) == 4 and parts[2] == 'equal'}
    if (float(variables.get('dice_temperature', 'nan')) != context.temperature_k
        or int(variables.get('dice_seed', '0')) != context.native_seed
        or int(variables.get('dice_run_steps', '0')) != context.steps
        or (context.ensemble == 'npt' and float(variables.get('dice_pressure', 'nan')) != context.pressure_bar)
        or (context.ensemble == 'nvt' and 'dice_pressure' in variables)):
        raise ValueError('MDMC native conditions disagree with context')
    seed = read(root/'seed.xyz')
    if (protocol['native_timestep'] != context.native_timestep or protocol['cadence'] != context.cadence
        or not np.array_equal(seed.numbers, context.md_mc['seed_numbers'])
        or not np.allclose(seed.cell.array, context.md_mc['seed_cell'], rtol=1e-10, atol=1e-12)
        or not np.all(seed.pbc) or seed.info.get('structure_id') != context.selected_structure_id):
        raise ValueError('MDMC recipe or seed disagrees with context')
    for name, d in zip(context.ordered_models, context.model_digests, strict=True):
        if Path(name).name != name or digest(root/name) != d:
            raise ValueError('MDMC ordered model identity is damaged')
    parameters = IdentitySwapParameters(**context.md_mc['parameters'])
    if (type(result.get('planned_steps')) is not int or result['planned_steps'] != context.steps
        or type(result.get('planned_attempts')) is not int or result['planned_attempts'] != parameters.planned_attempts(context.steps)):
        raise ValueError('MDMC planned budget mismatch')
    rows = result['observations']
    counters = [result['last_retained_step']]
    counters.extend(value for row in rows for value in (row['step'], row['attempts'], row['accepts']))
    if result['status'] == 'completed':
        counters.extend(result[key] for key in ('completed_steps', 'actual_attempts', 'actual_accepts'))
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError('MDMC counters must be nonnegative integers')
    if result['last_retained_step'] != (rows[-1]['step'] if rows else 0):
        raise ValueError('MDMC retained budget mismatch')
    if result['status'] == 'completed':
        if (result['failure'] is not None or result['completed_steps'] != context.steps
            or result['actual_attempts'] != parameters.planned_attempts(context.steps)
            or not 0 <= result['actual_accepts'] <= result['actual_attempts']
            or result.get('acceptance_ratio') != result['actual_accepts']/result['actual_attempts']
            or result.get('integrated_time_fs') != context.steps*context.native_timestep*1000
            or len(rows) != context.steps//context.cadence):
            raise ValueError('MDMC success budget mismatch')
    elif (not isinstance(result['failure'], str) or result['completed_steps'] is not None
          or result['actual_attempts'] is not None or result['actual_accepts'] is not None
          or result.get('acceptance_ratio') is not None or result.get('integrated_time_fs') is not None):
        raise ValueError('MDMC failure incorrectly claims completion')
    member_stream = (root/'md_mc_members.jsonl').open() if len(context.ordered_models) > 1 else None
    previous_accepts = 0
    try:
        seen = 0
        for seen, block in enumerate(native_blocks(root/'md_mc_frames.lammpstrj', expected_atom_count=len(context.md_mc['seed_numbers'])), 1):
            if seen > len(rows):
                raise ValueError('MDMC has unlisted formal frames')
            row = rows[seen-1]
            frame = frame_from_block(block, context)
            step = frame.info['md_step']
            if (row['step'] != step or step != seen*context.cadence or step > context.steps
                or row['geometry_id'] != structure_id_for_atoms(frame)
                or row['attempts'] != parameters.planned_attempts(step)
                or not previous_accepts <= row['accepts'] <= row['attempts']
                or row['accepts']-previous_accepts > row['attempts']-parameters.planned_attempts(step-context.cadence)):
                raise ValueError('invalid MDMC formal frame/audit binding')
            previous_accepts = row['accepts']
            if member_stream is not None:
                evidence = json.loads(member_stream.readline())
                energies, forces = np.asarray(evidence['energies']), np.asarray(evidence['forces'])
                if (evidence['step'] != step or evidence['geometry_id'] != row['geometry_id']
                    or energies.shape != (len(context.ordered_models),) or forces.shape != (len(context.ordered_models), len(frame), 3)
                    or not np.isfinite(energies).all() or not np.isfinite(forces).all()
                    or row['uncertainty'] != float(np.linalg.norm(np.std(forces, axis=0), axis=1).max())):
                    raise ValueError('invalid MDMC member/geometry evidence')
            elif row['uncertainty'] is not None:
                raise ValueError('single-model MDMC cannot claim committee uncertainty')
        if seen != len(rows) or (member_stream is not None and member_stream.readline()):
            raise ValueError('MDMC observation/member inventory mismatch')
        if result['status'] == 'completed' and (result['actual_accepts'] < previous_accepts
            or result['actual_accepts']-previous_accepts > result['actual_attempts']-(rows[-1]['attempts'] if rows else 0)):
            raise ValueError('MDMC terminal acceptance budget mismatch')
    finally:
        if member_stream is not None:
            member_stream.close()
    return context, result


def committed_frames(root):
    context, result = validate_result(root)
    for row, block in zip(result['observations'], native_blocks(Path(root)/'md_mc_frames.lammpstrj', expected_atom_count=len(context.md_mc['seed_numbers'])), strict=True):
        frame = frame_from_block(block, context)
        frame.info.update(mc_attempts=row['attempts'], mc_accepts=row['accepts'], sampling_task_status=result['status'],
                          md_time_fs=row['step']*context.native_timestep*1000, phase='md_mc',
                          ensemble=context.ensemble, temperature_k=context.temperature_k)
        if context.pressure_bar is not None:
            frame.info['pressure_bar'] = context.pressure_bar
        yield frame
