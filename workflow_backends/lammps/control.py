"""Resolve a bounded set of task-owned controls in native LAMMPS protocols."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from workflow_core.common.lammps_control import LammpsSamplingOptions

from workflow_backends.lammps.protocol import command_spans, assemble_lammps_protocol


@dataclass(frozen=True)
class ControlledProtocol:
    text: str
    cadence: int
    timestep: float
    settings: dict[str, dict[str, object]]


def _number(value, field):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{field} requires a literal positive finite number') from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f'{field} requires a literal positive finite number')
    return number


def _format_number(value):
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _cadence_fallback(commands):
    dump = next((row for _, row in commands if row[0] == 'dump'), None)
    if dump and len(dump) > 4:
        try:
            value = int(dump[4])
        except ValueError as exc:
            raise ValueError('trajectory cadence requires a literal integer') from exc
        if value <= 0:
            raise ValueError('trajectory cadence must be positive')
        return value, f'recipe:dump {dump[1]}'
    return 20, 'default'


def _rewrite(text, edits):
    lines = text.splitlines(keepends=True)
    ends = {start: end for start, end, _row in command_spans(lines)}
    for start, replacement in sorted(edits.items(), reverse=True):
        lines[start:ends[start]] = [replacement + '\n']
    return ''.join(lines)


def render_lammps_control(text: str, options: dict, *, ensemble: str, temperature: float,
                          pressure: float | None, steps: int, seed: int) -> ControlledProtocol:
    """Return executable protocol and the source of each effective control."""
    options = LammpsSamplingOptions.from_payload(options).to_payload()
    control = options['control']
    historical_placeholders = re.search(r'\$\{?dice_', text) is not None
    if 'mode' not in control or (control['mode'] == 'native' and not historical_placeholders):
        from workflow_core.common.native_protocol import NativeProtocolRequest
        from workflow_backends.lammps.native_protocol import compile_lammps_protocol
        resolved = compile_lammps_protocol(text, NativeProtocolRequest(
            ensemble=ensemble, temperature_k=temperature, pressure_bar=pressure,
            steps=steps, timestep_ps=options.get('timestep_ps'),
            trajectory_every_steps=options.get('trajectory_every_steps'),
            thermostat=control.get('fix_id'), group=control.get('group'),
            temperature_damping_ps=control.get('temperature_damping_ps'),
            pressure_damping_ps=control.get('pressure_damping_ps'),
            pressure_coupling=control.get('pressure_coupling')))
        if resolved.timestep_ps is None:
            raise ValueError('native timestep could not be resolved')
        settings = {name: {'value': value['value'], 'source': value['source']}
                    for name, value in resolved.settings.items()}
        for name, value in resolved.settings.items():
            if 'target' in value:
                settings[name + '_target'] = {'value': value['target'], 'source': 'native target'}
        return ControlledProtocol(assemble_lammps_protocol(resolved.text, validate_existing=True),
            resolved.cadence, resolved.timestep_ps, settings)
    if ensemble not in {'nvt', 'npt'} or (ensemble == 'npt') != (pressure is not None):
        raise ValueError('LAMMPS control requires matching NVT/NPT pressure conditions')
    _number(temperature, 'temperature')
    if pressure is not None and (isinstance(pressure, bool) or not math.isfinite(float(pressure))):
        raise ValueError('pressure must be finite')
    if type(steps) is not int or steps <= 0 or type(seed) is not int or seed <= 0:
        raise ValueError('steps and seed must be positive integers')
    commands = [(start, row) for start, _end, row in command_spans(text.splitlines(keepends=True))]
    for line, row in commands:
        if row[0] == 'timestep' and len(row) != 2:
            raise ValueError(f'line {line + 1}: timestep requires one value')
        if row[:2] == ['variable', 'dice_trajectory_every_steps'] and (len(row) != 4 or row[2] != 'equal'):
            raise ValueError(f'line {line + 1}: trajectory cadence requires one equal-style integer')
    first_run = min((line for line, row in commands if row[0] == 'run'), default=len(text.splitlines()))
    if any(row[0] == 'timestep' and line >= first_run for line, row in commands):
        raise ValueError('timestep must be set before the formal run')
    fixes = [(line, row) for line, row in commands if row[0] == 'fix' and len(row) > 3 and row[3] in {'nvt', 'npt'}]
    if control['mode'] == 'native':
        return _native(text, options, commands, ensemble, temperature, pressure, steps, seed)
    identifier = control.get('fix_id')
    candidates = [item for item in fixes if item[1][1] == identifier] if identifier else fixes
    if identifier and len(candidates) != 1:
        raise ValueError('lammps.control.fix_id must identify one existing NVT/NPT fix')
    if len(candidates) > 1:
        raise ValueError('managed control requires one unambiguous NVT/NPT fix; set fix_id')
    selected = candidates[0] if candidates else None
    units = [row for _, row in commands if row[0] == 'units']
    if units != [['units', 'metal']]:
        raise ValueError('managed control requires units metal')
    for line, row in commands:
        if row[0] in {'if', 'jump', 'next', 'label', 'python', 'shell'} or '$' in row[0]:
            raise ValueError(f'line {line + 1}: managed control flow requires native mode')
        if row[0] == 'fix' and len(row) > 3 and (selected is None or line != selected[0]):
            style = row[3].split('/')[0]
            if style in {'nvt', 'npt', 'nve', 'nph', 'langevin', 'temp', 'press', 'deform', 'rigid', 'move'} or '$' in style:
                raise ValueError(f'line {line + 1}: extra thermostat, barostat or integrator; use native mode')
    if selected:
        row = selected[1]
        if row[4:5] != ['temp'] or len(row) != (8 if row[3] == 'nvt' else 12) or (row[3] == 'npt' and row[8] not in {'iso', 'aniso', 'tri'}):
            raise ValueError(f'line {selected[0] + 1}: managed fix requires supported simple syntax; use native mode')
    runs = [(i, r) for i, r in commands if r[0] == 'run']
    if len(runs) != 1:
        raise ValueError('managed control requires exactly one formal run')
    run_line = runs[0][0]
    if len(runs[0][1]) != 2:
        raise ValueError('managed run requires supported simple syntax')
    if selected and selected[0] >= run_line:
        raise ValueError('managed fix must precede the formal run')
    if selected and any(row[:2] == ['unfix', selected[1][1]] and line < run_line for line, row in commands):
        raise ValueError('managed fix must remain active through the formal run')
    settings = {}
    def resolve(field, inherited=None, default=None):
        if field in control:
            value, origin = control[field], f'lammps.control.{field}'
        elif inherited is not None:
            value, origin = inherited, f'recipe:fix {selected[1][1]}:{field}'
        else:
            value, origin = default, 'default'
        if value is None:
            raise ValueError(f'lammps.control.{field} requires a value or a matching recipe setting')
        settings[field] = {'value': value, 'source': origin}
        return value
    row = selected[1] if selected else None
    group = resolve('group', row[2] if row else None, 'all')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', group):
        raise ValueError('managed group requires a literal identifier; use native mode')
    if row and 'group' not in control:
        settings['group']['source'] = f'recipe:fix {row[1]}:group'
    velocities = [(i, r) for i, r in commands if r[0] == 'velocity']
    group_limit = min([selected[0] if selected else run_line, *(i for i, _ in velocities)])
    if group != 'all' and not any(r[:2] == ['group', group] and i < group_limit for i, r in commands):
        raise ValueError(f'managed group {group} must be defined before its fix')
    damping = _number(resolve('temperature_damping_ps', _number(row[7], 'temperature damping') if row and 'temperature_damping_ps' not in control else None), 'temperature_damping_ps')
    identifier = row[1] if row else 'dice_integrator'
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', identifier):
        raise ValueError('managed fix_id requires a literal identifier; use native mode')
    replacement = f'fix {identifier} {group} {ensemble} temp ${{dice_temperature}} ${{dice_temperature}} {_format_number(damping)}'
    if ensemble == 'npt':
        coupling = resolve('pressure_coupling', row[8] if row and row[3] == 'npt' else None, 'iso')
        pdamp = _number(resolve('pressure_damping_ps', _number(row[11], 'pressure damping') if row and row[3] == 'npt' and 'pressure_damping_ps' not in control else None), 'pressure_damping_ps')
        replacement += f' {coupling} ${{dice_pressure}} ${{dice_pressure}} {_format_number(pdamp)}'
        settings['pressure_bar'] = {'value': pressure, 'source': 'pressures_bar'}
    edits = {runs[0][0]: 'run ${dice_run_steps}'}
    insertions = []
    for command, field in (('timestep', 'timestep_ps'), ('cadence', 'trajectory_every_steps')):
        entries = [(i, r) for i, r in commands if r[0] == command] if command == 'timestep' else [(i, r) for i, r in commands if r[:3] == ['variable', 'dice_trajectory_every_steps', 'equal']]
        if len(entries) > 1:
            raise ValueError(f'duplicate {command} settings')
        inherited = entries[0][1][1 if command == 'timestep' else 3] if entries else None
        fallback, fallback_source = _cadence_fallback(commands) if command == 'cadence' and field not in options and inherited is None else (None, 'default')
        value = options.get(field, inherited if inherited is not None else fallback)
        value = _number(value, field)
        if command == 'cadence' and not value.is_integer():
            raise ValueError('trajectory_every_steps must be a positive integer')
        settings[field] = {'value': value, 'source': f'lammps.{field}' if field in options else 'recipe' if entries else fallback_source}
        if command == 'timestep':
            timestep = value
            if entries:
                edits[entries[0][0]] = f'timestep {_format_number(value)}'
            else:
                insertions.append(f'timestep {_format_number(value)}')
        else:
            cadence = int(value)
            # Cadence is declared before the generated setup/dump.
            declaration = f'variable dice_trajectory_every_steps equal {cadence}'
            if entries:
                edits[entries[0][0]] = declaration
            else:
                insertions_cadence = declaration
    velocities = [(i, r) for i, r in commands if r[0] == 'velocity']
    if len(velocities) > 1 or any(len(r) < 5 or r[2] != 'create' or i >= run_line for i, r in velocities):
        raise ValueError('managed velocity supports one create before run; use native mode')
    initialization = f'velocity {group} create ${{dice_temperature}} ${{dice_seed}} mom yes rot no dist gaussian'
    if velocities:
        velocity_row = velocities[0][1]
        tail = velocity_row[5:]
        if len(tail) % 2 or any(tail[i] not in {'mom', 'rot', 'dist', 'loop', 'sum', 'units'} for i in range(0, len(tail), 2)):
            raise ValueError('managed velocity requires simple create syntax; use native mode')
        edits[velocities[0][0]] = f'velocity {group} create ${{dice_temperature}} ${{dice_seed}}' + (' ' + ' '.join(tail) if tail else '')
    else:
        insertions.append(initialization)
    if selected:
        edits[selected[0]] = replacement
    else:
        insertions.append(replacement)
    edits[runs[0][0]] = '\n'.join([*insertions, 'run ${dice_run_steps}'])
    rendered = _rewrite(text, edits)
    if not any(r[:3] == ['variable', 'dice_trajectory_every_steps', 'equal'] for _, r in commands):
        rendered = insertions_cadence + '\n' + rendered
    settings.update({
        'mode': {'value': 'managed', 'source': 'lammps.control.mode'},
        'velocity_initialization': {'value': 'create', 'source': 'managed'},
        'fix_id': {'value': identifier, 'source': 'recipe' if selected else 'reserved default'},
        'ensemble': {'value': ensemble, 'source': 'kind'},
        'temperature_k': {'value': temperature, 'source': 'temperatures_k'},
        'steps': {'value': steps, 'source': 'steps'},
        'seed': {'value': seed, 'source': 'task seed'},
    })
    if ensemble == 'nvt':
        inactive = {name: control[name] for name in ('pressure_damping_ps', 'pressure_coupling') if name in control}
        if inactive:
            settings['inactive_barostat'] = {'value': inactive, 'source': 'not applicable to nvt'}
    if selected:
        settings['original_control'] = {'value': ' '.join(selected[1]), 'source': f'recipe:line {selected[0] + 1}'}
    return ControlledProtocol(assemble_lammps_protocol(rendered, validate_existing=True), cadence, timestep, settings)


def _native(text, options, commands, ensemble, temperature, pressure, steps, seed):
    aliases = {row[1]: ' '.join(row[3:]) for _, row in commands if row[0] == 'variable' and len(row) >= 4}
    def depends(expression, target, seen=frozenset()):
        names = set(re.findall(r'\$\{([A-Za-z0-9_]+)\}|\bv_([A-Za-z0-9_]+)\b', expression))
        references = {part for pair in names for part in pair if part}
        return target in references or any(name in aliases and name not in seen and
            depends(aliases[name], target, seen | {name}) for name in references)
    for line, row in commands:
        if row[0] == 'fix' and len(row) > 3 and row[3] in {'nvt', 'npt'}:
            if row[3] != ensemble:
                raise ValueError(f'line {line + 1}: native {row[3]} conflicts with DICE {ensemble}')
        if row[0] == 'fix' and len(row) > 3 and row[3] in {'nvt', 'npt', 'langevin'}:
            offset = 5 if row[3] in {'nvt', 'npt'} else 4
            if len(row) < offset + 2 or not all(depends(token, 'dice_temperature') for token in row[offset:offset + 2]):
                raise ValueError(f'line {line + 1}: native thermostat must use ${{dice_temperature}}')
            if row[3] == 'npt':
                for index, token in enumerate(row[8:], 8):
                    if token in {'iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'xz', 'yz'}:
                        if not all(depends(value, 'dice_pressure') for value in row[index + 1:index + 3]) or len(row) < index + 3:
                            raise ValueError(f'line {line + 1}: native pressure targets must use ${{dice_pressure}}')
    rows = [row for _, row in commands]
    if sum(row == ['run', '${dice_run_steps}'] for row in rows) != 1 or sum(row[0] == 'run' for row in rows) != 1:
        raise ValueError('native protocol requires exactly one run ${dice_run_steps}')
    references = ' '.join(' '.join(row) for row in rows if row[0] not in {'print', 'variable'})
    for name in ('dice_temperature',) + (('dice_pressure',) if ensemble == 'npt' else ()):
        if not depends(references, name):
            raise ValueError(f'native protocol must consume ${{{name}}}')
    entries = [(i, r) for i, r in commands if r[0] == 'timestep']
    if len(entries) != 1 or len(entries[0][1]) != 2:
        if 'timestep_ps' not in options or entries:
            raise ValueError('native protocol requires one literal timestep or lammps.timestep_ps')
    timestep = _number(options.get('timestep_ps', entries[0][1][1] if entries else None), 'timestep_ps')
    declarations = [(i, r) for i, r in commands if r[:3] == ['variable', 'dice_trajectory_every_steps', 'equal']]
    if len(declarations) > 1:
        raise ValueError('duplicate trajectory cadence')
    fallback, fallback_source = _cadence_fallback(commands) if not declarations and 'trajectory_every_steps' not in options else (20, 'default')
    cadence_value = options.get('trajectory_every_steps', declarations[0][1][3] if declarations else fallback)
    cadence_number = _number(cadence_value, 'trajectory_every_steps')
    if not cadence_number.is_integer():
        raise ValueError('trajectory cadence must be an integer')
    cadence = int(cadence_number)
    if 'timestep_ps' in options and [r for _, r in commands if r[0] == 'units'] != [['units', 'metal']]:
        raise ValueError('lammps.timestep_ps override requires units metal')
    edits = {}
    if 'timestep_ps' in options and entries:
        edits[entries[0][0]] = f'timestep {_format_number(timestep)}'
    if 'trajectory_every_steps' in options and declarations:
        edits[declarations[0][0]] = f'variable dice_trajectory_every_steps equal {cadence}'
    rendered = _rewrite(text, edits)
    if not entries:
        run_line = next(i for i, row in commands if row[0] == 'run')
        edits[run_line] = f'timestep {_format_number(timestep)}\nrun ${{dice_run_steps}}'
        rendered = _rewrite(text, edits)
    if 'trajectory_every_steps' in options and not declarations:
        rendered = f'variable dice_trajectory_every_steps equal {cadence}\n' + rendered
    settings = {'mode': {'value': 'native', 'source': 'lammps.control.mode'},
                'ensemble': {'value': ensemble, 'source': 'kind'},
                'temperature_k': {'value': temperature, 'source': 'temperatures_k'},
                'steps': {'value': steps, 'source': 'steps'},
                'seed': {'value': seed, 'source': 'task seed'},
                'native_timestep': {'value': timestep, 'source': 'lammps.timestep_ps' if 'timestep_ps' in options else 'recipe'},
                'trajectory_every_steps': {'value': cadence, 'source': 'lammps.trajectory_every_steps' if 'trajectory_every_steps' in options else 'recipe' if declarations else fallback_source}}
    if pressure is not None:
        settings['pressure_bar'] = {'value': pressure, 'source': 'pressures_bar'}
    return ControlledProtocol(assemble_lammps_protocol(rendered, validate_existing=True), cadence, timestep, settings)


def validate_native_ensemble(text, *, ensemble):
    """Check directly stated ensemble contradictions without rewriting recipes."""
    for line, row in [(start, row) for start, _end, row in command_spans(text.splitlines(keepends=True))]:
        if row[0] == 'fix' and len(row) > 3 and row[3] in {'nvt', 'npt'} and row[3] != ensemble:
            raise ValueError(f'line {line + 1}: native {row[3]} conflicts with DICE {ensemble}; select managed control or update the recipe')
