"""Bind scientific controls in ordinary LAMMPS syntax without source markers."""
from dataclasses import dataclass
import ast
import math
import operator
import re

from workflow_core.common.native_protocol import NativeProtocolRequest
from workflow_backends.lammps.protocol import command_spans


@dataclass(frozen=True)
class ResolvedNativeProtocol:
    text: str
    settings: dict[str, dict[str, object]]
    ensemble: str | None
    timestep_ps: float | None
    cadence: int
    total_steps: int | None
    temperature_k: float | None
    pressure_bar: float | None


def _format(value):
    return format(float(value), '.17g')


def _select(candidates, identifier, field):
    selected = [item for item in candidates if identifier is None or item[2][1] == identifier]
    if len(selected) != 1:
        names = ', '.join(item[2][1] for item in candidates) or 'none'
        requested = '' if identifier is None else f' (requested {identifier!r})'
        raise ValueError(f'{field} requires one native target{requested}; candidates: {names}')
    return selected[0]


def _native_number(expression, variables, seen=frozenset()):
    """Resolve finite arithmetic only; dynamic LAMMPS expressions stay unknown."""
    if len(str(expression)) > 512 or len(seen) > 16:
        return None
    def replace(match):
        name = match.group(1) or match.group(2)
        if name in seen or name not in variables:
            raise ValueError('dynamic native expression')
        value = _native_number(variables[name], variables, seen | {name})
        if value is None:
            raise ValueError('dynamic native expression')
        return '(' + repr(value) + ')'
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub}:
            return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.Mod: operator.mod}
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError('dynamic native expression')
    try:
        expression = re.sub(r'\$\{(\w+)\}|\bv_(\w+)\b', replace, str(expression))
        value = evaluate(ast.parse(expression, mode='eval').body)
        return value if math.isfinite(value) else None
    except (ValueError, SyntaxError, TypeError, ArithmeticError, RecursionError):
        return None


def compile_lammps_protocol(text: str, request: NativeProtocolRequest) -> ResolvedNativeProtocol:
    text = _complete_protocol(text, request)
    commands = command_spans(text.splitlines(keepends=True))
    fixes = [item for item in commands if item[2][0] == 'fix' and len(item[2]) >= 4]
    baths = [item for item in fixes if item[2][3] in {'nvt', 'npt', 'langevin', 'temp/berendsen', 'temp/csvr', 'temp/csld'}]
    runs = [item for item in commands if item[2][0] == 'run']
    settings, edits = {}, {}
    variables = {row[1]: ' '.join(row[3:]) for _, _, row in commands
                 if len(row) >= 4 and row[0] == 'variable' and row[2] == 'equal'}
    number = lambda expression: _native_number(expression, variables)
    units = next((row[1] for _, _, row in commands if row[0] == 'units'), 'lj')
    pressure_unit_bar = {'metal': 1., 'real': 1.01325}.get(units)
    time_unit_ps = {'metal': 1., 'real': .001}.get(units)
    temperature, pressure = None, None
    ensemble = baths[0][2][3] if len(baths) == 1 else None
    if not baths and any(row[3]=='nve' for _,_,row in fixes):
        ensemble='nve'
    if ensemble in {'langevin', 'temp/berendsen', 'temp/csvr', 'temp/csld'}:
        ensemble = 'nvt'
    if len(baths) == 1:
        row = baths[0][2]
        offset = row.index('temp', 4) + 1 if row[3] in {'nvt', 'npt'} else 4
        targets = [number(v) for v in row[offset:offset + 2]]
        if len(targets) == 2 and targets[0] == targets[1]:
            temperature = targets[0]
        settings['temperature_k'] = dict(value=temperature, source='native', target=row[1])
        pressures = [number(row[i + 1]) for i in range(4, len(row) - 3)
                     if row[i] in {'iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'xz', 'yz'}
                     and number(row[i + 1]) == number(row[i + 2])]
        if pressures and all(v is not None and v == pressures[0] for v in pressures) and pressure_unit_bar:
            pressure = pressures[0] * pressure_unit_bar
            settings['pressure_bar'] = dict(value=pressure, source='native', target=row[1])
    if request.temperature_k is not None:
        start, end, row = _select(baths, request.thermostat, 'temperature_k')
        row = row.copy()
        offset = row.index('temp', 4) + 1 if row[3] in {'nvt', 'npt'} else 4
        row[offset:offset + 2] = [_format(request.temperature_k)] * 2
        edits[start] = (end, ' '.join(row))
        temperature = float(request.temperature_k)
        settings['temperature_k'] = dict(value=temperature, source='DICE', target=row[1])
    bars = [item for item in fixes if item[2][3] in {'npt', 'nph'}]
    if request.pressure_bar is not None:
        if pressure_unit_bar is None:
            raise ValueError(f'pressure_bar override is unsupported for units {units}')
        start, end, original = _select(bars, request.barostat or request.thermostat, 'pressure_bar')
        row = edits[start][1].split() if start in edits else original.copy()
        axes = [i for i in range(4, len(row) - 3) if row[i] in {'iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'xz', 'yz'}]
        if not axes:
            raise ValueError('pressure_bar requires a native pressure-axis target')
        for i in axes:
            row[i + 1:i + 3] = [_format(request.pressure_bar / pressure_unit_bar)] * 2
        edits[start] = (end, ' '.join(row))
        pressure = float(request.pressure_bar)
        settings['pressure_bar'] = dict(value=pressure, source='DICE', target=row[1])
    if request.ensemble is not None and request.ensemble != ensemble:
        if request.ensemble == 'nvt' and ensemble == 'npt':
            start, end, original = _select(bars, request.barostat or request.thermostat, 'ensemble')
            row = edits[start][1].split() if start in edits else original.copy()
            row[3] = 'nvt'
            stripped, i = row[:4], 4
            while i < len(row):
                if row[i] in {'iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'xz', 'yz'}:
                    i += 4
                elif row[i] in {'couple', 'pchain', 'ploop', 'dilate', 'nreset', 'mtk', 'flip', 'scalexy', 'scalexz', 'scaleyz'}:
                    i += 2
                else:
                    stripped.append(row[i]); i += 1
            edits[start] = (end, ' '.join(stripped))
            ensemble, pressure = 'nvt', None
            settings.pop('pressure_bar', None)
        else:
            raise ValueError(f'cannot bind requested ensemble {request.ensemble} to native {ensemble}; provide a matching controller')
    settings['ensemble'] = dict(value=ensemble, source='DICE' if request.ensemble is not None else 'native')
    if request.group is not None or request.temperature_damping_ps is not None:
        start, end, original = _select(baths, request.thermostat, 'thermostat control')
        row = edits[start][1].split() if start in edits else original.copy()
        if request.group is not None:
            row[2] = request.group
            settings['group'] = dict(value=request.group, source='DICE', target=row[1])
        if request.temperature_damping_ps is not None:
            if time_unit_ps is None:
                raise ValueError(f'temperature_damping_ps override is unsupported for units {units}')
            offset = row.index('temp', 4) + 3 if row[3] in {'nvt', 'npt'} else 6
            row[offset] = _format(request.temperature_damping_ps / time_unit_ps)
            settings['temperature_damping_ps'] = dict(value=request.temperature_damping_ps, source='DICE', target=row[1])
        edits[start] = (end, ' '.join(row))
    if request.pressure_damping_ps is not None or request.pressure_coupling is not None:
        start, end, original = _select(bars, request.barostat or request.thermostat, 'barostat control')
        row = edits[start][1].split() if start in edits else original.copy()
        if row[3] not in {'npt', 'nph'}:
            raise ValueError('barostat settings require an active native barostat')
        axes = [i for i in range(4, len(row) - 3) if row[i] in {'iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'xz', 'yz'}]
        if request.pressure_coupling is not None:
            if len(axes) != 1 or row[axes[0]] not in {'iso', 'aniso', 'tri'}:
                raise ValueError('pressure_coupling override requires one iso/aniso/tri target; retain directional settings in the native protocol')
            row[axes[0]] = request.pressure_coupling
            settings['pressure_coupling'] = dict(value=request.pressure_coupling, source='DICE', target=row[1])
        if request.pressure_damping_ps is not None:
            if time_unit_ps is None or not axes:
                raise ValueError('pressure_damping_ps requires known units and a native pressure target')
            for i in axes:
                row[i + 3] = _format(request.pressure_damping_ps / time_unit_ps)
            settings['pressure_damping_ps'] = dict(value=request.pressure_damping_ps, source='DICE', target=row[1])
        edits[start] = (end, ' '.join(row))
    if request.steps is not None:
        if request.run is None and len(runs) != 1:
            raise ValueError('steps requires one native run or targets.run')
        index = (request.run or 1) - 1
        if not 0 <= index < len(runs):
            raise ValueError('targets.run is outside the native run inventory')
        start, end, row = runs[index]
        row = row.copy(); row[1] = str(request.steps)
        edits[start] = (end, ' '.join(row))
        settings['steps'] = dict(value=request.steps, source='DICE', target=index + 1)
    timestep = next((row[1] for _, _, row in commands if row[0] == 'timestep'), None)
    native_time = number(timestep) if timestep is not None else {'metal': .001, 'real': 1.}.get(units)
    time_unit_ps = {'metal': 1., 'real': .001}.get(units)
    timestep_ps = native_time * time_unit_ps if native_time is not None and time_unit_ps is not None else None
    if request.timestep_ps is not None:
        if time_unit_ps is None:
            raise ValueError(f'timestep_ps override is unsupported for units {units}')
        entries=[item for item in commands if item[2][0]=='timestep']
        if len(entries)>1:
            raise ValueError('timestep_ps override is ambiguous across native stages')
        value='timestep '+_format(request.timestep_ps/time_unit_ps)
        if entries:
            start,end,_row=entries[0];edits[start]=(end,value)
        else:
            # A generated integration setting belongs before the first run.
            insertion=runs[0][0] if runs else len(text.splitlines())
            if insertion in edits:
                end,replacement=edits[insertion]
                edits[insertion]=(end,value+'\n'+replacement)
            else:
                edits[insertion]=(insertion,value)
        timestep_ps=request.timestep_ps
    settings['timestep_ps'] = dict(value=timestep_ps, source='native' if timestep is not None else 'default')
    if request.timestep_ps is not None:settings['timestep_ps']['source']='DICE'
    lines = text.splitlines(keepends=True)
    for start, (end, replacement) in sorted(edits.items(), reverse=True):
        lines[start:end] = [replacement + '\n']
    budgets = [request.steps if (request.steps is not None and i == (request.run or 1) - 1)
               else number(row[1]) if len(row) >= 2 and 'upto' not in row else None
               for i, (_, _, row) in enumerate(runs)]
    total = int(sum(budgets)) if budgets and all(v is not None and v >= 0 and float(v).is_integer() for v in budgets) else None
    if any(row[0] in {'jump','next','if','reset_timestep'} for _,_,row in commands):total=None
    if 'steps' not in settings:settings['steps']=dict(value=total,source='native')
    cadence=request.trajectory_every_steps
    if cadence is None:
        inherited=next((number(row[3]) for _,_,row in commands if row[:3]==['variable','dice_trajectory_every_steps','equal']),None)
        if inherited is None:
            inherited=next((number(row[4]) for _,_,row in commands if row[0]=='dump' and len(row)>4),None)
        cadence=int(inherited) if inherited is not None and inherited>0 and float(inherited).is_integer() else 20
    settings['trajectory_every_steps']=dict(value=cadence,source='DICE' if request.trajectory_every_steps is not None else 'native' if any(row[0]=='dump' for _,_,row in commands) else 'default')
    return ResolvedNativeProtocol(''.join(lines), settings, ensemble, timestep_ps,
        cadence, total, temperature, pressure)


def _complete_protocol(text, request):
    """Generate only an unambiguous, explicitly specified missing MD stage."""
    lines = text.splitlines(keepends=True)
    commands = command_spans(lines)
    runs = [item for item in commands if item[2][0] == 'run']
    fixes = [item for item in commands if item[2][0] == 'fix' and len(item[2]) > 3]
    dynamics = [row for _, _, row in fixes if row[3].split('/')[0] in {
        'nvt', 'npt', 'nph', 'nve', 'langevin', 'temp', 'press', 'rigid', 'move'}]
    additions = []
    if request.ensemble is not None and not dynamics:
        if any(row[0] in {'minimize', 'jump', 'next', 'if'} for _, _, row in commands):
            raise ValueError('cannot generate an MD controller inside an optimization or native control flow')
        units = next((row[1] for _, _, row in commands if row[0] == 'units'), 'lj')
        unit_ps = {'metal': 1., 'real': .001}.get(units)
        pressure_bar = {'metal': 1., 'real': 1.01325}.get(units)
        if unit_ps is None:
            raise ValueError(f'controller generation is unsupported for units {units}')
        identifier = 'dice_integrator'
        while any(row[1] == identifier for _, _, row in fixes):
            identifier += '_'
        controller = f'fix {identifier} {request.group or "all"} {request.ensemble}'
        if request.ensemble in {'nvt', 'npt'}:
            if request.temperature_k is None or request.temperature_damping_ps is None:
                raise ValueError('missing native thermostat requires temperatures_k and temperature_damping_ps')
            controller += ' temp ' + ' '.join((_format(request.temperature_k),
                _format(request.temperature_k), _format(request.temperature_damping_ps / unit_ps)))
        if request.ensemble == 'npt':
            if request.pressure_bar is None or request.pressure_damping_ps is None:
                raise ValueError('missing native barostat requires pressures_bar and pressure_damping_ps')
            target = _format(request.pressure_bar / pressure_bar)
            controller += f' {request.pressure_coupling or "iso"} {target} {target} ' + _format(request.pressure_damping_ps / unit_ps)
        additions.append(controller)
    if not runs and request.steps is not None:
        if any(row[0] in {'minimize', 'jump', 'next', 'if'} for _, _, row in commands):
            raise ValueError('steps cannot generate an MD run inside an optimization or native control flow')
        additions.append(f'run {request.steps}')
    if additions:
        insertion = runs[0][0] if runs else len(lines)
        lines[insertion:insertion] = ['\n' + '\n'.join(additions) + '\n']
    return ''.join(lines)
