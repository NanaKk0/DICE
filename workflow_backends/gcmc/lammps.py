"""Backend-owned fixed-N segment transport; recipes own the physical MD."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
from ase.calculators.lammps import Prism
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import iread_lammps_dump_text

from workflow_backends.common.model_files import link_model_file


class LammpsSegmentRunner:
    """One sequential native child in an already activated sampling worker."""
    def __init__(self, *, recipe, models, type_map, mode, executable):
        self.recipe, self.models = Path(recipe), tuple(Path(p) for p in models)
        resolved = shutil.which(str(executable))
        if resolved is None:
            raise ValueError('registered LAMMPS executable is absent from the activated worker environment')
        self.type_map, self.mode, self.executable = tuple(type_map), mode, str(Path(resolved).resolve())
        self.identity = {'backend': 'LAMMPS', 'mode': mode, 'executable': self.executable,
                         'recipe_sha256': hashlib.sha256(self.recipe.read_bytes()).hexdigest(),
                         'model_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.models],
                         'type_map': list(self.type_map)}

    def run(self, atoms, *, parameters, temperature_k, start_step, output_dir):
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
            raise ValueError('LAMMPS MD requires an explicit GPU assignment; no CPU fallback')
        prepared = prepare_segment(atoms, models=self.models, type_map=self.type_map, mode=self.mode,
                                   parameters=parameters, recipe=self.recipe, temperature_k=temperature_k,
                                   start_step=start_step, output_dir=output_dir)
        command = (sys.executable, '-m', 'workflow_backends.gcmc.native_child', '--parent', str(os.getpid()),
                   '--executable', self.executable, '--mode', self.mode)
        with (prepared.root/'native.stdout.log').open('wb') as stdout, (prepared.root/'native.stderr.log').open('wb') as stderr:
            result = subprocess.run(command, cwd=prepared.root, stdout=stdout, stderr=stderr, check=False)
        if result.returncode:
            raise MdSegmentFailure(f'LAMMPS MD segment exited {result.returncode}; diagnostics: {prepared.root}', prepared)
        try:
            return read_segment(prepared)
        except (OSError, ValueError, EOFError, KeyError, IndexError, RuntimeError) as exc:
            raise MdSegmentFailure(f'LAMMPS MD segment validation failed: {exc}; diagnostics: {prepared.root}', prepared) from exc


class MdSegmentFailure(RuntimeError):
    """Failed fixed-N block with a readable, strictly validated prefix.

    Iteration raises at the first invalid/missing observation. Consumers may
    retain frames yielded before that error, but must not resume this block.
    """
    def __init__(self, message, prepared):
        super().__init__(message)
        self.prepared = prepared

    @property
    def observations(self):
        return _native_frames(self.prepared, 'trajectory.dump', observations=True)


@dataclass(frozen=True)
class PreparedSegment:
    root: Path
    seed: object
    prism: object
    type_map: tuple[str, ...]
    expected_steps: tuple[int, ...]
    start_step: int
    end_step: int
    timestep_fs: float
    mode: str
    controls: dict


@dataclass(frozen=True)
class MdSegmentResult:
    prepared: PreparedSegment
    endpoint: object
    thermo: dict

    @property
    def observations(self):
        # Re-readable and streaming; do not hold all native snapshots in RAM.
        return _native_frames(self.prepared, 'trajectory.dump', observations=True)


def _dump_frames(prepared, name):
    # ase.io.iread routes this format through a list-building reader: one
    # truncated tail would hide all earlier frames and retain the dump in RAM.
    with (prepared.root/name).open() as stream:
        yield from iread_lammps_dump_text(stream, index=slice(None),
                                         specorder=list(prepared.type_map), units='metal')


def _native_frames(prepared, name, *, observations):
    prism = Prism(prepared.seed.cell.array, pbc=prepared.seed.pbc, reduce_cell=False)
    expected = iter(prepared.expected_steps)
    count = 0
    for native in _dump_frames(prepared, name):
        step = int(native.info['timestep'])
        if observations and step == prepared.start_step:
            continue
        if observations:
            if step != next(expected, None):
                raise ValueError('native MD observation cadence is incomplete or duplicated')
        elif step != prepared.end_step or count:
            raise ValueError('native MD endpoint is missing or has the wrong timestep')
        if (len(native) != len(prepared.seed)
            or not np.array_equal(native.numbers, prepared.seed.numbers)
            or not np.array_equal(native.arrays.get('v_dice_native_id'), np.arange(1, len(native)+1))
            or not native.has('momenta') or not np.all(native.pbc)):
            raise ValueError('native MD atom inventory, velocities or PBC changed')
        if any(not np.isfinite(value).all() for value in (native.positions, native.cell.array,
                                                          native.get_velocities(), native.get_forces())):
            raise ValueError('native MD state contains nonfinite values')
        frame = prepared.seed.copy()
        frame.calc = None
        frame.set_cell(prism.update_cell(native.cell.array), scale_atoms=False)
        # Never let ASE's default per-atom wrapping split a guest that straddles
        # a periodic face. xu/image flags preserve its continuous coordinates.
        frame.set_positions(prism.vector_to_ase(native.positions, wrap=False))
        frame.set_velocities(prism.vector_to_ase(native.get_velocities(), wrap=False))
        frame.set_array('dice_native_force', prism.vector_to_ase(native.get_forces(), wrap=False))
        if frame.cell.rank != 3 or frame.get_volume() <= 0:
            raise ValueError('native MD returned an invalid cell')
        if prepared.controls['ensemble'] != 'npt' and not np.allclose(
            frame.cell.array, prepared.seed.cell.array, atol=1e-10, rtol=1e-10
        ):
            raise ValueError('fixed-cell MD changed the cell')
        frame.info.update(phase='md', md_steps=step, phase_step=step-prepared.start_step)
        count += 1
        yield frame
    if observations:
        if next(expected, None) is not None:
            raise ValueError('native MD observation cadence is incomplete')
    elif count != 1:
        raise ValueError('native MD endpoint is absent')


def read_segment(prepared):
    columns, first, last, completed = None, None, None, False
    with (prepared.root/'log.lammps').open() as stream:
        for line in stream:
            if line.startswith('ERROR'):
                raise ValueError('native MD reported an error')
            if line.strip() == 'DICE_MD_SEGMENT_COMPLETE':
                completed = True
            fields = line.split()
            if fields and fields[0] == 'Step' and 'PotEng' in fields:
                columns = fields
                continue
            if columns and len(fields) == len(columns):
                try:
                    numbers = [float(value) for value in fields]
                except ValueError:
                    continue
                if not np.isfinite(numbers).all():
                    raise ValueError('native MD thermo is nonfinite')
                row = dict(zip(columns, numbers))
                first = row if first is None else first
                last = row
    if (not completed or first is None or last is None
        or first['Step'] != prepared.start_step or last['Step'] != prepared.end_step):
        raise ValueError('native MD did not reach its requested endpoint')
    endpoint, = tuple(_native_frames(prepared, 'endpoint.dump', observations=False))
    # Fully validate before exposing a successful segment; replay stays bounded.
    for _ in _native_frames(prepared, 'trajectory.dump', observations=True):
        pass
    return MdSegmentResult(prepared, endpoint, {'first': first, 'last': last,
                           'total_energy_change_ev': last['TotEng']-first['TotEng']})


def _positive(value, label):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f'{label} must be finite and positive')
    return number


def validate_recipe(path, parameters, temperature_k):
    """Resolve the closed recipe without loading a model or launching MD."""
    substitutions = {'${dice_temperature_k}': str(float(temperature_k))}
    if parameters.pressure_bar is not None:
        substitutions['${dice_pressure_bar}'] = str(float(parameters.pressure_bar))
    lines, commands = [], {}
    for raw in Path(path).read_text().splitlines():
        for key, value in substitutions.items():
            raw = raw.replace(key, value)
        parts = shlex.split(raw, comments=True)
        if not parts:
            continue
        if any('$' in item or '&' in item or ';' in item for item in parts):
            raise ValueError('unsupported MD recipe substitution or command continuation')
        command = parts[0]
        if command not in {'timestep', 'neighbor', 'neigh_modify', 'fix'} or command in commands:
            raise ValueError(f'managed or unsupported MD recipe command: {command}')
        commands[command] = parts
        lines.append(' '.join(parts))
    if set(commands) != {'timestep', 'neighbor', 'neigh_modify', 'fix'}:
        raise ValueError('MD recipe requires timestep, neighbor, neigh_modify and one integration fix')
    if len(commands['timestep']) != 2:
        raise ValueError('MD timestep requires one constant in metal units')
    timestep_fs = 1000*_positive(commands['timestep'][1], 'MD timestep')
    neighbor = commands['neighbor']
    if len(neighbor) != 3 or neighbor[2] not in {'bin', 'nsq'}:
        raise ValueError('MD neighbor requires an explicit skin and supported style')
    _positive(neighbor[1], 'neighbor skin')
    rebuild = commands['neigh_modify'][1:]
    if len(rebuild) != 6 or set(rebuild[::2]) != {'every', 'delay', 'check'}:
        raise ValueError('MD neigh_modify requires every, delay and check')
    rebuild = dict(zip(rebuild[::2], rebuild[1::2]))
    if (not rebuild['every'].isdigit() or int(rebuild['every']) <= 0
        or not rebuild['delay'].isdigit() or int(rebuild['delay']) % int(rebuild['every'])
        or rebuild['check'] not in {'yes', 'no'}):
        raise ValueError('unsupported MD neighbor rebuild controls')
    fix = commands['fix']
    if (len(fix) < 4 or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', fix[1])
        or fix[2] != 'all' or fix[3] != parameters.ensemble):
        raise ValueError('MD recipe must integrate all atoms with the declared ensemble')
    controls = {'ensemble': parameters.ensemble, 'cell_freedoms': (), 'cell_coupling': 'none',
                'controller_history': 'not_applicable' if parameters.ensemble == 'nve' else 'fresh_per_segment',
                'timestep_fs': timestep_fs, 'neighbor': neighbor[1:], 'neigh_modify': rebuild}
    options = fix[4:]
    if parameters.ensemble == 'nve':
        if options:
            raise ValueError('unsupported NVE fix options')
    else:
        if len(options) < 4 or options[0] != 'temp' or any(float(v) != temperature_k for v in options[1:3]):
            raise ValueError('MD thermostat must use the task temperature at both endpoints')
        controls['thermostat_tau_fs'] = 1000*_positive(options[3], 'thermostat damping')
        options = options[4:]
        if parameters.ensemble == 'nvt' and options:
            raise ValueError('unsupported NVT fix options')
        if parameters.ensemble == 'npt':
            if len(options) < 4 or options[0] not in {'iso', 'aniso', 'tri'}:
                raise ValueError('NPT recipe requires explicit iso, aniso or tri cell freedoms')
            if any(float(v) != parameters.pressure_bar for v in options[1:3]):
                raise ValueError('NPT recipe pressure differs from explicit md.pressure_bar')
            freedoms = {'iso': ('isotropic_volume',), 'aniso': ('x', 'y', 'z'),
                        'tri': ('x', 'y', 'z', 'xy', 'xz', 'yz')}
            controls.update(cell_freedoms=freedoms[options[0]],
                            cell_coupling='xyz' if options[0] == 'iso' else 'none',
                            pressure_bar=parameters.pressure_bar,
                            barostat_tau_fs=1000*_positive(options[3], 'barostat damping'))
            if options[4:]:
                raise ValueError('unsupported NPT fix options; cell controls must remain explicit')
    return tuple(lines), controls


def prepare_segment(seed, *, models, type_map, mode, parameters, recipe,
                    temperature_k, start_step, output_dir):
    lines, controls = validate_recipe(recipe, parameters, temperature_k)
    if (isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0
        or mode not in {'ordinary', 'kokkos'} or not models):
        raise ValueError('invalid native MD segment controls')
    _positive(temperature_k, 'temperature')
    if (not len(seed) or seed.constraints or not np.all(seed.pbc) or seed.cell.rank != 3
        or not seed.has('momenta') or not np.isfinite(seed.positions).all()
        or not np.isfinite(seed.cell.array).all() or not np.isfinite(seed.get_momenta()).all()):
        raise ValueError('native MD requires finite full-rank periodic state and explicit unconstrained momenta')
    if (not type_map or len(set(type_map)) != len(type_map)
        or not set(seed.get_chemical_symbols()) <= set(type_map)):
        raise ValueError('native MD type map does not cover the atomic state')
    masses = seed.get_masses()
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('native MD masses must be finite and positive')
    for symbol in set(seed.get_chemical_symbols()):
        selected = masses[np.asarray(seed.get_chemical_symbols()) == symbol]
        if not np.all(selected == selected[0]):
            raise ValueError('native atomic type mapping cannot represent different masses for one element')
    for path in models:
        if Path(path).is_symlink() or not Path(path).is_file():
            raise ValueError('native MD models must be existing regular prepared files')
    if mode == 'kokkos' and (len(models) != 1 or Path(models[0]).suffix != '.pt2'):
        raise ValueError('Kokkos MD requires exactly one qualified PT2 model')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    prism = Prism(seed.cell.array, pbc=seed.pbc, reduce_cell=False)
    transport = seed.copy()
    # ASE gives a pre-existing native `type` array precedence over specorder.
    # Rebind only the transport copy; DICE identity/membership remain untouched.
    transport.set_array('type', np.asarray([type_map.index(s)+1 for s in seed.get_chemical_symbols()], dtype=int))
    with (root/'state.data').open('w') as stream:
        write_lammps_data(stream, transport, specorder=list(type_map), prismobj=prism,
                          atom_style='atomic', units='metal', masses=True, velocities=True,
                          write_image_flags=True, force_skew=parameters.ensemble == 'npt')
    names = []
    for index, path in enumerate(models):
        name = f'model_{index:03d}'+Path(path).suffix
        link_model_file(path, root/name)
        names.append(name)
    pair = ('deepmd/kk' if mode == 'kokkos' else 'deepmd')+' '+' '.join(names)
    if len(models) > 1:
        pair += f' out_file model_devi.out out_freq {parameters.every}'
    columns = 'id type xu yu zu vx vy vz fx fy fz v_dice_native_id'
    text = ['clear', 'units metal', 'boundary p p p',
            'atom_style atomic/kk' if mode == 'kokkos' else 'atom_style atomic',
            'atom_modify map yes', 'read_data state.data',
            'run_style verlet/kk' if mode == 'kokkos' else 'run_style verlet',
            'pair_style '+pair, 'pair_coeff * * '+' '.join(type_map),
            f'reset_timestep {start_step}', *lines,
            'variable dice_native_id atom id', 'thermo 1',
            'thermo_style custom step temp pe ke etotal press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz',
            'thermo_modify format float %.17g lost error',
            f'dump observations all custom {parameters.every} trajectory.dump {columns}',
            'dump_modify observations sort id format float %.17g',
            'run 0', f'run {parameters.steps}',
            f'write_dump all custom endpoint.dump {columns} modify sort id format float %.17g',
            'print "DICE_MD_SEGMENT_COMPLETE"']
    (root/'input.lammps').write_text('\n'.join(text)+'\n')
    return PreparedSegment(root, seed.copy(), prism, tuple(type_map),
                           tuple(range((start_step//parameters.every+1)*parameters.every,
                                       start_step+parameters.steps+1, parameters.every)),
                           start_step, start_step+parameters.steps, controls['timestep_fs'], mode, controls)
