"""Version-bound public FLAMES steps, current-state identity and diagnostics.

This provider-internal phase is not a scheduler, public sample adapter or a
completion authority. No private upstream moves or acceptance are replaced.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from ase import units
from ase.data import vdw_radii

from workflow_backends.gcmc.calculator import SharedCalculator
from workflow_core.structures.gcmc_membership import validate_membership


class McPhase:
    def __init__(self, gcmc_class, host, guest, evaluator, *, temperature_k,
                 pressure_bar, random_seed, velocity_seed, output_dir, device,
                 radii_a=None, phi=1., translate_a=.75, rotate_deg=90.,
                 overlap_scale=.6, de_max_ev=None, max_guests=None, weights=None,
                 reference_tolerance_ev=0.):
        if (isinstance(reference_tolerance_ev, bool) or not np.isfinite(reference_tolerance_ev)
            or reference_tolerance_ev < 0):
            raise ValueError('reference_tolerance_ev must be finite and nonnegative')
        self._reference_tolerance_ev = float(reference_tolerance_ev)
        self._reference_revalidation = {'framework_reference_ev': 0., 'guest_reference_ev': 0.}
        for name, seed in [('random_seed', random_seed), ('velocity_seed', velocity_seed)]:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(f'{name} must be an explicit nonnegative integer')
        names = ('insertion', 'deletion', 'translation', 'rotation', 'reinsertion')
        if weights is None:
            weights = dict(zip(names, (.25, .25, .25, .25, 0.), strict=True))
        if not isinstance(weights, dict) or set(weights) != set(names) or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x < 0
            for x in weights.values()
        ):
            raise ValueError('weights require all five finite nonnegative move probabilities')
        if (abs(sum(weights.values())-1.) > 1e-12 or weights['insertion'] <= 0
            or weights['insertion'] != weights['deletion'] or weights['reinsertion'] != 0):
            raise ValueError('weights require sum one, equal positive exchange and zero reinsertion')
        weights = {name: float(weights[name]) for name in names}
        if max_guests is not None and (
            isinstance(max_guests, bool) or not isinstance(max_guests, int) or max_guests <= 0
        ):
            raise ValueError('max_guests must be a positive integer')
        self._max_guests = max_guests
        self._failed = False
        for name, value in [('temperature_k', temperature_k), ('pressure_bar', pressure_bar),
                            ('phi', phi), ('translate_a', translate_a),
                            ('rotate_deg', rotate_deg), ('overlap_scale', overlap_scale)]:
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if rotate_deg > 180:
            raise ValueError('rotate_deg must not exceed 180')
        if de_max_ev is not None and (
            isinstance(de_max_ev, bool) or not np.isfinite(de_max_ev) or de_max_ev <= 0
        ):
            raise ValueError('de_max_ev must be null or finite and positive')
        if not len(host) or not len(guest) or host.constraints or guest.constraints:
            raise ValueError('MC requires nonempty unconstrained host and guest')
        if not np.all(host.pbc) or not np.isfinite(host.cell.array).all() or host.get_volume() <= 0:
            raise ValueError('MC requires a finite full-rank periodic cell')
        if not np.isfinite(host.positions).all() or not np.isfinite(guest.positions).all():
            raise ValueError('MC positions must be finite')
        for atoms in (host, guest):
            if not np.isfinite(atoms.get_masses()).all() or np.any(atoms.get_masses() <= 0):
                raise ValueError('MC masses must be finite and positive')
        if any(key in guest.arrays for key in ('dice_atom_id', 'dice_molecule_id')):
            raise ValueError('insertion template cannot carry persistent identity')
        host, guest = host.copy(), guest.copy()
        if not any(key in host.arrays for key in ('dice_atom_id', 'dice_molecule_id')):
            host.set_array('dice_atom_id', np.arange(1, len(host)+1, dtype=np.int64))
            host.set_array('dice_molecule_id', np.zeros(len(host), dtype=np.int64))
        self.temperature_k = float(temperature_k)
        self._host_count = validate_membership(host, guest.numbers)
        self._host_ids = host.arrays['dice_atom_id'][:self._host_count].copy()
        self._guest_numbers = guest.numbers.copy()
        self._velocity_rng = np.random.default_rng(velocity_seed)
        self._next_atom_id = int(host.arrays['dice_atom_id'].max()) + 1
        self._next_molecule_id = int(host.arrays['dice_molecule_id'].max()) + 1
        self._attempts = 0
        self._empty_after = self._zero_to_one = self._one_to_zero = 0
        self._events = {name: {'attempts': 0, 'accepts': 0}
                        for name in ('insertion', 'deletion', 'translation', 'rotation', 'reinsertion')}
        guest.set_array('dice_atom_id', np.zeros(len(guest), dtype=np.int64))
        guest.set_array('dice_molecule_id', np.zeros(len(guest), dtype=np.int64))
        for atoms in (host, guest):
            atoms.set_masses(atoms.get_masses())
        if not host.has('momenta'):
            host.set_velocities(self._velocities(host.get_masses()))
        if not np.isfinite(host.get_momenta()).all():
            raise ValueError('MC momenta must be finite')
        guest.set_momenta(np.zeros((len(guest), 3)))
        radii = np.array(vdw_radii if radii_a is None else radii_a, dtype=float, copy=True)
        numbers = np.concatenate((host.numbers, guest.numbers))
        if radii.ndim != 1 or len(radii) <= numbers.max() or (
            not np.isfinite(radii[numbers]).all() or np.any(radii[numbers] < 0)
        ):
            raise ValueError('complete finite nonnegative relevant-element overlap radii required')
        self._sim = gcmc_class(
            model=SharedCalculator(evaluator, context=str(output_dir)),
            framework_atoms=host[:self._host_count], adsorbate_atoms=guest,
            device=device,
            temperature=float(temperature_k), pressure=float(pressure_bar)*1e5,
            fugacity_coeff=float(phi), automatic_supercell=False,
            random_seed=random_seed, vdw_radii=radii, vdw_factor=float(overlap_scale),
            max_translation=2*float(translate_a), max_rotation=np.deg2rad(rotate_deg),
            max_deltaE=float('inf') if de_max_ev is None else float(de_max_ev),
            move_weights=weights,
            output_folder=str(Path(output_dir)), output_to_file=False,
            save_frequency=100, save_rejected=False,
        )
        try:
            self._sim.set_state(host)
            self._sim.n_adsorbates = (len(host)-self._host_count)//len(guest)
        except BaseException:
            self._sim.trajectory.close()
            raise

    def _velocities(self, masses):
        return self._velocity_rng.normal(size=(len(masses), 3)) * np.sqrt(
            units.kB * self.temperature_k / masses[:, None])

    @property
    def atoms(self):
        return self._sim.current_system.copy()

    def advance(self, attempts, *, every, observe):
        if self._failed:
            raise RuntimeError('cannot continue a failed phase; replay from its valid input')
        for name, value in [('attempts', attempts), ('every', every)]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for _ in range(attempts):
            # Any inference, identity, observer or resource failure poisons this
            # phase: its advanced RNG must not be reused as a fresh attempt.
            self._failed = True
            before = self._sim.n_adsorbates
            if self._max_guests is not None and before >= self._max_guests:
                raise RuntimeError('max_guests resource budget reached')
            sizes = {name: len(values) for name, values in self._sim.mov_dict.items()}
            self._sim.step(self._attempts + 1)
            current = self._sim.current_system
            fresh = np.flatnonzero(current.arrays['dice_atom_id'] == 0)
            if len(fresh):
                if len(fresh) != len(self._guest_numbers):
                    raise ValueError('upstream returned an incomplete insertion')
                current.arrays['dice_atom_id'][fresh] = np.arange(
                    self._next_atom_id, self._next_atom_id + len(fresh))
                current.arrays['dice_molecule_id'][fresh] = self._next_molecule_id
                momenta = current.get_momenta()
                masses = current.get_masses()[fresh]
                momenta[fresh] = self._velocities(masses) * masses[:, None]
                current.set_momenta(momenta)
                self._next_atom_id += len(fresh)
                self._next_molecule_id += 1
            if validate_membership(current, self._guest_numbers) != self._host_count or not np.array_equal(
                current.arrays['dice_atom_id'][:self._host_count], self._host_ids
            ):
                raise ValueError('upstream changed host membership')
            changed = [name for name, values in self._sim.mov_dict.items()
                       if len(values) != sizes[name]]
            if len(changed) != 1:
                raise ValueError('upstream attempt accounting is inconsistent')
            move = changed[0]
            self._events[move]['attempts'] += 1
            self._events[move]['accepts'] += int(self._sim.mov_dict[move][-1])
            after = self._sim.n_adsorbates
            self._empty_after += int(after == 0)
            self._zero_to_one += int(before == 0 and after == 1)
            self._one_to_zero += int(before == 1 and after == 0)
            self._attempts += 1
            if self._max_guests is not None and after >= self._max_guests:
                raise RuntimeError('max_guests resource budget reached')
            if self._attempts % every == 0:
                frame = self.atoms
                frame.info.update(phase='gcmc', mc_attempts=self._attempts,
                                  n_guests=after, energy_ev=float(self._sim.current_total_energy))
                observe(frame)
            self._failed = False

    def evidence(self):
        sim = self._sim
        return {
            'mc_attempts': self._attempts, 'events': deepcopy(self._events),
            'empty_after_attempts': self._empty_after,
            'zero_to_one': self._zero_to_one, 'one_to_zero': self._one_to_zero,
            'limitations': ['flames-empty-state-forced-insertion', 'fixed-template-flexible-md'],
            'equilibrium_claim': False,
            'failed': self._failed,
            'reference_revalidation_ev': dict(self._reference_revalidation),
            'controls': {
                'temperature_k': float(sim.T), 'pressure_bar': float(sim.P)/1e5,
                'fugacity_pa': float(sim.fugacity)/units.J,
                'phi': float(sim.fugacity_coeff), 'weights': dict(sim.move_weights),
                'max_translation_a': float(sim.max_translation),
                'max_rotation_rad': float(sim.max_rotation),
                'de_max_ev': None if np.isinf(sim.max_deltaE) else float(sim.max_deltaE),
                'volume_a3': float(sim.V)*units.m**3,
                'framework_reference_ev': float(sim.framework_energy),
                'guest_reference_ev': float(sim.adsorbate_energy),
                'automatic_supercell': False,
                'max_guests': self._max_guests,
                'reference_tolerance_ev': self._reference_tolerance_ev,
            },
        }

    def checkpoint(self):
        """Operational atomic/RNG state only; never a public completion record.

        The owning task must bind model identity and serialize this state
        losslessly before it can be used for durable phase recovery.
        """
        if self._failed:
            raise RuntimeError('cannot checkpoint a failed phase')
        return {
            'revision': 1, 'atoms': self.atoms,
            'mc_rng': deepcopy(self._sim.rnd_generator.bit_generator.state),
            'velocity_rng': deepcopy(self._velocity_rng.bit_generator.state),
            'next_atom_id': self._next_atom_id, 'next_molecule_id': self._next_molecule_id,
            'evidence': self.evidence(),
            'template_numbers': self._sim.adsorbate.numbers.copy(),
            'template_positions': self._sim.adsorbate.positions.copy(),
            'overlap_radii': self._sim.vdw.copy(),
        }

    def restore(self, checkpoint):
        """Restore a fresh phase constructed from this exact atomic checkpoint."""
        if self._failed or self._attempts:
            raise RuntimeError('restore requires a fresh nonfailed phase')
        required = {'revision', 'atoms', 'mc_rng', 'velocity_rng', 'next_atom_id',
                    'next_molecule_id', 'evidence', 'template_numbers',
                    'template_positions', 'overlap_radii'}
        if not isinstance(checkpoint, dict) or set(checkpoint) != required or checkpoint['revision'] != 1:
            raise ValueError('unsupported MC checkpoint')
        atoms = checkpoint['atoms']
        if validate_membership(atoms, self._guest_numbers) != self._host_count:
            raise ValueError('checkpoint host membership differs')
        current = self.atoms
        if (set(atoms.arrays) != set(current.arrays)
            or any(not np.array_equal(atoms.arrays[k], current.arrays[k]) for k in atoms.arrays)
            or not np.array_equal(atoms.cell, current.cell)
            or not np.array_equal(atoms.pbc, current.pbc)):
            raise ValueError('phase was not constructed from the exact checkpoint atoms')
        for key, actual in [('template_numbers', self._sim.adsorbate.numbers),
                            ('template_positions', self._sim.adsorbate.positions),
                            ('overlap_radii', self._sim.vdw)]:
            if not np.array_equal(checkpoint[key], actual, equal_nan=True):
                raise ValueError(f'checkpoint {key} differs')
        evidence = checkpoint['evidence']
        actual = self.evidence()['controls']
        changes = {key: (value, actual.get(key)) for key, value in evidence['controls'].items()
                   if value != actual.get(key)}
        reference_keys = {'framework_reference_ev', 'guest_reference_ev'}
        if (evidence['failed'] or set(evidence['controls']) != set(actual)
            or any(key not in reference_keys or not np.isfinite(old) or not np.isfinite(new)
                   or abs(new-old) > self._reference_tolerance_ev
                   for key, (old, new) in changes.items())):
            raise ValueError(f'checkpoint controls differ or checkpoint belongs to a failed phase: {changes}')
        # Compare numerical reference recomputation, never overwrite/rebase the
        # actual model energy or modify upstream acceptance. CPU defaults exact;
        # qualified GPU providers may bind an explicit numerical parity bound.
        self._reference_revalidation = {key: actual[key]-evidence['controls'][key] for key in reference_keys}
        counts = [checkpoint['next_atom_id'], checkpoint['next_molecule_id'],
                  evidence['mc_attempts'], evidence['empty_after_attempts'],
                  evidence['zero_to_one'], evidence['one_to_zero']]
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts):
            raise ValueError('checkpoint counters must be nonnegative integers')
        if checkpoint['next_atom_id'] < self._next_atom_id or checkpoint['next_molecule_id'] < self._next_molecule_id:
            raise ValueError('checkpoint would reuse retired identities')
        events = evidence['events']
        if set(events) != set(self._events) or any(
            set(row) != {'attempts', 'accepts'}
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in row.values())
            or row['accepts'] > row['attempts'] for row in events.values()
        ) or sum(row['attempts'] for row in events.values()) != evidence['mc_attempts']:
            raise ValueError('checkpoint event counts are inconsistent')
        if any(x > evidence['mc_attempts'] for x in counts[3:]):
            raise ValueError('checkpoint boundary counts exceed attempts')
        mc_rng, velocity_rng = np.random.default_rng(), np.random.default_rng()
        mc_rng.bit_generator.state = deepcopy(checkpoint['mc_rng'])
        velocity_rng.bit_generator.state = deepcopy(checkpoint['velocity_rng'])
        self._sim.rnd_generator.bit_generator.state = mc_rng.bit_generator.state
        self._velocity_rng = velocity_rng
        self._next_atom_id, self._next_molecule_id = counts[:2]
        self._attempts, self._empty_after, self._zero_to_one, self._one_to_zero = counts[2:]
        self._events = deepcopy(events)

    def restore_after_md(self, checkpoint):
        """Carry MC/controller identity into a freshly constructed MD endpoint.

        The caller has validated a complete fixed-N native block. This is a
        physical phase transition, not permission to recover incompatible work.
        The official constructor has already recomputed its geometry-dependent
        references and volume. No upstream cached values are patched here.
        """
        current = self.atoms
        old = checkpoint['atoms']
        if validate_membership(old, self._guest_numbers) != self._host_count:
            raise ValueError('MD handoff changed host membership')
        for key in ('numbers', 'masses', 'dice_atom_id', 'dice_molecule_id'):
            if not np.array_equal(old.arrays[key], current.arrays[key]):
                raise ValueError(f'MD handoff changed fixed-N atomic identity: {key}')
        old_controls = checkpoint['evidence']['controls']
        actual = self.evidence()['controls']
        geometry_dependent = {'volume_a3', 'framework_reference_ev', 'guest_reference_ev'}
        if set(old_controls) != set(actual) or any(
            value != actual[key] for key, value in old_controls.items() if key not in geometry_dependent
        ):
            raise ValueError('MD handoff changed fixed MC controls')
        restored = deepcopy(checkpoint)
        restored['atoms'] = current
        restored['evidence']['controls'] = actual
        self.restore(restored)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._sim.trajectory.close()
