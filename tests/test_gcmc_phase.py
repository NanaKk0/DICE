"""Pinned official FLAMES + backend-owned current-state acquisition controls."""
from contextlib import redirect_stdout
from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes


class Zero(Calculator):
    implemented_properties = ['energy', 'forces']

    def __deepcopy__(self, memo):
        raise TypeError('live evaluator cannot be copied')

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': 0., 'forces': np.zeros((len(atoms), 3))}


@pytest.fixture
def official():
    source = os.environ.get('DICE_FLAMES_SOURCE')
    if source is None:
        pytest.skip('explicit official FLAMES source required; skip is not qualification')
    source = Path(source).resolve(strict=True)
    sys.path.insert(0, str(source))
    try:
        from flames.gcmc import GCMC
        import flames.gcmc as upstream
    finally:
        sys.path.remove(str(source))
    assert Path(upstream.__file__).resolve() == source / 'flames/gcmc.py'
    return GCMC


def inputs():
    host = Atoms('He', positions=[[1., 1., 1.]], cell=[10., 10., 10.], pbc=True)
    host.set_velocities([[.1, .2, .3]])
    guest = Atoms('Ne', positions=[[0., 0., 0.]])
    pressure_bar = units.kB*303 / units.J * units.m**3 / host.get_volume() / 1e5
    return host, guest, pressure_bar


@pytest.mark.parametrize('difference,tolerance,accepted', [(1e-8, 0., False), (2e-5, 2e-3, True), (3e-3, 2e-3, False)])
def test_reference_revalidation_is_bounded_and_does_not_rebase_model_energy(tmp_path, official, difference, tolerance, accepted):
    from workflow_backends.gcmc.phase import McPhase
    host, guest, pressure = inputs()
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119),
                     reference_tolerance_ev=tolerance) as phase:
            checkpoint = phase.checkpoint()
            checkpoint['evidence']['controls']['framework_reference_ev'] += difference
            if not accepted:
                with pytest.raises(ValueError, match='checkpoint controls differ'):
                    phase.restore(checkpoint)
            else:
                phase.restore(checkpoint)
                evidence = phase.evidence()
                assert evidence['controls']['framework_reference_ev'] == 0.
                assert evidence['reference_revalidation_ev']['framework_reference_ev'] == -difference
                assert evidence['controls']['reference_tolerance_ev'] == tolerance


def test_md_handoff_refreshes_cell_and_references_without_resetting_mc_state(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase
    host, guest, pressure = inputs()
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'before', device='cpu', radii_a=np.zeros(119)) as phase:
            phase.advance(137, every=100, observe=lambda _: None)
            checkpoint = phase.checkpoint()
        moved = checkpoint['atoms'].copy()
        moved.positions += .2
        moved.set_cell([[11., 0., 0.], [.3, 10., 0.], [.1, .2, 9.]], scale_atoms=True)
        moved.set_velocities(moved.get_velocities()+.01)
        with McPhase(official, moved, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'after', device='cpu', radii_a=np.zeros(119)) as phase:
            phase.restore_after_md(checkpoint)
            state = phase.checkpoint()
            assert state['mc_rng'] == checkpoint['mc_rng']
            assert state['velocity_rng'] == checkpoint['velocity_rng']
            assert state['next_atom_id'] == checkpoint['next_atom_id']
            assert phase.evidence()['controls']['volume_a3'] == pytest.approx(moved.get_volume())
            np.testing.assert_array_equal(phase.atoms.get_momenta(), moved.get_momenta())
            samples = []
            phase.advance(63, every=100, observe=samples.append)
            assert [a.info['mc_attempts'] for a in samples] == [200]


def test_official_phase_records_current_state_and_empty_boundary_evidence(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    snapshots = []
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119)) as phase:
            phase.advance(200, every=100, observe=snapshots.append)
            phase.advance(200, every=100, observe=snapshots.append)
            evidence = phase.evidence()
            assert evidence['mc_attempts'] == 400
            assert evidence['events']['insertion']['accepts'] > 0
            assert evidence['events']['deletion']['accepts'] > 0
            assert sum(row['attempts'] for row in evidence['events'].values()) == 400
            assert evidence['empty_after_attempts'] > 0
            assert evidence['zero_to_one'] > 0 and evidence['one_to_zero'] > 0
            assert 'flames-empty-state-forced-insertion' in evidence['limitations']
            assert evidence['equilibrium_claim'] is False
            assert phase.atoms.get_velocities()[0].tolist() == host.get_velocities()[0].tolist()
    assert [a.info['mc_attempts'] for a in snapshots] == [100, 200, 300, 400]
    assert all(a.info['phase'] == 'gcmc' for a in snapshots)
    assert all(len(set(a.arrays['dice_atom_id'])) == len(a) for a in snapshots)
    assert all(np.all(a.arrays['dice_atom_id'] > 0) for a in snapshots)
    assert not list((tmp_path/'native').glob('**/*rejected*'))


def test_all_rejected_attempts_still_record_and_stride_does_not_change_chain(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    final_states = []
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        for every in (1, 100):
            samples = []
            with McPhase(official, host, guest, Zero(), temperature_k=303,
                         pressure_bar=pressure, random_seed=139, velocity_seed=912,
                         output_dir=tmp_path/str(every), device='cpu',
                         radii_a=np.full(119, 100.)) as phase:
                phase.advance(137, every=every, observe=samples.append)
                phase.advance(263, every=every, observe=samples.append)
                assert len(samples) == 400 // every
                assert all(len(a) == len(host) for a in samples)
                assert phase.evidence()['events']['insertion'] == {'attempts': 400, 'accepts': 0}
                assert phase.evidence()['empty_after_attempts'] == 400
                final_states.append(phase.atoms)
                for a in samples:
                    np.testing.assert_array_equal(a.positions, host.positions)
    np.testing.assert_array_equal(final_states[0].positions, final_states[1].positions)


def test_molecular_survivor_ids_momenta_and_recording_independence(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, _, pressure = inputs()
    guest = Atoms('CO2', positions=[[0., 0., 0.], [1.16, 0., 0.], [-1.16, 0., 0.]])
    finals = []
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        for every in (1, 100):
            samples = []
            with McPhase(official, host, guest, Zero(), temperature_k=303,
                         pressure_bar=pressure, random_seed=139, velocity_seed=912,
                         output_dir=tmp_path/str(every), device='cpu',
                         radii_a=np.zeros(119)) as phase:
                phase.advance(400, every=every, observe=samples.append)
                finals.append(phase.atoms)
            max_seen = 0
            old = {}
            for frame in samples:
                ids = frame.arrays['dice_atom_id']
                mids = frame.arrays['dice_molecule_id']
                momenta = frame.get_momenta()
                assert len(set(ids)) == len(frame)
                for uid, momentum in zip(ids, momenta):
                    if uid in old:
                        np.testing.assert_array_equal(momentum, old[uid])
                    else:
                        assert uid > max_seen
                max_seen = max(max_seen, int(ids.max()))
                old = dict(zip(ids, momenta.copy()))
                assert mids[0] == 0
                assert (len(frame) - 1) % 3 == 0
                for start in range(1, len(frame), 3):
                    np.testing.assert_array_equal(frame.numbers[start:start+3], guest.numbers)
                    assert mids[start] > 0 and np.all(mids[start:start+3] == mids[start])
                assert len(set(mids[1::3])) == (len(frame)-1)//3
    for key in ('numbers', 'positions', 'momenta', 'dice_atom_id', 'dice_molecule_id'):
        np.testing.assert_array_equal(finals[0].arrays[key], finals[1].arrays[key])


@pytest.mark.parametrize('delta,window,expected_n', [(-2., None, 1), (2., None, 0), (-2., 1., 0)])
def test_energy_window_is_explicit_and_reference_energy_is_not_clipped(
    tmp_path, official, delta, window, expected_n
):
    from workflow_backends.gcmc.phase import McPhase

    class PairEnergy(Zero):
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            # Additive per-atom reference cancels in insertion; cross term does not.
            self.results['energy'] = float(10*len(atoms) + delta*len(atoms)*(len(atoms)-1)/2)

    host, guest, pressure = inputs()
    samples = []
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, PairEnergy(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119),
                     phi=.8, de_max_ev=window) as phase:
            phase.advance(1, every=1, observe=samples.append)
            evidence = phase.evidence()
            assert evidence['controls']['de_max_ev'] == window
            assert evidence['controls']['fugacity_pa'] == pytest.approx(.8*pressure*1e5)
            assert evidence['controls']['max_translation_a'] == 1.5
            assert evidence['controls']['max_rotation_rad'] == pytest.approx(np.pi/2)
            assert samples[0].info['n_guests'] == expected_n
            assert samples[0].info['energy_ev'] == pytest.approx(18. if expected_n else 10.)


def test_loaded_seed_uses_explicit_membership_and_preserves_existing_momenta(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    loaded = host + guest
    loaded.set_array('dice_atom_id', np.array([11, 27]))
    loaded.set_array('dice_molecule_id', np.array([0, 8]))
    loaded.set_velocities([[.1, .2, .3], [.3, -.2, .1]])
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, loaded, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119)) as phase:
            np.testing.assert_array_equal(phase.atoms.get_momenta(), loaded.get_momenta())
            np.testing.assert_array_equal(phase.atoms.arrays['dice_atom_id'], [11, 27])
            samples = []
            phase.advance(100, every=1, observe=samples.append)
            for frame in samples:
                ids = frame.arrays['dice_atom_id']
                assert ids[0] == 11
                assert np.all(ids[~np.isin(ids, [11, 27])] > 27)


@pytest.mark.parametrize('atom_ids,molecule_ids', [([1, 1], [0, 1]), ([1, 2], [1, 0]),
                                                  ([1, 2], [0, -1]), ([1., 2.], [0, 1])])
def test_malformed_loaded_identity_fails_before_native_output(tmp_path, official, atom_ids, molecule_ids):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    loaded = host + guest
    loaded.set_array('dice_atom_id', np.array(atom_ids))
    loaded.set_array('dice_molecule_id', np.array(molecule_ids))
    with pytest.raises(ValueError, match='identity|membership'):
        McPhase(official, loaded, guest, Zero(), temperature_k=303,
                pressure_bar=pressure, random_seed=139, velocity_seed=912,
                output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119))
    assert not (tmp_path/'native').exists()


def test_nonfinite_trial_stops_phase_without_fake_rejection_or_retry(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    class BadTrial(Zero):
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if len(atoms) > 1:
                self.results['energy'] = np.nan

    host, guest, pressure = inputs()
    samples = []
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, BadTrial(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119)) as phase:
            with pytest.raises(ValueError, match='nonfinite'):
                phase.advance(1, every=1, observe=samples.append)
            assert not samples
            assert phase.evidence()['mc_attempts'] == 0
            assert phase.evidence()['events']['insertion'] == {'attempts': 0, 'accepts': 0}
            with pytest.raises(RuntimeError, match='failed phase'):
                phase.advance(1, every=1, observe=samples.append)


def test_guest_limit_aborts_instead_of_rejecting_or_completing(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119),
                     max_guests=1) as phase:
            with pytest.raises(RuntimeError, match='max_guests'):
                phase.advance(100, every=100, observe=lambda _: None)
            assert len(phase.atoms) == 2
            assert phase.evidence()['mc_attempts'] == 1
            assert phase.evidence()['events']['insertion']['accepts'] == 1


def test_plain_checkpoint_replays_exact_rng_identity_and_cumulative_cadence(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    settings = dict(temperature_k=303, pressure_bar=pressure, random_seed=139,
                    velocity_seed=912, device='cpu', radii_a=np.zeros(119))
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, Zero(), output_dir=tmp_path/'original', **settings) as phase:
            phase.advance(137, every=100, observe=lambda _: None)
            checkpoint = phase.checkpoint()
            samples1 = []
            phase.advance(263, every=100, observe=samples1.append)
            final = phase.atoms
            expected = phase.evidence()
        with McPhase(official, checkpoint['atoms'], guest, Zero(),
                     output_dir=tmp_path/'restored', **settings) as replay:
            replay.restore(checkpoint)
            samples2 = []
            replay.advance(263, every=100, observe=samples2.append)
            assert replay.evidence() == expected
            for key in final.arrays:
                np.testing.assert_array_equal(replay.atoms.arrays[key], final.arrays[key])
            assert [a.info['mc_attempts'] for a in samples2] == [200, 300, 400]
            for first, second in zip(samples1, samples2, strict=True):
                for key in first.arrays:
                    np.testing.assert_array_equal(first.arrays[key], second.arrays[key])


def test_closed_exchange_weights_disable_translation_rotation(tmp_path, official):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    weights = dict(insertion=.5, deletion=.5, translation=0., rotation=0., reinsertion=0.)
    with (tmp_path/'log').open('w') as log, redirect_stdout(log):
        with McPhase(official, host, guest, Zero(), temperature_k=303,
                     pressure_bar=pressure, random_seed=139, velocity_seed=912,
                     output_dir=tmp_path/'native', device='cpu', radii_a=np.zeros(119),
                     weights=weights) as phase:
            phase.advance(100, every=100, observe=lambda _: None)
            evidence = phase.evidence()
            assert evidence['controls']['weights'] == weights
            assert evidence['events']['translation']['attempts'] == 0
            assert evidence['events']['rotation']['attempts'] == 0
            assert evidence['events']['deletion']['accepts'] > 0


@pytest.mark.parametrize('weights', [dict(insertion=.5, deletion=.25, translation=.25, rotation=0., reinsertion=0.),
                                    dict(insertion=.25, deletion=.25, translation=.25, rotation=0., reinsertion=.25),
                                    dict(insertion=True, deletion=True, translation=0., rotation=0., reinsertion=0.)])
def test_unqualified_move_weights_fail_before_native_output(tmp_path, official, weights):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    with pytest.raises(ValueError, match='weights'):
        McPhase(official, host, guest, Zero(), temperature_k=303,
                pressure_bar=pressure, random_seed=139, velocity_seed=912,
                output_dir=tmp_path/'native', device='cpu', weights=weights)
    assert not (tmp_path/'native').exists()


@pytest.mark.parametrize('seed', [None, True, -1, 1.5])
def test_phase_requires_explicit_reproducible_rng_seed(tmp_path, official, seed):
    from workflow_backends.gcmc.phase import McPhase

    host, guest, pressure = inputs()
    with pytest.raises(ValueError, match='random_seed'):
        McPhase(official, host, guest, Zero(), temperature_k=303,
                pressure_bar=pressure, random_seed=seed, velocity_seed=912,
                output_dir=tmp_path/'native', device='cpu')
    assert not (tmp_path/'native').exists()
