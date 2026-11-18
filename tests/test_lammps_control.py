"""Public LAMMPS protocol control: physical settings, sources and overrides."""
import pytest


def test_historical_native_mode_accepts_clean_numeric_protocol_and_explicit_overrides():
    from workflow_backends.lammps.control import render_lammps_control
    result = render_lammps_control('units metal\ntimestep .001\nvelocity all create 350 917\n'
        'fix bath all nvt temp 300 300 .1 drag .2\nrun 80\n',
        {'control':{'mode':'native'}}, ensemble='nvt', temperature=600,
        pressure=None, steps=40, seed=17)
    assert 'fix bath all nvt temp 600 600 .1 drag .2' in result.text
    assert 'velocity all create 350 917' in result.text
    assert 'run 40' in result.text


def test_managed_nvt_replaces_npt_and_preserves_mobile_and_constraints():
    from workflow_backends.lammps.control import render_lammps_control

    source = (
        'units metal\natom_style atomic\nboundary p p p\n'
        'variable dice_trajectory_every_steps equal 20\n'
        'group mobile type 1\ngroup fixed subtract all mobile\n'
        'fix freeze fixed setforce 0 0 0\ntimestep 0.001\n'
        'fix bath mobile npt temp 1200 1200 0.1 iso 1 1 1.0\n'
        'run 999\nunfix bath\n'
    )
    result = render_lammps_control(source, {'control': {'mode': 'managed'}},
        ensemble='nvt', temperature=600, pressure=None, steps=40, seed=17)
    assert 'fix bath mobile nvt temp ${dice_temperature} ${dice_temperature} 0.1' in result.text
    assert 'fix freeze fixed setforce 0 0 0' in result.text
    assert 'run ${dice_run_steps}' in result.text
    assert result.settings['group'] == {'value': 'mobile', 'source': 'recipe:fix bath:group'}
    assert result.settings['temperature_k']['value'] == 600
    assert result.cadence == 20


def test_native_langevin_retains_protocol_and_rejects_direct_ensemble_conflict():
    from workflow_backends.lammps.control import render_lammps_control

    source = ('units metal\natom_style atomic\ntimestep 0.001\n'
              'fix integrate mobile nve\n'
              'fix bath mobile langevin ${dice_temperature} ${dice_temperature} 0.1 ${dice_seed}\n'
              'run ${dice_run_steps}\n')
    result = render_lammps_control(source, {'control': {'mode': 'native'}},
        ensemble='nvt', temperature=600, pressure=None, steps=40, seed=17)
    assert result.text.replace('include dice_task_setup.lammps\n', '') == source
    assert result.settings['mode']['value'] == 'native'
    conflicting = source.replace('fix integrate mobile nve',
        'fix integrate mobile npt temp ${dice_temperature} ${dice_temperature} 0.1 iso ${dice_pressure} ${dice_pressure} 1')
    with pytest.raises(ValueError, match='line .*npt.*nvt'):
        render_lammps_control(conflicting, {'control': {'mode': 'native'}},
            ensemble='nvt', temperature=600, pressure=None, steps=40, seed=17)
    assert result.timestep == 0.001


def test_managed_npt_generates_dynamics_and_records_explicit_sources():
    from workflow_backends.lammps.control import render_lammps_control

    source = 'units metal\natom_style atomic\nboundary p p p\nthermo 50\nrun ${dice_run_steps}\n'
    result = render_lammps_control(source, {
        'control': {'mode': 'managed', 'temperature_damping_ps': 0.2,
                    'pressure_damping_ps': 2.0},
        'timestep_ps': 0.002, 'trajectory_every_steps': 20,
    }, ensemble='npt', temperature=900, pressure=5, steps=40, seed=17)
    assert 'fix dice_integrator all npt temp ${dice_temperature} ${dice_temperature} 0.2 iso ${dice_pressure} ${dice_pressure} 2' in result.text
    assert 'velocity all create ${dice_temperature} ${dice_seed}' in result.text
    assert result.text.index('timestep 0.002') < result.text.index('run ${dice_run_steps}')
    assert result.settings['pressure_bar']['value'] == 5
    assert result.settings['pressure_damping_ps']['source'] == 'lammps.control.pressure_damping_ps'
    assert result.cadence == 20


@pytest.mark.parametrize('body, options, error', [
    ('fix a all nvt temp 1 1 0.1\nfix b all nve\n', {}, 'extra.*integrat'),
    ('fix a all nvt temp 1 1 0.1 drag 0.2\n', {}, 'supported.*syntax'),
    ('fix a all nvt temp 1 1 0.1\n', {'fix_id': 'missing'}, 'fix_id'),
    ('fix a all nvt temp 1 1 0.1\n', {'group': 'missing'}, 'group.*missing'),
    ('fix a all nvt temp 1 1 0.1\nvelocity all set 1 0 0\n', {}, 'velocity'),
    ('fix a all nvt temp 1 1 0.1\nif "1" then "run 200"\n', {}, 'control flow'),
])
def test_managed_control_rejects_ambiguous_physics(body, options, error):
    from workflow_backends.lammps.control import render_lammps_control

    with pytest.raises(ValueError, match=error):
        render_lammps_control('units metal\natom_style atomic\ntimestep 0.001\n' + body + 'run ${dice_run_steps}\n',
            {'control': {'mode': 'managed', **options}}, ensemble='nvt',
            temperature=600, pressure=None, steps=40, seed=17)


def test_native_conditions_follow_variable_aliases_but_reject_hardcoded_pressure():
    from workflow_backends.lammps.control import render_lammps_control

    source = ('units metal\natom_style atomic\ntimestep 0.001\n'
        'variable T equal ${dice_temperature}\nvariable P equal ${dice_pressure}\n'
        'fix bath all npt temp ${T} ${T} 0.1 iso ${P} ${P} 1.0\nrun ${dice_run_steps}\n')
    result = render_lammps_control(source, {'control': {'mode': 'native'}},
        ensemble='npt', temperature=600, pressure=5, steps=40, seed=17)
    assert 'fix bath all npt temp ${T} ${T}' in result.text
    with pytest.raises(ValueError, match='pressure'):
        render_lammps_control(source.replace('iso ${P} ${P}', 'iso 1 1'),
            {'control': {'mode': 'native'}}, ensemble='npt', temperature=600,
            pressure=5, steps=40, seed=17)


def test_managed_continuations_preserve_comments_and_validate_explicit_setup():
    from workflow_backends.lammps.control import render_lammps_control

    source = ('units metal\natom_style atomic\ninclude dice_task_setup.lammps\n'
        'timestep 0.001\nfix bath all &\n# control comment\n\n nvt temp 1200 1200 0.1\n'
        '# retain analysis\ncompute disp all msd\nrun ${dice_run_steps}\n')
    result = render_lammps_control(source, {'control': {'mode': 'managed'}},
        ensemble='nvt', temperature=600, pressure=None, steps=40, seed=17)
    assert '# retain analysis\ncompute disp all msd' in result.text
    assert result.text.count('nvt temp') == 1
    assert '1200' not in result.text
    with pytest.raises(ValueError, match='pair_style.*DICE'):
        render_lammps_control(source.replace('timestep 0.001', 'pair_style lj/cut 2\ntimestep 0.001'),
            {'control': {'mode': 'managed'}}, ensemble='nvt', temperature=600,
            pressure=None, steps=40, seed=17)


@pytest.mark.parametrize('mode', ['managed', 'native'])
def test_explicit_standard_cadence_leaves_auxiliary_dynamic_dump_alone(mode):
    from workflow_backends.lammps.control import render_lammps_control

    source = ('units metal\natom_style atomic\ntimestep 0.001\n'
        'variable extra_stride equal 50\ndump extra all custom ${extra_stride} extra.dump id type x y z\n'
        'fix bath all nvt temp ${dice_temperature} ${dice_temperature} 0.1\nrun ${dice_run_steps}\n')
    result = render_lammps_control(source, {'control': {'mode': mode}, 'trajectory_every_steps': 20},
        ensemble='nvt', temperature=600, pressure=None, steps=40, seed=17)
    assert 'dump extra all custom ${extra_stride} extra.dump' in result.text
    assert result.cadence == 20


@pytest.mark.parametrize('mode', ['managed', 'native'])
def test_timestep_must_apply_before_formal_run(mode):
    from workflow_backends.lammps.control import render_lammps_control

    source = ('units metal\natom_style atomic\n'
        'fix bath all nvt temp ${dice_temperature} ${dice_temperature} 0.1\n'
        'run ${dice_run_steps}\ntimestep 0.002\n')
    with pytest.raises(ValueError, match='timestep.*before.*run'):
        render_lammps_control(source, {'control': {'mode': mode}}, ensemble='nvt',
            temperature=600, pressure=None, steps=40, seed=17)
