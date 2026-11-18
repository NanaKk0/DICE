"""Native protocol assembly, including physical command ordering."""
import pytest

from workflow_backends.lammps.protocol import assemble_lammps_protocol


def test_assembly_keeps_multiline_preamble_and_protocol_commands_intact():
    protocol = (
        '# Protocol initialization\nunits metal\natom_style atomic\n'
        'processors * * & # keep this command together\n  *\n'
        'variable dice_trajectory_every_steps equal 20\n'
        'variable atom_count index $(count(all))\n'
        'fix thermostat all &\n nvt temp ${dice_temperature} ${dice_temperature} 0.1\n'
        'print "# physical protocol"\nrun ${dice_run_steps}'
    )
    rendered = assemble_lammps_protocol(protocol)
    setup = rendered.index('include dice_task_setup.lammps')
    assert rendered.index('processors * * &') < rendered.index('  *\n') < setup
    assert setup < rendered.index('variable atom_count index $(count(all))')
    assert rendered.replace('include dice_task_setup.lammps\n', '') == protocol


def test_explicit_setup_protocol_keeps_its_original_bytes():
    protocol = 'include dice_task_setup.lammps\nprint """first\nsecond"""\nrun ${dice_run_steps}'
    assert assemble_lammps_protocol(protocol) == protocol


def test_quoted_ampersand_is_preserved_as_print_text():
    protocol = 'units metal\natom_style atomic\nrun ${dice_run_steps}\nprint "&"\n'
    rendered = assemble_lammps_protocol(protocol)
    assert rendered.endswith('print "&"\n')
    assert rendered.replace('include dice_task_setup.lammps\n', '') == protocol


@pytest.mark.parametrize('protocol', [
    'units metal\natom_style atomic\nfix thermostat all &\n',
    'units metal\natom_style atomic\nprint "unterminated\n',
])
def test_ambiguous_incomplete_protocol_is_rejected(protocol):
    with pytest.raises(ValueError, match='automatic LAMMPS protocol'):
        assemble_lammps_protocol(protocol)


def test_backend_plugin_load_precedes_generated_model_setup():
    protocol = (
        'plugin load /opt/deepmd/lib/libdeepmd_lmp.so\n'
        'units metal\natom_style atomic\nboundary p p p\n'
        'timestep 0.001\nrun ${dice_run_steps}\n'
    )
    rendered = assemble_lammps_protocol(protocol)
    assert rendered.index('plugin load') < rendered.index('units metal')
    assert rendered.index('boundary p p p') < rendered.index('include dice_task_setup.lammps')
    assert rendered.index('include dice_task_setup.lammps') < rendered.index('timestep 0.001')
