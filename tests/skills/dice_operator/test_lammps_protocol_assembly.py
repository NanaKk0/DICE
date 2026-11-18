from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only selected-protocol preflight for the automatic LAMMPS interface."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

REPO = CODE_ROOT
sys.path.insert(0, str(REPO))

from workflow_app.composition import preflight_sampling_protocols
from workflow_core.lifecycle.config import load_run_config


class AutomaticLammpsProtocol(unittest.TestCase):
    def test_selected_protocol_preflight_accepts_automatic_and_rejects_conflict(self):
        reference = FIXTURES/'md-mc.json'
        payload = json.loads(reference.read_text())
        recipe_source = FIXTURES/'md-mc.lammps'
        protocol = recipe_source.read_text().replace('include dice_task_setup.lammps\n', '')
        with tempfile.TemporaryDirectory(prefix='dice-operator-protocol-') as directory:
            root = Path(directory)
            group = payload['sample']['condition_groups']['ysz_swap']
            recipe = root / group['recipe']
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(protocol)
            run = root / 'run.json'
            run.write_text(json.dumps(payload))
            launch = SimpleNamespace(
                config=load_run_config(run),
                plan=SimpleNamespace(blocks=(SimpleNamespace(block_id='r0/sample'),)),
                preflight=None,
            )
            preflight_sampling_protocols(launch, first_block='r0/sample')
            self.assertEqual(recipe.read_text(), protocol)
            self.assertEqual({p.name for p in root.iterdir()}, {'run.json', 'recipes'})
            recipe.write_text(protocol + '\npair_style lj/cut 2.5\n')
            with self.assertRaisesRegex(ValueError, 'automatic LAMMPS protocol'):
                preflight_sampling_protocols(launch, first_block='r0/sample')
            self.assertFalse((root / 'iter0').exists())

    def test_managed_controls_resolve_sources_without_creating_results(self):
        reference = FIXTURES/'md-mc.json'
        payload = json.loads(reference.read_text())
        group = payload['sample']['condition_groups']['ysz_swap']
        group['kind'] = 'nvt'
        for name in ('mc', 'ensemble', 'pressures_bar'):
            group.pop(name, None)
        group['lammps'] = {'control': {'mode': 'managed'}, 'timestep_ps': 0.001}
        with tempfile.TemporaryDirectory(prefix='dice-operator-controls-') as directory:
            root = Path(directory)
            recipe = root/group['recipe']
            recipe.parent.mkdir(parents=True)
            protocol = ('units metal\natom_style atomic\n'
                        'fix bath all npt temp 900 900 0.1 iso 1 1 1\nrun 999\n')
            recipe.write_text(protocol)
            run = root/'run.json'
            run.write_text(json.dumps(payload))
            launch = SimpleNamespace(config=load_run_config(run),
                plan=SimpleNamespace(blocks=(SimpleNamespace(block_id='r0/sample'),)), preflight=None)
            summaries = preflight_sampling_protocols(launch, first_block='r0/sample')
            self.assertEqual(summaries[0]['settings']['ensemble']['value'], 'nvt')
            self.assertEqual(summaries[0]['settings']['temperature_damping_ps']['value'], 0.1)
            self.assertEqual(recipe.read_text(), protocol)
            self.assertFalse((root/'iter0').exists())
            group['lammps']['control']['unsupported_field'] = 1
            run.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_run_config(run)

    def test_clean_native_requests_preserve_inheritance_and_inspect_explicit_fields(self):
        import runpy
        inspector = runpy.run_path(str(SKILL_ROOT/'scripts/inspect_dice_operation.py'))
        reference = FIXTURES/'md-mc.json'
        payload = json.loads(reference.read_text())
        payload['sample']['condition_groups'] = {'native':{'recipe':'recipes/native.lammps'}}
        with tempfile.TemporaryDirectory(prefix='dice-operator-clean-native-') as directory:
            root = Path(directory)
            (root/'recipes').mkdir()
            (root/'recipes/native.lammps').write_text('units metal\natom_style atomic\ntimestep .001\n'
                'velocity all create 350 917\nfix bath all nvt temp 900 900 .1 drag .2\nrun 80\n')
            run = root/'run.json'
            for explicit in (False,True):
                if explicit:
                    payload['sample']['condition_groups']['native'].update(temperatures_k=[600],steps=20)
                run.write_text(json.dumps(payload))
                launch = SimpleNamespace(config=load_run_config(run),
                    plan=SimpleNamespace(blocks=(SimpleNamespace(block_id='r0/sample'),)),preflight=None)
                task = inspector['_sampling_summary'](launch)['effective_rounds'][0]['tasks'][0]
                self.assertEqual(task['temperature_k'],600 if explicit else None)
                self.assertEqual(task['protocol_request']['steps'],20 if explicit else None)
                summary, = preflight_sampling_protocols(launch,first_block='r0/sample')
                self.assertEqual(summary['settings']['temperature_k']['value'],600 if explicit else 900)
                self.assertEqual(summary['settings']['temperature_k']['source'],'DICE' if explicit else 'native')
                self.assertFalse((root/'iter0').exists())


if __name__ == '__main__':
    unittest.main()
