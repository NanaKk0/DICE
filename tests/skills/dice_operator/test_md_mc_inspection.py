from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only normalized MDMC controls, activations and rejection scenarios."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MdMcInspection(unittest.TestCase):
    def test_closed_budgets_and_suffix(self):
        package = SKILL_ROOT
        reference = FIXTURES/'md-mc.json'
        for mode in ('normal', 'npt', 'npt_missing_pressure', 'nondivisible', 'interval_one', 'rounds', 'resume', 'invalid_move', 'bool_count', 'too_short', 'nep'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='dice-operator-mdmc-') as name:
                root = Path(name)
                payload = json.loads(reference.read_text())
                group = payload['sample']['condition_groups']['ysz_swap']
                if mode == 'nondivisible':
                    group['steps'] = 103
                    group['mc']['attempts_per_event'] = 2
                elif mode == 'interval_one':
                    group['steps'] = 7
                    group['mc']['every_md_steps'] = 1
                elif mode in ('npt', 'npt_missing_pressure'):
                    group['ensemble'] = 'npt'
                    if mode == 'npt':
                        group['pressures_bar'] = [1., 100.]
                elif mode == 'rounds':
                    payload['sample'].pop('round_count')
                    payload['sample']['condition_groups']['md'] = dict(kind='nvt', temperatures_k=[1773], steps=20, recipe='recipes/md.in')
                    payload['sample']['rounds'] = [{'condition_groups': ['md']}, {'condition_groups': ['ysz_swap']}]
                elif mode == 'invalid_move':
                    group['mc']['move'] = 'transmute'
                elif mode == 'bool_count':
                    group['mc']['every_md_steps'] = True
                elif mode == 'too_short':
                    group['steps'] = 9
                elif mode == 'nep':
                    payload['student'].update(backend='NEP', recipe='recipes/nep.in')
                    payload['student'].pop('cli')
                    payload['sampler']['backend'] = 'GPUMD'
                run = root/'run.json'
                run.write_text(json.dumps(payload))
                original = run.read_bytes()
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                env.pop('DICE_ENVIRONMENT', None)
                env.pop('DICE_EXECUTION', None)
                command = [sys.executable, str(package/'scripts/inspect_dice_operation.py'), str(run)]
                if mode == 'resume':
                    command += ['--resume', 'r0/sample']
                result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20)
                output = json.loads(result.stdout)
                self.assertEqual(list(root.iterdir()), [run])
                self.assertEqual(run.read_bytes(), original)
                if mode in ('invalid_move', 'bool_count', 'too_short', 'nep', 'npt_missing_pressure'):
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(output['ok'])
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                rounds = output['sampling']['effective_rounds']
                if mode == 'rounds':
                    self.assertEqual(rounds[0]['tasks'][0]['kind'], 'nvt')
                    self.assertIsNone(rounds[0]['tasks'][0]['md_mc'])
                task = rounds[-1]['tasks'][0]
                self.assertEqual(task['kind'], 'md_mc')
                self.assertEqual(task['ensemble'], 'npt' if mode == 'npt' else 'nvt')
                if mode == 'npt':
                    self.assertEqual([t['pressure_bar'] for t in rounds[-1]['tasks']], [1., 100.])
                self.assertEqual(task['temperature_k'], 1773)
                self.assertEqual(task['md_mc']['elements'], ['Y', 'Zr'])
                self.assertEqual(task['md_mc']['move'], 'identity_swap')
                expected = 20 if mode == 'nondivisible' else 7 if mode == 'interval_one' else 10000
                self.assertEqual(task['planned_mc_attempts'], expected)
                self.assertEqual(task['selection_group_id'], 'YSZ_all')
                self.assertEqual(output['block_operation_roles'][f'r{len(rounds)-1}/sample'], ['sampler.lammps.pt'])
                self.assertFalse({'gpu', 'gpus', 'per_gpu', 'mps'} & set(task['md_mc']))
                if mode == 'resume':
                    self.assertEqual(output['selected_block'], 'r0/sample')


if __name__ == '__main__':
    unittest.main()
