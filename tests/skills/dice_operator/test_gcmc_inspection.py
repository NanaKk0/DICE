from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only CLI scenarios; these do not grade free-text agent behavior."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class GcmcInspection(unittest.TestCase):
    def test_success_and_rejection(self):
        package = SKILL_ROOT
        reference = FIXTURES/'gcmc.json'
        for mode in ('nvt', 'npt', 'pure_mc', 'unknown', 'nep'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='dice-operator-gcmc-') as name:
                root = Path(name)
                payload = json.loads(reference.read_text())
                group = payload['sample']['condition_groups']['ads']
                if mode == 'npt':
                    group['md'].update(ensemble='npt', pressure_bar=1.5)
                elif mode == 'pure_mc':
                    del group['md']
                elif mode == 'unknown':
                    group['mc']['flames_kwargs'] = {}
                elif mode == 'nep':
                    payload['student']['backend'] = 'NEP'
                    del payload['student']['cli']
                    payload['student']['recipe'] = 'recipes/nep.in'
                    payload['sampler']['backend'] = 'GPUMD'
                run = root/'run.json'
                run.write_text(json.dumps(payload))
                before = run.read_bytes()
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                env.pop('DICE_ENVIRONMENT', None)
                env.pop('DICE_EXECUTION', None)
                result = subprocess.run([sys.executable, str(package/'scripts/inspect_dice_operation.py'), str(run)],
                    env=env, text=True, capture_output=True, timeout=20)
                output = json.loads(result.stdout)
                self.assertEqual(list(root.iterdir()), [run])
                self.assertEqual(run.read_bytes(), before)
                if mode in ('unknown', 'nep'):
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(output['ok'])
                    self.assertIn('flames_kwargs' if mode == 'unknown' else 'GPUMD', output['error'])
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                tasks = output['sampling']['effective_rounds'][0]['tasks']
                self.assertEqual([t['pressure_bar'] for t in tasks], [.1, 1.])
                self.assertEqual(output['block_operation_roles']['r0/sample'], ['sampler.lammps.pt'])
                for task in tasks:
                    self.assertEqual(task['cycles'], 2)
                    self.assertEqual(task['gcmc']['every'], 100)
                    self.assertEqual(task['gcmc']['moves'], 100)
                    if mode == 'pure_mc':
                        self.assertIsNone(task['gcmc']['md'])
                    elif mode == 'npt':
                        self.assertEqual(task['gcmc']['md']['pressure_bar'], 1.5)


if __name__ == '__main__':
    unittest.main()
