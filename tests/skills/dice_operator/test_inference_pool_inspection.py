from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only acceptance/rejection scenarios for native inference frame pools."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class InferencePoolInspection(unittest.TestCase):
    def test_public_inspector_accepts_pool_and_rejects_mps_distributed(self):
        package = SKILL_ROOT
        repo = CODE_ROOT
        baseline = json.loads((FIXTURES/'inference.json').read_text())
        run = json.loads((FIXTURES/'widom.json').read_text())
        for binding in baseline['bindings']:
            role = binding['operation_role']
            for mode in ('pool', 'mps', 'distributed'):
                with self.subTest(role=role, mode=mode), tempfile.TemporaryDirectory(prefix='dice-inference-inspect-') as name:
                    root = Path(name)
                    profile = json.loads(json.dumps(baseline))
                    profile['bindings'] = [binding]
                    if mode == 'mps':
                        profile['resource_classes']['inference']['per_gpu'] = 2
                    elif mode == 'distributed':
                        resource = profile['resource_classes']['inference']
                        resource['mode'] = 'distributed'
                        resource['launcher'] = {'kind': 'torchrun'}
                        resource.pop('per_gpu')
                    (root/'run.json').write_text(json.dumps(run))
                    (root/'execution.json').write_text(json.dumps(profile))
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(repo))
                    env.pop('DICE_ENVIRONMENT', None)
                    env.pop('DICE_EXECUTION', None)
                    result = subprocess.run([sys.executable, str(package/'scripts/inspect_dice_operation.py'),
                        str(root/'run.json'), '--execution', str(root/'execution.json')],
                        env=env, capture_output=True, text=True, timeout=30)
                    output = json.loads(result.stdout)
                    self.assertEqual({path.name for path in root.iterdir()}, {'run.json', 'execution.json'})
                    if mode == 'pool':
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        actual = output['execution']['resource_classes']['inference']
                        self.assertEqual(actual['gpus_per_node'], 4)
                        self.assertEqual(actual['per_gpu'], 1)
                    else:
                        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                        self.assertFalse(output['ok'])


if __name__ == '__main__':
    unittest.main()
