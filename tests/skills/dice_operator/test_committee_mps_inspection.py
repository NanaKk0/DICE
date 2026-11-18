from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only public-loader scenarios for committee training placement."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CommitteeMpsInspection(unittest.TestCase):
    def test_placement_rejections_and_train_resume(self):
        package = SKILL_ROOT
        repo = CODE_ROOT
        for owner in ('teacher', 'student'):
            for family in ('tf', 'pt'):
                for mode in ('single', 'dual', 'ordinary', 'resume', 'ssh', 'slurm', 'multi_target', 'unsupported'):
                    with self.subTest(owner=owner, family=family, mode=mode), tempfile.TemporaryDirectory(prefix='dice-committee-inspect-') as name:
                        root = Path(name)
                        run = json.loads((FIXTURES/'widom.json').read_text())
                        role = f'{owner}.deepmd.{family}.train'
                        run[owner].update(backend='DeepMD', cli=family, model='committee', members=5, recipe='recipes/train.json')
                        run['teacher' if owner == 'teacher' else 'screen'].update(lo=0.05, hi=0.25)
                        if owner == 'teacher': run['screen'].pop('fp_pool')
                        execution = json.loads((FIXTURES/'committee.json').read_text())
                        execution['bindings'] = [b for b in execution['bindings'] if b['operation_role'] == role]
                        resource = execution['resource_classes']['committee']
                        if mode == 'dual': resource['gpus_per_node'] = 2
                        if mode == 'ordinary': resource['per_gpu'] = 1
                        if mode == 'ssh':
                            execution['targets']['local'] = dict(adapter='ssh', max_in_flight=1, host_alias='test', remote_root='dice', stage_mode='copy')
                        if mode == 'slurm':
                            execution['targets']['local'] = dict(adapter='slurm', max_in_flight=1, working_root='/shared/dice', stage_mode='shared')
                        if mode == 'multi_target':
                            execution['targets']['second'] = dict(adapter='local', max_in_flight=1)
                            execution['target_pools']['local']['targets'].append('second')
                        if mode == 'unsupported':
                            execution['bindings'][0]['operation_role'] = 'teacher.mace.train' if owner == 'teacher' else 'student.nep.train'
                        run_path, execution_path = root/'run.json', root/'execution.json'
                        run_path.write_text(json.dumps(run)); execution_path.write_text(json.dumps(execution))
                        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                        env.pop('DICE_ENVIRONMENT', None); env.pop('DICE_EXECUTION', None)
                        args = [sys.executable, str(package/'scripts/inspect_dice_operation.py'), str(run_path), '--execution', str(execution_path)]
                        block = 'teach/train' if owner == 'teacher' else 'r0/train'
                        if mode == 'resume': args += ['--resume', block]
                        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
                        output = json.loads(result.stdout)
                        self.assertEqual({p.name for p in root.iterdir()}, {'run.json', 'execution.json'})
                        if mode in ('ssh', 'slurm', 'multi_target', 'unsupported'):
                            self.assertEqual(result.returncode, 2, output)
                            self.assertFalse(output['ok'])
                            if mode == 'unsupported':
                                self.assertIn('pool', output['error'].lower())
                            else:
                                self.assertIn('one explicit local target', output['error'])
                            continue
                        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                        self.assertIn(role, output['block_operation_roles'][block])
                        actual = output['execution']['resource_classes']['committee']
                        self.assertEqual(actual['gpus_per_node'], resource['gpus_per_node'])
                        self.assertEqual(actual['per_gpu'], resource['per_gpu'])
                        self.assertEqual(output['normalized_run_config'][owner]['members'], 5)
                        if mode == 'resume':
                            self.assertEqual(output['selected_block'], block)
                            self.assertEqual(output['selected_suffix'][0], block)


if __name__ == '__main__': unittest.main()
