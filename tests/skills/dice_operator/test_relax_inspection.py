from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only loader scenarios; scientific failure evidence is tested in DICE."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RelaxInspection(unittest.TestCase):
    def test_mps_placement_and_exact_resume(self):
        package = SKILL_ROOT
        repo = CODE_ROOT
        for family in ('tf', 'pt'):
            for mode in ('mps2', 'mps4', 'per_gpu1', 'resume', 'final_only',
                         'ssh', 'slurm', 'multi_gpu', 'nep', 'cpu_budget'):
                with self.subTest(family=family, mode=mode), tempfile.TemporaryDirectory(prefix='dice-relax-inspect-') as name:
                    root = Path(name)
                    run = json.loads((FIXTURES/'widom.json').read_text())
                    run['student']['cli'] = family
                    group = dict(kind='relax', fmax_ev_a=.02, max_steps=8)
                    if mode == 'final_only':
                        group['include_history'] = False
                    run['sample']['condition_groups'] = {'relax': group}
                    execution = json.loads((FIXTURES/'relax.json').read_text())
                    role = f'student.deepmd.{family}.relax'
                    execution['bindings'] = [b for b in execution['bindings'] if b['operation_role'] == role]
                    resource = execution['resource_classes']['relax-mps']
                    if mode == 'mps4': resource['per_gpu'] = 4
                    if mode == 'per_gpu1': resource['per_gpu'] = 1
                    if mode == 'multi_gpu': resource['gpus_per_node'] = 2
                    if mode == 'cpu_budget': resource['cpus_per_process'] = 1
                    if mode == 'ssh':
                        execution['targets']['local'] = dict(adapter='ssh', max_in_flight=1, host_alias='test', remote_root='dice', stage_mode='copy')
                    if mode == 'slurm':
                        execution['targets']['local'] = dict(adapter='slurm', max_in_flight=1, working_root='/shared/dice', stage_mode='shared')
                    if mode == 'nep':
                        run['student'].update(backend='NEP', recipe='recipes/nep.in')
                        del run['student']['cli']
                        run['sampler']['backend'] = 'GPUMD'
                        execution['bindings'][0]['operation_role'] = 'student.nep.relax'
                    run_path, execution_path = root/'run.json', root/'execution.json'
                    run_path.write_text(json.dumps(run)); execution_path.write_text(json.dumps(execution))
                    before = {p.name: p.read_bytes() for p in root.iterdir()}
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                    env.pop('DICE_ENVIRONMENT', None); env.pop('DICE_EXECUTION', None)
                    args = [sys.executable, str(package/'scripts/inspect_dice_operation.py'), str(run_path), '--execution', str(execution_path)]
                    if mode == 'resume': args += ['--resume', 'r0/sample']
                    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
                    output = json.loads(result.stdout)
                    self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)
                    if mode in ('ssh', 'slurm', 'multi_gpu', 'nep', 'cpu_budget'):
                        self.assertEqual(result.returncode, 2, output)
                        self.assertFalse(output['ok'])
                        continue
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    self.assertEqual(output['block_operation_roles']['r0/sample'], [role])
                    task = output['sampling']['effective_rounds'][0]['tasks'][0]
                    self.assertEqual(task['kind'], 'relax')
                    self.assertFalse(task['relax']['accept_unconverged'])
                    self.assertEqual(task['relax']['include_history'], mode != 'final_only')
                    self.assertEqual(output['execution']['resource_classes']['relax-mps']['per_gpu'], resource['per_gpu'])
                    if mode == 'resume':
                        self.assertEqual(output['selected_block'], 'r0/sample')
                        self.assertEqual(output['selected_suffix'][0], 'r0/sample')


if __name__ == '__main__': unittest.main()
