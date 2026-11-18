from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only Widom inspection scenarios; free-text judgment is not automated."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class WidomInspection(unittest.TestCase):
    def test_closed_controls_and_resume(self):
        package = SKILL_ROOT
        reference = FIXTURES/'widom.json'
        for mode in ('default','fixed_attempts','no_overlap','resume','invalid_budget','nep'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='dice-operator-widom-') as name:
                root=Path(name); payload=json.loads(reference.read_text())
                group=payload['sample']['condition_groups']['insert']
                group.update(frames=100,max_attempts=1000)
                group.pop('random_seed'); group.pop('overlap')
                if mode=='fixed_attempts': group['max_attempts']=100
                elif mode=='no_overlap': group['overlap']=None
                elif mode=='invalid_budget': group['max_attempts']=99
                elif mode=='nep':
                    payload['student'].update(backend='NEP',recipe='recipes/nep.in')
                    del payload['student']['cli']; payload['sampler']['backend']='GPUMD'
                run=root/'run.json'; run.write_text(json.dumps(payload)); original=run.read_bytes()
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
                env.pop('DICE_ENVIRONMENT',None); env.pop('DICE_EXECUTION',None)
                args=[sys.executable,str(package/'scripts/inspect_dice_operation.py'),str(run)]
                if mode=='resume': args+=['--resume','r0/sample']
                result=subprocess.run(args,env=env,text=True,capture_output=True,timeout=20)
                output=json.loads(result.stdout)
                self.assertEqual(list(root.iterdir()),[run]); self.assertEqual(run.read_bytes(),original)
                if mode in ('invalid_budget','nep'):
                    self.assertEqual(result.returncode,2); self.assertFalse(output['ok']); continue
                self.assertEqual(result.returncode,0,result.stderr)
                tasks=output['sampling']['effective_rounds'][0]['tasks']
                self.assertEqual(len(tasks),1)
                task=tasks[0]
                self.assertEqual(task['kind'],'widom'); self.assertIsNone(task['temperature_k'])
                self.assertIsNone(task['steps']); self.assertIsNone(task['recipe_binding'])
                self.assertEqual(task['widom']['frames'],100)
                self.assertEqual(task['widom']['max_attempts'],100 if mode=='fixed_attempts' else 1000)
                self.assertEqual(task['widom']['random_seed'],0)
                self.assertEqual(task['widom']['overlap_scale'],None if mode=='no_overlap' else .6)
                self.assertEqual(output['block_operation_roles']['r0/sample'],['sampler.lammps.pt'])


if __name__=='__main__': unittest.main()
