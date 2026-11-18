from support import CODE_ROOT, FIXTURES, SKILL_ROOT
"""Read-only public-inspector scenarios for native DeePMD presets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DeepMDPresetInspection(unittest.TestCase):
    def test_canonical_zbl_pt_roles_and_tf_rejection(self):
        package = SKILL_ROOT
        repo = CODE_ROOT
        sys.path.insert(0, str(repo))
        from workflow_core.lifecycle.config import load_run_config, validate_role_recipe_feasibility
        template = json.loads((FIXTURES/'widom.json').read_text())
        for cli in ('pt', 'tf'):
            with self.subTest(cli=cli), tempfile.TemporaryDirectory(prefix='dice-zbl-inspect-') as name:
                root = Path(name)
                run = json.loads(json.dumps(template))
                for role, family in [('teacher', 'dpa4'), ('student', 'dpa4c')]:
                    run[role] = dict(backend='DeepMD', cli='pt' if role=='teacher' else cli,
                        model='single', type_map=['O','H'], recipe=role+'.json',
                        policy='from_scratch', valid_frac=0)
                    model = {'type':'linear_ener', 'weights':'sum', 'type_map':['O','H'],
                        'models':[{'descriptor':{'type':family}, 'fitting_net':{'neuron':[96]}},
                                  {'type':'inner_potential','mode':'zbl'}]}
                    (root/(role+'.json')).write_text(json.dumps({'model':model}))
                run['student']['compress'] = True
                (root/'run.json').write_text(json.dumps(run))
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(repo))
                env.pop('DICE_ENVIRONMENT', None);env.pop('DICE_EXECUTION', None)
                result = subprocess.run([sys.executable, str(package/'scripts/inspect_dice_operation.py'),
                    str(root/'run.json')], env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                output = json.loads(result.stdout)
                self.assertEqual(output['scope'], 'structural-only')
                config = load_run_config(root/'run.json')
                if cli == 'pt':
                    validate_role_recipe_feasibility(config, 'teacher')
                    validate_role_recipe_feasibility(config, 'student')
                    rendered = json.dumps(output)
                    self.assertIn('teacher.deepmd.pt.gmm_reference', rendered)
                    self.assertIn('student.deepmd.pt.compress', rendered)
                else:
                    with self.assertRaisesRegex(ValueError, 'DPA4C student recipes require'):
                        validate_role_recipe_feasibility(config, 'student')
                self.assertEqual({p.name for p in root.iterdir()}, {'run.json','teacher.json','student.json'})

    def test_preset_pt_roles_and_tf_rejection(self):
        package = SKILL_ROOT
        repo = CODE_ROOT
        sys.path.insert(0, str(repo))
        from workflow_core.lifecycle.config import load_run_config, validate_role_recipe_feasibility
        # Use the checked public sampling fixture; all changes remain local to
        # this temporary structural input and never preflight or launch work.
        template = json.loads((FIXTURES/'widom.json').read_text())
        for cli in ('pt', 'tf'):
            with self.subTest(cli=cli), tempfile.TemporaryDirectory(prefix='dice-preset-inspect-') as name:
                root = Path(name)
                run = json.loads(json.dumps(template))
                for role, family in [('teacher', 'dpa4'), ('student', 'dpa4c')]:
                    run[role] = dict(backend='DeepMD', cli='pt' if role=='teacher' else cli,
                        model='single', type_map=['O','H'], recipe=role+'.json',
                        policy='from_scratch', valid_frac=0)
                    (root/(role+'.json')).write_text(json.dumps({'model': {'preset':family+'-nano-v20260911'}}))
                run['student']['compress'] = True
                (root/'run.json').write_text(json.dumps(run))
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(repo))
                env.pop('DICE_ENVIRONMENT', None);env.pop('DICE_EXECUTION', None)
                result = subprocess.run([sys.executable, str(package/'scripts/inspect_dice_operation.py'),
                    str(root/'run.json')], env=env, capture_output=True, text=True, timeout=30)
                output = json.loads(result.stdout)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('external input existence and native recipe feasibility', output['not_checked'])
                config = load_run_config(root/'run.json')
                if cli == 'pt':
                    validate_role_recipe_feasibility(config, 'teacher')
                    validate_role_recipe_feasibility(config, 'student')
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.assertTrue(output['ok'])
                    rendered = json.dumps(output)
                    self.assertIn('teacher.deepmd.pt.gmm_reference', rendered)
                    self.assertIn('student.deepmd.pt.compress', rendered)
                else:
                    with self.assertRaisesRegex(ValueError, 'DPA4C student recipes require'):
                        validate_role_recipe_feasibility(config, 'student')
                self.assertEqual({p.name for p in root.iterdir()}, {'run.json','teacher.json','student.json'})


if __name__ == '__main__':
    unittest.main()
