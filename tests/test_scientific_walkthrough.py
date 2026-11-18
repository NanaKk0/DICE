"""Validate public examples and native snippets without running native engines."""
import copy
import json
import re
from pathlib import Path

from workflow_core.lifecycle.config import load_run_config
from workflow_runtime.lifecycle.environment import load_environment_profile

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/dpa4_dpa4c'


def example_run():
    return json.loads((EXAMPLE / 'run.json').read_text())


def test_complete_documented_run_and_environment(tmp_path):
    scientific = example_run()
    run = tmp_path / 'run.json'
    run.write_text(json.dumps(scientific))
    config = load_run_config(run)
    assert config.round_count == 2
    assert config.teacher['cli'] == config.student['cli'] == 'pt'
    assert config.selection_plan.group_ids == ('bulk',)
    assert config.teacher['type_map'] == config.student['type_map'] == ['O', 'H']
    loaded = load_environment_profile(EXAMPLE / 'environment.json')
    expected = {'teacher.deepmd.pt.train', 'teacher.deepmd.pt.label',
                'teacher.deepmd.pt.gmm_reference', 'teacher.deepmd.pt.gmm_score',
                'student.deepmd.pt.train', 'student.deepmd.pt.compress',
                'student.deepmd.pt.gmm_screen', 'sampler.lammps.pt'}
    assert expected == set(loaded.bindings)
    scientific.update(iter=1, student_train_bootstrap=[], fp_data=['data/returned.extxyz'])
    run.write_text(json.dumps(scientific))
    assert load_run_config(run).iter_index == 1


def test_native_nvt_documentation_contract(tmp_path):
    from workflow_backends.lammps.native_protocol import compile_lammps_protocol
    from workflow_backends.lammps.protocol_sampler import bind_protocol_setup
    doc = (ROOT / 'docs/guides/sampling.md').read_text()
    recipe = re.search(r'```lammps\n(.*?)\n```', doc, re.S).group(1)
    scientific = example_run()
    scientific['sample'] = json.loads(re.findall(r'```json\n(.*?)\n```', doc, re.S)[0])
    run = tmp_path / 'run.json'
    run.write_text(json.dumps(scientific))
    task, = load_run_config(run).sampling_plan.tasks_for_round(0)
    result = compile_lammps_protocol(recipe+'\n', task.protocol_request)
    assert result.temperature_k == 300 and result.total_steps == 200
    assert result.cadence == 10 and result.timestep_ps == .00025
    assert '${dice_' not in recipe and 'include dice_task_setup.lammps' not in recipe
    rendered = bind_protocol_setup(result.text)
    assert rendered.index('boundary p p p') < rendered.index('include dice_task_setup.lammps')
    assert rendered.index('include dice_task_setup.lammps') < rendered.index('velocity all create')
    assert 'run 200' in rendered


def test_model_guide_describes_portable_native_interfaces():
    text = (ROOT / 'docs/guides/deepmd.md').read_text()
    for value in ['foundation_finetune', 'from_scratch', 'type_map',
                  'dp --pt-expt', 'dp --pt', 'deepmd/kk']:
        assert value in text
    assert 'cli: "pt"' in text
    assert 'https://docs.deepmodeling.com/' in text
    assert 'https://huggingface.co/' in text
    pages = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md')),
             EXAMPLE / 'README.md', EXAMPLE / 'data/NOTICE.md']
    for page in pages:
        body = page.read_text()
        assert not re.search(r'/(?:home|Users)/[^/\s]+/', body), page
        assert not re.search(r'\b(?:RTX|GeForce|Quadro)\s*\d{3,4}\b', body), page
        assert 'in the validated run' not in body.lower(), page


def test_portable_pt_profile_matches_the_walkthrough():
    template_path = ROOT / 'examples/profiles/dpa4c_dpa4.json'
    template = load_environment_profile(template_path)
    example = load_environment_profile(EXAMPLE / 'environment.json')
    assert template.bindings == example.bindings
    assert template.bindings['student.deepmd.pt.gmm_screen'] == 'models'
    assert template.bindings['sampler.lammps.pt'] == 'sampling'


def test_sampling_guide_fragments_load_in_the_complete_workflow(tmp_path):
    base = example_run()
    document = (ROOT / 'docs/reference/sampling.md').read_text()
    fragments = [json.loads(block) for block in re.findall(r'```json\n(.*?)\n```', document, re.S)]
    assert fragments
    for fragment in fragments:
        payload = copy.deepcopy(base)
        if fragment.get('kind') == 'md_mc':
            # This reference fragment describes Y/Zr swaps, not the water example.
            elements = fragment['mc']['elements']
            for role in ('teacher', 'student'):
                payload[role]['type_map'] = list(dict.fromkeys([*payload[role]['type_map'], *elements]))
        payload['sample'] = {'round_count': 1, 'seeds': {'bulk': {'source': 'seeds/seed.extxyz'}},
                             'condition_groups': {'example': fragment}}
        run = tmp_path / 'run.json'
        run.write_text(json.dumps(payload))
        assert load_run_config(run).round_count == 1


def test_other_backend_and_fp_fragments_use_accepted_public_shapes(tmp_path):
    base = example_run()
    for name, role in [('mace.md', 'teacher'), ('nep.md', 'student'), ('cp2k-abacus.md', 'fp')]:
        payload = copy.deepcopy(base)
        document = (ROOT / 'docs/guides/experimental' / name).read_text()
        fragment = json.loads(re.findall(r'```json\n(.*?)\n```', document, re.S)[0])
        if role in ('teacher', 'student'):
            fragment['type_map'] = base[role]['type_map']
        payload[role] = fragment
        if name == 'nep.md':
            payload['sampler'] = {'backend': 'GPUMD'}
            payload['sample']['condition_groups']['nvt'].pop('lammps', None)
        run = tmp_path / 'run.json'
        run.write_text(json.dumps(payload))
        assert load_run_config(run).round_count == 2
