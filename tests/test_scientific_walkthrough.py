"""Check documented native shapes without activating or running native engines."""
import json
import re
from pathlib import Path

from workflow_core.lifecycle.config import load_run_config
from workflow_runtime.lifecycle.environment import load_environment_profile
from workflow_backends.lammps.sampler import LammpsSampler

ROOT = Path(__file__).resolve().parents[1]


def test_complete_documented_run_and_environment(tmp_path):
    doc = (ROOT / "docs/own-system.md").read_text()
    objects = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", doc, re.S)]
    assert len(objects) == 2
    scientific, environment = objects
    run = tmp_path / "run.json"
    run.write_text(json.dumps(scientific))
    config = load_run_config(run)
    assert config.round_count == 2
    assert config.teacher["cli"] == config.student["cli"] == "pt"
    assert config.selection_plan.group_ids == ("bulk",)
    assert config.teacher["type_map"] == config.student["type_map"] == ["Al", "O"]
    profile = tmp_path / "environment.json"
    profile.write_text(json.dumps(environment))
    loaded = load_environment_profile(profile)
    expected = {"teacher.deepmd.pt.train", "teacher.deepmd.pt.label",
                "teacher.deepmd.pt.gmm_reference", "teacher.deepmd.pt.gmm_score",
                "student.deepmd.pt.train", "student.deepmd.pt.compress",
                "student.deepmd.pt.gmm_screen", "sampler.lammps.pt"}
    assert expected == set(loaded.bindings)
    scientific.update(iter=1, student_train_bootstrap=[], fp_data=["data/returned.extxyz"])
    run.write_text(json.dumps(scientific))
    assert load_run_config(run).iter_index == 1


def test_managed_nvt_documentation_contract(tmp_path):
    doc = (ROOT / "docs/own-system.md").read_text()
    recipe = re.search(r"```lammps\n(.*?)\n```", doc, re.S).group(1)
    path = tmp_path / "nvt.lammps"
    path.write_text(recipe+"\n")
    # The validator reads text only; no executor, subprocess or calculator.
    sampler = object.__new__(LammpsSampler)
    sampler._validate_recipe(path, kind="nvt", ensemble="nvt")
    assert "include dice_task_setup.lammps" in recipe
    assert "run ${dice_run_steps}" in recipe


def test_dependencies_explain_current_and_historical_interfaces():
    text = (ROOT / "docs/native-models.md").read_text()
    for value in ["3.2.0b1.dev20260728", "3.2.0+g3b2555c46", "2.11.0+cu130",
                  "13.0", "4.6.2", "dp --pt-expt", "dp --pt",
                  "DPA4-Neo-OMat24-v20260627.pt"]:
        assert value in text
    assert 'cli: "pt"' in text
    assert "specialized historical exporter" in text
    assert "deepmd-compatibility.md" in text


def test_portable_pt_profile_matches_the_walkthrough():
    doc = (ROOT / 'docs/own-system.md').read_text()
    _, documented = [json.loads(block) for block in re.findall(r'```json\n(.*?)\n```', doc, re.S)]
    template_path = ROOT / 'examples/profiles/dpa4c_dpa4.json'
    assert json.loads(template_path.read_text()) == documented
    loaded = load_environment_profile(template_path)
    assert loaded.bindings['student.deepmd.pt.gmm_screen'] == 'models'
    assert loaded.bindings['sampler.lammps.pt'] == 'sampling'


def test_sampling_guide_fragments_load_in_the_complete_workflow(tmp_path):
    import copy
    guide = (ROOT / 'docs/own-system.md').read_text()
    base = json.loads(re.findall(r'```json\n(.*?)\n```', guide, re.S)[0])
    fragments = [json.loads(block) for block in re.findall(r'```json\n(.*?)\n```', (ROOT/'docs/sampling.md').read_text(), re.S)]
    assert fragments
    for fragment in fragments:
        payload = copy.deepcopy(base)
        payload['sample'] = {'round_count': 1, 'seeds': {'bulk': {'source': 'seeds/md.extxyz'}},
                             'condition_groups': {'example': fragment}}
        run = tmp_path / 'run.json'
        run.write_text(json.dumps(payload))
        config = load_run_config(run)
        assert config.round_count == 1


def test_other_backend_and_fp_fragments_use_accepted_public_shapes(tmp_path):
    import copy
    guide = (ROOT/'docs/own-system.md').read_text()
    base = json.loads(re.findall(r'```json\n(.*?)\n```', guide, re.S)[0])
    for name, role in [('mace.md','teacher'), ('nep.md','student'), ('first-principles.md','fp')]:
        payload = copy.deepcopy(base)
        fragment = json.loads(re.findall(r'```json\n(.*?)\n```', (ROOT/'docs'/name).read_text(), re.S)[0])
        payload[role] = fragment
        if name == 'nep.md':
            payload['sampler'] = {'backend': 'GPUMD'}
        run = tmp_path/'run.json'
        run.write_text(json.dumps(payload))
        assert load_run_config(run).round_count == 2
