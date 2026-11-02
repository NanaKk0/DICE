"""Bounded CPU checks for shipped examples; no native engines or inference."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from workflow_core.lifecycle.config import load_run_config
from workflow_core.lifecycle.data import LabelContract, validate_label_properties
from workflow_core.lifecycle.screen import reserve_teacher_gmm_fp, select_student_gmm_train
from workflow_core.structures.formats import read_extxyz_structures

ROOT = Path(__file__).resolve().parents[1]


def test_generated_loader_inventory_is_current_and_includes_conditional_additions():
    subprocess.run([sys.executable, str(ROOT / "tools/generate_config_reference.py"), "--check"], check=True)
    document = (ROOT / "docs/configuration-fields.md").read_text()
    condition_row = next(line for line in document.splitlines() if "_resolve_explicit_sampling_plan:" in line and "`prefix`" in line)
    assert "`recipe`" in condition_row and "`pressures_bar`" in condition_row
    p_row = next(line for line in document.splitlines() if "_p_band_config:" in line)
    assert all(f"`{key}`" in p_row for key in ("floor", "ceiling", "trust"))


def test_minimal_labelled_extxyz_schema_and_unlabelled_rejection():
    frames = read_extxyz_structures(ROOT / "examples/data/labelled_h2.extxyz", allow_empty=False)
    validate_label_properties(frames, LabelContract())
    assert len(frames) == 1 and len(frames[0]) == 2
    assert frames[0].get_potential_energy() == -1.0
    np.testing.assert_allclose(frames[0].get_forces(), np.zeros((2, 3)))
    assert not frames[0].pbc.any()
    unlabelled = frames[0].copy()
    with pytest.raises(ValueError, match="energy"):
        validate_label_properties([unlabelled], LabelContract())
    with pytest.raises(ValueError, match="stress"):
        validate_label_properties(frames, LabelContract(require_stress=True))


def scored_frames():
    result = []
    for index, probability in enumerate((0.05, 0.1, 0.2, 0.5, 1.0)):
        atoms = Atoms("H", positions=[[index, 0, 0]])
        atoms.info.update(structure_id=f"frame{index}", system_id="demo", teacher_p=probability, student_p=probability)
        result.append(atoms)
    return result


def test_documented_p_boundaries_and_independent_teacher_views():
    fp, student, flags = reserve_teacher_gmm_fp(
        scored_frames(), fp_pool=-1, teacher_floor=0.1,
        teacher_trust=0.1, teacher_ceiling=0.5, entropy="doc-test",
    )
    assert [f.info["teacher_p"] for f in fp] == [0.1, 0.2]
    assert [f.info["teacher_p"] for f in student] == [0.2, 0.5, 1.0]
    assert flags["frame1"]["teacher_trust_reject"]
    selected, _ = select_student_gmm_train(
        scored_frames(), pool=-1, keep=5, policy="random", entropy="doc-test",
        feature_vectors=None, student_floor=0.1, student_ceiling=0.5,
    )
    assert [f.info["student_p"] for f in selected] == [0.1, 0.2]
    selected, _ = select_student_gmm_train(
        scored_frames(), pool=5, keep=5, policy="random", entropy="doc-test",
        feature_vectors=None, student_floor=0.1, student_ceiling=None,
    )
    assert selected[-1].info["student_p"] == 1.0


def test_documented_next_iteration_handoff(tmp_path):
    script = ROOT / "examples/iteration_handoff/check.py"
    run_root = tmp_path / "handoff"
    subprocess.run([sys.executable, str(script), str(run_root)], check=True)
    summary = json.loads((run_root / "handoff_check.json").read_text())
    assert summary["returned_frames"] > 0
    assert summary["iter1_teacher_frames"] > 8
    config = load_run_config(run_root / "run.json")
    assert config.iter_index == 1
    assert config.raw["student_train_bootstrap"] == []
    assert config.student.get("include_teacher_train_data", False) is False
    payload = json.loads((run_root / "run.json").read_text())
    payload["fp_data"] = []
    invalid = run_root / "invalid.json"
    invalid.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fp_data"):
        load_run_config(invalid)
    result = subprocess.run([sys.executable, str(script), str(run_root)], capture_output=True, text=True)
    assert result.returncode != 0 and "new or empty" in result.stderr


def test_documented_explicit_sampling_snippet_uses_real_loader(tmp_path):
    # Exercise the official loader with the retained synthetic fixture, not an
    # archived training configuration that is deliberately not distributed.
    import importlib.util
    spec = importlib.util.spec_from_file_location("_dice_documented_mock", ROOT / "examples/cpu_mock/fixtures.py")
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    payload = json.loads(fixture._write_public_run(tmp_path).read_text())
    payload["sampler"] = {"backend": "LAMMPS"}
    payload["sample"] = {
        "seeds": {"bulk": {"source": "seeds/md.extxyz", "selection_group": "bulk"}},
        "condition_groups": {"warm": {"kind": "nvt", "recipe": "recipes/nvt.lammps", "temperatures_k": [300], "steps": 200}},
        "round_count": 2,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    config = load_run_config(path)
    assert config.round_count == 2
    assert config.selection_plan.group_ids == ("bulk",)


def test_html_build_includes_new_pages_and_data_asset(tmp_path):
    destination = tmp_path / "site"
    subprocess.run([sys.executable, str(ROOT / "tools/build_docs.py"), "--output", str(destination)], check=True)
    for path in ("docs/data.html", "docs/own-system.html", "docs/configuration-fields.html", "examples/iteration_handoff/README.html", "examples/data/labelled_h2.extxyz"):
        assert (destination / path).is_file()
