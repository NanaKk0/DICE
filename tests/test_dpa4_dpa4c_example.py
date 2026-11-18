"""Validate the bundled scientific inputs without running native engines."""
import json
from pathlib import Path

import numpy as np
from ase.io import read

from workflow_backends.lammps.native_protocol import compile_lammps_protocol
from workflow_core.lifecycle.config import load_run_config
from workflow_core.lifecycle.launch import prepare_lifecycle_launch


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/dpa4_dpa4c"


def test_example_supplies_all_scientific_inputs():
    config = load_run_config(EXAMPLE / "run.json")
    assert config.round_count == 2
    for role in (config.teacher, config.student):
        assert role["type_map"] == ["O", "H"]
        native = json.loads((EXAMPLE / role["recipe"]).read_text())
        assert native["model"]["descriptor"]["precision"] == "float32"
        assert native["loss"]["start_pref_v"] == native["loss"]["limit_pref_v"] == 0
        for population in ("training_data", "validation_data"):
            assert "systems" not in native["training"].get(population, {})
    teacher = read(EXAMPLE / "data/teacher_initial.extxyz", ":")
    student = read(EXAMPLE / "data/student_initial.extxyz", ":")
    additional = read(EXAMPLE / "data/demo_iter1_labeled.extxyz", ":")
    seeds = read(EXAMPLE / "seeds/seed.extxyz", ":")
    assert (len(teacher), len(student), len(additional), len(seeds)) == (40, 40, 20, 2)
    assert config.teacher["policy"] == "foundation_finetune"
    assert config.teacher["foundation_model"] == "models/DPA4-Neo-OMat24-v20260805.pt"
    assert config.teacher["train_options"]["skip_neighbor_stat"]
    assert config.student["policy"] == "from_scratch"
    assert config.student["include_teacher_train_data"] is True
    for first, second in zip(teacher, student):
        assert first.get_chemical_symbols() == second.get_chemical_symbols()
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.cell, second.cell)
    initial_ids = {atoms.info["structure_id"] for atoms in teacher}
    assert initial_ids.isdisjoint(atoms.info["structure_id"] for atoms in additional)
    for atoms in teacher + student + additional + seeds:
        assert len(atoms) == 192
        assert set(atoms.info) <= {"system_id", "structure_id", "structure_source"}
        assert set(atoms.arrays) == {"numbers", "positions"}
        assert set(atoms.get_chemical_symbols()) == {"O", "H"}
        assert atoms.pbc.all() and np.linalg.det(atoms.cell) > 0
        assert np.isfinite(atoms.positions).all()
    for atoms in teacher + student + additional:
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()
    for atoms in seeds:
        assert atoms.calc is None
    assert config.raw["screen"]["prescreen"] == "JOINT_DIRECT"
    assert "student_p" not in config.raw["screen"]
    assert config.raw["screen"]["teacher_p"] == {"floor": 0.0, "ceiling": 0.1, "trust": 0.1}
    assert config.raw["screen"]["pool"] == 200
    assert config.raw["screen"]["keep"] == 50
    assert config.raw["screen"]["policy"] == "DIRECT"
    assert config.raw["fp"]["keep"] == 20
    assert config.raw["fp"]["all_rn"]
    protocol = (EXAMPLE / "recipes/nvt.lammps").read_text()
    assert "${dice_" not in protocol
    for rn in range(2):
        tasks = config.sampling_plan.tasks_for_round(rn)
        assert {task.temperature_k for task in tasks} == {300, 600}
        observations = 0
        for task in tasks:
            compiled = compile_lammps_protocol(protocol, task.protocol_request)
            assert compiled.temperature_k == task.temperature_k
            assert compiled.total_steps == 2500
            assert compiled.timestep_ps == 0.00025
            assert compiled.cadence == 10
            observations += len(seeds) * (compiled.total_steps // compiled.cadence)
        assert observations == 1000


def test_next_iteration_preserves_config_and_adds_separate_reference_data():
    initial = json.loads((EXAMPLE / "run.json").read_text())
    following = json.loads((EXAMPLE / "run_iter1.json").read_text())
    config = load_run_config(EXAMPLE / "run_iter1.json")
    assert config.iter_index == 1
    assert following["student_train_bootstrap"] == []
    assert following["fp_data"] == ["data/demo_iter1_labeled.extxyz"]
    assert config.student["include_teacher_train_data"] is True
    assert {key for key in initial if initial[key] != following[key]} == {
        "iter", "student_train_bootstrap", "fp_data",
    }


def test_example_profiles_bind_to_the_complete_lifecycle():
    launch = prepare_lifecycle_launch(
        EXAMPLE / "run.json",
        environment_path=EXAMPLE / "environment.json",
        execution_path=EXAMPLE / "execution.json",
    )
    assert [block.block_id for block in launch.plan.blocks] == [
        "teach/train", "teach/screen", "teach/scan",
        "r0/train", "r0/sample", "r0/screen",
        "r1/train", "r1/sample", "r1/screen", "fp",
    ]
