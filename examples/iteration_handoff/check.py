"""Demonstrate an external-label handoff with synthetic CPU services only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/cpu_mock"))
from backend_roles import compose_test_backend_roles
from fixtures import _FakeSampler, _FakeStudent, _FakeTeacher, _write_public_run

from workflow_app.entrypoint import main as dice_main
from workflow_core.lifecycle.config import load_run_config, validate_role_recipe_feasibility, validate_sampler_recipe_feasibility
from workflow_core.lifecycle.data import LabelContract, validate_label_properties
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


def check(run_root: Path) -> dict[str, int]:
    root = run_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("choose a new or empty run directory; existing data will not be overwritten")
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_public_run(root)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["visualize"] = False

    def execute(iter_index: int) -> None:
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (root / f"run_iter{iter_index}.input.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        config = load_run_config(config_path)
        validate_role_recipe_feasibility(config, "teacher")
        validate_role_recipe_feasibility(config, "student")
        validate_sampler_recipe_feasibility(config)
        dice_main([str(config_path)], backend_composition=compose_test_backend_roles(
            config_path, teacher=_FakeTeacher(), student=_FakeStudent(), sampling=_FakeSampler(),
        ))
        if not (root / f"iter{iter_index}/fp/selected.xyz").is_file():
            raise AssertionError(f"iter{iter_index} did not publish its terminal selection")

    execute(0)
    selected = read_extxyz_structures(root / "iter0/fp/selected.xyz", allow_empty=False)
    returned = []
    for atoms in selected:
        frame = atoms.copy()
        frame.positions[:, 0] += 0.007
        frame.info.pop("structure_id", None)
        frame.info.pop("structure_source", None)
        frame.info["label_origin"] = "synthetic_documentation_test_not_DFT"
        frame.calc = SinglePointCalculator(frame, energy=-1.0, forces=np.zeros((len(frame), 3)))
        returned.append(frame)
    validate_label_properties(returned, LabelContract())
    returned_path = root / "data/fp_returned_iter0.extxyz"
    write_extxyz_structures(returned_path, returned, allow_empty=False)
    # Compare what was actually serialized/read, not higher-precision in-memory
    # coordinates: extxyz is the public handoff boundary.
    returned = read_extxyz_structures(returned_path, allow_empty=False)
    payload.update(iter=1, student_train_bootstrap=[], fp_data=["data/fp_returned_iter0.extxyz"])
    execute(1)
    teacher = read_extxyz_structures(root / "iter1/teach/train/dataset/train.xyz", allow_empty=False)
    teacher_by_geometry = {exact_structure_hash(frame): frame for frame in teacher}
    student = read_extxyz_structures(root / "iter1/teach/screen/train.xyz", allow_empty=False)
    student_geometries = {exact_structure_hash(frame) for frame in student}
    for frame in returned:
        geometry = exact_structure_hash(frame)
        assert geometry in teacher_by_geometry
        assert float(teacher_by_geometry[geometry].get_potential_energy()) == -1.0
        assert geometry not in student_geometries
    prior = [
        *read_extxyz_structures(root / "iter0/r1/train/dataset/train.xyz", allow_empty=False),
        *read_extxyz_structures(root / "iter0/r1/screen/train.xyz", allow_empty=True),
    ]
    assert student_geometries == {exact_structure_hash(frame) for frame in prior}
    validate_label_properties(teacher, LabelContract())
    validate_label_properties(student, LabelContract())
    summary = {"selected_frames": len(selected), "returned_frames": len(returned),
               "iter1_teacher_frames": len(teacher), "iter1_student_seed_geometries": len(student_geometries)}
    (root / "handoff_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    try:
        summary = check(args.run_directory)
    except ValueError as exc:
        parser.error(str(exc))
    print("Synthetic iter-0 → iter-1 handoff passed: " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
