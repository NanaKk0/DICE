from __future__ import annotations

import json
from pathlib import Path


def test_dpa4_preset_selects_teacher_readout_roles_without_importing_backend(tmp_path: Path) -> None:
    from workflow_backends.teacher.training import preflight_teacher_train_capability

    recipe = tmp_path / "teacher.json"
    recipe.write_text(json.dumps({"model": {"preset": "dpa4-nano-v20260911"}}))

    capability = preflight_teacher_train_capability(
        backend="DeepMD", cli="pt", recipe_path=recipe, model_shape="single",
        policy="from_scratch", foundation_asset_present=False,
    )

    assert "teacher.deepmd.pt.gmm_reference" in capability.required_operation_roles
    assert "teacher.deepmd.pt.gmm_score" in capability.required_operation_roles


def test_dpa4c_preset_selects_exportable_student_training(tmp_path: Path) -> None:
    from workflow_backends.student.deepmd.adapter import DeepMDStudentAdapter

    class NativeRuntime:
        def train(self, input_json, member_dir, cli, *, recipe_kind, **kwargs):
            assert cli == "pt"
            assert recipe_kind == "dpa4c"
            rendered = json.loads(input_json.read_text())
            assert rendered["model"]["preset"] == "dpa4c-nano-v20260911"
            assert rendered["model"]["descriptor"]["seed"] == rendered["training"]["seed"]
            output = member_dir / "student.pt2"
            output.write_bytes(b"native terminal")
            return output

    recipe = tmp_path / "student.json"
    recipe.write_text(json.dumps({"model": {
        "preset": "dpa4c-nano-v20260911", "descriptor": {"seed": 42},
    }}))
    dataset = tmp_path / "train.xyz"
    dataset.write_text("")
    result = DeepMDStudentAdapter(tmp_path / "run.json", NativeRuntime()).train(
        recipe=recipe, dataset_path=dataset, train_dir=tmp_path / "train", cli="pt",
        type_map=("Zr", "Y", "O"), model="single", members=1,
        policy="from_scratch", foundation_model=None,
    )

    assert result[0].terminal_model_path.suffix == ".pt2"
    assert result[0].evidence["native_selector"] == "pt_expt"


def test_minimal_preset_recipe_still_gets_member_owned_random_seeds(tmp_path: Path) -> None:
    from workflow_backends.student.deepmd.adapter import DeepMDStudentAdapter

    seeds = []

    class NativeRuntime:
        def train(self, input_json, member_dir, cli, **kwargs):
            rendered = json.loads(input_json.read_text())
            model = rendered["model"]
            seed = rendered["training"]["seed"]
            assert model["descriptor"]["seed"] == seed
            assert model["fitting_net"]["seed"] == seed
            seeds.append(seed)
            output = member_dir / "student.pt2"
            output.write_bytes(b"native terminal")
            return output

    recipe = tmp_path / "student.json"
    recipe.write_text(json.dumps({"model": {"preset": "dpa4c-nano-v20260911"}}))
    dataset = tmp_path / "train.xyz"
    dataset.write_text("")
    DeepMDStudentAdapter(tmp_path / "run.json", NativeRuntime()).train(
        recipe=recipe, dataset_path=dataset, train_dir=tmp_path / "train", cli="pt",
        type_map=("Zr", "Y", "O"), model="committee", members=2,
        policy="from_scratch", foundation_model=None,
    )

    assert len(set(seeds)) == 2


def test_public_recipe_feasibility_accepts_preset_single_student(tmp_path: Path) -> None:
    from workflow_core.lifecycle.config import load_run_config, validate_role_recipe_feasibility

    recipe = tmp_path / "student.json"
    recipe.write_text(json.dumps({"model": {"preset": "dpa4c-nano-v20260911"}}))
    payload = {
        "iter": 0, "teacher_train_bootstrap": ["teacher.xyz"],
        "student_train_bootstrap": ["student.xyz"], "fp_data": [],
        "teacher": {"backend": "DeepMD", "cli": "pt", "model": "single",
                    "type_map": ["O", "H"], "recipe": "teacher.json",
                    "policy": "from_scratch", "valid_frac": 0},
        "student": {"backend": "DeepMD", "cli": "pt", "model": "single",
                    "type_map": ["O", "H"], "recipe": "student.json",
                    "policy": "from_scratch", "valid_frac": 0},
        "sampler": {"backend": "LAMMPS"},
        "sample": {"round_count": 1, "seeds": {"water": {"source": "seed.xyz"}},
                   "condition_groups": {"nvt": {"kind": "nvt", "recipe": "md.in",
                                                "steps": 20, "temperatures_k": [300]}}},
        "screen": {"pool": 2, "keep": 1, "fp_pool": 2, "policy": "random"},
        "fp": {"keep": 1, "policy": "random"},
    }
    run = tmp_path / "run.json"
    run.write_text(json.dumps(payload))

    validate_role_recipe_feasibility(load_run_config(run), "student")
