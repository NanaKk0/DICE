from __future__ import annotations

import json
from pathlib import Path

import pytest


def _bridged_model(family: str) -> dict:
    return {
        "type": "linear_ener", "weights": "sum", "type_map": ["O", "Y", "Zr"],
        "models": [
            {"type": "dpa4" if family == "dpa4" else "standard",
             "descriptor": {"type": family}, "fitting_net": {"neuron": [96]}},
            {"type": "inner_potential", "mode": "zbl"},
        ],
    }


def test_canonical_zbl_teacher_selects_dpa4_readout_roles(tmp_path: Path) -> None:
    from workflow_backends.teacher.training import preflight_teacher_train_capability

    recipe = tmp_path / "teacher.json"
    recipe.write_text(json.dumps({"model": _bridged_model("dpa4")}))
    capability = preflight_teacher_train_capability(
        backend="DeepMD", cli="pt", recipe_path=recipe, model_shape="single",
        policy="from_scratch", foundation_asset_present=False,
    )

    assert "teacher.deepmd.pt.gmm_reference" in capability.required_operation_roles
    assert "teacher.deepmd.pt.gmm_score" in capability.required_operation_roles


def test_canonical_zbl_students_keep_native_route_and_distinct_learned_seeds(tmp_path: Path) -> None:
    from workflow_backends.student.deepmd.adapter import DeepMDStudentAdapter

    seeds = []

    class NativeRuntime:
        def train(self, input_json, member_dir, cli, *, recipe_kind, **kwargs):
            rendered = json.loads(input_json.read_text())
            assert cli == "pt" and recipe_kind == "dpa4c"
            model = rendered["model"]
            assert "descriptor" not in model and "fitting_net" not in model
            learned = model["models"][0]
            seed = rendered["training"]["seed"]
            assert learned["descriptor"]["seed"] == seed
            assert learned["fitting_net"]["seed"] == seed
            seeds.append(seed)
            output = member_dir / "student.pt2"
            output.write_bytes(b"native terminal")
            return output

    recipe = tmp_path / "student.json"
    recipe.write_text(json.dumps({"model": _bridged_model("dpa4c")}))
    dataset = tmp_path / "train.xyz"
    dataset.write_text("")
    result = DeepMDStudentAdapter(tmp_path / "run.json", NativeRuntime()).train(
        recipe=recipe, dataset_path=dataset, train_dir=tmp_path / "train", cli="pt",
        type_map=("O", "Y", "Zr"), model="committee", members=2,
        policy="from_scratch", foundation_model=None,
    )

    assert len(set(seeds)) == 2
    assert all(model.evidence["native_selector"] == "pt_expt" for model in result)


def test_zbl_finetune_rejects_a_reordered_learned_type_map_before_training(tmp_path: Path) -> None:
    from workflow_backends.student.deepmd.adapter import DeepMDStudentAdapter

    class NativeRuntime:
        def foundation_type_map(self, model, *, work_dir):
            return ("O", "Y", "Zr")

        def train(self, *args, **kwargs):
            raise AssertionError("conflicting foundation type maps must fail before native training")

    model = _bridged_model("dpa4c")
    model["models"][0]["type_map"] = ["Y", "O", "Zr"]
    recipe = tmp_path / "student.json"
    recipe.write_text(json.dumps({"model": model}))
    foundation = tmp_path / "foundation.pt"
    foundation.write_bytes(b"foundation asset")

    with pytest.raises(ValueError, match="recipe type_map must exactly preserve"):
        DeepMDStudentAdapter(tmp_path / "run.json", NativeRuntime()).train(
            recipe=recipe, dataset_path=tmp_path / "train.xyz", train_dir=tmp_path / "train",
            cli="pt", type_map=("Zr", "O"), model="single", members=1,
            policy="foundation_finetune", foundation_model=foundation,
        )


@pytest.mark.parametrize("mutation", ["mean", "two_learned", "two_inner"])
def test_unsupported_bridging_shapes_fail_before_teacher_launch(tmp_path: Path, mutation: str) -> None:
    from workflow_backends.teacher.training import preflight_teacher_train_capability

    model = _bridged_model("dpa4")
    if mutation == "mean":
        model["weights"] = "mean"
    else:
        model["models"].append(model["models"][0 if mutation == "two_learned" else 1])
    recipe = tmp_path / "teacher.json"
    recipe.write_text(json.dumps({"model": model}))

    with pytest.raises(ValueError, match="one learned model and one inner_potential"):
        preflight_teacher_train_capability(
            backend="DeepMD", cli="pt", recipe_path=recipe, model_shape="single",
            policy="from_scratch", foundation_asset_present=False,
        )
