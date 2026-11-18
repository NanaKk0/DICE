from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from workflow_core.lifecycle.config import load_run_config
from workflow_core.lifecycle.seed_frame_sampling import select_frame_indices


def test_lammps_controls_survive_round_and_multiframe_expansion(tmp_path):
    options = {'control': {'mode': 'managed', 'temperature_damping_ps': 0.1},
               'timestep_ps': 0.001, 'trajectory_every_steps': 20}
    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz'}}, 'round_count': 2,
        'condition_groups': {'bulk': {'kind': 'nvt', 'recipe': 'recipes/nvt.lammps',
            'temperatures_k': [300, 600], 'steps': [20, 40], 'lammps': options}}})
    config = load_run_config(run)
    expanded = config.sampling_plan.expand_source_frames({'A': (0, 2)})
    tasks = expanded.tasks_for_round(1)
    assert len(tasks) == 4
    assert all(task.lammps.to_payload() == options and task.steps == 40 for task in tasks)


@pytest.mark.parametrize('max_frames', [None, 2])
def test_control_preflight_reports_sources_and_fails_before_mutation(tmp_path, max_frames):
    from workflow_app.composition import preflight_sampling_protocols
    from workflow_core.lifecycle.launch import prepare_lifecycle_launch

    run = _write_lammps_run(tmp_path, sampler={'backend': 'LAMMPS'}, sample={
        'seeds': {'A': {'source': 'data/A.xyz', **({'max_frames': max_frames} if max_frames else {})}}, 'round_count': 1,
        'condition_groups': {'bulk': {'kind': 'npt', 'recipe': 'recipes/nvt.lammps',
            'temperatures_k': [300, 600], 'pressures_bar': [1], 'steps': 20,
            'lammps': {'control': {'mode': 'managed', 'pressure_damping_ps': 1.0}}}}})
    recipe = tmp_path/'recipes/nvt.lammps'
    recipe.write_text('units metal\natom_style atomic\ntimestep 0.001\n'
                     'fix bath all nvt temp 300 300 0.1\nrun ${dice_run_steps}\n')
    if max_frames:
        from ase import Atoms
        from workflow_core.structures.formats import write_extxyz_structures
        write_extxyz_structures(tmp_path/'data/A.xyz', [Atoms('H', cell=[4]*3, pbc=True),
            Atoms('H', positions=[[1, 0, 0]], cell=[4]*3, pbc=True)])
    launch = prepare_lifecycle_launch(run)
    summary = preflight_sampling_protocols(launch, first_block='r0/sample')
    assert summary[0]['task_count'] == (4 if max_frames else 2)
    assert summary[0]['settings']['pressure_damping_ps']['value'] == 1.0
    payload = json.loads(run.read_text())
    del payload['sample']['condition_groups']['bulk']['lammps']['control']['pressure_damping_ps']
    run.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='bulk.*pressure_damping_ps'):
        preflight_sampling_protocols(prepare_lifecycle_launch(run), first_block='r0/sample')
    assert not (tmp_path/'iter0').exists()


def _write_run(root: Path, *, sample: dict[str, object]) -> Path:
    (root / "data").mkdir()
    (root / "recipes").mkdir()
    for name in ("teacher.xyz", "student.xyz", "A.xyz", "B.xyz"):
        (root / "data" / name).write_text("seed\n", encoding="utf-8")
    (root / "recipes" / "teacher.yaml").write_text("model: MACE\n", encoding="utf-8")
    (root / "recipes" / "nep.in").write_text("version 4\n", encoding="utf-8")
    (root / "recipes" / "run.in").write_text(
        "velocity 300\nensemble nvt_ber 300 300 100\nrun 20\n",
        encoding="utf-8",
    )
    payload = {
        "iter": 0,
        "teacher_train_bootstrap": ["data/teacher.xyz"],
        "student_train_bootstrap": ["data/student.xyz"],
        "fp_data": [],
        "teacher": {
            "backend": "MACE", "model": "single", "type_map": ["H"],
            "recipe": "recipes/teacher.yaml", "policy": "from_scratch", "valid_frac": 0.0,
        },
        "student": {
            "backend": "NEP", "model": "single", "type_map": ["H"],
            "recipe": "recipes/nep.in", "policy": "from_scratch", "valid_frac": 0.0,
        },
        "sampler": {"backend": "GPUMD", "recipe": "recipes/run.in"},
        "sample": sample,
        "screen": {"pool": 1, "keep": 1, "fp_pool": 1, "policy": "random"},
        "fp": {"keep": 1, "policy": "random"},
    }
    path = root / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_lammps_run(
    root: Path, *, sample: dict[str, object], sampler: dict[str, object],
) -> Path:
    path = _write_run(root, sample=sample)
    (root / "recipes" / "student.json").write_text(
        json.dumps({"model": {"descriptor": {"type": "se_e2_a"}}}), encoding="utf-8",
    )
    (root / "recipes" / "nvt.lammps").write_text(
        "include dice_task_setup.lammps\nfix thermostat all nvt temp 300 300 100\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    (root / "recipes" / "npt.lammps").write_text(
        "include dice_task_setup.lammps\nfix barostat all npt temp 300 300 100 iso 1 1 1000\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["student"] = {
        "backend": "DeepMD", "cli": "tf", "model": "single", "type_map": ["H"],
        "recipe": "recipes/student.json", "policy": "from_scratch", "valid_frac": 0.0,
    }
    payload["sampler"] = sampler
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fully_system_specific_conditions_expand_in_canonical_order(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "systems": {
                "A": {"seed": "data/A.xyz", "ensemble": "nvt", "temperatures_k": [300, 500]},
                "B": {
                    "seed": "data/B.xyz", "ensemble": "npt",
                    "temperatures_k": [400, 600], "pressures_bar": [1, 10],
                },
            },
            "steps": [20],
        },
    )
    config = load_run_config(run)
    plan = config.sampling_plan

    assert plan.system_task_counts == {"A": 2, "B": 4}
    assert plan.total_tasks == 6
    assert [
        (task.system_id, task.ensemble, task.temperature_k, task.pressure_bar)
        for task in plan.tasks
    ] == [
        ("A", "nvt", 300.0, None),
        ("A", "nvt", 500.0, None),
        ("B", "npt", 400.0, 1.0),
        ("B", "npt", 400.0, 10.0),
        ("B", "npt", 600.0, 1.0),
        ("B", "npt", 600.0, 10.0),
    ]


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (
            {"ensemble": "nvt", "temperatures_k": [300, 500]},
            [("A", "nvt", 300.0, None), ("A", "nvt", 500.0, None)],
        ),
        (
            {"ensemble": "npt", "temperatures_k": [300], "pressures_bar": -5},
            [("A", "npt", 300.0, -5.0)],
        ),
    ],
)
def test_historical_common_only_conditions_remain_valid(
    tmp_path: Path,
    condition: dict[str, object],
    expected: list[tuple[str, str, float, float | None]],
) -> None:
    run = _write_run(
        tmp_path,
        sample={"systems": {"A": {"seed": "data/A.xyz"}}, "steps": [20], **condition},
    )

    config = load_run_config(run)
    plan = config.sampling_plan

    assert [(item.system_id, item.ensemble, item.temperature_k, item.pressure_bar) for item in plan.tasks] == expected


def test_boolean_pressure_is_rejected_with_its_field_name(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "systems": {"A": {"seed": "data/A.xyz"}},
            "steps": [20], "ensemble": "npt", "temperatures_k": [300], "pressures_bar": True,
        },
    )

    with pytest.raises(ValueError, match=r"sample\.pressures_bar"):
        load_run_config(run)


def test_cross_ensemble_fallback_uses_only_relevant_defaults(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "systems": {
                "A": {"seed": "data/A.xyz"},
                "B": {"seed": "data/B.xyz", "ensemble": "npt", "temperatures_k": [400, 500]},
            },
            "steps": [20], "ensemble": "nvt", "temperatures_k": [300], "pressures_bar": 5,
        },
    )

    config = load_run_config(run)
    plan = config.sampling_plan

    assert [(item.system_id, item.temperature_k, item.pressure_bar) for item in plan.tasks] == [
        ("A", 300.0, None),
        ("B", 400.0, 5.0),
        ("B", 500.0, 5.0),
    ]


@pytest.mark.parametrize(
    ("sample_patch", "error"),
    [
        ({}, r"sample\.systems\.A\.ensemble"),
        ({"ensemble": "nve", "temperatures_k": [300]}, r"sample\.ensemble"),
        ({"ensemble": "nvt"}, r"sample\.systems\.A\.temperatures_k"),
        ({"ensemble": "npt", "temperatures_k": [300]}, r"sample\.systems\.A\.pressures_bar"),
        ({"ensemble": "nvt", "temperatures_k": [300, 300]}, r"sample\.temperatures_k"),
        ({"ensemble": "nvt", "temperatures_k": [0]}, r"sample\.temperatures_k"),
        ({"ensemble": "nvt", "temperatures_k": [True]}, r"sample\.temperatures_k"),
        ({"ensemble": "nvt", "temperatures_k": 300}, r"sample\.temperatures_k"),
        ({"ensemble": "npt", "temperatures_k": [300], "pressures_bar": []}, r"sample\.pressures_bar"),
        ({"ensemble": "npt", "temperatures_k": [300], "pressures_bar": [1, 1]}, r"sample\.pressures_bar"),
        ({"ensemble": "npt", "temperatures_k": [300], "pressures_bar": None}, r"sample\.pressures_bar"),
        ({"ensemble": "npt", "temperatures_k": [300], "pressures_bar": float("inf")}, r"sample\.pressures_bar"),
        ({"ensemble": "nvt", "temperatures_k": [300], "unknown": 1}, r"sample contains unsupported"),
    ],
)
def test_sampling_condition_error_boundaries_are_field_specific(
    tmp_path: Path, sample_patch: dict[str, object], error: str,
) -> None:
    run = _write_run(
        tmp_path,
        sample={"systems": {"A": {"seed": "data/A.xyz"}}, "steps": [20], **sample_patch},
    )

    with pytest.raises(ValueError, match=error):
        load_run_config(run)


def test_effective_nvt_rejects_only_an_explicit_system_pressure(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "systems": {
                "A": {"seed": "data/A.xyz", "ensemble": "nvt", "pressures_bar": 1},
                "B": {"seed": "data/B.xyz", "ensemble": "npt"},
            },
            "steps": [20], "temperatures_k": [300], "pressures_bar": 5,
        },
    )

    with pytest.raises(ValueError, match=r"sample\.systems\.A\.pressures_bar.*forbidden"):
        load_run_config(run)


def test_heterogeneous_lammps_plan_accepts_exact_ensemble_recipe_map(tmp_path: Path) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "systems": {
                "A": {"seed": "data/A.xyz", "ensemble": "nvt", "temperatures_k": [300]},
                "B": {
                    "seed": "data/B.xyz", "ensemble": "npt",
                    "temperatures_k": [400], "pressures_bar": [1, 10],
                },
            },
            "steps": [20],
        },
        sampler={
            "backend": "LAMMPS",
            "recipes": {"nvt": "recipes/nvt.lammps", "npt": "recipes/npt.lammps"},
        },
    )

    config = load_run_config(run)

    assert config.sampling_plan.ensembles == ("nvt", "npt")
    assert config.sampler["recipes"] == {
        "nvt": "recipes/nvt.lammps", "npt": "recipes/npt.lammps",
    }


@pytest.mark.parametrize(
    ("sampler", "error"),
    [
        (
            {"backend": "LAMMPS", "recipe": "recipes/nvt.lammps", "recipes": {"nvt": "recipes/nvt.lammps"}},
            "exactly one of recipe or recipes",
        ),
        (
            {"backend": "LAMMPS", "recipe": "recipes/nvt.lammps"},
            "heterogeneous LAMMPS ensemble plans require sampler.recipes",
        ),
        (
            {"backend": "LAMMPS", "recipes": {"nvt": "recipes/nvt.lammps"}},
            "keys must exactly match effective ensembles",
        ),
        (
            {
                "backend": "LAMMPS",
                "recipes": {
                    "nvt": "recipes/nvt.lammps", "npt": "recipes/npt.lammps", "nve": "recipes/nvt.lammps",
                },
            },
            "unsupported field",
        ),
    ],
)
def test_lammps_recipe_map_rejects_ambiguous_or_inexact_ownership(
    tmp_path: Path, sampler: dict[str, object], error: str,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "systems": {
                "A": {"seed": "data/A.xyz", "ensemble": "nvt", "temperatures_k": [300]},
                "B": {
                    "seed": "data/B.xyz", "ensemble": "npt",
                    "temperatures_k": [400], "pressures_bar": 1,
                },
            },
            "steps": [20],
        },
        sampler=sampler,
    )

    with pytest.raises(ValueError, match=error):
        load_run_config(run)


def test_gpumd_rejects_sampler_recipes_map(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "systems": {"A": {"seed": "data/A.xyz"}},
            "steps": [20], "ensemble": "nvt", "temperatures_k": [300],
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD", "recipes": {"nvt": "recipes/run.in"}}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sampler.recipes is valid only for LAMMPS"):
        load_run_config(run)


def test_explicit_seed_condition_groups_expand_in_group_major_round_order(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 2,
            "seeds": {
                "seed-a": {"source": "data/A.xyz"},
                "seed-b": {"source": "data/B.xyz"},
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300, 500],
                    "steps": [20, 30],
                },
                "pressure": {
                    "kind": "npt",
                    "recipe": "recipes/npt.lammps",
                    "seeds": ["seed-b"],
                    "temperatures_k": [400],
                    "pressures_bar": [1, 10],
                    "steps": 40,
                },
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    config = load_run_config(run)
    plan = config.sampling_plan

    assert config.round_count == 2
    assert [
        (
            task.condition_group_id,
            task.sampling_role_id,
            task.ensemble,
            task.temperature_k,
            task.pressure_bar,
            task.steps,
        )
        for task in plan.tasks_for_round(1)
    ] == [
        ("fixed", "seed-a", "nvt", 300.0, None, 30),
        ("fixed", "seed-a", "nvt", 500.0, None, 30),
        ("fixed", "seed-b", "nvt", 300.0, None, 30),
        ("fixed", "seed-b", "nvt", 500.0, None, 30),
        ("pressure", "seed-b", "npt", 400.0, 1.0, 40),
        ("pressure", "seed-b", "npt", 400.0, 10.0, 40),
    ]


def test_explicit_neb_groups_expand_as_independent_tasks_in_registry_order(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "hop-a": {"source": "data/A.xyz", "selection_group": "paths"},
                "hop-b": {"source": "data/B.xyz", "selection_group": "paths"},
            },
            "condition_groups": {
                "neb-a": {
                    "kind": "neb",
                    "seeds": ["hop-a"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                    "include_history": True,
                },
                "neb-b": {
                    "kind": "neb",
                    "seeds": ["hop-b"],
                    "images": 5,
                    "fmax_ev_a": 0.1,
                    "max_steps": 80,
                },
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    config = load_run_config(run)

    tasks = config.sampling_plan.tasks_for_round(0)
    assert [task.task_id for task in tasks] == ["task_000000", "task_000001"]
    assert [
        (
            task.condition_group_id,
            task.sampling_role_id,
            task.selection_group_id,
            task.kind,
            task.neb.images,
            task.neb.fmax_ev_a,
            task.neb.max_steps,
            task.neb.include_history,
        )
        for task in tasks
    ] == [
        ("neb-a", "hop-a", "paths", "neb", 3, 0.05, 40, True),
        ("neb-b", "hop-b", "paths", "neb", 5, 0.1, 80, False),
    ]


def test_explicit_relax_group_normalizes_defaults_and_expands_seed_tasks(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "relax-a": {"source": "data/A.xyz", "selection_group": "basins"},
                "relax-b": {"source": "data/B.xyz", "selection_group": "basins"},
            },
            "condition_groups": {
                "basin-relax": {
                    "kind": "relax",
                    "seeds": ["relax-a", "relax-b"],
                    "fmax_ev_a": 0.02,
                    "max_steps": 400,
                    "accept_unconverged": True,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    config = load_run_config(run)
    tasks = config.sampling_plan.tasks_for_round(0)

    assert [task.task_id for task in tasks] == ["task_000000", "task_000001"]
    assert [
        (
            task.condition_group_id,
            task.sampling_role_id,
            task.selection_group_id,
            task.kind,
            task.recipe_binding,
            task.steps,
            task.relax.fmax_ev_a,
            task.relax.max_steps,
            task.relax.max_step_a,
            task.relax.include_history,
            task.relax.history_stride,
            task.relax.accept_unconverged,
            task.relax.fixed_atoms,
        )
        for task in tasks
    ] == [
        (
            "basin-relax", "relax-a", "basins", "relax", None, None,
            0.02, 400, None, True, 1, True, (),
        ),
        (
            "basin-relax", "relax-b", "basins", "relax", None, None,
            0.02, 400, None, True, 1, True, (),
        ),
    ]


def test_relax_scientific_intent_is_preserved_in_the_sampling_plan(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"relax-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "basin-relax": {
                    "kind": "relax",
                    "fmax_ev_a": 0.02,
                    "max_steps": 400,
                    "max_step_a": 0.1,
                    "include_history": True,
                    "history_stride": 5,
                    "fixed_atoms": [0],
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")
    config = load_run_config(run)
    task = config.sampling_plan.tasks_for_round(0)[0]

    assert task.kind == "relax"
    assert task.relax.fmax_ev_a == 0.02
    assert task.relax.max_steps == 400
    assert task.relax.max_step_a == 0.1
    assert task.relax.include_history is True
    assert task.relax.history_stride == 5
    assert task.relax.fixed_atoms == (0,)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"fmax_ev_a": 0}, r"\.fmax_ev_a must be a finite positive number"),
        ({"max_steps": True}, r"\.max_steps must be a positive integer"),
        ({"max_step_a": float("inf")}, r"\.max_step_a must be a finite positive number"),
        ({"include_history": 1}, r"\.include_history must be a boolean"),
        ({"accept_unconverged": 1}, r"\.accept_unconverged must be a boolean"),
        ({"history_stride": 0}, r"\.history_stride must be a positive integer"),
        (
            {"include_history": False, "history_stride": 2},
            r"\.history_stride is valid only when include_history=true",
        ),
        ({"fixed_atoms": [0, 0]}, r"\.fixed_atoms must be ordered unique"),
        ({"unknown": 1}, r"contains unsupported field\(s\): unknown"),
    ],
)
def test_relax_group_rejects_invalid_closed_configuration(
    tmp_path: Path,
    update: dict[str, object],
    error: str,
) -> None:
    declaration: dict[str, object] = {
        "kind": "relax",
        "fmax_ev_a": 0.02,
        "max_steps": 100,
    }
    declaration.update(update)
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"relax-a": {"source": "data/A.xyz"}},
            "condition_groups": {"basin-relax": declaration},
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_run_config(run)


def test_relax_only_round_selects_model_family_relax_role_not_md_sampler_role(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.launch import block_operation_roles, prepare_lifecycle_launch

    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"basin": {"source": "data/A.xyz"}},
            "condition_groups": {
                "relax": {
                    "kind": "relax",
                    "fmax_ev_a": 0.02,
                    "max_steps": 100,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    launch = prepare_lifecycle_launch(run)

    assert block_operation_roles(launch.config, "r0/sample") == (
        "student.nep.relax",
    )


def test_explicit_neb_group_accepts_canonical_ordinary_protocol(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "path": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 16,
                    "fmax_ev_a": 0.05,
                    "max_steps": 500,
                    "include_history": False,
                    "init": "idpp",
                    "method": "neb",
                    "climb": True,
                    "climb_preconverge_fmax_ev_a": 0.1,
                    "fixed_atoms": [0, 2, 5],
                    "relax": True,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    task = load_run_config(run).sampling_plan.tasks_for_round(0)[0]

    assert task.neb.images == 16
    assert task.neb.init == "idpp"
    assert task.neb.method == "neb"
    assert task.neb.climb is True
    assert task.neb.climb_preconverge_fmax_ev_a == 0.1
    assert task.neb.fixed_atoms == (0, 2, 5)
    assert task.neb.relax is True


def test_explicit_neb_group_omitted_algorithm_fields_use_improved_tangent(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "path": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 4,
                    "fmax_ev_a": 0.05,
                    "max_steps": 100,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    parameters = load_run_config(run).sampling_plan.tasks_for_round(0)[0].neb

    assert parameters.init == "linear"
    assert parameters.method == "neb"
    assert parameters.climb is False
    assert parameters.fixed_atoms == ()
    assert parameters.relax is False


def test_explicit_neb_group_accepts_closed_dyneb_method(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "path": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 4,
                    "fmax_ev_a": 0.05,
                    "max_steps": 100,
                    "method": "dyneb",
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    parameters = load_run_config(run).sampling_plan.tasks_for_round(0)[0].neb

    assert parameters.method == "dyneb"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("images", True, r"\.images must be a positive integer"),
        ("images", 0, r"\.images must be a positive integer"),
        ("images", 1.5, r"\.images must be a positive integer"),
        ("init", "cubic", r"\.init must be one of"),
        ("method", "aseneb", r"\.method must be one of"),
        ("climb", 1, r"\.climb must be a boolean"),
        ("fixed_atoms", "0", r"\.fixed_atoms must be an ordered unique"),
        ("fixed_atoms", [True], r"\.fixed_atoms must be an ordered unique"),
        ("fixed_atoms", [-1], r"\.fixed_atoms must be an ordered unique"),
        ("fixed_atoms", [0, 0], r"\.fixed_atoms must be an ordered unique"),
        ("relax", 1, r"\.relax must be a boolean"),
    ],
)
def test_explicit_neb_group_rejects_invalid_canonical_protocol_values(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    declaration: dict[str, object] = {
        "kind": "neb",
        "seeds": ["hop"],
        "images": 4,
        "fmax_ev_a": 0.05,
        "max_steps": 100,
    }
    declaration[field] = value
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {"path": declaration},
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_run_config(run)


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        (
            {"climb_preconverge_fmax_ev_a": 0.1},
            r"climb_preconverge_fmax_ev_a requires climb=true",
        ),
        (
            {"climb": True, "climb_preconverge_fmax_ev_a": 0.05},
            r"climb_preconverge_fmax_ev_a must be finite and greater than fmax_ev_a",
        ),
        (
            {
                "method": "dyneb",
                "climb": True,
                "climb_preconverge_fmax_ev_a": 0.1,
            },
            r"climb_preconverge_fmax_ev_a requires method='neb'",
        ),
    ],
)
def test_explicit_neb_group_rejects_invalid_two_stage_climb_combinations(
    tmp_path: Path,
    updates: dict[str, object],
    error: str,
) -> None:
    declaration: dict[str, object] = {
        "kind": "neb",
        "seeds": ["hop"],
        "images": 4,
        "fmax_ev_a": 0.05,
        "max_steps": 100,
    }
    declaration.update(updates)
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {"path": declaration},
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_run_config(run)


def test_explicit_neb_group_rejects_retired_intermediate_images(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "path": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "intermediate_images": 4,
                    "fmax_ev_a": 0.05,
                    "max_steps": 100,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"sample\.condition_groups\.path contains unsupported field\(s\): intermediate_images",
    ):
        load_run_config(run)


def test_public_neb_examples_use_only_the_canonical_images_field() -> None:
    root = Path(__file__).resolve().parents[1]
    public_owners = (
        root / "README.md",
        root / "docs/reference/sampling.md",
        root / "docs/reference/run-json.md",
        root / "docs/guides/sampling.md",
    )

    for path in public_owners:
        text = path.read_text(encoding="utf-8")
        assert '"intermediate_images"' not in text
    sampling_reference = public_owners[1].read_text(encoding="utf-8")
    for field in ('images', 'init', 'method', 'climb', 'fixed_atoms', 'relax'):
        assert any(spelling in sampling_reference for spelling in (
            f'"{field}"', f'`{field}`', f'`{field}:',
        )), field


def test_neb_endpoint_role_forbids_frame_sampling(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "hop": {"source": "data/A.xyz", "max_frames": 2},
            },
            "condition_groups": {
                "neb": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                },
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"sample\.seeds\.hop\.max_frames is forbidden for NEB endpoint roles",
    ):
        load_run_config(run)


def test_one_sampling_role_cannot_bind_both_neb_and_md_groups(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "neb": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                },
                "md": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "seeds": ["hop"],
                    "temperatures_k": [300],
                    "steps": 20,
                },
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    with pytest.raises(
        ValueError,
        match=r"sample\.seeds\.hop cannot be assigned to both NEB and MD groups",
    ):
        load_run_config(run)


def test_explicit_round_activation_filters_in_registry_order(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "seed-a": {"source": "data/A.xyz", "selection_group": "joint"},
                "seed-b": {"source": "data/B.xyz", "selection_group": "joint"},
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300],
                    "steps": [20, 30],
                },
                "pressure": {
                    "kind": "npt",
                    "recipe": "recipes/npt.lammps",
                    "temperatures_k": [400],
                    "pressures_bar": [1],
                    "steps": [40, 50],
                },
            },
            "rounds": [
                {"seeds": ["seed-b"], "condition_groups": ["pressure"]},
                {
                    "seeds": ["seed-b", "seed-a"],
                    "condition_groups": ["pressure", "fixed"],
                },
            ],
        },
        sampler={"backend": "LAMMPS"},
    )

    config = load_run_config(run)
    plan = config.sampling_plan

    assert config.round_count == 2
    assert [
        (task.condition_group_id, task.sampling_role_id, task.steps)
        for task in plan.tasks_for_round(0)
    ] == [("pressure", "seed-b", 40)]
    assert [
        (task.condition_group_id, task.sampling_role_id, task.steps)
        for task in plan.tasks_for_round(1)
    ] == [
        ("fixed", "seed-a", 30),
        ("fixed", "seed-b", 30),
        ("pressure", "seed-a", 50),
        ("pressure", "seed-b", 50),
    ]
    assert plan.selection_plan_for_round(0).role_to_group == (
        ("seed-b", "joint"),
    )
    assert plan.selection_plan_for_round(1).role_to_group == (
        ("seed-a", "joint"),
        ("seed-b", "joint"),
    )


def test_explicit_all_active_rounds_share_plan_with_round_count(
    tmp_path: Path,
) -> None:
    count_root = tmp_path / "count"
    rounds_root = tmp_path / "rounds"
    count_root.mkdir()
    rounds_root.mkdir()
    common = {
        "seeds": {
            "seed-a": {"source": "data/A.xyz", "selection_group": "joint"},
            "seed-b": {"source": "data/B.xyz", "selection_group": "joint"},
        },
        "condition_groups": {
            "fixed": {
                "kind": "nvt",
                "recipe": "recipes/nvt.lammps",
                "temperatures_k": [300],
                "steps": 20,
            }
        },
    }
    count_run = _write_lammps_run(
        count_root,
        sample={**common, "round_count": 1},
        sampler={"backend": "LAMMPS"},
    )
    rounds_run = _write_lammps_run(
        rounds_root,
        sample={**common, "rounds": [{}]},
        sampler={"backend": "LAMMPS"},
    )
    count = load_run_config(count_run)
    rounds = load_run_config(rounds_run)
    assert count.sampling_plan.tasks_for_round(0) == rounds.sampling_plan.tasks_for_round(0)
    assert count.sampling_plan.selection_plan_for_round(0) == rounds.sampling_plan.selection_plan_for_round(0)


@pytest.mark.parametrize(
    "mutation",
    [
        "mixed-count",
        "empty-rounds",
        "empty-seeds",
        "duplicate-seed",
        "unknown-seed",
        "empty-groups",
        "duplicate-group",
        "unknown-group",
        "ineffective-intersection",
        "inactive-seed",
    ],
)
def test_explicit_rounds_reject_invalid_activation_before_execution(
    tmp_path: Path, mutation: str,
) -> None:
    sample: dict[str, object] = {
        "seeds": {
            "seed-a": {"source": "data/A.xyz"},
            "seed-b": {"source": "data/B.xyz"},
        },
        "condition_groups": {
            "for-a": {
                "kind": "nvt",
                "recipe": "recipes/nvt.lammps",
                "seeds": ["seed-a"],
                "temperatures_k": [300],
                "steps": 20,
            },
            "for-b": {
                "kind": "nvt",
                "recipe": "recipes/npt.lammps",
                "seeds": ["seed-b"],
                "temperatures_k": [400],
                "steps": 20,
            },
        },
        "rounds": [{}],
    }
    if mutation == "mixed-count":
        sample["round_count"] = 1
    elif mutation == "empty-rounds":
        sample["rounds"] = []
    elif mutation == "empty-seeds":
        sample["rounds"] = [{"seeds": []}]
    elif mutation == "duplicate-seed":
        sample["rounds"] = [{"seeds": ["seed-a", "seed-a"]}]
    elif mutation == "unknown-seed":
        sample["rounds"] = [{"seeds": ["missing"]}]
    elif mutation == "empty-groups":
        sample["rounds"] = [{"condition_groups": []}]
    elif mutation == "duplicate-group":
        sample["rounds"] = [{"condition_groups": ["for-a", "for-a"]}]
    elif mutation == "unknown-group":
        sample["rounds"] = [{"condition_groups": ["missing"]}]
    elif mutation == "ineffective-intersection":
        sample["rounds"] = [
            {"seeds": ["seed-a"], "condition_groups": ["for-b"]}
        ]
    else:
        sample["rounds"] = [
            {
                "seeds": ["seed-a", "seed-b"],
                "condition_groups": ["for-a"],
            }
        ]
    run = _write_lammps_run(
        tmp_path,
        sample=sample,
        sampler={"backend": "LAMMPS"},
    )

    with pytest.raises(ValueError):
        load_run_config(run)


def test_selected_sample_preflight_reads_only_active_round_inputs(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )
    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "seed-a": {"source": "data/A.xyz"},
                "seed-b": {"source": "data/B.xyz"},
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "seeds": ["seed-a"],
                    "temperatures_k": [300],
                    "steps": 20,
                },
                "future": {
                    "kind": "npt",
                    "recipe": "recipes/npt.lammps",
                    "seeds": ["seed-b"],
                    "temperatures_k": [400],
                    "pressures_bar": [1],
                    "steps": 30,
                },
            },
            "rounds": [
                {"seeds": ["seed-a"], "condition_groups": ["fixed"]}
            ],
        },
        sampler={"backend": "LAMMPS"},
    )
    write(
        tmp_path / "data" / "A.xyz",
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        format="extxyz",
    )
    (tmp_path / "data" / "B.xyz").unlink()
    (tmp_path / "recipes" / "npt.lammps").unlink()

    prepared = prepare_execution_suffix(
        prepare_lifecycle_launch(run),
        first_block="r0/sample",
        require_environment=False,
    )

    assert prepared.config.sampling_plan.tasks_for_round(0)[0].sampling_role_id == "seed-a"


def test_explicit_seed_first_frame_is_validated_at_selected_sample_preflight(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300],
                    "steps": 20,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    (tmp_path / "data" / "A.xyz").write_text("not extxyz\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"sample\.seeds\.seed-a\.source.*first frame"):
        prepare_execution_suffix(
            prepare_lifecycle_launch(run),
            first_block="r0/sample",
            require_environment=False,
        )


def test_neb_endpoint_bundle_requires_exactly_two_frames_at_selected_preflight(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )

    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "neb": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")
    write(
        tmp_path / "data" / "A.xyz",
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        format="extxyz",
    )

    with pytest.raises(
        ValueError,
        match=r"sample\.seeds\.hop\.source.*exactly two canonical endpoint frames",
    ):
        prepare_execution_suffix(
            prepare_lifecycle_launch(run),
            first_block="r0/sample",
            require_environment=False,
        )


def test_neb_fixed_atom_index_is_checked_at_selected_preflight(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )

    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "neb": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                    "fixed_atoms": [1],
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")
    write(
        tmp_path / "data" / "A.xyz",
        [
            Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
            Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
        ],
        format="extxyz",
    )

    with pytest.raises(
        ValueError,
        match=r"sample\.condition_groups\.neb\.fixed_atoms.*outside.*atom order",
    ):
        prepare_execution_suffix(
            prepare_lifecycle_launch(run),
            first_block="r0/sample",
            require_environment=False,
        )


def test_neb_fixed_endpoint_coordinate_mismatch_fails_selected_preflight(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )

    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "neb": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                    "fixed_atoms": [0],
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")
    write(
        tmp_path / "data" / "A.xyz",
        [
            Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True),
            Atoms("HH", positions=[[0.1, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True),
        ],
        format="extxyz",
    )

    with pytest.raises(
        ValueError,
        match=r"sample\.condition_groups\.neb\.fixed_atoms.*identical endpoint coordinates",
    ):
        prepare_execution_suffix(
            prepare_lifecycle_launch(run),
            first_block="r0/sample",
            require_environment=False,
        )


def test_neb_only_round_selects_model_family_neb_role_not_md_sampler_role(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        block_operation_roles,
        prepare_lifecycle_launch,
    )

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"hop": {"source": "data/A.xyz"}},
            "condition_groups": {
                "neb": {
                    "kind": "neb",
                    "seeds": ["hop"],
                    "images": 3,
                    "fmax_ev_a": 0.05,
                    "max_steps": 40,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    write(
        tmp_path / "data" / "A.xyz",
        [
            Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
            Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
        ],
        format="extxyz",
    )

    launch = prepare_lifecycle_launch(run)

    assert block_operation_roles(launch.config, "r0/sample") == (
        "student.deepmd.tf.neb",
    )


def test_explicit_duplicate_effective_recipe_bytes_fail_before_task_creation(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "first": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300],
                    "steps": 20,
                },
                "duplicate": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt-copy.lammps",
                    "temperatures_k": [300],
                    "steps": 20,
                },
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    write(tmp_path / "data" / "A.xyz", Atoms("H", positions=[[0, 0, 0]]), format="extxyz")
    recipe = (tmp_path / "recipes" / "nvt.lammps").read_bytes()
    (tmp_path / "recipes" / "nvt-copy.lammps").write_bytes(recipe)

    with pytest.raises(ValueError, match="duplicate effective native task"):
        prepare_execution_suffix(
            prepare_lifecycle_launch(run),
            first_block="r0/sample",
            require_environment=False,
        )


def test_explicit_nvt_program_normalizes_segments_boundaries_and_rounds(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 2,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "cycle": {
                    "kind": "nvt_program",
                    "recipe": "recipes/nvt.lammps",
                    "segments": [
                        {"temperature_k": 300, "steps": 5},
                        {
                            "start_temperature_k": 300,
                            "stop_temperature_k": 700,
                            "steps": 7,
                        },
                        {"temperature_k": 700, "steps": 3},
                    ],
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    config = load_run_config(run)
    plan = config.sampling_plan
    first = plan.tasks_for_round(0)
    second = plan.tasks_for_round(1)

    assert len(first) == len(second) == 1
    assert first[0].steps == second[0].steps == 15
    assert first[0].program.boundaries == (5, 12, 15)
    assert [
        (segment.start_temperature_k, segment.stop_temperature_k, segment.steps)
        for segment in first[0].program.segments
    ] == [(300.0, 300.0, 5), (300.0, 700.0, 7), (700.0, 700.0, 3)]


def test_explicit_npt_program_keeps_one_continuous_task_per_pressure(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 2,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "cycle": {
                    "kind": "npt_program",
                    "recipe": "recipes/nvt.lammps",
                    "pressures_bar": [1, 2],
                    "segments": [
                        {"temperature_k": 1800, "steps": 5},
                        {"start_temperature_k": 1800, "stop_temperature_k": 1200, "steps": 7},
                        {"temperature_k": 1200, "steps": 3},
                    ],
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    plan = load_run_config(run).sampling_plan
    for round_index in (0, 1):
        tasks = plan.tasks_for_round(round_index)
        assert len(tasks) == 2
        assert [task.pressure_bar for task in tasks] == [1.0, 2.0]
        assert all(task.kind == "npt_program" and task.ensemble == "npt" for task in tasks)
        assert all(task.temperature_k is None and task.steps == 15 for task in tasks)
        assert all(task.program.boundaries == (5, 12, 15) for task in tasks)

    payload = json.loads(run.read_text())
    del payload["sample"]["condition_groups"]["cycle"]["pressures_bar"]
    run.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pressures_bar"):
        load_run_config(run)


def test_explicit_nvt_cell_program_plans_closed_endpoints(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "seed-a": {
                    "source": "data/A.xyz",
                    "selection_group": "joint",
                }
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300],
                    "steps": [5, 5],
                },
                "deform": {
                    "kind": "nvt_cell_program",
                    "recipe": "recipes/npt.lammps",
                    "segments": [
                        {
                            "start_temperature_k": 300,
                            "stop_temperature_k": 700,
                            "steps": 7,
                            "cell": {
                                "mode": "isotropic_density",
                                "stop_density_g_cm3": 3.8,
                            },
                        },
                        {
                            "temperature_k": 700,
                            "steps": 3,
                            "cell": {
                                "mode": "symmetric_strain",
                                "stop": {
                                    "xx": -0.04,
                                    "yy": 0.02,
                                    "zz": 0.01,
                                    "yz": 0.015,
                                    "xz": 0.0,
                                    "xy": -0.02,
                                },
                            },
                        },
                    ],
                },
            },
            "rounds": [
                {"condition_groups": ["fixed"]},
                {"condition_groups": ["deform"]},
            ],
        },
        sampler={"backend": "LAMMPS"},
    )

    config = load_run_config(run)
    task = config.sampling_plan.tasks_for_round(1)[0]

    assert task.kind == "nvt_cell_program"
    assert task.ensemble == "nvt"
    assert task.selection_group_id == "joint"
    assert task.steps == 10
    assert task.program.boundaries == (7, 10)
    assert task.program.segments[0].cell.stop_density_g_cm3 == 3.8
    assert task.program.segments[1].cell.symmetric_strain == (
        ("xx", -0.04),
        ("yy", 0.02),
        ("zz", 0.01),
        ("yz", 0.015),
        ("xz", 0.0),
        ("xy", -0.02),
    )


@pytest.mark.parametrize(
    ("kind", "cell", "error"),
    [
        (
            "nvt_cell_program",
            {
                "mode": "isotropic_density",
                "stop_density_g_cm3": 3.8,
                "units": "g/cm3",
            },
            "unsupported field",
        ),
        (
            "nvt_cell_program",
            {"mode": "isotropic_density", "stop_density_g_cm3": True},
            "finite and positive",
        ),
        (
            "nvt_cell_program",
            {"mode": "isotropic_density", "stop_density_g_cm3": float("inf")},
            "finite and positive",
        ),
        (
            "nvt_cell_program",
            {"mode": "absolute_cell", "stop_density_g_cm3": 3.8},
            "mode must be one of",
        ),
        (
            "nvt_cell_program",
            {
                "mode": "symmetric_strain",
                "stop": {"xx": 0, "yy": 0, "zz": 0, "yz": 0, "xz": 0},
            },
            "contain exactly",
        ),
        (
            "nvt_cell_program",
            {
                "mode": "symmetric_strain",
                "stop": {
                    "xx": False,
                    "yy": 0,
                    "zz": 0,
                    "yz": 0,
                    "xz": 0,
                    "xy": 0,
                },
            },
            "finite non-boolean",
        ),
        (
            "nvt_cell_program",
            {
                "mode": "symmetric_strain",
                "stop": {
                    "xx": -1.0,
                    "yy": 0,
                    "zz": 0,
                    "yz": 0,
                    "xz": 0,
                    "xy": 0,
                },
            },
            "positive definite",
        ),
        (
            "nvt_program",
            {"mode": "isotropic_density", "stop_density_g_cm3": 3.8},
            "valid only for nvt_cell_program",
        ),
    ],
)
def test_cell_program_rejects_open_or_nonphysical_endpoints_at_config_boundary(
    tmp_path: Path,
    kind: str,
    cell: dict[str, object],
    error: str,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "deform": {
                    "kind": kind,
                    "recipe": "recipes/nvt.lammps",
                    "segments": [
                        {"temperature_k": 300, "steps": 5, "cell": cell}
                    ],
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    with pytest.raises(ValueError, match=error):
        load_run_config(run)


def test_cell_endpoints_distinguish_effective_tasks_with_identical_recipe_bytes(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.launch import (
        prepare_execution_suffix,
        prepare_lifecycle_launch,
    )

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "density-low": {
                    "kind": "nvt_cell_program",
                    "recipe": "recipes/nvt.lammps",
                    "segments": [
                        {
                            "temperature_k": 300,
                            "steps": 5,
                            "cell": {
                                "mode": "isotropic_density",
                                "stop_density_g_cm3": 3.0,
                            },
                        }
                    ],
                },
                "density-high": {
                    "kind": "nvt_cell_program",
                    "recipe": "recipes/nvt-copy.lammps",
                    "segments": [
                        {
                            "temperature_k": 300,
                            "steps": 5,
                            "cell": {
                                "mode": "isotropic_density",
                                "stop_density_g_cm3": 3.8,
                            },
                        }
                    ],
                },
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    write(
        tmp_path / "data" / "A.xyz",
        Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),
        format="extxyz",
    )
    recipe = (tmp_path / "recipes" / "nvt.lammps").read_bytes()
    (tmp_path / "recipes" / "nvt-copy.lammps").write_bytes(recipe)

    prepared = prepare_execution_suffix(
        prepare_lifecycle_launch(run),
        first_block="r0/sample",
        require_environment=False,
    )
    assert len(prepared.config.sampling_plan.tasks_for_round(0)) == 2

    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sample"]["condition_groups"]["density-high"]["segments"][0][
        "cell"
    ]["stop_density_g_cm3"] = 3.0
    run.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate effective native task"):
        prepare_execution_suffix(
            prepare_lifecycle_launch(run),
            first_block="r0/sample",
            require_environment=False,
        )


@pytest.mark.parametrize(
    "segment",
    [
        {"temperature_k": 300, "steps": True},
        {"temperature_k": float("inf"), "steps": 5},
    ],
)
def test_invalid_cell_program_temperature_or_steps_fail_before_artifact_mutation(
    tmp_path: Path,
    segment: dict[str, object],
) -> None:
    from workflow_core.lifecycle.launch import prepare_lifecycle_launch

    segment["cell"] = {
        "mode": "isotropic_density",
        "stop_density_g_cm3": 3.8,
    }
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "deform": {
                    "kind": "nvt_cell_program",
                    "recipe": "recipes/nvt.lammps",
                    "segments": [segment],
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    marker = tmp_path / "iter0" / "r0" / "sample" / ".complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"existing-marker-evidence")

    with pytest.raises(ValueError):
        prepare_lifecycle_launch(run)

    assert marker.read_bytes() == b"existing-marker-evidence"
    assert not (tmp_path / "iter0" / "r0" / "sample" / ".attempts").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "mixed-form",
        "unused-seed",
        "empty-assignment",
        "repeated-assignment",
        "unknown-assignment",
        "wrong-round-steps",
        "ambiguous-segment",
    ],
)
def test_explicit_sampling_rejects_ambiguous_shapes_before_execution(
    tmp_path: Path, mutation: str,
) -> None:
    sample: dict[str, object] = {
        "round_count": 2,
        "seeds": {
            "seed-a": {"source": "data/A.xyz"},
            "seed-b": {"source": "data/B.xyz"},
        },
        "condition_groups": {
            "fixed": {
                "kind": "nvt",
                "recipe": "recipes/nvt.lammps",
                "temperatures_k": [300],
                "steps": [20, 30],
            }
        },
    }
    group = sample["condition_groups"]["fixed"]
    if mutation == "mixed-form":
        sample["systems"] = {"A": {"seed": "data/A.xyz"}}
    elif mutation == "unused-seed":
        group["seeds"] = ["seed-a"]
    elif mutation == "empty-assignment":
        group["seeds"] = []
    elif mutation == "repeated-assignment":
        group["seeds"] = ["seed-a", "seed-a"]
    elif mutation == "unknown-assignment":
        group["seeds"] = ["seed-a", "missing"]
    elif mutation == "wrong-round-steps":
        group["steps"] = [20]
    else:
        group.clear()
        group.update(
            {
                "kind": "nvt_program",
                "recipe": "recipes/nvt.lammps",
                "segments": [{"temperature_k": 300, "stop_temperature_k": 400, "steps": 5}],
            }
        )

    run = _write_lammps_run(
        tmp_path,
        sample=sample,
        sampler={"backend": "LAMMPS"},
    )

    with pytest.raises(ValueError):
        load_run_config(run)


def test_explicit_gpumd_md_condition_groups_use_group_local_recipes(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/run.in",
                    "temperatures_k": [300],
                    "steps": 20,
                }
            },
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    config = load_run_config(run)

    assert config.sampling_plan.tasks_for_round(0)[0].kind == "nvt"
    assert config.sampling_plan.tasks_for_round(0)[0].recipe_binding == "fixed"


@pytest.mark.parametrize("kind", ["nvt_program", "nvt_cell_program", "npt_program"])
def test_explicit_gpumd_rejects_lammps_only_program_groups(
    tmp_path: Path, kind: str
) -> None:
    group: dict[str, object] = {
        "kind": kind,
        "recipe": "recipes/run.in",
        "segments": [{"temperature_k": 300, "steps": 20}],
    }
    if kind == "npt_program":
        group["pressures_bar"] = [1]
    if kind == "nvt_cell_program":
        group["segments"] = [
            {
                "temperature_k": 300,
                "steps": 20,
                "cell": {
                    "mode": "isotropic_density",
                    "stop_density_g_cm3": 3.8,
                },
            }
        ]
    run = _write_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {"program": group},
        },
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["sampler"] = {"backend": "GPUMD"}
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="GPUMD explicit sampling supports only"):
        load_run_config(run)


def test_gpumd_native_sampler_rejects_program_plan_when_config_was_bypassed(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from workflow_backends.gpumd.sampler import GpumdSampler

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"seed-a": {"source": "data/A.xyz"}},
            "condition_groups": {
                "program": {
                    "kind": "nvt_program",
                    "recipe": "recipes/nvt.lammps",
                    "segments": [{"temperature_k": 300, "steps": 20}],
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    plan = load_run_config(run).sampling_plan

    with pytest.raises(ValueError, match="GPUMD supports only nvt and npt MD tasks"):
        GpumdSampler(run, 0, 0, object()).sample(
            group_recipes={"program": tmp_path / "recipes" / "nvt.lammps"},
            sample_dir=tmp_path / "sample",
            models=(tmp_path / "missing-nep.txt",),
            systems={"seed-a": [Atoms("H")]},
            steps=None,
            plan=plan,
            plumed_sources={},
        )


def test_explicit_seed_selection_groups_resolve_once_without_changing_tasks(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "role-a": {"source": "data/A.xyz", "selection_group": "joint"},
                "role-b": {"source": "data/B.xyz", "selection_group": "joint"},
                "role-c": {"source": "data/A.xyz"},
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300, 500],
                    "steps": 20,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    config = load_run_config(run)
    plan = config.sampling_plan

    assert config.selection_plan is plan.selection_plan
    assert plan.selection_plan.group_ids == ("joint", "role-c")
    assert plan.selection_plan.role_to_group == (
        ("role-a", "joint"),
        ("role-b", "joint"),
        ("role-c", "role-c"),
    )
    assert tuple(group.sampling_role_ids for group in plan.selection_plan.groups) == (
        ("role-a", "role-b"),
        ("role-c",),
    )
    assert [
        (task.sampling_role_id, task.selection_group_id)
        for task in plan.tasks_for_round(0)
    ] == [
        ("role-a", "joint"),
        ("role-a", "joint"),
        ("role-b", "joint"),
        ("role-b", "joint"),
        ("role-c", "role-c"),
        ("role-c", "role-c"),
    ]
    assert plan.selection_plan.uniform(7) == (("joint", 7), ("role-c", 7))


def test_explicit_multiframe_seed_expands_selected_indices_before_conditions(
    tmp_path: Path,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 2,
            "seeds": {
                "pool80": {
                    "source": "data/A.xyz",
                    "selection_group": "natoms80",
                    "max_frames": 3,
                }
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [100, 600],
                    "steps": 200,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    plan = load_run_config(run).sampling_plan
    expanded = plan.expand_source_frames({"pool80": (1, 4, 7)})

    assert [seed.sampling_role_id for seed in expanded.seeds] == [
        "pool80__frame_000001",
        "pool80__frame_000004",
        "pool80__frame_000007",
    ]
    assert expanded.selection_plan.group_ids == ("natoms80",)
    assert [
        (task.sampling_role_id, task.source_frame_index, task.temperature_k)
        for task in expanded.tasks_for_round(1)
    ] == [
        ("pool80__frame_000001", 1, 100.0),
        ("pool80__frame_000001", 1, 600.0),
        ("pool80__frame_000004", 4, 100.0),
        ("pool80__frame_000004", 4, 600.0),
        ("pool80__frame_000007", 7, 100.0),
        ("pool80__frame_000007", 7, 600.0),
    ]


def test_multiframe_random_selection_is_stable_unique_and_never_refills() -> None:
    arguments = {
        "source_sha256": "a" * 64,
        "iter_index": 0,
        "round_index": 1,
        "sampling_role_id": "pool80",
    }
    selected, seed = select_frame_indices(frame_count=9, max_frames=4, **arguments)
    repeated, repeated_seed = select_frame_indices(frame_count=9, max_frames=4, **arguments)
    underfilled, _ = select_frame_indices(frame_count=3, max_frames=84, **arguments)

    assert selected == repeated
    assert seed == repeated_seed
    assert len(selected) == len(set(selected)) == 4
    assert underfilled == (0, 1, 2)


def test_sampling_evidence_resolves_multiframe_source_per_round(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.runner import _sampling_evidence_request

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 2,
            "seeds": {
                "pool80": {
                    "source": "data/A.xyz",
                    "selection_group": "natoms80",
                    "max_frames": 3,
                }
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [100, 600],
                    "steps": 200,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    frames = []
    for index in range(5):
        atoms = Atoms("H", positions=[[float(index), 0.0, 0.0]])
        atoms.info["structure_id"] = f"source-{index}"
        frames.append(atoms)
    write(tmp_path / "data" / "A.xyz", frames, format="extxyz")
    model = tmp_path / "graph.pb"
    model.write_bytes(b"model")
    config = load_run_config(run)

    evidence = _sampling_evidence_request(
        config,
        1,
        tmp_path / "iter0" / "r1" / "sample",
        models=OrderedSamplingModels((model,), member_ids=("student",)),
    )

    assert len(evidence.seed_selections) == 1
    selection = evidence.seed_selections[0]
    assert len(selection.selected_indices) == 3
    assert tuple(source.system_id for source in evidence.systems) == tuple(
        f"pool80__frame_{index:06d}" for index in selection.selected_indices
    )
    assert evidence.plan.total_tasks == 6
    assert evidence.plan.selection_plan.group_ids == ("natoms80",)


def test_sampling_evidence_uses_only_each_rounds_effective_roles_and_tasks(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.runner import _sampling_evidence_request

    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "role-a": {"source": "data/A.xyz", "selection_group": "shared"},
                "role-b": {"source": "data/B.xyz", "selection_group": "shared"},
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "seeds": ["role-a"],
                    "temperatures_k": [300],
                    "steps": [20, 30],
                },
                "pressure": {
                    "kind": "npt",
                    "recipe": "recipes/npt.lammps",
                    "seeds": ["role-b"],
                    "temperatures_k": [400],
                    "pressures_bar": [1],
                    "steps": [40, 50],
                },
            },
            "rounds": [
                {"seeds": ["role-a"], "condition_groups": ["fixed"]},
                {"seeds": ["role-b"], "condition_groups": ["pressure"]},
            ],
        },
        sampler={"backend": "LAMMPS"},
    )
    for name, x in (("A.xyz", 0.0), ("B.xyz", 1.0)):
        write(
            tmp_path / "data" / name,
            Atoms("H", positions=[[x, 0.0, 0.0]]),
            format="extxyz",
        )
    model = tmp_path / "graph.pb"
    model.write_bytes(b"model")
    config = load_run_config(run)

    first = _sampling_evidence_request(
        config,
        0,
        tmp_path / "iter0" / "r0" / "sample",
        models=OrderedSamplingModels((model,), member_ids=("student",)),
    )
    second = _sampling_evidence_request(
        config,
        1,
        tmp_path / "iter0" / "r1" / "sample",
        models=OrderedSamplingModels((model,), member_ids=("student",)),
    )

    assert tuple(source.system_id for source in first.systems) == ("role-a",)
    assert tuple(source.system_id for source in second.systems) == ("role-b",)
    assert [task.condition_group_id for task in first.plan.tasks_for_round(0)] == [
        "fixed"
    ]
    assert [task.condition_group_id for task in second.plan.tasks_for_round(1)] == [
        "pressure"
    ]
    assert first.plan.selection_plan_for_round(0).group_ids == ("shared",)
    assert second.plan.selection_plan_for_round(1).group_ids == ("shared",)


def test_terminal_selection_plan_filters_candidate_rounds_in_global_role_order(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.runner import _artifact_checks, _terminal_selection_plan

    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "role-a": {"source": "data/A.xyz", "selection_group": "joint"},
                "role-b": {"source": "data/B.xyz", "selection_group": "solo"},
                "role-c": {"source": "data/A.xyz", "selection_group": "joint"},
            },
            "condition_groups": {
                "first": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "seeds": ["role-a"],
                    "temperatures_k": [300],
                    "steps": 20,
                },
                "final": {
                    "kind": "npt",
                    "recipe": "recipes/npt.lammps",
                    "seeds": ["role-b", "role-c"],
                    "temperatures_k": [400],
                    "pressures_bar": [1],
                    "steps": 30,
                },
            },
            "rounds": [
                {"seeds": ["role-a"], "condition_groups": ["first"]},
                {
                    "seeds": ["role-c", "role-b"],
                    "condition_groups": ["final"],
                },
            ],
        },
        sampler={"backend": "LAMMPS"},
    )
    for name, x in (("A.xyz", 0.0), ("B.xyz", 1.0)):
        write(
            tmp_path / "data" / name,
            Atoms("H", positions=[[x, 0.0, 0.0]]),
            format="extxyz",
        )
    config = load_run_config(run)

    final_only = _terminal_selection_plan(config, (1,))
    all_rounds = _terminal_selection_plan(config, (0, 1))

    assert final_only.group_ids == ("solo", "joint")
    assert final_only.role_to_group == (
        ("role-b", "solo"),
        ("role-c", "joint"),
    )
    assert all_rounds.group_ids == ("joint", "solo")
    assert all_rounds.role_to_group == (
        ("role-a", "joint"),
        ("role-b", "solo"),
        ("role-c", "joint"),
    )

    class CaptureRecovery:
        facts = None

        @classmethod
        def bind_final_fp_recovery_validator(
            cls, facts, *, candidate_paths, teacher_router
        ):
            cls.facts = facts
            return lambda _path: True

    _artifact_checks(config, blocks=("fp",), composition=CaptureRecovery)

    assert CaptureRecovery.facts.system_ids == ("solo", "joint")
    assert CaptureRecovery.facts.selection_plan.role_to_group == (
        ("role-b", "solo"),
        ("role-c", "joint"),
    )


def test_screen_recovery_validates_only_round_local_selection_groups(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.runner import (
        _artifact_checks,
        _configured_screen_report_system_ids,
    )

    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "role-a": {"source": "data/A.xyz", "selection_group": "first"},
                "role-b": {"source": "data/B.xyz", "selection_group": "second"},
            },
            "condition_groups": {
                "cold": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "seeds": ["role-a"],
                    "temperatures_k": [100],
                    "steps": 20,
                },
                "warm": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "seeds": ["role-b"],
                    "temperatures_k": [200],
                    "steps": 20,
                },
            },
            "rounds": [
                {"seeds": ["role-a"], "condition_groups": ["cold"]},
                {"seeds": ["role-b"], "condition_groups": ["warm"]},
            ],
        },
        sampler={"backend": "LAMMPS"},
    )
    for name, x in (("A.xyz", 0.0), ("B.xyz", 1.0)):
        write(
            tmp_path / "data" / name,
            Atoms("H", positions=[[x, 0.0, 0.0]]),
            format="extxyz",
        )
    captured = []

    def capture(request):
        captured.append(request)
        return True

    monkeypatch.setattr(
        "workflow_artifacts.lifecycle.screen_validation.validate_screen_artifacts",
        capture,
    )
    config = load_run_config(run)
    checks = _artifact_checks(config, blocks=("r0/screen", "r1/screen"))

    assert checks["r0/screen"](tmp_path)
    assert checks["r1/screen"](tmp_path)
    assert [request.expected_system_ids for request in captured] == [
        ("first",),
        ("second",),
    ]
    assert [request.selection_plan.group_ids for request in captured] == [
        ("first",),
        ("second",),
    ]
    assert _configured_screen_report_system_ids(SimpleNamespace(config=config)) == {
        "r0/screen": ("first",),
        "r1/screen": ("second",),
    }


def test_joint_direct_screen_recovery_freezes_model_and_dataset_identity_per_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.runner import _artifact_checks

    run = _write_lammps_run(
        tmp_path,
        sample={
            "seeds": {
                "role-a": {
                    "source": "data/A.xyz",
                    "selection_group": "first",
                }
            },
            "condition_groups": {
                "cold": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [100],
                    "steps": 20,
                }
            },
            "rounds": [
                {"condition_groups": ["cold"]},
                {"condition_groups": ["cold"]},
            ],
        },
        sampler={"backend": "LAMMPS"},
    )
    write(
        tmp_path / "data" / "A.xyz",
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        format="extxyz",
    )
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["screen"] = {
        "prescreen": "JOINT_DIRECT",
        "teacher_p": {"floor": 0.0, "trust": 0.1, "ceiling": 0.1},
        "pool": 1,
        "keep": 1,
        "fp_pool": -1,
        "policy": "random",
    }
    run.write_text(json.dumps(payload), encoding="utf-8")
    captured = []

    def capture(request):
        captured.append(request)
        return True

    monkeypatch.setattr(
        "workflow_artifacts.lifecycle.screen_validation.validate_screen_artifacts",
        capture,
    )
    config = load_run_config(run)
    checks = _artifact_checks(config, blocks=("r0/screen", "r1/screen"))

    assert checks["r0/screen"](tmp_path)
    assert checks["r1/screen"](tmp_path)
    assert [request.joint_model_identity for request in captured] == [
        "iter0/r0/train/student",
        "iter0/r1/train/student",
    ]
    assert [request.joint_dataset_identity for request in captured] == [
        "iter0/r0/train/dataset",
        "iter0/r1/train/dataset",
    ]


def test_sample_recovery_uses_the_same_multiframe_expansion_as_execution(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_core.lifecycle.runner import _artifact_checks

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "pool80": {
                    "source": "data/A.xyz",
                    "selection_group": "natoms80",
                    "max_frames": 3,
                }
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [100, 600],
                    "steps": 200,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    frames = []
    for index in range(5):
        atoms = Atoms("H", positions=[[float(index), 0.0, 0.0]])
        atoms.info["structure_id"] = f"source-{index}"
        frames.append(atoms)
    write(tmp_path / "data" / "A.xyz", frames, format="extxyz")

    class CaptureRecovery:
        sample_facts = None

        @staticmethod
        def bind_student_training_recovery_validator(_facts):
            return lambda _path: True

        @classmethod
        def bind_sample_recovery_validator(cls, facts):
            cls.sample_facts = facts
            return lambda _path: True

    config = load_run_config(run)
    _artifact_checks(
        config,
        blocks=("r0/sample",),
        composition=CaptureRecovery,
    )

    facts = CaptureRecovery.sample_facts
    assert facts is not None
    expected_indices, _ = select_frame_indices(
        source_sha256=hashlib.sha256(
            (tmp_path / "data" / "A.xyz").read_bytes()
        ).hexdigest(),
        frame_count=5,
        max_frames=3,
        iter_index=0,
        round_index=0,
        sampling_role_id="pool80",
    )
    assert facts.system_ids == tuple(
        f"pool80__frame_{index:06d}" for index in expected_indices
    )
    assert len(facts.tasks) == 6
    assert tuple(task.source_frame_index for task in facts.tasks) == tuple(
        index for index in expected_indices for _temperature in (100, 600)
    )


def test_sample_recovery_rebuilds_multiframe_plan_from_committed_manifest(
    tmp_path: Path,
) -> None:
    from ase import Atoms
    from ase.io import write

    from workflow_backends.lammps.sampler import _publish_seed_selection_manifest
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.runner import (
        _artifact_checks,
        _sampling_evidence_request,
    )

    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "pool80": {
                    "source": "data/A.xyz",
                    "selection_group": "natoms80",
                    "max_frames": 3,
                }
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [100, 600],
                    "steps": 200,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )
    frames = [
        Atoms("H", positions=[[float(index), 0.0, 0.0]])
        for index in range(5)
    ]
    for index, atoms in enumerate(frames):
        atoms.info["structure_id"] = f"source-{index}"
    source = tmp_path / "data" / "A.xyz"
    write(source, frames, format="extxyz")
    model = tmp_path / "graph.pb"
    model.write_bytes(b"model")
    config = load_run_config(run)
    evidence = _sampling_evidence_request(
        config,
        0,
        tmp_path / "iter0" / "r0" / "sample",
        models=OrderedSamplingModels((model,), member_ids=("student",)),
    )
    manifest = tmp_path / "iter0" / "r0" / "sample" / "seed_selection.json"
    _publish_seed_selection_manifest(
        manifest,
        iter_index=0,
        round_index=0,
        selections=evidence.seed_selections,
    )
    source.unlink()

    class CaptureRecovery:
        sample_facts = None

        @staticmethod
        def bind_student_training_recovery_validator(_facts):
            return lambda _path: True

        @classmethod
        def bind_sample_recovery_validator(cls, facts):
            cls.sample_facts = facts
            return lambda _path: True

    _artifact_checks(
        config,
        blocks=("r0/sample",),
        composition=CaptureRecovery,
    )

    facts = CaptureRecovery.sample_facts
    assert facts is not None
    assert facts.system_ids == tuple(
        f"pool80__frame_{index:06d}"
        for index in evidence.seed_selections[0].selected_indices
    )


def test_seed_selection_manifest_is_idempotent_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    from workflow_backends.lammps.sampler import _publish_seed_selection_manifest
    from workflow_backends.sampling import SamplingSeedSelection

    selection = SamplingSeedSelection(
        source_role_id="pool80",
        source_sha256="a" * 64,
        frame_count=5,
        max_frames=3,
        selected_indices=(0, 2, 4),
        random_seed_sha256="b" * 64,
        algorithm="sha256-python-random-v1",
    )
    target = tmp_path / "sample" / "seed_selection.json"

    _publish_seed_selection_manifest(
        target, iter_index=0, round_index=1, selections=(selection,)
    )
    original = target.read_bytes()
    _publish_seed_selection_manifest(
        target, iter_index=0, round_index=1, selections=(selection,)
    )
    assert target.read_bytes() == original

    with pytest.raises(ValueError, match="disagrees"):
        _publish_seed_selection_manifest(
            target, iter_index=0, round_index=0, selections=(selection,)
        )


@pytest.mark.parametrize("max_frames", [0, -1, True, 1.5, "3"])
def test_explicit_multiframe_seed_rejects_invalid_max_frames(
    tmp_path: Path, max_frames: object,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {"pool80": {"source": "data/A.xyz", "max_frames": max_frames}},
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [100],
                    "steps": 200,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    with pytest.raises(ValueError, match="max_frames"):
        load_run_config(run)


def test_generic_thirteen_role_fixture_resolves_seven_first_appearance_groups(
    tmp_path: Path,
) -> None:
    seeds = {
        f"role-{index:02d}": {
            "source": "data/A.xyz",
            "selection_group": f"group-{min(index // 2, 6):02d}",
        }
        for index in range(13)
    }
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 2,
            "seeds": seeds,
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300],
                    "steps": [20, 30],
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    plan = load_run_config(run).sampling_plan

    assert len(plan.seeds) == len(plan.tasks_for_round(0)) == len(plan.tasks_for_round(1)) == 13
    assert plan.selection_plan.group_ids == tuple(f"group-{index:02d}" for index in range(7))


@pytest.mark.parametrize("selection_group", ["", "bad/group", True, 3])
def test_explicit_seed_rejects_invalid_selection_group(
    tmp_path: Path, selection_group: object,
) -> None:
    run = _write_lammps_run(
        tmp_path,
        sample={
            "round_count": 1,
            "seeds": {
                "role-a": {
                    "source": "data/A.xyz",
                    "selection_group": selection_group,
                }
            },
            "condition_groups": {
                "fixed": {
                    "kind": "nvt",
                    "recipe": "recipes/nvt.lammps",
                    "temperatures_k": [300],
                    "steps": 20,
                }
            },
        },
        sampler={"backend": "LAMMPS"},
    )

    with pytest.raises(ValueError, match="selection_group"):
        load_run_config(run)
