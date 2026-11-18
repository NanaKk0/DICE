from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from workflow_backends.sampling import (
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingSeedSource,
)
from workflow_backends.sampling_models import OrderedSamplingModels
from workflow_core.common.sampling_plan import (
    ExplicitSamplingPlan,
    NebSamplingParameters,
    SamplingConditionGroup,
    SeedRole,
)


def test_compressed_neb_recovery_uses_neb_model_presentations(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sampling_models import FIXED_COMPRESSION
    from workflow_backends.sample_recovery import _validate_compression

    sample = tmp_path / "sample"
    task = sample / "tasks" / "task_000000"
    derived = sample / "sampling_models" / "student" / "model_compressed.pt2"
    task.mkdir(parents=True)
    derived.parent.mkdir(parents=True)
    derived.write_bytes(b"compressed-model")
    presentation = task / "model_000.pt2"
    presentation.write_bytes(derived.read_bytes())
    digest = hashlib.sha256(derived.read_bytes()).hexdigest()
    (sample / "sampling_model_assets.json").write_text(
        json.dumps(
            {
                "schema": "dice.lifecycle.sampling-model-assets",
                "schema_revision": 1,
                "cli": "pt",
                "compression": FIXED_COMPRESSION,
                "operation_profile": {
                    "role": "student.deepmd.pt.compress",
                    "profile_id": "test-compress",
                    "source_scripts": [],
                    "executables": {"dp": "/usr/bin/dp"},
                    "runtime": {},
                },
                "members": [
                    {
                        "member_id": "student",
                        "source_path": "r0/train/student/model.pt2",
                        "source_sha256": "0" * 64,
                        "derived_path": "sampling_models/student/model_compressed.pt2",
                        "derived_sha256": digest,
                        "deepmd_recipe_kind": "dpa4c",
                        "cli": "pt",
                        "native_selector": "pt_expt",
                        "training_script": "r0/train/student/input.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    facts = SimpleNamespace(
        compress=True,
        cli="pt",
        member_ids=("student",),
        tasks=(SimpleNamespace(kind="neb"),),
    )

    _validate_compression(sample, facts)


class _HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        self.results = {
            "energy": float(np.sum(positions**2)),
            "forces": -2.0 * positions,
        }


class _CalculatorProvider:
    policy_identity = {"name": "committee-force-deviation", "revision": 1}

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        assert model_paths

    def create(self, model_path: Path) -> Calculator:
        assert model_path.is_file()
        return _HarmonicCalculator()


class _PerpendicularCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        displacement = positions.copy()
        displacement[:, 1] -= 1.0
        self.results = {
            "energy": float(np.sum(displacement**2)),
            "forces": -2.0 * displacement,
        }


class _PerpendicularProvider(_CalculatorProvider):
    def create(self, model_path: Path) -> Calculator:
        assert model_path.is_file()
        return _PerpendicularCalculator()


class _DoubleWellCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        x = float(positions[1, 0])
        forces = np.zeros_like(positions)
        forces[1, 0] = -4.0 * x * (x * x - 1.0)
        self.results = {
            "energy": float((x * x - 1.0) ** 2),
            "forces": forces,
        }


class _DoubleWellProvider(_CalculatorProvider):
    def create(self, model_path: Path) -> Calculator:
        assert model_path.is_file()
        return _DoubleWellCalculator()


def test_neb_worker_accepts_declared_siblings_and_global_task_numbers(tmp_path,monkeypatch):
    import workflow_backends.ase_neb.adapter as adapter
    import workflow_backends.ase_neb.worker as worker
    from workflow_runtime.lifecycle.environment import OperationContext
    model=tmp_path/'model.pb';model.write_text('calculator fixture')
    endpoints=(Atoms('HH',positions=[[0,2,0],[-1,0,0]],cell=[6]*3,pbc=True),
               Atoms('HH',positions=[[0,2,0],[1,0,0]],cell=[6]*3,pbc=True))
    plan=ExplicitSamplingPlan((SeedRole('md','joint'),SeedRole('hop','joint')),
        (SamplingConditionGroup('md','nvt',('md',),steps_by_round=(20,),temperatures_k=(300,)),
         SamplingConditionGroup('path','neb',('hop',),neb=NebSamplingParameters(2,1.e6,1))),1)
    evidence=SamplingEvidenceRequest(tmp_path/'run.json',0,0,tmp_path/'sample',plan,
        (SamplingSeedSource('md',(endpoints[0],)),SamplingSeedSource('hop',endpoints)),
        OrderedSamplingModels((model,),member_ids=('student',)),None)
    (evidence.sample_dir/'tasks/task_000000').mkdir(parents=True)
    monkeypatch.setattr(worker,'_DeepMDCalculatorProvider',_DoubleWellProvider)
    def execute(context,executable,arguments,**kwargs):worker.execute_manifest(arguments[-1])
    monkeypatch.setattr(adapter,'run_operation',execute)
    backend=adapter.SubprocessAseNebSamplingNativeOps(OperationContext(role='student.deepmd.pt.neb',
        profile_id='test',source_scripts=(),executables={'python':'python'},runtime={}),model_family='deepmd-pt')
    assert backend.execute(SamplingExecutionRequest(evidence)).task_count==1
    assert {row.task_index for row in backend.open_current(evidence).source.catalog()}=={1}


def test_relax_true_optimizes_both_endpoints_before_band_initialization(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_core.structures.formats import read_extxyz_structures

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms(
            "HH",
            positions=[[0.0, 2.0, 0.0], [-1.5, 0.0, 0.0]],
            cell=[6, 6, 6],
            pbc=True,
        ),
        Atoms(
            "HH",
            positions=[[0.0, 2.0, 0.0], [1.5, 0.0, 0.0]],
            cell=[6, 6, 6],
            pbc=True,
        ),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=2,
                    fmax_ev_a=0.05,
                    max_steps=500,
                    include_history=False,
                    method="neb",
                    fixed_atoms=(0,),
                    relax=True,
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_DoubleWellProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    raw = read_extxyz_structures(task_dir / "raw_endpoints.extxyz", allow_empty=False)
    optimized = read_extxyz_structures(task_dir / "endpoints.extxyz", allow_empty=False)
    preparation = json.loads(
        (task_dir / "endpoint_relaxation.json").read_text(encoding="utf-8")
    )
    context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    result = json.loads((task_dir / "neb_result.json").read_text(encoding="utf-8"))
    assert [float(frame.positions[1, 0]) for frame in raw] == [-1.5, 1.5]
    assert np.allclose(
        [float(frame.positions[1, 0]) for frame in optimized],
        [-1.0, 1.0],
        atol=0.02,
    )
    assert context["endpoint_preparation"] == preparation
    assert result["endpoint_preparation"] == preparation
    assert preparation["driver_member_id"] == "student"
    assert [row["role"] for row in preparation["endpoints"]] == ["initial", "final"]
    assert all(row["converged"] is True for row in preparation["endpoints"])
    assert all(0 <= row["optimizer_steps"] <= 500 for row in preparation["endpoints"])
    assert all(
        np.array_equal(frame.positions[0], endpoints[0].positions[0])
        and np.array_equal(frame.cell.array, endpoints[0].cell.array)
        for frame in optimized
    )
    catalog = backend.open_current(evidence).source.catalog()
    assert [record.image_index for record in catalog] == [0, 1, 2, 3]
    visible = tuple(
        frame
        for chunk in backend.open_current(evidence).source.replay(
            [record.structure_id for record in catalog], chunk_size=4
        )
        for frame in chunk
    )
    assert np.allclose(visible[0].positions, optimized[0].positions)
    assert np.allclose(visible[-1].positions, optimized[-1].positions)
    backend.operation_profile_id = 'dormant-completed-prefix'
    assert backend.validate_current(evidence).complete
    assert len(backend.open_current(evidence).source.catalog()) == len(catalog)


def test_relax_true_rejects_one_sided_endpoint_convergence_before_interpolation(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms(
            "HH",
            positions=[[0.0, 2.0, 0.0], [-1.0, 0.0, 0.0]],
            cell=[6, 6, 6],
            pbc=True,
        ),
        Atoms(
            "HH",
            positions=[[0.0, 2.0, 0.0], [1.5, 0.0, 0.0]],
            cell=[6, 6, 6],
            pbc=True,
        ),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=2,
                    fmax_ev_a=1.0e-12,
                    max_steps=1,
                    method="neb",
                    fixed_atoms=(0,),
                    relax=True,
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_DoubleWellProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    with pytest.raises(ValueError, match="final endpoint relaxation exhausted"):
        backend.execute(SamplingExecutionRequest(evidence))

    task_root = evidence.sample_dir / "tasks"
    assert not (task_root / "task_000000").exists()
    assert not (task_root / ".task_000000.staging").exists()


def test_canonical_ordinary_neb_publishes_exact_algorithm_and_constraint_identity(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_core.structures.formats import read_extxyz_structures

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms(
            "HH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        Atoms(
            "HH",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=2,
                    fmax_ev_a=1.0e6,
                    max_steps=2,
                    init="idpp",
                    method="neb",
                    climb=True,
                    fixed_atoms=(0,),
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    result = json.loads((task_dir / "neb_result.json").read_text(encoding="utf-8"))
    expected_algorithm = {
        "name": "ase-neb-fire",
        "revision": 4,
        "band": "NEB",
        "interpolation": "idpp",
        "interpolation_mic": True,
        "method": "improvedtangent",
        "spring_ev_a2": 0.1,
        "optimizer": "FIRE",
        "climb": True,
        "fixed_cell": True,
        "fixed_endpoints": True,
    }
    expected_constraint = {
        "name": "ase-fix-atoms",
        "revision": 1,
        "indices": [0],
        "zero_based": True,
    }
    assert context["parameters"] == {
        "images": 2,
        "fmax_ev_a": 1.0e6,
        "max_steps": 2,
        "include_history": False,
        "init": "idpp",
        "method": "neb",
        "climb": True,
        "fixed_atoms": [0],
    }
    assert context["algorithm"] == result["algorithm"] == expected_algorithm
    assert context["constraint"] == result["constraint"] == expected_constraint
    assert "endpoint_preparation" not in context
    assert "endpoint_preparation" not in result
    assert not (task_dir / "raw_endpoints.extxyz").exists()
    assert not (task_dir / "endpoint_relaxation.json").exists()
    for name in ("endpoints.extxyz", "initial_band.extxyz", "final_band.extxyz"):
        frames = read_extxyz_structures(task_dir / name, allow_empty=False)
        assert all(tuple(frame.constraints[0].get_indices()) == (0,) for frame in frames)
        assert all(np.array_equal(frame.cell.array, endpoints[0].cell.array) for frame in frames)
        assert all(np.array_equal(frame.positions[0], endpoints[0].positions[0]) for frame in frames)


def test_idpp_interpolation_uses_minimum_image_convention_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workflow_backends.ase_neb.adapter as adapter_module
    from ase.mep import NEB as RealNEB

    recorded: dict[str, object] = {}

    class RecordingNEB(RealNEB):
        def interpolate(self, *args, **kwargs):
            recorded.update(kwargs)
            return super().interpolate(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "NEB", RecordingNEB)
    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        Atoms("HH", positions=[[0.0, 0.0, 0.0], [4.9, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=1,
                    fmax_ev_a=1.0e6,
                    max_steps=2,
                    init="idpp",
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = adapter_module.AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    result = json.loads(
        (evidence.sample_dir / "tasks" / "task_000000" / "neb_result.json")
        .read_text(encoding="utf-8")
    )
    assert recorded["mic"] is True
    assert result["algorithm"]["interpolation_mic"] is True


def test_dyneb_uses_exact_dynamic_relaxation_constructor_and_optimizer_fmax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workflow_backends.ase_neb.adapter as adapter_module
    from ase.mep import DyNEB as RealDyNEB
    from ase.optimize import FIRE as RealFIRE

    recorded: dict[str, object] = {}

    def recording_dyneb(images, **kwargs):
        recorded["constructor"] = dict(kwargs)
        return RealDyNEB(images, **kwargs)

    class RecordingFIRE(RealFIRE):
        def run(self, fmax=0.05, steps=None):
            recorded["optimizer_fmax"] = fmax
            return super().run(fmax=fmax, steps=steps)

    monkeypatch.setattr(adapter_module, "DyNEB", recording_dyneb, raising=False)
    monkeypatch.setattr(adapter_module, "FIRE", RecordingFIRE)
    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms(
            "HH",
            positions=[[0, 2, 0], [0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        Atoms(
            "HH",
            positions=[[0, 2, 0], [1, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=1,
                    fmax_ev_a=1.0e6,
                    max_steps=2,
                    init="idpp",
                    method="dyneb",
                    climb=True,
                    fixed_atoms=(0,),
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = adapter_module.AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    constructor = recorded["constructor"]
    assert constructor["method"] == "improvedtangent"
    assert constructor["dynamic_relaxation"] is True
    assert constructor["scale_fmax"] == 0.0
    assert constructor["fmax"] == recorded["optimizer_fmax"] == 1.0e6
    assert constructor["k"] == 0.1
    assert constructor["climb"] is True
    result = json.loads(
        (
            evidence.sample_dir
            / "tasks"
            / "task_000000"
            / "neb_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["algorithm"]["band"] == "DyNEB"
    assert result["algorithm"]["dynamic_relaxation"] is True
    assert result["algorithm"]["scale_fmax"] == 0.0
    assert result["algorithm"]["fmax_ev_a"] == 1.0e6


def test_two_stage_climb_preconverges_then_uses_remaining_step_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workflow_backends.ase_neb.adapter as adapter_module

    calls: list[dict[str, object]] = []

    class RecordingFIRE:
        def __init__(self, band, logfile=None):
            self.band = band
            self.logfile = logfile
            self.nsteps = 0
            self.callbacks = []

        def attach(self, function, interval=1):
            self.callbacks.append(function)

        def run(self, fmax=0.05, steps=None):
            self.nsteps = 2 if not calls else 3
            calls.append(
                {
                    "climb": bool(self.band.climb),
                    "fmax": float(fmax),
                    "steps": int(steps),
                    "logfile": Path(self.logfile).name,
                }
            )
            for callback in self.callbacks:
                callback()
            return True

    monkeypatch.setattr(adapter_module, "FIRE", RecordingFIRE)
    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms(
            "HH",
            positions=[[0, 2, 0], [0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        Atoms(
            "HH",
            positions=[[0, 2, 0], [1, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
    )
    parameters = NebSamplingParameters(
        images=1,
        fmax_ev_a=0.02,
        max_steps=10,
        method="neb",
        climb=True,
        climb_preconverge_fmax_ev_a=0.05,
        fixed_atoms=(0,),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a", "neb", ("hop",), neb=parameters
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = adapter_module.AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    assert calls == [
        {
            "climb": False,
            "fmax": 0.05,
            "steps": 10,
            "logfile": "optimizer_preclimb.log",
        },
        {
            "climb": True,
            "fmax": 0.02,
            "steps": 8,
            "logfile": "optimizer.log",
        },
    ]
    result = json.loads(
        (
            evidence.sample_dir
            / "tasks"
            / "task_000000"
            / "neb_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["converged"] is True
    assert result["optimizer_steps"] == 5
    assert result["optimization_stages"] == [
        {
            "name": "preclimb",
            "climb": False,
            "fmax_ev_a": 0.05,
            "step_budget": 10,
            "optimizer_steps": 2,
            "converged": True,
        },
        {
            "name": "climb",
            "climb": True,
            "fmax_ev_a": 0.02,
            "step_budget": 8,
            "optimizer_steps": 3,
            "converged": True,
        },
    ]


def test_neb_semantic_validation_rejects_resealed_constraint_drift(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_core.structures.formats import (
        read_extxyz_structures,
        write_extxyz_structures,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True),
        Atoms("HH", positions=[[0, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=1,
                    fmax_ev_a=1.0e6,
                    max_steps=2,
                    method="neb",
                    fixed_atoms=(0,),
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )
    backend.execute(SamplingExecutionRequest(evidence))
    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    band_path = task_dir / "initial_band.extxyz"
    frames = read_extxyz_structures(band_path, allow_empty=False)
    for frame in frames:
        frame.set_constraint()
    write_extxyz_structures(band_path, frames, allow_empty=False)
    result_path = task_dir / "neb_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["files"]["initial_band.extxyz"] = hashlib.sha256(
        band_path.read_bytes()
    ).hexdigest()
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    success_path = task_dir / "task_completion_receipt.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    success_path.write_text(
        json.dumps(success, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    assert backend.validate_current(evidence).complete is False
    with pytest.raises(ValueError, match="constraint"):
        backend.open_current(evidence)


def test_final_only_neb_publishes_ordered_endpoints_and_interior_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import workflow_backends.ase_neb.adapter as adapter_module
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
    )
    parameters = NebSamplingParameters(
        images=3,
        fmax_ev_a=0.05,
        max_steps=10,
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                condition_group_id="neb-a",
                kind="neb",
                sampling_role_ids=("hop",),
                neb=parameters,
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    source = backend.open_current(evidence).source
    catalog = source.catalog()
    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    context = json.loads((task_dir / "task_context.json").read_text(encoding="utf-8"))
    native_result = json.loads((task_dir / "neb_result.json").read_text(encoding="utf-8"))
    success = json.loads((task_dir / "task_completion_receipt.json").read_text(encoding="utf-8"))

    assert result.task_count == 1
    assert context["schema_revision"] == 3
    assert native_result["schema_revision"] == 3
    assert success["schema_revision"] == 3
    assert len(catalog) == 5
    assert [record.optimizer_step for record in catalog] == [0, 0, 0, 0, 0]
    assert [record.image_index for record in catalog] == [0, 1, 2, 3, 4]
    assert all(record.condition_group_id == "neb-a" for record in catalog)
    native_iread = adapter_module.iread
    scan_count = 0

    def counted_iread(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return native_iread(*args, **kwargs)

    monkeypatch.setattr(adapter_module, "iread", counted_iread)
    frames = tuple(
        frame
        for chunk in source.replay(
            [record.structure_id for record in catalog], chunk_size=2
        )
        for frame in chunk
    )
    assert scan_count == 0  # Indexed replay never invokes the whole-file iterator.
    assert [float(frame.positions[0, 0]) for frame in frames] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    scan_count = 0
    reversed_frames = tuple(
        frame
        for chunk in source.replay(
            [record.structure_id for record in reversed(catalog)], chunk_size=2
        )
        for frame in chunk
    )
    assert scan_count == 0  # Indexed replay never invokes the whole-file iterator.
    assert [float(frame.positions[0, 0]) for frame in reversed_frames] == [
        1.0,
        0.75,
        0.5,
        0.25,
        0.0,
    ]


def test_history_neb_admits_complete_initial_and_max_step_bands(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
    )
    parameters = NebSamplingParameters(
        images=1,
        fmax_ev_a=1.0e-12,
        max_steps=2,
        include_history=True,
        method="dyneb",
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                condition_group_id="neb-a",
                kind="neb",
                sampling_role_ids=("hop",),
                neb=parameters,
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_PerpendicularProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))
    catalog = backend.open_current(evidence).source.catalog()
    result = json.loads(
        (
            tmp_path
            / "sample"
            / "tasks"
            / "task_000000"
            / "neb_result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["converged"] is False
    assert result["termination"] == "max_steps"
    assert result["optimizer_steps"] == 2
    assert [record.optimizer_step for record in catalog] == [0, 1, 2]
    assert [record.image_index for record in catalog] == [1, 1, 1]


def test_neb_replay_rejects_a_missing_task_success_marker(tmp_path: Path) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                condition_group_id="neb-a",
                kind="neb",
                sampling_role_ids=("hop",),
                neb=NebSamplingParameters(2, 0.05, 10),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )
    backend.execute(SamplingExecutionRequest(evidence))
    (evidence.sample_dir / "tasks" / "task_000000" / "task_completion_receipt.json").unlink()

    assert backend.validate_current(evidence).complete is False
    with pytest.raises(ValueError, match="completion receipt"):
        backend.open_current(evidence)


def test_mixed_round_allows_unconverged_final_only_neb_to_publish_no_images(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "student_model.pb"
    model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"), SeedRole("md", "paths")),
        condition_groups=(
            SamplingConditionGroup(
                condition_group_id="neb-a",
                kind="neb",
                sampling_role_ids=("hop",),
                neb=NebSamplingParameters(1, 1.0e-12, 1),
            ),
            SamplingConditionGroup(
                condition_group_id="md-a",
                kind="nvt",
                sampling_role_ids=("md",),
                temperatures_k=(300.0,),
                steps_by_round=(10,),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource("hop", endpoints),
            SamplingSeedSource("md", (Atoms("H"),)),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_PerpendicularProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    result = backend.execute(SamplingExecutionRequest(evidence))

    assert result.task_count == 1
    assert backend.open_current(evidence).source.catalog() == ()
    assert backend.validate_current(evidence).complete is True


@pytest.mark.parametrize(
    ("final", "error"),
    [
        (Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]]), "atom counts"),
        (Atoms("He", positions=[[1, 0, 0]]), "element order"),
        (
            Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=False),
            "identical PBC",
        ),
        (
            Atoms("H", positions=[[1, 0, 0]], cell=[6, 5, 5], pbc=True),
            "variable-cell",
        ),
        (
            Atoms("H", positions=[[np.nan, 0, 0]], cell=[5, 5, 5], pbc=True),
            "finite",
        ),
    ],
)
def test_neb_endpoint_invariants_fail_before_execution(
    final: Atoms, error: str
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    initial = Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], cell=[5, 5, 5], pbc=True
    )

    with pytest.raises(ValueError, match=error):
        AseNebSamplingNativeOps._validate_endpoints((initial, final))


def test_neb_endpoint_constraints_must_match_the_declared_fixed_mask() -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    initial = Atoms(
        "HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True
    )
    final = Atoms(
        "HH", positions=[[0, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True
    )
    initial.set_constraint(FixAtoms(indices=[0]))

    with pytest.raises(ValueError, match="endpoint constraints"):
        AseNebSamplingNativeOps._validate_endpoints(
            (initial, final), fixed_atoms=(0,)
        )


def test_neb_fixed_endpoint_coordinates_must_be_identical() -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    initial = Atoms(
        "HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True
    )
    final = Atoms(
        "HH", positions=[[0.1, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True
    )

    with pytest.raises(ValueError, match="fixed endpoint coordinates"):
        AseNebSamplingNativeOps._validate_endpoints(
            (initial, final), fixed_atoms=(0,)
        )


def test_committee_first_member_drives_neb_and_all_members_postevaluate(
    tmp_path: Path,
) -> None:
    from collections import Counter
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    class OffsetCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = np.asarray(self.atoms.positions, dtype=float)
            self.results = {
                "energy": float(np.sum(positions**2) + self.offset),
                "forces": -2.0 * positions + self.offset,
            }

    class RecordingProvider:
        policy_identity = {"name": "committee-force-deviation", "revision": 1}

        def __init__(self) -> None:
            self.created: list[str] = []

        def preflight(self, model_paths: tuple[Path, ...]) -> None:
            assert len(model_paths) == 2

        def create(self, model_path: Path) -> Calculator:
            self.created.append(model_path.name)
            return OffsetCalculator(0.0 if "000" in model_path.name else 1.0)

    models = (tmp_path / "first.pb", tmp_path / "second.pb")
    for model in models:
        model.write_text("model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a", "neb", ("hop",),
                neb=NebSamplingParameters(1, 1.0e6, 2, relax=True),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "hop",
                (
                    Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
        ),
        models=OrderedSamplingModels(models, member_ids=("first", "second")),
        steps=None,
    )
    provider = RecordingProvider()
    backend = AseNebSamplingNativeOps(
        calculator_provider=provider,
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))
    result = json.loads(
        (evidence.sample_dir / "tasks" / "task_000000" / "neb_result.json").read_text(
            encoding="utf-8"
        )
    )
    counts = Counter(provider.created)

    assert counts["first.pb"] == counts["second.pb"] == 1
    assert counts["model_000.pb"] == 5
    assert counts["model_001.pb"] == 1
    assert result["observations"][1]["member_energies_ev"] == [0.25, 1.25]
    assert result["observations"][1]["force_deviation"] > 0.0


def test_history_reuses_driver_results_and_caches_committee_calculators(
    tmp_path: Path,
) -> None:
    from collections import Counter
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    class PerpendicularOffsetCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            displacement = np.asarray(self.atoms.positions, dtype=float).copy()
            displacement[:, 1] -= 1.0
            self.results = {
                "energy": float(np.sum(displacement**2) + self.offset),
                "forces": -2.0 * displacement + self.offset,
            }

    class RecordingProvider:
        policy_identity = {"name": "committee-force-deviation", "revision": 1}

        def __init__(self) -> None:
            self.created: list[str] = []

        def preflight(self, model_paths: tuple[Path, ...]) -> None:
            assert len(model_paths) == 2

        def create(self, model_path: Path) -> Calculator:
            self.created.append(model_path.name)
            return PerpendicularOffsetCalculator(
                0.0 if "000" in model_path.name else 1.0
            )

    models = (tmp_path / "first.pb", tmp_path / "second.pb")
    for model in models:
        model.write_text("model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=1,
                    fmax_ev_a=1.0e-12,
                    max_steps=2,
                    include_history=True,
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "hop",
                (
                    Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
        ),
        models=OrderedSamplingModels(models, member_ids=("first", "second")),
        steps=None,
    )
    provider = RecordingProvider()
    backend = AseNebSamplingNativeOps(
        calculator_provider=provider,
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))
    result = json.loads(
        (evidence.sample_dir / "tasks/task_000000/neb_result.json").read_text(
            encoding="utf-8"
        )
    )
    counts = Counter(provider.created)

    assert len(result["observations"]) == 3
    assert counts["model_000.pb"] == 3
    assert counts["model_001.pb"] == 1
    assert all(
        row["member_energies_ev"][1] - row["member_energies_ev"][0]
        == pytest.approx(1.0)
        for row in result["observations"]
    )


def test_neb_routes_execute_concurrently_up_to_max_workers(
    tmp_path: Path,
) -> None:
    from threading import Condition, Lock
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    class ConcurrencyProbe:
        def __init__(self) -> None:
            self.condition = Condition()
            self.active = 0
            self.max_active = 0

        def overlap(self) -> None:
            with self.condition:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.condition.notify_all()
                self.condition.wait_for(lambda: self.max_active >= 2, timeout=1.0)
                self.active -= 1

    class ProbeCalculator(_HarmonicCalculator):
        def __init__(self, probe: ConcurrencyProbe | None) -> None:
            super().__init__()
            self.probe = probe

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ) -> None:
            if self.probe is not None:
                self.probe.overlap()
                self.probe = None
            super().calculate(atoms, properties, system_changes)

    class ProbeProvider:
        policy_identity = {"name": "committee-force-deviation", "revision": 1}

        def __init__(self, probe: ConcurrencyProbe) -> None:
            self.probe = probe
            self.lock = Lock()
            self.seen_task_dirs: set[str] = set()

        def preflight(self, model_paths: tuple[Path, ...]) -> None:
            assert len(model_paths) == 1

        def create(self, model_path: Path) -> Calculator:
            task_dir = model_path.parent.name
            should_probe = False
            if task_dir.startswith(".task_"):
                with self.lock:
                    if task_dir not in self.seen_task_dirs:
                        self.seen_task_dirs.add(task_dir)
                        should_probe = True
            return ProbeCalculator(self.probe if should_probe else None)

    model = tmp_path / "student.pb"
    model.write_text("model", encoding="utf-8")
    roles = ("hop-a", "hop-b")
    plan = ExplicitSamplingPlan(
        seeds=tuple(SeedRole(role, "paths") for role in roles),
        condition_groups=tuple(
            SamplingConditionGroup(
                f"neb-{index}",
                "neb",
                (role,),
                neb=NebSamplingParameters(1, 1.0e6, 1),
            )
            for index, role in enumerate(roles)
        ),
        round_count=1,
    )
    endpoints = (
        Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=tuple(
            SamplingSeedSource(role, endpoints) for role in roles
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    probe = ConcurrencyProbe()
    backend = AseNebSamplingNativeOps(
        calculator_provider=ProbeProvider(probe),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
        max_workers=2,
    )

    result = backend.execute(SamplingExecutionRequest(evidence))

    assert result.task_count == 2
    assert probe.max_active == 2
    assert backend.validate_current(evidence).complete is True


def test_subprocess_neb_projects_cpu_allocation_into_worker_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import workflow_backends.ase_neb.adapter as adapter_module
    from workflow_runtime.execution.allocation import Allocation
    from workflow_runtime.lifecycle.environment import OperationContext

    captured: dict[str, object] = {}

    def capture_manifest(context, executable, arguments, **kwargs):
        del context, executable, kwargs
        captured.update(
            json.loads(Path(arguments[-1]).read_text(encoding="utf-8"))
        )
        raise RuntimeError("captured worker manifest")

    monkeypatch.setattr(adapter_module, "run_operation", capture_manifest)
    model = tmp_path / "student.pb"
    model.write_text("model", encoding="utf-8")
    roles = tuple(f"hop-{index}" for index in range(9))
    plan = ExplicitSamplingPlan(
        seeds=tuple(SeedRole(role, "paths") for role in roles),
        condition_groups=tuple(
            SamplingConditionGroup(
                f"neb-{index}",
                "neb",
                (role,),
                neb=(
                    NebSamplingParameters(
                        1,
                        0.02,
                        10,
                        climb=True,
                        climb_preconverge_fmax_ev_a=0.05,
                    )
                    if index == 0
                    else NebSamplingParameters(1, 1.0e6, 1)
                ),
            )
            for index, role in enumerate(roles)
        ),
        round_count=1,
    )
    endpoints = (
        Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=tuple(
            SamplingSeedSource(role, endpoints) for role in roles
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = adapter_module.SubprocessAseNebSamplingNativeOps(
        OperationContext(
            role="student.deepmd.pt.neb",
            profile_id="test-neb",
            source_scripts=(),
            executables={"python": "python"},
            runtime={},
            allocation=Allocation(
                nodes=1,
                gpus_per_node=1,
                cpus_per_task=8,
                memory_mb=24576,
                launcher="direct",
                mode="pool",
                per_gpu=1,
            ),
        ),
        model_family="deepmd-pt",
    )

    with pytest.raises(RuntimeError, match="captured worker manifest"):
        backend.execute(SamplingExecutionRequest(evidence))

    assert captured["schema_revision"] == 5
    assert captured["max_workers"] == 4
    assert captured["groups"][0]["climb_preconverge_fmax_ev_a"] == 0.05


def test_neb_runtime_failure_leaves_no_committed_or_staging_task(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    class NonFiniteCalculator(_HarmonicCalculator):
        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["forces"][:] = np.nan

    class FailAfterPreflight(_CalculatorProvider):
        def __init__(self) -> None:
            self.calls = 0

        def create(self, model_path: Path) -> Calculator:
            self.calls += 1
            return _HarmonicCalculator() if self.calls == 1 else NonFiniteCalculator()

    model = tmp_path / "model.pb"
    model.write_text("model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a", "neb", ("hop",),
                neb=NebSamplingParameters(1, 0.05, 2),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "hop",
                (
                    Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=FailAfterPreflight(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    with pytest.raises(Exception):
        backend.execute(SamplingExecutionRequest(evidence))

    assert not (evidence.sample_dir / "tasks" / "task_000000").exists()
    assert not (evidence.sample_dir / "tasks" / ".task_000000.staging").exists()


def test_endpoint_relaxation_non_finite_failure_leaves_no_task_evidence(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    class NonFiniteCalculator(_HarmonicCalculator):
        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["forces"][:] = np.nan

    class FailFirstEndpoint(_CalculatorProvider):
        def __init__(self) -> None:
            self.calls = 0

        def create(self, model_path: Path) -> Calculator:
            self.calls += 1
            return _HarmonicCalculator() if self.calls == 1 else NonFiniteCalculator()

    model = tmp_path / "model.pb"
    model.write_text("model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a",
                "neb",
                ("hop",),
                neb=NebSamplingParameters(
                    images=1,
                    fmax_ev_a=0.05,
                    max_steps=2,
                    relax=True,
                ),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "hop",
                (
                    Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=FailFirstEndpoint(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    with pytest.raises(ValueError, match="endpoint relaxation produced"):
        backend.execute(SamplingExecutionRequest(evidence))

    assert not (evidence.sample_dir / "tasks" / "task_000000").exists()
    assert not (evidence.sample_dir / "tasks" / ".task_000000.staging").exists()


def test_neb_band_corruption_invalidates_read_only_replay(tmp_path: Path) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "model.pb"
    model.write_text("model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a", "neb", ("hop",),
                neb=NebSamplingParameters(1, 1.0e6, 2),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "hop",
                (
                    Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )
    backend.execute(SamplingExecutionRequest(evidence))
    final_band = evidence.sample_dir / "tasks" / "task_000000" / "final_band.extxyz"
    final_band.write_bytes(final_band.read_bytes() + b"\n")

    assert backend.validate_current(evidence).complete is False
    with pytest.raises(ValueError, match="digest mismatch"):
        backend.open_current(evidence)


def test_multiple_neb_paths_execute_and_replay_in_group_order(tmp_path: Path) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    model = tmp_path / "model.pb"
    model.write_text("model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop-a", "joint"), SeedRole("hop-b", "joint")),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a", "neb", ("hop-a",),
                neb=NebSamplingParameters(1, 1.0e6, 2),
            ),
            SamplingConditionGroup(
                "neb-b", "neb", ("hop-b",),
                neb=NebSamplingParameters(2, 1.0e6, 2),
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "hop-a",
                (
                    Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
            SamplingSeedSource(
                "hop-b",
                (
                    Atoms("H", positions=[[2, 0, 0]], cell=[5, 5, 5], pbc=True),
                    Atoms("H", positions=[[4, 0, 0]], cell=[5, 5, 5], pbc=True),
                ),
            ),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.neb",
        operation_profile_id="test-neb",
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    catalog = backend.open_current(evidence).source.catalog()

    assert result.task_count == 2
    assert [record.task_id for record in catalog] == [
        "task_000000", "task_000000", "task_000000",
        "task_000001", "task_000001", "task_000001", "task_000001",
    ]
    assert [record.image_index for record in catalog] == [0, 1, 2, 0, 1, 2, 3]
    assert tuple(
        path.name for path in sorted((evidence.sample_dir / "tasks").iterdir())
    ) == ("task_000000", "task_000001")


@pytest.mark.parametrize(
    ("model_family", "operation_role"),
    [
        ("deepmd-tf", "student.deepmd.tf.neb"),
        ("deepmd-pt", "student.deepmd.pt.neb"),
        ("nep", "student.nep.neb"),
    ],
)
@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.parametrize("method", ["neb", "dyneb"])
@pytest.mark.parametrize("relax", [False, True])
def test_complete_fake_model_shape_protocol_matrix(
    tmp_path: Path,
    model_family: str,
    operation_role: str,
    members: int,
    method: str,
    relax: bool,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps

    suffix = ".txt" if model_family == "nep" else ".pb"
    models = tuple(tmp_path / f"model_{index:03d}{suffix}" for index in range(members))
    for model in models:
        model.write_text("fake-model", encoding="utf-8")
    endpoints = (
        Atoms(
            "HH",
            positions=[[0, 2, 0], [0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        Atoms(
            "HH",
            positions=[[0, 2, 0], [1, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
    )
    include_history = members == 2
    parameters = NebSamplingParameters(
        images=1,
        fmax_ev_a=1.0e6,
        max_steps=2,
        include_history=include_history,
        init="idpp" if relax else "linear",
        method=method,
        climb=method == "dyneb",
        fixed_atoms=(0,),
        relax=relax,
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "paths"),),
        condition_groups=(
            SamplingConditionGroup(
                "neb-a", "neb", ("hop",), neb=parameters
            ),
        ),
        round_count=1,
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("hop", endpoints),),
        models=OrderedSamplingModels(
            models,
            member_ids=tuple(f"student_{index:02d}" for index in range(members)),
        ),
        steps=None,
    )
    backend = AseNebSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family=model_family,
        operation_role=operation_role,
        operation_profile_id="test-neb",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    result = json.loads((task_dir / "neb_result.json").read_text(encoding="utf-8"))
    assert result["algorithm"]["band"] == (
        "DyNEB" if method == "dyneb" else "NEB"
    )
    assert ("endpoint_preparation" in result) is relax
    assert len(result["observations"]) == (1 if include_history else 3)
    assert len(result["observations"][0]["member_energies_ev"]) == members
    assert len(result["observations"][0]["member_forces_ev_a"]) == members
    assert backend.validate_current(evidence).complete is True
