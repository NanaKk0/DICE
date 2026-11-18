from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from workflow_backends.sampling import (
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingSeedSource,
)
from workflow_backends.sampling_models import OrderedSamplingModels
from workflow_core.common.sampling_plan import (
    ExplicitSamplingPlan,
    RelaxSamplingParameters,
    SamplingConditionGroup,
    SeedRole,
)
from workflow_core.structures.formats import read_extxyz_structures


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


class _ShiftedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, center: float) -> None:
        super().__init__()
        self.center = float(center)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        displacement = positions.copy()
        displacement[1, 0] -= self.center
        self.results = {
            "energy": float(np.sum(displacement**2)),
            "forces": -2.0 * displacement,
        }


class _CountingCommitteeProvider(_CalculatorProvider):
    def __init__(self) -> None:
        self.create_calls: list[str] = []

    def create(self, model_path: Path) -> Calculator:
        self.create_calls.append(model_path.name)
        center = 0.0 if "000" in model_path.name else 2.0
        return _ShiftedCalculator(center)


def _evidence(
    tmp_path: Path,
    *,
    parameters: RelaxSamplingParameters,
    members: int = 1,
) -> SamplingEvidenceRequest:
    models = tuple(tmp_path / f"model_{index:03d}.pb" for index in range(members))
    for model in models:
        model.write_text("fake-model", encoding="utf-8")
    seed = Atoms(
        "HH",
        positions=[[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("basin", "relax"),),
        condition_groups=(
            SamplingConditionGroup(
                "relax-path",
                "relax",
                ("basin",),
                relax=parameters,
            ),
        ),
        round_count=1,
    )
    return SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("basin", (seed,)),),
        models=OrderedSamplingModels(
            models,
            member_ids=tuple(f"student_{index:02d}" for index in range(members)),
        ),
        steps=None,
    )


def test_relax_sparse_replay_reads_only_requested_geometry_after_catalog(tmp_path, monkeypatch):
    import builtins
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps
    evidence = _evidence(tmp_path, parameters=RelaxSamplingParameters(fmax_ev_a=.05,max_steps=300))
    backend = AseRelaxSamplingNativeOps(calculator_provider=_CalculatorProvider(),
        model_family='deepmd-pt',operation_role='student.deepmd.pt.relax',operation_profile_id='test')
    backend.execute(SamplingExecutionRequest(evidence))
    backend.operation_profile_id = 'dormant-completed-prefix'
    assert backend.validate_current(evidence).complete
    source=backend.open_current(evidence).source
    rows=source.catalog()
    path=evidence.sample_dir/'tasks/task_000000/observations.extxyz'
    original_open, original_builtin_open, reads=Path.open, builtins.open, []
    class Counted:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def __getattr__(self,name):return getattr(self.stream,name)
        def read(self,*args):
            value=self.stream.read(*args);reads.append(len(value));return value
        def readline(self,*args):
            value=self.stream.readline(*args);reads.append(len(value));return value
        def __iter__(self):return self
        def __next__(self):
            line=self.readline()
            if not line:raise StopIteration
            return line
    def opened(p,*args,**kwargs):
        stream=original_open(p,*args,**kwargs)
        return Counted(stream) if p==path else stream
    monkeypatch.setattr(Path,'open',opened)
    def builtin_open(p,*args,**kwargs):
        stream=original_builtin_open(p,*args,**kwargs)
        return Counted(stream) if isinstance(p,(str,Path)) and Path(p)==path else stream
    monkeypatch.setattr(builtins,'open',builtin_open)
    frame,=next(source.replay((rows[-1].structure_id,),chunk_size=1))
    assert frame.info['structure_id']==rows[-1].structure_id
    assert sum(reads)<path.stat().st_size/4
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):
        tuple(source.replay((rows[-1].structure_id,),chunk_size=1))


@pytest.mark.parametrize('worker_death', [False, True])
def test_persistent_relax_executes_inventory_once_and_replays_in_global_order(tmp_path, monkeypatch, worker_death):
    from dataclasses import replace
    import sys
    from contextlib import contextmanager
    from workflow_backends.ase_relax.adapter import SubprocessAseRelaxSamplingNativeOps
    from workflow_backends.execution import ProcessGroupRegistry
    from workflow_backends.common.sampling_pool import DeviceSessionEnvironment
    from workflow_runtime.execution.allocation import Allocation
    from workflow_runtime.lifecycle.environment import OperationContext

    evidence = _evidence(tmp_path, parameters=RelaxSamplingParameters(
        fmax_ev_a=0.05, max_steps=300, fixed_atoms=(0,)))
    evidence = replace(evidence, plan=replace(evidence.plan, rounds=None, condition_groups=tuple(
        replace(evidence.plan.condition_groups[0], condition_group_id=f'group-{i}')
        for i in range(7))))
    # A real child interpreter with a deterministic stand-in at the native library boundary.
    native = tmp_path / 'deepmd'
    native.mkdir()
    (native / '__init__.py').write_text('')
    (native / 'calculator.py').write_text('''
import os, time
from tests.test_ase_relax_sampling import _HarmonicCalculator
class DP(_HarmonicCalculator):
    def __init__(self, model):
        super().__init__()
        self.calls = 0
        with open(os.environ['RELAX_LOAD_TRACE'], 'a') as f:
            f.write(str(os.getpid()) + '\\n')
    def calculate(self, *args, **kwargs):
        self.calls += 1
        if os.environ.get('RELAX_TEST_DEATH') and self.calls == 6:
            try:
                fd = os.open(os.environ['RELAX_LOAD_TRACE'] + '.death', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pass
            else:
                os.close(fd)
                os._exit(7)
        time.sleep(0.002)
        return super().calculate(*args, **kwargs)
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    monkeypatch.setenv('RELAX_LOAD_TRACE', str(tmp_path / 'loads'))
    if worker_death:
        monkeypatch.setenv('RELAX_TEST_DEATH', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-fixture')
    @contextmanager
    def session(*args):
        yield DeviceSessionEnvironment('GPU-fixture', '0', {}, ProcessGroupRegistry())
    monkeypatch.setattr('workflow_backends.ase_relax.pool.open_relax_session', session)
    context = OperationContext(role='student.deepmd.pt.relax', profile_id='fixture',
        source_scripts=(), executables={'python': sys.executable}, runtime={},
        allocation=Allocation(1, 1, 4, 1024, 'direct', 'pool', 2))
    backend = SubprocessAseRelaxSamplingNativeOps(context, model_family='deepmd-pt')
    execution = backend.execute(SamplingExecutionRequest(evidence))
    assert execution.failed_task_count == int(worker_death)
    assert len(set((tmp_path / 'loads').read_text().splitlines())) == 2 + int(worker_death)
    assert len((tmp_path / 'loads').read_text().splitlines()) == 2 + int(worker_death)
    catalog = backend.open_current(evidence).source.catalog()
    assert list(dict.fromkeys(row.task_index for row in catalog)) == list(range(7))
    assert sum((evidence.sample_dir / 'tasks' / f'task_{i:06d}' / 'final_structure.extxyz').is_file()
               for i in range(7)) == 7 - int(worker_death)


@pytest.mark.parametrize('history', [True, False])
def test_failed_relax_prefix_retained_with_history_eligibility(tmp_path, history):
    from dataclasses import replace
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    class Failing(_HarmonicCalculator):
        def calculate(self, atoms=None, *args, **kwargs):
            if atoms.info.get('fail') and atoms.positions[1, 0] < 0.9:
                raise RuntimeError('calculator stopped after saved steps')
            return super().calculate(atoms, *args, **kwargs)

    class Provider(_CalculatorProvider):
        def create(self, path):
            return Failing()

    evidence = _evidence(tmp_path, parameters=RelaxSamplingParameters(
        fmax_ev_a=.05, max_steps=300, include_history=history, fixed_atoms=(0,)))
    bad = evidence.systems[0].frames[0].copy()
    bad.info['fail'] = True
    evidence = replace(evidence, systems=(*evidence.systems, SamplingSeedSource('bad', (bad,))),
        plan=replace(evidence.plan, rounds=None, selection_plan=None, seeds=(*evidence.plan.seeds, SeedRole('bad', 'relax')),
            condition_groups=(replace(evidence.plan.condition_groups[0], sampling_role_ids=('bad', 'basin')),)))
    backend = AseRelaxSamplingNativeOps(calculator_provider=Provider(), model_family='deepmd-pt',
        operation_role='student.deepmd.pt.relax', operation_profile_id='test', retain_prefix=True)
    execution = backend.execute(SamplingExecutionRequest(evidence))
    assert execution.failed_task_count == 1
    tasks = evidence.plan.tasks_for_round(0)
    failed = next(task for task in tasks if task.sampling_role_id == 'bad')
    failed_dir = evidence.sample_dir / 'tasks' / failed.task_id
    result = json.loads((failed_dir / 'relax_result.json').read_text())
    assert result['termination'] == 'failed' and result['converged'] is False
    assert list((failed_dir / 'accepted').glob('step_*'))
    source = backend.open_current(evidence).source
    rows = source.catalog()
    assert any(row.sampling_role_id == 'basin' for row in rows)
    assert any(row.sampling_role_id == 'bad' for row in rows) == history
    frames = [f for chunk in source.replay(tuple(r.structure_id for r in rows), chunk_size=2) for f in chunk]
    assert all(f.info['relax_termination'] == 'failed' for f in frames if f.info['sampling_role_id'] == 'bad')


@pytest.mark.parametrize('accept', [False, True])
def test_persistent_exhaustion_cannot_bypass_accept_unconverged(tmp_path, accept):
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps
    evidence = _evidence(tmp_path, parameters=RelaxSamplingParameters(
        fmax_ev_a=1.e-12, max_steps=2, include_history=True, accept_unconverged=accept))
    backend = AseRelaxSamplingNativeOps(calculator_provider=_CalculatorProvider(),
        model_family='deepmd-tf', operation_role='student.deepmd.tf.relax',
        operation_profile_id='test', retain_prefix=True)
    if accept:
        backend.execute(SamplingExecutionRequest(evidence))
        assert backend.open_current(evidence).source.catalog()
    else:
        with pytest.raises(ValueError, match='no screen-visible observations'):
            backend.execute(SamplingExecutionRequest(evidence))
        assert not backend.validate_current(evidence).complete
    result = json.loads((evidence.sample_dir / 'tasks/task_000000/relax_result.json').read_text())
    assert result['termination'] == ('max_steps' if accept else 'failed')
    assert result['converged'] is False
    if not accept:
        assert result['observations'] == []


@pytest.mark.parametrize('history', [True, False])
def test_partial_committee_step_is_diagnostic_and_corruption_rejects_publication(tmp_path, history):
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    class LateFailure(_HarmonicCalculator):
        def calculate(self, atoms=None, *args, **kwargs):
            if atoms.positions[1, 0] < .99:
                raise RuntimeError('second member unavailable')
            super().calculate(atoms, *args, **kwargs)

    class Provider(_CalculatorProvider):
        def create(self, model):
            return LateFailure() if '001' in model.name else _HarmonicCalculator()

    evidence = _evidence(tmp_path, parameters=RelaxSamplingParameters(
        fmax_ev_a=.01, max_steps=30, include_history=history), members=2)
    backend = AseRelaxSamplingNativeOps(calculator_provider=Provider(), model_family='deepmd-pt',
        operation_role='student.deepmd.pt.relax', operation_profile_id='test', retain_prefix=True)
    if history:
        backend.execute(SamplingExecutionRequest(evidence))
        assert [r.optimizer_step for r in backend.open_current(evidence).source.catalog()] == [0]
    else:
        with pytest.raises(ValueError, match='no screen-visible observations'):
            backend.execute(SamplingExecutionRequest(evidence))
        assert not backend.validate_current(evidence).complete
    directory = evidence.sample_dir / 'tasks/task_000000'
    assert (directory / 'accepted/step_00000001/frame.extxyz').is_file()
    assert not (directory / 'accepted/step_00000001/evidence.json').exists()
    (directory / 'accepted/step_00000000/evidence.json').write_text('{}')
    assert not backend.validate_current(evidence).complete


def test_pure_relax_sampling_composition_requires_no_native_recipe(
    tmp_path: Path,
) -> None:
    from workflow_app.composition import sampling_backend_binding

    config = SimpleNamespace(
        root=tmp_path,
        sampler={"backend": "LAMMPS"},
        student={"type_map": ["H"]},
        raw={
            "sample": {
                "condition_groups": {
                    "relax-path": {
                        "kind": "relax",
                        "seeds": ["basin"],
                        "fmax_ev_a": 0.02,
                        "max_steps": 100,
                    }
                }
            }
        },
    )

    binding = sampling_backend_binding(config)

    assert binding.recipes == ()


def test_fixed_cell_fire_relaxation_publishes_strided_path_and_terminal(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    evidence = _evidence(
        tmp_path,
        parameters=RelaxSamplingParameters(
            fmax_ev_a=0.05,
            max_steps=500,
            max_step_a=0.1,
            include_history=True,
            history_stride=3,
            fixed_atoms=(0,),
        ),
    )
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    source = backend.open_current(evidence).source
    catalog = source.catalog()
    frames = tuple(frame for chunk in source.replay(
        tuple(row.structure_id for row in catalog), chunk_size=2
    ) for frame in chunk)
    task_result = json.loads(
        (evidence.sample_dir / "tasks" / "task_000000" / "relax_result.json")
        .read_text(encoding="utf-8")
    )

    steps = [row.optimizer_step for row in catalog]
    assert result.task_count == 1
    assert task_result["converged"] is True
    assert steps[0] == 0
    assert steps[-1] == task_result["optimizer_steps"]
    assert all(step % 3 == 0 for step in steps[1:-1])
    assert len(steps) == len(set(steps))
    assert all(np.array_equal(frame.cell.array, frames[0].cell.array) for frame in frames)
    assert all(np.array_equal(frame.positions[0], frames[0].positions[0]) for frame in frames)
    assert all(row.locator.backend == "ASE-RELAX" for row in catalog)
    assert all(row.condition_group_id == "relax-path" for row in catalog)
    assert backend.validate_current(evidence).complete is True


def test_committee_uses_first_member_path_and_caches_calculators_across_tasks(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    models = (tmp_path / "model_000.pb", tmp_path / "model_001.pb")
    for model in models:
        model.write_text("fake-model", encoding="utf-8")
    seeds = (
        SeedRole("basin-a", "relax"),
        SeedRole("basin-b", "relax"),
    )
    parameters = RelaxSamplingParameters(
        fmax_ev_a=0.05,
        max_steps=500,
        include_history=False,
        fixed_atoms=(0,),
    )
    plan = ExplicitSamplingPlan(
        seeds=seeds,
        condition_groups=(
            SamplingConditionGroup(
                "relax-path",
                "relax",
                tuple(seed.sampling_role_id for seed in seeds),
                relax=parameters,
            ),
        ),
        round_count=1,
    )
    systems = tuple(
        SamplingSeedSource(
            seed.sampling_role_id,
            (
                Atoms(
                    "HH",
                    positions=[[0.0, 2.0, 0.0], [start, 0.0, 0.0]],
                    cell=[6.0, 6.0, 6.0],
                    pbc=True,
                ),
            ),
        )
        for seed, start in zip(seeds, (1.0, -1.0), strict=True)
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=systems,
        models=OrderedSamplingModels(
            models, member_ids=("student_00", "student_01")
        ),
        steps=None,
    )
    provider = _CountingCommitteeProvider()
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=provider,
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    assert provider.create_calls == ["model_000.pb", "model_001.pb"]
    for task_index in range(2):
        task_dir = evidence.sample_dir / "tasks" / f"task_{task_index:06d}"
        result = json.loads((task_dir / "relax_result.json").read_text(encoding="utf-8"))
        final = tuple(backend.open_current(evidence).source.replay(
            (result["observations"][0]["structure_id"],), chunk_size=1
        ))[0][0]
        assert abs(float(final.positions[1, 0])) < 0.03
        assert len(result["observations"][0]["member_energies_ev"]) == 2
        assert len(result["observations"][0]["member_forces_ev_a"]) == 2
        assert result["observations"][0]["force_deviation"] > 0.0


def test_lifecycle_sampling_intent_accepts_relax_without_md_steps() -> None:
    from workflow_core.lifecycle.sampling import SamplingTaskIntent

    parameters = RelaxSamplingParameters(fmax_ev_a=0.02, max_steps=100)

    task = SamplingTaskIntent(
        task_id="task_000000",
        system_id="basin",
        ensemble="relax",
        temperature_k=None,
        condition_group_id="relax-path",
        sampling_role_id="basin",
        selection_group_id="basins",
        kind="relax",
        steps=None,
        source_frame_index=0,
        relax=parameters,
    )

    assert task.relax is parameters


def test_relax_task_is_accepted_by_canonical_sample_artifact_validator(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps
    from workflow_backends.sample_recovery import validate_sample_artifacts
    from workflow_core.lifecycle.sampling import SampleRecoveryFacts, SamplingTaskIntent

    parameters = RelaxSamplingParameters(
        fmax_ev_a=0.05,
        max_steps=500,
        include_history=True,
        history_stride=2,
        fixed_atoms=(0,),
    )
    evidence = _evidence(tmp_path, parameters=parameters)
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )
    backend.execute(SamplingExecutionRequest(evidence))
    task = evidence.plan.tasks_for_round(0)[0]
    facts = SampleRecoveryFacts(
        backend="LAMMPS",
        cli="tf",
        member_ids=("student_00",),
        type_map=("H",),
        system_ids=("basin",),
        tasks=(
            SamplingTaskIntent(
                task_id=task.task_id,
                system_id=task.system_id,
                ensemble=task.ensemble,
                temperature_k=None,
                condition_group_id=task.condition_group_id,
                sampling_role_id=task.sampling_role_id,
                selection_group_id=task.selection_group_id,
                kind=task.kind,
                steps=None,
                source_frame_index=task.source_frame_index,
                relax=task.relax,
            ),
        ),
        steps=None,
        plumed_enabled=False,
        compress=False,
        iter_index=0,
        round_index=0,
    )

    assert validate_sample_artifacts(evidence.sample_dir, facts) is True


def test_mixed_neb_and_relax_round_preserves_global_task_order(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps
    from workflow_backends.hybrid_sampling import HybridSamplingNativeOps
    from workflow_core.common.sampling_plan import NebSamplingParameters

    model = tmp_path / "model.pb"
    model.write_text("fake-model", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("basin", "joint"), SeedRole("hop", "joint")),
        condition_groups=(
            SamplingConditionGroup(
                "relax-first",
                "relax",
                ("basin",),
                relax=RelaxSamplingParameters(1.0e6, 2),
            ),
            SamplingConditionGroup(
                "neb-second",
                "neb",
                ("hop",),
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
                "basin",
                (Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),),
            ),
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
    provider = _CalculatorProvider()
    backend = HybridSamplingNativeOps(
        neb=AseNebSamplingNativeOps(
            calculator_provider=provider,
            model_family="deepmd-tf",
            operation_role="student.deepmd.tf.neb",
            operation_profile_id="test-neb",
            allow_empty_observations=True,
        ),
        relax=AseRelaxSamplingNativeOps(
            calculator_provider=provider,
            model_family="deepmd-tf",
            operation_role="student.deepmd.tf.relax",
            operation_profile_id="test-relax",
            allow_empty_observations=True,
        ),
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    catalog = backend.open_current(evidence).source.catalog()

    assert result.task_count == 2
    assert list(dict.fromkeys(row.task_id for row in catalog)) == [
        "task_000000",
        "task_000001",
    ]
    assert list(dict.fromkeys(row.locator.backend for row in catalog)) == [
        "ASE-RELAX",
        "ASE-NEB",
    ]


def test_relax_progress_reports_the_real_ase_backend(tmp_path: Path) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    evidence = _evidence(
        tmp_path,
        parameters=RelaxSamplingParameters(fmax_ev_a=1.0e6, max_steps=2),
    )
    snapshots = []
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )

    backend.execute(
        SamplingExecutionRequest(evidence, progress_callback=snapshots.append)
    )

    assert snapshots
    assert snapshots[-1].backend == "ASE"
    assert snapshots[-1].ensemble == "relax"
    assert snapshots[-1].completed_tasks == snapshots[-1].total_tasks == 1


def test_runtime_sampling_intent_preserves_relax_semantics_and_inventory(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from workflow_core.lifecycle.artifact_handles import ModelHandle
    from workflow_core.lifecycle.runner import _sampling_evidence_intent

    parameters = RelaxSamplingParameters(
        fmax_ev_a=0.02,
        max_steps=400,
        max_step_a=0.1,
        include_history=True,
        history_stride=5,
        fixed_atoms=(0,),
    )
    evidence = _evidence(tmp_path, parameters=parameters)

    intent = _sampling_evidence_intent(
        SimpleNamespace(iter_index=0, sampler={"backend": "LAMMPS"}),
        0,
        evidence,
        (ModelHandle("student", "model-id"),),
    )

    assert intent.tasks[0].relax is parameters
    assert intent.evidence_inventory == (
        "seeds",
        "task-contexts",
        "relax-evidence",
    )


def test_sampling_summary_counts_relax_steps_and_reports_mixed_kinds() -> None:
    from types import SimpleNamespace

    from workflow_core.lifecycle.runner import _sampling_block_summary

    parameters = RelaxSamplingParameters(fmax_ev_a=0.02, max_steps=400)
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("thermal", "p40"), SeedRole("basin", "p40")),
        condition_groups=(
            SamplingConditionGroup(
                "md",
                "nvt",
                ("thermal",),
                steps_by_round=(10,),
                temperatures_k=(300.0, 500.0),
            ),
            SamplingConditionGroup(
                "relax",
                "relax",
                ("basin",),
                relax=parameters,
            ),
        ),
        round_count=1,
    )
    summary = _sampling_block_summary(
        SimpleNamespace(
            sampler={"backend": "LAMMPS"},
            student={"backend": "DeepMD", "cli": "pt"},
        ),
        plan=plan,
        round_index=0,
        system_count=2,
        model_count=1,
        elapsed_seconds=1.5,
    )

    assert summary.backend == "LAMMPS+ASE"
    assert summary.total_steps == 420
    assert summary.kind_counts == (("nvt", 2), ("relax", 1))


def test_nonconverged_relax_task_never_publishes_a_task_receipt(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    evidence = _evidence(
        tmp_path,
        parameters=RelaxSamplingParameters(
            fmax_ev_a=1.0e-12,
            max_steps=1,
            include_history=True,
            fixed_atoms=(0,),
        ),
    )
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )

    with pytest.raises(
        ValueError,
        match="exhausted max_steps without satisfying fmax_ev_a",
    ):
        backend.execute(SamplingExecutionRequest(evidence))

    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    assert not task_dir.exists()
    assert not (task_dir / "task_completion_receipt.json").exists()
    assert backend.validate_current(evidence).complete is False


def test_explicitly_accepted_nonconverged_relax_publishes_history_and_status(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps

    evidence = _evidence(
        tmp_path,
        parameters=RelaxSamplingParameters(
            fmax_ev_a=1.0e-12,
            max_steps=1,
            include_history=True,
            history_stride=1,
            accept_unconverged=True,
            fixed_atoms=(0,),
        ),
    )
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )

    backend.execute(SamplingExecutionRequest(evidence))

    task_dir = evidence.sample_dir / "tasks" / "task_000000"
    result = json.loads((task_dir / "relax_result.json").read_text(encoding="utf-8"))
    catalog = backend.open_current(evidence).source.catalog()
    frames = read_extxyz_structures(task_dir / "observations.extxyz", allow_empty=False)
    assert result["converged"] is False
    assert result["termination"] == "max_steps"
    assert result["optimizer_steps"] == 1
    assert result["final_max_force_ev_a"] > 1.0e-12
    assert [row.optimizer_step for row in catalog] == [0, 1]
    assert all(frame.info["relax_converged"] is False for frame in frames)
    assert all(frame.info["relax_termination"] == "max_steps" for frame in frames)
    assert (task_dir / "trajectory.extxyz").is_file()
    assert (task_dir / "task_completion_receipt.json").is_file()
    assert backend.validate_current(evidence).complete is True

    from workflow_backends.sample_recovery import validate_sample_artifacts
    from workflow_core.lifecycle.sampling import SampleRecoveryFacts, SamplingTaskIntent

    task = evidence.plan.tasks_for_round(0)[0]
    facts = SampleRecoveryFacts(
        backend="LAMMPS",
        cli="tf",
        member_ids=("student_00",),
        type_map=("H",),
        system_ids=("basin",),
        tasks=(
            SamplingTaskIntent(
                task_id=task.task_id,
                system_id=task.system_id,
                ensemble=task.ensemble,
                temperature_k=None,
                condition_group_id=task.condition_group_id,
                sampling_role_id=task.sampling_role_id,
                selection_group_id=task.selection_group_id,
                kind=task.kind,
                steps=None,
                source_frame_index=task.source_frame_index,
                relax=task.relax,
            ),
        ),
        steps=None,
        plumed_enabled=False,
        compress=False,
        iter_index=0,
        round_index=0,
    )
    assert validate_sample_artifacts(evidence.sample_dir, facts) is True


def test_relax_artifact_corruption_is_rejected_by_native_and_lifecycle_validation(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_relax.adapter import AseRelaxSamplingNativeOps
    from workflow_backends.sample_recovery import validate_sample_artifacts
    from workflow_core.lifecycle.sampling import SampleRecoveryFacts, SamplingTaskIntent

    parameters = RelaxSamplingParameters(fmax_ev_a=1.0e6, max_steps=2)
    evidence = _evidence(tmp_path, parameters=parameters)
    backend = AseRelaxSamplingNativeOps(
        calculator_provider=_CalculatorProvider(),
        model_family="deepmd-tf",
        operation_role="student.deepmd.tf.relax",
        operation_profile_id="test-relax",
    )
    backend.execute(SamplingExecutionRequest(evidence))
    task = evidence.plan.tasks_for_round(0)[0]
    facts = SampleRecoveryFacts(
        backend="LAMMPS",
        cli="tf",
        member_ids=("student_00",),
        type_map=("H",),
        system_ids=("basin",),
        tasks=(
            SamplingTaskIntent(
                task_id=task.task_id,
                system_id=task.system_id,
                ensemble=task.ensemble,
                temperature_k=None,
                condition_group_id=task.condition_group_id,
                sampling_role_id=task.sampling_role_id,
                selection_group_id=task.selection_group_id,
                kind=task.kind,
                steps=None,
                source_frame_index=task.source_frame_index,
                relax=task.relax,
            ),
        ),
        steps=None,
        plumed_enabled=False,
        compress=False,
        iter_index=0,
        round_index=0,
    )
    observations = evidence.sample_dir / "tasks" / "task_000000" / "observations.extxyz"
    observations.write_bytes(observations.read_bytes() + b"\n")

    assert backend.validate_current(evidence).complete is False
    assert validate_sample_artifacts(evidence.sample_dir, facts) is False


def test_relax_worker_preserves_distinct_task_ids_for_one_group_with_many_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workflow_backends.ase_relax import worker
    from workflow_core.structures.formats import write_extxyz_structures

    model = tmp_path / "model.pb"
    model.write_text("fake-model", encoding="utf-8")
    seeds = []
    groups = []
    for index, x in enumerate((0.5, 1.0)):
        role = f"basin-{index}"
        seed_path = tmp_path / f"{role}.extxyz"
        write_extxyz_structures(
            seed_path,
            (
                Atoms(
                    "H",
                    positions=[[x, 0.0, 0.0]],
                    cell=[5.0, 5.0, 5.0],
                    pbc=True,
                ),
            ),
        )
        seeds.append(
            {
                "sampling_role_id": role,
                "selection_group_id": "basins",
                "path": str(seed_path),
            }
        )
        groups.append(
            {
                "condition_group_id": "relax-all",
                "task_id": f"task_{index:06d}",
                "sampling_role_id": role,
                "source_frame_index": index + 4,
                "fmax_ev_a": 1.0e6,
                "max_steps": 2,
                "max_step_a": None,
                "include_history": True,
                "history_stride": 1,
                "fixed_atoms": [],
            }
        )
    manifest = tmp_path / "request.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": worker.REQUEST_SCHEMA,
                "schema_revision": worker.REQUEST_REVISION,
                "run_json_path": str(tmp_path / "run.json"),
                "iter": 0,
                "round": 0,
                "sample_dir": str(tmp_path / "sample"),
                "model_family": "deepmd-tf",
                "operation_role": "student.deepmd.tf.relax",
                "operation_profile_id": "test-relax",
                "allow_empty_observations": False,
                "max_workers": 1,
                "models": [{"member_id": "student", "path": str(model)}],
                "seeds": seeds,
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker, "_DeepMDCalculatorProvider", _CalculatorProvider)

    worker.execute_manifest(manifest)

    task_dirs = sorted((tmp_path / "sample" / "tasks").glob("task_*"))
    assert [path.name for path in task_dirs] == ["task_000000", "task_000001"]
    assert [
        json.loads((path / "task_context.json").read_text(encoding="utf-8"))[
            "source_frame_index"
        ]
        for path in task_dirs
    ] == [4, 5]
