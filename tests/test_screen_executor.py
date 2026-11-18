from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from types import SimpleNamespace

from ase import Atoms
from workflow_artifacts.lifecycle.screen_publication import (
    ScreenArtifactPublicationCapability,
)
import numpy as np
import pytest

from workflow_core.structures.formats import read_extxyz_structures


def test_screen_round_inputs_use_ordered_selection_groups_instead_of_roles() -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan
    from workflow_core.lifecycle.screen_module import _round_inputs

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    selection_plan = SelectionPlan.from_seeds(
        (
            SeedRole("role-a", "joint"),
            SeedRole("role-b", "joint"),
            SeedRole("role-c", "solo"),
        )
    )
    config = SimpleNamespace(
        selection_plan=selection_plan,
        sampling_plan=SimpleNamespace(explicit=True),
        path="/controlled/run.json",
        iter_index=0,
        teacher={"lo": 0.1, "hi": 0.2},
        raw={
            "sample": {"seeds": {"role-a": {}, "role-b": {}, "role-c": {}}},
            "teacher": {"lo": 0.1, "hi": 0.2},
            "screen": {
                "policy": "DIRECT",
                "pool": 4,
                "keep": 2,
                "fp_pool": 1,
                "teacher_p": {"floor": 0.0, "trust": 0.5},
                "student_p": {"floor": 0.0},
                "fill_underfilled_from_safe_region": False,
            },
            "fp": {"policy": "DIRECT"},
        },
    )

    inputs = _round_inputs(config, route, 0)

    assert inputs.system_ids == ("joint", "solo")
    assert inputs.selection_plan is selection_plan
    assert inputs.require_explicit_provenance is True


def test_explicit_screen_rejects_role_group_conflict_before_route_work(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    selection_plan = SelectionPlan.from_seeds(
        (SeedRole("role-a", "joint"), SeedRole("role-b", "joint"))
    )
    record = SimpleNamespace(
        system_id="wrong",
        structure_id="candidate-1",
        md_step=20,
        student_uncertainty=0.0,
        sampling_role_id="role-a",
        selection_group_id="wrong",
        condition_group_id="fixed",
        task_id="task_000000",
    )

    class Source:
        def catalog(self):
            return (record,)

        def replay(self, identities, *, chunk_size: int):
            raise AssertionError("invalid provenance reached replay")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("invalid provenance reached route work")

    with pytest.raises(ValueError, match="selection plan"):
        execute_screen(
            ScreenExecutionRequest(
                route=route,
                round_index=0,
                stage=tmp_path,
                source=Source(),
                inputs=ScreenRoundInputs(
                    system_ids=selection_plan.group_ids,
                    selection_plan=selection_plan,
                    require_explicit_provenance=True,
                    pair_guard=None,
                    screen_policy="random",
                    fp_policy="random",
                    pool=1,
                    keep=1,
                    fp_pool=1,
                    teacher_p={"floor": 0.0, "trust": 0.1},
                    student_p={"floor": 0.0},
                    entropy="test/iter0/r0",
                ),
                services=ScreenServices(unexpected, unexpected, unexpected),
                report=_Reporter(),
                artifacts=ScreenArtifactPublicationCapability(),
            )
        )


def test_explicit_screen_publishes_ordered_zero_and_underfilled_group_rows(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    plan = SelectionPlan.from_seeds(
        (SeedRole("role-a", "joint"), SeedRole("role-b", "joint"), SeedRole("role-c", "solo"))
    )
    record = SimpleNamespace(
        system_id="joint",
        structure_id="joint-rejected",
        md_step=20,
        student_uncertainty=0.0,
        sampling_role_id="role-a",
        selection_group_id="joint",
        condition_group_id="fixed",
        task_id="task_000000",
    )
    frame = Atoms("H2", positions=[[0, 0, 0], [0.5, 0, 0]])
    frame.info.update(
        system_id="joint",
        structure_id=record.structure_id,
        sampling_role_id="role-a",
        selection_group_id="joint",
        condition_group_id="fixed",
    )

    class Source:
        def catalog(self):
            return (record,)

        def replay(self, identities, *, chunk_size: int):
            assert tuple(identities) == (record.structure_id,)
            yield (frame.copy(),)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("all-reject grouped screen invoked route work")

    execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=plan.group_ids,
                selection_plan=plan,
                require_explicit_provenance=True,
                pair_guard={"min": 0.6, "pairs": {}},
                screen_policy="DIRECT",
                fp_policy="DIRECT",
                pool=4,
                keep=2,
                fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p={"floor": 0.0},
                entropy="test/iter0/r0",
                fill_underfilled_from_safe_region=True,
            ),
            services=ScreenServices(unexpected, unexpected, unexpected),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    audit = json.loads(
        (tmp_path / "selection_group_outcomes.json").read_text(encoding="utf-8")
    )
    assert audit["selection_groups"] == ["joint", "solo"]
    assert audit["role_to_group"] == [
        ["role-a", "joint"],
        ["role-b", "joint"],
        ["role-c", "solo"],
    ]
    assert [row["selection_group_id"] for row in audit["rows"]] == ["joint", "solo"]
    assert all(row["final_selected_count"] == 0 for row in audit["rows"])
    assert all(row["underfill"] == 2 for row in audit["rows"])
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert all(
        [row["system_id"] for row in role["systems"]] == ["joint", "solo"]
        for role in report["roles"]
    )
    from workflow_artifacts.lifecycle.screen_validation import (
        ScreenArtifactValidationRequest,
        validate_screen_artifacts,
    )

    validation = ScreenArtifactValidationRequest(
        contract=route.screen_artifact_contract,
        screen_dir=tmp_path,
        screen_policy="DIRECT",
        fp_policy="DIRECT",
        pair_guard={"min": 0.6, "pairs": {}},
        expected_system_ids=plan.group_ids,
        expected_type_map=("H",),
        selection_plan=plan,
        require_selection_group_audit=True,
    )
    assert validate_screen_artifacts(validation)
    audit["rows"][0]["final_selected_count"] = 1
    (tmp_path / "selection_group_outcomes.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    assert not validate_screen_artifacts(validation)


@dataclass(frozen=True)
class _CatalogRecord:
    system_id: str
    structure_id: str
    md_step: int
    student_uncertainty: float


class _ObservationSource:
    def __init__(self, record: _CatalogRecord, frame: Atoms) -> None:
        self._record = record
        self._frame = frame

    def catalog(self):
        # The screen executor owns canonical first-seen deduplication.
        return (self._record, self._record)

    def replay(self, identities, *, chunk_size: int):
        assert chunk_size > 0
        assert tuple(identities) == (self._record.structure_id,)
        yield [self._frame.copy()]


class _Reporter:
    def phase(self, _name: str) -> None:
        pass

    def snapshot(self, _snapshot) -> None:
        pass

    def summary(self, _summary) -> None:
        pass

    def warning(self, exc: Exception) -> None:
        raise AssertionError(f"screen report warning: {exc}")


@pytest.mark.parametrize("prescreen", ["JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"])
@pytest.mark.parametrize('uncertainty', [0.0, None])
@pytest.mark.parametrize('empty_group', [False, True])
def test_joint_direct_single_student_uses_structural_readout_without_gmm(
    tmp_path: Path,
    prescreen: str,
    uncertainty,
    monkeypatch,
    empty_group,
) -> None:
    from workflow_core.lifecycle import visualization as visualization_core
    element_rows = visualization_core.element_rows_from_element_vectors
    streamed_visualization = []
    def checked_rows(frames, *args, **kwargs):
        if not isinstance(frames, (list, tuple)):
            streamed_visualization.append(True)
        return element_rows(frames, *args, **kwargs)
    monkeypatch.setattr(visualization_core, 'element_rows_from_element_vectors', checked_rows)
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        JointVectorReadout,
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    from workflow_core.structures.canonical_identity import stamp_canonical_identity

    frames = []
    for index in range(3):
        frame = Atoms(
            "H2", positions=[[0.0, 0.0, 0.0], [0.8 + 0.1 * index, 0.0, 0.0]]
        )
        frames.append(
            stamp_canonical_identity(
                frame, system_id="water", structure_source="workflow"
            )
        )
    candidate_ids = tuple(str(frame.info["structure_id"]) for frame in frames)
    train_frame = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    train_frame.info.update(structure_id="train-0", system_id="water")
    from workflow_core.structures.formats import write_extxyz_structures

    student_train_path = write_extxyz_structures(
        tmp_path / "student_train.xyz", [train_frame]
    )
    background_id = str(
        read_extxyz_structures(student_train_path, allow_empty=False)[0].info[
            "structure_id"
        ]
    )
    records = tuple(
        _CatalogRecord("water", identity, index, uncertainty)
        for index, identity in enumerate(candidate_ids)
    )
    plan = None
    if empty_group:
        from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan
        plan = SelectionPlan.from_seeds((SeedRole('role-water', 'water'), SeedRole('role-empty', 'empty')))
        records = tuple(SimpleNamespace(**vars(record), sampling_role_id='role-water',
            selection_group_id='water', condition_group_id='fixed', task_id='task_000000') for record in records)
        for frame in frames:
            frame.info.update(sampling_role_id='role-water', selection_group_id='water', condition_group_id='fixed')

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    calls = []

    def unexpected_gmm(*_args, **_kwargs):
        raise AssertionError("JOINT_DIRECT invoked student GMM")

    def read_vectors(*, frames, **_kwargs):
        calls.append(tuple(frame.info["structure_id"] for frame in frames))
        return JointVectorReadout(
            candidate_vectors={
                candidate_ids[0]: np.asarray([1.0, -0.2]),
                candidate_ids[1]: np.asarray([1.0, 0.2]),
                candidate_ids[2]: np.asarray([0.0, 1.0]),
            },
            background_vectors={background_id: np.asarray([1.0, 0.0])},
            candidate_element_vectors={
                "H": {
                    candidate_ids[0]: np.asarray([11.0, -0.2]),
                    candidate_ids[1]: np.asarray([11.0, 0.2]),
                    candidate_ids[2]: np.asarray([10.0, 1.0]),
                }
            },
            background_element_vectors={
                "H": {background_id: np.asarray([11.0, 0.0])}
            },
            descriptor_schema={"name": "test", "revision": 1, "width": 2},
            model_identity="student",
            dataset_identity="iter0/r0/train/dataset",
        )

    def label_teacher(*, frames, **_kwargs):
        from ase.calculators.singlepoint import SinglePointCalculator

        result = []
        for frame in frames:
            item = frame.copy()
            item.info["force_uncertainty"] = 0.05
            item.calc = SinglePointCalculator(
                item, energy=0.0, forces=np.zeros((len(item), 3))
            )
            result.append(item)
        return result

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=plan.group_ids if plan else ("water",),
                selection_plan=plan,
                require_explicit_provenance=empty_group,
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=2,
                keep=1,
                fp_pool=None,
                teacher_p=None,
                student_p=None,
                teacher_force_lo=0.1,
                teacher_force_hi=0.2,
                entropy="test/iter0/r0",
                fill_underfilled_from_safe_region=False,
                prescreen=prescreen,
            ),
            services=ScreenServices(
                score_teacher=None,
                score_student=unexpected_gmm,
                label_teacher=label_teacher,
                read_student_vectors=read_vectors,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            student_train_path=student_train_path,
            student_type_map=("H",),
        )
    )

    assert calls == [candidate_ids]
    if empty_group:
        rows = json.loads((tmp_path/'selection_group_outcomes.json').read_text())['rows']
        assert rows[1]['selection_group_id'] == 'empty'
        assert rows[1]['candidate_count'] == rows[1]['final_selected_count'] == 0
        assert rows[1]['underfill'] == 1
    assert [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    ] == [candidate_ids[2]]
    assert not (tmp_path / "gmm_router").exists()
    assert not (tmp_path / "gmm_scores.npz").exists()
    evidence_name = (
        "element_direct_screen.npz"
        if prescreen == "ELEMENT_JOINT_DIRECT"
        else "joint_direct_screen.npz"
    )
    assert (tmp_path / evidence_name).is_file()
    import pickle
    from workflow_core.lifecycle.visualization import visualization_from_draft

    assert not (tmp_path / ".visualize_cache").exists()
    assert result.visualization_draft is not None
    visualization = visualization_from_draft(
        pickle.loads(pickle.dumps(result.visualization_draft)), (candidate_ids[2],)
    )
    assert streamed_visualization == []
    assert [layer.name for layer in visualization.layers] == [
        "sample", "pool", "train", "select"
    ]
    assert visualization.layers[0].frame_ids == candidate_ids
    assert visualization.layers[1].frame_ids == candidate_ids[1:]
    assert visualization.layers[2].frame_ids == (background_id,)
    assert visualization.layers[3].frame_ids == (candidate_ids[2],)
    np.testing.assert_array_equal(
        visualization.elements["H"].vectors,
        np.asarray(
            [[11.0, -0.2], [11.0, 0.2], [10.0, 1.0], [11.0, 0.0]],
            dtype=np.float32,
        ),
    )
    from workflow_artifacts.lifecycle.screen_validation import (
        ScreenArtifactValidationRequest,
        validate_screen_artifacts,
    )
    from workflow_policies.identities import selector_identity

    validation = ScreenArtifactValidationRequest(
        contract=route.screen_artifact_contract,
        screen_dir=tmp_path,
        screen_policy="random",
        fp_policy="random",
        pair_guard=None,
        expected_system_ids=plan.group_ids if plan else ("water",),
        selection_plan=plan,
        require_selection_group_audit=empty_group,
            expected_type_map=("H",),
            expected_thresholds={
                "prescreen": selector_identity(prescreen).as_dict(),
                "teacher_force_deviation": {"lo": 0.1, "hi": 0.2},
                "quota": {"pool": 2, "keep": 1},
                "fill_underfilled_from_safe_region": False,
                "selector": selector_identity("random").as_dict(),
            },
            prescreen=prescreen,
            joint_model_identity="student",
            joint_dataset_identity="iter0/r0/train/dataset",
        )
    assert validate_screen_artifacts(validation)
    if empty_group:
        summary_path = tmp_path/'selection_group_outcomes.json'
        original_summary = summary_path.read_text()
        malformed = json.loads(original_summary)
        malformed['rows'][1]['normal_selected_count'] = 1
        summary_path.write_text(json.dumps(malformed))
        assert not validate_screen_artifacts(validation)
        summary_path.write_text(original_summary)
        with np.load(tmp_path/evidence_name, allow_pickle=False) as archive:
            original_evidence = {name: np.array(archive[name]) for name in archive.files}
        malformed_evidence = dict(original_evidence)
        outcomes = json.loads(str(malformed_evidence['outcomes_json'].item()))
        del outcomes['water']
        malformed_evidence['outcomes_json'] = np.asarray(json.dumps(outcomes))
        np.savez_compressed(tmp_path/evidence_name, **malformed_evidence)
        assert not validate_screen_artifacts(validation)
        np.savez_compressed(tmp_path/evidence_name, **original_evidence)
        assert validate_screen_artifacts(validation)
    if prescreen == "ELEMENT_JOINT_DIRECT":
        with np.load(tmp_path / evidence_name, allow_pickle=False) as archive:
            corrupted = {name: np.array(archive[name]) for name in archive.files}
        corrupted["candidate_element_0000"][0, 0] = np.inf
        np.savez_compressed(tmp_path / evidence_name, **corrupted)
        assert not validate_screen_artifacts(validation)


def test_element_joint_screen_use_case_selects_per_group_with_full_background() -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_use_case import ScreenRoundUseCase
    from workflow_policies.selection import select_element_joint_representatives

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    frames = []
    for identity, system_id, symbols in (
        ("a0", "a", "H2O"),
        ("a1", "a", "H"),
        ("a2", "a", "O"),
        ("b0", "b", "H"),
        ("b1", "b", "O"),
    ):
        frame = Atoms(symbols)
        frame.info.update(structure_id=identity, system_id=system_id)
        frames.append(frame)
    background = []
    for identity, symbols in (("train-h", "H2"), ("train-o", "O")):
        frame = Atoms(symbols)
        frame.info.update(structure_id=identity, system_id="train")
        background.append(frame)
    candidate_elements = {
        "H": {
            "a0": np.asarray([1.0, 0.0]),
            "a1": np.asarray([0.9, 0.1]),
            "b0": np.asarray([-1.0, 0.0]),
        },
        "O": {
            "a0": np.asarray([0.0, 1.0]),
            "a2": np.asarray([0.1, 0.9]),
            "b1": np.asarray([0.0, -1.0]),
        },
    }
    background_elements = {
        "H": {"train-h": np.asarray([1.0, 0.0])},
        "O": {"train-o": np.asarray([0.0, 1.0])},
    }

    selected, outcomes = ScreenRoundUseCase(route).select_student_element_joint_proposal(
        frames,
        background_frames=background,
        pool=2,
        candidate_element_vectors=candidate_elements,
        background_element_vectors=background_elements,
        type_map=("H", "O"),
    )

    expected_a = select_element_joint_representatives(
        ("a0", "a1", "a2"),
        (
            {"H": np.asarray([[1.0, 0.0]]), "O": np.asarray([[0.0, 1.0]])},
            {"H": np.asarray([[0.9, 0.1]])},
            {"O": np.asarray([[0.1, 0.9]])},
        ),
        candidate_presence=(("H", "O"), ("H",), ("O",)),
        background_identities=("train-h", "train-o"),
        background_element_rows=(
            {"H": np.asarray([[1.0, 0.0]])},
            {"O": np.asarray([[0.0, 1.0]])},
        ),
        background_presence=(("H",), ("O",)),
        type_map=("H", "O"),
        keep=2,
    )
    assert tuple(frame.info["structure_id"] for frame in selected) == (
        *expected_a.selected_identities,
        "b0",
        "b1",
    )
    assert outcomes["a"]["background_count"] == 2
    assert outcomes["a"]["active_elements"] == ("H", "O")
    assert outcomes["b"]["bypass"] is True


def test_element_direct_screen_use_case_selects_candidates_without_background() -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_use_case import ScreenRoundUseCase

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    frames = []
    for identity, system_id, symbols in (
        ("a0", "a", "H2O"),
        ("a1", "a", "H"),
        ("a2", "a", "O"),
        ("b0", "b", "H"),
    ):
        frame = Atoms(symbols)
        frame.info.update(structure_id=identity, system_id=system_id)
        frames.append(frame)
    element_vectors = {
        "H": {
            "a0": np.asarray([1.0, 0.0]),
            "a1": np.asarray([0.9, 0.1]),
            "b0": np.asarray([-1.0, 0.0]),
        },
        "O": {
            "a0": np.asarray([0.0, 1.0]),
            "a2": np.asarray([0.1, 0.9]),
        },
    }

    selected, outcomes = ScreenRoundUseCase(route).select_element_per_system(
        frames,
        keep=2,
        element_vectors=element_vectors,
        type_map=("H", "O"),
    )

    assert tuple(frame.info["structure_id"] for frame in selected) == (
        "a0",
        "a2",
        "b0",
    )
    assert outcomes["a"]["candidate_count"] == 3
    assert outcomes["a"]["background_count"] == 0
    assert outcomes["b"]["selected_candidate_count"] == 1


@pytest.mark.parametrize("reuse", [False, True])
def test_element_direct_single_student_runs_after_teacher_trust_without_background(
    tmp_path: Path, reuse: bool,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        JointVectorReadout,
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )
    from workflow_core.structures.formats import write_extxyz_structures

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    frames = []
    records = []
    for index in range(3):
        identity = f"candidate-{index}"
        frame = Atoms("H", positions=[[float(index), 0.0, 0.0]])
        frame.info.update(structure_id=identity, system_id="water")
        frames.append(frame)
        records.append(_CatalogRecord("water", identity, index, 0.0))
    train_frame = Atoms("H")
    train_frame.info.update(structure_id="train-0", system_id="water")
    student_train_path = write_extxyz_structures(
        tmp_path / "student_train.xyz", [train_frame]
    )

    class Source:
        def catalog(self):
            return tuple(records)

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {
        "name": "readout-input",
        "revision": 1,
        "backend": "test",
        "model_family": "test-model",
        "definition": "atomic-energy readout inputs",
        "dtype": "float32",
        "width": 2,
        "normalized": False,
        "constant_terms_included": False,
        "deduplication": "unique-native-input",
        "components": [
            {
                "id": "layer",
                "native_source": "layer",
                "invariance": "scalar",
                "offset": 0,
                "width": 2,
            }
        ],
    }
    teacher_router = tmp_path / "teacher_router"
    teacher_router.mkdir()
    (teacher_router / "manifest.json").write_text(
        json.dumps(
            {
                "model_digest": "teacher-model",
                "reference_digest": "teacher-reference",
                "descriptor_schema": schema,
            }
        ),
        encoding="utf-8",
    )
    readout_calls = []

    def score_student(*, frames, router_dir: Path, **_kwargs):
        router_dir.mkdir(parents=True)
        (router_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_digest": "student-model",
                    "reference_digest": "student-reference",
                    "descriptor_schema": schema,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            values={str(frame.info["structure_id"]): 0.2 for frame in frames},
            frame_means=None,
            descriptor_schema=schema,
            **({"element_frame_means": {"H": {
                f"candidate-{index}": np.asarray([float(index), 1.0]) for index in range(3)
            }}} if reuse else {}),
        )

    def score_teacher(*, frames, **_kwargs):
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = 0.2
            scored.append(item)
        return SimpleNamespace(
            frames=tuple(scored), frame_means=None, descriptor_schema=schema
        )

    def read_vectors(*, frames, **_kwargs):
        assert ("reused_score" in _kwargs) == reuse
        readout_calls.append(tuple(frame.info["structure_id"] for frame in frames))
        return JointVectorReadout(
            candidate_vectors={
                f"candidate-{index}": np.asarray([float(index), 1.0])
                for index in range(3)
            },
            background_vectors={"train-0": np.asarray([0.0, 1.0])},
            candidate_element_vectors={
                "H": {
                    "candidate-0": np.asarray([1.0, 0.0]),
                    "candidate-1": np.asarray([0.0, 1.0]),
                    "candidate-2": np.asarray([-1.0, 0.0]),
                }
            },
            background_element_vectors={
                "H": {"train-0": np.asarray([0.5, 0.5])}
            },
            descriptor_schema=schema,
            model_identity="student",
            dataset_identity="iter0/r0/train/dataset",
        )

    def label_teacher(*, frames, **_kwargs):
        from ase.calculators.singlepoint import SinglePointCalculator

        labeled = []
        for frame in frames:
            item = frame.copy()
            item.calc = SinglePointCalculator(
                item,
                energy=0.0,
                forces=np.zeros((len(item), 3)),
            )
            labeled.append(item)
        return labeled

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="ELEMENT_DIRECT",
                fp_policy="random",
                pool=-1,
                keep=2,
                fp_pool=-1,
                teacher_p={"floor": 0.0, "trust": 0.1, "ceiling": 0.3},
                student_p={"floor": 0.0, "ceiling": 0.3},
                entropy="test/iter0/r0",
                visualize=False,
            ),
            services=ScreenServices(
                score_teacher=score_teacher,
                score_student=score_student,
                label_teacher=label_teacher,
                read_student_vectors=read_vectors,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            teacher_router=teacher_router,
            student_train_path=student_train_path,
            student_type_map=("H",),
        )
    )

    assert readout_calls == [("candidate-0", "candidate-1", "candidate-2")]
    assert tuple(
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    ) == ("candidate-0", "candidate-1")
    assert (tmp_path / "gmm_scores.npz").is_file()
    assert (tmp_path / "element_direct_screen.npz").is_file()
    from workflow_artifacts.lifecycle.screen_validation import (
        ScreenArtifactValidationRequest,
        validate_screen_artifacts,
    )

    validation = ScreenArtifactValidationRequest(
            contract=route.screen_artifact_contract,
            screen_dir=tmp_path,
            screen_policy="ELEMENT_DIRECT",
            fp_policy="random",
            pair_guard=None,
            expected_system_ids=("water",),
            expected_type_map=("H",),
            expected_thresholds={
                "routing_policy": route.screen_routing_policy_id.as_dict(),
                "teacher_p": {"floor": 0.0, "trust": 0.1, "ceiling": 0.3},
                "student_p": {"floor": 0.0, "ceiling": 0.3},
                "quota": {"fp_pool": -1, "pool": -1, "keep": 2},
                "fill_underfilled_from_safe_region": True,
                "selector": {"name": "element-direct", "revision": 1},
            },
            prescreen="GMM",
            joint_model_identity="student",
            joint_dataset_identity="iter0/r0/train/dataset",
            teacher_router=teacher_router,
        )
    from workflow_artifacts.lifecycle.screen_validation import (
        _valid_element_direct_selector,
    )

    assert _valid_element_direct_selector(validation)
    with np.load(tmp_path / "element_direct_screen.npz", allow_pickle=False) as archive:
        corrupted = {name: np.array(archive[name]) for name in archive.files}
    corrupted["candidate_element_0000"][0, 0] = np.nan
    np.savez_compressed(tmp_path / "element_direct_screen.npz", **corrupted)
    assert not _valid_element_direct_selector(validation)


def test_single_single_all_reject_short_circuits_route_services(tmp_path: Path) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    record = _CatalogRecord(
        system_id="water",
        structure_id="water-0001",
        md_step=1,
        student_uncertainty=0.0,
    )
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    frame.info.update(system_id="water", structure_id=record.structure_id)
    calls: list[str] = []

    def unexpected_service(*_args, **_kwargs):
        calls.append("route service")
        raise AssertionError("all-reject screen invoked route-specific work")

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=_ObservationSource(record, frame),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard={"min": 0.6, "pairs": {}},
                screen_policy="random",
                fp_policy="random",
                pool=1,
                keep=1,
                fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p={"floor": 0.0},
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=unexpected_service,
                score_student=unexpected_service,
                label_teacher=unexpected_service,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    assert result.observation_count == 1
    assert result.train_count == 0
    assert result.fp_candidate_count == 0
    assert calls == []
    assert (tmp_path / "pair_guard.npz").is_file()
    assert (tmp_path / "train.xyz").is_file()
    assert (tmp_path / "fp_candidate.xyz").is_file()
    assert (tmp_path / "screen_summary.json").is_file()
    assert not (tmp_path / "gmm_scores.npz").exists()
    assert not (tmp_path / "gmm_router").exists()
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert all(role["total"]["denominator"] == 0 for role in report["roles"])
    assert report["published"] == {"train": 0, "fp_candidate": 0}


@pytest.mark.parametrize("prescreen", ["JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"])
def test_joint_direct_full_pool_reads_only_for_enabled_visualization(
    tmp_path: Path, prescreen: str
) -> None:
    from types import SimpleNamespace

    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        JointVectorReadout,
        ScreenRoundInputs,
        ScreenServices,
        _joint_student_proposal,
    )
    from workflow_core.lifecycle.screen_use_case import ScreenRoundUseCase

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    frames = []
    for index in range(2):
        frame = Atoms("H", positions=[[index, 0, 0]])
        frame.info.update(structure_id=f"frame-{index}", system_id="water")
        frames.append(frame)
    calls = []

    def read_vectors(**_kwargs):
        calls.append("read")
        return JointVectorReadout(
            {f"frame-{index}": np.asarray([index, 1.0]) for index in range(2)},
            {"background": np.asarray([0.5, 1.0])},
            {"width": 2},
            "student",
            "dataset",
        )

    def request(visualize: bool):
        return SimpleNamespace(
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=2,
                keep=1,
                fp_pool=None,
                teacher_p=None,
                student_p=None,
                entropy="test",
                prescreen=prescreen,
                visualize=visualize,
            ),
            services=ScreenServices(None, None, lambda **_kwargs: [], read_vectors),
        )

    proposal, _readout, outcomes = _joint_student_proposal(
        request(True),
        core=ScreenRoundUseCase(route),
        observations=frames,
        work=tmp_path,
        report=SimpleNamespace(phase=lambda _phase: None),
    )
    assert [frame.info["structure_id"] for frame in proposal] == ["frame-0", "frame-1"]
    assert outcomes["water"]["bypass"] is True
    assert calls == []

    _joint_student_proposal(
        request(False),
        core=ScreenRoundUseCase(route),
        observations=frames,
        work=tmp_path,
        report=SimpleNamespace(phase=lambda _phase: None),
    )
    assert calls == []


def test_element_joint_full_pool_publishes_replayable_readout_free_bypass(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )
    from workflow_core.structures.formats import write_extxyz_structures

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    frame = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    frame.info.update(structure_id="candidate", system_id="water")
    record = _CatalogRecord("water", "candidate", 1, 0.0)
    train = Atoms("H")
    train.info.update(structure_id="train", system_id="water")
    student_train_path = write_extxyz_structures(
        tmp_path / "student_train.xyz", [train]
    )

    class Source:
        def catalog(self):
            return (record,)

        def replay(self, identities, *, chunk_size: int):
            assert tuple(identities) == ("candidate",)
            assert chunk_size > 0
            yield [frame.copy()]

    def label_teacher(*, frames, **_kwargs):
        from ase.calculators.singlepoint import SinglePointCalculator

        result = []
        for candidate in frames:
            item = candidate.copy()
            item.info["force_uncertainty"] = 0.05
            item.calc = SinglePointCalculator(
                item, energy=0.0, forces=np.zeros((len(item), 3))
            )
            result.append(item)
        return result

    def unexpected_readout(**_kwargs):
        raise AssertionError("element joint full-pool bypass invoked native readout")

    execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=1,
                keep=1,
                fp_pool=None,
                teacher_p=None,
                student_p=None,
                teacher_force_lo=0.1,
                teacher_force_hi=0.2,
                entropy="test",
                fill_underfilled_from_safe_region=False,
                prescreen="ELEMENT_JOINT_DIRECT",
                visualize=False,
            ),
            services=ScreenServices(
                score_teacher=None,
                score_student=None,
                label_teacher=label_teacher,
                read_student_vectors=unexpected_readout,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            student_train_path=student_train_path,
            student_type_map=("H",),
        )
    )

    from workflow_artifacts.lifecycle.element_direct import (
        load_element_direct_screen_evidence,
    )

    audit = load_element_direct_screen_evidence(
        tmp_path / "element_direct_screen.npz"
    )
    assert not bool(audit["readout_performed"].item())
    assert tuple(audit["proposal_ids"].tolist()) == ("candidate",)

def test_single_teacher_committee_student_all_reject_still_publishes_zero_report(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "committee"},
        sampler={"backend": "LAMMPS"},
    )
    record = _CatalogRecord("water", "mixed-rejected", 1, 0.15)
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    frame.info.update(system_id="water", structure_id=record.structure_id)

    def unexpected_service(*_args, **_kwargs):
        raise AssertionError("all-reject mixed route invoked route-specific work")

    execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=_ObservationSource(record, frame),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard={"min": 0.6, "pairs": {}},
                screen_policy="random",
                fp_policy="random",
                pool=1,
                keep=1,
                fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p=None,
                student_force_lo=0.1,
                student_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=unexpected_service,
                score_student=unexpected_service,
                label_teacher=unexpected_service,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert report["roles"][0]["total"]["denominator"] == 0
    assert report["roles"][1]["total"]["denominator"] == 0
    assert report["roles"][1]["total"]["counts"] == {
        "accurate": 0,
        "candidate": 0,
        "failed": 0,
    }


def test_committee_element_direct_all_reject_publishes_empty_fallback_evidence(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.policy_resolution import (
        load_policy_resolution,
    )
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "committee"},
        sampler={"backend": "LAMMPS"},
    )
    record = _CatalogRecord("water", "fallback-rejected", 1, 0.15)
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    frame.info.update(system_id="water", structure_id=record.structure_id)

    def unexpected_service(*_args, **_kwargs):
        raise AssertionError("all-reject fallback invoked route-specific work")

    class FallbackReporter(_Reporter):
        def warning(self, exc):
            assert "committee_descriptor_unavailable" in str(exc)

    execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=_ObservationSource(record, frame),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard={"min": 0.6, "pairs": {}},
                screen_policy="ELEMENT_DIRECT",
                fp_policy="random",
                pool=1,
                keep=1,
                fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p=None,
                student_force_lo=0.1,
                student_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=unexpected_service,
                score_student=unexpected_service,
                label_teacher=unexpected_service,
                read_student_vectors=unexpected_service,
            ),
            report=FallbackReporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    evidence = load_policy_resolution(tmp_path / "policy_resolution.json")
    assert evidence["rows"][0]["candidate_ids"] == []
    assert evidence["rows"][0]["selected_ids"] == []
    summary = json.loads(
        (tmp_path / "screen_summary.json").read_text(encoding="utf-8")
    )
    assert summary["selector_resolution"] == {
        "requested_policy": "ELEMENT_DIRECT",
        "effective_policy": "random",
        "fallback_reason": "committee_descriptor_unavailable",
    }
    assert not (tmp_path / "element_direct_screen.npz").exists()


def test_single_single_keeps_independent_overlapping_train_and_fp_views(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord(
            system_id="water",
            structure_id=f"water-{index:04d}",
            md_step=index,
            student_uncertainty=0.0,
        )
        for index in (1, 2)
    )
    frames = []
    for index, record in enumerate(records):
        frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8 + index, 0.0, 0.0]])
        frame.info.update(system_id="water", structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 2}
    teacher_router = tmp_path / "teacher_router"
    teacher_router.mkdir()
    manifest = {
        "model_digest": "teacher-model",
        "reference_digest": "teacher-reference",
        "descriptor_schema": schema,
    }
    (teacher_router / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    def score_teacher(*, frames, **_kwargs):
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = 0.2
            scored.append(item)
        return SimpleNamespace(
            frames=tuple(scored), frame_means=None, descriptor_schema=schema
        )

    def score_student(*, frames, router_dir: Path, **_kwargs):
        router_dir.mkdir(parents=True)
        (router_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_digest": "student-model",
                    "reference_digest": "student-reference",
                    "descriptor_schema": schema,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            values={str(frame.info["structure_id"]): 0.2 for frame in frames},
            frame_means=None,
            descriptor_schema=schema,
        )

    def label_teacher(*, frames, **_kwargs):
        return [frame.copy() for frame in frames]

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=-1,
                keep=2,
                fp_pool=-1,
                teacher_p={"floor": 0.0, "trust": 0.1, "ceiling": 0.3},
                student_p={"floor": 0.0, "ceiling": 0.3},
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=score_teacher,
                score_student=score_student,
                label_teacher=label_teacher,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            teacher_router=teacher_router,
        )
    )

    train_ids = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    ]
    fp_ids = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.fp_candidate_path, allow_empty=True)
    ]
    assert train_ids == [record.structure_id for record in records]
    assert fp_ids == train_ids
    assert result.train_count == result.fp_candidate_count == 2
    assert (tmp_path / "gmm_scores.npz").is_file()
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "dice.lifecycle.screen-report"
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert report["roles"][0]["total"]["counts"] == {
        "covered": 2,
        "candidate": 2,
        "discard": 0,
    }
    assert report["roles"][1]["total"]["denominator"] == 2
    assert report["published"] == {"train": 2, "fp_candidate": 2}


def test_student_first_single_single_scores_only_the_bounded_proposal_with_teacher(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord("water", f"proposal-{index:04d}", index, 0.0)
        for index in (1, 2, 3, 4)
    )
    source_frames = []
    for index, record in enumerate(records):
        frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8 + index, 0.0, 0.0]])
        frame.info.update(system_id=record.system_id, structure_id=record.structure_id)
        source_frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            by_id = {str(frame.info["structure_id"]): frame for frame in source_frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 2}
    teacher_router = tmp_path / "teacher_router"
    teacher_router.mkdir()
    (teacher_router / "manifest.json").write_text(
        json.dumps(
            {
                "model_digest": "teacher",
                "reference_digest": "reference",
                "descriptor_schema": schema,
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def score_student(*, frames, router_dir: Path, **_kwargs):
        identities = tuple(str(frame.info["structure_id"]) for frame in frames)
        calls.append(("student", identities))
        router_dir.mkdir(parents=True)
        (router_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_digest": "student",
                    "reference_digest": "reference",
                    "descriptor_schema": schema,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            values={
                "proposal-0001": 0.1,
                "proposal-0002": 0.2,
                "proposal-0003": 0.7,
                "proposal-0004": 0.8,
            },
            frame_means=None,
            descriptor_schema=schema,
        )

    def score_teacher(*, frames, **_kwargs):
        identities = tuple(str(frame.info["structure_id"]) for frame in frames)
        calls.append(("teacher", identities))
        probabilities = {"proposal-0001": 0.2, "proposal-0002": 0.05}
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = probabilities[str(item.info["structure_id"])]
            scored.append(item)
        return SimpleNamespace(
            frames=tuple(scored), frame_means=None, descriptor_schema=schema
        )

    def label_teacher(*, frames, **_kwargs):
        identities = tuple(str(frame.info["structure_id"]) for frame in frames)
        calls.append(("label", identities))
        return [frame.copy() for frame in frames]

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=2,
                keep=1,
                fp_pool=-1,
                teacher_p={"floor": 0.0, "trust": 0.1, "ceiling": 0.3},
                student_p={"floor": 0.0, "ceiling": 0.5},
                entropy="test/iter0/r0",
                fill_underfilled_from_safe_region=False,
            ),
            services=ScreenServices(
                score_teacher=score_teacher,
                score_student=score_student,
                label_teacher=label_teacher,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            teacher_router=teacher_router,
        )
    )

    train_ids = [
        str(frame.info["structure_id"])
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    ]
    fp_ids = [
        str(frame.info["structure_id"])
        for frame in read_extxyz_structures(result.fp_candidate_path, allow_empty=True)
    ]
    assert calls == [
        ("student", tuple(record.structure_id for record in records)),
        ("teacher", ("proposal-0001", "proposal-0002")),
        ("label", ("proposal-0001",)),
    ]
    assert train_ids == ["proposal-0001"]
    assert fp_ids == ["proposal-0001", "proposal-0002"]
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert report["route"] == {
        "name": "single-teacher/single-student",
        "revision": 1,
        "directional_policy": {
            "name": "student-reference-tail_teacher-reference-tail",
            "revision": 1,
        },
    }
    assert report["roles"][0]["total"]["denominator"] == 2
    assert report["roles"][1]["total"]["denominator"] == 4


@pytest.mark.parametrize('variable_n', [False, True])
def test_single_single_fills_an_underfilled_gmm_system_from_covered_frames(
    tmp_path: Path,
    variable_n: bool,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord("water", f"safe-fill-{index:04d}", index, None if variable_n else 0.0)
        for index in (1, 2, 3)
    )
    frames = []
    for index, record in enumerate(records):
        frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8 + index, 0.0, 0.0]])
        if variable_n:
            frame = Atoms('H'*(index+1), positions=[[.8*j, 0., 0.] for j in range(index+1)])
        frame.info.update(system_id=record.system_id, structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 2}
    teacher_router = tmp_path / "teacher_router"
    teacher_router.mkdir()
    (teacher_router / "manifest.json").write_text(
        json.dumps({"model_digest": "teacher", "reference_digest": "reference", "descriptor_schema": schema}),
        encoding="utf-8",
    )

    teacher_score_calls: list[tuple[str, ...]] = []

    def score_teacher(*, frames, **_kwargs):
        teacher_score_calls.append(
            tuple(str(frame.info["structure_id"]) for frame in frames)
        )
        probabilities = {
            "safe-fill-0001": 0.2,
            "safe-fill-0002": 0.2,
            "safe-fill-0003": 0.05,
        }
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = probabilities[str(item.info["structure_id"])]
            scored.append(item)
        return SimpleNamespace(frames=tuple(scored), frame_means=None, descriptor_schema=schema)

    def score_student(*, frames, router_dir: Path, **_kwargs):
        router_dir.mkdir(parents=True)
        (router_dir / "manifest.json").write_text(
            json.dumps({"model_digest": "student", "reference_digest": "reference", "descriptor_schema": schema}),
            encoding="utf-8",
        )
        return SimpleNamespace(
            values={
                "safe-fill-0001": 0.2,
                "safe-fill-0002": 0.5,
                "safe-fill-0003": 0.7,
            },
            frame_means=None,
            descriptor_schema=schema,
        )

    label_calls: list[tuple[str, ...]] = []

    def label_teacher(*, frames, **_kwargs):
        label_calls.append(tuple(str(frame.info["structure_id"]) for frame in frames))
        return [frame.copy() for frame in frames]

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",), pair_guard=None, screen_policy="random",
                fp_policy="random", pool=1, keep=3, fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p={"floor": 0.0, "ceiling": 0.5}, entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=score_teacher, score_student=score_student,
                label_teacher=label_teacher,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(), teacher_router=teacher_router,
        )
    )

    train_ids = [
        str(frame.info["structure_id"])
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    ]
    assert teacher_score_calls == [
        ("safe-fill-0001",),
        ("safe-fill-0002", "safe-fill-0003"),
    ]
    assert train_ids == ["safe-fill-0001", "safe-fill-0002"]
    assert label_calls == [tuple(train_ids)]
    with np.load(tmp_path / "gmm_scores.npz", allow_pickle=False) as audit:
        assert int(audit["schema_revision"].item()) == 5
        assert audit["normal_teacher_scored"].tolist() == [True, False, False]
        assert audit["teacher_scored"].tolist() == [True, True, True]
        assert audit["normal_train_selected"].tolist() == [True, False, False]
        assert audit["safe_fill_eligible"].tolist() == [False, True, True]
        assert audit["safe_fill_attempted"].tolist() == [False, True, True]
        assert audit["safe_fill_accepted"].tolist() == [False, True, False]
        assert audit["train_selected"].tolist() == [True, True, False]
        assert audit["fp_candidate"].tolist() == [True, False, False]
    summary = json.loads((tmp_path / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_revision"] == 6
    assert "gate_order" not in summary
    assert summary["directional_policy"] == {
        "name": "student-reference-tail_teacher-reference-tail",
        "revision": 1,
    }
    assert summary["fill_underfilled_from_safe_region"] is True
    assert summary["per_system"]["water"]["normal_teacher_scored"] == 1
    assert summary["per_system"]["water"]["normal_train"] == 1
    assert summary["per_system"]["water"]["safe_fill_eligible"] == 2
    assert summary["per_system"]["water"]["safe_fill_attempted"] == 2
    assert summary["per_system"]["water"]["safe_fill_accepted"] == 1
    assert summary["per_system"]["water"]["train_underfill"] == 1
    from workflow_artifacts.lifecycle.gmm_screen import validate_gmm_screen_summary

    assert "gate_order" not in validate_gmm_screen_summary(
        tmp_path / "screen_summary.json"
    )


@pytest.mark.parametrize('variable_n', [False, True])
def test_single_teacher_committee_student_keeps_directional_overlap_without_student_gmm(
    tmp_path: Path,
    variable_n: bool,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={
            "backend": "DeepMD",
            "cli": "tf",
            "model": "committee",
        },
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord("water", f"mixed-{index:04d}", index, deviation)
        for index, deviation in enumerate((0.15, 0.14, 0.05, 0.25), start=1)
    )
    frames = []
    for index, record in enumerate(records):
        frame = Atoms(
            "H2", positions=[[0.0, 0.0, 0.0], [0.8 + index, 0.0, 0.0]]
        )
        if variable_n and index:
            frame += Atoms('H'*index, positions=[[4.+j, 0., 0.] for j in range(index)])
        frame.info.update(system_id="water", structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 2}
    teacher_router = tmp_path / "teacher_router"
    teacher_router.mkdir()
    (teacher_router / "manifest.json").write_text(
        json.dumps(
            {
                "model_digest": "teacher-model",
                "reference_digest": "teacher-reference",
                "descriptor_schema": schema,
            }
        ),
        encoding="utf-8",
    )

    teacher_calls: list[tuple[str, ...]] = []

    def score_teacher(*, frames, **_kwargs):
        identities = tuple(str(frame.info["structure_id"]) for frame in frames)
        teacher_calls.append(identities)
        probabilities = {"mixed-0001": 0.2, "mixed-0003": 0.05}
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = probabilities[str(item.info["structure_id"])]
            scored.append(item)
        return SimpleNamespace(
            frames=tuple(scored), frame_means=None, descriptor_schema=schema
        )

    def unexpected_student_gmm(*_args, **_kwargs):
        raise AssertionError("single-teacher/committee-student invoked student GMM")

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=1,
                keep=2,
                fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p=None,
                student_force_lo=0.1,
                student_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=score_teacher,
                score_student=unexpected_student_gmm,
                label_teacher=lambda *, frames, **_kwargs: [
                    frame.copy() for frame in frames
                ],
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            teacher_router=teacher_router,
        )
    )

    train_ids = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    ]
    fp_ids = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.fp_candidate_path, allow_empty=True)
    ]
    assert teacher_calls == [("mixed-0001",), ("mixed-0003",)]
    assert train_ids == ["mixed-0001"]
    assert fp_ids == ["mixed-0001"]
    assert (tmp_path / "mixed_route_scores.npz").is_file()
    assert not (tmp_path / "gmm_router").exists()
    assert not (tmp_path / "gmm_scores.npz").exists()
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert "gate_order" not in report["route"]
    assert report["route"]["directional_policy"] == {
        "name": "student-force-deviation_teacher-reference-tail",
        "revision": 1,
    }
    assert report["roles"][0]["total"] == {
        "system_id": "TOTAL",
        "denominator": 2,
        "counts": {"covered": 1, "candidate": 2, "discard": 0},
    }
    assert report["roles"][1]["total"] == {
        "system_id": "TOTAL",
        "denominator": 4,
        "counts": {"accurate": 1, "candidate": 2, "failed": 1},
    }
    assert report["published"] == {"train": 1, "fp_candidate": 1}
    from workflow_artifacts.lifecycle.mixed_route import load_mixed_route_audit

    audit_path = tmp_path / "mixed_route_scores.npz"
    audit = load_mixed_route_audit(audit_path)
    assert audit["normal_teacher_scored"].tolist() == [True, False, False, False]
    assert audit["teacher_scored"].tolist() == [True, False, True, False]
    assert np.isnan(audit["teacher_p"][[1, 3]]).all()
    assert audit["fp_candidate"].tolist() == [True, False, False, False]

    corrupted = {
        name: np.asarray(value)
        for name, value in audit.items()
        if name != "thresholds"
    }
    corrupted["thresholds"] = np.asarray(
        json.dumps(audit["thresholds"], sort_keys=True, separators=(",", ":"))
    )
    corrupted["student_pool"] = corrupted["student_pool"].copy()
    corrupted["student_pool"][[0, 1]] = [False, True]
    corrupted_path = tmp_path / "corrupted_mixed_route_scores.npz"
    np.savez_compressed(corrupted_path, **corrupted)
    try:
        load_mixed_route_audit(corrupted_path)
    except ValueError as exc:
        assert "proposal" in str(exc)
    else:
        raise AssertionError("corrupt student-first mixed proposal was accepted")


def test_committee_student_element_direct_falls_back_to_exact_random_without_readout(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.policy_resolution import (
        load_policy_resolution,
    )
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "committee"},
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord("water", f"fallback-{index}", index, 0.15)
        for index in range(4)
    )
    frames = []
    for index, record in enumerate(records):
        frame = Atoms("H", positions=[[float(index), 0.0, 0.0]])
        frame.info.update(system_id="water", structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    class Reporter:
        def __init__(self):
            self.warnings = []

        def phase(self, _name):
            pass

        def snapshot(self, _snapshot):
            pass

        def summary(self, _summary):
            pass

        def warning(self, exc):
            self.warnings.append(str(exc))

    def score_teacher(*, frames, **_kwargs):
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = 0.5
            scored.append(item)
        return SimpleNamespace(
            frames=tuple(scored), frame_means=None, descriptor_schema=None
        )

    outputs = {}
    for policy in ("random", "ELEMENT_DIRECT"):
        stage = tmp_path / policy.lower()
        stage.mkdir()
        reporter = Reporter()
        result = execute_screen(
            ScreenExecutionRequest(
                route=route,
                round_index=0,
                stage=stage,
                source=Source(),
                inputs=ScreenRoundInputs(
                    system_ids=("water",), pair_guard=None,
                    screen_policy=policy, fp_policy="random",
                    pool=4, keep=2, fp_pool=1,
                    teacher_p={"floor": 0.0, "trust": 0.1}, student_p=None,
                    student_force_lo=0.1, student_force_hi=0.2,
                    entropy="iter0/r0/screen",
                ),
                services=ScreenServices(
                    score_teacher=score_teacher,
                    score_student=None,
                    read_student_vectors=(
                        lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("committee fallback requested descriptors")
                        )
                    ),
                    label_teacher=lambda *, frames, **_kwargs: [
                        frame.copy() for frame in frames
                    ],
                ),
                report=reporter,
                artifacts=ScreenArtifactPublicationCapability(),
            )
        )
        outputs[policy] = [
            str(frame.info["structure_id"])
            for frame in read_extxyz_structures(result.train_path, allow_empty=True)
        ]
        if policy == "ELEMENT_DIRECT":
            assert reporter.warnings == [
                "screen.policy ELEMENT_DIRECT fell back to random for water: "
                "committee_descriptor_unavailable"
            ]
            evidence = load_policy_resolution(stage / "policy_resolution.json")
            assert evidence["rows"][0]["selected_ids"] == outputs[policy]
            summary = json.loads(
                (stage / "screen_summary.json").read_text(encoding="utf-8")
            )
            assert summary["selector_resolution"] == {
                "requested_policy": "ELEMENT_DIRECT",
                "effective_policy": "random",
                "fallback_reason": "committee_descriptor_unavailable",
            }
        else:
            assert reporter.warnings == []
            assert not (stage / "policy_resolution.json").exists()

    assert outputs["ELEMENT_DIRECT"] == outputs["random"]


def test_single_teacher_committee_student_report_includes_safe_fill_and_all_force_bins(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "single"},
        student={"backend": "DeepMD", "cli": "tf", "model": "committee"},
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord("water", f"safe-mixed-{index:04d}", index, value)
        for index, value in enumerate((0.05, 0.1, 0.2), start=1)
    )
    frames = []
    for record in records:
        frame = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        frame.info.update(system_id=record.system_id, structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            by_id = {record.structure_id: frame for record, frame in zip(records, frames, strict=True)}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 1}
    teacher_router = tmp_path / "teacher_router"
    teacher_router.mkdir()
    (teacher_router / "manifest.json").write_text(
        json.dumps({"model_digest": "teacher", "reference_digest": "reference", "descriptor_schema": schema}),
        encoding="utf-8",
    )

    def score_teacher(*, frames, **_kwargs):
        scored = []
        for frame in frames:
            item = frame.copy()
            item.info["teacher_p"] = 0.2
            scored.append(item)
        return SimpleNamespace(frames=tuple(scored), frame_means=None, descriptor_schema=schema)

    def label_teacher(*, frames, **_kwargs):
        return [frame.copy() for frame in frames]

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=2,
                keep=2,
                fp_pool=1,
                teacher_p={"floor": 0.0, "trust": 0.1},
                student_p=None,
                student_force_lo=0.1,
                student_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=score_teacher,
                score_student=None,
                label_teacher=label_teacher,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            teacher_router=teacher_router,
        )
    )

    train_ids = {
        str(frame.info["structure_id"])
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    }
    assert train_ids == {"safe-mixed-0001", "safe-mixed-0002"}
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    teacher, student = report["roles"]
    assert teacher["total"]["denominator"] == 2
    assert teacher["total"]["counts"] == {"covered": 2, "candidate": 2, "discard": 0}
    assert student["total"]["denominator"] == 3
    assert student["total"]["counts"] == {"accurate": 1, "candidate": 1, "failed": 1}
    assert report["published"] == {
        "train": result.train_count,
        "fp_candidate": result.fp_candidate_count,
    }
    summary = json.loads((tmp_path / "screen_summary.json").read_text(encoding="utf-8"))
    assert "gate_order" not in summary
    assert summary["counts"]["normal_teacher_scored"] == 1
    assert summary["counts"]["teacher_scored"] == 2
    assert summary["counts"]["safe_fill_attempted"] == 1
    assert summary["counts"]["safe_fill_accepted"] == 1


def test_committee_teacher_single_student_all_reject_publishes_zero_roles(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    record = _CatalogRecord("water", "committee-rejected", 1, 0.0)
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    frame.info.update(system_id="water", structure_id=record.structure_id)

    def unexpected_service(*_args, **_kwargs):
        raise AssertionError("all-reject committee-teacher route invoked route work")

    execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=_ObservationSource(record, frame),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard={"min": 0.6, "pairs": {}},
                screen_policy="random",
                fp_policy="random",
                pool=1,
                keep=1,
                fp_pool=None,
                teacher_p=None,
                student_p={"floor": 0.0, "ceiling": 0.3},
                teacher_force_lo=0.1,
                teacher_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=unexpected_service,
                score_student=unexpected_service,
                label_teacher=unexpected_service,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert all(role["total"]["denominator"] == 0 for role in report["roles"])


def test_committee_teacher_single_student_bounds_teacher_evaluation_and_keeps_outputs_disjoint(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord(
            system_id="water",
            structure_id=f"bounded-{index:04d}",
            md_step=index,
            student_uncertainty=0.0,
        )
        for index in (1, 2, 3)
    )
    frames = []
    for index, record in enumerate(records):
        frame = Atoms(
            "H2", positions=[[0.0, 0.0, 0.0], [0.8 + index, 0.0, 0.0]]
        )
        frame.info.update(system_id="water", structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 2}

    def score_student(*, frames, router_dir: Path, **_kwargs):
        router_dir.mkdir(parents=True)
        (router_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model_digest": "student-model",
                    "reference_digest": "student-reference",
                    "descriptor_schema": schema,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            values={str(frame.info["structure_id"]): 0.2 for frame in frames},
            frame_means=None,
            descriptor_schema=schema,
        )

    evaluated_ids: list[str] = []

    def label_teacher(*, frames, **_kwargs):
        result = []
        for index, frame in enumerate(frames):
            item = frame.copy()
            evaluated_ids.append(str(item.info["structure_id"]))
            item.info["force_uncertainty"] = 0.05 if index == 0 else 0.15
            result.append(item)
        return result

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=2,
                keep=1,
                fp_pool=None,
                teacher_p=None,
                student_p={"floor": 0.0, "ceiling": 0.3},
                teacher_force_lo=0.1,
                teacher_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=None,
                score_student=score_student,
                label_teacher=label_teacher,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    train_ids = {
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    }
    fp_ids = {
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.fp_candidate_path, allow_empty=True)
    }
    assert len(evaluated_ids) == 2
    assert len(train_ids) == len(fp_ids) == 1
    assert train_ids.isdisjoint(fp_ids)
    assert (tmp_path / "gmm_router").is_dir()
    assert (tmp_path / "mixed_route_scores.npz").is_file()
    assert not (tmp_path / "gmm_scores.npz").exists()
    summary = json.loads((tmp_path / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["fill_underfilled_from_safe_region"] is True
    assert summary["counts"]["normal_train"] == 1
    assert summary["counts"]["safe_fill_eligible"] == 0
    assert summary["counts"]["safe_fill_attempted"] == 0
    assert summary["counts"]["safe_fill_accepted"] == 0
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert report["roles"][0]["total"]["denominator"] == 2
    assert report["roles"][0]["total"]["counts"] == {
        "accurate": 1,
        "candidate": 1,
        "failed": 0,
    }
    assert report["roles"][1]["total"]["denominator"] == 3
    assert report["roles"][1]["total"]["counts"] == {
        "covered": 0,
        "candidate": 3,
        "discard": 0,
    }
    assert report["published"] == {
        "train": result.train_count,
        "fp_candidate": result.fp_candidate_count,
    }


def test_committee_teacher_single_student_safe_fill_accepts_ceiling_and_preserves_fp(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "single"},
        sampler={"backend": "LAMMPS"},
    )
    records = (
        _CatalogRecord("water", "normal", 1, 0.0),
        _CatalogRecord("water", "at-ceiling", 2, 0.0),
    )
    frames = []
    for record in records:
        frame = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        frame.info.update(system_id=record.system_id, structure_id=record.structure_id)
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    schema = {"schema": "readout-input@1", "width": 1}

    def run(stage: Path, *, enabled: bool, safe_teacher_deviation: float = 0.05):
        def score_student(*, frames, router_dir: Path, **_kwargs):
            router_dir.mkdir(parents=True)
            (router_dir / "manifest.json").write_text(
                json.dumps({
                    "model_digest": "student", "reference_digest": "reference",
                    "descriptor_schema": schema,
                }), encoding="utf-8",
            )
            return SimpleNamespace(
                values={"normal": 0.2, "at-ceiling": 0.3},
                frame_means=None, descriptor_schema=schema,
            )

        def label_teacher(*, frames, **_kwargs):
            labeled = []
            for frame in frames:
                item = frame.copy()
                item.info["force_uncertainty"] = (
                    0.15
                    if item.info["structure_id"] == "normal"
                    else safe_teacher_deviation
                )
                labeled.append(item)
            return labeled

        return execute_screen(
            ScreenExecutionRequest(
                route=route, round_index=0, stage=stage, source=Source(),
                inputs=ScreenRoundInputs(
                    system_ids=("water",), pair_guard=None, screen_policy="random",
                    fp_policy="random", pool=1, keep=1, fp_pool=None,
                    teacher_p=None, student_p={"floor": 0.0, "ceiling": 0.3},
                    teacher_force_lo=0.1, teacher_force_hi=0.2,
                    fill_underfilled_from_safe_region=enabled, entropy="p-ceiling-fill",
                ),
                services=ScreenServices(
                    score_teacher=None, score_student=score_student,
                label_teacher=label_teacher,
                ),
                report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            )
        )

    enabled = run(tmp_path / "enabled", enabled=True)
    disabled = run(tmp_path / "disabled", enabled=False)
    rejected = run(tmp_path / "rejected", enabled=True, safe_teacher_deviation=0.3)
    enabled_train = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(enabled.train_path, allow_empty=True)
    ]
    disabled_train = read_extxyz_structures(disabled.train_path, allow_empty=True)
    enabled_fp = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(enabled.fp_candidate_path, allow_empty=True)
    ]
    disabled_fp = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(disabled.fp_candidate_path, allow_empty=True)
    ]
    rejected_train = read_extxyz_structures(rejected.train_path, allow_empty=True)

    assert enabled_train == ["at-ceiling"]
    assert disabled_train == []
    assert enabled_fp == disabled_fp == ["normal"]
    assert rejected_train == []
    summary = json.loads((tmp_path / "enabled" / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["normal_train"] == 0
    assert summary["counts"]["safe_fill_eligible"] == 1
    assert summary["counts"]["safe_fill_attempted"] == 1
    assert summary["counts"]["safe_fill_accepted"] == 1
    assert summary["counts"]["train_underfill"] == 0
    report = json.loads((tmp_path / "enabled" / "screen_report.json").read_text(encoding="utf-8"))
    assert report["roles"][0]["total"]["denominator"] == 2
    assert report["roles"][0]["total"]["counts"] == {
        "accurate": 1,
        "candidate": 1,
        "failed": 0,
    }
    assert report["roles"][1]["total"]["denominator"] == 2
    assert report["roles"][1]["total"]["counts"] == {
        "covered": 1,
        "candidate": 1,
        "discard": 0,
    }
    assert report["published"] == {"train": 1, "fp_candidate": 1}
    with np.load(tmp_path / "rejected" / "mixed_route_scores.npz", allow_pickle=False) as audit:
        assert audit["safe_fill_attempted"].tolist() == [False, True]
        assert audit["safe_fill_accepted"].tolist() == [False, False]
        assert audit["teacher_high"].tolist() == [False, True]


def test_committee_committee_all_reject_publishes_zero_roles(tmp_path: Path) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "committee"},
        sampler={"backend": "LAMMPS"},
    )
    record = _CatalogRecord("water", "committee-both-rejected", 1, 0.15)
    frame = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    frame.info.update(system_id="water", structure_id=record.structure_id)

    def unexpected_service(*_args, **_kwargs):
        raise AssertionError("all-reject committee route invoked route work")

    execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=_ObservationSource(record, frame),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                pair_guard={"min": 0.6, "pairs": {}},
                screen_policy="random",
                fp_policy="random",
                pool=1,
                keep=1,
                fp_pool=None,
                teacher_p=None,
                student_p=None,
                student_force_lo=0.1,
                student_force_hi=0.2,
                teacher_force_lo=0.1,
                teacher_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=unexpected_service,
                score_student=unexpected_service,
                label_teacher=unexpected_service,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert all(role["total"]["denominator"] == 0 for role in report["roles"])


@pytest.mark.parametrize('grouped',[False,True])
def test_committee_committee_uses_force_deviation_without_gmm_artifacts(
    tmp_path: Path, grouped: bool,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan
    plan = SelectionPlan.from_seeds((SeedRole('role-a','water'),)) if grouped else None
    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={
            "backend": "DeepMD",
            "cli": "tf",
            "model": "committee",
        },
        sampler={"backend": "LAMMPS"},
    )
    records = tuple(
        _CatalogRecord(
            system_id="water",
            structure_id=f"committee-{index:04d}",
            md_step=index,
            student_uncertainty=0.15,
        )
        for index in (1, 2)
    )
    if grouped:
        records = tuple(SimpleNamespace(**vars(record), sampling_role_id='role-a',
            selection_group_id='water', condition_group_id='native', task_id='task_000000') for record in records)
    frames = []
    for index, record in enumerate(records):
        frame = Atoms(
            "H2", positions=[[0.0, 0.0, 0.0], [0.8 + index, 0.0, 0.0]]
        )
        frame.info.update(system_id="water", structure_id=record.structure_id)
        if grouped:
            frame.info.update(sampling_role_id='role-a', selection_group_id='water', condition_group_id='native', task_id='task_000000')
        frames.append(frame)

    if grouped:
        from workflow_core.structures.canonical_identity import stamp_canonical_identity
        frames = [stamp_canonical_identity(frame, system_id='water', structure_source='workflow') for frame in frames]
        records = tuple(SimpleNamespace(**{**vars(record),'structure_id':frame.info['structure_id']})
                        for record,frame in zip(records,frames,strict=True))

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            assert chunk_size > 0
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    def label_teacher(*, frames, **_kwargs):
        result = []
        for index, frame in enumerate(frames):
            item = frame.copy()
            item.info["force_uncertainty"] = 0.05 if index == 0 else 0.15
            from ase.calculators.singlepoint import SinglePointCalculator
            item.calc = SinglePointCalculator(item, energy=0., forces=np.zeros((len(item),3)))
            result.append(item)
        return result

    result = execute_screen(
        ScreenExecutionRequest(
            route=route,
            round_index=0,
            stage=tmp_path,
            source=Source(),
            inputs=ScreenRoundInputs(
                system_ids=("water",),
                selection_plan=plan,
                require_explicit_provenance=grouped,
                pair_guard=None,
                screen_policy="random",
                fp_policy="random",
                pool=-1,
                keep=2,
                fp_pool=None,
                teacher_p=None,
                student_p=None,
                student_force_lo=0.1,
                student_force_hi=0.2,
                teacher_force_lo=0.1,
                teacher_force_hi=0.2,
                entropy="test/iter0/r0",
            ),
            services=ScreenServices(
                score_teacher=None,
                score_student=None,
                label_teacher=label_teacher,
            ),
            report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
        )
    )

    train_ids = {
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.train_path, allow_empty=True)
    }
    fp_ids = {
        frame.info["structure_id"]
        for frame in read_extxyz_structures(result.fp_candidate_path, allow_empty=True)
    }
    assert len(train_ids) == len(fp_ids) == 1
    assert train_ids.isdisjoint(fp_ids)
    assert not (tmp_path / "gmm_router").exists()
    assert not (tmp_path / "gmm_scores.npz").exists()
    assert (tmp_path / "committee_force_scores.npz").is_file()
    summary = json.loads((tmp_path / "screen_summary.json").read_text(encoding="utf-8"))
    assert summary["fill_underfilled_from_safe_region"] is True
    assert summary["counts"]["normal_train"] == 1
    assert summary["counts"]["safe_fill_eligible"] == 0
    assert summary["counts"]["safe_fill_attempted"] == 0
    assert summary["counts"]["safe_fill_accepted"] == 0
    with np.load(tmp_path / "committee_force_scores.npz", allow_pickle=False) as audit:
        assert audit["normal_train_selected"].tolist() == [True, False]
        assert audit["safe_fill_enabled"].item() is True
        assert audit["safe_fill_eligible"].tolist() == [False, False]
        assert audit["safe_fill_attempted"].tolist() == [False, False]
        assert audit["safe_fill_accepted"].tolist() == [False, False]
    report = json.loads((tmp_path / "screen_report.json").read_text(encoding="utf-8"))
    assert [role["role"] for role in report["roles"]] == ["teacher", "student"]
    assert report["roles"][0]["total"]["denominator"] == 2
    assert report["roles"][0]["total"]["counts"] == {
        "accurate": 1,
        "candidate": 1,
        "failed": 0,
    }
    assert report["roles"][1]["total"]["denominator"] == 2
    assert report["roles"][1]["total"]["counts"] == {
        "accurate": 0,
        "candidate": 2,
        "failed": 0,
    }
    assert report["published"] == {"train": 1, "fp_candidate": 1}

    if grouped:
        from workflow_artifacts.lifecycle.screen_validation import ScreenArtifactValidationRequest, validate_screen_artifacts
        validation = ScreenArtifactValidationRequest(route.screen_artifact_contract, tmp_path,
            'random','random',None,('water',),('H',),summary['thresholds'],
            selection_plan=plan, require_selection_group_audit=True)
        assert validate_screen_artifacts(validation)
        (tmp_path/'screen_report.json').unlink()
        assert validate_screen_artifacts(validation)
        (tmp_path/'screen_report.json').write_text('damaged diagnostic')
        assert validate_screen_artifacts(validation)


def test_committee_committee_safe_fill_uses_strict_lower_region_and_preserves_fp(
    tmp_path: Path,
) -> None:
    from workflow_core.lifecycle.routes import resolve_route
    from workflow_core.lifecycle.screen_executor import (
        ScreenExecutionRequest,
        ScreenRoundInputs,
        ScreenServices,
        execute_screen,
    )

    route = resolve_route(
        teacher={"backend": "MACE", "model": "committee"},
        student={"backend": "DeepMD", "cli": "tf", "model": "committee"},
        sampler={"backend": "LAMMPS"},
    )
    records = (
        _CatalogRecord("water", "at-lo", 1, 0.1),
        _CatalogRecord("water", "below-lo", 2, 0.05),
    )
    frames = []
    for record in records:
        frame = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        frame.info.update(
            system_id=record.system_id,
            structure_id=record.structure_id,
            force_uncertainty=record.student_uncertainty,
        )
        frames.append(frame)

    class Source:
        def catalog(self):
            return records

        def replay(self, identities, *, chunk_size: int):
            by_id = {str(frame.info["structure_id"]): frame for frame in frames}
            yield [by_id[identity].copy() for identity in identities]

    def run(stage: Path, *, enabled: bool, safe_teacher_deviation: float = 0.05):
        def label_teacher(*, frames, **_kwargs):
            labeled = []
            for frame in frames:
                item = frame.copy()
                item.info["force_uncertainty"] = (
                    0.15
                    if item.info["structure_id"] == "at-lo"
                    else safe_teacher_deviation
                )
                labeled.append(item)
            return labeled

        return execute_screen(
            ScreenExecutionRequest(
                route=route,
                round_index=0,
                stage=stage,
                source=Source(),
                inputs=ScreenRoundInputs(
                    system_ids=("water",), pair_guard=None, screen_policy="random",
                    fp_policy="random", pool=-1, keep=1, fp_pool=None,
                    teacher_p=None, student_p=None, student_force_lo=0.1,
                    student_force_hi=0.2, teacher_force_lo=0.1, teacher_force_hi=0.2,
                    fill_underfilled_from_safe_region=enabled, entropy="committee-safe-fill",
                ),
                services=ScreenServices(
                    score_teacher=None, score_student=None, label_teacher=label_teacher,
                ),
                report=_Reporter(),
            artifacts=ScreenArtifactPublicationCapability(),
            )
        )

    enabled = run(tmp_path / "enabled", enabled=True)
    disabled = run(tmp_path / "disabled", enabled=False)
    rejected = run(tmp_path / "rejected", enabled=True, safe_teacher_deviation=0.3)
    enabled_train = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(enabled.train_path, allow_empty=True)
    ]
    disabled_train = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(disabled.train_path, allow_empty=True)
    ]
    enabled_fp = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(enabled.fp_candidate_path, allow_empty=True)
    ]
    disabled_fp = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(disabled.fp_candidate_path, allow_empty=True)
    ]
    rejected_train = read_extxyz_structures(rejected.train_path, allow_empty=True)
    rejected_fp = [
        frame.info["structure_id"]
        for frame in read_extxyz_structures(rejected.fp_candidate_path, allow_empty=True)
    ]

    assert enabled_train == ["below-lo"]
    assert disabled_train == []
    assert enabled_fp == disabled_fp == ["at-lo"]
    assert rejected_train == []
    assert rejected_fp == ["at-lo"]
    report = json.loads((tmp_path / "enabled" / "screen_report.json").read_text(encoding="utf-8"))
    assert report["roles"][0]["total"]["counts"] == {
        "accurate": 1,
        "candidate": 1,
        "failed": 0,
    }
    assert report["roles"][0]["total"]["denominator"] == 2
    assert report["roles"][1]["total"]["counts"] == {
        "accurate": 1,
        "candidate": 1,
        "failed": 0,
    }
    assert report["roles"][1]["total"]["denominator"] == 2
    assert report["published"] == {"train": 1, "fp_candidate": 1}
    with np.load(tmp_path / "enabled" / "committee_force_scores.npz", allow_pickle=False) as audit:
        assert audit["student_in_band"].tolist() == [True, False]
        assert audit["safe_fill_eligible"].tolist() == [False, True]
        assert audit["safe_fill_attempted"].tolist() == [False, True]
        assert audit["safe_fill_accepted"].tolist() == [False, True]
    with np.load(tmp_path / "rejected" / "committee_force_scores.npz", allow_pickle=False) as audit:
        assert audit["safe_fill_attempted"].tolist() == [False, True]
        assert audit["safe_fill_accepted"].tolist() == [False, False]
        assert audit["teacher_high"].tolist() == [False, True]
