from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
import pytest


class _FakeLammpsExecutor:
    def run(self, task_dir: Path, task: dict) -> None:
        del task
        (task_dir / "traj.lammpstrj").write_text(
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n"
            "ITEM: ATOMS id type x y z\n1 1 0 0 0\n"
            "ITEM: TIMESTEP\n20\n"
            "ITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\n"
            "ITEM: ATOMS id type x y z\n1 1 0.1 0 0\n",
            encoding="utf-8",
        )
        (task_dir / "model_devi.out").write_text(
            "0 0 0 0 0.1 0 0.01\n20 0 0 0 0.2 0 0.01\n",
            encoding="utf-8",
        )


class _FakeGpumdExecutor:
    def run(self, task_dir: Path, task: dict) -> None:
        del task
        (task_dir / "active.out").write_text("20 0.3\n", encoding="utf-8")
        (task_dir / "active.xyz").write_text(
            "1\nProperties=species:S:1:pos:R:3\nH 0.1 0 0\n",
            encoding="utf-8",
        )


class _IdentitySamplingModelProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prepare_sampling_models(self, request):
        from workflow_backends.sampling_models import (
            OrderedSamplingModels,
            SamplingModelsResult,
        )

        self.calls.append("prepare")
        return SamplingModelsResult(
            OrderedSamplingModels(
                tuple(request.models.paths),
                member_ids=tuple(request.models.member_ids),
            )
        )

    def load_sampling_models(self, request):
        self.calls.append("load")
        from workflow_backends.sampling_models import SamplingModelsResult

        return SamplingModelsResult(request.models)


class _PathBackedEvidenceObserver:
    def observe_evidence(self, request):
        from workflow_backends.sampling import observe_path_backed_sampling_evidence

        return observe_path_backed_sampling_evidence(request)


def test_core_sampling_intent_accepts_closed_neb_parameters() -> None:
    from workflow_core.common.sampling_plan import NebSamplingParameters
    from workflow_core.lifecycle.sampling import SamplingTaskIntent

    parameters = NebSamplingParameters(
        images=5,
        fmax_ev_a=0.05,
        max_steps=100,
        include_history=False,
    )

    task = SamplingTaskIntent(
        task_id="task_000000",
        system_id="hop",
        ensemble="neb",
        temperature_k=None,
        condition_group_id="path-a",
        sampling_role_id="hop",
        selection_group_id="paths",
        kind="neb",
        neb=parameters,
    )

    assert task.neb == parameters
    assert task.steps is None
    assert task.program is None


def _coordinator_evidence(tmp_path: Path, models):
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingSeedSource

    return SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=SamplingPlan.create(
            (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
        ),
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=models,
        steps=20,
    )


def test_lifecycle_sampling_coordinator_prepares_before_native_execute(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import (
        SamplingExecutionRequest,
        SamplingExecutionResult,
    )
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    requested = OrderedSamplingModels((model,), member_ids=("student",))
    resolved = OrderedSamplingModels((model,), member_ids=("student",))
    events: list[str] = []

    class Provider:
        def prepare_sampling_models(self, request):
            events.append("prepare")
            return SamplingModelsResult(resolved)

        def load_sampling_models(self, request):
            raise AssertionError("fresh execution must not load models")

    class NativeOps:
        def execute(self, request):
            events.append("native_execute")
            assert request.evidence.models is resolved
            return SamplingExecutionResult(request.evidence.sample_dir, 1)

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )
    result = coordinator.execute(
        SamplingExecutionRequest(_coordinator_evidence(tmp_path, requested))
    )

    assert result.task_count == 1
    from workflow_core.lifecycle.sampling import SamplingExecutionStatus

    assert result.status is SamplingExecutionStatus.SUCCEEDED
    assert events == ["prepare", "native_execute"]


def test_lifecycle_sampling_coordinator_blocks_reordered_models_before_native_execute(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    first = tmp_path / "student_01.pb"
    second = tmp_path / "student_02.pb"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    requested = OrderedSamplingModels(
        (first, second), member_ids=("student_01", "student_02")
    )

    class Provider:
        def prepare_sampling_models(self, request):
            return SamplingModelsResult(
                OrderedSamplingModels(
                    (second, first), member_ids=("student_02", "student_01")
                )
            )

        def load_sampling_models(self, request):
            raise AssertionError("fresh execution must not load models")

    class NativeOps:
        def execute(self, request):
            raise AssertionError("invalid provider result reached native execute")

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )
    with pytest.raises(ValueError, match="ordered member identities"):
        coordinator.execute(
            SamplingExecutionRequest(_coordinator_evidence(tmp_path, requested))
        )

    assert not (_coordinator_evidence(tmp_path, requested).sample_dir / "tasks").exists()


def test_lifecycle_sampling_coordinator_preserves_native_execution_failure(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    requested = OrderedSamplingModels((model,), member_ids=("student",))
    events: list[str] = []

    class NativeFailure(RuntimeError):
        pass

    class Provider:
        def prepare_sampling_models(self, request):
            events.append("prepare")
            return SamplingModelsResult(request.models)

        def load_sampling_models(self, request):
            raise AssertionError("failed execution must not load models")

    class NativeOps:
        def execute(self, request):
            events.append("native_execute")
            raise NativeFailure("native sampling failed")

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )

    with pytest.raises(NativeFailure, match="native sampling failed"):
        coordinator.execute(
            SamplingExecutionRequest(_coordinator_evidence(tmp_path, requested))
        )
    assert events == ["prepare", "native_execute"]


def test_lifecycle_sampling_coordinator_loads_before_current_native_validation(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingValidationResult
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    requested = OrderedSamplingModels((model,), member_ids=("student",))
    resolved = OrderedSamplingModels((model,), member_ids=("student",))
    evidence = _coordinator_evidence(tmp_path, requested)
    (evidence.sample_dir / "tasks" / "task_000000").mkdir(parents=True)
    (evidence.sample_dir / "seeds.xyz").write_text("current", encoding="utf-8")
    (evidence.sample_dir / "tasks" / "task_000000" / "task_context.json").write_text(
        "{}", encoding="utf-8"
    )
    events: list[str] = []

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("completion validation must not prepare models")

        def load_sampling_models(self, request):
            events.append("load")
            return SamplingModelsResult(resolved)

    class NativeOps:
        def validate_current(self, request):
            events.append("native_validate")
            assert request.models is resolved
            return SamplingValidationResult(False)

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )
    result = coordinator.validate_completion(evidence)

    assert result == SamplingValidationResult(False)
    assert events == ["load", "native_validate"]


def test_lifecycle_sampling_coordinator_blocks_invalid_load_before_native_validation(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    first = tmp_path / "student_01.pb"
    second = tmp_path / "student_02.pb"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    requested = OrderedSamplingModels(
        (first, second), member_ids=("student_01", "student_02")
    )

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("completion must not prepare models")

        def load_sampling_models(self, request):
            return SamplingModelsResult(
                OrderedSamplingModels(
                    (second, first), member_ids=("student_02", "student_01")
                )
            )

    class NativeOps:
        def validate_current(self, request):
            raise AssertionError("invalid loaded models reached native validation")

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )
    evidence = _coordinator_evidence(tmp_path, requested)
    (evidence.sample_dir / "tasks" / "task_000000").mkdir(parents=True)
    (evidence.sample_dir / "seeds.xyz").write_text("current", encoding="utf-8")
    (evidence.sample_dir / "tasks" / "task_000000" / "task_context.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ordered member identities"):
        coordinator.validate_completion(evidence)


def test_lifecycle_sampling_coordinator_opens_current_without_provider_or_revalidation(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingObservationResult
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    evidence = _coordinator_evidence(
        tmp_path, OrderedSamplingModels((model,), member_ids=("student",))
    )
    (evidence.sample_dir / "tasks" / "task_000000").mkdir(parents=True)
    (evidence.sample_dir / "seeds.xyz").write_text("current", encoding="utf-8")
    (evidence.sample_dir / "tasks" / "task_000000" / "task_context.json").write_text(
        "{}", encoding="utf-8"
    )
    source = SimpleNamespace(catalog=lambda: ("strict",))
    events: list[str] = []

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("current open must not prepare models")

        def load_sampling_models(self, request):
            raise AssertionError("current open must not load models")

    class NativeOps:
        def validate_current(self, request):
            raise AssertionError("current open must not repeat native validation")

        def open_current(self, request):
            events.append("native_open")
            return SamplingObservationResult(source)

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )
    result = coordinator.open_observations(evidence)

    assert result.source.catalog() == ("strict",)
    assert events == ["native_open"]


def test_lifecycle_sampling_coordinator_validates_legacy_with_one_load_and_catalog(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import (
        SamplingObservationResult,
        SamplingValidationResult,
    )
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    requested = OrderedSamplingModels((model,), member_ids=("student",))
    resolved = OrderedSamplingModels((model,), member_ids=("student",))
    evidence = _coordinator_evidence(tmp_path, requested)
    events: list[str] = []

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("legacy completion must not prepare models")

        def load_sampling_models(self, request):
            events.append("load")
            return SamplingModelsResult(resolved)

    class Source:
        def catalog(self):
            events.append("catalog")
            return ("row",)

    class NativeOps:
        def validate_current(self, request):
            raise AssertionError("legacy completion must not use current validation")

        def reconstruct_compatible_legacy(self, request):
            events.append("native_reconstruct")
            assert request.models is resolved
            return SamplingObservationResult(Source(), historical=True)

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )

    assert coordinator.validate_completion(evidence) == SamplingValidationResult(True)
    assert events == ["load", "native_reconstruct", "catalog"]


def test_lifecycle_sampling_coordinator_returns_incomplete_for_partial_without_calls(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingValidationResult
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    evidence = _coordinator_evidence(
        tmp_path, OrderedSamplingModels((model,), member_ids=("student",))
    )
    (evidence.sample_dir / "tasks" / "task_000000").mkdir(parents=True)
    (evidence.sample_dir / "seeds.xyz").write_text("partial", encoding="utf-8")

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("partial completion reached provider prepare")

        def load_sampling_models(self, request):
            raise AssertionError("partial completion reached provider load")

    class NativeOps:
        def __getattr__(self, name):
            raise AssertionError(f"partial completion reached native {name}")

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )

    result = coordinator.validate_completion(evidence)
    assert result == SamplingValidationResult(False)
    from workflow_core.lifecycle.sampling import SamplingValidationStatus

    assert result.status is SamplingValidationStatus.INCOMPLETE


def test_lifecycle_sampling_coordinator_rejects_legacy_catalog_failure(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingObservationResult
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    evidence = _coordinator_evidence(
        tmp_path, OrderedSamplingModels((model,), member_ids=("student",))
    )
    loads = 0

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("legacy completion must not prepare models")

        def load_sampling_models(self, request):
            nonlocal loads
            loads += 1
            return SamplingModelsResult(request.models)

    class Source:
        def catalog(self):
            raise ValueError("bad catalog")

    class NativeOps:
        def reconstruct_compatible_legacy(self, request):
            return SamplingObservationResult(Source(), historical=True)

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )

    from workflow_core.lifecycle.sampling import (
        SamplingEvidenceFailure,
        SamplingEvidenceRebuildRequired,
    )

    with pytest.raises(
        SamplingEvidenceRebuildRequired,
        match=r"legacy sample evidence is incompatible; rebuild from r0/sample",
    ) as rejected:
        coordinator.validate_completion(evidence)
    assert rejected.value.failure is SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE
    assert rejected.value.resumable is True
    assert rejected.value.resume_block == "r0/sample"
    assert loads == 1


def test_lifecycle_sampling_coordinator_opens_legacy_with_one_load_per_call(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling import SamplingObservationResult
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    evidence = _coordinator_evidence(
        tmp_path, OrderedSamplingModels((model,), member_ids=("student",))
    )
    events: list[str] = []
    source = SimpleNamespace(catalog=lambda: ("strict",))

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("legacy open must not prepare models")

        def load_sampling_models(self, request):
            events.append("load")
            return SamplingModelsResult(request.models)

    class NativeOps:
        def open_current(self, request):
            raise AssertionError("legacy open must not use current source")

        def reconstruct_compatible_legacy(self, request):
            events.append("native_reconstruct")
            return SamplingObservationResult(source, historical=True)

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )

    first = coordinator.open_observations(evidence)
    second = coordinator.open_observations(evidence)

    assert first.historical is True and second.historical is True
    assert events == ["load", "native_reconstruct", "load", "native_reconstruct"]


def test_lifecycle_sampling_coordinator_rejects_partial_open_before_calls(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")
    evidence = _coordinator_evidence(
        tmp_path, OrderedSamplingModels((model,), member_ids=("student",))
    )
    (evidence.sample_dir / "tasks" / "task_000000").mkdir(parents=True)
    (evidence.sample_dir / "seeds.xyz").write_text("partial", encoding="utf-8")

    class Provider:
        def prepare_sampling_models(self, request):
            raise AssertionError("partial open reached provider prepare")

        def load_sampling_models(self, request):
            raise AssertionError("partial open reached provider load")

    class NativeOps:
        def __getattr__(self, name):
            raise AssertionError(f"partial open reached native {name}")

    coordinator = LifecycleSamplingCoordinator(
        NativeOps(), Provider(), _PathBackedEvidenceObserver()
    )

    from workflow_core.lifecycle.sampling import (
        SamplingEvidenceFailure,
        SamplingEvidenceRebuildRequired,
    )

    with pytest.raises(
        SamplingEvidenceRebuildRequired,
        match=r"sample artifact evidence is partial or ambiguous; rebuild from r0/sample",
    ) as rejected:
        coordinator.open_observations(evidence)
    assert rejected.value.failure is SamplingEvidenceFailure.PARTIAL_OR_AMBIGUOUS
    assert rejected.value.resumable is True
    assert rejected.value.resume_block == "r0/sample"


def test_neutral_sampling_model_contract_requires_complete_member_identity(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling_models import OrderedSamplingModels

    model = tmp_path / "student_model.pb"
    model.write_text("model", encoding="utf-8")

    for member_ids in ((), ("",), ("student", "student")):
        paths = (model, model) if len(member_ids) == 2 else (model,)
        try:
            OrderedSamplingModels(paths, member_ids=member_ids)
        except ValueError as exc:
            assert "identities" in str(exc)
        else:
            raise AssertionError(
                f"sampling-model inputs accepted invalid identities {member_ids!r}"
            )


def test_neutral_sampling_model_contract_requires_nonempty_regular_files(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling_models import OrderedSamplingModels

    empty = tmp_path / "empty.pb"
    empty.touch()
    directory = tmp_path / "directory.pb"
    directory.mkdir()
    target = tmp_path / "target.pb"
    target.write_text("model", encoding="utf-8")
    symlink = tmp_path / "symlink.pb"
    symlink.symlink_to(target)

    for invalid in (tmp_path / "missing.pb", empty, directory, symlink):
        try:
            OrderedSamplingModels((invalid,), member_ids=("student",))
        except ValueError as exc:
            assert "regular" in str(exc)
        else:
            raise AssertionError(f"sampling-model inputs accepted {invalid}")


def test_neutral_sampling_model_contract_rejects_reordered_provider_result(
    tmp_path: Path,
) -> None:
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsRequest,
        SamplingModelsResult,
        validate_sampling_models_result,
    )

    first = tmp_path / "student_01.pb"
    second = tmp_path / "student_02.pb"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    request = SamplingModelsRequest(
        models=OrderedSamplingModels(
            (first, second), member_ids=("student_01", "student_02")
        ),
        sample_dir=tmp_path / "sample",
    )
    result = SamplingModelsResult(
        OrderedSamplingModels(
            (second, first), member_ids=("student_02", "student_01")
        )
    )

    try:
        validate_sampling_models_result(request, result)
    except ValueError as exc:
        assert "ordered member identities" in str(exc)
    else:
        raise AssertionError("sampling-model provider result changed request order")


def test_sampling_model_contract_has_logic_owned_capability_and_backend_adapter_value() -> None:
    from workflow_backends import sampling
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_models import (
        SamplingModelInputs,
        SamplingModelProvider,
        SamplingModelsResult,
    )

    owned = (
        SamplingModelInputs,
        SamplingModelsResult,
        SamplingModelProvider,
    )
    assert {item.__module__ for item in owned} == {
        "workflow_core.lifecycle.sampling_models"
    }
    assert OrderedSamplingModels.__module__ == "workflow_backends.sampling_models"
    assert set(sampling.__all__).isdisjoint(
        {"SamplingModelInputs", "OrderedSamplingModels"}
    )


def test_sampling_capability_contract_is_logic_owned_and_path_free() -> None:
    from workflow_backends.sampling import LifecycleSampling as CompatibilitySampling
    from workflow_core.lifecycle import sampling

    assert CompatibilitySampling is sampling.LifecycleSampling
    source = inspect.getsource(sampling)
    assert "pathlib" not in source
    assert "from pathlib import Path" not in source
    assert "workflow_backends" not in source
    assert "workflow_runtime" not in source
    assert set(sampling.SamplingEvidence.__dict__).isdisjoint(
        {"sample_dir", "run_json_path", "workspace", "path"}
    )


def test_sampling_evidence_classification_is_logic_owned_and_closed() -> None:
    from workflow_core.lifecycle.sampling import (
        SamplingEvidenceFacts,
        SamplingEvidenceState,
        classify_sampling_evidence,
    )

    assert classify_sampling_evidence(
        SamplingEvidenceFacts(True, (True, True))
    ) is SamplingEvidenceState.CURRENT
    assert classify_sampling_evidence(
        SamplingEvidenceFacts(False, (False, False))
    ) is SamplingEvidenceState.LEGACY_COMPATIBLE
    assert classify_sampling_evidence(
        SamplingEvidenceFacts(True, (True, False))
    ) is SamplingEvidenceState.REBUILD_REQUIRED
    assert classify_sampling_evidence(
        SamplingEvidenceFacts(False, (True, False))
    ) is SamplingEvidenceState.REBUILD_REQUIRED


def test_sampling_evidence_request_uses_shared_model_input_validation(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.sampling import SamplingEvidenceRequest, SamplingSeedSource

    with pytest.raises(ValueError, match="regular"):
        SamplingEvidenceRequest(
            run_json_path=tmp_path / "run.json",
            iter_index=0,
            round_index=0,
            sample_dir=tmp_path / "sample",
            plan=SamplingPlan.create(
                (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
            ),
            systems=(SamplingSeedSource("water", (Atoms("H"),)),),
            models=SimpleNamespace(
                paths=(tmp_path / "missing.pb",),
                member_ids=("student",),
            ),
            steps=20,
        )


def test_sampling_backend_interface_is_typed_and_backend_neutral() -> None:
    from workflow_backends.sampling import SamplingEvidenceRequest
    from workflow_core.lifecycle.sampling import LifecycleSampling

    for operation in (
        LifecycleSampling.execute,
        LifecycleSampling.open_observations,
        LifecycleSampling.validate_completion,
    ):
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(operation).parameters.values()
        )
    assert set(SamplingEvidenceRequest.__dataclass_fields__).isdisjoint(
        {
            "backend",
            "command",
            "compression",
            "deepmd_recipe_kind",
            "marker",
            "quota",
            "selector",
        }
    )
    assert "reconstruct_historical" not in LifecycleSampling.__dict__


def test_lifecycle_runner_delegates_sample_evidence_transitions_to_coordinator() -> None:
    from workflow_core.lifecycle import runner

    open_source = inspect.getsource(runner._sample_observation_source)
    completion_source = inspect.getsource(runner._valid_sample_artifacts)
    runner_source = inspect.getsource(runner)

    assert "_sampling_evidence_mode" not in runner_source
    assert "reconstruct_historical" not in runner_source
    assert "open_observations(intent)" in open_source
    assert "validate_completion(intent)" in completion_source


def test_lammps_sampling_factory_binds_native_ops_without_a_provider(
    tmp_path: Path,
) -> None:
    from workflow_backends.lammps.adapter import (
        LammpsSamplingBackendFactory,
        LammpsSamplingNativeOps,
    )
    from workflow_backends.sampling import SamplingBackendBinding, SamplingRecipeSource

    recipe = tmp_path / "sample.in"
    recipe.write_text("run ${dice_run_steps}\n", encoding="utf-8")
    factory = LammpsSamplingBackendFactory(_FakeLammpsExecutor())
    binding = SamplingBackendBinding(
        recipes=(SamplingRecipeSource(recipe),),
        type_map=("H",),
        system_inputs={},
    )

    native_ops = factory.bind(binding)

    assert isinstance(native_ops, LammpsSamplingNativeOps)
    assert "model_provider" not in native_ops.__dataclass_fields__


def test_gpumd_sampling_factory_binds_native_ops_without_a_provider(
    tmp_path: Path,
) -> None:
    from workflow_backends.gpumd.adapter import (
        GpumdSamplingBackendFactory,
        GpumdSamplingNativeOps,
    )
    from workflow_backends.sampling import SamplingBackendBinding, SamplingRecipeSource

    recipe = tmp_path / "run.in"
    recipe.write_text("run 10\n", encoding="utf-8")
    factory = GpumdSamplingBackendFactory(_FakeGpumdExecutor())
    binding = SamplingBackendBinding(
        recipes=(SamplingRecipeSource(recipe),),
        type_map=("H",),
        system_inputs={},
    )

    native_ops = factory.bind(binding)

    assert isinstance(native_ops, GpumdSamplingNativeOps)
    assert "model_provider" not in native_ops.__dataclass_fields__


def test_lammps_sampling_backend_executes_validates_and_replays_through_one_seam(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\n"
        "dump native all custom 20 native.xyz id type x y z\n"
        "run ${dice_run_steps}\n",
        encoding="utf-8",
    )
    first = tmp_path / "student_01.pb"
    second = tmp_path / "student_02.pb"
    first.write_text("member-one", encoding="utf-8")
    second.write_text("member-two", encoding="utf-8")
    plan = SamplingPlan.create(
        (SystemSamplingCondition("water", "nvt", (300.0, 500.0), ()),)
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=1,
        sample_dir=tmp_path / "iter0" / "r1" / "sample",
        plan=plan,
        systems=(
            SamplingSeedSource(
                "water",
                (Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),),
            ),
        ),
        models=OrderedSamplingModels(
            (first, second), member_ids=("student_01", "student_02")
        ),
        steps=40,
    )
    snapshots = []
    provider = _IdentitySamplingModelProvider()
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            recipe=recipe,
        ),
        provider,
    )

    result = backend.execute(
        SamplingExecutionRequest(evidence=evidence, progress_callback=snapshots.append)
    )

    assert result.sample_dir == evidence.sample_dir
    assert result.task_count == 2
    tasks = tuple(sorted((evidence.sample_dir / "tasks").iterdir()))
    assert tuple(task.name for task in tasks) == ("task_000000", "task_000001")
    assert tuple(
        (task / name).read_text(encoding="utf-8")
        for task in tasks[:1]
        for name in ("graph_000.pb", "graph_001.pb")
    ) == ("member-one", "member-two")
    assert all((task / "traj.lammpstrj").is_file() for task in tasks)
    assert all((task / "model_devi.out").is_file() for task in tasks)
    assert snapshots[-1].completed_tasks == 2
    assert provider.calls == ["prepare"]
    assert backend.validate_completion(evidence).complete is True
    assert provider.calls == ["prepare", "load"]

    observation = backend.open_observations(evidence)
    assert provider.calls == ["prepare", "load"]
    catalog = observation.source.catalog()
    assert tuple(row.task_index for row in catalog) == (0, 1)
    assert tuple(row.student_uncertainty for row in catalog) == (0.2, 0.2)
    requested = (catalog[1].structure_id, catalog[0].structure_id)
    chunks = tuple(observation.source.replay(requested, chunk_size=1))
    assert tuple(len(chunk) for chunk in chunks) == (1, 1)
    assert tuple(frame.info["structure_id"] for chunk in chunks for frame in chunk) == requested


def test_lammps_coordinator_preserves_effective_sampling_model_plan(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_artifacts.lifecycle.sampling_models import (
        EffectiveSamplingModel,
        EffectiveSamplingModelPlan,
        file_sha256,
    )
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    model = tmp_path / "student_model.pt2"
    model.write_bytes(b"dpa4c")
    plan = EffectiveSamplingModelPlan(
        (
            EffectiveSamplingModel(
                member_id="student",
                source_path=model,
                source_sha256=file_sha256(model),
                effective_path=model,
                effective_sha256=file_sha256(model),
                deepmd_recipe_kind="dpa4c",
                cli="pt",
            ),
        ),
        compressed=False,
    )

    class Provider:
        def prepare_sampling_models(self, request):
            return SamplingModelsResult(plan)

        def load_sampling_models(self, request):
            raise AssertionError("fresh execution must not load models")

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=SamplingPlan.create(
            (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
        ),
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=20,
    )
    coordinator = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(), type_map=("H",), recipe=recipe
        ),
        Provider(),
    )

    coordinator.execute(SamplingExecutionRequest(evidence))

    receipt = json.loads(
        (
            evidence.sample_dir
            / "tasks"
            / "task_000000"
            / "native_execution.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["schema_revision"] == 2
    assert receipt["pair_style"] == "deepmd/kk"


def test_lammps_explicit_execution_validates_only_round_local_roles_and_recipes(
    tmp_path: Path,
) -> None:
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_plan import (
        ExplicitSamplingPlan,
        SamplingConditionGroup,
        SamplingRound,
        SeedRole,
    )

    active_recipe = tmp_path / "active.lammps"
    active_recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    inactive_recipe = tmp_path / "inactive-missing.lammps"
    model = tmp_path / "student.pb"
    model.write_bytes(b"model")
    plan = ExplicitSamplingPlan(
        seeds=(
            SeedRole("water-a", "shared"),
            SeedRole("water-b", "shared"),
            SeedRole("future", "future"),
        ),
        condition_groups=(
            SamplingConditionGroup(
                "active",
                "nvt",
                ("water-a", "water-b"),
                (20, 20),
                temperatures_k=(100.0,),
            ),
            SamplingConditionGroup(
                "future",
                "nvt",
                ("future",),
                (20, 20),
                temperatures_k=(200.0,),
            ),
        ),
        round_count=2,
        rounds=(
            SamplingRound(("water-a", "water-b"), ("active",)),
            SamplingRound(("future",), ("future",)),
        ),
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=tuple(
            SamplingSeedSource(
                role_id,
                (Atoms("H", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True),),
            )
            for role_id in ("water-a", "water-b")
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )

    result = LammpsSamplingNativeOps(
        executor=_FakeLammpsExecutor(),
        type_map=("H",),
        group_recipes={"active": active_recipe, "future": inactive_recipe},
    ).execute(SamplingExecutionRequest(evidence))

    assert result.task_count == 2
    assert tuple(
        json.loads((task / "task_context.json").read_text(encoding="utf-8"))[
            "sampling_role_id"
        ]
        for task in sorted((evidence.sample_dir / "tasks").iterdir())
    ) == ("water-a", "water-b")


def test_gpumd_sampling_backend_executes_validates_and_replays_through_same_seam(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.gpumd.adapter import GpumdSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    recipe = tmp_path / "run.in"
    recipe.write_text(
        "dump_position 20\n"
        "velocity 300\n"
        "ensemble nvt_ber 300 300 100\n"
        "run 10\n",
        encoding="utf-8",
    )
    first = tmp_path / "student_01.txt"
    second = tmp_path / "student_02.txt"
    first.write_text("member-one", encoding="utf-8")
    second.write_text("member-two", encoding="utf-8")
    plan = SamplingPlan.create(
        (SystemSamplingCondition("water", "nvt", (300.0, 500.0), ()),)
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=1,
        sample_dir=tmp_path / "iter0" / "r1" / "sample",
        plan=plan,
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=OrderedSamplingModels(
            (first, second), member_ids=("student_01", "student_02")
        ),
        steps=40,
    )
    snapshots = []
    provider = _IdentitySamplingModelProvider()
    backend = LifecycleSamplingCoordinator(
        GpumdSamplingNativeOps(
            executor=_FakeGpumdExecutor(),
            recipe=recipe,
        ),
        provider,
    )

    result = backend.execute(
        SamplingExecutionRequest(evidence=evidence, progress_callback=snapshots.append)
    )

    assert result.sample_dir == evidence.sample_dir
    assert result.task_count == 2
    tasks = tuple(sorted((evidence.sample_dir / "tasks").iterdir()))
    assert tuple(task.name for task in tasks) == ("task_000000", "task_000001")
    assert tuple(
        (tasks[0] / name).read_text(encoding="utf-8")
        for name in ("nep_0.txt", "nep_1.txt")
    ) == ("member-one", "member-two")
    assert all((task / "active.xyz").is_file() for task in tasks)
    assert all((task / "active.out").is_file() for task in tasks)
    assert snapshots[-1].completed_tasks == 2
    assert provider.calls == ["prepare"]
    assert backend.validate_completion(evidence).complete is True
    assert provider.calls == ["prepare", "load"]

    observation = backend.open_observations(evidence)
    assert provider.calls == ["prepare", "load"]
    catalog = observation.source.catalog()
    assert tuple(row.task_index for row in catalog) == (0, 1)
    assert tuple(row.student_uncertainty for row in catalog) == (0.3, 0.3)
    requested = (catalog[1].structure_id, catalog[0].structure_id)
    chunks = tuple(observation.source.replay(requested, chunk_size=1))
    assert tuple(len(chunk) for chunk in chunks) == (1, 1)
    assert tuple(frame.info["structure_id"] for chunk in chunks for frame in chunk) == requested


def test_lammps_sampling_backend_fails_closed_on_invalid_committed_evidence(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    first = tmp_path / "student_01.pb"
    second = tmp_path / "student_02.pb"
    first.write_text("member-one", encoding="utf-8")
    second.write_text("member-two", encoding="utf-8")
    plan = SamplingPlan.create(
        (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
    )
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            recipe=recipe,
        ),
        _IdentitySamplingModelProvider(),
    )

    for mutation in ("partial", "reordered", "malformed", "identity"):
        evidence = SamplingEvidenceRequest(
            run_json_path=tmp_path / "run.json",
            iter_index=0,
            round_index=0,
            sample_dir=tmp_path / mutation / "sample",
            plan=plan,
            systems=(SamplingSeedSource("water", (Atoms("H"),)),),
            models=OrderedSamplingModels(
                (first, second), member_ids=("student_01", "student_02")
            ),
            steps=20,
        )
        backend.execute(SamplingExecutionRequest(evidence))
        task = evidence.sample_dir / "tasks" / "task_000000"
        if mutation == "partial":
            (task / "traj.lammpstrj").unlink()
        elif mutation == "reordered":
            (task / "graph_000.pb").unlink()
            (task / "graph_000.pb").write_text("member-two", encoding="utf-8")
        else:
            context_path = task / "task_context.json"
            if mutation == "malformed":
                context_path.write_text("{", encoding="utf-8")
            else:
                context = json.loads(context_path.read_text(encoding="utf-8"))
                context["selected_seed_id"] = "sha256:wrong"
                context_path.write_text(json.dumps(context), encoding="utf-8")

        assert backend.validate_completion(evidence).complete is False, mutation


def test_lammps_sampling_backend_rejects_reordered_provider_result_before_rendering(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    first = tmp_path / "student_01.pb"
    second = tmp_path / "student_02.pb"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    class _ReorderingProvider:
        def prepare_sampling_models(self, request):
            return SamplingModelsResult(
                OrderedSamplingModels(
                    (second, first),
                    member_ids=("student_02", "student_01"),
                )
            )

        def load_sampling_models(self, request):
            raise AssertionError("fresh execution must not load models")

    recipe = tmp_path / "sample.in"
    recipe.write_text("run ${dice_run_steps}\n", encoding="utf-8")
    sample_dir = tmp_path / "sample"
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=sample_dir,
        plan=SamplingPlan.create(
            (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
        ),
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=OrderedSamplingModels(
            (first, second), member_ids=("student_01", "student_02")
        ),
        steps=20,
    )
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            recipe=recipe,
        ),
        _ReorderingProvider(),
    )

    with pytest.raises(ValueError, match="ordered member identities"):
        backend.execute(SamplingExecutionRequest(evidence))

    assert not (sample_dir / "tasks").exists()


def test_gpumd_sampling_backend_fails_closed_on_invalid_committed_evidence(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.gpumd.adapter import GpumdSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    recipe = tmp_path / "run.in"
    recipe.write_text(
        "velocity 300\nensemble nvt_ber 300 300 100\nrun 10\n",
        encoding="utf-8",
    )
    first = tmp_path / "student_01.txt"
    second = tmp_path / "student_02.txt"
    first.write_text("member-one", encoding="utf-8")
    second.write_text("member-two", encoding="utf-8")
    plan = SamplingPlan.create(
        (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
    )
    backend = LifecycleSamplingCoordinator(
        GpumdSamplingNativeOps(
            executor=_FakeGpumdExecutor(),
            recipe=recipe,
        ),
        _IdentitySamplingModelProvider(),
    )

    for mutation in ("partial", "reordered", "malformed", "identity"):
        evidence = SamplingEvidenceRequest(
            run_json_path=tmp_path / "run.json",
            iter_index=0,
            round_index=0,
            sample_dir=tmp_path / mutation / "sample",
            plan=plan,
            systems=(SamplingSeedSource("water", (Atoms("H"),)),),
            models=OrderedSamplingModels(
                (first, second), member_ids=("student_01", "student_02")
            ),
            steps=20,
        )
        backend.execute(SamplingExecutionRequest(evidence))
        task = evidence.sample_dir / "tasks" / "task_000000"
        if mutation == "partial":
            (task / "active.xyz").unlink()
        elif mutation == "reordered":
            (task / "nep_0.txt").unlink()
            (task / "nep_0.txt").write_text("member-two", encoding="utf-8")
        else:
            context_path = task / "task_context.json"
            if mutation == "malformed":
                context_path.write_text("{", encoding="utf-8")
            else:
                context = json.loads(context_path.read_text(encoding="utf-8"))
                context["selected_seed_id"] = "sha256:wrong"
                context_path.write_text(json.dumps(context), encoding="utf-8")

        assert backend.validate_completion(evidence).complete is False, mutation


def test_gpumd_explicit_md_preserves_global_task_identity(tmp_path: Path) -> None:
    from workflow_backends.gpumd.adapter import GpumdSamplingNativeOps
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

    recipe = tmp_path / "run.in"
    recipe.write_text(
        "velocity 300\nensemble nvt_ber 300 300 100\nrun 10\n",
        encoding="utf-8",
    )
    model = tmp_path / "student.txt"
    model.write_text("member", encoding="utf-8")
    endpoint_frames = (
        Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "all"), SeedRole("md", "all")),
        condition_groups=(
            SamplingConditionGroup(
                "neb", "neb", ("hop",),
                neb=NebSamplingParameters(1, 0.05, 5),
            ),
            SamplingConditionGroup(
                "md", "nvt", ("md",),
                temperatures_k=(300.0,), steps_by_round=(20,),
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
            SamplingSeedSource("hop", endpoint_frames),
            SamplingSeedSource("md", (Atoms("H"),)),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = GpumdSamplingNativeOps(
        executor=_FakeGpumdExecutor(), group_recipes={"md": recipe}
    )

    result = backend.execute(SamplingExecutionRequest(evidence))

    assert result.task_count == 1
    assert not (evidence.sample_dir / "tasks" / "task_000000").exists()
    assert (evidence.sample_dir / "tasks" / "task_000001" / "active.xyz").is_file()


def test_lammps_explicit_md_preserves_global_task_identity(tmp_path: Path) -> None:
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
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

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    model = tmp_path / "student.pb"
    model.write_text("member", encoding="utf-8")
    endpoints = (
        Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True),
        Atoms("H", positions=[[1, 0, 0]], cell=[5, 5, 5], pbc=True),
    )
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "all"), SeedRole("md", "all")),
        condition_groups=(
            SamplingConditionGroup(
                "neb", "neb", ("hop",),
                neb=NebSamplingParameters(1, 0.05, 5),
            ),
            SamplingConditionGroup(
                "md", "nvt", ("md",),
                steps_by_round=(20,), temperatures_k=(300.0,),
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
    backend = LammpsSamplingNativeOps(
        executor=_FakeLammpsExecutor(),
        type_map=("H",),
        group_recipes={"md": recipe},
    )

    result = backend.execute(SamplingExecutionRequest(evidence))

    assert result.task_count == 1
    assert not (evidence.sample_dir / "tasks" / "task_000000").exists()
    assert (evidence.sample_dir / "tasks" / "task_000001" / "traj.lammpstrj").is_file()


def test_mixed_lammps_neb_executes_and_replays_in_declaration_order(
    tmp_path: Path,
) -> None:
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_backends.hybrid_sampling import HybridSamplingNativeOps
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
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

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = np.asarray(self.atoms.positions, dtype=float)
            self.results = {
                "energy": float(np.sum(positions**2)),
                "forces": -2.0 * positions,
            }

    class Provider:
        policy_identity = {"name": "committee-force-deviation", "revision": 1}

        def preflight(self, model_paths):
            assert model_paths

        def create(self, model_path):
            assert model_path.is_file()
            return Harmonic()

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    model = tmp_path / "student.pb"
    model.write_text("member", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "joint"), SeedRole("md", "joint")),
        condition_groups=(
            SamplingConditionGroup(
                "neb", "neb", ("hop",),
                neb=NebSamplingParameters(2, 0.05, 5),
            ),
            SamplingConditionGroup(
                "md", "nvt", ("md",),
                steps_by_round=(20,), temperatures_k=(300.0,),
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
            SamplingSeedSource("md", (Atoms("H"),)),
        ),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = HybridSamplingNativeOps(
        md=LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            group_recipes={"md": recipe},
        ),
        neb=AseNebSamplingNativeOps(
            calculator_provider=Provider(),
            model_family="deepmd-tf",
            operation_role="student.deepmd.tf.neb",
            operation_profile_id="test",
        ),
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    source = backend.open_current(evidence).source
    catalog = source.catalog()

    assert result.task_count == 2
    assert tuple(path.name for path in sorted((evidence.sample_dir / "tasks").iterdir())) == (
        "task_000000", "task_000001"
    )
    assert [record.task_index for record in catalog] == [0, 0, 0, 0, 1]
    assert backend.validate_current(evidence).complete is True


def test_hybrid_sampling_executes_and_replays_neb_only_round(
    tmp_path: Path,
) -> None:
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_backends.hybrid_sampling import HybridSamplingNativeOps
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
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
        SamplingRound,
        SeedRole,
    )

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = np.asarray(self.atoms.positions, dtype=float)
            self.results = {
                "energy": float(np.sum(positions**2)),
                "forces": -2.0 * positions,
            }

    class Provider:
        policy_identity = {"name": "committee-force-deviation", "revision": 1}

        def preflight(self, model_paths):
            assert model_paths

        def create(self, model_path):
            assert model_path.is_file()
            return Harmonic()

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    model = tmp_path / "student.pb"
    model.write_text("member", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("md", "joint"), SeedRole("hop", "joint")),
        condition_groups=(
            SamplingConditionGroup(
                "md", "nvt", ("md",),
                steps_by_round=(20, 20), temperatures_k=(300.0,),
            ),
            SamplingConditionGroup(
                "neb", "neb", ("hop",),
                neb=NebSamplingParameters(2, 0.05, 5),
            ),
        ),
        round_count=2,
        rounds=(
            SamplingRound(("md",), ("md",)),
            SamplingRound(("hop",), ("neb",)),
        ),
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=1,
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
    backend = HybridSamplingNativeOps(
        md=LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            group_recipes={"md": recipe},
        ),
        neb=AseNebSamplingNativeOps(
            calculator_provider=Provider(),
            model_family="deepmd-tf",
            operation_role="student.deepmd.tf.neb",
            operation_profile_id="test",
        ),
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    source = backend.open_current(evidence).source

    assert result.task_count == 1
    assert len(source.catalog()) == 4
    assert backend.validate_current(evidence).complete is True


def test_hybrid_sampling_executes_and_replays_md_only_round(
    tmp_path: Path,
) -> None:
    from workflow_backends.ase_neb.adapter import AseNebSamplingNativeOps
    from workflow_backends.hybrid_sampling import HybridSamplingNativeOps
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
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
        SamplingRound,
        SeedRole,
    )

    class InactiveNebProvider:
        policy_identity = {"name": "committee-force-deviation", "revision": 1}

        def preflight(self, model_paths):
            raise AssertionError("inactive NEB branch must not be preflighted")

        def create(self, model_path):
            raise AssertionError("inactive NEB branch must not create calculators")

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    model = tmp_path / "student.pb"
    model.write_text("member", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("hop", "joint"), SeedRole("md", "joint")),
        condition_groups=(
            SamplingConditionGroup(
                "neb", "neb", ("hop",),
                neb=NebSamplingParameters(2, 0.05, 5),
            ),
            SamplingConditionGroup(
                "md", "nvt", ("md",),
                steps_by_round=(20, 20), temperatures_k=(300.0,),
            ),
        ),
        round_count=2,
        rounds=(
            SamplingRound(("hop",), ("neb",)),
            SamplingRound(("md",), ("md",)),
        ),
    )
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=1,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("md", (Atoms("H", cell=[4, 4, 4], pbc=True),)),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = HybridSamplingNativeOps(
        md=LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            group_recipes={"md": recipe},
        ),
        neb=AseNebSamplingNativeOps(
            calculator_provider=InactiveNebProvider(),
            model_family="deepmd-tf",
            operation_role="student.deepmd.tf.neb",
            operation_profile_id="test",
        ),
    )

    result = backend.execute(SamplingExecutionRequest(evidence))
    source = backend.open_current(evidence).source

    assert result.task_count == 1
    assert len(source.catalog()) == 1
    assert backend.validate_current(evidence) == backend.md.validate_current(evidence)


def test_mixed_sampling_legacy_recovery_is_typed_rebuild_required(
    tmp_path: Path,
) -> None:
    from workflow_backends.hybrid_sampling import HybridSamplingNativeOps
    from workflow_backends.sampling import SamplingEvidenceRequest
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_backends.sampling import SamplingSeedSource
    from workflow_core.common.sampling_plan import (
        ExplicitSamplingPlan,
        SamplingConditionGroup,
        SeedRole,
    )
    from workflow_core.lifecycle.sampling import (
        SamplingEvidenceFailure,
        SamplingEvidenceRebuildRequired,
    )

    model = tmp_path / "model"
    model.write_text("member", encoding="utf-8")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("md", "all"),),
        condition_groups=(
            SamplingConditionGroup(
                "md-a",
                "nvt",
                ("md",),
                temperatures_k=(300.0,),
                steps_by_round=(10, 10, 10),
            ),
        ),
        round_count=3,
    )
    request = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=2,
        sample_dir=tmp_path / "sample",
        plan=plan,
        systems=(SamplingSeedSource("md", (Atoms("H"),)),),
        models=OrderedSamplingModels((model,), member_ids=("student",)),
        steps=None,
    )
    backend = HybridSamplingNativeOps(md=object(), neb=object())

    with pytest.raises(SamplingEvidenceRebuildRequired) as caught:
        backend.reconstruct_compatible_legacy(request)

    assert caught.value.failure is SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE
    assert caught.value.resume_block == "r2/sample"


def test_gpumd_sampling_backend_rejects_reordered_provider_result_before_rendering(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.gpumd.adapter import GpumdSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import (
        OrderedSamplingModels,
        SamplingModelsResult,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    first = tmp_path / "student_01.txt"
    second = tmp_path / "student_02.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    class _ReorderingProvider:
        def prepare_sampling_models(self, request):
            return SamplingModelsResult(
                OrderedSamplingModels(
                    (second, first),
                    member_ids=("student_02", "student_01"),
                )
            )

        def load_sampling_models(self, request):
            raise AssertionError("fresh execution must not load models")

    recipe = tmp_path / "run.in"
    recipe.write_text(
        "velocity 300\nensemble nvt_ber 300 300 100\nrun 10\n",
        encoding="utf-8",
    )
    sample_dir = tmp_path / "sample"
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=sample_dir,
        plan=SamplingPlan.create(
            (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
        ),
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=OrderedSamplingModels(
            (first, second), member_ids=("student_01", "student_02")
        ),
        steps=20,
    )
    backend = LifecycleSamplingCoordinator(
        GpumdSamplingNativeOps(
            executor=_FakeGpumdExecutor(),
            recipe=recipe,
        ),
        _ReorderingProvider(),
    )

    with pytest.raises(ValueError, match="ordered member identities"):
        backend.execute(SamplingExecutionRequest(evidence))

    assert not (sample_dir / "tasks").exists()


def test_lammps_sampling_backend_reconstructs_compatible_history_read_only(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    recipe = tmp_path / "sample.in"
    recipe.write_text(
        "include dice_task_setup.lammps\nrun ${dice_run_steps}\n",
        encoding="utf-8",
    )
    models = []
    for index in (1, 2):
        model = tmp_path / f"student_{index:02d}.pb"
        model.write_text(f"member-{index}", encoding="utf-8")
        models.append(model)
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=SamplingPlan.create(
            (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
        ),
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=OrderedSamplingModels(
            tuple(models), member_ids=("student_01", "student_02")
        ),
        steps=20,
    )
    provider = _IdentitySamplingModelProvider()
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(
            executor=_FakeLammpsExecutor(),
            type_map=("H",),
            recipe=recipe,
        ),
        provider,
    )
    backend.execute(SamplingExecutionRequest(evidence))
    (evidence.sample_dir / "seeds.xyz").unlink()
    for context in evidence.sample_dir.glob("tasks/*/task_context.json"):
        context.unlink()
    before = {
        path.relative_to(evidence.sample_dir): path.read_bytes()
        for path in evidence.sample_dir.rglob("*")
        if path.is_file()
    }

    historical = backend.open_observations(evidence)
    assert provider.calls == ["prepare", "load"]
    catalog = historical.source.catalog()
    replayed = tuple(
        frame
        for chunk in historical.source.replay(
            tuple(row.structure_id for row in catalog), chunk_size=1
        )
        for frame in chunk
    )

    assert historical.historical is True
    assert tuple(frame.info["structure_id"] for frame in replayed) == tuple(
        row.structure_id for row in catalog
    )
    after = {
        path.relative_to(evidence.sample_dir): path.read_bytes()
        for path in evidence.sample_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_gpumd_sampling_backend_reconstructs_compatible_history_read_only(
    tmp_path: Path,
) -> None:
    from workflow_artifacts.lifecycle.sample_conditions import (
        SamplingPlan,
        SystemSamplingCondition,
    )
    from workflow_backends.gpumd.adapter import GpumdSamplingNativeOps
    from workflow_backends.sampling import (
        SamplingEvidenceRequest,
        SamplingExecutionRequest,
        SamplingSeedSource,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import (
        LifecycleSamplingCoordinator,
    )

    recipe = tmp_path / "run.in"
    recipe.write_text(
        "velocity 300\nensemble nvt_ber 300 300 100\nrun 10\n",
        encoding="utf-8",
    )
    models = []
    for index in (1, 2):
        model = tmp_path / f"student_{index:02d}.txt"
        model.write_text(f"member-{index}", encoding="utf-8")
        models.append(model)
    evidence = SamplingEvidenceRequest(
        run_json_path=tmp_path / "run.json",
        iter_index=0,
        round_index=0,
        sample_dir=tmp_path / "sample",
        plan=SamplingPlan.create(
            (SystemSamplingCondition("water", "nvt", (300.0,), ()),)
        ),
        systems=(SamplingSeedSource("water", (Atoms("H"),)),),
        models=OrderedSamplingModels(
            tuple(models), member_ids=("student_01", "student_02")
        ),
        steps=20,
    )
    provider = _IdentitySamplingModelProvider()
    backend = LifecycleSamplingCoordinator(
        GpumdSamplingNativeOps(
            executor=_FakeGpumdExecutor(),
            recipe=recipe,
        ),
        provider,
    )
    backend.execute(SamplingExecutionRequest(evidence))
    (evidence.sample_dir / "seeds.xyz").unlink()
    for context in evidence.sample_dir.glob("tasks/*/task_context.json"):
        context.unlink()
    before = {
        path.relative_to(evidence.sample_dir): path.read_bytes()
        for path in evidence.sample_dir.rglob("*")
        if path.is_file()
    }

    historical = backend.open_observations(evidence)
    assert provider.calls == ["prepare", "load"]
    catalog = historical.source.catalog()
    replayed = tuple(
        frame
        for chunk in historical.source.replay(
            tuple(row.structure_id for row in catalog), chunk_size=1
        )
        for frame in chunk
    )

    assert historical.historical is True
    assert tuple(frame.info["structure_id"] for frame in replayed) == tuple(
        row.structure_id for row in catalog
    )
    after = {
        path.relative_to(evidence.sample_dir): path.read_bytes()
        for path in evidence.sample_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_lammps_sampling_assembles_protocol_without_user_setup_include(tmp_path: Path) -> None:
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator

    recipe = tmp_path / "protocol.lammps"
    protocol = (
        "units metal\natom_style atomic\nboundary p p p\n"
        "variable dice_trajectory_every_steps equal 20\n"
        "group mobile type 1\n"
        "timestep 0.001\n"
        "velocity mobile create ${dice_temperature} ${dice_seed}\n"
        "fix thermostat mobile nvt temp ${dice_temperature} ${dice_temperature} 0.1\n"
        "run ${dice_run_steps}\n"
    )
    recipe.write_text(protocol, encoding="utf-8")
    model = tmp_path / "student.pb"
    model.write_text("model", encoding="utf-8")
    evidence = _coordinator_evidence(tmp_path, OrderedSamplingModels((model,), member_ids=("student",)))
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(), type_map=("H",), recipe=recipe),
        _IdentitySamplingModelProvider(),
    )

    result = backend.execute(SamplingExecutionRequest(evidence))

    rendered = (result.sample_dir / "tasks/task_000000/input.lammps").read_text()
    assert rendered.count("include dice_task_setup.lammps") == 1
    assert rendered.index("boundary p p p") < rendered.index("include dice_task_setup.lammps")
    assert rendered.index("include dice_task_setup.lammps") < rendered.index("group mobile type 1")
    assert protocol == recipe.read_text()
    assert backend.validate_completion(evidence).complete is True


@pytest.mark.parametrize("command", [
    "read_data other.lmp", "read_restart saved.restart", "create_box 1 box",
    "pair_style lj/cut 2.5", "pair_coeff * * 1 1", "mass 1 99",
    "dump 1 all custom 5 other.dump id type x y z",
    "dump extra all custom 5 traj.lammpstrj id type x y z",
    "dump_modify 1 sort no", "undump 1", "clear", "reset_timestep 0",
    "include unknown.lammps", "variable dice_temperature equal 999",
    "group mobile type 1\nboundary f f f",
])
def test_lammps_automatic_protocol_rejects_conflicting_setup_before_execution(
    tmp_path: Path, command: str,
) -> None:
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator

    recipe = tmp_path / "protocol.lammps"
    recipe.write_text(f"units metal\natom_style atomic\n{command}\nrun ${{dice_run_steps}}\n")
    model = tmp_path / "student.pb"
    model.write_text("model")
    evidence = _coordinator_evidence(tmp_path, OrderedSamplingModels((model,), member_ids=("student",)))
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(), type_map=("H",), recipe=recipe),
        _IdentitySamplingModelProvider(),
    )
    with pytest.raises(ValueError, match="automatic LAMMPS protocol"):
        backend.execute(SamplingExecutionRequest(evidence))
    assert not (evidence.sample_dir / "tasks").exists()


@pytest.mark.parametrize("kind", ["nvt", "npt"])
def test_explicit_lammps_automatic_protocol_preserves_completion_and_receipts(tmp_path: Path, kind: str) -> None:
    from dataclasses import replace
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
    from workflow_core.lifecycle.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole

    recipe = tmp_path / "protocol.lammps"
    protocol = (
        "units metal\natom_style atomic\nboundary p p p\n"
        "variable dice_trajectory_every_steps equal 20\n"
        "timestep 0.001\n"
        "velocity all create ${dice_temperature} ${dice_seed}\n"
        f"fix thermostat all {kind} temp ${{dice_temperature}} ${{dice_temperature}} 0.1"
        + (" iso ${dice_pressure} ${dice_pressure} 1.0" if kind == "npt" else "")
        + "\nrun ${dice_run_steps}\n"
    )
    recipe.write_text(protocol)
    model = tmp_path / "student.pb"
    model.write_text("model")
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("water", "water"),),
        condition_groups=(SamplingConditionGroup(
            "sample", kind, ("water",), (20,), (300.0,), (1.0,) if kind == "npt" else (),
        ),), round_count=1,
    )
    evidence = replace(_coordinator_evidence(tmp_path, OrderedSamplingModels((model,), member_ids=("student",))), plan=plan, steps=None)
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(), type_map=("H",), group_recipes={"sample": recipe}),
        _IdentitySamplingModelProvider(),
    )
    result = backend.execute(SamplingExecutionRequest(evidence))
    assert backend.validate_completion(evidence).complete is True
    task = result.sample_dir / "tasks/task_000000"
    assert (task / "protocol.lammps").read_text() == protocol
    context = json.loads((task / "task_context.json").read_text())
    receipt = json.loads((task / "native_execution.json").read_text())
    assert context["recipe_sha256"] != context["rendered_input_sha256"]
    assert receipt["recipe_sha256"] == context["recipe_sha256"]
    assert receipt["rendered_input_sha256"] == context["rendered_input_sha256"]
    (task / "protocol.lammps").write_text(protocol.replace("0.001", "0.002"))
    assert backend.validate_completion(evidence).complete is False


@pytest.mark.parametrize('ensemble, coupling', [('nvt', None), ('npt', 'iso'), ('npt', 'aniso'), ('npt', 'tri')])
def test_managed_control_is_bound_to_execution_completion_and_resume(tmp_path, ensemble, coupling):
    from dataclasses import replace
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
    from workflow_core.common.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole
    from workflow_core.common.lammps_control import LammpsSamplingOptions

    recipe = tmp_path / 'protocol.lammps'
    recipe.write_text('units metal\natom_style atomic\nboundary p p p\n'
        'group mobile type 1\nfix bath mobile npt temp 1200 1200 0.1 iso 1 1 1\n'
        'timestep 0.002\nrun 999\n')
    options = LammpsSamplingOptions.from_payload({'control': {'mode': 'managed',
        **({'pressure_coupling': coupling} if coupling else {})},
        'timestep_ps': 0.001, 'trajectory_every_steps': 20})
    plan = ExplicitSamplingPlan(seeds=(SeedRole('water', 'water'),),
        condition_groups=(SamplingConditionGroup('sample', ensemble, ('water',),
            (20,), (300.0,), (1.0,) if ensemble == 'npt' else (), lammps=options),), round_count=1)
    model = tmp_path / 'student.pb'
    model.write_text('model')
    evidence = replace(_coordinator_evidence(tmp_path,
        OrderedSamplingModels((model,), member_ids=('student',))), plan=plan, steps=None,
        systems=(SamplingSeedSource('water', (Atoms('H', cell=[4]*3, pbc=True),)),))
    backend = LifecycleSamplingCoordinator(LammpsSamplingNativeOps(
        executor=_FakeLammpsExecutor(), type_map=('H',), group_recipes={'sample': recipe}),
        _IdentitySamplingModelProvider())
    result = backend.execute(SamplingExecutionRequest(evidence))
    task = result.sample_dir / 'tasks/task_000000'
    assert f'fix bath mobile {ensemble} temp' in (task / 'input.lammps').read_text()
    assert ('change_box all triclinic' in (task / 'dice_task_setup.lammps').read_text()) == (coupling == 'tri')
    context = json.loads((task / 'task_context.json').read_text())
    assert context['schema_revision'] == 5
    assert context['native_timestep'] == 0.001
    assert context['lammps_control']['settings']['group']['value'] == 'mobile'
    assert backend.validate_completion(evidence).complete
    receipt_path = task / 'native_execution.json'
    original_receipt = receipt_path.read_text()
    downgraded = json.loads(original_receipt)
    downgraded.pop('lammps_control')
    downgraded['schema_revision'] = 1
    receipt_path.write_text(json.dumps(downgraded))
    assert not backend.validate_completion(evidence).complete
    receipt_path.write_text(original_receipt)
    changed = replace(options, timestep_ps=0.002)
    changed_plan = replace(plan, condition_groups=(replace(plan.condition_groups[0], lammps=changed),))
    assert not backend.validate_completion(replace(evidence, plan=changed_plan)).complete


@pytest.mark.parametrize("kind", ["nvt_program", "npt_program"])
@pytest.mark.parametrize('source_style', ['legacy','clean'])
def test_lammps_automatic_temperature_program_keeps_one_bound_continuous_task(tmp_path: Path, kind: str, source_style: str) -> None:
    from dataclasses import replace
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
    from workflow_core.lifecycle.sampling_plan import (
        ExplicitSamplingPlan, SamplingConditionGroup, SeedRole, TemperatureProgram, TemperatureProgramSegment,
    )

    ensemble = kind.removesuffix("_program")
    recipe = tmp_path / "program.lammps"
    recipe.write_text(
        "units metal\natom_style atomic\nboundary p p p\n"
        "variable dice_trajectory_every_steps equal 20\n"
        "timestep 0.001\nvelocity all create ${dice_temperature} ${dice_seed}\n"
        "variable dice_tstart file dice_temperature_start.dat\n"
        "variable dice_tstop file dice_temperature_stop.dat\n"
        "variable dice_segment_steps file dice_temperature_steps.dat\n"
        "label dice_segment_loop\n"
        f"fix dice_{ensemble} all {ensemble} temp ${{dice_tstart}} ${{dice_tstop}} 0.1"
        + (" iso ${dice_pressure} ${dice_pressure} 1.0" if ensemble == "npt" else "")
        + f"\nrun ${{dice_segment_steps}}\nunfix dice_{ensemble}\n"
        "next dice_tstart dice_tstop dice_segment_steps\njump SELF dice_segment_loop\n"
    )
    model = tmp_path / "student.pb"
    if source_style=='clean':
        recipe.write_text('units metal\natom_style atomic\nboundary p p p\ntimestep 0.001\n'
            'velocity all create 300 917\n'+f'fix bath all {ensemble} temp 300 300 0.1'
            + (' iso 1 1 1.0' if ensemble=='npt' else '')+'\nrun 20\n')
    model.write_text("model")
    program = TemperatureProgram((TemperatureProgramSegment(300., 300., 10, 10), TemperatureProgramSegment(300., 500., 10, 20)))
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("water", "water"),),
        condition_groups=(SamplingConditionGroup(
            "sample", kind, ("water",), (20,), pressures_bar=(1.,) if ensemble == "npt" else (), program=program,
        ),), round_count=1,
    )
    evidence = replace(_coordinator_evidence(tmp_path, OrderedSamplingModels((model,), member_ids=("student",))), plan=plan, steps=None)
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(), type_map=("H",), group_recipes={"sample": recipe}),
        _IdentitySamplingModelProvider(),
    )
    result = backend.execute(SamplingExecutionRequest(evidence))
    assert result.task_count == 1
    task = result.sample_dir / "tasks/task_000000"
    assert (task / "dice_temperature_steps.dat").read_text() == "10\n10\n"
    assert (task / "dice_temperature_stop.dat").read_text() == "300.0\n500.0\n"
    assert backend.validate_completion(evidence).complete is True


@pytest.mark.parametrize('source_style',['legacy','clean'])
def test_lammps_automatic_cell_program_keeps_rendered_schedule_and_completion(tmp_path: Path, source_style: str) -> None:
    from dataclasses import replace
    from workflow_backends.lammps.adapter import LammpsSamplingNativeOps
    from workflow_backends.sampling import SamplingExecutionRequest, SamplingSeedSource
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator
    from workflow_core.lifecycle.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole, TemperatureProgram, TemperatureProgramSegment

    recipe = tmp_path / "cell.lammps"
    recipe.write_text(
        "# DICE_CAPABILITY: nvt_cell_program\n"
        "units metal\natom_style atomic\nboundary p p p\n"
        "variable dice_trajectory_every_steps equal 20\n"
        "timestep 0.001\nvelocity all create ${dice_temperature} ${dice_seed}\n"
    )
    model = tmp_path / "student.pb"
    if source_style=='clean':
        recipe.write_text('units metal\natom_style atomic\nboundary p p p\ntimestep 0.001\n'
            'velocity all create 300 917\nfix bath all nvt temp 300 300 0.2 drag 0.3\nrun 20\n')
    model.write_text("model")
    program = TemperatureProgram((TemperatureProgramSegment(300., 500., 20, 20),))
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("water", "water"),),
        condition_groups=(SamplingConditionGroup("sample", "nvt_cell_program", ("water",), (20,), program=program),),
        round_count=1,
    )
    evidence = replace(
        _coordinator_evidence(tmp_path, OrderedSamplingModels((model,), member_ids=("student",))),
        plan=plan, steps=None,
        systems=(SamplingSeedSource("water", (Atoms("H", cell=[4]*3, pbc=True),)),),
    )
    backend = LifecycleSamplingCoordinator(
        LammpsSamplingNativeOps(executor=_FakeLammpsExecutor(), type_map=("H",), group_recipes={"sample": recipe}),
        _IdentitySamplingModelProvider(),
    )
    result = backend.execute(SamplingExecutionRequest(evidence))
    rendered = (result.sample_dir / "tasks/task_000000/input.lammps").read_text()
    assert rendered.index("include dice_task_setup.lammps") < rendered.index("velocity all create")
    assert rendered.index("velocity all create") < rendered.index("fix dice_cell_nvt_0")
    if source_style=='clean':assert '500.0 0.2 drag 0.3' in rendered
    assert backend.validate_completion(evidence).complete is True


@pytest.mark.parametrize("compressed", [False, True])
def test_automatic_protocol_preserves_dpa4c_kokkos_model_selection(tmp_path: Path, compressed: bool) -> None:
    from workflow_backends.lammps.sampler import LammpsSampler
    from workflow_backends.lammps.observation_source import validate_lammps_sample_block
    from workflow_artifacts.lifecycle.sampling_models import EffectiveSamplingModel, EffectiveSamplingModelPlan, file_sha256
    from workflow_core.lifecycle.sampling_plan import ExplicitSamplingPlan, SamplingConditionGroup, SeedRole

    class KokkosExecutor(_FakeLammpsExecutor):
        def run(self, task_dir, task):
            super().run(task_dir, task)
            (task_dir / "model_devi.out").unlink()

    recipe = tmp_path / "protocol.lammps"
    recipe.write_text("units metal\natom_style atomic\nboundary p p p\ntimestep 0.001\nrun ${dice_run_steps}\n")
    source = tmp_path / "source.pt2"
    effective = tmp_path / "effective.pt2"
    source.write_text("source-model")
    effective.write_text("compressed-model" if compressed else "source-model")
    manifest = tmp_path / "sampling_model_assets.json"
    manifest.write_text("{}")
    models = EffectiveSamplingModelPlan((EffectiveSamplingModel(
        member_id="student", source_path=source, source_sha256=file_sha256(source),
        effective_path=effective, effective_sha256=file_sha256(effective), deepmd_recipe_kind="dpa4c", cli="pt",
    ),), compressed=compressed, manifest_path=manifest if compressed else None)
    plan = ExplicitSamplingPlan(
        seeds=(SeedRole("water", "water"),),
        condition_groups=(SamplingConditionGroup("sample", "nvt", ("water",), (20,), (300.,)),), round_count=1,
    )
    systems = {"water": [Atoms("H", cell=[4]*3, pbc=True)]}
    sample = tmp_path / "sample"
    LammpsSampler(tmp_path / "run.json", 0, 0, KokkosExecutor()).sample(
        group_recipes={"sample": recipe}, sample_dir=sample, models=models, type_map=("H",),
        systems=systems, steps=None, plan=plan, plumed_sources={},
    )
    task = sample / "tasks/task_000000"
    assert (task / "graph_000.pt2").read_text() == effective.read_text()
    assert "atom_modify map yes" in (task / "dice_task_setup.lammps").read_text()
    receipt = json.loads((task / "native_execution.json").read_text())
    assert receipt["pair_style"] == "deepmd/kk"
    assert receipt["command"] == ["lmp", "-k", "on", "g", "1", "-sf", "kk", "-in", "input.lammps"]
    assert validate_lammps_sample_block(
        sample_dir=sample, run_json_path=tmp_path / "run.json", iter_index=0, round_index=0,
        group_recipes={"sample": recipe}, models=models.paths, systems=systems, steps=None, plan=plan,
        require_native_execution=True, deepmd_recipe_kind="dpa4c", plumed_sources={},
    ) is True
