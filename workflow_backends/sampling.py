from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from ase import Atoms

from workflow_artifacts.lifecycle.sample_conditions import SamplingPlan
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
from workflow_backends.sampling_models import (
    SamplingModelInputs,
    SamplingModelsRequest,
    SamplingModelsResult,
    validate_sampling_models_result,
    validate_sampling_model_inputs,
)
from workflow_core.lifecycle.sampling import (
    LifecycleSampling,
    SamplingEvidenceFacts,
    SamplingEvidenceIntent,
    SamplingExecutionIntent,
    SamplingExecutionStatus,
    SamplingObservationResult,
    SamplingValidationResult,
)
from workflow_core.lifecycle.sampling_models import SamplingModelProvider
from workflow_core.lifecycle.sampling_models import SamplingModelPreparationExpectation
from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_runtime.lifecycle.progress import (
    OperationProgressReporter,
    SamplingSnapshot,
)
from workflow_core.structures.canonical_identity import structure_id_for_atoms


@dataclass(frozen=True, slots=True)
class SamplingRecipeSource:
    """One selected native recipe, optionally scoped to an ensemble."""

    path: Path
    ensemble: str | None = None
    condition_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if self.ensemble is not None and not str(self.ensemble):
            raise ValueError("sampling recipe ensemble must be non-empty when present")
        if self.condition_group_id is not None and not str(self.condition_group_id):
            raise ValueError("sampling recipe condition group must be non-empty when present")
        if self.ensemble is not None and self.condition_group_id is not None:
            raise ValueError("sampling recipe cannot bind both ensemble and condition group")


@dataclass(frozen=True, slots=True)
class SamplingBackendBinding:
    """Route-resolved assets used once to bind the selected sampling role."""

    recipes: tuple[SamplingRecipeSource, ...]
    type_map: tuple[str, ...]
    system_inputs: Mapping[str, Path]
    guest_inputs: Mapping[str, Path] = field(default_factory=dict)
    widom_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        recipes = tuple(self.recipes)
        keys = tuple((recipe.ensemble, recipe.condition_group_id) for recipe in recipes)
        if len(set(keys)) != len(keys):
            raise ValueError("sampling backend binding recipe identities must be unique")
        type_map = tuple(str(symbol) for symbol in self.type_map)
        if not type_map:
            raise ValueError("sampling backend binding requires a non-empty type map")
        system_inputs = {
            str(system_id): Path(path).resolve()
            for system_id, path in self.system_inputs.items()
        }
        object.__setattr__(self, "recipes", recipes)
        object.__setattr__(self, "type_map", type_map)
        object.__setattr__(self, "system_inputs", system_inputs)
        object.__setattr__(self, "guest_inputs", {
            str(group_id): Path(path).resolve() for group_id, path in self.guest_inputs.items()
        })


@dataclass(frozen=True, slots=True)
class SamplingSeedSource:
    system_id: str
    frames: tuple[Atoms, ...]

    def __post_init__(self) -> None:
        system_id = str(self.system_id)
        frames = tuple(self.frames)
        if not system_id or not frames or any(not isinstance(frame, Atoms) for frame in frames):
            raise ValueError("sampling seed sources require a system id and native seed frames")
        object.__setattr__(self, "system_id", system_id)
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True, slots=True)
class SamplingSeedSelection:
    source_role_id: str
    source_sha256: str
    frame_count: int
    max_frames: int
    selected_indices: tuple[int, ...]
    random_seed_sha256: str
    algorithm: str

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.selected_indices)
        if not self.source_role_id or len(self.source_sha256) != 64:
            raise ValueError("sampling seed selection requires role and source SHA-256")
        if self.frame_count <= 0 or self.max_frames <= 0:
            raise ValueError("sampling seed selection counts must be positive")
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("sampling seed selection indices must be unique and non-empty")
        if any(index < 0 or index >= self.frame_count for index in indices):
            raise ValueError("sampling seed selection index is outside its source")
        if len(indices) != min(self.frame_count, self.max_frames):
            raise ValueError("sampling seed selection count must equal min(frame_count, max_frames)")
        object.__setattr__(self, "selected_indices", indices)


def seed_selection_manifest_payload(
    *,
    iter_index: int,
    round_index: int,
    selections: tuple[SamplingSeedSelection, ...],
) -> dict[str, object]:
    return {
        "schema": "dice.lifecycle.seed-frame-selection",
        "schema_revision": 1,
        "iter": int(iter_index),
        "round": int(round_index),
        "sources": [
            {
                "sampling_role_id": selection.source_role_id,
                "source_sha256": selection.source_sha256,
                "frame_count": selection.frame_count,
                "max_frames": selection.max_frames,
                "selected_indices": list(selection.selected_indices),
                "algorithm": selection.algorithm,
                "random_seed_sha256": selection.random_seed_sha256,
            }
            for selection in selections
        ],
    }


@dataclass(frozen=True, slots=True)
class SamplingEvidenceRequest:
    """Resolved backend-neutral inputs shared by sample evidence operations."""

    run_json_path: Path
    iter_index: int
    round_index: int
    sample_dir: Path
    plan: SamplingPlan | ExplicitSamplingPlan
    systems: tuple[SamplingSeedSource, ...]
    models: SamplingModelInputs
    steps: int | None
    seed_selections: tuple[SamplingSeedSelection, ...] = ()

    def __post_init__(self) -> None:
        systems = tuple(self.systems)
        system_ids = tuple(system.system_id for system in systems)
        validate_sampling_model_inputs(self.models)
        model_paths = tuple(Path(path).resolve() for path in self.models.paths)
        if not isinstance(self.plan, (SamplingPlan, ExplicitSamplingPlan)):
            raise TypeError("sampling requires one resolved SamplingPlan")
        if not systems or len(set(system_ids)) != len(system_ids):
            raise ValueError("sampling systems must be ordered, unique, and non-empty")
        plan_system_ids = (
            tuple(
                role_id
                for role_id, _group_id in self.plan.selection_plan_for_round(
                    self.round_index
                ).role_to_group
            )
            if isinstance(self.plan, ExplicitSamplingPlan)
            else tuple(system.system_id for system in self.plan.systems)
        )
        if plan_system_ids != system_ids:
            raise ValueError("sampling seed sources must match the resolved plan order")
        if isinstance(self.iter_index, bool) or int(self.iter_index) < 0:
            raise ValueError("sampling iter index must be non-negative")
        if isinstance(self.round_index, bool) or int(self.round_index) < 0:
            raise ValueError("sampling round index must be non-negative")
        if isinstance(self.plan, SamplingPlan):
            if isinstance(self.steps, bool) or self.steps is None or int(self.steps) <= 0:
                raise ValueError("legacy sampling steps must be positive")
        elif self.steps is not None:
            raise ValueError("explicit sampling steps are owned by condition groups")
        object.__setattr__(self, "run_json_path", Path(self.run_json_path).resolve())
        object.__setattr__(self, "sample_dir", Path(self.sample_dir).resolve())
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "round_index", int(self.round_index))
        object.__setattr__(self, "systems", systems)
        object.__setattr__(self, "seed_selections", tuple(self.seed_selections))
        object.__setattr__(
            self, "steps", None if self.steps is None else int(self.steps)
        )

    @property
    def model_paths(self) -> tuple[Path, ...]:
        return tuple(Path(path).resolve() for path in self.models.paths)

    @property
    def seed_mapping(self) -> dict[str, tuple[Atoms, ...]]:
        return {source.system_id: source.frames for source in self.systems}

@dataclass(frozen=True, slots=True)
class SamplingExecutionRequest:
    evidence: SamplingEvidenceRequest
    progress_callback: Callable[[SamplingSnapshot], None] | None = None
    progress_reporter_factory: Callable[[str], OperationProgressReporter] | None = None


@dataclass(frozen=True, slots=True)
class SamplingExecutionResult:
    sample_dir: Path
    task_count: int
    failed_task_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_dir", Path(self.sample_dir).resolve())

    @property
    def status(self) -> SamplingExecutionStatus:
        return SamplingExecutionStatus.SUCCEEDED


class SamplingNativeOps(Protocol):
    """Backend-native operations over coordinator-resolved model inputs."""

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult: ...

    def validate_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingValidationResult: ...

    def open_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult: ...

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult: ...


class SamplingBackendFactory(Protocol):
    """Selected native capability that binds one route-specific sampling role."""

    def bind(
        self,
        binding: SamplingBackendBinding,
    ) -> SamplingNativeOps: ...


@dataclass(slots=True)
class BoundSamplingHandleSession:
    """Resolve canonical model handles once at a physical sampling boundary.

    One instance is scoped to one coordinator-facing block operation.  It
    preserves the coordinator's prepare/load/native ordering while keeping
    paths, ASE frames, recipes, replay sources, and progress callbacks behind
    this adapter.
    """

    expectation: ExpectedArtifactSemantics[SamplingEvidenceIntent]
    preparation_expectation: SamplingModelPreparationExpectation
    physical_evidence: SamplingEvidenceRequest
    physical_execution: SamplingExecutionRequest
    model_provider: SamplingModelProvider
    native_ops: SamplingNativeOps
    evidence_observer: object | None = None
    resolved_models: tuple[object, ...] = ()
    _resolved_evidence: SamplingEvidenceRequest | None = field(
        default=None, init=False, repr=False
    )
    _resolution_kind: str | None = field(default=None, init=False, repr=False)

    @property
    def intent(self) -> SamplingEvidenceIntent:
        return self.expectation.facts

    def __post_init__(self) -> None:
        if (
            self.expectation.artifact_kind != "sampling"
            or self.expectation.block.value
            != f"r{self.expectation.facts.round_index}/sample"
        ):
            raise ValueError("bound sampling session requires a sample expectation")
        if (
            self.preparation_expectation.iter_index != self.intent.iter_index
            or self.preparation_expectation.round_index != self.intent.round_index
            or self.preparation_expectation.models != self.intent.models
        ):
            raise ValueError(
                "sampling model preparation diverges from the sample expectation"
            )
        self._require_intent(self.intent)
        if self.physical_execution.evidence != self.physical_evidence:
            raise ValueError(
                "sampling execution and evidence bindings must share one native request"
            )
        if tuple(self.physical_evidence.models.member_ids) != tuple(
            model.member_id for model in self.intent.models
        ):
            raise ValueError(
                "canonical sampling handles diverge from bound native model order"
            )
        resolved_models = tuple(self.resolved_models)
        if resolved_models and (
            tuple(getattr(item, "handle") for item in resolved_models)
            != self.intent.models
            or tuple(Path(getattr(item, "path")).resolve() for item in resolved_models)
            != tuple(self.physical_evidence.models.paths)
        ):
            raise ValueError(
                "resolved sampling models diverge from canonical handles or physical inputs"
            )
        if (
            self.physical_evidence.iter_index != self.intent.iter_index
            or self.physical_evidence.round_index != self.intent.round_index
        ):
            raise ValueError("sampling indices diverge from the canonical intent")
        physical_system_ids = tuple(
            system.system_id for system in self.physical_evidence.systems
        )
        semantic_system_ids = tuple(system.system_id for system in self.intent.systems)
        plan_system_ids = (
            tuple(
                role_id
                for role_id, _group_id in self.physical_evidence.plan.selection_plan_for_round(
                    self.physical_evidence.round_index
                ).role_to_group
            )
            if isinstance(self.physical_evidence.plan, ExplicitSamplingPlan)
            else tuple(
                system.system_id for system in self.physical_evidence.plan.systems
            )
        )
        if (
            physical_system_ids != semantic_system_ids
            or plan_system_ids != semantic_system_ids
        ):
            raise ValueError("sampling systems diverge from the canonical intent")
        for semantic, physical in zip(
            self.intent.systems, self.physical_evidence.systems, strict=True
        ):
            identities = tuple(
                str(frame.info.get("structure_id") or structure_id_for_atoms(frame))
                for frame in physical.frames
            )
            if identities != semantic.structures.ordered_structure_ids:
                raise ValueError(
                    "sampling seed structures diverge from the canonical intent"
                )
        resolved_tasks = (
            self.physical_evidence.plan.tasks_for_round(self.physical_evidence.round_index)
            if isinstance(self.physical_evidence.plan, ExplicitSamplingPlan)
            else self.physical_evidence.plan.tasks
        )
        physical_tasks = tuple(
            (
                task.system_id,
                task.ensemble,
                None if task.temperature_k is None else float(task.temperature_k),
                None if task.pressure_bar is None else float(task.pressure_bar),
            )
            for task in resolved_tasks
        )
        semantic_tasks = tuple(
            (
                task.system_id,
                task.ensemble,
                task.temperature_k,
                task.pressure_bar,
            )
            for task in self.intent.tasks
        )
        if physical_tasks != semantic_tasks:
            raise ValueError("sampling tasks diverge from the canonical intent")

    def prepare_sampling_models(
        self, request: SamplingEvidenceIntent
    ) -> SamplingModelsResult:
        self._require_fresh_resolution("prepare")
        self._resolve_physical_models(request, prepare=True)
        return SamplingModelsResult(self.intent.models)

    def load_sampling_models(
        self, request: SamplingEvidenceIntent
    ) -> SamplingModelsResult:
        self._require_fresh_resolution("load")
        self._resolve_physical_models(request, prepare=False)
        return SamplingModelsResult(self.intent.models)

    def execute(self, request: SamplingExecutionIntent) -> SamplingExecutionResult:
        if not isinstance(request, SamplingExecutionIntent):
            raise TypeError("bound sampling execution requires its canonical intent")
        evidence = self._take_resolved(request.evidence, expected_kind="prepare")
        return self.native_ops.execute(
            replace(self.physical_execution, evidence=evidence)
        )

    def validate_current(
        self, request: SamplingEvidenceIntent
    ) -> SamplingValidationResult:
        evidence = self._take_resolved(request, expected_kind="load")
        return self.native_ops.validate_current(evidence)

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceIntent
    ) -> SamplingObservationResult:
        evidence = self._take_resolved(request, expected_kind="load")
        return self.native_ops.reconstruct_compatible_legacy(evidence)

    def open_current(
        self, request: SamplingEvidenceIntent
    ) -> SamplingObservationResult:
        self._require_intent(request)
        if self._resolved_evidence is not None:
            raise ValueError("current sampling evidence unexpectedly resolved models")
        return self.native_ops.open_current(self.physical_evidence)

    def observe_evidence(self, request: SamplingEvidenceIntent) -> SamplingEvidenceFacts:
        self._require_intent(request)
        source = self.native_ops if self.evidence_observer is None else self.evidence_observer
        observe = getattr(source, "observe_evidence", None)
        if not callable(observe):
            return observe_path_backed_sampling_evidence(self.physical_evidence)
        return observe(self.physical_evidence)

    def _resolve_physical_models(
        self, request: SamplingEvidenceIntent, *, prepare: bool
    ) -> None:
        self._require_intent(request)
        physical_request = SamplingModelsRequest(
            models=self.physical_evidence.models,
            sample_dir=self.physical_evidence.sample_dir,
        )
        resolve = (
            self.model_provider.prepare_sampling_models
            if prepare
            else self.model_provider.load_sampling_models
        )
        result = resolve(physical_request)
        physical_models = validate_sampling_models_result(physical_request, result)
        if tuple(physical_models.member_ids) != tuple(
            model.member_id for model in self.intent.models
        ):
            raise ValueError(
                "physical sampling model resolution changed canonical handle order"
            )
        self._resolved_evidence = replace(
            self.physical_evidence,
            models=physical_models,
        )
        self._resolution_kind = "prepare" if prepare else "load"

    def _take_resolved(
        self, request: SamplingEvidenceIntent, *, expected_kind: str
    ) -> SamplingEvidenceRequest:
        self._require_intent(request)
        if self._resolution_kind != expected_kind or self._resolved_evidence is None:
            raise ValueError(
                f"sampling native {expected_kind} consumer ran without model resolution"
            )
        evidence = self._resolved_evidence
        self._resolved_evidence = None
        self._resolution_kind = None
        return evidence

    def _require_fresh_resolution(self, kind: str) -> None:
        if self._resolved_evidence is not None:
            raise ValueError(
                f"sampling {kind} attempted while another model resolution is pending"
            )

    def _require_intent(self, request: SamplingEvidenceIntent) -> None:
        if request != self.intent:
            raise ValueError("sampling intent does not match its physical binding")


def observe_path_backed_sampling_evidence(
    request: SamplingEvidenceRequest,
) -> SamplingEvidenceFacts:
    """Observe the common physical current/history discriminator for an adapter."""

    root = request.sample_dir
    task_root = root / "tasks"
    task_dirs = (
        tuple(sorted(path for path in task_root.iterdir() if path.is_dir()))
        if task_root.is_dir()
        else ()
    )
    gcmc_ids = {
        task.task_id for task in request.plan.tasks_for_round(request.round_index)
        if task.kind == 'gcmc_md'
    } if isinstance(request.plan, ExplicitSamplingPlan) else set()
    widom_ids = {
        task.task_id for task in request.plan.tasks_for_round(request.round_index)
        if task.kind == 'widom'
    } if isinstance(request.plan, ExplicitSamplingPlan) else set()
    return SamplingEvidenceFacts(
        seeds_present=(root / "seeds.xyz").exists(),
        task_context_presence=tuple(
            (task / ('widom_result.json' if task.name in widom_ids else ('gcmc_result.json' if task.name in gcmc_ids else 'task_context.json'))).exists()
            for task in task_dirs
        ),
    )


__all__ = [
    "SamplingBackendBinding",
    "SamplingBackendFactory",
    "BoundSamplingHandleSession",
    "SamplingEvidenceRequest",
    "SamplingExecutionRequest",
    "SamplingExecutionResult",
    "SamplingObservationResult",
    "SamplingNativeOps",
    "LifecycleSampling",
    "SamplingRecipeSource",
    "SamplingSeedSource",
    "SamplingValidationResult",
    "observe_path_backed_sampling_evidence",
]
