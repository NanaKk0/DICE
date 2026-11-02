from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from typing import Protocol

from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.lifecycle.artifact_handles import ModelHandle, StructureBatchHandle
from workflow_core.lifecycle.state_model import PublicBlockId
from workflow_core.common.sampling_plan import (
    NebSamplingParameters,
    RelaxSamplingParameters,
    neb_algorithm_identity,
    neb_constraint_identity,
    neb_parameters_payload,
    relax_algorithm_identity,
    relax_constraint_identity,
    relax_parameters_payload,
)
from workflow_core.common.widom import WidomSamplingParameters
from workflow_core.common.gcmc import GcmcSamplingParameters
from workflow_core.common.md_mc import IdentitySwapParameters


class SamplingModelSet(Protocol):
    """Ordered model identity carried by a sampling command."""

    @property
    def member_ids(self) -> tuple[str, ...]: ...


class SamplingEvidence(Protocol):
    """Path-free evidence identity consumed by the sampling use case."""

    @property
    def iter_index(self) -> int: ...

    @property
    def round_index(self) -> int: ...

    @property
    def models(self) -> SamplingModelSet: ...


class SamplingExecutionCommand(Protocol):
    @property
    def evidence(self) -> SamplingEvidence: ...


class SamplingExecutionOutcome(Protocol):
    @property
    def task_count(self) -> int: ...

    @property
    def failed_task_count(self) -> int: ...

    @property
    def status(self) -> SamplingExecutionStatus: ...


class SamplingObservationReplay(Protocol):
    """Bounded ordered replay without exposing a backend source type."""

    def catalog(self) -> tuple[object, ...]: ...

    def replay(
        self,
        ordered_identities: Sequence[str],
        *,
        chunk_size: int,
    ) -> Iterator[tuple[object, ...]]: ...


class SamplingExecutionStatus(Enum):
    SUCCEEDED = "succeeded"


@dataclass(frozen=True, slots=True)
class SamplingSystemIntent:
    """One ordered logical seed system selected by core policy."""

    system_id: str
    structures: StructureBatchHandle

    def __post_init__(self) -> None:
        system_id = str(self.system_id).strip()
        if not system_id:
            raise ValueError("sampling system identity must be non-empty")
        if not self.structures.ordered_structure_ids:
            raise ValueError("sampling seed system requires structures")
        object.__setattr__(self, "system_id", system_id)


@dataclass(frozen=True, slots=True)
class SamplingTaskIntent:
    """One ordered path-free native task condition selected by core policy."""

    task_id: str
    system_id: str
    ensemble: str
    temperature_k: float | None
    pressure_bar: float | None = None
    condition_group_id: str | None = None
    sampling_role_id: str | None = None
    selection_group_id: str | None = None
    kind: str | None = None
    steps: int | None = None
    program: object | None = None
    source_frame_index: int = 0
    neb: NebSamplingParameters | None = None
    relax: RelaxSamplingParameters | None = None
    gcmc: GcmcSamplingParameters | None = None
    widom: WidomSamplingParameters | None = None
    md_mc: IdentitySwapParameters | None = None
    cycles: int | None = None

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        system_id = str(self.system_id).strip()
        ensemble = str(self.ensemble).strip().lower()
        explicit = self.condition_group_id is not None
        if not task_id or not system_id:
            raise ValueError("sampling task and system identities must be non-empty")
        if ensemble not in {"neb", "relax", "nvt", "npt", "gcmc_md", "widom"}:
            raise ValueError("sampling task ensemble must be neb, relax, nvt, npt, gcmc_md, or widom")
        if (ensemble in {"npt", "gcmc_md"}) != (self.pressure_bar is not None):
            raise ValueError("sampling pressure presence must match its ensemble")
        if explicit:
            values = (
                self.condition_group_id,
                self.sampling_role_id,
                self.selection_group_id,
                self.kind,
            )
            if any(value is None or not str(value).strip() for value in values):
                raise ValueError("explicit sampling tasks require group, role, selection, and kind identities")
            if self.kind not in {"neb", "relax", "nvt", "npt", "nvt_program", "nvt_cell_program", "npt_program", "gcmc_md", "widom", "md_mc"}:
                raise ValueError("explicit sampling task kind is unsupported")
            if (self.kind == 'md_mc') != isinstance(self.md_mc, IdentitySwapParameters):
                raise ValueError('only MDMC tasks contain closed identity-swap parameters')
            if (self.kind == 'widom') != (self.widom is not None):
                raise ValueError('only Widom tasks contain Widom parameters')
            is_neb = self.kind == "neb"
            is_relax = self.kind == "relax"
            is_program = self.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
            if is_program != (self.program is not None):
                raise ValueError("explicit program task must contain exactly one program")
            if (is_program or is_neb or is_relax or self.kind == "widom") != (self.temperature_k is None):
                raise ValueError("explicit program task cannot contain a fixed temperature")
            if self.kind == 'gcmc_md':
                if (ensemble != 'gcmc_md' or not isinstance(self.gcmc, GcmcSamplingParameters)
                    or isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0
                    or self.steps is not None or self.neb is not None or self.relax is not None):
                    raise ValueError('explicit GCMC task requires closed parameters, cycles and no standalone MD steps')
            elif self.gcmc is not None or self.cycles is not None:
                raise ValueError('only GCMC tasks may contain GCMC parameters/cycles')
            elif self.kind == 'widom':
                if not isinstance(self.widom, WidomSamplingParameters) or self.steps is not None or self.neb is not None or self.relax is not None:
                    raise ValueError('explicit widom requires closed parameters and no MD steps')
            elif is_neb:
                if self.steps is not None or not isinstance(
                    self.neb, NebSamplingParameters
                ):
                    raise ValueError(
                        "explicit NEB task requires closed NEB parameters and no MD steps"
                    )
            elif is_relax:
                if self.steps is not None or not isinstance(
                    self.relax, RelaxSamplingParameters
                ) or self.neb is not None:
                    raise ValueError(
                        "explicit relax task requires closed relax parameters and no MD steps"
                    )
            elif (
                isinstance(self.steps, bool)
                or self.steps is None
                or int(self.steps) <= 0
                or self.neb is not None
                or self.relax is not None
            ):
                raise ValueError(
                    "explicit MD sampling task requires positive steps and no NEB parameters"
                )
            if (
                isinstance(self.source_frame_index, bool)
                or not isinstance(self.source_frame_index, int)
                or self.source_frame_index < 0
            ):
                raise ValueError(
                    "explicit sampling task source frame index must be non-negative"
                )
        elif any(
            value is not None
            for value in (
                self.sampling_role_id,
                self.selection_group_id,
                self.kind,
                self.steps,
                self.program,
                self.neb,
                self.relax,
                self.gcmc,
                self.cycles,
                self.md_mc,
            )
        ):
            raise ValueError("legacy sampling tasks cannot contain partial explicit identity")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "system_id", system_id)
        object.__setattr__(self, "ensemble", ensemble)
        object.__setattr__(
            self,
            "temperature_k",
            None if self.temperature_k is None else float(self.temperature_k),
        )
        object.__setattr__(
            self,
            "pressure_bar",
            None if self.pressure_bar is None else float(self.pressure_bar),
        )
        if explicit:
            object.__setattr__(self, "condition_group_id", str(self.condition_group_id).strip())
            object.__setattr__(self, "sampling_role_id", str(self.sampling_role_id).strip())
            object.__setattr__(self, "selection_group_id", str(self.selection_group_id).strip())
            object.__setattr__(self, "kind", str(self.kind))
            object.__setattr__(
                self,
                "steps",
                None if self.steps is None else int(self.steps),
            )
            object.__setattr__(self, "source_frame_index", int(self.source_frame_index))


@dataclass(frozen=True, slots=True)
class SamplingEvidenceIntent:
    """Canonical path-free identity of one round's sampling evidence."""

    iter_index: int
    round_index: int
    models: tuple[ModelHandle, ...]
    systems: tuple[SamplingSystemIntent, ...]
    tasks: tuple[SamplingTaskIntent, ...]
    evidence_inventory: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.iter_index, bool) or int(self.iter_index) < 0:
            raise ValueError("sampling iter index must be non-negative")
        if isinstance(self.round_index, bool) or int(self.round_index) < 0:
            raise ValueError("sampling round index must be non-negative")
        models = tuple(self.models)
        if not models or any(not isinstance(model, ModelHandle) for model in models):
            raise TypeError("sampling evidence requires canonical model handles")
        members = tuple(model.member_id for model in models)
        model_ids = tuple(model.model_id for model in models)
        if len(set(members)) != len(members) or len(set(model_ids)) != len(model_ids):
            raise ValueError("sampling model handles must be ordered and unique")
        systems = tuple(self.systems)
        system_ids = tuple(system.system_id for system in systems)
        if (
            not systems
            or any(not isinstance(system, SamplingSystemIntent) for system in systems)
            or len(set(system_ids)) != len(system_ids)
        ):
            raise ValueError("sampling systems must be ordered and unique")
        tasks = tuple(self.tasks)
        task_ids = tuple(task.task_id for task in tasks)
        if (
            not tasks
            or any(not isinstance(task, SamplingTaskIntent) for task in tasks)
            or len(set(task_ids)) != len(task_ids)
            or any(task.system_id not in system_ids for task in tasks)
        ):
            raise ValueError("sampling tasks must be ordered, unique, and system-bound")
        evidence_inventory = tuple(
            str(item).strip() for item in self.evidence_inventory
        )
        if (
            not evidence_inventory
            or any(not item for item in evidence_inventory)
            or len(set(evidence_inventory)) != len(evidence_inventory)
        ):
            raise ValueError("sampling evidence inventory must be ordered and unique")
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "round_index", int(self.round_index))
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "systems", systems)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "evidence_inventory", evidence_inventory)

    def semantic_fields(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "iter_index": self.iter_index,
                "round_index": self.round_index,
                "models": tuple(
                    {"member_id": model.member_id, "model_id": model.model_id}
                    for model in self.models
                ),
                "systems": tuple(
                    {
                        "system_id": system.system_id,
                        "batch_id": system.structures.batch_id,
                        "ordered_structure_ids": system.structures.ordered_structure_ids,
                    }
                    for system in self.systems
                ),
                "tasks": tuple(
                    {
                        "task_id": task.task_id,
                        "system_id": task.system_id,
                        "ensemble": task.ensemble,
                        "temperature_k": task.temperature_k,
                        "pressure_bar": task.pressure_bar,
                        **(
                            {
                                "condition_group_id": task.condition_group_id,
                                "sampling_role_id": task.sampling_role_id,
                                "selection_group_id": task.selection_group_id,
                                "kind": task.kind,
                                "steps": task.steps,
                                "source_frame_index": task.source_frame_index,
                                "program": (
                                    tuple(
                                        {
                                            "start_temperature_k": segment.start_temperature_k,
                                            "stop_temperature_k": segment.stop_temperature_k,
                                            "steps": segment.steps,
                                            "end_step": segment.end_step,
                                            **(
                                                {
                                                    "cell": (
                                                        {
                                                            "mode": "isotropic_density",
                                                            "stop_density_g_cm3": segment.cell.stop_density_g_cm3,
                                                        }
                                                        if segment.cell.mode
                                                        == "isotropic_density"
                                                        else {
                                                            "mode": "symmetric_strain",
                                                            "stop": dict(
                                                                segment.cell.symmetric_strain
                                                            ),
                                                        }
                                                    )
                                                }
                                                if getattr(segment, "cell", None)
                                                is not None
                                                else {}
                                            ),
                                        }
                                        for segment in task.program.segments
                                    )
                                    if task.program is not None
                                    else None
                                ),
                                "neb": (
                                    {
                                        **neb_parameters_payload(task.neb),
                                        "fixed_atoms": task.neb.fixed_atoms,
                                        "algorithm": neb_algorithm_identity(task.neb),
                                        "constraint": {
                                            **neb_constraint_identity(task.neb),
                                            "indices": task.neb.fixed_atoms,
                                        },
                                    }
                                    if task.neb is not None
                                    else None
                                ),
                                **(
                                    {'gcmc': asdict(task.gcmc), 'cycles': task.cycles}
                                    if task.gcmc is not None else {}
                                ),
                                **({'widom': asdict(task.widom)} if task.widom is not None else {}),
                                **({'md_mc': asdict(task.md_mc)} if task.md_mc is not None else {}),
                                **(
                                    {"relax": {
                                        **relax_parameters_payload(task.relax),
                                        "fixed_atoms": task.relax.fixed_atoms,
                                        "algorithm": relax_algorithm_identity(task.relax),
                                        "constraint": {
                                            **relax_constraint_identity(task.relax),
                                            "indices": task.relax.fixed_atoms,
                                        },
                                    }}
                                    if task.relax is not None
                                    else {}
                                ),
                            }
                            if task.condition_group_id is not None
                            else {}
                        ),
                    }
                    for task in self.tasks
                ),
                "evidence_inventory": self.evidence_inventory,
            }
        )

    def for_block(
        self, block: PublicBlockId
    ) -> ExpectedArtifactSemantics[SamplingEvidenceIntent]:
        if block != PublicBlockId.round(self.round_index, "sample"):
            raise ValueError("sampling evidence expectation is bound to another block")
        return ExpectedArtifactSemantics(block, "sampling", self)


@dataclass(frozen=True, slots=True)
class SamplingExecutionIntent:
    """Canonical execution command; adapters bind native paths and callbacks."""

    evidence: SamplingEvidenceIntent


class SamplingValidationStatus(Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class SamplingValidationResult:
    complete: bool

    @property
    def status(self) -> SamplingValidationStatus:
        return (
            SamplingValidationStatus.COMPLETE
            if self.complete
            else SamplingValidationStatus.INCOMPLETE
        )


@dataclass(frozen=True, slots=True)
class SamplingObservationResult:
    source: SamplingObservationReplay
    historical: bool = False


@dataclass(frozen=True, slots=True)
class SamplingEvidenceFacts:
    """Physical evidence facts observed by an adapter and classified by logic."""

    seeds_present: bool
    task_context_presence: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_context_presence",
            tuple(bool(item) for item in self.task_context_presence),
        )


class SamplingEvidenceState(Enum):
    CURRENT = "current"
    LEGACY_COMPATIBLE = "legacy-compatible"
    REBUILD_REQUIRED = "rebuild-required"


class SamplingEvidenceFailure(Enum):
    PARTIAL_OR_AMBIGUOUS = "partial-or-ambiguous"
    HISTORICAL_INCOMPATIBLE = "historical-incompatible"


class SamplingEvidenceRebuildRequired(ValueError):
    """Typed resumable rejection of committed sample evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure: SamplingEvidenceFailure,
        round_index: int,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.resume_block = f"r{int(round_index)}/sample"
        self.resumable = True


@dataclass(frozen=True, slots=True)
class SampleRecoveryFacts:
    """Path-free normalized settings for one committed sample block."""

    backend: str
    cli: str | None
    member_ids: tuple[str, ...]
    type_map: tuple[str, ...]
    system_ids: tuple[str, ...]
    tasks: tuple[SamplingTaskIntent, ...]
    steps: int | None
    plumed_enabled: bool
    compress: bool
    iter_index: int = 0
    round_index: int = 0

    def __post_init__(self) -> None:
        backend = str(self.backend).strip()
        cli = None if self.cli is None else str(self.cli).strip()
        members = tuple(str(item).strip() for item in self.member_ids)
        type_map = tuple(str(item).strip() for item in self.type_map)
        systems = tuple(str(item).strip() for item in self.system_ids)
        tasks = tuple(self.tasks)
        if backend not in {"LAMMPS", "GPUMD"}:
            raise ValueError("sample recovery backend must be LAMMPS or GPUMD")
        if backend == "LAMMPS" and cli not in {"tf", "pt"}:
            raise ValueError("LAMMPS sample recovery requires DeepMD CLI")
        if backend == "GPUMD" and cli is not None:
            raise ValueError("GPUMD sample recovery cannot declare DeepMD CLI")
        for label, values in (
            ("members", members), ("type map", type_map), ("systems", systems)
        ):
            if not values or any(not item for item in values) or len(set(values)) != len(values):
                raise ValueError(f"sample recovery {label} must be ordered unique")
        if (
            not tasks
            or any(not isinstance(task, SamplingTaskIntent) for task in tasks)
            or tuple(task.task_id for task in tasks)
            != tuple(f"task_{index:06d}" for index in range(len(tasks)))
            or any(task.system_id not in systems for task in tasks)
        ):
            raise ValueError("sample recovery tasks must be exact ordered intents")
        explicit = all(task.condition_group_id is not None for task in tasks)
        if any((task.condition_group_id is not None) != explicit for task in tasks):
            raise ValueError("sample recovery cannot mix legacy and explicit task intents")
        if explicit:
            if self.steps is not None:
                raise ValueError(
                    "explicit sample recovery requires task-local MD or NEB parameters"
                )
        elif isinstance(self.steps, bool) or self.steps is None or int(self.steps) <= 0:
            raise ValueError("legacy sample recovery steps must be positive")
        if not isinstance(self.plumed_enabled, bool) or not isinstance(self.compress, bool):
            raise TypeError("sample recovery mode flags must be boolean")
        if (
            isinstance(self.iter_index, bool)
            or int(self.iter_index) < 0
            or isinstance(self.round_index, bool)
            or int(self.round_index) < 0
        ):
            raise ValueError("sample recovery iteration coordinates must be non-negative")
        if backend == "GPUMD" and (self.plumed_enabled or self.compress):
            raise ValueError("GPUMD sample recovery forbids PLUMED/compression")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "cli", cli)
        object.__setattr__(self, "member_ids", members)
        object.__setattr__(self, "type_map", type_map)
        object.__setattr__(self, "system_ids", systems)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(
            self, "steps", None if self.steps is None else int(self.steps)
        )
        object.__setattr__(self, "iter_index", int(self.iter_index))
        object.__setattr__(self, "round_index", int(self.round_index))


def classify_sampling_evidence(facts: SamplingEvidenceFacts) -> SamplingEvidenceState:
    """Classify the accepted current, historical, and partial evidence shapes."""

    contexts = facts.task_context_presence
    if facts.seeds_present and all(contexts):
        return SamplingEvidenceState.CURRENT
    if not facts.seeds_present and not any(contexts):
        return SamplingEvidenceState.LEGACY_COMPATIBLE
    return SamplingEvidenceState.REBUILD_REQUIRED


class SamplingNativeOperations(Protocol):
    """Provider operations invoked by the sampling use case."""

    def execute(self, request: SamplingExecutionCommand) -> SamplingExecutionOutcome: ...

    def validate_current(
        self, request: SamplingEvidence
    ) -> SamplingValidationResult: ...

    def open_current(self, request: SamplingEvidence) -> SamplingObservationResult: ...

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidence
    ) -> SamplingObservationResult: ...


class SamplingEvidenceObserver(Protocol):
    """Adapter seam that observes physical evidence without classifying it."""

    def observe_evidence(self, request: SamplingEvidence) -> SamplingEvidenceFacts: ...


class LifecycleSampling(Protocol):
    """Logic-owned sampling capability with closed recovery behavior."""

    def execute(self, request: SamplingExecutionCommand) -> SamplingExecutionOutcome: ...

    def open_observations(
        self, request: SamplingEvidence
    ) -> SamplingObservationResult: ...

    def validate_completion(
        self, request: SamplingEvidence
    ) -> SamplingValidationResult: ...


__all__ = [
    "LifecycleSampling",
    "SampleRecoveryFacts",
    "SamplingEvidence",
    "SamplingEvidenceFacts",
    "SamplingEvidenceFailure",
    "SamplingEvidenceIntent",
    "SamplingEvidenceObserver",
    "SamplingEvidenceRebuildRequired",
    "SamplingEvidenceState",
    "SamplingExecutionCommand",
    "SamplingExecutionIntent",
    "SamplingExecutionOutcome",
    "SamplingExecutionStatus",
    "SamplingModelSet",
    "SamplingSystemIntent",
    "SamplingTaskIntent",
    "SamplingNativeOperations",
    "SamplingObservationReplay",
    "SamplingObservationResult",
    "SamplingValidationResult",
    "SamplingValidationStatus",
    "classify_sampling_evidence",
]
