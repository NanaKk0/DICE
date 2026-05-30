from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from workflow_core.student_pipeline.config.stage_config import canonical_stage_type
from workflow_artifacts.first_principles.handoff import first_principles_handoff_input_path_for_stage


StagePathResolver = Callable[[Path], Path | None]


@dataclass(frozen=True, slots=True)
class StagePathRole:
    role: str
    relative_path: Path
    allow_empty_extxyz: bool = False
    completion_evidence: bool = True
    successor: bool = False
    include_descendants: bool = False
    path_resolver: StagePathResolver | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "relative_path", Path(self.relative_path))
        object.__setattr__(self, "allow_empty_extxyz", bool(self.allow_empty_extxyz))
        object.__setattr__(self, "completion_evidence", bool(self.completion_evidence))
        object.__setattr__(self, "successor", bool(self.successor))
        object.__setattr__(self, "include_descendants", bool(self.include_descendants))

    def resolve_path(self, stage_dir: str | Path) -> Path:
        root = Path(stage_dir).resolve()
        if self.path_resolver is None:
            return (root / self.relative_path).resolve()
        resolved = self.path_resolver(root)
        if resolved is None:
            raise ValueError(f"student stage artifact role {self.role!r} cannot be resolved for {root}")
        return Path(resolved).resolve()


@dataclass(frozen=True, slots=True)
class StudentStageContractDescriptor:
    stage_type: str
    materialized_inputs: tuple[StagePathRole, ...] = ()
    published_outputs: tuple[StagePathRole, ...] = ()
    adapter_context: bool = False
    stdout_contract_id: str = ""

    def __post_init__(self) -> None:
        stage_type = canonical_stage_type(self.stage_type)
        object.__setattr__(self, "stage_type", stage_type)
        object.__setattr__(self, "materialized_inputs", tuple(self.materialized_inputs or ()))
        object.__setattr__(self, "published_outputs", tuple(self.published_outputs or ()))
        object.__setattr__(self, "adapter_context", bool(self.adapter_context))
        object.__setattr__(self, "stdout_contract_id", str(self.stdout_contract_id or f"student_stage:{stage_type}"))

    @property
    def artifact_contract_id(self) -> str:
        return f"student_stage:{self.stage_type}"

    @property
    def materialized_input_roles(self) -> tuple[str, ...]:
        return tuple(role.role for role in self.materialized_inputs)

    @property
    def published_output_roles(self) -> tuple[str, ...]:
        return tuple(role.role for role in self.published_outputs)

    @property
    def completion_evidence_roles(self) -> tuple[str, ...]:
        return tuple(role.role for role in self.published_outputs if role.completion_evidence)

    @property
    def successor_roles(self) -> tuple[str, ...]:
        return tuple(role.role for role in self.published_outputs if role.successor)

    @property
    def allow_empty_extxyz_roles(self) -> tuple[str, ...]:
        return tuple(role.role for role in self.published_outputs if role.allow_empty_extxyz)

    def materialized_input_path(self, role: str, stage_dir: str | Path) -> Path:
        return _role_path(self.materialized_inputs, role, stage_dir)

    def published_output_path(self, role: str, stage_dir: str | Path) -> Path:
        return _role_path(self.published_outputs, role, stage_dir)

    def successor_path(self, role: str, stage_dir: str | Path) -> Path | None:
        for item in self.published_outputs:
            if item.role == str(role) and item.successor:
                return item.resolve_path(stage_dir)
        return None

    def artifact_root(self, stage_dir: str | Path) -> Path:
        return Path(stage_dir).resolve()


def student_stage_contract_descriptor(stage_type: str) -> StudentStageContractDescriptor:
    stage = canonical_stage_type(stage_type)
    try:
        return _STAGE_CONTRACTS[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported student pipeline stage contract: {stage_type!r}") from exc


def student_stage_contract_descriptors() -> tuple[StudentStageContractDescriptor, ...]:
    return tuple(_STAGE_CONTRACTS[stage] for stage in sorted(_STAGE_CONTRACTS))


def _role_path(roles: tuple[StagePathRole, ...], role: str, stage_dir: str | Path) -> Path:
    requested = str(role)
    for item in roles:
        if item.role == requested:
            return item.resolve_path(stage_dir)
    raise ValueError(f"unknown student stage artifact role: {requested!r}")


def _first_principles_handoff_from_stage(stage_dir: Path) -> Path | None:
    return first_principles_handoff_input_path_for_stage(stage_dir)


_STAGE_CONTRACTS: dict[str, StudentStageContractDescriptor] = {
    "relabel": StudentStageContractDescriptor(
        "relabel",
        materialized_inputs=(
            StagePathRole("relabel_input", Path("input") / "relabel_input.xyz", completion_evidence=False),
            StagePathRole(
                "failed_backflow_input",
                Path("input") / "failed_backflow_input.xyz",
                completion_evidence=False,
            ),
        ),
        published_outputs=(
            StagePathRole("student_training_labeled", Path("label") / "train.xyz"),
            StagePathRole(
                "student_training_dataset",
                Path("label"),
                completion_evidence=False,
                include_descendants=True,
            ),
            StagePathRole("teacher_dft_candidate", Path("active_learning") / "dft.xyz", allow_empty_extxyz=True),
        ),
        adapter_context=True,
    ),
    "train": StudentStageContractDescriptor(
        "train",
        materialized_inputs=(
            StagePathRole("student_training_input", Path("input") / "train.xyz", completion_evidence=False),
        ),
        published_outputs=(
            StagePathRole("student_committee", Path("student_committee"), successor=True, include_descendants=True),
            StagePathRole("student_committee_summary", Path("student_committee_summary.json"), completion_evidence=False),
        ),
        adapter_context=True,
    ),
    "sampling": StudentStageContractDescriptor(
        "sampling",
        materialized_inputs=(
            StagePathRole("seed_pool", Path("input") / "seed_pool.xyz", completion_evidence=False),
        ),
        published_outputs=(
            StagePathRole("selected_seed", Path("seed_pool.xyz"), completion_evidence=False),
            StagePathRole("student_candidate", Path("candidate.xyz"), allow_empty_extxyz=True),
            StagePathRole("student_failed", Path("failed.xyz"), allow_empty_extxyz=True),
            StagePathRole(
                "seed_pool_successor",
                Path("seed_pool_after_sampling.xyz"),
                allow_empty_extxyz=True,
                successor=True,
            ),
        ),
        adapter_context=True,
    ),
    "student_relabel_input_filtering": StudentStageContractDescriptor(
        "student_relabel_input_filtering",
        materialized_inputs=(
            StagePathRole("student_candidate", Path("input") / "candidate.xyz", completion_evidence=False),
            StagePathRole("student_failed", Path("input") / "failed.xyz", completion_evidence=False),
        ),
        published_outputs=(
            StagePathRole("filtered_candidate", Path("filtered_candidate.xyz"), allow_empty_extxyz=True),
            StagePathRole("filtered_failed", Path("filtered_failed.xyz"), allow_empty_extxyz=True),
        ),
    ),
    "seed_pool_refresh": StudentStageContractDescriptor(
        "seed_pool_refresh",
        materialized_inputs=(
            StagePathRole("seed_pool", Path("input") / "seed_pool.xyz", completion_evidence=False),
            StagePathRole("candidate_increment", Path("input") / "candidate_increment.xyz", completion_evidence=False),
            StagePathRole("failed_increment", Path("input") / "failed_increment.xyz", completion_evidence=False),
        ),
        published_outputs=(
            StagePathRole("seed_pool", Path("seed_pool.xyz"), successor=True),
        ),
    ),
    "teacher_dft_candidate_filtering": StudentStageContractDescriptor(
        "teacher_dft_candidate_filtering",
        materialized_inputs=(
            StagePathRole(
                "teacher_dft_candidate",
                Path("input") / "teacher_dft_candidates.xyz",
                completion_evidence=False,
            ),
        ),
        published_outputs=(
            StagePathRole("screened_dft_candidates", Path("screened_dft_candidates.xyz"), allow_empty_extxyz=True),
            StagePathRole(
                "first_principles_input",
                Path("first_principles") / "prepare" / "source" / "input_for_first_principles.xyz",
                allow_empty_extxyz=True,
                path_resolver=_first_principles_handoff_from_stage,
            ),
        ),
    ),
}
