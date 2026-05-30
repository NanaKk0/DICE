from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from workflow_core.common.uncertainty import UncertaintyRecord


@dataclass(slots=True)
class CommitteeModelArtifact:
    backend: str
    member_index: int
    model_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudentExportContract:
    canonical_format: str = "extxyz"
    requires_element_order: bool = True
    requires_train_valid_split: bool = True
    writes_deepmd_dataset: bool = False


@dataclass(slots=True)
class StudentContinuationContract:
    supports_bootstrap_from_summary: bool = True
    supports_finetune_from_previous_round: bool = True
    requires_restart_assets: bool = False
    required_member_assets: tuple[str, ...] = ()


@dataclass(slots=True)
class StudentFoundationBootstrapSpec:
    backend_name: str = ""
    model_path: str | None = None
    restart_path: str | None = None
    type_map: list[str] = field(default_factory=list)
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudentCommitteeSummary:
    backend: str
    summary_path: str
    committee_root: str
    member_count: int
    usable_member_count: int
    members: list[CommitteeModelArtifact]
    supports_continuation: bool
    continuation_assets_complete: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_summary_path(self) -> Path:
        return Path(self.summary_path).resolve()


@dataclass(slots=True)
class StudentTrainRequest:
    backend_name: str
    outer_round: int
    student_round: int
    work_root: str
    summary_path: str
    backend_cfg: Any
    dataset_path: str | None = None
    dataset_roots: list[str] = field(default_factory=list)
    previous_model_root: str | None = None
    continue_training: bool = False
    training_stage: str = "bootstrap"
    execution_context: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def teacher_round(self) -> int:
        return int(self.outer_round)


@dataclass(slots=True, kw_only=True)
class StudentSamplingRequest:
    backend_name: str
    outer_round: int
    student_round: int
    committee_summary_path: str
    seed_xyz_path: str
    work_root: str
    backend_cfg: Any
    reference_xyz_files: list[str] = field(default_factory=list)
    uncertainty_cfg: dict[str, Any] = field(default_factory=dict)
    summary_extra: dict[str, Any] = field(default_factory=dict)
    execution_context: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    committee_model_paths: list[str] = field(default_factory=list)
    backend_identity: dict[str, Any] = field(default_factory=dict)
    stage_execution: dict[str, Any] = field(default_factory=dict)
    md_expansion: dict[str, Any] = field(default_factory=dict)
    native_recipe: dict[str, Any] = field(default_factory=dict)
    scalar_overrides: dict[str, bool] = field(default_factory=dict)
    output_requirements: dict[str, Any] = field(default_factory=dict)

    @property
    def teacher_round(self) -> int:
        return int(self.outer_round)


@dataclass(slots=True)
class StudentEvaluationRequest:
    backend_name: str
    dataset_path: str
    committee_summary_path: str
    backend_cfg: Any
    work_root: str | None = None
    execution_context: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudentSamplingSummary:
    backend: str
    sampling_backend: str
    summary_path: str
    work_root: str
    selected_xyz: str
    candidate_xyz: str
    failed_xyz: str
    uncertainty_records_jsonl: str | None
    effective_sample_count: int
    selected_count: int
    candidate_count: int
    failed_count: int
    high_unc_count: int
    high_unc_ratio: float
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StudentTrainExecutionBackend(Protocol):
    backend_name: str

    def train_committee(self, paths, request: StudentTrainRequest) -> str:
        ...

    def reset_interrupted_training(self, request: StudentTrainRequest) -> None:
        ...


@runtime_checkable
class StudentSamplingExecutionBackend(Protocol):
    backend_name: str

    def sample_with_committee(self, paths, request: StudentSamplingRequest) -> str:
        ...


@runtime_checkable
class StudentSamplingRequestAdapter(Protocol):
    """Build a backend-specific sampling request from canonical stage input only."""

    def build_sampling_request(self, request_input: Any) -> StudentSamplingRequest:
        ...


@runtime_checkable
class StudentModelCatalogBackend(Protocol):
    backend_name: str

    def load_committee_summary(self, summary_path: str | Path) -> StudentCommitteeSummary:
        ...

    def collect_models(self, model_root: str) -> list[CommitteeModelArtifact]:
        ...


@runtime_checkable
class StudentEvaluationBackend(Protocol):
    backend_name: str

    def build_evaluation_request(
        self,
        cfg: dict[str, Any],
        *,
        dataset_path: str | Path,
        committee_summary_path: str | Path,
        work_root: str | Path | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> StudentEvaluationRequest:
        ...

    def evaluate_committee(self, paths, request: StudentEvaluationRequest) -> dict[str, Any]:
        ...


@runtime_checkable
class StudentConfigBackend(Protocol):
    backend_name: str

    def export_contract(self, cfg: dict[str, Any]) -> StudentExportContract:
        ...

    def continuation_contract(self, cfg: dict[str, Any]) -> StudentContinuationContract:
        ...

    def resolve_train_ratio(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> float:
        ...

    def resolve_element_order(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> list[str]:
        ...


class StudentBackend(ABC):
    backend_name = ""
    aliases: tuple[str, ...] = ()

    def matches_name(self, name: str) -> bool:
        normalized = str(name or "").strip().lower()
        return normalized == self.backend_name or normalized in self.aliases

    @abstractmethod
    def export_contract(self, cfg: dict[str, Any]) -> StudentExportContract:
        raise NotImplementedError

    @abstractmethod
    def continuation_contract(self, cfg: dict[str, Any]) -> StudentContinuationContract:
        raise NotImplementedError

    @abstractmethod
    def resolve_train_ratio(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def resolve_element_order(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        training_stage: str | None = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def load_committee_summary(self, summary_path: str | Path) -> StudentCommitteeSummary:
        raise NotImplementedError

    @abstractmethod
    def committee_work_root(self, paths, outer_round: int, student_round: int, *, create: bool = True) -> Path:
        raise NotImplementedError

    @abstractmethod
    def committee_summary_path(self, paths, outer_round: int, student_round: int) -> Path:
        raise NotImplementedError

    @abstractmethod
    def sampling_work_root(self, paths, outer_round: int, student_round: int, *, create: bool = True) -> Path:
        raise NotImplementedError

    @abstractmethod
    def normalize_sampling_summary(
        self,
        paths,
        outer_round: int,
        student_round: int,
        summary_path: str | Path,
        summary_payload: dict[str, Any],
    ) -> StudentSamplingSummary:
        raise NotImplementedError

    @abstractmethod
    def build_train_request(
        self,
        cfg: dict[str, Any],
        paths,
        outer_round: int,
        student_round: int,
        *,
        work_root: str | Path,
        summary_path: str | Path,
        dataset_roots: list[str],
        dataset_path: str | Path | None,
        previous_model_root: str | Path | None,
        continue_training: bool,
        training_stage: str,
        extra_context: dict[str, Any] | None = None,
    ) -> StudentTrainRequest:
        raise NotImplementedError

    @abstractmethod
    def train_committee(self, paths, request: StudentTrainRequest) -> str:
        raise NotImplementedError

    def reset_interrupted_training(self, request: StudentTrainRequest) -> None:
        """Reset incomplete current-stage training state before a resumed run.

        Completed workflow steps are skipped before this hook is called. Backends can
        use this hook to remove stale checkpoints for members that did not finish, so
        a rerun starts from the beginning of the current training stage instead of
        accidentally resuming an interrupted optimizer state.
        """
        del request

    def validate_foundation_bootstrap(
        self,
        cfg: dict[str, Any],
        spec: StudentFoundationBootstrapSpec,
        *,
        element_order: list[str],
    ) -> None:
        del cfg, spec, element_order

    def prepare_foundation_bootstrap_context(
        self,
        cfg: dict[str, Any],
        spec: StudentFoundationBootstrapSpec,
    ) -> dict[str, Any]:
        del cfg
        return {
            "backend_name": str(spec.backend_name),
            "model_path": spec.model_path,
            "restart_path": spec.restart_path,
            "type_map": list(spec.type_map),
            **dict(spec.backend_options or {}),
        }

    def build_sampling_request(self, *args: Any, **kwargs: Any) -> StudentSamplingRequest:
        del args, kwargs
        raise RuntimeError(
            "StudentBackend.build_sampling_request is retired; use the canonical student sampling request adapter"
        )

    @abstractmethod
    def sample_with_committee(
        self,
        paths,
        request: StudentSamplingRequest,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_evaluation_request(
        self,
        cfg: dict[str, Any],
        *,
        dataset_path: str | Path,
        committee_summary_path: str | Path,
        work_root: str | Path | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> StudentEvaluationRequest:
        raise NotImplementedError

    @abstractmethod
    def evaluate_committee(self, paths, request: StudentEvaluationRequest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_uncertainty_outputs(self, task_dir: str, cfg: dict[str, Any]) -> list[UncertaintyRecord]:
        raise NotImplementedError

    @abstractmethod
    def collect_models(self, model_root: str) -> list[CommitteeModelArtifact]:
        raise NotImplementedError
