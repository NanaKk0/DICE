from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class TrajectoryFrameRef:
    trajectory_path: str | Path
    trajectory_format: str
    step: int | None = None
    frame_index: int | None = None
    fallback_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingObservation:
    task_id: str
    force_std_max: float
    force_std_avg: float | None = None
    force_std_p90: float | None = None
    step: int | None = None
    frame_index: int | None = None
    temperature: float | None = None
    atoms: Any | None = None
    trajectory_ref: TrajectoryFrameRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingBackendResult:
    backend: str
    sampling_backend: str
    work_root: str | Path
    jobs: list[dict[str, Any]]
    observations: list[SamplingObservation]
    task_summaries: list[dict[str, Any]] = field(default_factory=list)
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class MDSamplingRequest:
    student_backend: str
    teacher_round: int
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


@runtime_checkable
class SamplingExecutionBackend(Protocol):
    backend_name: str

    def sample_with_committee(self, paths, request: MDSamplingRequest) -> str:
        ...

    def normalize_sampling_summary(
        self,
        paths,
        request: MDSamplingRequest,
        summary_path: str | Path,
        summary_payload: dict[str, Any],
    ) -> Any:
        ...
