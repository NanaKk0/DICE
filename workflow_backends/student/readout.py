from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from ase import Atoms

from workflow_backends.observation_source import ReplayableObservationSource
from workflow_policies.gmm_reference_tail import (
    GMMReferenceFitResult,
    GMMScoreResult,
)
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter


@dataclass(frozen=True, slots=True)
class StudentGMMScore:
    values: Mapping[str, float]
    frame_means: Mapping[str, np.ndarray] | None
    fit: GMMReferenceFitResult
    descriptor_schema: Mapping[str, object] | None = None
    compact_view: Any | None = None
    element_frame_means: Mapping[str, Mapping[str, np.ndarray]] | None = None

    @property
    def status(self) -> str:
        return "succeeded"

    def close(self) -> None:
        close = getattr(self.compact_view, "close", None)
        if callable(close):
            close(cleanup=True)


@dataclass(frozen=True, slots=True)
class StudentStructureVectorResult:
    candidate_vectors: Mapping[str, np.ndarray]
    background_vectors: Mapping[str, np.ndarray]
    descriptor_schema: Mapping[str, object]
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]] | None = None
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]] | None = None

    def __post_init__(self) -> None:
        candidates = _validated_vector_mapping(
            self.candidate_vectors, label="candidate"
        )
        background = _validated_vector_mapping(
            self.background_vectors, label="background"
        )
        if not background:
            raise ValueError("student structural-vector background must be non-empty")
        widths = {
            len(vector) for vector in (*candidates.values(), *background.values())
        }
        if len(widths) != 1:
            raise ValueError("student structural vectors must share one feature width")
        schema = dict(self.descriptor_schema)
        if int(schema.get("width", -1)) != next(iter(widths)):
            raise ValueError("student structural vectors do not match descriptor schema")
        if (self.candidate_element_vectors is None) != (
            self.background_element_vectors is None
        ):
            raise ValueError(
                "student structural-vector element evidence must cover both partitions"
            )
        candidate_elements = background_elements = None
        if self.candidate_element_vectors is not None:
            width = next(iter(widths))
            candidate_elements = _validated_element_vector_mapping(
                self.candidate_element_vectors,
                partition_ids=tuple(candidates),
                width=width,
                label="candidate",
            )
            background_elements = _validated_element_vector_mapping(
                self.background_element_vectors or {},
                partition_ids=tuple(background),
                width=width,
                label="background",
            )
        object.__setattr__(self, "candidate_vectors", candidates)
        object.__setattr__(self, "background_vectors", background)
        object.__setattr__(self, "descriptor_schema", schema)
        object.__setattr__(self, "candidate_element_vectors", candidate_elements)
        object.__setattr__(self, "background_element_vectors", background_elements)


@dataclass(frozen=True, slots=True)
class StudentReadoutRequest:
    model: Path
    reference_path: Path
    source: ReplayableObservationSource
    candidates: tuple[Atoms, ...]
    router_dir: Path
    work_dir: Path
    type_map: tuple[str, ...]
    emit_frame_means: bool
    chunk_size: int
    progress_reporter_factory: (
        Callable[[str, str], OperationProgressReporter] | None
    ) = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(frame, Atoms) for frame in candidates):
            raise TypeError("student readout candidates must be ASE Atoms")
        if isinstance(self.chunk_size, bool) or int(self.chunk_size) <= 0:
            raise ValueError("student readout chunk size must be positive")
        object.__setattr__(self, "model", Path(self.model).resolve())
        object.__setattr__(self, "reference_path", Path(self.reference_path).resolve())
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "router_dir", Path(self.router_dir).resolve())
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())
        object.__setattr__(
            self, "type_map", tuple(str(symbol) for symbol in self.type_map)
        )
        object.__setattr__(self, "chunk_size", int(self.chunk_size))

    def progress(
        self, operation_id: str, phase: str
    ) -> OperationProgressReporter | None:
        if self.progress_reporter_factory is None:
            return None
        return self.progress_reporter_factory(operation_id, phase)


@dataclass(frozen=True, slots=True)
class PhaseStudentReferenceRequest:
    model: Path
    reference_path: Path
    router_dir: Path
    work_dir: Path
    type_map: tuple[str, ...]
    progress_reporter: OperationProgressReporter | None = None


@dataclass(frozen=True, slots=True)
class PhaseStudentScoreRequest:
    model: Path
    reference_path: Path
    frame_chunks: Iterable[Sequence[Atoms]]
    ordered_identities: tuple[str, ...]
    router_dir: Path
    work_dir: Path
    emit_frame_means: bool
    progress_reporter: OperationProgressReporter | None = None
    progress_systems: Mapping[str, str] | None = None


class PhaseStudentReadoutOps(Protocol):
    def fit_reference(
        self, request: PhaseStudentReferenceRequest
    ) -> GMMReferenceFitResult: ...

    def score_candidates(self, request: PhaseStudentScoreRequest) -> GMMScoreResult: ...


@dataclass(frozen=True, slots=True)
class FusedStudentReferenceRequest:
    model: Path
    reference_path: Path
    router_dir: Path
    work_dir: Path
    type_map: tuple[str, ...]
    progress_reporter: OperationProgressReporter | None = None


@dataclass(frozen=True, slots=True)
class FusedStudentScreenRequest:
    model: Path
    reference_path: Path
    frame_chunks: Iterable[Sequence[Atoms]]
    ordered_identities: tuple[str, ...]
    router_dir: Path
    work_dir: Path
    type_map: tuple[str, ...]
    emit_frame_means: bool
    reference_progress_reporter: OperationProgressReporter | None = None
    score_progress_reporter: OperationProgressReporter | None = None
    progress_systems: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class FusedStudentReadoutResult:
    fit: GMMReferenceFitResult
    score: Any
    phase_timings: Mapping[str, float]


@runtime_checkable
class FusedStudentReadoutOps(Protocol):
    def fit_reference(
        self, request: FusedStudentReferenceRequest
    ) -> GMMReferenceFitResult: ...

    def fit_and_score(
        self, request: FusedStudentScreenRequest
    ) -> FusedStudentReadoutResult: ...


class StudentReadoutCapability(Protocol):
    def readout_gmm(self, request: StudentReadoutRequest) -> StudentGMMScore: ...


def _validated_vector_mapping(
    values: Mapping[str, np.ndarray], *, label: str
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for raw_identity, raw_vector in values.items():
        identity = str(raw_identity)
        vector = np.asarray(raw_vector, dtype=np.float32)
        if not identity or identity in result:
            raise ValueError(f"student structural-vector {label} identities must be unique")
        if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
            raise ValueError(
                f"student structural-vector {label} rows must be finite vectors"
            )
        result[identity] = vector
    return result


def _validated_element_vector_mapping(
    values: Mapping[str, Mapping[str, np.ndarray]],
    *,
    partition_ids: tuple[str, ...],
    width: int,
    label: str,
) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(values, Mapping):
        raise ValueError(
            f"student structural-vector {label} element evidence must be a mapping"
        )
    positions = {identity: index for index, identity in enumerate(partition_ids)}
    result: dict[str, dict[str, np.ndarray]] = {}
    for raw_element, raw_rows in values.items():
        element = str(raw_element)
        if not element or element in result or not isinstance(raw_rows, Mapping):
            raise ValueError(
                f"student structural-vector {label} element identities are invalid"
            )
        rows: dict[str, np.ndarray] = {}
        indices: list[int] = []
        for raw_identity, raw_vector in raw_rows.items():
            identity = str(raw_identity)
            if identity not in positions or identity in rows:
                raise ValueError(
                    f"student structural-vector {label} element rows contain an unknown identity"
                )
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError(
                    f"student structural-vector {label} element rows must be finite schema-width vectors"
                )
            rows[identity] = vector
            indices.append(positions[identity])
        if indices != sorted(indices):
            raise ValueError(
                f"student structural-vector {label} element rows changed canonical order"
            )
        if rows:
            result[element] = rows
    return result


@dataclass(slots=True)
class StudentNativeReadoutAdapter:
    """Translate the temporary path wrapper into backend-native readout calls."""

    operations: PhaseStudentReadoutOps | FusedStudentReadoutOps

    def readout_gmm(self, request: StudentReadoutRequest) -> StudentGMMScore:
        if isinstance(self.operations, FusedStudentReadoutOps):
            return self._readout_fused(request, self.operations)
        return self._readout_phased(request, self.operations)

    def _readout_phased(
        self,
        request: StudentReadoutRequest,
        operations: PhaseStudentReadoutOps,
    ) -> StudentGMMScore:
        candidates, identities = _canonical_candidates(request.candidates)
        reference_reporter = request.progress(
            "student_gmm_reference", "student reference"
        )
        try:
            fit = operations.fit_reference(
                PhaseStudentReferenceRequest(
                    model=request.model,
                    reference_path=request.reference_path,
                    router_dir=request.router_dir,
                    work_dir=request.work_dir / "reference",
                    type_map=request.type_map,
                    progress_reporter=reference_reporter,
                )
            )
        finally:
            _close_reporter(reference_reporter)
        if not candidates:
            return StudentGMMScore(
                values={},
                frame_means={} if request.emit_frame_means else None,
                fit=fit,
                descriptor_schema=fit.descriptor_schema,
            )

        replay, consumed = _validated_replay(
            request.source,
            identities,
            request.chunk_size,
            label="student GMM",
        )
        progress_systems = {
            identity: str(frame.info.get("system_id", "unknown"))
            for identity, frame in zip(identities, candidates, strict=True)
        }
        score_reporter = request.progress("student_gmm_score", "student scoring")
        try:
            scored = operations.score_candidates(
                PhaseStudentScoreRequest(
                    model=request.model,
                    reference_path=request.reference_path,
                    frame_chunks=replay,
                    ordered_identities=identities,
                    router_dir=request.router_dir,
                    work_dir=request.work_dir / "score_operation",
                    emit_frame_means=request.emit_frame_means,
                    progress_reporter=score_reporter,
                    progress_systems=progress_systems,
                )
            )
            if consumed() != len(identities):
                raise ValueError(
                    "student GMM native operation did not consume the complete canonical candidate order"
                )
        finally:
            _close_reporter(score_reporter)
        return _phase_score(request, identities, fit, scored)

    def _readout_fused(
        self,
        request: StudentReadoutRequest,
        operations: FusedStudentReadoutOps,
    ) -> StudentGMMScore:
        candidates, identities = _canonical_candidates(
            request.candidates, label="DPA4C fused screen"
        )
        if not candidates:
            reporter = request.progress("student_gmm_reference", "student reference")
            try:
                fit = operations.fit_reference(
                    FusedStudentReferenceRequest(
                        model=request.model,
                        reference_path=request.reference_path,
                        router_dir=request.router_dir,
                        work_dir=request.work_dir / "reference",
                        type_map=request.type_map,
                        progress_reporter=reporter,
                    )
                )
            finally:
                _close_reporter(reporter)
            return StudentGMMScore(
                values={},
                frame_means={} if request.emit_frame_means else None,
                fit=fit,
                descriptor_schema=fit.descriptor_schema,
            )

        replay, consumed = _validated_replay(
            request.source,
            identities,
            request.chunk_size,
            label="DPA4C fused",
        )
        progress_systems = {
            identity: str(frame.info.get("system_id", "unknown"))
            for identity, frame in zip(identities, candidates, strict=True)
        }
        reference_reporter = request.progress(
            "student_gmm_reference", "student reference"
        )
        score_reporter = request.progress("student_gmm_score", "student scoring")
        try:
            result = operations.fit_and_score(
                FusedStudentScreenRequest(
                    model=request.model,
                    reference_path=request.reference_path,
                    frame_chunks=replay,
                    ordered_identities=identities,
                    router_dir=request.router_dir,
                    work_dir=request.work_dir / "screen_operation",
                    type_map=request.type_map,
                    emit_frame_means=request.emit_frame_means,
                    reference_progress_reporter=reference_reporter,
                    score_progress_reporter=score_reporter,
                    progress_systems=progress_systems,
                )
            )
            if consumed() != len(identities):
                raise ValueError(
                    "DPA4C fused operation did not consume complete canonical candidates"
                )
        finally:
            _close_reporter(reference_reporter)
            _close_reporter(score_reporter)
        return _fused_score(request, identities, result)


def _canonical_candidates(
    candidates: Sequence[Atoms], *, label: str = "student GMM candidates"
) -> tuple[tuple[Atoms, ...], tuple[str, ...]]:
    ordered = tuple(candidates)
    identities = tuple(str(frame.info.get("structure_id", "")) for frame in ordered)
    if any(not identity for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError(f"{label} require unique non-empty canonical identities")
    return ordered, identities


def _validated_replay(source, identities, chunk_size: int, *, label: str):
    consumed = 0

    def replay():
        nonlocal consumed
        for chunk_items in source.replay(identities, chunk_size=chunk_size):
            chunk = tuple(chunk_items)
            if not chunk or len(chunk) > chunk_size:
                raise ValueError(
                    f"{label} replay violated the bounded canonical candidate order"
                )
            actual = tuple(str(frame.info.get("structure_id", "")) for frame in chunk)
            expected = identities[consumed : consumed + len(chunk)]
            if actual != expected:
                raise ValueError(f"{label} replay changed canonical candidate order")
            consumed += len(chunk)
            yield chunk
        if consumed != len(identities):
            raise ValueError(f"{label} replay omitted canonical candidate order evidence")

    return replay(), lambda: consumed


def _phase_score(request, identities, fit, scored) -> StudentGMMScore:
    accepted = False
    try:
        result = _validated_phase_score(request, identities, fit, scored)
        accepted = True
        return result
    finally:
        if not accepted and callable(getattr(scored, "close", None)):
            scored.close(cleanup=True)


def _validated_phase_score(request, identities, fit, scored) -> StudentGMMScore:
    actual = tuple(str(identity) for identity in (
        scored.frame_ids if isinstance(scored, GMMScoreResult) else scored.identities
    ))
    if actual != identities:
        raise ValueError(
            "student GMM score identities do not match canonical candidate order"
        )
    if scored.descriptor_schema != fit.descriptor_schema or scored.descriptor_schema is None:
        raise ValueError("student GMM reference and score descriptor schemas do not match")
    frame_p = np.asarray(scored.frame_p, dtype=np.float64)
    frame_statistics = np.asarray(scored.frame_statistics, dtype=np.float64)
    if (
        frame_p.shape != (len(actual),)
        or frame_statistics.shape != (len(actual),)
        or not np.isfinite(frame_p).all()
        or not np.isfinite(frame_statistics).all()
    ):
        raise ValueError("student GMM score returned non-finite or incomplete evidence")
    means = None
    if request.emit_frame_means:
        if scored.frame_means is None or scored.frame_means.shape[0] != len(actual):
            raise ValueError("student GMM score is missing requested readout-input frame means")
        if not np.isfinite(scored.frame_means).all():
            raise ValueError("student GMM score returned non-finite readout-input frame means")
        means = {
            identity: np.array(row, dtype=np.float32, copy=True)
            for identity, row in zip(actual, scored.frame_means, strict=True)
        }
    return StudentGMMScore(
        values={
            identity: float(value)
            for identity, value in zip(actual, frame_p, strict=True)
        },
        frame_means=means,
        fit=fit,
        descriptor_schema=scored.descriptor_schema,
        compact_view=None if isinstance(scored, GMMScoreResult) else scored,
        element_frame_means=_element_mapping(getattr(scored, "element_means", None)),
    )


def _fused_score(request, identities, result) -> StudentGMMScore:
    score = result.score
    accepted = False
    try:
        actual = tuple(str(identity) for identity in score.identities)
        if actual != identities or tuple(score.values) != identities:
            raise ValueError("DPA4C fused score identities do not match canonical order")
        if score.descriptor_schema != result.fit.descriptor_schema or score.descriptor_schema is None:
            raise ValueError("DPA4C fused reference and score descriptor schemas differ")
        probabilities = getattr(score, "frame_p", None)
        values = (
            {identity: float(value) for identity, value in zip(identities, probabilities, strict=True)}
            if probabilities is not None else
            {identity: float(score.values[identity]) for identity in identities}
        )
        if not np.isfinite(tuple(values.values())).all():
            raise ValueError("DPA4C fused score returned non-finite evidence")
        means = score.means if request.emit_frame_means else None
        ordinal_means = getattr(score, "frame_means", None)
        if request.emit_frame_means and ordinal_means is not None:
            means = {identity: row for identity, row in zip(identities, ordinal_means, strict=True)}
        if request.emit_frame_means:
            if means is None or tuple(means) != identities:
                raise ValueError("DPA4C fused score is missing requested frame means")
            if any(
                not np.isfinite(np.asarray(means[identity])).all()
                for identity in identities
            ):
                raise ValueError("DPA4C fused score returned non-finite frame means")
        completed = StudentGMMScore(
            values=values,
            frame_means=means,
            fit=result.fit,
            descriptor_schema=score.descriptor_schema,
            compact_view=score,
            element_frame_means=_element_mapping(getattr(score, "element_means", None)),
        )
        accepted = True
        return completed
    finally:
        if not accepted:
            close = getattr(score, "close", None)
            if callable(close):
                close(cleanup=True)


def _close_reporter(reporter) -> None:
    if reporter is not None and not reporter.closed:
        reporter.close()


def _element_mapping(values):
    return None if values is None else {
        element: {identity: np.asarray(row, dtype=np.float32) for identity, row in zip(identities, rows, strict=True)}
        for element, (identities, rows) in values.items()
    }


__all__ = [
    "FusedStudentReadoutOps",
    "FusedStudentReadoutResult",
    "FusedStudentReferenceRequest",
    "FusedStudentScreenRequest",
    "PhaseStudentReadoutOps",
    "PhaseStudentReferenceRequest",
    "PhaseStudentScoreRequest",
    "StudentGMMScore",
    "StudentReadoutCapability",
    "StudentNativeReadoutAdapter",
    "StudentReadoutRequest",
    "StudentStructureVectorResult",
]
