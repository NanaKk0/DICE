from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from ase import Atoms

from workflow_backends.observation_source import ReplayableObservationSource
from workflow_policies.gmm_reference_tail import GMMReferenceFitResult
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter
from workflow_core.structures.formats import write_extxyz_structures
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherReadoutCapability,
    TeacherReferenceIntent,
    TeacherReferenceOutcome,
    TeacherScoreIntent,
    TeacherScoreOutcome,
)


@dataclass(frozen=True, slots=True)
class TeacherReadoutReferenceRequest:
    model: Path
    reference_path: Path
    router_dir: Path
    work_dir: Path
    type_map: tuple[str, ...]
    progress_reporter_factory: (
        Callable[[str, str], OperationProgressReporter] | None
    ) = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", Path(self.model).resolve())
        object.__setattr__(
            self, "reference_path", Path(self.reference_path).resolve()
        )
        object.__setattr__(self, "router_dir", Path(self.router_dir).resolve())
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())
        object.__setattr__(
            self, "type_map", tuple(str(symbol) for symbol in self.type_map)
        )

    def progress(
        self, operation_id: str, phase: str
    ) -> OperationProgressReporter | None:
        if self.progress_reporter_factory is None:
            return None
        return self.progress_reporter_factory(operation_id, phase)


@dataclass(frozen=True, slots=True)
class TeacherReadoutScoreRequest:
    model: Path
    reference_path: Path
    source: ReplayableObservationSource
    candidates: tuple[Atoms, ...]
    router_dir: Path
    work_dir: Path
    emit_frame_means: bool
    chunk_size: int
    reference_descriptor_schema: Mapping[str, object]
    progress_reporter_factory: (
        Callable[[str, str], OperationProgressReporter] | None
    ) = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(frame, Atoms) for frame in candidates):
            raise TypeError("teacher readout candidates must be ASE Atoms")
        if isinstance(self.chunk_size, bool) or int(self.chunk_size) <= 0:
            raise ValueError("teacher readout chunk size must be positive")
        object.__setattr__(self, "model", Path(self.model).resolve())
        object.__setattr__(
            self, "reference_path", Path(self.reference_path).resolve()
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "router_dir", Path(self.router_dir).resolve())
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())
        object.__setattr__(self, "chunk_size", int(self.chunk_size))
        object.__setattr__(
            self,
            "reference_descriptor_schema",
            dict(self.reference_descriptor_schema),
        )

    def progress(
        self, operation_id: str, phase: str
    ) -> OperationProgressReporter | None:
        if self.progress_reporter_factory is None:
            return None
        return self.progress_reporter_factory(operation_id, phase)


@dataclass(frozen=True, slots=True)
class TeacherReferenceFitRequest:
    model: Path
    reference_path: Path
    router_dir: Path
    work_dir: Path
    type_map: tuple[str, ...]
    progress_reporter: OperationProgressReporter | None = None


@dataclass(frozen=True, slots=True)
class TeacherCandidateScoreRequest:
    model: Path
    reference_path: Path
    frame_chunks: Iterable[Sequence[Atoms]]
    ordered_identities: tuple[str, ...]
    router_dir: Path
    work_dir: Path
    emit_frame_means: bool
    progress_reporter: OperationProgressReporter | None = None
    progress_systems: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class TeacherCandidateScoreResult:
    ordered_identities: tuple[str, ...]
    frame_p: np.ndarray
    frame_statistics: np.ndarray
    frame_means: np.ndarray | None
    descriptor_schema: Mapping[str, object] | None
    compact_view: Any | None = None


@dataclass(frozen=True, slots=True)
class TeacherStructureVectors:
    candidate_vectors: Mapping[str, np.ndarray]
    background_vectors: Mapping[str, np.ndarray]
    descriptor_schema: Mapping[str, object]
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]]
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]]


def partitioned_teacher_structure_vectors(
    executor: object,
    *,
    model: Path,
    candidate_frames: Sequence[Atoms],
    background_frames: Sequence[Atoms],
    type_map: tuple[str, ...],
    work_dir: Path,
) -> TeacherStructureVectors:
    candidates = tuple(candidate_frames)
    background = tuple(background_frames)
    candidate_ids = tuple(str(frame.info.get("structure_id", "")) for frame in candidates)
    background_ids = tuple(str(frame.info.get("structure_id", "")) for frame in background)
    if any(not identity for identity in (*candidate_ids, *background_ids)):
        raise ValueError("teacher structural vectors require canonical identities")
    combined: list[Atoms] = []
    tokens: list[str] = []
    for partition, frames, identities in (
        ("candidate", candidates, candidate_ids),
        ("background", background, background_ids),
    ):
        for ordinal, (frame, identity) in enumerate(zip(frames, identities, strict=True)):
            item = frame.copy()
            token = f"{partition}:{ordinal:08d}:{identity}"
            item.info["structure_id"] = token
            combined.append(item)
            tokens.append(token)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    input_path = write_extxyz_structures(
        work / "teacher_structure_vectors.xyz", combined, allow_empty=False
    )
    operation = getattr(executor, "structure_vectors", None)
    if not callable(operation):
        raise ValueError("teacher backend omitted structural-vector capability")
    schema, actual, rows, raw_elements = operation(
        model=Path(model),
        input_path=input_path,
        type_map=type_map,
        work_dir=work / "operation",
    )
    if tuple(actual) != tuple(tokens):
        raise ValueError("teacher structural readout changed partition order")
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(tokens) or not np.isfinite(values).all():
        raise ValueError("teacher structural readout returned invalid vectors")
    if not isinstance(raw_elements, Mapping):
        raise ValueError("teacher structural readout omitted per-element vectors")
    allowed_elements = tuple(str(element) for element in type_map)
    if set(raw_elements) - set(allowed_elements):
        raise ValueError("teacher structural readout returned an unknown element")
    for element in allowed_elements:
        expected_ids = tuple(
            token
            for token, frame in zip(tokens, combined, strict=True)
            if element in frame.get_chemical_symbols()
        )
        if not expected_ids:
            if element in raw_elements and len(raw_elements[element][0]):
                raise ValueError(
                    "teacher structural readout returned evidence for an absent element"
                )
            continue
        if element not in raw_elements:
            raise ValueError(
                "teacher structural readout omitted a chemically present element"
            )
        raw_ids, element_rows = raw_elements[element]
        element_values = np.asarray(element_rows, dtype=np.float32)
        if (
            tuple(raw_ids) != expected_ids
            or element_values.ndim != 2
            or len(element_values) != len(expected_ids)
            or element_values.shape[1:] != values.shape[1:]
            or not np.isfinite(element_values).all()
        ):
            raise ValueError(
                "teacher structural readout returned invalid ordered per-element vectors"
            )
    split = len(candidates)

    def elements_for(token_ids, identities):
        token_to_identity = dict(zip(token_ids, identities, strict=True))
        result = {}
        for element, (raw_ids, element_rows) in raw_elements.items():
            mapped = {
                token_to_identity[token]: np.asarray(row, dtype=np.float32)
                for token, row in zip(raw_ids, element_rows, strict=True)
                if token in token_to_identity
            }
            if mapped:
                result[str(element)] = mapped
        return result

    return TeacherStructureVectors(
        candidate_vectors=dict(zip(candidate_ids, values[:split], strict=True)),
        background_vectors=dict(zip(background_ids, values[split:], strict=True)),
        descriptor_schema=dict(schema),
        candidate_element_vectors=elements_for(tokens[:split], candidate_ids),
        background_element_vectors=elements_for(tokens[split:], background_ids),
    )


class TeacherReadoutOps(Protocol):
    def fit_reference(
        self, request: TeacherReferenceFitRequest
    ) -> GMMReferenceFitResult: ...

    def score_candidates(
        self, request: TeacherCandidateScoreRequest
    ) -> TeacherCandidateScoreResult: ...


__all__ = [
    "TeacherReadoutCapability",
    "TeacherReferenceIntent",
    "TeacherReferenceOutcome",
    "TeacherScoreIntent",
    "TeacherScoreOutcome",
    "TeacherCandidateScoreRequest",
    "TeacherCandidateScoreResult",
    "TeacherStructureVectors",
    "partitioned_teacher_structure_vectors",
    "TeacherReadoutOps",
    "TeacherReadoutReferenceRequest",
    "TeacherReadoutScoreRequest",
    "TeacherReferenceFitRequest",
]
