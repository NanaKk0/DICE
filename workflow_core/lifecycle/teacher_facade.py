from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from workflow_core.lifecycle.gmm import TeacherGMMScore
from workflow_core.lifecycle.data import LabelContract, validate_label_properties
from workflow_core.lifecycle.model_handles import (
    ModelArtifact,
    PublishedModelHandle,
    publish_model_handles,
    resolve_model_handles,
)
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherLabelIntent,
    TeacherLabelOutcome,
    TeacherLabelingCapability,
    TeacherModelCapability,
    TeacherModelHandle,
    TeacherOutcomeStatus,
    TeacherReadoutCapability,
    TeacherReferenceArtifactFacts,
    TeacherReferenceIntent,
    TeacherReferenceOutcome,
    TeacherScoreIntent,
    TeacherScoreOutcome,
    TeacherTrainingCapability,
    TeacherTrainingIntent,
    TeacherTrainingOutcome,
    TEACHER_LABEL_ARTIFACT_KIND,
    TEACHER_READOUT_ARTIFACT_KIND,
    TeacherTrainingArtifactFacts,
    TeacherStructureArtifactFacts,
    require_teacher_models,
    require_teacher_reference,
    require_teacher_structures,
)
from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
from workflow_core.structures.canonical_identity import (
    copy_atoms_preserving_singlepoint_results,
)
from workflow_core.structures.exact_identity import exact_structure_hash


@dataclass(frozen=True, slots=True)
class TeacherStructureVectorResult:
    candidate_vectors: Mapping[str, np.ndarray]
    background_vectors: Mapping[str, np.ndarray]
    descriptor_schema: Mapping[str, object]
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]] | None = None
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]] | None = None


@dataclass(frozen=True, slots=True)
class _TeacherStructureReadoutRequest:
    model: Path
    reference_path: Path
    source: Any
    candidates: tuple[Any, ...]
    router_dir: Path
    work_dir: Path
    emit_frame_means: bool
    chunk_size: int
    reference_descriptor_schema: Mapping[str, object]
    report: Any = None

    def progress(self, operation_id: str, phase: str):
        operation = getattr(self.report, "operation", None)
        return None if not callable(operation) else operation(operation_id, phase=phase)


class _TeacherStructureSource:
    def __init__(self, frames: Sequence[Any]) -> None:
        self._frames = {
            str(frame.info["structure_id"]): frame.copy() for frame in frames
        }

    def replay(self, ordered_identities, *, chunk_size: int):
        identities = tuple(str(identity) for identity in ordered_identities)
        for start in range(0, len(identities), int(chunk_size)):
            yield tuple(
                self._frames[identity].copy()
                for identity in identities[start : start + int(chunk_size)]
            )


@dataclass(slots=True)
class TeacherTrainingUseCase:
    """Validate lifecycle member semantics around one logic-owned train port."""

    capability: TeacherTrainingCapability

    def execute(
        self,
        request: TeacherTrainingIntent,
        expected_artifacts: ExpectedArtifactSemantics[
            TeacherTrainingArtifactFacts
        ] | None = None,
    ) -> TeacherTrainingOutcome:
        if expected_artifacts is not None:
            require_teacher_models(
                expected_artifacts,
                member_ids=request.member_ids,
                dataset=request.dataset,
            )
        outcome = self.capability.train(request)
        actual = tuple(model.member_id for model in outcome.models)
        expected = request.member_ids
        if outcome.status in {
            TeacherOutcomeStatus.SUCCEEDED,
            TeacherOutcomeStatus.VALIDATED_REUSE,
        }:
            if actual != expected:
                raise ValueError(
                    "completed teacher training must return the exact ordered members"
                )
        elif outcome.status is TeacherOutcomeStatus.PARTIAL:
            if not actual or actual == expected or not _is_ordered_subsequence(
                actual, expected
            ):
                raise ValueError(
                    "partial teacher training must identify a proper ordered member subset"
                )
        elif actual:
            raise ValueError("failed teacher training cannot return reusable models")
        return outcome


@dataclass(slots=True)
class TeacherLabelingUseCase:
    """Keep canonical structure order authoritative around one label port."""

    capability: TeacherLabelingCapability

    def execute(
        self,
        request: TeacherLabelIntent,
        expected_artifacts: ExpectedArtifactSemantics[
            TeacherStructureArtifactFacts
        ] | None = None,
    ) -> TeacherLabelOutcome:
        if expected_artifacts is not None:
            require_teacher_structures(
                expected_artifacts,
                artifact_kind=TEACHER_LABEL_ARTIFACT_KIND,
                structures=request.structures,
            )
        outcome = self.capability.label(request)
        actual = outcome.labeled_structure_ids
        expected = request.structures.ordered_structure_ids
        if outcome.status in {
            TeacherOutcomeStatus.SUCCEEDED,
            TeacherOutcomeStatus.VALIDATED_REUSE,
        }:
            if actual != expected:
                raise ValueError(
                    "completed teacher labeling must restore the exact canonical order"
                )
        elif outcome.status is TeacherOutcomeStatus.PARTIAL:
            if not actual or actual == expected or not _is_ordered_subsequence(
                actual, expected
            ):
                raise ValueError(
                    "partial teacher labeling must identify a proper ordered structure subset"
                )
        elif actual:
            raise ValueError("failed teacher labeling cannot return reusable labels")
        return outcome


@dataclass(slots=True)
class TeacherReadoutUseCase:
    """Keep descriptor compatibility, order and finite evidence in logic."""

    capability: TeacherReadoutCapability

    def fit_reference(
        self,
        request: TeacherReferenceIntent,
        expected_artifacts: ExpectedArtifactSemantics[
            TeacherReferenceArtifactFacts
        ] | None = None,
    ) -> TeacherReferenceOutcome:
        expected_schema = None
        if expected_artifacts is not None:
            expected_schema = require_teacher_reference(
                expected_artifacts,
                model=request.model,
                reference=request.reference,
            )
        outcome = self.capability.fit_reference(request)
        if expected_schema is not None and outcome.descriptor_schema != expected_schema:
            raise ValueError("teacher reference fit changed descriptor semantics")
        return outcome

    def score(
        self,
        request: TeacherScoreIntent,
        expected_artifacts: ExpectedArtifactSemantics[
            TeacherStructureArtifactFacts
        ] | None = None,
    ) -> TeacherScoreOutcome:
        if expected_artifacts is not None:
            facts = require_teacher_structures(
                expected_artifacts,
                artifact_kind=TEACHER_READOUT_ARTIFACT_KIND,
                structures=request.candidates,
            )
            if (
                facts.descriptor_schema_name
                != request.reference_descriptor_schema.name
                or facts.descriptor_schema_revision
                != request.reference_descriptor_schema.revision
                or facts.emit_frame_means != request.emit_frame_means
            ):
                raise ValueError("teacher readout artifact expectation changed schema intent")
        outcome = self.capability.score_candidates(request)
        if outcome.ordered_structure_ids != request.candidates.ordered_structure_ids:
            raise ValueError("teacher score changed canonical candidate order")
        if outcome.descriptor_schema != request.reference_descriptor_schema:
            raise ValueError("teacher score descriptor schema does not match the reference")
        if request.emit_frame_means != (outcome.frame_means is not None):
            raise ValueError("teacher score frame-mean evidence does not match the request")
        return outcome


@dataclass(slots=True)
class TeacherModelUseCase:
    """Expose strict semantic validation and historical recovery without paths."""

    capability: TeacherModelCapability

    def validate(
        self,
        models: tuple[TeacherModelHandle, ...],
        expected_artifacts: ExpectedArtifactSemantics[
            TeacherTrainingArtifactFacts
        ] | None = None,
    ) -> bool:
        ordered = tuple(models)
        member_ids = tuple(model.member_id for model in ordered)
        if not ordered or len(set(member_ids)) != len(member_ids):
            raise ValueError("teacher model validation requires ordered unique members")
        if expected_artifacts is not None and require_teacher_models(
            expected_artifacts,
            member_ids=member_ids,
        ) != ordered:
            raise ValueError("teacher model validation changed canonical model identities")
        return bool(self.capability.validate_published_training(ordered))

    def recover(self, member_id: str) -> TeacherModelHandle:
        expected = str(member_id).strip()
        if not expected:
            raise ValueError("teacher historical recovery requires a member identity")
        recovered = self.capability.recover_historical_model(expected)
        if recovered.member_id != expected:
            raise ValueError("teacher historical recovery changed the requested member")
        return recovered


def _is_ordered_subsequence(actual: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    iterator = iter(expected)
    return all(any(candidate == item for candidate in iterator) for item in actual)


class _TeacherReadoutOps(Protocol):
    def fit_reference(self, request: object) -> Any: ...

    def score_candidates(self, request: object) -> Any: ...


class _TeacherTrainingOps(Protocol):
    def train(self, request: object) -> Any: ...

    def finalize_published_training(self, request: object) -> None: ...


class _TeacherLabelOps(Protocol):
    def label(self, request: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class _TeacherReferenceFitRequest:
    model: Any
    reference_path: Any
    router_dir: Any
    work_dir: Any
    type_map: tuple[str, ...]
    progress_reporter: Any = None


@dataclass(frozen=True, slots=True)
class _TeacherCandidateScoreRequest:
    model: Any
    reference_path: Any
    frame_chunks: Any
    ordered_identities: tuple[str, ...]
    router_dir: Any
    work_dir: Any
    emit_frame_means: bool
    progress_reporter: Any = None
    progress_systems: Any = None


@dataclass(frozen=True, slots=True)
class _TeacherNativeLabelRequest:
    models: Any
    frames: Any
    work_dir: Any
    trusted_below: float | None = None
    progress_reporter: Any = None


@dataclass(frozen=True, slots=True)
class _TeacherTrainPublicationRequest:
    train: Any
    handles: tuple[PublishedModelHandle, ...]
    stage_root: Any
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        returned = tuple(item.member_id for item in self.train.member_artifacts)
        published = tuple(item.member_id for item in self.handles)
        if returned != self.member_ids or published != self.member_ids:
            raise ValueError(
                "teacher publication requires ordered train artifacts and handles"
            )


@dataclass(frozen=True, slots=True)
class _TeacherPublishedTrainingValidationRequest:
    stage_root: Any
    handles: tuple[PublishedModelHandle, ...]
    member_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TeacherHistoricalModelRequest:
    member_id: str
    legacy_handle: Any


@dataclass(slots=True)
class LifecycleTeacherReadoutCoordinator:
    """Own canonical replay and validation around backend-native readout ops."""

    readout_ops: _TeacherReadoutOps

    def fit_reference(self, request: Any):
        reporter = request.progress("gmm_reference", "descriptors")
        try:
            return self.readout_ops.fit_reference(
                _TeacherReferenceFitRequest(
                    model=request.model,
                    reference_path=request.reference_path,
                    router_dir=request.router_dir,
                    work_dir=request.work_dir,
                    type_map=request.type_map,
                    progress_reporter=reporter,
                )
            )
        finally:
            if reporter is not None and not reporter.closed:
                reporter.close()

    def score_candidates(
        self, request: Any
    ) -> TeacherGMMScore:
        candidates = tuple(request.candidates)
        identities = tuple(
            str(frame.info.get("structure_id", "")) for frame in candidates
        )
        if any(not identity for identity in identities) or len(set(identities)) != len(
            identities
        ):
            raise ValueError(
                "teacher GMM candidates require unique non-empty canonical identities"
            )
        progress_systems = {
            identity: str(frame.info.get("system_id", "unknown"))
            for identity, frame in zip(identities, candidates, strict=True)
        }
        replay_complete = False

        def validated_replay():
            nonlocal replay_complete
            consumed = 0
            for chunk_items in request.source.replay(
                identities, chunk_size=request.chunk_size
            ):
                chunk = tuple(chunk_items)
                if not chunk or len(chunk) > request.chunk_size:
                    raise ValueError(
                        "teacher GMM replay violated the bounded canonical candidate order"
                    )
                expected = identities[consumed : consumed + len(chunk)]
                actual = tuple(
                    str(frame.info.get("structure_id", "")) for frame in chunk
                )
                if actual != expected:
                    raise ValueError(
                        "teacher GMM replay changed canonical candidate order"
                    )
                canonical = candidates[consumed : consumed + len(chunk)]
                for replayed, metadata in zip(chunk, canonical, strict=True):
                    replayed.info.update(metadata.info)
                consumed += len(chunk)
                yield chunk
            if consumed != len(identities):
                raise ValueError(
                    "teacher GMM replay omitted canonical candidate order evidence"
                )
            replay_complete = True

        reporter = request.progress("teacher_gmm_score", "teacher scoring")
        result = None
        try:
            result = self.readout_ops.score_candidates(
                _TeacherCandidateScoreRequest(
                    model=request.model,
                    reference_path=request.reference_path,
                    frame_chunks=validated_replay(),
                    ordered_identities=identities,
                    router_dir=request.router_dir,
                    work_dir=request.work_dir,
                    emit_frame_means=request.emit_frame_means,
                    progress_reporter=reporter,
                    progress_systems=progress_systems,
                )
            )
            if not replay_complete:
                raise ValueError(
                    "teacher GMM native operation did not consume the complete canonical candidate order"
                )
        finally:
            if reporter is not None and not reporter.closed:
                reporter.close()

        actual = tuple(str(identity) for identity in result.ordered_identities)
        if actual != identities:
            raise ValueError(
                "teacher GMM score identities do not match canonical candidate order"
            )
        schema = result.descriptor_schema
        if schema is None or dict(schema) != dict(request.reference_descriptor_schema):
            raise ValueError(
                "teacher GMM reference and score descriptor schemas do not match"
            )
        frame_p = np.asarray(result.frame_p, dtype=np.float64)
        statistics = np.asarray(result.frame_statistics, dtype=np.float64)
        if (
            frame_p.shape != (len(actual),)
            or statistics.shape != (len(actual),)
            or not np.isfinite(frame_p).all()
            or not np.isfinite(statistics).all()
        ):
            raise ValueError("teacher GMM score returned non-finite or incomplete evidence")
        means = result.frame_means
        if request.emit_frame_means:
            expected_width = schema.get("width")
            means_array = None if means is None else np.asarray(means)
            if (
                isinstance(expected_width, bool)
                or not isinstance(expected_width, int)
                or expected_width <= 0
                or means_array is None
                or means_array.shape != (len(actual), expected_width)
            ):
                raise ValueError(
                    "teacher GMM score is missing requested readout-input frame means"
                )
            if not np.isfinite(means_array).all():
                raise ValueError(
                    "teacher GMM score returned non-finite readout-input frame means"
                )
        elif means is not None:
            raise ValueError(
                "teacher GMM score unexpectedly returned readout-input frame means"
            )
        scored_frames = []
        for frame, probability in zip(candidates, frame_p, strict=True):
            item = copy_atoms_preserving_singlepoint_results(frame)
            item.info["teacher_p"] = float(probability)
            scored_frames.append(item)
        return TeacherGMMScore(
            frames=tuple(scored_frames),
            frame_means=means,
            descriptor_schema=dict(schema),
            compact_view=result.compact_view,
        )


@dataclass(slots=True)
class LifecycleTeacherTrainingCoordinator:
    """Publish ordered teacher terminals before backend-owned finalization."""

    training_ops: _TeacherTrainingOps

    def train_and_publish(
        self, request: Any, *, stage_root
    ) -> tuple[PublishedModelHandle, ...]:
        result = self.training_ops.train(request)
        artifacts = tuple(result.member_artifacts)
        if any(not isinstance(artifact, ModelArtifact) for artifact in artifacts):
            raise TypeError(
                "teacher training must return core-publishable model artifacts"
            )
        returned = tuple(artifact.member_id for artifact in artifacts)
        if returned != request.member_ids:
            raise ValueError(
                "teacher training artifacts must match ordered requested members"
            )
        handles = publish_model_handles(
            artifacts,
            expected_member_ids=request.member_ids,
            stage_root=stage_root,
            role="teacher",
        )
        self.training_ops.finalize_published_training(
            _TeacherTrainPublicationRequest(
                train=result,
                handles=handles,
                stage_root=stage_root,
                member_ids=request.member_ids,
            )
        )
        return handles


_CANONICAL_LABEL_METADATA = (
    "structure_id",
    "system_id",
    "structure_source",
    "teacher_p",
    "student_p",
    "acquisition_route",
)
_FORBIDDEN_NATIVE_LABEL_METADATA = (
    "internal_type_index",
    "deepmd_type_index",
    "type_index",
    "atom_types",
)


@dataclass(slots=True)
class LifecycleTeacherLabelCoordinator:
    """Apply canonical label invariants after backend-native parsing."""

    label_ops: _TeacherLabelOps
    label_contract: LabelContract

    def label(self, request: Any) -> list:
        sources = tuple(request.frames)
        expected_ids = tuple(
            str(frame.info.get("structure_id", "")) for frame in sources
        )
        if any(not identity for identity in expected_ids) or len(
            set(expected_ids)
        ) != len(expected_ids):
            raise ValueError(
                "teacher label requires unique non-empty canonical identities"
            )
        reporter = request.progress("labeling", "labeling")
        try:
            native = self.label_ops.label(
                _TeacherNativeLabelRequest(
                    models=request.models,
                    frames=sources,
                    work_dir=request.work_dir,
                    trusted_below=request.trusted_below,
                    progress_reporter=reporter,
                )
            )
        finally:
            if reporter is not None and not reporter.closed:
                reporter.close()
        if len(native.frames) != len(sources):
            raise ValueError(
                "teacher native label result changed the canonical frame count"
            )
        actual_ids = tuple(
            str(frame.info.get("structure_id", "")) for frame in native.frames
        )
        if actual_ids != expected_ids:
            raise ValueError(
                "teacher native label identities do not match canonical frame order"
            )
        canonical = []
        for index, (source, labeled) in enumerate(
            zip(sources, native.frames, strict=True)
        ):
            if exact_structure_hash(labeled) != exact_structure_hash(source):
                raise ValueError(
                    f"teacher native label changed canonical geometry at frame {index}"
                )
            for field_name in ("system_id", "structure_source"):
                if field_name in source.info and labeled.info.get(field_name) != source.info[field_name]:
                    raise ValueError(
                        f"teacher native label changed canonical {field_name} at frame {index}"
                    )
            item = copy_atoms_preserving_singlepoint_results(labeled)
            for field_name in _CANONICAL_LABEL_METADATA:
                if field_name in source.info:
                    item.info[field_name] = source.info[field_name]
            for field_name in _FORBIDDEN_NATIVE_LABEL_METADATA:
                item.info.pop(field_name, None)
                item.arrays.pop(field_name, None)
            canonical.append(item)
        validate_label_properties(canonical, self.label_contract)
        return canonical


@dataclass(slots=True)
class LifecycleTeacherFacade:
    """The single lifecycle-owned facade around one route-bound teacher role."""

    delegate: object
    label_contract: LabelContract
    _training: LifecycleTeacherTrainingCoordinator = field(init=False, repr=False)
    _labeling: LifecycleTeacherLabelCoordinator = field(init=False, repr=False)
    _readout: LifecycleTeacherReadoutCoordinator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for method_name in (
            "train",
            "finalize_published_training",
            "label",
            "validate_published_training",
            "recover_historical_model",
        ):
            if not callable(getattr(self.delegate, method_name, None)):
                raise TypeError(
                    f"bound teacher role is missing required capability {method_name!r}"
                )
        try:
            readout_ops = self.delegate.readout_ops
        except AttributeError as exc:
            raise TypeError("bound teacher role is missing typed readout operations")
        self._training = LifecycleTeacherTrainingCoordinator(self.delegate)
        self._labeling = LifecycleTeacherLabelCoordinator(
            self.delegate, self.label_contract
        )
        self._readout = LifecycleTeacherReadoutCoordinator(readout_ops)

    @property
    def readout_ops(self) -> object:
        return self.delegate.readout_ops

    @property
    def screen_operation_profiles(self):
        return self.delegate.screen_operation_profiles

    def train_and_publish(
        self, request: Any, *, stage_root
    ) -> tuple[PublishedModelHandle, ...]:
        return self._training.train_and_publish(request, stage_root=stage_root)

    def label(self, request: Any) -> list:
        return self._labeling.label(request)

    def fit_reference(self, request: Any):
        return self._readout.fit_reference(request)

    def score_candidates(self, request: Any) -> TeacherGMMScore:
        return self._readout.score_candidates(request)

    def read_structure_vectors(
        self,
        *,
        model: str | Path,
        candidate_frames: Sequence[Any],
        background_frames: Sequence[Any],
        reference_path: str | Path,
        router_dir: str | Path,
        work_dir: str | Path,
        chunk_size: int,
        reference_descriptor_schema: Mapping[str, object],
        report: Any = None,
        type_map: Sequence[str] | None = None,
    ) -> TeacherStructureVectorResult:
        """Read one candidate/background population through the bound teacher."""
        candidates = tuple(candidate_frames)
        background = tuple(background_frames)
        candidate_ids = tuple(str(frame.info.get("structure_id", "")) for frame in candidates)
        background_ids = tuple(str(frame.info.get("structure_id", "")) for frame in background)
        if any(not identity for identity in (*candidate_ids, *background_ids)):
            raise ValueError("joint DIRECT teacher structures require canonical identities")
        direct = getattr(self.readout_ops, "read_structure_vectors", None)
        if callable(direct):
            resolved_type_map = (
                tuple(str(item) for item in type_map)
                if type_map is not None
                else tuple(
                    dict.fromkeys(
                        symbol
                        for frame in (*candidates, *background)
                        for symbol in frame.get_chemical_symbols()
                    )
                )
            )
            outcome = direct(
                model=Path(model),
                candidate_frames=candidates,
                background_frames=background,
                type_map=resolved_type_map,
                work_dir=Path(work_dir),
            )
            return TeacherStructureVectorResult(
                candidate_vectors=outcome.candidate_vectors,
                background_vectors=outcome.background_vectors,
                descriptor_schema=outcome.descriptor_schema,
                candidate_element_vectors=outcome.candidate_element_vectors,
                background_element_vectors=outcome.background_element_vectors,
            )
        combined = []
        tokens = []
        for partition, frames, identities in (
            ("candidate", candidates, candidate_ids),
            ("background", background, background_ids),
        ):
            for ordinal, (frame, identity) in enumerate(
                zip(frames, identities, strict=True)
            ):
                item = frame.copy()
                token = f"{partition}:{ordinal:08d}:{identity}"
                item.info["structure_id"] = token
                combined.append(item)
                tokens.append(token)
        request = _TeacherStructureReadoutRequest(
            model=Path(model),
            reference_path=Path(reference_path),
            source=_TeacherStructureSource(combined),
            candidates=tuple(combined),
            router_dir=Path(router_dir),
            work_dir=Path(work_dir),
            emit_frame_means=True,
            chunk_size=int(chunk_size),
            reference_descriptor_schema=dict(reference_descriptor_schema),
            report=report,
        )
        scored = self._readout.score_candidates(request)
        try:
            rows = np.array(
                scored.frame_means,
                dtype=np.float32,
                order="C",
                copy=True,
            )
            if (
                rows.ndim != 2
                or rows.shape[0] != len(tokens)
                or not np.isfinite(rows).all()
            ):
                raise ValueError("teacher structural readout returned invalid vectors")
            split = len(candidates)
            return TeacherStructureVectorResult(
                candidate_vectors=dict(
                    zip(candidate_ids, rows[:split], strict=True)
                ),
                background_vectors=dict(
                    zip(background_ids, rows[split:], strict=True)
                ),
                descriptor_schema=dict(scored.descriptor_schema or {}),
            )
        finally:
            scored.close()

    def resolve_published_models(
        self, *, stage_root, member_ids: tuple[str, ...]
    ) -> tuple[PublishedModelHandle, ...]:
        def recover(member_id, legacy_handle):
            result = self.delegate.recover_historical_model(
                _TeacherHistoricalModelRequest(member_id, legacy_handle)
            )
            if result.member_id != member_id:
                raise ValueError(
                    "teacher historical recovery changed the requested member identity"
                )
            return result.path

        return resolve_model_handles(
            stage_root=stage_root,
            role="teacher",
            expected_member_ids=member_ids,
            legacy_resolver=recover,
        )

    def validate_published_training(
        self, *, stage_root, member_ids: tuple[str, ...]
    ) -> bool:
        try:
            handles = self.resolve_published_models(
                stage_root=stage_root, member_ids=member_ids
            )
            return bool(
                self.delegate.validate_published_training(
                    _TeacherPublishedTrainingValidationRequest(
                        stage_root=stage_root,
                        handles=handles,
                        member_ids=member_ids,
                    )
                )
            )
        except (OSError, TypeError, ValueError):
            return False


__all__ = [
    "TeacherLabelingUseCase",
    "TeacherModelUseCase",
    "TeacherReadoutUseCase",
    "TeacherTrainingUseCase",
    "LifecycleTeacherReadoutCoordinator",
    "LifecycleTeacherLabelCoordinator",
    "LifecycleTeacherTrainingCoordinator",
    "LifecycleTeacherFacade",
    "TeacherStructureVectorResult",
]
