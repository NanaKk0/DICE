from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from workflow_core.lifecycle.descriptor_stream import validated_readout_input_schema
from workflow_core.lifecycle.artifact_handles import (
    DatasetHandle,
    ModelHandle,
    StructureBatchHandle,
)
from workflow_core.lifecycle.student_capabilities import StudentReadoutIntent
from workflow_core.lifecycle.teacher_capabilities import (
    TeacherDatasetHandle,
    TeacherDescriptorSchema,
    TeacherScoreIntent,
    TeacherScoreOutcome,
)
from workflow_core.lifecycle.teacher_facade import TeacherReadoutUseCase
from workflow_core.lifecycle.gmm import TeacherGMMScore
from workflow_core.structures.canonical_identity import (
    copy_atoms_preserving_singlepoint_results,
)
from workflow_core.structures.formats import read_extxyz_structures

from workflow_core.lifecycle.routes import RouteKind, RouteSpec


@dataclass(frozen=True, slots=True)
class LifecycleScreenIOAdapter:
    """Bind physical screening paths and provider DTOs at the app boundary."""

    config: Any
    teacher: Any
    student: Any
    gmm_score_chunk_size: int
    student_readout_request_factory: Any
    teacher_score_request_factory: Any
    teacher_label_request_factory: Any
    router_validator: Any
    artifacts: Any
    cpu_pool_factory: Any = None

    @classmethod
    def bind(
        cls,
        *,
        config: Any,
        teacher: Any,
        student: Any,
        gmm_score_chunk_size: int,
        student_readout_request_factory: Any,
        teacher_score_request_factory: Any,
        teacher_label_request_factory: Any,
        router_validator: Any,
        artifacts: Any,
        cpu_pool_factory: Any = None,
    ) -> LifecycleScreenIOAdapter:
        route = config.route
        if not isinstance(route, RouteSpec):
            raise TypeError("lifecycle screen I/O requires a resolved RouteSpec")
        return cls(
            config=config,
            teacher=teacher,
            student=student,
            gmm_score_chunk_size=_positive_int(gmm_score_chunk_size),
            student_readout_request_factory=student_readout_request_factory,
            teacher_score_request_factory=teacher_score_request_factory,
            teacher_label_request_factory=teacher_label_request_factory,
            router_validator=router_validator,
            artifacts=artifacts,
            cpu_pool_factory=cpu_pool_factory,
        )

    def score_teacher(
        self,
        *,
        frames,
        source,
        work_dir,
        emit_frame_means,
        report,
        round_index,
    ):
        del round_index
        router_dir = (
            self.config.root
            / f"iter{self.config.iter_index}"
            / "teach"
            / "scan"
            / "gmm_router"
        )
        manifest = self.router_validator(router_dir)
        reference_schema = manifest.get("descriptor_schema")
        if not isinstance(reference_schema, dict):
            raise ValueError(
                "teacher GMM router omitted its readout-input descriptor schema"
            )
        models = _teacher_models(self.config, self.teacher)
        if len(models) != 1:
            raise ValueError("teacher GMM requires exactly one canonical model")
        reference_path = (
            self.config.root
            / f"iter{self.config.iter_index}"
            / "teach"
            / "train"
            / "dataset"
            / "train.xyz"
        )
        reference_ids = _structure_ids(read_extxyz_structures(reference_path))
        candidate_frames = tuple(frames)
        candidate_ids = _structure_ids(candidate_frames)
        member_id = _member_ids("teacher", self.config.teacher)[0]
        descriptor_schema = _teacher_schema(reference_schema)
        intent = TeacherScoreIntent(
            model=ModelHandle(
                member_id,
                f"iter{self.config.iter_index}/teach/train/{member_id}",
            ),
            reference=TeacherDatasetHandle(
                f"iter{self.config.iter_index}/teach/train/dataset",
                reference_ids,
                reference_ids,
                "self",
            ),
            candidates=StructureBatchHandle(
                f"iter{self.config.iter_index}/screen/candidates", candidate_ids
            ),
            chunk_size=self.gmm_score_chunk_size,
            emit_frame_means=bool(emit_frame_means),
            reference_descriptor_schema=descriptor_schema,
        )
        physical = self.teacher_score_request_factory(
            source=source,
            model=models[0],
            reference_path=reference_path,
            router_dir=router_dir,
            candidates=candidate_frames,
            work_dir=work_dir / "score_operation",
            emit_frame_means=emit_frame_means,
            chunk_size=self.gmm_score_chunk_size,
            reference_descriptor_schema=reference_schema,
            progress_reporter_factory=lambda operation_id, phase: report.operation(
                operation_id, phase=phase
            ),
        )
        outcome = TeacherReadoutUseCase(
            _BoundTeacherScreenReadout(self.teacher, intent, physical)
        ).score(intent)
        scored_frames = []
        for frame, probability in zip(
            candidate_frames, outcome.frame_probabilities, strict=True
        ):
            item = copy_atoms_preserving_singlepoint_results(frame)
            item.info["teacher_p"] = float(probability)
            scored_frames.append(item)
        return TeacherGMMScore(
            frames=tuple(scored_frames),
            frame_means=(
                None
                if outcome.frame_means is None
                else np.asarray(outcome.frame_means, dtype=np.float32)
            ),
            descriptor_schema=_teacher_schema_mapping(outcome.descriptor_schema),
        )

    def score_student(
        self,
        *,
        frames,
        source,
        router_dir,
        work_dir,
        emit_features,
        report,
        round_index,
    ):
        stage_root = (
            self.config.root
            / f"iter{self.config.iter_index}"
            / f"r{round_index}"
            / "train"
        )
        model_handles = _student_model_handles(self.config, round_index)
        if len(model_handles) != 1:
            raise ValueError(
                "lifecycle student GMM requires exactly one canonical model"
            )
        resolved = self.student.resolve_model_expectations(
            model_handles, stage_root=stage_root
        )
        if len(resolved) != 1:
            raise ValueError("student GMM model resolution changed canonical membership")
        reference_path = stage_root / "dataset" / "train.xyz"
        reference_ids = _structure_ids(read_extxyz_structures(reference_path))
        candidate_frames = tuple(frames)
        candidate_ids = _structure_ids(candidate_frames)
        intent = StudentReadoutIntent(
            model=model_handles[0],
            reference=DatasetHandle(
                f"iter{self.config.iter_index}/r{round_index}/train/dataset",
                reference_ids,
            ),
            candidates=StructureBatchHandle(
                f"iter{self.config.iter_index}/r{round_index}/screen/candidates",
                candidate_ids,
            ),
            chunk_size=self.gmm_score_chunk_size,
            emit_frame_means=bool(emit_features),
            type_map=tuple(self.config.student["type_map"]),
        )
        physical = self.student_readout_request_factory(
            model=resolved[0].path,
            reference_path=reference_path,
            source=source,
            candidates=candidate_frames,
            router_dir=router_dir,
            work_dir=work_dir,
            type_map=tuple(self.config.student["type_map"]),
            emit_frame_means=emit_features,
            chunk_size=self.gmm_score_chunk_size,
            progress_reporter_factory=lambda operation_id, phase: report.operation(
                operation_id, phase=phase
            ),
        )
        outcome = self.student.execute_readout(
            intent,
            physical_request=physical,
            resolved_model=resolved[0],
        )
        values = MappingProxyType(
            dict(zip(outcome.ordered_structure_ids, outcome.frame_probabilities, strict=True))
        )
        means = None
        if outcome.frame_means is not None:
            means = MappingProxyType(
                {
                    identity: np.asarray(row, dtype=np.float32)
                    for identity, row in zip(
                        outcome.ordered_structure_ids,
                        outcome.frame_means,
                        strict=True,
                    )
                }
            )
        return _StudentScreenScore(
            values=values,
            frame_means=means,
            descriptor_schema=_student_schema_mapping(outcome.descriptor_schema),
            element_frame_means=outcome.element_frame_means,
            model_identity=model_handles[0].model_id,
            dataset_identity=intent.reference.dataset_id,
        )

    def label_teacher(
        self, *, frames, work_dir, report, trusted_below, round_index
    ) -> list[Any]:
        del round_index
        return self.teacher.label(
            self.teacher_label_request_factory(
                models=_teacher_models(self.config, self.teacher),
                frames=tuple(frames),
                work_dir=work_dir,
                trusted_below=trusted_below,
                progress_reporter_factory=lambda operation_id, phase: report.operation(
                    operation_id, phase=phase
                ),
            )
        )

    def read_student_vectors(
        self,
        *,
        frames,
        source,
        work_dir,
        report,
        round_index,
        materialized: bool = False,
        reused_score=None,
    ):
        from workflow_core.lifecycle.screen_executor import JointVectorReadout

        stage_root = (
            self.config.root
            / f"iter{self.config.iter_index}"
            / f"r{round_index}"
            / "train"
        )
        model_handles = _student_model_handles(self.config, round_index)
        if len(model_handles) != 1:
            raise ValueError(
                "joint DIRECT student readout requires exactly one canonical model"
            )
        resolved = self.student.resolve_model_expectations(
            model_handles, stage_root=stage_root
        )
        if len(resolved) != 1:
            raise ValueError(
                "joint DIRECT student model resolution changed canonical membership"
            )
        reference_path = stage_root / "dataset" / "train.xyz"
        background_frames = tuple(
            read_extxyz_structures(reference_path, allow_empty=False)
        )
        candidate_metadata = {
            str(frame.info["structure_id"]): dict(frame.info) for frame in frames
        }
        candidate_ids = tuple(candidate_metadata)
        candidate_frames = []
        if reused_score is not None:
            pass
        elif materialized:
            candidate_frames = [frame.copy() for frame in frames]
        else:
            for chunk in source.replay(
                candidate_ids, chunk_size=self.gmm_score_chunk_size
            ):
                for frame in chunk:
                    identity = str(frame.info.get("structure_id", ""))
                    if identity not in candidate_metadata:
                        raise ValueError(
                            "joint DIRECT student replay returned an unexpected identity"
                        )
                    frame.info.update(candidate_metadata[identity])
                    candidate_frames.append(frame)
        if tuple(
            str(frame.info["structure_id"]) for frame in candidate_frames
        ) != candidate_ids and reused_score is None:
            raise ValueError(
                "joint DIRECT student replay changed candidate identity order"
            )
        outcome = self.student._read_structure_vectors(
            model=resolved[0].path,
            candidate_frames=tuple(candidate_frames) if reused_score is None else (),
            background_frames=background_frames,
            work_dir=work_dir,
            type_map=tuple(self.config.student["type_map"]),
            chunk_size=self.gmm_score_chunk_size,
            report=report,
        )
        if reused_score is not None:
            if (
                reused_score.model_identity != model_handles[0].model_id
                or reused_score.dataset_identity != f"iter{self.config.iter_index}/r{round_index}/train/dataset"
                or dict(reused_score.descriptor_schema) != dict(outcome.descriptor_schema)
                or reused_score.frame_means is None
                or tuple(reused_score.frame_means) != candidate_ids
                or reused_score.element_frame_means is None
            ):
                raise ValueError("reused student features changed model, dataset, schema or candidate order")
            expected_elements = {}
            for frame in frames:
                identity = str(frame.info["structure_id"])
                for element in dict.fromkeys(frame.get_chemical_symbols()):
                    expected_elements.setdefault(element, []).append(identity)
            width = int(outcome.descriptor_schema["width"])
            if set(reused_score.element_frame_means) != set(expected_elements):
                raise ValueError("reused student element features changed chemical presence")
            for element, expected_ids in expected_elements.items():
                rows = reused_score.element_frame_means[element]
                if tuple(rows) != tuple(expected_ids) or any(
                    np.asarray(row).shape != (width,) or not np.isfinite(row).all()
                    for row in rows.values()
                ):
                    raise ValueError("reused student element features changed chemical presence or width")
        return JointVectorReadout(
            candidate_vectors=MappingProxyType(dict(outcome.candidate_vectors if reused_score is None else reused_score.frame_means)),
            background_vectors=MappingProxyType(dict(outcome.background_vectors)),
            descriptor_schema=MappingProxyType(dict(outcome.descriptor_schema)),
            model_identity=model_handles[0].model_id,
            dataset_identity=(
                f"iter{self.config.iter_index}/r{round_index}/train/dataset"
            ),
            candidate_element_vectors=outcome.candidate_element_vectors if reused_score is None else reused_score.element_frame_means,
            background_element_vectors=outcome.background_element_vectors,
            candidate_frames=tuple(candidate_frames) if reused_score is None else None,
            background_frames=background_frames,
        )

    @property
    def teacher_gmm_profile(self):
        if not self.config.route.teacher_requires_reference_tail:
            return None
        return self.teacher.screen_operation_profiles.gmm_score

    @property
    def teacher_label_profile(self):
        return self.teacher.screen_operation_profiles.label

    @property
    def student_profile_values(self) -> tuple[Any, Any]:
        if self.config.route.kind not in {
            RouteKind.SINGLE_TEACHER_SINGLE_STUDENT,
            RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
        }:
            return (None, None)
        profiles = self.student.screen_operation_profiles
        return (profiles.gmm_reference, profiles.gmm_score)


def _teacher_models(config: Any, teacher: Any) -> tuple[Path, ...]:
    handles = teacher.resolve_published_models(
        stage_root=config.root / f"iter{config.iter_index}" / "teach" / "train",
        member_ids=_member_ids("teacher", config.teacher),
    )
    return tuple(handle.path for handle in handles)


@dataclass(frozen=True, slots=True)
class _StudentScreenScore:
    values: Any
    frame_means: Any
    descriptor_schema: Any
    element_frame_means: Any = None
    model_identity: str | None = None
    dataset_identity: str | None = None

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _BoundTeacherScreenReadout:
    """Bind the teacher's physical request behind the logic-owned readout port."""

    teacher: Any
    intent: TeacherScoreIntent
    request: Any

    def fit_reference(self, request):
        del request
        raise TypeError("screen scoring does not fit the teacher reference")

    def score_candidates(self, intent: TeacherScoreIntent) -> TeacherScoreOutcome:
        if intent != self.intent:
            raise ValueError("teacher score intent does not match its physical binding")
        physical_ids = _structure_ids(self.request.candidates)
        physical_schema = _teacher_schema(self.request.reference_descriptor_schema)
        if (
            physical_ids != intent.candidates.ordered_structure_ids
            or self.request.chunk_size != intent.chunk_size
            or self.request.emit_frame_means != intent.emit_frame_means
            or physical_schema != intent.reference_descriptor_schema
        ):
            raise ValueError("teacher score intent diverges from bound native inputs")
        score = self.teacher.score_candidates(self.request)
        try:
            actual = _structure_ids(score.frames)
            probabilities = tuple(
                float(frame.info["teacher_p"]) for frame in score.frames
            )
            compact = getattr(score, "compact_view", None)
            raw_statistics = getattr(compact, "frame_statistics", None)
            statistics = (
                probabilities
                if raw_statistics is None
                else tuple(float(value) for value in raw_statistics)
            )
            means = None
            if score.frame_means is not None:
                means = tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(score.frame_means)
                )
            return TeacherScoreOutcome(
                ordered_structure_ids=actual,
                frame_probabilities=probabilities,
                frame_statistics=statistics,
                descriptor_schema=_teacher_schema(score.descriptor_schema),
                frame_means=means,
            )
        finally:
            score.close()


def _student_model_handles(config: Any, round_index: int) -> tuple[ModelHandle, ...]:
    return tuple(
        ModelHandle(
            member_id,
            f"iter{config.iter_index}/r{round_index}/train/{member_id}",
        )
        for member_id in _member_ids("student", config.student)
    )


def _structure_ids(frames) -> tuple[str, ...]:
    identities = tuple(str(frame.info.get("structure_id", "")) for frame in frames)
    if any(not identity for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("screen readout requires ordered unique structure identities")
    return identities


def _student_schema_mapping(schema: Any) -> MappingProxyType:
    if schema is None:
        raise ValueError("successful student screen readout omitted descriptor schema")
    return MappingProxyType(
        {
            "name": schema.name,
            "revision": schema.revision,
            "backend": schema.backend,
            "width": schema.width,
            **dict(schema.attributes),
        }
    )


def _teacher_schema(value: Any) -> TeacherDescriptorSchema:
    schema = validated_readout_input_schema(value)
    known = {"name", "revision", "backend", "width"}
    return TeacherDescriptorSchema(
        name=str(schema["name"]),
        revision=int(schema["revision"]),
        backend=str(schema["backend"]),
        width=int(schema["width"]),
        attributes={key: item for key, item in schema.items() if key not in known},
    )


def _teacher_schema_mapping(schema: TeacherDescriptorSchema) -> MappingProxyType:
    return MappingProxyType(
        {
            "name": schema.name,
            "revision": schema.revision,
            "backend": schema.backend,
            "width": schema.width,
            **dict(schema.attributes),
        }
    )


def _member_ids(role: str, spec: Any) -> tuple[str, ...]:
    members = int(spec.get("members", 1))
    return tuple(
        role if members == 1 else f"{role}_{index:02d}"
        for index in range(1, members + 1)
    )


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("lifecycle screen GMM score chunk size must be positive")
    return value


__all__ = ["LifecycleScreenIOAdapter"]
