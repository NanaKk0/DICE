from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable, Protocol
import shutil

import numpy as np

from workflow_backends.common.committee_pool import execute_committee_members
from workflow_backends.sampling_models import (
    OrderedSamplingModels,
    SamplingModelsRequest,
    SamplingModelsResult,
    validate_sampling_models_result,
)
from workflow_backends.student.role import (
    BoundStudentModelCapability,
    BoundStudentReadoutCapability,
    bind_student_training_artifact_validator,
    BoundStudentTrainingCapability,
    StudentBackendBinding,
    StudentHistoricalModelRequest,
    StudentHistoricalModelResult,
    StudentTrainPublicationRequest,
    StudentTrainRequest,
    StudentTrainResult,
    resolve_student_model_expectations,
)
from workflow_backends.student.readout import (
    PhaseStudentReferenceRequest,
    PhaseStudentScoreRequest,
    StudentNativeReadoutAdapter,
    StudentStructureVectorResult,
)
from workflow_core.lifecycle.model_handles import ModelArtifact
from workflow_core.structures.formats import write_extxyz_structures
from workflow_policies.gmm_reference_tail import GMMReferenceFitResult, GMMScoreResult
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter


class NepExecutor(Protocol):
    def train(
        self,
        recipe: Path,
        member: Path,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> tuple[Path, Path]: ...

    def gmm_reference(self, **kwargs) -> GMMReferenceFitResult: ...

    def gmm_score(self, **kwargs) -> GMMScoreResult: ...

    def structure_vectors(self, **kwargs) -> tuple[dict[str, object], tuple[str, ...], np.ndarray]: ...


@dataclass(slots=True)
class NepStudentAdapter:
    executor: NepExecutor

    def structure_vectors(
        self,
        *,
        model: Path,
        candidate_frames,
        background_frames,
        type_map: tuple[str, ...],
        work_dir: Path,
    ) -> StudentStructureVectorResult:
        del type_map
        candidates = tuple(candidate_frames)
        background = tuple(background_frames)
        if not background:
            raise ValueError("joint DIRECT student training background is empty")
        candidate_ids = tuple(str(frame.info.get("structure_id", "")) for frame in candidates)
        background_ids = tuple(str(frame.info.get("structure_id", "")) for frame in background)
        if any(not identity for identity in (*candidate_ids, *background_ids)):
            raise ValueError("joint DIRECT student structures require canonical identities")
        combined = []
        tokens = []
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
            work / "joint_direct_structures.xyz", combined, allow_empty=False
        )
        vector_kwargs = {
            "model": Path(model),
            "input_path": input_path,
            "work_dir": work / "operation",
        }
        parameters = inspect.signature(self.executor.structure_vectors).parameters
        if "include_element_vectors" in parameters:
            vector_kwargs["include_element_vectors"] = True
        raw_vectors = self.executor.structure_vectors(**vector_kwargs)
        if len(raw_vectors) == 3:
            schema, actual, rows = raw_vectors
            element_rows = None
        else:
            schema, actual, rows, element_rows = raw_vectors
        if tuple(actual) != tuple(tokens):
            raise ValueError("student structural readout changed partition order")
        values = np.asarray(rows, dtype=np.float32)
        if values.ndim != 2 or len(values) != len(tokens) or not np.isfinite(values).all():
            raise ValueError("student structural readout returned invalid vectors")
        split = len(candidates)
        return StudentStructureVectorResult(
            candidate_vectors=dict(zip(candidate_ids, values[:split], strict=True)),
            background_vectors=dict(zip(background_ids, values[split:], strict=True)),
            descriptor_schema=schema,
            candidate_element_vectors=_partition_element_vectors(
                element_rows, candidate_ids, token_ids=tokens[:split]
            ),
            background_element_vectors=_partition_element_vectors(
                element_rows, background_ids, token_ids=tokens[split:]
            ),
        )
    def train(self, *, recipe: str | Path, dataset_path: str | Path, train_dir: str | Path, model: str, members: int, policy: str, foundation_model: str | Path | None, foundation_restart: str | Path | None, progress_reporter_factory: Callable[[str], OperationProgressReporter] | None = None) -> tuple[ModelArtifact, ...]:
        if model not in {"single", "committee"} or (model == "single" and members != 1) or (model == "committee" and members < 2):
            raise ValueError("NEP lifecycle student model shape is invalid")
        if policy == "foundation_finetune" and (foundation_model is None or foundation_restart is None):
            raise ValueError("NEP lifecycle foundation_finetune requires the foundation model and restart pair")
        source=Path(recipe); root=Path(train_dir)

        def execute_member(index: int) -> ModelArtifact:
            member=root/("student" if model=="single" else f"student_{index:02d}"); member.mkdir(parents=True,exist_ok=True)
            rendered=member/"nep.in"; text=source.read_text(encoding="utf-8")
            if foundation_model is not None:
                shutil.copy2(foundation_model,member/"foundation_nep.txt"); shutil.copy2(foundation_restart,member/"foundation_nep.restart")
                text += "\nfine_tune foundation_nep.txt foundation_nep.restart\n"
            rendered.write_text(text,encoding="utf-8"); shutil.copy2(dataset_path,member/"train.xyz"); shutil.copy2(dataset_path,member/"test.xyz")
            reporter = (
                progress_reporter_factory(member.name)
                if progress_reporter_factory is not None
                else None
            )
            native, restart = self._execute_train(
                rendered,
                member,
                progress_reporter=reporter,
                member_index=index,
                total_members=members,
            )
            return ModelArtifact(
                member_id=member.name,
                terminal_model_path=native,
                evidence={"restart_path": Path(restart).resolve()},
            )
        return execute_committee_members(
            tuple(range(1, members + 1)),
            context=getattr(self.executor, "train_context", None),
            task_id=lambda index: (
                "student" if model == "single" else f"student_{index:02d}"
            ),
            execute=execute_member,
        )

    def _execute_train(
        self,
        rendered: Path,
        member: Path,
        *,
        progress_reporter: OperationProgressReporter | None,
        member_index: int,
        total_members: int,
    ) -> tuple[Path, Path]:
        parameters = inspect.signature(self.executor.train).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, object] = {}
        if accepts_kwargs or "progress_reporter" in parameters:
            kwargs["progress_reporter"] = progress_reporter
        if accepts_kwargs or "member_index" in parameters:
            kwargs["member_index"] = member_index
        if accepts_kwargs or "total_members" in parameters:
            kwargs["total_members"] = total_members
        return self.executor.train(rendered, member, **kwargs)

    def fit_gmm_reference(
        self,
        *,
        model: str | Path,
        reference_path: str | Path,
        router_dir: str | Path,
        work_dir: str | Path,
        type_map: tuple[str, ...],
        progress_reporter: OperationProgressReporter | None = None,
    ) -> GMMReferenceFitResult:
        return _invoke_with_optional_progress(self.executor.gmm_reference,
            model=_canonical_nep_model_path(model),
            reference_path=Path(reference_path).resolve(),
            router_dir=Path(router_dir).absolute(),
            work_dir=Path(work_dir).absolute(),
            type_map=tuple(str(item) for item in type_map),
            progress_reporter=progress_reporter,
        )

    def fit_reference(
        self, request: PhaseStudentReferenceRequest
    ) -> GMMReferenceFitResult:
        return self.fit_gmm_reference(
            model=request.model,
            reference_path=request.reference_path,
            router_dir=request.router_dir,
            work_dir=request.work_dir,
            type_map=request.type_map,
            progress_reporter=request.progress_reporter,
        )

    def score_candidates(self, request: PhaseStudentScoreRequest) -> GMMScoreResult:
        if callable(getattr(self.executor, "gmm_score_stream", None)):
            return self.executor.gmm_score_stream(
                model=_canonical_nep_model_path(request.model), reference_path=request.reference_path,
                frame_chunks=request.frame_chunks, ordered_identities=request.ordered_identities,
                router_dir=request.router_dir, work_dir=request.work_dir,
                emit_frame_means=request.emit_frame_means, progress_reporter=request.progress_reporter,
                progress_systems=request.progress_systems,
            )
        results: list[GMMScoreResult] = []
        consumed = 0
        for index, chunk_items in enumerate(request.frame_chunks):
            chunk = tuple(chunk_items)
            if not chunk:
                raise ValueError("NEP phase readout received an empty candidate chunk")
            identities = tuple(
                str(frame.info.get("structure_id", "")) for frame in chunk
            )
            expected = request.ordered_identities[consumed : consumed + len(chunk)]
            if identities != expected:
                raise ValueError(
                    "NEP phase readout chunks do not match canonical candidate order"
                )
            chunk_root = request.work_dir / f"score_chunk_{index:06d}"
            chunk_root.mkdir(parents=True, exist_ok=True)
            try:
                path = write_extxyz_structures(
                    chunk_root / "candidates.xyz", chunk, allow_empty=False
                )
                results.append(
                    self.score_gmm_router(
                        model=request.model,
                        reference_path=request.reference_path,
                        observation_path=path,
                        ordered_identities=identities,
                        router_dir=request.router_dir,
                        work_dir=chunk_root / "operation",
                        emit_frame_means=request.emit_frame_means,
                        progress_reporter=request.progress_reporter,
                        progress_systems=(
                            None
                            if request.progress_systems is None
                            else dict(request.progress_systems)
                        ),
                    )
                )
            finally:
                shutil.rmtree(chunk_root, ignore_errors=True)
            consumed += len(chunk)
        if consumed != len(request.ordered_identities) or not results:
            raise ValueError(
                "NEP phase readout did not receive the complete candidate stream"
            )
        from workflow_backends.common.inference_pool import merge_gmm_shards
        return merge_gmm_shards(results, request.ordered_identities)

    def score_gmm_router(
        self,
        *,
        model: str | Path,
        reference_path: str | Path,
        observation_path: str | Path,
        ordered_identities: tuple[str, ...],
        router_dir: str | Path,
        work_dir: str | Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        return _invoke_with_optional_progress(self.executor.gmm_score,
            model=_canonical_nep_model_path(model),
            reference_path=Path(reference_path).resolve(),
            observation_path=Path(observation_path).resolve(),
            ordered_identities=tuple(str(item) for item in ordered_identities),
            router_dir=Path(router_dir).absolute(),
            work_dir=Path(work_dir).absolute(),
            emit_frame_means=bool(emit_frame_means),
            progress_reporter=progress_reporter,
            progress_systems=progress_systems,
        )


@dataclass(frozen=True, slots=True)
class NepStudentBackendFactory:
    executor: NepExecutor

    def bind(self, binding: StudentBackendBinding) -> NepStudentBackend:
        from workflow_backends.student.role import (
            StudentScreenOperationProfiles,
            screen_profile_id,
        )

        return NepStudentBackend(
            adapter=NepStudentAdapter(self.executor),
            binding=binding,
            sampling_models=NepSamplingModelProvider(),
            screen_operation_profiles=StudentScreenOperationProfiles(
                gmm_reference=screen_profile_id(
                    getattr(self.executor, "gmm_reference_context", None)
                ),
                gmm_score=screen_profile_id(
                    getattr(self.executor, "gmm_score_context", None)
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class NepSamplingModelProvider:
    """Backend-owned identity provider for ordered native NEP models."""

    def prepare_sampling_models(
        self, request: SamplingModelsRequest
    ) -> SamplingModelsResult:
        result = SamplingModelsResult(
            OrderedSamplingModels(
                tuple(request.models.paths),
                member_ids=tuple(request.models.member_ids),
            )
        )
        validate_sampling_models_result(request, result)
        return result

    def load_sampling_models(
        self, request: SamplingModelsRequest
    ) -> SamplingModelsResult:
        return self.prepare_sampling_models(request)


@dataclass(slots=True)
class NepStudentBackend:
    """Bound NEP train and direct sampling-model capabilities."""

    adapter: NepStudentAdapter
    binding: StudentBackendBinding
    sampling_models: NepSamplingModelProvider
    screen_operation_profiles: object

    @property
    def training(self) -> NepStudentBackend:
        return self

    @property
    def models(self) -> NepStudentBackend:
        return self

    def bind_training_capability(
        self,
        intent: object,
        request: StudentTrainRequest,
        models: tuple[object, ...],
        published_root: Path | None = None,
    ) -> BoundStudentTrainingCapability:
        return BoundStudentTrainingCapability(
            self, intent, request, tuple(models), published_root
        )

    def structure_vectors(self, **kwargs) -> StudentStructureVectorResult:
        kwargs.pop("source", None)
        kwargs.pop("chunk_size", None)
        kwargs.pop("report", None)
        return self.adapter.structure_vectors(**kwargs)

    def bind_training_expectation(
        self,
        expectation: object,
        request: StudentTrainRequest,
        *,
        published_root: Path,
    ) -> BoundStudentTrainingCapability:
        return BoundStudentTrainingCapability(
            self,
            getattr(expectation, "intent"),
            request,
            tuple(getattr(expectation, "models")),
            published_root,
            expectation,
        )

    def bind_training_artifact_validator(
        self, expectation: object
    ) -> object:
        return bind_student_training_artifact_validator(self, expectation)

    def resolve_model_expectations(
        self, models: tuple[object, ...], *, stage_root: Path
    ) -> tuple[object, ...]:
        return resolve_student_model_expectations(
            self, tuple(models), stage_root=stage_root
        )

    def bind_readout_capability(
        self, intent: object, request: object, resolved_model: object | None = None
    ) -> BoundStudentReadoutCapability:
        return BoundStudentReadoutCapability(self, intent, request, resolved_model)

    def bind_model_capability(
        self,
        models: tuple[object, ...],
        requests: tuple[StudentHistoricalModelRequest, ...],
    ) -> BoundStudentModelCapability:
        return BoundStudentModelCapability(self, tuple(models), tuple(requests))

    @property
    def readout_ops(self):
        return self.adapter

    @property
    def readout(self) -> StudentNativeReadoutAdapter:
        return StudentNativeReadoutAdapter(self.adapter)

    @property
    def gmm_reference_context(self) -> object | None:
        return getattr(self.adapter.executor, "gmm_reference_context", None)

    @property
    def gmm_score_context(self) -> object | None:
        return getattr(self.adapter.executor, "gmm_score_context", None)

    def train(self, request: StudentTrainRequest) -> StudentTrainResult:
        _validate_student_member_ids(request.member_ids, self.binding.model_shape)
        artifacts = self.adapter.train(
            recipe=self.binding.recipe_path,
            dataset_path=request.dataset_path,
            train_dir=request.train_dir,
            model=self.binding.model_shape,
            members=len(request.member_ids),
            policy=self.binding.policy,
            foundation_model=self.binding.foundation_model,
            foundation_restart=self.binding.foundation_restart,
            progress_reporter_factory=request.progress_reporter_factory,
        )
        result = StudentTrainResult(artifacts)
        if tuple(item.member_id for item in result.artifacts) != request.member_ids:
            raise ValueError("NEP student train changed ordered member identities")
        return result

    def finalize_published_training(
        self, request: StudentTrainPublicationRequest
    ) -> None:
        stage = request.stage_root
        for artifact in request.train.artifacts:
            restart = Path(artifact.evidence.get("restart_path", ""))
            member_root = stage / artifact.member_id
            if (
                restart.is_symlink()
                or not restart.is_file()
                or restart.stat().st_size <= 0
            ):
                raise ValueError(
                    f"NEP student train lacks restart evidence for {artifact.member_id}"
                )
            try:
                restart.resolve().relative_to(member_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"NEP student restart evidence escapes {artifact.member_id}"
                ) from exc

    def recover_historical_model(
        self, request: StudentHistoricalModelRequest
    ) -> StudentHistoricalModelResult:
        from workflow_backends.student.nep.historical_model_recovery import (
            resolve_legacy_nep_student_model,
        )

        return StudentHistoricalModelResult(
            member_id=request.member_id,
            path=resolve_legacy_nep_student_model(
                request.member_id, request.legacy_handle
            ),
        )


def _partition_element_vectors(
    raw,
    identities: tuple[str, ...],
    *,
    token_ids,
):
    if raw is None:
        return None
    token_to_identity = dict(zip(token_ids, identities, strict=True))
    result = {}
    for element, (raw_ids, rows) in raw.items():
        values = {
            token_to_identity[token]: np.asarray(row, dtype=np.float32)
            for token, row in zip(raw_ids, rows, strict=True)
            if token in token_to_identity
        }
        if values:
            result[str(element)] = values
    return result


def _validate_student_member_ids(
    member_ids: tuple[str, ...], model_shape: str
) -> None:
    expected = (
        ("student",)
        if model_shape == "single"
        else tuple(f"student_{index:02d}" for index in range(1, len(member_ids) + 1))
    )
    if member_ids != expected or (model_shape == "committee" and len(member_ids) < 2):
        raise ValueError("NEP student train requires canonical ordered member identities")


def _invoke_with_optional_progress(operation, **kwargs):
    parameters = inspect.signature(operation).parameters
    for optional_name in ("progress_reporter", "progress_systems"):
        if optional_name not in parameters:
            kwargs.pop(optional_name, None)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return operation(**kwargs)


def _canonical_nep_model_paths(models: tuple[str | Path, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for model in models:
        path = Path(model)
        if path.is_symlink() or path.suffix != ".txt" or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"NEP lifecycle consumer requires a non-empty regular native .txt model: {path}")
        paths.append(path.resolve())
    return tuple(paths)


def _canonical_nep_model_path(model: str | Path) -> Path:
    return _canonical_nep_model_paths((model,))[0]
