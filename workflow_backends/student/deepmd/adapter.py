from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import inspect
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from workflow_backends.common.committee_pool import execute_committee_members
from workflow_backends.deepmd_common.training_pool import DEEPMD_TRAINING_POOL
from workflow_backends.common.model_files import link_model_file

from workflow_core.lifecycle.model_handles import (
    ModelArtifact,
    PublishedModelHandle,
)
from workflow_core.structures.formats import write_extxyz_structures
from workflow_artifacts.lifecycle.sampling_models import (
    FIXED_COMPRESSION,
    EffectiveSamplingModel,
    EffectiveSamplingModelPlan,
    file_sha256,
    operation_profile_evidence,
    publish_sampling_model_assets,
    load_sampling_model_assets,
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
    FusedStudentReadoutOps,
    FusedStudentReadoutResult,
    FusedStudentReferenceRequest,
    FusedStudentScreenRequest,
    PhaseStudentReferenceRequest,
    PhaseStudentReadoutOps,
    PhaseStudentScoreRequest,
    StudentNativeReadoutAdapter,
    StudentStructureVectorResult,
)
from workflow_backends.sampling_models import (
    OrderedSamplingModels,
    SamplingModelsRequest,
    SamplingModelsResult,
    validate_sampling_models_result,
)
from workflow_backends.deepmd_common.dataset import materialize_native_dataset
from workflow_backends.deepmd_common.model_recipe import synchronize_model_random_seeds
from workflow_core.model_families import deepmd_recipe_model_family
from workflow_backends.deepmd_common.type_maps import (
    resolve_training_type_maps,
    validated_type_map,
)
from workflow_policies.gmm_reference_tail import GMMReferenceFitResult, GMMScoreResult
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter


class DeepMDStudentExecutor(Protocol):
    def train(
        self,
        input_json: Path,
        member_dir: Path,
        cli: str,
        *,
        recipe_kind: str = "standard",
        foundation_model: Path | None = None,
        train_options: Mapping[str, bool] | None = None,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path: ...

    def gmm_reference(
        self,
        *,
        model: Path,
        reference_path: Path,
        router_dir: Path,
        work_dir: Path,
        type_map: tuple[str, ...],
        progress_reporter: OperationProgressReporter | None = None,
    ) -> GMMReferenceFitResult: ...

    def structure_vectors(
        self,
        *,
        model: Path,
        input_path: Path,
        type_map: tuple[str, ...],
        work_dir: Path,
    ) -> tuple[dict[str, object], tuple[str, ...], np.ndarray]: ...

    def compress(
        self,
        source_model: Path,
        output_model: Path,
        cli: str,
        *,
        recipe_kind: str,
        training_script: Path | None,
        step: float,
        extrapolate: int,
        frequency: int,
    ) -> Path: ...

    def gmm_score(
        self,
        *,
        model: Path,
        reference_path: Path,
        observation_path: Path,
        ordered_identities: tuple[str, ...],
        router_dir: Path,
        work_dir: Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult: ...

    def gmm_score_chunks(
        self,
        *,
        model: Path,
        reference_path: Path,
        observation_paths: tuple[Path, ...],
        ordered_identities_by_chunk: tuple[tuple[str, ...], ...],
        router_dir: Path,
        work_dir: Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult: ...

    def gmm_score_rolling(
        self,
        *,
        model: Path,
        reference_path: Path,
        frame_chunks: Iterable[Sequence[object]],
        ordered_identities: tuple[str, ...],
        router_dir: Path,
        work_dir: Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult: ...

    def gmm_score_stream(
        self,
        *,
        model: Path,
        reference_path: Path,
        frame_chunks: Iterable[Sequence[object]],
        ordered_identities: tuple[str, ...],
        router_dir: Path,
        work_dir: Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult: ...

    def gmm_screen(
        self,
        *,
        model: Path,
        reference_path: Path,
        frame_chunks: Iterable[Sequence[object]],
        ordered_identities: tuple[str, ...],
        router_dir: Path,
        work_dir: Path,
        type_map: tuple[str, ...],
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ): ...

    def foundation_type_map(self, model: Path, *, work_dir: Path) -> tuple[str, ...]: ...


@dataclass(slots=True)
class DeepMDStudentAdapter:
    run_json_path: Path
    executor: DeepMDStudentExecutor

    def structure_vectors(
        self,
        *,
        model: Path,
        candidate_frames: Sequence[object],
        background_frames: Sequence[object],
        type_map: tuple[str, ...],
        work_dir: Path,
    ) -> StudentStructureVectorResult:
        return _partitioned_structure_vectors(
            self.executor,
            model=model,
            candidate_frames=candidate_frames,
            background_frames=background_frames,
            type_map=type_map,
            work_dir=work_dir,
        )

    def prepare_sampling_models(
        self,
        *,
        models: Sequence[PublishedModelHandle],
        sample_dir: str | Path,
        cli: str,
        compress: bool,
    ) -> EffectiveSamplingModelPlan:
        if cli not in {"tf", "pt"}:
            raise ValueError("DeepMD sampling model preparation requires tf or pt CLI")
        ordered = tuple(models)
        if not ordered:
            raise ValueError("DeepMD sampling model preparation requires models")
        if not compress:
            effective: list[EffectiveSamplingModel] = []
            for model in ordered:
                input_json = model.path.parent / "input.json"
                recipe_kind = _rendered_recipe_kind(
                    input_json, member_id=model.member_id
                )
                if recipe_kind == "dpa4c" and cli != "pt":
                    raise ValueError("DPA4C sampling requires the PT ABI")
                effective.append(
                    EffectiveSamplingModel(
                        member_id=model.member_id,
                        source_path=model.path,
                        source_sha256=file_sha256(model.path),
                        effective_path=model.path,
                        effective_sha256=file_sha256(model.path),
                        deepmd_recipe_kind=recipe_kind,
                        cli=cli,
                    )
                )
            return EffectiveSamplingModelPlan(
                tuple(effective), compressed=False
            )

        context = getattr(self.executor, "compress_context", None)
        if context is None:
            raise ValueError(
                "DeepMD sampling compression requires its selected operation context"
            )
        expected_role = f"student.deepmd.{cli}.compress"
        if getattr(context, "role", None) != expected_role:
            raise ValueError(
                f"DeepMD sampling compression requires operation role {expected_role}"
            )
        root = Path(sample_dir).resolve()
        compressed_models: list[EffectiveSamplingModel] = []
        for model in ordered:
            source = model.path
            input_json = source.parent / "input.json"
            recipe_kind = _rendered_recipe_kind(
                input_json, member_id=model.member_id
            )
            if recipe_kind == "dpa4c" and cli != "pt":
                raise ValueError("DPA4C sampling compression requires the PT ABI")
            output = (
                root
                / "sampling_models"
                / model.member_id
                / f"model_compressed{source.suffix}"
            )
            if output.exists() or output.is_symlink():
                raise ValueError(
                    f"DeepMD sampling compression output already exists for {model.member_id}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            returned = Path(
                self.executor.compress(
                    source,
                    output,
                    cli,
                    recipe_kind=recipe_kind,
                    training_script=(
                        input_json if recipe_kind == "dpa4c" else None
                    ),
                    step=float(FIXED_COMPRESSION["step"]),
                    extrapolate=int(FIXED_COMPRESSION["extrapolate"]),
                    frequency=int(FIXED_COMPRESSION["frequency"]),
                )
            ).resolve()
            if returned != output:
                raise ValueError(
                    f"DeepMD sampling compression returned an unexpected path for {model.member_id}"
                )
            compressed_models.append(
                EffectiveSamplingModel(
                    member_id=model.member_id,
                    source_path=source,
                    source_sha256=file_sha256(source),
                    effective_path=returned,
                    effective_sha256=file_sha256(returned),
                    deepmd_recipe_kind=recipe_kind,
                    cli=cli,
                    training_script=(
                        input_json if recipe_kind == "dpa4c" else None
                    ),
                )
            )
        manifest = publish_sampling_model_assets(
            sample_dir=root,
            models=compressed_models,
            cli=cli,
            operation_profile=operation_profile_evidence(context),
        )
        return EffectiveSamplingModelPlan(
            tuple(compressed_models),
            compressed=True,
            manifest_path=manifest,
        )

    def train(self, *, recipe: str | Path, dataset_path: str | Path, train_dir: str | Path, cli: str, type_map: tuple[str, ...], model: str, members: int, policy: str, foundation_model: str | Path | None, train_options: Mapping[str, bool] | None = None, progress_reporter_factory: Callable[[str], OperationProgressReporter] | None = None) -> tuple[ModelArtifact, ...]:
        if cli not in {"tf", "pt"} or model not in {"single", "committee"} or (model == "single" and members != 1) or (model == "committee" and members < 2):
            raise ValueError("DeepMD lifecycle student shape/cli is invalid")
        if policy == "foundation_finetune" and foundation_model is None:
            raise ValueError("DeepMD lifecycle student foundation_finetune requires foundation_model")
        source = json.loads(Path(recipe).read_text(encoding="utf-8"))
        recipe_kind = _deepmd_recipe_kind(source)
        if recipe_kind == "dpa4c" and cli != "pt":
            raise ValueError(
                "DPA4C lifecycle students require a DeepMD/PT declaration"
            )
        foundation = (
            None if foundation_model is None else Path(foundation_model).resolve()
        )
        if recipe_kind == "dpa4c" and foundation is not None:
            if (
                foundation.suffix not in {".pt", ".pt2"}
                or not foundation.is_file()
                or foundation.stat().st_size <= 0
            ):
                raise ValueError(
                    "DPA4C lifecycle foundation_model must be a non-empty .pt or .pt2 file"
                )
        root = Path(train_dir)
        if recipe_kind == "dpa4c":
            active_type_map, native_type_map = resolve_training_type_maps(
                active_type_map=type_map,
                native_recipe=source,
                foundation_model=foundation,
                executor=self.executor,
                work_dir=root,
            )
        else:
            active_type_map = validated_type_map(type_map, role="active workflow")
            native_type_map = active_type_map
        native_datasets = materialize_native_dataset(
            self.executor, dataset_path, work_root=root, type_map=active_type_map
        )
        def execute_member(index: int) -> ModelArtifact:
            member_dir = root / ("student" if model == "single" else f"student_{index:02d}"); member_dir.mkdir(parents=True, exist_ok=True)
            rendered = copy.deepcopy(source); native_model = rendered.setdefault("model", {}); native_model["type_map"] = list(native_type_map)
            training = rendered.setdefault("training", {})
            training_data = dict(training.get("training_data", {}))
            training_data["systems"] = [str(path.resolve()) for path in native_datasets]
            training["training_data"] = training_data
            synchronize_model_random_seeds(rendered, self._seed(member_dir, index))
            if foundation is not None and recipe_kind != "dpa4c":
                training["foundation_model"] = str(foundation)
            input_json = member_dir / "input.json"; input_json.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
            reporter = (
                progress_reporter_factory(member_dir.name)
                if progress_reporter_factory is not None
                else None
            )
            native = Path(
                self._execute_train(
                    input_json,
                    member_dir,
                    cli,
                    recipe_kind=recipe_kind,
                    foundation_model=(
                        foundation if recipe_kind == "dpa4c" else None
                    ),
                    train_options=train_options,
                    progress_reporter=reporter,
                    member_index=index,
                    total_members=members,
                )
            ).resolve()
            expected_suffix = ".pb" if cli == "tf" else ".pt2"
            if native.suffix != expected_suffix:
                raise ValueError(
                    f"DeepMD lifecycle student {cli.upper()} terminal must use {expected_suffix}: {native}"
                )
            evidence = {
                "deepmd_cli": cli,
                "deepmd_recipe_kind": recipe_kind,
                "native_selector": "pt_expt" if recipe_kind == "dpa4c" else cli,
            }
            if train_options and train_options.get("skip_neighbor_stat") is True:
                evidence["train_options"] = {"skip_neighbor_stat": True}
            return ModelArtifact(
                member_id=member_dir.name,
                terminal_model_path=native,
                evidence=evidence,
            )
        return execute_committee_members(
            tuple(range(1, members + 1)),
            context=getattr(self.executor, "train_context", None),
            capability=DEEPMD_TRAINING_POOL,
            diagnostic_root=root,
            task_id=lambda index: (
                "student" if model == "single" else f"student_{index:02d}"
            ),
            execute=execute_member,
        )

    def _execute_train(
        self,
        input_json: Path,
        member_dir: Path,
        cli: str,
        *,
        recipe_kind: str,
        foundation_model: Path | None,
        train_options: Mapping[str, bool] | None,
        progress_reporter: OperationProgressReporter | None,
        member_index: int,
        total_members: int,
    ) -> Path:
        parameters = inspect.signature(self.executor.train).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, object] = {}
        if accepts_kwargs or "recipe_kind" in parameters:
            kwargs["recipe_kind"] = recipe_kind
        if accepts_kwargs or "foundation_model" in parameters:
            kwargs["foundation_model"] = foundation_model
        if accepts_kwargs or "train_options" in parameters:
            kwargs["train_options"] = train_options
        if accepts_kwargs or "progress_reporter" in parameters:
            kwargs["progress_reporter"] = progress_reporter
        if accepts_kwargs or "member_index" in parameters:
            kwargs["member_index"] = member_index
        if accepts_kwargs or "total_members" in parameters:
            kwargs["total_members"] = total_members
        return self.executor.train(input_json, member_dir, cli, **kwargs)

    def _seed(self, train_dir: Path, index: int) -> int:
        return int.from_bytes(hashlib.sha256(f"{self.run_json_path.resolve()}\0{train_dir}\0{index}".encode()).digest()[:4], "big")

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
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        return _invoke_with_optional_progress(self.executor.gmm_reference,
            model=_deepmd_model_file(model, presentation_dir=work),
            reference_path=Path(reference_path).resolve(),
            router_dir=Path(router_dir).absolute(),
            work_dir=work.absolute(),
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
        progress_systems = (
            None
            if request.progress_systems is None
            else dict(request.progress_systems)
        )
        if callable(getattr(self.executor, "gmm_score_stream", None)):
            return self.score_gmm_router_stream(
                model=request.model,
                reference_path=request.reference_path,
                frame_chunks=request.frame_chunks,
                ordered_identities=request.ordered_identities,
                router_dir=request.router_dir,
                work_dir=request.work_dir,
                emit_frame_means=request.emit_frame_means,
                progress_reporter=request.progress_reporter,
                progress_systems=progress_systems,
            )
        if callable(getattr(self.executor, "gmm_score_rolling", None)):
            return self.score_gmm_router_rolling(
                model=request.model,
                reference_path=request.reference_path,
                frame_chunks=request.frame_chunks,
                ordered_identities=request.ordered_identities,
                router_dir=request.router_dir,
                work_dir=request.work_dir,
                emit_frame_means=request.emit_frame_means,
                progress_reporter=request.progress_reporter,
                progress_systems=progress_systems,
            )

        chunks_root = request.work_dir / "score_chunks"
        paths: list[Path] = []
        identities_by_chunk: list[tuple[str, ...]] = []
        consumed = 0
        try:
            for index, chunk_items in enumerate(request.frame_chunks):
                chunk = tuple(chunk_items)
                if not chunk:
                    raise ValueError(
                        "DeepMD phase readout received an empty candidate chunk"
                    )
                identities = tuple(
                    str(frame.info.get("structure_id", "")) for frame in chunk
                )
                expected = request.ordered_identities[
                    consumed : consumed + len(chunk)
                ]
                if identities != expected:
                    raise ValueError(
                        "DeepMD phase readout chunks do not match canonical candidate order"
                    )
                paths.append(
                    write_extxyz_structures(
                        chunks_root / f"chunk_{index:06d}.xyz",
                        chunk,
                        allow_empty=False,
                    )
                )
                identities_by_chunk.append(identities)
                consumed += len(chunk)
            if consumed != len(request.ordered_identities) or not paths:
                raise ValueError(
                    "DeepMD phase readout did not receive the complete candidate stream"
                )
            return self.score_gmm_router_chunks(
                model=request.model,
                reference_path=request.reference_path,
                observation_paths=tuple(paths),
                ordered_identities_by_chunk=tuple(identities_by_chunk),
                router_dir=request.router_dir,
                work_dir=request.work_dir,
                emit_frame_means=request.emit_frame_means,
                progress_reporter=request.progress_reporter,
                progress_systems=progress_systems,
            )
        finally:
            shutil.rmtree(chunks_root, ignore_errors=True)

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
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        return _invoke_with_optional_progress(self.executor.gmm_score,
            model=_deepmd_model_file(model, presentation_dir=work),
            reference_path=Path(reference_path).resolve(),
            observation_path=Path(observation_path).resolve(),
            ordered_identities=tuple(str(item) for item in ordered_identities),
            router_dir=Path(router_dir).absolute(),
            work_dir=work.absolute(),
            emit_frame_means=bool(emit_frame_means),
            progress_reporter=progress_reporter,
            progress_systems=progress_systems,
        )

    def score_gmm_router_chunks(
        self,
        *,
        model: str | Path,
        reference_path: str | Path,
        observation_paths: tuple[Path, ...],
        ordered_identities_by_chunk: tuple[tuple[str, ...], ...],
        router_dir: str | Path,
        work_dir: str | Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        paths = tuple(Path(path).resolve() for path in observation_paths)
        identities_by_chunk = tuple(
            tuple(str(item) for item in identities)
            for identities in ordered_identities_by_chunk
        )
        if not paths or len(paths) != len(identities_by_chunk):
            raise ValueError("DeepMD-TF GMM chunk score requires aligned non-empty inputs")
        presented_model = _deepmd_model_file(model, presentation_dir=work)
        chunk_operation = getattr(self.executor, "gmm_score_chunks", None)
        if callable(chunk_operation):
            return _invoke_with_optional_progress(chunk_operation,
                model=presented_model,
                reference_path=Path(reference_path).resolve(),
                observation_paths=paths,
                ordered_identities_by_chunk=identities_by_chunk,
                router_dir=Path(router_dir).absolute(),
                work_dir=work.absolute(),
                emit_frame_means=bool(emit_frame_means),
                progress_reporter=progress_reporter,
                progress_systems=progress_systems,
            )
        if presented_model.suffix == ".pt2":
            raise TypeError(
                "DPA4C GMM scoring requires one model-owning chunk operation"
            )
        results = [
            _invoke_with_optional_progress(self.executor.gmm_score,
                model=presented_model,
                reference_path=Path(reference_path).resolve(),
                observation_path=path,
                ordered_identities=chunk_ids,
                router_dir=Path(router_dir).absolute(),
                work_dir=(work / f"chunk_{index:06d}").absolute(),
                emit_frame_means=bool(emit_frame_means),
                progress_reporter=progress_reporter,
                progress_systems=progress_systems,
            )
            for index, (path, chunk_ids) in enumerate(zip(paths, identities_by_chunk, strict=True))
        ]
        return _combine_gmm_scores(results, emit_frame_means=bool(emit_frame_means))

    def score_gmm_router_rolling(
        self,
        *,
        model: str | Path,
        reference_path: str | Path,
        frame_chunks: Iterable[Sequence[object]],
        ordered_identities: tuple[str, ...],
        router_dir: str | Path,
        work_dir: str | Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        operation = getattr(self.executor, "gmm_score_rolling", None)
        if not callable(operation):
            raise TypeError("DeepMD rolling GMM score requires one streaming worker")
        return _invoke_with_optional_progress(
            operation,
            model=_deepmd_model_file(model, presentation_dir=work),
            reference_path=Path(reference_path).resolve(),
            frame_chunks=frame_chunks,
            ordered_identities=tuple(str(item) for item in ordered_identities),
            router_dir=Path(router_dir).absolute(),
            work_dir=work.absolute(),
            emit_frame_means=bool(emit_frame_means),
            progress_reporter=progress_reporter,
            progress_systems=progress_systems,
        )


    def score_gmm_router_stream(
        self,
        *,
        model: str | Path,
        reference_path: str | Path,
        frame_chunks: Iterable[Sequence[object]],
        ordered_identities: tuple[str, ...],
        router_dir: str | Path,
        work_dir: str | Path,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        operation = getattr(self.executor, "gmm_score_stream", None)
        if not callable(operation):
            raise TypeError("DeepMD framed GMM score requires one streaming worker")
        return _invoke_with_optional_progress(
            operation,
            model=_deepmd_model_file(model, presentation_dir=work),
            reference_path=Path(reference_path).resolve(),
            frame_chunks=frame_chunks,
            ordered_identities=tuple(str(item) for item in ordered_identities),
            router_dir=Path(router_dir).absolute(),
            work_dir=work.absolute(),
            emit_frame_means=bool(emit_frame_means),
            progress_reporter=progress_reporter,
            progress_systems=progress_systems,
        )

    def fit_and_score_gmm_screen(
        self,
        *,
        model: str | Path,
        reference_path: str | Path,
        frame_chunks: Iterable[Sequence[object]],
        ordered_identities: tuple[str, ...],
        router_dir: str | Path,
        work_dir: str | Path,
        type_map: tuple[str, ...],
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ):
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        operation = getattr(self.executor, "gmm_screen", None)
        if not callable(operation):
            raise TypeError("DPA4C fused screen requires one native gmm_screen worker")
        return _invoke_with_optional_progress(
            operation,
            model=_deepmd_model_file(model, presentation_dir=work),
            reference_path=Path(reference_path).resolve(),
            frame_chunks=frame_chunks,
            ordered_identities=tuple(str(item) for item in ordered_identities),
            router_dir=Path(router_dir).absolute(),
            work_dir=work.absolute(),
            type_map=tuple(str(item) for item in type_map),
            emit_frame_means=bool(emit_frame_means),
            progress_reporter=progress_reporter,
            progress_systems=progress_systems,
        )


@dataclass(frozen=True, slots=True)
class DeepMDStudentBackendFactory:
    executor: DeepMDStudentExecutor
    cli: str

    def __post_init__(self) -> None:
        if self.cli not in {"tf", "pt"}:
            raise ValueError("DeepMD student role factory requires tf or pt CLI")

    def bind(self, binding: StudentBackendBinding) -> DeepMDStudentBackend:
        recipe = json.loads(binding.recipe_path.read_text(encoding="utf-8"))
        has_stream = callable(getattr(self.executor, "gmm_score_stream", None))
        has_rolling = callable(getattr(self.executor, "gmm_score_rolling", None))
        has_fused = callable(getattr(self.executor, "gmm_screen", None))
        fused_readout = _deepmd_recipe_kind(recipe) == "dpa4c" or (
            has_fused and not has_stream and not has_rolling
        )
        adapter = DeepMDStudentAdapter(
            run_json_path=binding.run_json_path,
            executor=self.executor,
        )
        readout_ops = (
            DeepMDFusedStudentReadoutOps(adapter)
            if fused_readout
            else adapter
        )
        sampling_models = DeepMDSamplingModelProvider(
            adapter=adapter,
            binding=binding,
            cli=self.cli,
        )
        from workflow_backends.student.role import (
            StudentScreenOperationProfiles,
            screen_profile_id,
        )

        return DeepMDStudentBackend(
            adapter=adapter,
            binding=binding,
            cli=self.cli,
            readout_ops=readout_ops,
            readout=StudentNativeReadoutAdapter(readout_ops),
            sampling_models=sampling_models,
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
class DeepMDFusedStudentReadoutOps:
    adapter: DeepMDStudentAdapter

    def fit_reference(
        self, request: FusedStudentReferenceRequest
    ) -> GMMReferenceFitResult:
        return self.adapter.fit_gmm_reference(
            model=request.model,
            reference_path=request.reference_path,
            router_dir=request.router_dir,
            work_dir=request.work_dir,
            type_map=request.type_map,
            progress_reporter=request.progress_reporter,
        )

    def fit_and_score(
        self, request: FusedStudentScreenRequest
    ) -> FusedStudentReadoutResult:
        result = self.adapter.fit_and_score_gmm_screen(
            model=request.model,
            reference_path=request.reference_path,
            frame_chunks=request.frame_chunks,
            ordered_identities=request.ordered_identities,
            router_dir=request.router_dir,
            work_dir=request.work_dir,
            type_map=request.type_map,
            emit_frame_means=request.emit_frame_means,
            progress_reporter=request.score_progress_reporter,
            progress_systems=(
                None
                if request.progress_systems is None
                else dict(request.progress_systems)
            ),
        )
        return FusedStudentReadoutResult(
            fit=result.fit,
            score=result.score,
            phase_timings=result.phase_timings,
        )


@dataclass(frozen=True, slots=True)
class DeepMDSamplingModelProvider:
    """Dedicated DeepMD capability for effective sampling-model resolution."""

    adapter: DeepMDStudentAdapter
    binding: StudentBackendBinding
    cli: str

    def prepare_sampling_models(
        self, request: SamplingModelsRequest
    ) -> SamplingModelsResult:
        result = SamplingModelsResult(
            self.adapter.prepare_sampling_models(
                models=_published_model_handles(request.models),
                sample_dir=request.sample_dir,
                cli=self.cli,
                compress=self.binding.compress_sampling_models,
            )
        )
        validate_sampling_models_result(request, result)
        return result

    def load_sampling_models(
        self, request: SamplingModelsRequest
    ) -> SamplingModelsResult:
        handles = _published_model_handles(request.models)
        if not self.binding.compress_sampling_models:
            recipe = json.loads(
                self.binding.recipe_path.read_text(encoding="utf-8")
            )
            if _deepmd_recipe_kind(recipe) != "dpa4c":
                result = SamplingModelsResult(
                    OrderedSamplingModels(
                        tuple(handle.path for handle in handles),
                        member_ids=tuple(handle.member_id for handle in handles),
                    )
                )
            else:
                result = SamplingModelsResult(
                    self.adapter.prepare_sampling_models(
                        models=handles,
                        sample_dir=request.sample_dir,
                        cli=self.cli,
                        compress=False,
                    )
                )
        else:
            context = getattr(self.adapter.executor, "compress_context", None)
            if context is None:
                raise ValueError(
                    "DeepMD sampling compression requires its selected operation context"
                )
            result = SamplingModelsResult(
                load_sampling_model_assets(
                    sample_dir=request.sample_dir,
                    source_models=handles,
                    cli=self.cli,
                    expected_operation_profile=operation_profile_evidence(context),
                )
            )
        validate_sampling_models_result(request, result)
        return result


@dataclass(slots=True)
class DeepMDStudentBackend:
    """Bound DeepMD train and effective-sampling-model capabilities."""

    adapter: DeepMDStudentAdapter
    binding: StudentBackendBinding
    cli: str
    readout_ops: PhaseStudentReadoutOps | FusedStudentReadoutOps
    readout: StudentNativeReadoutAdapter
    sampling_models: DeepMDSamplingModelProvider
    screen_operation_profiles: object

    @property
    def training(self) -> DeepMDStudentBackend:
        return self

    @property
    def models(self) -> DeepMDStudentBackend:
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
    def gmm_reference_context(self) -> object | None:
        return getattr(self.adapter.executor, "gmm_reference_context", None)

    @property
    def gmm_score_context(self) -> object | None:
        return getattr(self.adapter.executor, "gmm_score_context", None)

    def train(self, request: StudentTrainRequest) -> StudentTrainResult:
        _validate_member_ids(request.member_ids, self.binding.model_shape)
        artifacts = self.adapter.train(
            recipe=self.binding.recipe_path,
            dataset_path=request.dataset_path,
            train_dir=request.train_dir,
            cli=self.cli,
            type_map=self.binding.type_map,
            model=self.binding.model_shape,
            members=len(request.member_ids),
            policy=self.binding.policy,
            foundation_model=self.binding.foundation_model,
            train_options=self.binding.train_options,
            progress_reporter_factory=request.progress_reporter_factory,
        )
        result = StudentTrainResult(artifacts)
        if tuple(item.member_id for item in result.artifacts) != request.member_ids:
            raise ValueError("DeepMD student train changed ordered member identities")
        return result

    def finalize_published_training(
        self, request: StudentTrainPublicationRequest
    ) -> None:
        _rebase_retained_training_inputs(
            stage_root=request.stage_root,
            published_root=request.published_root,
            member_ids=request.member_ids,
        )

    def recover_historical_model(
        self, request: StudentHistoricalModelRequest
    ) -> StudentHistoricalModelResult:
        from workflow_backends.student.deepmd.historical_model_recovery import (
            resolve_legacy_deepmd_student_model,
        )

        return StudentHistoricalModelResult(
            member_id=request.member_id,
            path=resolve_legacy_deepmd_student_model(
                request.member_id, request.legacy_handle
            ),
        )


def _partitioned_structure_vectors(
    executor: DeepMDStudentExecutor,
    *,
    model: Path,
    candidate_frames: Sequence[object],
    background_frames: Sequence[object],
    type_map: tuple[str, ...],
    work_dir: Path,
) -> StudentStructureVectorResult:
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
        "type_map": tuple(type_map),
        "work_dir": work / "operation",
    }
    parameters = inspect.signature(executor.structure_vectors).parameters
    if "include_element_vectors" in parameters:
        vector_kwargs["include_element_vectors"] = True
    raw_vectors = executor.structure_vectors(**vector_kwargs)
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


def _partition_element_vectors(
    raw: Mapping[str, tuple[Sequence[str], np.ndarray]] | None,
    identities: tuple[str, ...],
    *,
    token_ids: Sequence[str],
) -> Mapping[str, Mapping[str, np.ndarray]] | None:
    if raw is None:
        return None
    token_to_identity = dict(zip(token_ids, identities, strict=True))
    result: dict[str, dict[str, np.ndarray]] = {}
    for element, (raw_ids, rows) in raw.items():
        values = {
            token_to_identity[token]: np.asarray(row, dtype=np.float32)
            for token, row in zip(raw_ids, rows, strict=True)
            if token in token_to_identity
        }
        if values:
            result[str(element)] = values
    return result


def _published_model_handles(models: object) -> tuple[PublishedModelHandle, ...]:
    paths = tuple(getattr(models, "paths", ()))
    member_ids = tuple(getattr(models, "member_ids", ()))
    if len(paths) != len(member_ids):
        raise ValueError("DeepMD sampling models require complete member identities")
    return tuple(
        PublishedModelHandle(member_id, path)
        for member_id, path in zip(member_ids, paths, strict=True)
    )


def _validate_member_ids(member_ids: tuple[str, ...], model_shape: str) -> None:
    expected = (
        ("student",)
        if model_shape == "single"
        else tuple(f"student_{index:02d}" for index in range(1, len(member_ids) + 1))
    )
    if member_ids != expected or (model_shape == "committee" and len(member_ids) < 2):
        raise ValueError(
            "DeepMD student train requires canonical ordered member identities"
        )


def _rebase_retained_training_inputs(
    *,
    stage_root: Path,
    published_root: Path,
    member_ids: tuple[str, ...],
) -> None:
    stage = Path(stage_root).resolve()
    published = Path(published_root).resolve()
    for member_id in member_ids:
        input_json = stage / member_id / "input.json"
        try:
            rendered = json.loads(input_json.read_text(encoding="utf-8"))
            training_data = rendered["training"]["training_data"]
            systems = training_data["systems"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"DeepMD retained input is missing training systems for {member_id}"
            ) from exc
        if (
            not isinstance(systems, list)
            or not systems
            or any(not isinstance(item, str) or not item for item in systems)
        ):
            raise ValueError(
                f"DeepMD retained input has invalid training systems for {member_id}"
            )
        rebased: list[str] = []
        for item in systems:
            source = Path(item).resolve()
            try:
                relative = source.relative_to(stage)
            except ValueError as exc:
                raise ValueError(
                    f"DeepMD retained training system escapes train staging for {member_id}"
                ) from exc
            rebased.append(str(published / relative))
        training_data["systems"] = rebased
        input_json.write_text(json.dumps(rendered, indent=2), encoding="utf-8")


def _deepmd_recipe_kind(recipe: object) -> str:
    if isinstance(recipe, Mapping) and deepmd_recipe_model_family(recipe) == "dpa4c":
        return "dpa4c"
    return "standard"


def _rendered_recipe_kind(input_json: Path, *, member_id: str) -> str:
    if (
        input_json.is_symlink()
        or not input_json.is_file()
        or input_json.stat().st_size <= 0
    ):
        raise ValueError(
            f"DeepMD sampling is missing rendered input.json for {member_id}"
        )
    try:
        rendered = json.loads(input_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"DeepMD sampling input.json is invalid for {member_id}"
        ) from exc
    return _deepmd_recipe_kind(rendered)


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


def _deepmd_model_file(model: str | Path, *, presentation_dir: str | Path) -> Path:
    """Present one canonical DeepMD handle to the suffix-sensitive GMM loader."""
    canonical = Path(model).absolute()
    if canonical.is_symlink():
        raise ValueError(f"DeepMD GMM requires a regular canonical model handle: {canonical}")
    if canonical.suffix not in {".pb", ".pt2"}:
        raise ValueError(
            f"DeepMD GMM requires a canonical .pb or .pt2 model handle: {canonical}"
        )
    if not canonical.is_file() or canonical.stat().st_size <= 0:
        raise ValueError(f"DeepMD GMM model handle is missing or empty: {canonical}")
    return link_model_file(
        canonical, Path(presentation_dir) / f"deepmd_model{canonical.suffix}"
    )


def _combine_gmm_scores(
    results: tuple[GMMScoreResult, ...] | list[GMMScoreResult], *, emit_frame_means: bool
) -> GMMScoreResult:
    if not results:
        raise ValueError("DeepMD-TF GMM chunk score returned no results")
    schema = results[0].descriptor_schema
    if schema is None or any(result.descriptor_schema != schema for result in results):
        raise ValueError("DeepMD-TF GMM score chunks changed readout-input descriptor schema")
    means = None
    if emit_frame_means:
        if any(result.frame_means is None for result in results):
            raise ValueError("DeepMD-TF GMM score chunks omitted requested frame means")
        means = np.concatenate(
            [np.asarray(result.frame_means, dtype=np.float32) for result in results], axis=0
        )
    elif any(result.frame_means is not None for result in results):
        raise ValueError("DeepMD-TF random GMM score unexpectedly returned frame means")
    from workflow_backends.common.inference_pool import merge_gmm_shards
    return merge_gmm_shards(results, tuple(str(identity) for result in results for identity in result.frame_ids))
