from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Protocol, Sequence

from ase import Atoms
import numpy as np
import yaml

from workflow_backends.common.committee_pool import execute_committee_members
from workflow_backends.teacher.training import (
    TeacherTrainPublicationRequest,
    TeacherTrainRequest,
    TeacherTrainResult,
)
from workflow_backends.teacher.readout import (
    TeacherCandidateScoreRequest,
    TeacherCandidateScoreResult,
    TeacherReadoutOps,
    TeacherReferenceFitRequest,
    TeacherStructureVectors,
    partitioned_teacher_structure_vectors as _teacher_structure_vectors,
)
from workflow_backends.teacher.label_result import (
    TeacherNativeLabelRequest,
    TeacherNativeLabelResult,
)
from workflow_backends.teacher.role import (
    TeacherBackendBinding,
    TeacherHistoricalModelRequest,
    TeacherHistoricalModelResult,
    TeacherPublishedTrainingValidationRequest,
)
from workflow_artifacts.lifecycle.model_publication import ModelArtifact
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)
from workflow_policies.gmm_reference_tail import GMMReferenceFitResult, GMMScoreResult
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter
from workflow_runtime.execution.task_scheduler import DistributedAssignment


class MaceExecutor(Protocol):
    def train(
        self,
        rendered_yaml: Path,
        work_dir: Path,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path: ...

    def train_distributed(
        self,
        rendered_yaml: Path,
        work_dir: Path,
        assignment: DistributedAssignment,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path: ...

    def label(
        self,
        models: tuple[Path, ...],
        frames: list[Atoms],
        work_dir: Path,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        trusted_below: float | None = None,
    ) -> list[Atoms]: ...

    def fit_gmm_router(
        self,
        model: Path,
        *,
        reference_path: Path,
        router_dir: Path,
        work_dir: Path,
        type_map: tuple[str, ...],
        model_digest: str,
        reference_digest: str,
        progress_reporter: OperationProgressReporter | None = None,
    ) -> GMMReferenceFitResult: ...

    def score_gmm_router(
        self,
        model: Path,
        *,
        candidate_path: Path,
        frame_ids: tuple[str, ...],
        router_dir: Path,
        work_dir: Path,
        model_digest: str,
        reference_digest: str,
        emit_frame_means: bool,
    ) -> GMMScoreResult: ...

    def score_gmm_router_chunks(
        self,
        model: Path,
        *,
        candidate_paths: tuple[Path, ...],
        frame_ids_by_chunk: tuple[tuple[str, ...], ...],
        router_dir: Path,
        work_dir: Path,
        model_digest: str,
        reference_digest: str,
        emit_frame_means: bool,
    ) -> GMMScoreResult: ...


@dataclass(slots=True)
class MaceTeacherAdapter:
    """MACE-native lifecycle teacher adapter; lifecycle publication stays in lifecycle core."""

    executor: MaceExecutor
    run_json_path: Path = Path("run.json")
    iter_index: int = 0

    def __post_init__(self) -> None:
        self.run_json_path = Path(self.run_json_path).resolve()
        self.iter_index = int(self.iter_index)

    @property
    def readout_ops(self) -> TeacherReadoutOps:
        return self

    def fit_reference(
        self, request: TeacherReferenceFitRequest
    ) -> GMMReferenceFitResult:
        model = self._single_gmm_model((request.model,))
        return self.executor.fit_gmm_router(
            model,
            reference_path=request.reference_path,
            router_dir=request.router_dir,
            work_dir=request.work_dir,
            type_map=request.type_map,
            model_digest=_file_digest(model),
            reference_digest=_file_digest(request.reference_path),
            progress_reporter=request.progress_reporter,
        )

    def score_candidates(
        self, request: TeacherCandidateScoreRequest
    ) -> TeacherCandidateScoreResult:
        model = self._single_gmm_model((request.model,))
        chunks_root = request.work_dir / "score_chunks"
        paths: list[Path] = []
        identities_by_chunk: list[tuple[str, ...]] = []
        try:
            for chunk_index, raw_chunk in enumerate(request.frame_chunks):
                chunk = tuple(raw_chunk)
                identities = tuple(
                    str(frame.info.get("structure_id", "")) for frame in chunk
                )
                if not chunk or any(not identity for identity in identities):
                    raise ValueError(
                        "MACE lifecycle GMM score requires non-empty canonical chunks"
                    )
                chunk_root = chunks_root / f"chunk_{chunk_index:06d}"
                chunk_root.mkdir(parents=True, exist_ok=True)
                paths.append(
                    write_extxyz_structures(
                        chunk_root / "candidates.xyz", chunk, allow_empty=False
                    )
                )
                identities_by_chunk.append(identities)
            flattened = tuple(
                identity for chunk in identities_by_chunk for identity in chunk
            )
            if flattened != request.ordered_identities:
                raise ValueError(
                    "MACE lifecycle GMM chunks changed canonical candidate order"
                )
            model_digest = _file_digest(model)
            reference_digest = _file_digest(request.reference_path)
            chunk_operation = getattr(
                self.executor, "score_gmm_router_chunks", None
            )
            if callable(chunk_operation):
                native = chunk_operation(
                    model,
                    candidate_paths=tuple(paths),
                    frame_ids_by_chunk=tuple(identities_by_chunk),
                    router_dir=request.router_dir,
                    work_dir=request.work_dir,
                    model_digest=model_digest,
                    reference_digest=reference_digest,
                    emit_frame_means=request.emit_frame_means,
                    progress_reporter=request.progress_reporter,
                    progress_systems=request.progress_systems,
                )
            else:
                results = [
                    self.executor.score_gmm_router(
                        model,
                        candidate_path=path,
                        frame_ids=identities,
                        router_dir=request.router_dir,
                        work_dir=request.work_dir / f"chunk_{index:06d}",
                        model_digest=model_digest,
                        reference_digest=reference_digest,
                        emit_frame_means=request.emit_frame_means,
                        progress_reporter=request.progress_reporter,
                        progress_systems=request.progress_systems,
                    )
                    for index, (path, identities) in enumerate(
                        zip(paths, identities_by_chunk, strict=True)
                    )
                ]
                native = _combine_gmm_scores(
                    results, emit_frame_means=request.emit_frame_means
                )
            return TeacherCandidateScoreResult(
                ordered_identities=tuple(str(item) for item in native.frame_ids.tolist()),
                frame_p=native.frame_p,
                frame_statistics=native.frame_statistics,
                frame_means=native.frame_means,
                descriptor_schema=native.descriptor_schema,
            )
        finally:
            shutil.rmtree(chunks_root, ignore_errors=True)

    def read_structure_vectors(
        self,
        *,
        model: Path,
        candidate_frames: Sequence[Atoms],
        background_frames: Sequence[Atoms],
        type_map: tuple[str, ...],
        work_dir: Path,
    ) -> TeacherStructureVectors:
        return _teacher_structure_vectors(
            self.executor,
            model=self._single_gmm_model((model,)),
            candidate_frames=candidate_frames,
            background_frames=background_frames,
            type_map=type_map,
            work_dir=work_dir,
        )

    def finalize_published_training(
        self, request: TeacherTrainPublicationRequest
    ) -> None:
        if any(
            artifact.backend_asset is not None
            for artifact in request.train.member_artifacts
        ):
            raise ValueError("MACE teacher training returned an unsupported backend asset")

    def train(self, request: TeacherTrainRequest) -> TeacherTrainResult:
        native_recipe = yaml.safe_load(request.recipe_path.read_text(encoding="utf-8")) or {}
        if not isinstance(native_recipe, dict):
            raise ValueError("MACE lifecycle recipe must be a YAML object")
        if "foundation_model" in native_recipe:
            raise ValueError(
                "MACE lifecycle native recipe foundation_model conflicts with the resolved workflow policy"
            )
        train_root = request.train_dir
        native_phase = "stage_two" if native_recipe.get("swa") is True else "stage_one"

        def execute_member(
            member: tuple[int, str],
            assignment: DistributedAssignment | None = None,
        ) -> ModelArtifact:
            member_number, member_id = member
            work_dir = train_root / member_id
            validation_path, validation_evidence = self._mace_validation_input(
                request,
                work_dir=work_dir,
                replica_count=(
                    1 if assignment is None else len(assignment.visible_devices)
                ),
            )
            rendered = dict(native_recipe)
            rendered.update(
                {
                    "name": member_id,
                    "model_dir": str(work_dir.resolve()),
                    "train_file": str(request.train_data_path),
                    "valid_file": str(validation_path),
                    "valid_fraction": 0.0,
                    "seed": self._member_seed(member_number, iter_index=request.iter_index),
                }
            )
            if validation_evidence["mode"] == "ddp_monitor":
                rendered["valid_batch_size"] = validation_evidence["frame_count"]
            if request.foundation_asset is not None:
                rendered["foundation_model"] = str(request.foundation_asset.path)
            rendered_yaml = work_dir / "rendered_mace.yaml"
            work_dir.mkdir(parents=True, exist_ok=True)
            rendered_yaml.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
            reporter = (
                request.progress_reporter_factory(member_id)
                if request.progress_reporter_factory is not None
                else None
            )
            operation = (
                self._execute_train(
                    rendered_yaml,
                    work_dir,
                    progress_reporter=reporter,
                    member_index=member_number,
                    total_members=len(request.member_ids),
                )
                if assignment is None
                else self._execute_train_distributed(
                    rendered_yaml,
                    work_dir,
                    assignment,
                    progress_reporter=reporter,
                    member_index=member_number,
                    total_members=len(request.member_ids),
                )
            )
            terminal_model = Path(operation).resolve()
            return ModelArtifact(
                member_id=member_id,
                terminal_model_path=terminal_model,
                evidence={
                    "native_phase": native_phase,
                    "output_kind": "mace_model",
                    "validation": validation_evidence,
                },
            )
        members = tuple(enumerate(request.member_ids, start=1))
        artifacts = execute_committee_members(
            members,
            context=getattr(self.executor, "train_context", None),
            task_id=lambda member: member[1],
            execute=execute_member,
            execute_distributed=lambda member, assignment: execute_member(
                member, assignment
            ),
        )
        return TeacherTrainResult(
            member_artifacts=artifacts,
            backend_evidence={"backend": "mace", "cli": "mace_run_train"},
        )

    @staticmethod
    def _mace_validation_input(
        request: TeacherTrainRequest,
        *,
        work_dir: Path,
        replica_count: int,
    ) -> tuple[Path, dict[str, object]]:
        """Present canonical validation intent through MACE's mandatory loader.

        ``self`` means that no structures are withheld from optimization. MACE
        still requires a non-empty validation loader for scheduler, early-stop,
        and checkpoint control flow, so use a small deterministic monitor rather
        than replaying the complete training set at every validation interval.
        Distributed MACE drops incomplete validation batches; one evenly spaced
        structure per local rank keeps every rank's loader non-empty.
        """

        if request.validation_mode == "holdout":
            return request.validation_data_path, {
                "mode": "holdout",
                "frame_count": None,
            }
        if replica_count < 1:
            raise ValueError("MACE validation monitor replica_count must be positive")

        frames = read_extxyz_structures(request.train_data_path, allow_empty=False)
        if len(frames) < replica_count:
            raise ValueError(
                "MACE self-validation monitor requires at least one training "
                "structure per local DDP rank"
            )
        if replica_count == 1:
            indices = (len(frames) // 2,)
        else:
            indices = tuple(
                rank * (len(frames) - 1) // (replica_count - 1)
                for rank in range(replica_count)
            )
        monitor_path = Path(work_dir) / "validation_monitor.xyz"
        write_extxyz_structures(
            monitor_path,
            [frames[index] for index in indices],
            allow_empty=False,
        )
        return monitor_path.resolve(), {
            "mode": "ddp_monitor",
            "frame_count": replica_count,
            "source_indices": indices,
        }

    def _execute_train(
        self,
        rendered_yaml: Path,
        work_dir: Path,
        *,
        progress_reporter: OperationProgressReporter | None,
        member_index: int,
        total_members: int,
    ) -> Path:
        return self.executor.train(
            rendered_yaml,
            work_dir,
            progress_reporter=progress_reporter,
            member_index=member_index,
            total_members=total_members,
        )

    def _execute_train_distributed(
        self,
        rendered_yaml: Path,
        work_dir: Path,
        assignment: DistributedAssignment,
        *,
        progress_reporter: OperationProgressReporter | None,
        member_index: int,
        total_members: int,
    ) -> Path:
        return self.executor.train_distributed(
            rendered_yaml,
            work_dir,
            assignment,
            progress_reporter=progress_reporter,
            member_index=member_index,
            total_members=total_members,
        )

    def label(
        self, request: TeacherNativeLabelRequest
    ) -> TeacherNativeLabelResult:
        aliases = tuple(Path(model).resolve() for model in request.models)
        labeled = self.executor.label(
            aliases,
            list(request.frames),
            request.work_dir,
            progress_reporter=request.progress_reporter,
            trusted_below=request.trusted_below,
        )
        return TeacherNativeLabelResult(tuple(labeled))

    @staticmethod
    def _single_gmm_model(models: Sequence[str | Path]) -> Path:
        if len(models) != 1:
            raise ValueError("MACE lifecycle GMM operation requires exactly one model")
        return Path(models[0]).resolve()

    def _member_seed(self, member: int, *, iter_index: int | None = None) -> int:
        resolved_iter = self.iter_index if iter_index is None else int(iter_index)
        entropy = f"{self.run_json_path}\0{resolved_iter}\0teach/train\0mace\0{member}"
        return int.from_bytes(hashlib.sha256(entropy.encode("utf-8")).digest()[:4], "big")


@dataclass(frozen=True, slots=True)
class MaceTeacherBackendFactory:
    """Select MACE once, then bind only validated route-stable inputs."""

    executor: MaceExecutor

    def bind(self, binding: TeacherBackendBinding) -> MaceTeacherBackend:
        from workflow_backends.teacher.role import (
            TeacherScreenOperationProfiles,
            screen_profile_id,
        )

        return MaceTeacherBackend(
            adapter=MaceTeacherAdapter(
                executor=self.executor,
                run_json_path=binding.run_json_path,
                iter_index=binding.iter_index,
            ),
            binding=binding,
            screen_operation_profiles=TeacherScreenOperationProfiles(
                gmm_score=screen_profile_id(
                    getattr(self.executor, "gmm_score_context", None)
                ),
                label=screen_profile_id(getattr(self.executor, "label_context", None)),
            ),
        )


@dataclass(slots=True)
class MaceTeacherBackend:
    adapter: MaceTeacherAdapter
    binding: TeacherBackendBinding
    screen_operation_profiles: object

    @property
    def readout_ops(self) -> TeacherReadoutOps:
        return self.adapter

    @property
    def gmm_reference_context(self):
        return getattr(self.adapter.executor, "gmm_reference_context", None)

    @property
    def gmm_score_context(self):
        return getattr(self.adapter.executor, "gmm_score_context", None)

    def train(self, request: TeacherTrainRequest) -> TeacherTrainResult:
        return self.adapter.train(request)

    def finalize_published_training(
        self, request: TeacherTrainPublicationRequest
    ) -> None:
        self.adapter.finalize_published_training(request)

    def label(
        self, request: TeacherNativeLabelRequest
    ) -> TeacherNativeLabelResult:
        return self.adapter.label(request)

    def validate_published_training(
        self, request: TeacherPublishedTrainingValidationRequest
    ) -> bool:
        if request.member_ids != self.binding.member_ids:
            return False
        return all(
            handle.path.suffix == ".model"
            and handle.path.parent.name == handle.member_id
            for handle in request.handles
        )

    def recover_historical_model(
        self, request: TeacherHistoricalModelRequest
    ) -> TeacherHistoricalModelResult:
        from workflow_backends.teacher.mace.historical_model_recovery import (
            resolve_legacy_mace_teacher_model,
        )

        return TeacherHistoricalModelResult(
            request.member_id,
            resolve_legacy_mace_teacher_model(
                request.member_id, request.legacy_handle
            ),
        )


def _combine_gmm_scores(
    results: Sequence[GMMScoreResult], *, emit_frame_means: bool
) -> GMMScoreResult:
    if not results:
        raise ValueError("MACE lifecycle GMM chunk score returned no results")
    schema = results[0].descriptor_schema
    if schema is None or any(result.descriptor_schema != schema for result in results):
        raise ValueError("MACE lifecycle GMM score chunks changed readout-input descriptor schema")
    means = None
    if emit_frame_means:
        if any(result.frame_means is None for result in results):
            raise ValueError("MACE lifecycle GMM score chunks omitted requested frame means")
        means = np.concatenate(
            [np.asarray(result.frame_means, dtype=np.float32) for result in results], axis=0
        )
    elif any(result.frame_means is not None for result in results):
        raise ValueError("MACE lifecycle random GMM score unexpectedly returned frame means")
    return GMMScoreResult(
        frame_ids=np.concatenate([result.frame_ids for result in results]),
        frame_p=np.concatenate([result.frame_p for result in results]),
        frame_statistics=np.concatenate([result.frame_statistics for result in results]),
        frame_means=means,
        descriptor_schema=schema,
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
