from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np

from workflow_backends.common.inference_pool import (
    inference_pool_enabled, shard_labels, shard_stream_score, shard_file_score,
    single_gpu_context,
)
from ase import Atoms

from workflow_artifacts.gmm_worker_evidence import gmm_score_chunks_payload
from workflow_backends.common.model_files import link_model_file
from workflow_backends.deepmd_common.progress import DeepMDTrainingProgressProbe
from workflow_core.model_families import deepmd_recipe_model_family
from workflow_backends.deepmd_common.train_options import (
    append_train_options,
    reject_automatic_sel_for_skip_neighbor_stat,
)
from workflow_backends.deepmd_common.type_maps import (
    resolve_foundation_type_map_operation,
)
from workflow_backends.descriptor_stream import validated_readout_input_schema
from workflow_backends.execution import run_operation, structured_result_stdout
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)
from workflow_core.structures.exact_identity import EXACT_STRUCTURE_DECIMALS
from workflow_artifacts.lifecycle.gmm_io import (
    POLICY_ID,
    GMMReferenceFitResult,
    GMMScoreResult,
    validate_gmm_router,
)
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.progress import (
    GenericWorkSnapshot,
    OperationProgressReporter,
)
from workflow_runtime.lifecycle.progress.worker_observations import (
    DescriptorObservationProjector,
    LabelObservationProjector,
    ScoreObservationProjector,
    WorkerObservationCursor,
    WorkerObservationProgressProbe,
    WorkerObservationScope,
)


@dataclass(frozen=True, slots=True)
class DeepMDNativeTrainResult:
    terminal_model_path: Path
    freeze_source_checkpoint_path: Path | None = None
    checkpoint_family: str | None = None
    selected_checkpoint_argument: str | None = None
    freeze_command_profile: Mapping[str, Any] = field(default_factory=dict)
    model_family: str | None = None
    native_type_map: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_model_path", Path(self.terminal_model_path).resolve())
        if self.freeze_source_checkpoint_path is not None:
            object.__setattr__(
                self,
                "freeze_source_checkpoint_path",
                Path(self.freeze_source_checkpoint_path).resolve(),
            )
        if self.checkpoint_family is not None and self.checkpoint_family not in {"regular", "ema"}:
            raise ValueError("DeepMD native checkpoint family must be regular or ema")
        if self.selected_checkpoint_argument is not None and not str(self.selected_checkpoint_argument).strip():
            raise ValueError("DeepMD native selected checkpoint argument must be non-empty")
        object.__setattr__(self, "freeze_command_profile", MappingProxyType(dict(self.freeze_command_profile)))
        object.__setattr__(self, "native_type_map", tuple(str(item) for item in self.native_type_map))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def __fspath__(self) -> str:
        return str(self.terminal_model_path)

    @property
    def readout_source_path(self) -> Path | None:
        return self.freeze_source_checkpoint_path

    @property
    def freeze_source_checkpoint(self) -> Path | None:
        return self.freeze_source_checkpoint_path


@dataclass(frozen=True, slots=True)
class NativeDeepMDTeacherExecutor:
    requires_native_dataset: ClassVar[bool] = True
    train_context: OperationContext
    label_context: OperationContext | None = None
    gmm_reference_context: OperationContext | None = None
    gmm_score_context: OperationContext | None = None

    def foundation_type_map(
        self, model: Path, *, work_dir: Path
    ) -> tuple[str, ...]:
        return resolve_foundation_type_map_operation(
            self.train_context, model, work_dir=work_dir
        )

    def structure_vectors(
        self,
        *,
        model: Path,
        input_path: Path,
        type_map: tuple[str, ...],
        work_dir: Path,
    ):
        context = self.gmm_score_context
        if context is None:
            raise ValueError("DeepMD teacher structural readout requires score context")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        type_map_path = work / "teacher_structure_type_map.json"
        compact_root = work / "compact_teacher_structure_vectors"
        type_map_path.write_text(json.dumps(list(type_map)), encoding="utf-8")
        from workflow_backends.teacher.structure_vector_worker import RESULT_PREFIX

        result = run_operation(
            context,
            "python",
            (
                "-m",
                "workflow_backends.teacher.structure_vector_worker",
                "--backend",
                "deepmd",
                "--model",
                str(model),
                "--input",
                str(input_path),
                "--type-map",
                str(type_map_path),
                "--output",
                str(compact_root),
                "--chunk-frames",
                str(int(context.runtime.get("descriptor_chunk_frames", 16))),
                "--batch-size",
                str(int(context.runtime.get("batch_size", 9472))),
            ),
            cwd=work,
            diagnostic_name="deepmd_teacher_structure_vectors",
            structured_result_prefix=RESULT_PREFIX,
            structured_result_limit_bytes=64 * 1024,
        )
        payload = structured_result_stdout(result, prefix=RESULT_PREFIX)
        lines = [
            line[len(RESULT_PREFIX) :]
            for line in payload.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        if len(lines) != 1:
            raise ValueError("DeepMD teacher structural readout returned no receipt")
        from workflow_backends.compact_structure_vector_store import (
            read_compact_structure_vector_store,
        )

        return read_compact_structure_vector_store(
            json.loads(lines[0]),
            expected_root=compact_root,
            expected_model_digest=_file_digest(Path(model).resolve()),
            include_elements=True,
        )

    def train(
        self,
        input_json: Path,
        member_dir: Path,
        cli: str,
        *,
        foundation_model: Path | None = None,
        train_options: Mapping[str, Any] | None = None,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> DeepMDNativeTrainResult:
        if cli not in {"tf", "pt"}:
            raise ValueError("DeepMD lifecycle native executor requires tf or pt CLI")
        input_json = Path(input_json).resolve()
        member_dir = Path(member_dir).resolve()
        train_args = [f"--{cli}", "train", str(input_json)]
        if foundation_model is not None:
            foundation = Path(foundation_model).resolve()
            if not foundation.is_file() or foundation.stat().st_size <= 0:
                raise ValueError(f"DeepMD lifecycle foundation model is missing or empty: {foundation}")
            train_args.extend(("--finetune", str(foundation)))
        rendered = json.loads(input_json.read_text(encoding="utf-8"))
        if not isinstance(rendered, dict):
            raise ValueError("DeepMD lifecycle native input must be an object")
        reject_automatic_sel_for_skip_neighbor_stat(rendered, train_options)
        append_train_options(train_args, train_options)
        probe = None
        if progress_reporter is not None:
            try:
                probe = DeepMDTrainingProgressProbe.from_rendered_input(
                    rendered,
                    member_dir=member_dir,
                    member_id=progress_reporter.operation_id,
                    member_index=member_index,
                    total_members=total_members,
                    cli=cli,
                )
            except Exception:
                probe = None
        try:
            run_operation(
                self.train_context,
                "dp",
                train_args,
                cwd=member_dir,
                diagnostic_name="deepmd_train",
                progress_reporter=progress_reporter,
                progress_probe=probe,
                probe_interval_seconds=float(
                    self.train_context.runtime.get("progress_probe_interval_seconds", 1.0)
                ),
                close_progress_reporter=False,
            )
            checkpoint_arg, checkpoint_path = _selected_checkpoint(rendered, member_dir=member_dir, cli=cli)
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
                raise ValueError(f"DeepMD lifecycle selected checkpoint is missing or empty: {checkpoint_path}")
            output_arg = "native" if cli == "pt" else "native.pb"
            model = member_dir / ("native.pt2" if cli == "pt" else "native.pb")
            freeze_args = [f"--{cli}", "freeze", "-c", checkpoint_arg, "-o", output_arg]
            if progress_reporter is not None:
                progress_reporter.report(GenericWorkSnapshot(activity="freezing model"))
            run_operation(
                self.train_context,
                "dp",
                freeze_args,
                cwd=member_dir,
                diagnostic_name="deepmd_freeze",
                progress_reporter=progress_reporter,
            )
        except BaseException:
            if progress_reporter is not None and not progress_reporter.closed:
                progress_reporter.close()
            raise
        if not model.is_file() or model.stat().st_size <= 0:
            raise ValueError("DeepMD lifecycle native freeze did not produce a model")
        type_map = rendered.get("model", {}).get("type_map")
        if not isinstance(type_map, list) or not type_map or not all(isinstance(item, str) and item for item in type_map):
            raise ValueError("DeepMD lifecycle native input must declare a non-empty model.type_map")
        (member_dir / "deepmd_model_metadata.json").write_text(
            json.dumps(
                {
                    "model_family": _model_family(rendered),
                    "type_map": type_map,
                    "native_model_suffix": ".pt2" if cli == "pt" else ".pb",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return DeepMDNativeTrainResult(
            terminal_model_path=model,
            freeze_source_checkpoint_path=(checkpoint_path if cli == "pt" else None),
            checkpoint_family=(
                _checkpoint_family(rendered) if cli == "pt" else None
            ),
            selected_checkpoint_argument=(checkpoint_arg if cli == "pt" else None),
            freeze_command_profile={
                "role": self.train_context.role,
                "profile_id": self.train_context.profile_id,
                "cli": cli,
                "freeze": "dp --<cli> freeze",
            },
            model_family=_model_family(rendered),
            native_type_map=tuple(type_map),
            evidence={
                "train_command": ("dp", *train_args),
                "freeze_command": ("dp", *freeze_args),
                "freeze_output_path": str(model.resolve()),
            },
        )

    def label(self, models: tuple[Path, ...], frames: list[Atoms], cli: str, work_dir: Path, *, progress_reporter: OperationProgressReporter | None = None, trusted_below: float | None = None) -> list[Atoms]:
        if cli not in {"tf", "pt"}:
            raise ValueError("DeepMD lifecycle native executor requires tf or pt CLI")
        if self.label_context is None:
            raise ValueError("DeepMD lifecycle ASE label execution requires its selected operation context")
        if not models:
            raise ValueError("DeepMD lifecycle ASE label execution requires at least one model")
        if frames and inference_pool_enabled(self.label_context):
            return shard_labels(
                self, context=self.label_context, models=models, frames=frames,
                work_dir=work_dir, trusted_below=trusted_below, cli=cli,
                progress_reporter=progress_reporter,
            )
        task = Path(work_dir) / "deepmd_ase_label"
        task.mkdir(parents=True, exist_ok=True)
        input_path = task / "input.xyz"
        output_path = task / "labeled.xyz"
        uncertainty_path = task / "uncertainties.json"
        config_path = task / "worker_config.json"
        serialized_frames = []
        for frame in frames:
            item = frame.copy()
            item.positions[:] = np.round(
                item.positions,
                EXACT_STRUCTURE_DECIMALS,
            )
            serialized_frames.append(item)
        write_extxyz_structures(input_path, serialized_frames, allow_empty=True)
        models = _validated_label_models(models, cli=cli)
        type_map = _model_metadata(models)
        presented_models = tuple(
            link_model_file(model, task / "models" / f"model_{index:03d}{model.suffix}")
            for index, model in enumerate(models)
        )
        config_path.write_text(
            json.dumps(
                {
                    "model_paths": [str(path) for path in presented_models],
                    "type_map": type_map,
                    "device": "cuda",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with WorkerObservationScope(task) as observations:
            probe = None
            if progress_reporter is not None:
                projector = LabelObservationProjector(
                    {str(frame.info.get("structure_id", "")): str(frame.info.get("system_id", "unknown")) for frame in frames},
                    total=len(frames),
                    trusted_below=trusted_below,
                )
                probe = WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), projector)
            run_operation(
                self.label_context,
                "python",
                [
                "-m",
                "workflow_backends.teacher.deepmd.ase_label_worker",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--uncertainties",
                str(uncertainty_path),
                "--config",
                str(config_path),
                "--observation-channel",
                str(observations.path),
                ],
                cwd=task,
                diagnostic_name="deepmd_label",
                progress_reporter=progress_reporter,
                progress_probe=probe,
            )
        labeled = read_extxyz_structures(output_path, allow_empty=True)
        values = json.loads(uncertainty_path.read_text(encoding="utf-8"))
        if len(labeled) != len(frames) or len(values) != len(frames):
            raise ValueError("DeepMD lifecycle ASE label worker returned mismatched outputs")
        for atoms, value in zip(labeled, values):
            atoms.info["force_uncertainty"] = float(value)
        return labeled

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
    ) -> GMMReferenceFitResult:
        context = self.gmm_reference_context
        if context is None:
            raise ValueError("DPA4 lifecycle gmm_reference requires its selected operation context")
        task = Path(work_dir) / "dpa4_gmm_reference"
        arguments = self._gmm_arguments(
            operation="reference",
            model=model,
            input_path=reference_path,
            router_dir=router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        arguments.extend(("--type-map-json", json.dumps(list(type_map), separators=(",", ":"))))
        if context.runtime.get("gmm_reference_spill", False):
            arguments.extend(("--spill-work-dir", str(task.resolve())))
        with WorkerObservationScope(task) as observations:
            arguments.extend(("--observation-channel", str(observations.path)))
            self._append_descriptor_runtime(arguments, context)
            probe = None if progress_reporter is None else WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), DescriptorObservationProjector(label="reference"))
            run_operation(context, "python", arguments, cwd=task, diagnostic_name="dpa4_gmm_reference", progress_reporter=progress_reporter, progress_probe=probe)
        manifest = validate_gmm_router(
            router_dir, model_digest=model_digest, reference_digest=reference_digest
        )
        return GMMReferenceFitResult(
            router_dir=Path(router_dir),
            policy_id=POLICY_ID,
            element_pca_dimensions={
                str(element): int(metadata["pca_dimensions"])
                for element, metadata in dict(manifest["elements"]).items()
            },
            descriptor_schema=dict(manifest["descriptor_schema"]),
        )

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
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        context = self.gmm_score_context
        if context is None:
            raise ValueError("DPA4 lifecycle gmm_score requires its selected operation context")
        task = Path(work_dir) / "dpa4_gmm_score"
        task.mkdir(parents=True, exist_ok=True)
        identity_filter = task / "identity_filter.json"
        identity_filter.write_text(json.dumps(list(frame_ids)), encoding="utf-8")
        arguments = self._gmm_arguments(
            operation="score",
            model=model,
            input_path=candidate_path,
            router_dir=router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        arguments.extend(("--identity-filter", str(identity_filter)))
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        try:
            with WorkerObservationScope(task) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                self._append_descriptor_runtime(arguments, context)
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), ScoreObservationProjector(progress_systems or {}, total=len(frame_ids), role="teacher"))
                operation = run_operation(
                context,
                "python",
                arguments,
                cwd=task,
                diagnostic_name="dpa4_gmm_score",
                structured_result_prefix="DICE_DPA4_GMM_RESULT=",
                structured_result_limit_bytes=64 * 1024 * 1024,
                progress_reporter=progress_reporter,
                progress_probe=probe,
            )
                return self._gmm_score_result(
                    structured_result_stdout(operation, prefix="DICE_DPA4_GMM_RESULT="),
                    frame_ids,
                    emit_frame_means,
                )
        finally:
            identity_filter.unlink(missing_ok=True)

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
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        """Score all bounded replay chunks in one terminal-owning process."""
        if inference_pool_enabled(self.gmm_score_context):
            return shard_file_score(
                self, model, context=self.gmm_score_context,
                candidate_paths=candidate_paths, frame_ids_by_chunk=frame_ids_by_chunk,
                router_dir=router_dir, work_dir=work_dir, model_digest=model_digest,
                reference_digest=reference_digest, emit_frame_means=emit_frame_means,
                progress_systems=progress_systems,
                progress_reporter=progress_reporter,
            )
        context = self.gmm_score_context
        if context is None:
            raise ValueError("DPA4 lifecycle gmm_score requires its selected operation context")
        if not candidate_paths or len(candidate_paths) != len(frame_ids_by_chunk):
            raise ValueError(
                "DPA4 lifecycle chunked gmm_score requires aligned non-empty inputs"
            )
        frame_ids = tuple(identity for chunk in frame_ids_by_chunk for identity in chunk)
        if (
            not frame_ids
            or any(not identity for identity in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or any(not chunk for chunk in frame_ids_by_chunk)
        ):
            raise ValueError(
                "DPA4 lifecycle chunked gmm_score requires unique canonical identities"
            )
        task = Path(work_dir) / "dpa4_gmm_score"
        task.mkdir(parents=True, exist_ok=True)
        chunk_manifest = task / "chunk_manifest.json"
        chunk_manifest.write_text(
            json.dumps(
                gmm_score_chunks_payload(
                    frame_count=len(frame_ids),
                    chunks=[
                        {
                            "path": str(Path(path).resolve()),
                            "frame_ids": list(ids),
                        }
                        for path, ids in zip(
                            candidate_paths, frame_ids_by_chunk, strict=True
                        )
                    ],
                ),
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        arguments = self._gmm_arguments(
            operation="score",
            model=model,
            input_path=None,
            router_dir=router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        arguments.extend(("--chunk-manifest", str(chunk_manifest.resolve())))
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        self._append_descriptor_runtime(arguments, context)
        try:
            with WorkerObservationScope(task) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = (
                    None
                    if progress_reporter is None
                    else WorkerObservationProgressProbe(
                        WorkerObservationCursor(observations.path),
                        ScoreObservationProjector(
                            progress_systems or {},
                            total=len(frame_ids),
                            role="teacher",
                        ),
                    )
                )
                operation = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=task,
                    diagnostic_name="dpa4_gmm_score",
                    structured_result_prefix="DICE_DPA4_GMM_RESULT=",
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                )
                return self._gmm_score_result(
                    structured_result_stdout(
                        operation, prefix="DICE_DPA4_GMM_RESULT="
                    ),
                    frame_ids,
                    emit_frame_means,
                )
        finally:
            chunk_manifest.unlink(missing_ok=True)

    def score_gmm_router_stream(
        self,
        model: Path,
        *,
        frame_chunks: Iterable[Sequence[object]],
        frame_ids: tuple[str, ...],
        router_dir: Path,
        work_dir: Path,
        model_digest: str,
        reference_digest: str,
        emit_frame_means: bool,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ):
        from workflow_backends.geometry_stream import GeometryStreamServer

        if inference_pool_enabled(self.gmm_score_context):
            return shard_stream_score(
                self, "score_gmm_router_stream", context=self.gmm_score_context,
                identities=frame_ids, frame_chunks=frame_chunks, identity_key="frame_ids",
                model=model, router_dir=router_dir, work_dir=work_dir,
                model_digest=model_digest, reference_digest=reference_digest,
                emit_frame_means=emit_frame_means, progress_systems=progress_systems,
                progress_reporter=progress_reporter,
            )
        context = self.gmm_score_context
        if context is None or context.role != "teacher.deepmd.pt.gmm_score":
            raise ValueError("DPA4 streamed score requires its stage-owned operation context")
        identities = tuple(str(item) for item in frame_ids)
        if not identities or any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("DPA4 streamed score requires unique canonical identities")
        task = Path(work_dir) / "dpa4_gmm_score"
        task.mkdir(parents=True, exist_ok=True)
        identity_filter = task / "identity_filter.json"
        compact_root = task / "compact_score_result"
        identity_filter.write_text(json.dumps(list(identities)), encoding="utf-8")
        byte_budget = 64 * 1024 * 1024
        atom_budget = 1_000_000
        channel = GeometryStreamServer(
            task / "geometry.sock",
            ordered_identities=identities,
            byte_budget=byte_budget,
            atom_budget=atom_budget,
        )
        arguments = self._gmm_arguments(
            operation="score_stream",
            model=model,
            input_path=None,
            router_dir=router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        arguments.extend((
            "--identity-filter", str(identity_filter),
            "--geometry-socket", str(channel.socket_path),
            "--geometry-byte-budget", str(byte_budget),
            "--geometry-atom-budget", str(atom_budget),
            "--compact-result-root", str(compact_root),
        ))
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        self._append_descriptor_runtime(arguments, context)
        try:
            with WorkerObservationScope(task) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    ScoreObservationProjector(
                        progress_systems or {}, total=len(identities), role="teacher"
                    ),
                )
                operation = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=task,
                    diagnostic_name="dpa4_gmm_score",
                    structured_result_prefix="DICE_DPA4_GMM_RESULT=",
                    structured_result_limit_bytes=64 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    input_producer=lambda process: channel.produce(
                        frame_chunks, consumer_alive=lambda: process.poll() is None
                    ),
                )
        finally:
            identity_filter.unlink(missing_ok=True)
            channel.close()
        marker = "DICE_DPA4_GMM_RESULT="
        stdout = structured_result_stdout(operation, prefix=marker)
        lines = [line[len(marker):] for line in stdout.splitlines() if line.startswith(marker)]
        if len(lines) != 1:
            raise ValueError("DPA4 streamed score returned no compact receipt")
        try:
            receipt = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError("DPA4 streamed score receipt is malformed") from exc
        from workflow_core.lifecycle.compact_scores import CompactScoreView

        return CompactScoreView(
            receipt,
            ordered_identities=identities,
            expected_model_digest=model_digest,
            expected_reference_digest=reference_digest,
        )

    @staticmethod
    def _gmm_arguments(
        *,
        operation: str,
        model: Path,
        input_path: Path | None,
        router_dir: Path,
        model_digest: str,
        reference_digest: str,
    ) -> list[str]:
        raw_model = Path(model)
        if raw_model.suffix == ".pt":
            model = _validated_gmm_model(raw_model, suffix=".pt", role="readout source")
            model_option = "--checkpoint"
        elif raw_model.suffix == ".pt2":
            model = _validated_gmm_model(raw_model, suffix=".pt2", role="terminal")
            model_option = "--terminal"
        else:
            raise ValueError(
                "DPA4 lifecycle GMM readout source must use the native .pt checkpoint "
                "or the DPA4C terminal .pt2"
            )
        arguments = [
            "-m", "workflow_backends.teacher.deepmd.descriptor_worker",
            "--operation", operation,
            model_option, str(model),
            "--router-dir", str(Path(router_dir).resolve()),
            "--model-digest", str(model_digest),
            "--reference-digest", str(reference_digest),
        ]
        if input_path is not None:
            arguments.extend(("--input", str(Path(input_path).resolve())))
        return arguments

    @staticmethod
    def _append_descriptor_runtime(arguments: list[str], context: OperationContext) -> None:
        if "descriptor_chunk_frames" in context.runtime:
            arguments.extend(("--chunk-frames", str(context.runtime["descriptor_chunk_frames"])))
        if "batch_size" in context.runtime:
            arguments.extend(("--batch-size", str(context.runtime["batch_size"])))

    @staticmethod
    def _gmm_score_result(
        stdout: str, expected_ids: tuple[str, ...], emit_frame_means: bool
    ) -> GMMScoreResult:
        marker = "DICE_DPA4_GMM_RESULT="
        position = stdout.rfind(marker)
        if position < 0:
            raise ValueError("DPA4 lifecycle gmm_score worker did not return a frame result")
        try:
            payload = json.loads(stdout[position + len(marker):].splitlines()[0])
            frame_ids = np.asarray(payload["frame_ids"], dtype=np.str_)
            frame_p = np.asarray(payload["frame_p"], dtype=np.float64)
            statistics = np.asarray(payload["frame_statistics"], dtype=np.float64)
            raw_means = payload.get("frame_means")
            schema = validated_readout_input_schema(payload["descriptor_schema"])
            means = None if raw_means is None else np.asarray(raw_means, dtype=np.float32)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("DPA4 lifecycle gmm_score worker returned malformed JSON") from exc
        count = len(expected_ids)
        if (
            tuple(frame_ids.tolist()) != expected_ids
            or frame_p.shape != (count,)
            or statistics.shape != (count,)
            or not np.all(np.isfinite(frame_p))
            or not np.all(np.isfinite(statistics))
            or np.any(frame_p <= 0.0)
            or np.any(frame_p > 1.0)
        ):
            raise ValueError("DPA4 lifecycle gmm_score changed canonical frame results")
        if emit_frame_means:
            if (
                means is None
                or means.shape != (count, int(schema["width"]))
                or not np.all(np.isfinite(means))
            ):
                raise ValueError("DPA4 lifecycle gmm_score omitted dynamic readout frame means")
        elif means is not None:
            raise ValueError("DPA4 lifecycle random gmm_score unexpectedly returned frame means")
        return GMMScoreResult(frame_ids, frame_p, statistics, means, schema)
def _selected_checkpoint(rendered: dict, *, member_dir: Path, cli: str) -> tuple[str, Path]:
    training = rendered.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("DeepMD lifecycle native input training must be an object")
    configured = str(training.get("save_ckpt", "model.ckpt")).strip()
    if not configured:
        raise ValueError("DeepMD lifecycle native training.save_ckpt must be non-empty when present")
    if any(token in configured for token in "*?["):
        raise ValueError(
            "DeepMD lifecycle native training.save_ckpt must identify one published checkpoint, "
            "not a wildcard or ambiguous pattern"
        )
    checkpoint_base = Path(configured)
    if not checkpoint_base.is_absolute():
        checkpoint_base = member_dir / checkpoint_base
    if cli == "pt":
        family = "ema" if training.get("enable_ema") is True else "regular"
        selected_configured = (
            _ema_checkpoint_prefix(configured) if family == "ema" else configured
        )
        checkpoint_arg = (
            selected_configured
            if selected_configured.endswith(".pt")
            else f"{selected_configured}.pt"
        )
        checkpoint_path = member_dir / Path(checkpoint_arg) if not Path(checkpoint_arg).is_absolute() else Path(checkpoint_arg)
        checkpoint_path = checkpoint_path.resolve()
        if checkpoint_path.is_symlink():
            try:
                resolved = checkpoint_path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"DeepMD lifecycle selected {family} checkpoint is a broken published pointer: {checkpoint_path}"
                ) from exc
        else:
            resolved = checkpoint_path
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise ValueError(
                f"DeepMD lifecycle selected {family} checkpoint is missing or empty: {checkpoint_path}"
            )
        return checkpoint_arg, resolved
    else:
        checkpoint_path = checkpoint_base if checkpoint_base.suffix == ".index" else Path(f"{checkpoint_base}.index")
        return configured, checkpoint_path.resolve()


def _ema_checkpoint_prefix(save_ckpt: str) -> str:
    """Return DeepMD's published EMA prefix for a regular checkpoint prefix."""
    path = Path(save_ckpt)
    suffix = path.suffix
    if suffix:
        return str(path.with_name(f"{path.stem}_ema{suffix}"))
    return str(path.with_name(f"{path.name}_ema"))


def _checkpoint_family(rendered: dict) -> str:
    training = rendered.get("training", {})
    enabled = isinstance(training, dict) and training.get("enable_ema") is True
    return "ema" if enabled else "regular"


def _model_family(rendered: dict) -> str:
    return deepmd_recipe_model_family(rendered)


def _validated_gmm_model(path: Path, *, suffix: str, role: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file() or raw.stat().st_size <= 0:
        raise ValueError(f"DPA4 lifecycle GMM {role} must be a non-empty regular file: {raw}")
    if raw.suffix != suffix:
        raise ValueError(f"DPA4 lifecycle GMM {role} requires native suffix {suffix}")
    return raw.resolve()


def _validated_label_models(models: tuple[Path, ...], *, cli: str) -> tuple[Path, ...]:
    expected_suffix = ".pt2" if cli == "pt" else ".pb"
    validated: list[Path] = []
    for model in models:
        path = Path(model)
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"DeepMD lifecycle ASE label model must be a non-empty regular file: {path}")
        if path.suffix != expected_suffix:
            raise ValueError(
                f"DeepMD lifecycle ASE label model {path} has suffix {path.suffix!r}; "
                f"the selected {cli!r} CLI requires {expected_suffix!r}"
            )
        validated.append(path.absolute())
    if len(set(validated)) != len(validated):
        raise ValueError("DeepMD lifecycle ASE label requires unique ordered model handles")
    return tuple(validated)


def _model_metadata(models: tuple[Path, ...]) -> list[str]:
    type_maps: set[tuple[str, ...]] = set()
    for model in models:
        if not model.is_file() or model.stat().st_size <= 0:
            raise ValueError(f"DeepMD lifecycle ASE label model is missing or empty: {model}")
        metadata_path = model.parent / "deepmd_model_metadata.json"
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"DeepMD lifecycle ASE label requires model metadata beside {model}") from exc
        type_map = payload.get("type_map")
        if not isinstance(type_map, list) or not type_map or not all(isinstance(item, str) and item for item in type_map):
            raise ValueError(f"DeepMD lifecycle ASE label metadata has invalid type_map beside {model}")
        type_maps.add(tuple(type_map))
    if len(type_maps) != 1:
        raise ValueError("DeepMD lifecycle ASE label models have incompatible metadata")
    return list(next(iter(type_maps)))


def publish_terminal_digest_metadata(handles: tuple[object, ...]) -> None:
    """Seal each published PT terminal digest into its existing model metadata."""
    for handle in handles:
        terminal = Path(handle.path)
        metadata_path = terminal.parent / "deepmd_model_metadata.json"
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"DeepMD lifecycle terminal metadata is invalid beside {terminal}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"DeepMD lifecycle terminal metadata must be an object beside {terminal}"
            )
        input_path = terminal.parent / "input.json"
        try:
            rendered = json.loads(input_path.read_text(encoding="utf-8"))
            rendered_type_map = tuple(rendered["model"]["type_map"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"DeepMD lifecycle terminal publication requires rendered type_map beside {terminal}"
            ) from exc
        declared = tuple(payload.get("type_map", rendered_type_map))
        if (
            declared != rendered_type_map
            or not declared
            or any(not isinstance(item, str) or not item for item in declared)
            or len(set(declared)) != len(declared)
        ):
            raise ValueError("DeepMD lifecycle terminal metadata type_map drifted")
        payload.update(
            {
                "type_map": list(declared),
                "native_model_suffix": terminal.suffix,
                "terminal_model_sha256": _file_digest(terminal),
            }
        )
        metadata_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )


def validate_terminal_digest_metadata(
    handles: tuple[object, ...], *, active_type_map: tuple[str, ...]
) -> bool:
    """Validate completed PT prefixes against their exact terminal bytes."""
    try:
        for handle in handles:
            terminal = Path(handle.path)
            payload = json.loads(
                (terminal.parent / "deepmd_model_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(payload, dict):
                return False
            native_type_map = tuple(payload.get("type_map", ()))
            if (
                not native_type_map
                or any(
                    not isinstance(item, str) or not item for item in native_type_map
                )
                or len(set(native_type_map)) != len(native_type_map)
                or any(item not in native_type_map for item in active_type_map)
                or payload.get("native_model_suffix") != terminal.suffix
                or payload.get("terminal_model_sha256") != _file_digest(terminal)
            ):
                return False
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return True


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
