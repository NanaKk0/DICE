from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from workflow_backends.common.inference_pool import (
    inference_pool_enabled, shard_labels, shard_stream_score, shard_file_score,
    single_gpu_context,
)

from workflow_backends.descriptor_stream import validated_readout_input_schema
from workflow_artifacts.gmm_worker_evidence import (
    parse_element_means,
    gmm_score_chunks_payload,
    normalize_gmm_frame_score_result,
)
from workflow_backends.execution import run_operation, structured_result_stdout
from workflow_backends.deepmd_common.progress import DeepMDTrainingProgressProbe
from workflow_backends.deepmd_common.train_options import (
    append_train_options,
    reject_automatic_sel_for_skip_neighbor_stat,
)
from workflow_backends.deepmd_common.type_maps import (
    resolve_foundation_type_map_operation,
)
from workflow_artifacts.lifecycle.gmm_io import (
    GMMReferenceFitResult,
    GMMScoreResult,
    validate_gmm_router,
)
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.progress import GenericWorkSnapshot, OperationProgressReporter
from workflow_runtime.lifecycle.progress.worker_observations import (
    DescriptorObservationProjector,
    ScoreObservationProjector,
    WorkerObservationCursor,
    WorkerObservationProgressProbe,
    WorkerObservationScope,
)
from workflow_backends.student.deepmd.descriptor_worker import (
    STRUCTURE_VECTOR_RESULT_PREFIX,
)


@dataclass(frozen=True, slots=True)
class FusedStudentGMMScreenResult:
    fit: GMMReferenceFitResult
    score: object
    phase_timings: Mapping[str, float]


def dpa4c_distributed_train_argv(
    input_json: Path,
    *,
    foundation_model: Path | None = None,
    train_options: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    arguments = ["dp", "--pt-expt", "train", str(input_json)]
    if foundation_model is not None:
        arguments.extend(("--finetune", str(foundation_model)))
    append_train_options(arguments, train_options)
    return tuple(arguments)


def dpa4c_graph_freeze_argv(output_name: str) -> tuple[str, ...]:
    if not isinstance(output_name, str) or not output_name or Path(output_name).name != output_name:
        raise ValueError("DPA4C graph freeze output name must be one safe filename")
    return (
        "dp",
        "--pt-expt",
        "freeze",
        "-o",
        output_name,
        "--lower-kind",
        "graph",
    )


@dataclass(frozen=True, slots=True)
class NativeDeepMDStudentExecutor:
    requires_native_dataset: ClassVar[bool] = True
    train_context: OperationContext
    compress_context: OperationContext | None = None
    gmm_screen_context: OperationContext | None = None
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
        include_element_vectors: bool = False,
    ):
        context = self.gmm_score_context or self.gmm_screen_context
        if context is None:
            raise ValueError(
                "DeepMD structural-vector readout requires a selected screen context"
            )
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime_path = work / "structure_vectors_runtime.json"
        type_map_path = work / "structure_vectors_type_map.json"
        compact_root = work / "compact_structure_vectors"
        runtime_path.write_text(
            json.dumps(context.runtime, sort_keys=True), encoding="utf-8"
        )
        type_map_path.write_text(json.dumps(list(type_map)), encoding="utf-8")
        try:
            result = run_operation(
                context,
                "python",
                (
                    "-m",
                    "workflow_backends.student.deepmd.descriptor_worker",
                    "--operation",
                    "structure_vectors",
                    "--model",
                    str(model),
                    "--runtime",
                    str(runtime_path),
                    "--type-map",
                    str(type_map_path),
                    "--observations",
                    str(input_path),
                    "--compact-result-root",
                    str(compact_root),
                ),
                cwd=work,
                diagnostic_name="deepmd_student_structure_vectors",
                structured_result_prefix=STRUCTURE_VECTOR_RESULT_PREFIX,
                structured_result_limit_bytes=64 * 1024,
            )
        finally:
            runtime_path.unlink(missing_ok=True)
            type_map_path.unlink(missing_ok=True)
        receipt = _parse_structure_vector_receipt(
            structured_result_stdout(result, prefix=STRUCTURE_VECTOR_RESULT_PREFIX)
        )
        from workflow_backends.compact_structure_vector_store import (
            read_compact_structure_vector_store,
        )

        return read_compact_structure_vector_store(
            receipt,
            expected_root=compact_root,
            expected_model_digest=_file_digest(Path(model).resolve()),
            include_elements=include_element_vectors,
        )

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
    ) -> Path:
        context = self.compress_context
        expected_role = f"student.deepmd.{cli}.compress"
        if context is None or context.role != expected_role:
            raise ValueError(
                f"DeepMD sampling compression requires operation context {expected_role}"
            )
        if cli not in {"tf", "pt"}:
            raise ValueError("DeepMD sampling compression CLI is invalid")
        if recipe_kind not in {"standard", "dpa4c"}:
            raise ValueError("DeepMD sampling compression recipe kind is unsupported")
        if recipe_kind == "dpa4c" and cli != "pt":
            raise ValueError("DPA4C sampling compression requires the PT ABI")
        if (float(step), int(extrapolate), int(frequency)) != (0.01, 5, -1):
            raise ValueError("DeepMD sampling compression values are not DICE-fixed")
        source = Path(source_model).resolve()
        output = Path(output_model).resolve()
        expected_suffix = ".pb" if cli == "tf" else ".pt2"
        if (
            source.suffix != expected_suffix
            or Path(source_model).is_symlink()
            or not source.is_file()
            or source.stat().st_size <= 0
        ):
            raise ValueError(
                f"DeepMD sampling compression source must be a regular {expected_suffix}"
            )
        if output.suffix != expected_suffix or output == source:
            raise ValueError(
                "DeepMD sampling compression output must preserve the source suffix"
            )
        training = None if training_script is None else Path(training_script).resolve()
        if recipe_kind == "dpa4c" and (
            training is None
            or Path(training_script).is_symlink()
            or not training.is_file()
            or training.stat().st_size <= 0
        ):
            raise ValueError(
                "DPA4C sampling compression requires a rendered training script"
            )
        arguments = [
            "--pt-expt" if recipe_kind == "dpa4c" else f"--{cli}",
            "compress",
            "-i",
            str(source),
            "-o",
            str(output),
            "-s",
            str(float(step)),
            "-e",
            str(int(extrapolate)),
            "-f",
            str(int(frequency)),
        ]
        if training is not None:
            arguments.extend(("--training-script", str(training)))
        run_operation(
            context,
            "dp",
            arguments,
            cwd=output.parent,
            diagnostic_name="deepmd_compress",
        )
        if (
            Path(output_model).is_symlink()
            or not output.is_file()
            or output.stat().st_size <= 0
        ):
            raise ValueError(
                "DeepMD sampling compression did not produce a regular model"
            )
        return output

    def train(
        self,
        input_json: Path,
        member_dir: Path,
        cli: str,
        *,
        recipe_kind: str = "standard",
        foundation_model: Path | None = None,
        train_options: Mapping[str, object] | None = None,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path:
        if cli not in {"tf", "pt"}:
            raise ValueError("DeepMD lifecycle student CLI is invalid")
        if recipe_kind not in {"standard", "dpa4c"}:
            raise ValueError(
                f"DeepMD lifecycle student recipe kind is unsupported: {recipe_kind}"
            )
        if recipe_kind == "dpa4c" and cli != "pt":
            raise ValueError("DPA4C lifecycle student execution requires the PT ABI")
        native_selector = "--pt-expt" if recipe_kind == "dpa4c" else f"--{cli}"
        train_arguments = [native_selector, "train", str(input_json)]
        if foundation_model is not None:
            foundation = Path(foundation_model).resolve()
            if not foundation.is_file() or foundation.stat().st_size <= 0:
                raise ValueError(
                    f"DeepMD lifecycle foundation model is missing or empty: {foundation}"
                )
            train_arguments.extend(("--finetune", str(foundation)))
        rendered = json.loads(Path(input_json).read_text(encoding="utf-8"))
        if not isinstance(rendered, dict):
            raise ValueError("DeepMD lifecycle student input must be an object")
        reject_automatic_sel_for_skip_neighbor_stat(rendered, train_options)
        if recipe_kind == "dpa4c":
            train_arguments = list(
                dpa4c_distributed_train_argv(
                    input_json,
                    foundation_model=foundation_model,
                    train_options=train_options,
                )[1:]
            )
        else:
            append_train_options(train_arguments, train_options)
        probe = None
        if progress_reporter is not None:
            try:
                probe = DeepMDTrainingProgressProbe.from_rendered_input(
                    rendered,
                    member_dir=Path(member_dir),
                    member_id=progress_reporter.operation_id,
                    member_index=member_index,
                    total_members=total_members,
                    cli=cli,
                )
            except Exception:
                probe = None
        frozen = Path(member_dir) / ("student_native.pb" if cli == "tf" else "student_native.pt2")
        try:
            run_operation(
                self.train_context,
                "dp",
                train_arguments,
                cwd=member_dir,
                diagnostic_name="deepmd_train",
                progress_reporter=progress_reporter,
                progress_probe=probe,
                probe_interval_seconds=float(
                    self.train_context.runtime.get("progress_probe_interval_seconds", 1.0)
                ),
                close_progress_reporter=False,
            )
            if progress_reporter is not None:
                progress_reporter.report(GenericWorkSnapshot(activity="freezing model"))
            freeze_arguments = (
                list(dpa4c_graph_freeze_argv(frozen.name)[1:])
                if recipe_kind == "dpa4c"
                else [native_selector, "freeze", "-o", frozen.name]
            )
            run_operation(
                self.train_context,
                "dp",
                freeze_arguments,
                cwd=member_dir,
                diagnostic_name="deepmd_freeze",
                progress_reporter=progress_reporter,
            )
        except BaseException:
            if progress_reporter is not None and not progress_reporter.closed:
                progress_reporter.close()
            raise
        if not frozen.is_file() or frozen.stat().st_size <= 0:
            raise ValueError("DeepMD lifecycle student native train did not produce a frozen model")
        return frozen

    def gmm_reference(
        self,
        *,
        model: Path,
        reference_path: Path,
        router_dir: Path,
        work_dir: Path,
        type_map: tuple[str, ...],
        progress_reporter: OperationProgressReporter | None = None,
    ) -> GMMReferenceFitResult:
        context = self.gmm_reference_context
        if context is None:
            raise ValueError("DeepMD-TF gmm_reference execution requires its selected operation context")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime = work / "gmm_reference_runtime.json"
        type_map_path = work / "gmm_reference_type_map.json"
        runtime.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        type_map_path.write_text(json.dumps(list(type_map)), encoding="utf-8")
        try:
            with WorkerObservationScope(work) as observations:
                arguments = [
                    "-m", "workflow_backends.student.deepmd.descriptor_worker",
                    "--operation", "gmm_reference",
                    "--model", str(model),
                    "--reference", str(reference_path),
                    "--router", str(router_dir),
                    "--runtime", str(runtime),
                    "--type-map", str(type_map_path),
                    "--observation-channel", str(observations.path),
                ]
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    DescriptorObservationProjector(label="student reference"),
                )
                run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=work,
                    diagnostic_name="deepmd_student_gmm_reference",
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                )
        finally:
            runtime.unlink(missing_ok=True)
            type_map_path.unlink(missing_ok=True)
        manifest = validate_gmm_router(router_dir)
        dimensions = {
            str(element): int(metadata["pca_dimensions"])
            for element, metadata in dict(manifest.get("elements", {})).items()
        }
        return GMMReferenceFitResult(
            router_dir=Path(router_dir),
            policy_id=str(manifest["policy_name"]),
            policy_revision=int(manifest["policy_revision"]),
            element_pca_dimensions=dimensions,
            descriptor_schema=dict(manifest["descriptor_schema"]),
        )

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
    ) -> GMMScoreResult:
        context = self.gmm_score_context
        if context is None:
            raise ValueError("DeepMD-TF gmm_score execution requires its selected operation context")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime = work / "gmm_score_runtime.json"
        identity_filter = work / "gmm_score_identity_filter.json"
        runtime.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        identity_filter.write_text(json.dumps(list(ordered_identities)), encoding="utf-8")
        arguments = [
            "-m", "workflow_backends.student.deepmd.descriptor_worker",
            "--operation", "gmm_score",
            "--model", str(model),
            "--reference", str(reference_path),
            "--router", str(router_dir),
            "--runtime", str(runtime),
            "--observations", str(observation_path),
            "--identity-filter", str(identity_filter),
        ]
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        try:
            with WorkerObservationScope(work) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    ScoreObservationProjector(
                        progress_systems or {}, total=len(ordered_identities), role="student"
                    ),
                )
                result = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=work,
                    diagnostic_name="deepmd_student_gmm_score",
                    structured_result_prefix="DICE_GMM_SCORE_RESULT=",
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                )
        finally:
            runtime.unlink(missing_ok=True)
            identity_filter.unlink(missing_ok=True)
        return _parse_score_result(
            structured_result_stdout(result, prefix="DICE_GMM_SCORE_RESULT="),
            ordered_identities=ordered_identities,
            emit_frame_means=emit_frame_means,
        )

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
    ) -> GMMScoreResult:
        context = self.gmm_score_context
        if context is None:
            raise ValueError("DeepMD-TF gmm_score execution requires its selected operation context")
        if not observation_paths or len(observation_paths) != len(ordered_identities_by_chunk):
            raise ValueError("DeepMD-TF chunked gmm_score requires aligned non-empty input chunks")
        ordered_identities = tuple(
            identity for chunk in ordered_identities_by_chunk for identity in chunk
        )
        if (
            not ordered_identities
            or any(not identity for identity in ordered_identities)
            or len(set(ordered_identities)) != len(ordered_identities)
            or any(not chunk for chunk in ordered_identities_by_chunk)
        ):
            raise ValueError("DeepMD-TF chunked gmm_score requires unique canonical frame identities")

        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime = work / "gmm_score_runtime.json"
        chunk_manifest = work / "gmm_score_chunk_manifest.json"
        runtime.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        chunk_manifest.write_text(
            json.dumps(
                gmm_score_chunks_payload(
                    frame_count=len(ordered_identities),
                    chunks=[
                        {
                            "path": str(Path(path).resolve()),
                            "frame_ids": list(identities),
                        }
                        for path, identities in zip(observation_paths, ordered_identities_by_chunk)
                    ],
                ),
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        arguments = [
            "-m", "workflow_backends.student.deepmd.descriptor_worker",
            "--operation", "gmm_score",
            "--model", str(model),
            "--reference", str(reference_path),
            "--router", str(router_dir),
            "--runtime", str(runtime),
            "--chunk-manifest", str(chunk_manifest.resolve()),
        ]
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        try:
            with WorkerObservationScope(work) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    ScoreObservationProjector(
                        progress_systems or {}, total=len(ordered_identities), role="student"
                    ),
                )
                result = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=work,
                    diagnostic_name="deepmd_student_gmm_score",
                    structured_result_prefix="DICE_GMM_SCORE_RESULT=",
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                )
        finally:
            runtime.unlink(missing_ok=True)
            chunk_manifest.unlink(missing_ok=True)
        return _parse_score_result(
            structured_result_stdout(result, prefix="DICE_GMM_SCORE_RESULT="),
            ordered_identities=ordered_identities,
            emit_frame_means=emit_frame_means,
        )

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
        slot_count: int = 2,
        progress_reporter: OperationProgressReporter | None = None,
        progress_systems: dict[str, str] | None = None,
    ) -> GMMScoreResult:
        from workflow_backends.rolling_extxyz import RollingExtxyzChannel

        context = self.gmm_score_context
        if context is None:
            raise ValueError(
                "DeepMD rolling gmm_score requires its selected operation context"
            )
        identities = tuple(str(item) for item in ordered_identities)
        if not identities or any(not item for item in identities) or len(
            set(identities)
        ) != len(identities):
            raise ValueError(
                "DeepMD rolling gmm_score requires unique canonical identities"
            )
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime = work / "gmm_score_runtime.json"
        identity_filter = work / "gmm_score_identity_filter.json"
        channel = RollingExtxyzChannel(
            work / "rolling_extxyz",
            ordered_identities=identities,
            slot_count=slot_count,
        )
        runtime.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        identity_filter.write_text(json.dumps(list(identities)), encoding="utf-8")
        arguments = [
            "-m",
            "workflow_backends.student.deepmd.descriptor_worker",
            "--operation",
            "gmm_score",
            "--model",
            str(model),
            "--reference",
            str(reference_path),
            "--router",
            str(router_dir),
            "--runtime",
            str(runtime),
            "--identity-filter",
            str(identity_filter),
            "--rolling-root",
            str(channel.root),
            "--rolling-slots",
            str(channel.slot_count),
        ]
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        try:
            with WorkerObservationScope(work) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    ScoreObservationProjector(
                        progress_systems or {}, total=len(identities), role="student"
                    ),
                )
                result = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=work,
                    diagnostic_name="deepmd_student_gmm_score",
                    structured_result_prefix="DICE_GMM_SCORE_RESULT=",
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                    input_producer=lambda process: channel.produce(
                        frame_chunks, consumer_alive=lambda: process.poll() is None
                    ),
                )
        finally:
            runtime.unlink(missing_ok=True)
            identity_filter.unlink(missing_ok=True)
            channel.close()
        return _parse_score_result(
            structured_result_stdout(result, prefix="DICE_GMM_SCORE_RESULT="),
            ordered_identities=identities,
            emit_frame_means=emit_frame_means,
        )

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
    ):
        from workflow_backends.geometry_stream import GeometryStreamServer

        if inference_pool_enabled(self.gmm_score_context):
            return shard_stream_score(
                self, "gmm_score_stream", context=self.gmm_score_context,
                identities=ordered_identities, frame_chunks=frame_chunks,
                identity_key="ordered_identities", model=model,
                reference_path=reference_path, router_dir=router_dir, work_dir=work_dir,
                emit_frame_means=emit_frame_means, progress_systems=progress_systems,
                progress_reporter=progress_reporter,
            )
        context = self.gmm_score_context
        if context is None:
            raise ValueError(
                "DeepMD geometry gmm_score requires its selected operation context"
            )
        identities = tuple(str(item) for item in ordered_identities)
        if not identities or any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("DeepMD geometry gmm_score requires unique canonical identities")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime = work / "gmm_score_runtime.json"
        identity_filter = work / "gmm_score_identity_filter.json"
        compact_root = work / "compact_score_result"
        byte_budget = 64 * 1024 * 1024
        atom_budget = 1_000_000
        channel = GeometryStreamServer(
            work / "geometry.sock",
            ordered_identities=identities,
            byte_budget=byte_budget,
            atom_budget=atom_budget,
        )
        runtime.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        identity_filter.write_text(json.dumps(list(identities)), encoding="utf-8")
        arguments = [
            "-m", "workflow_backends.student.deepmd.descriptor_worker",
            "--operation", "gmm_score",
            "--model", str(model),
            "--reference", str(reference_path),
            "--router", str(router_dir),
            "--runtime", str(runtime),
            "--identity-filter", str(identity_filter),
            "--geometry-socket", str(channel.socket_path),
            "--geometry-byte-budget", str(byte_budget),
            "--geometry-atom-budget", str(atom_budget),
            "--compact-result-root", str(compact_root),
        ]
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        try:
            with WorkerObservationScope(work) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    ScoreObservationProjector(
                        progress_systems or {}, total=len(identities), role="student"
                    ),
                )
                result = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=work,
                    diagnostic_name="deepmd_student_gmm_score",
                    structured_result_prefix="DICE_GMM_SCORE_RESULT=",
                    structured_result_limit_bytes=64 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                    input_producer=lambda process: channel.produce(
                        frame_chunks, consumer_alive=lambda: process.poll() is None
                    ),
                )
        finally:
            runtime.unlink(missing_ok=True)
            identity_filter.unlink(missing_ok=True)
            channel.close()
        return _parse_compact_score_result(
            structured_result_stdout(result, prefix="DICE_GMM_SCORE_RESULT="),
            ordered_identities=identities,
            model_digest=_file_digest(Path(model).resolve()),
            reference_digest=_file_digest(Path(reference_path).resolve()),
        )

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
    ) -> FusedStudentGMMScreenResult:
        from workflow_backends.geometry_stream import GeometryStreamServer

        context = self.gmm_screen_context
        if context is None or context.role != "student.deepmd.pt.gmm_screen":
            raise ValueError("DPA4C fused screen requires its selected gmm_screen context")
        if inference_pool_enabled(context):
            from time import perf_counter
            from workflow_backends.compact_score_store import write_compact_score_store
            from workflow_backends.execution import ProcessGroupRegistry, operation_execution_scope
            from workflow_backends.common.sampling_pool import sampling_visible_devices

            devices = sampling_visible_devices(context.allocation.gpus_per_node)
            worker = replace(
                self, gmm_reference_context=single_gpu_context(context),
                gmm_score_context=context,
            )
            started = perf_counter()
            registry = ProcessGroupRegistry()
            try:
                with operation_execution_scope(environment_overrides={"CUDA_VISIBLE_DEVICES": devices[0]}, process_registry=registry):
                    fit = worker.gmm_reference(
                        model=model, reference_path=reference_path, router_dir=router_dir,
                        work_dir=Path(work_dir) / "reference", type_map=type_map,
                        progress_reporter=progress_reporter,
                    )
            finally:
                registry.cancel_all()
            reference_seconds = perf_counter() - started
            started = perf_counter()
            scored = worker.gmm_score_stream(
                model=model, reference_path=reference_path, frame_chunks=frame_chunks,
                ordered_identities=ordered_identities, router_dir=router_dir,
                work_dir=Path(work_dir) / "score", emit_frame_means=emit_frame_means,
                progress_systems=progress_systems, progress_reporter=progress_reporter,
            )
            model_digest, reference_digest = _file_digest(Path(model)), _file_digest(Path(reference_path))
            receipt = write_compact_score_store(
                Path(work_dir) / "compact_score_result", scored,
                model_digest=model_digest, reference_digest=reference_digest,
            )
            score = _compact_view_from_receipt(
                receipt, ordered_identities=ordered_identities,
                model_digest=model_digest, reference_digest=reference_digest,
            )
            return FusedStudentGMMScreenResult(
                fit=fit, score=score, phase_timings={
                    # Worker model loads are included in the two wall-clock phases.
                    "model_load_seconds": 0.0, "reference_seconds": reference_seconds,
                    "score_seconds": perf_counter() - started,
                },
            )
        identities = tuple(str(item) for item in ordered_identities)
        if not identities or any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("DPA4C fused screen requires unique canonical identities")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime = work / "gmm_screen_runtime.json"
        identity_filter = work / "gmm_screen_identity_filter.json"
        type_map_path = work / "gmm_screen_type_map.json"
        compact_root = work / "compact_score_result"
        byte_budget = 64 * 1024 * 1024
        atom_budget = 1_000_000
        channel = GeometryStreamServer(
            work / "geometry.sock",
            ordered_identities=identities,
            byte_budget=byte_budget,
            atom_budget=atom_budget,
        )
        runtime.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        identity_filter.write_text(json.dumps(list(identities)), encoding="utf-8")
        type_map_path.write_text(json.dumps(list(type_map)), encoding="utf-8")
        arguments = [
            "-m", "workflow_backends.student.deepmd.descriptor_worker",
            "--operation", "gmm_screen",
            "--model", str(model),
            "--reference", str(reference_path),
            "--router", str(router_dir),
            "--runtime", str(runtime),
            "--type-map", str(type_map_path),
            "--identity-filter", str(identity_filter),
            "--geometry-socket", str(channel.socket_path),
            "--geometry-byte-budget", str(byte_budget),
            "--geometry-atom-budget", str(atom_budget),
            "--compact-result-root", str(compact_root),
        ]
        if emit_frame_means:
            arguments.append("--emit-frame-means")
        try:
            with WorkerObservationScope(work) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(
                    WorkerObservationCursor(observations.path),
                    ScoreObservationProjector(
                        progress_systems or {}, total=len(identities), role="student"
                    ),
                )
                result = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=work,
                    diagnostic_name="deepmd_student_gmm_screen",
                    structured_result_prefix="DICE_GMM_SCORE_RESULT=",
                    structured_result_limit_bytes=64 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                    input_producer=lambda process: channel.produce(
                        frame_chunks, consumer_alive=lambda: process.poll() is None
                    ),
                )
        finally:
            for path in (runtime, identity_filter, type_map_path):
                path.unlink(missing_ok=True)
            channel.close()
        stdout = structured_result_stdout(result, prefix="DICE_GMM_SCORE_RESULT=")
        receipt = _parse_receipt_json(stdout)
        raw_fit = receipt.get("fit")
        raw_timings = receipt.get("phase_timings")
        if not isinstance(raw_fit, dict) or not isinstance(raw_timings, dict):
            raise ValueError("DPA4C fused worker receipt lacks phase evidence")
        fit = GMMReferenceFitResult(
            router_dir=Path(router_dir),
            policy_id=str(raw_fit.get("policy_id", "")),
            policy_revision=int(raw_fit.get("policy_revision", -1)),
            element_pca_dimensions={
                str(key): int(value)
                for key, value in dict(raw_fit.get("element_pca_dimensions", {})).items()
            },
            descriptor_schema=dict(raw_fit.get("descriptor_schema", {})),
        )
        score = _compact_view_from_receipt(
            receipt,
            ordered_identities=identities,
            model_digest=_file_digest(Path(model).resolve()),
            reference_digest=_file_digest(Path(reference_path).resolve()),
        )
        timings = {str(key): float(value) for key, value in raw_timings.items()}
        if set(timings) != {
            "model_load_seconds",
            "reference_seconds",
            "score_seconds",
        } or any(
            value < 0 for value in timings.values()
        ):
            raise ValueError("DPA4C fused worker phase timings are invalid")
        return FusedStudentGMMScreenResult(fit=fit, score=score, phase_timings=timings)


def _parse_compact_score_result(
    stdout: str,
    *,
    ordered_identities: tuple[str, ...],
    model_digest: str,
    reference_digest: str,
):
    receipt = _parse_receipt_json(stdout)
    return _compact_view_from_receipt(
        receipt,
        ordered_identities=ordered_identities,
        model_digest=model_digest,
        reference_digest=reference_digest,
    )


def _parse_structure_vector_receipt(stdout: str) -> dict[str, object]:
    prefix = STRUCTURE_VECTOR_RESULT_PREFIX
    lines = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(
            "DeepMD compact structure-vector worker did not return exactly one receipt"
        )
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "DeepMD compact structure-vector worker returned malformed receipt JSON"
        ) from exc
    if not isinstance(receipt, dict):
        raise ValueError(
            "DeepMD compact structure-vector worker receipt must be an object"
        )
    return receipt


def _parse_receipt_json(stdout: str) -> dict[str, object]:
    prefix = "DICE_GMM_SCORE_RESULT="
    lines = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("DeepMD compact GMM worker did not return exactly one receipt")
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("DeepMD compact GMM worker returned malformed receipt JSON") from exc
    if not isinstance(receipt, dict):
        raise ValueError("DeepMD compact GMM worker receipt must be an object")
    return receipt


def _compact_view_from_receipt(
    receipt: Mapping[str, object],
    *,
    ordered_identities: tuple[str, ...],
    model_digest: str,
    reference_digest: str,
):
    from workflow_core.lifecycle.compact_scores import CompactScoreView

    return CompactScoreView(
        receipt,
        ordered_identities=ordered_identities,
        expected_model_digest=model_digest,
        expected_reference_digest=reference_digest,
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_score_result(
    stdout: str,
    *,
    ordered_identities: tuple[str, ...],
    emit_frame_means: bool,
) -> GMMScoreResult:
    prefix = "DICE_GMM_SCORE_RESULT="
    lines = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("DeepMD-TF gmm_score worker did not return exactly one frame result")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("DeepMD-TF gmm_score worker returned malformed frame JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("DeepMD-TF gmm_score worker returned an unsupported frame result schema")
    try:
        payload = normalize_gmm_frame_score_result(payload)
    except ValueError as exc:
        raise ValueError(
            "DeepMD-TF gmm_score worker returned an unsupported frame result schema"
        ) from exc
    if set(payload) - {"element_means"} != {
        "schema",
        "schema_revision",
        "frame_ids",
        "frame_p",
        "frame_statistics",
        "frame_means",
        "descriptor_schema",
    }:
        raise ValueError("DeepMD-TF gmm_score worker returned an unsupported frame result schema")
    descriptor_schema = validated_readout_input_schema(payload["descriptor_schema"])
    frame_ids = tuple(str(item) for item in payload["frame_ids"])
    if frame_ids != tuple(ordered_identities):
        raise ValueError("DeepMD-TF gmm_score worker changed the ordered identity filter")
    frame_p = np.asarray(payload["frame_p"], dtype=np.float64)
    statistics = np.asarray(payload["frame_statistics"], dtype=np.float64)
    if (
        frame_p.shape != (len(frame_ids),)
        or statistics.shape != (len(frame_ids),)
        or not np.all(np.isfinite(frame_p))
        or not np.all(np.isfinite(statistics))
        or np.any(frame_p <= 0.0)
        or np.any(frame_p > 1.0)
    ):
        raise ValueError("DeepMD-TF gmm_score worker returned invalid finite frame values")
    raw_means = payload["frame_means"]
    if emit_frame_means:
        means = np.asarray(raw_means, dtype=np.float32)
        if means.ndim != 2 or means.shape[0] != len(frame_ids) or not np.all(np.isfinite(means)):
            raise ValueError("DeepMD-TF gmm_score worker returned invalid readout-input frame means")
        if means.shape[1] != int(descriptor_schema["width"]):
            raise ValueError("DeepMD-TF gmm_score worker frame means do not match descriptor schema")
        means = np.ascontiguousarray(means)
    else:
        if raw_means is not None:
            raise ValueError("DeepMD-TF random gmm_score unexpectedly returned readout-input frame means")
        means = None
    return GMMScoreResult(
        frame_ids=np.asarray(frame_ids, dtype=np.str_),
        frame_p=frame_p,
        frame_statistics=statistics,
        frame_means=means,
        descriptor_schema=descriptor_schema,
        element_means=parse_element_means(payload.get("element_means"), frame_ids=frame_ids, width=int(descriptor_schema["width"])),
    )
