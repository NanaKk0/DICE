from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from workflow_backends.common.inference_pool import inference_pool_enabled, shard_stream_score
from workflow_backends.descriptor_stream import validated_readout_input_schema
from workflow_artifacts.gmm_worker_evidence import normalize_gmm_frame_score_result, parse_element_means
from workflow_backends.student.nep.descriptor_worker import (
    GMM_RESULT_PREFIX,
    RESULT_PREFIX,
)
from workflow_backends.student.nep.progress import NepTrainingProgressProbe
from workflow_backends.execution import run_operation, structured_result_stdout
from workflow_artifacts.lifecycle.gmm_io import (
    GMMReferenceFitResult,
    GMMScoreResult,
    validate_gmm_router,
)
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.progress import OperationProgressReporter
from workflow_runtime.lifecycle.progress.worker_observations import (
    DescriptorObservationProjector,
    ScoreObservationProjector,
    WorkerObservationCursor,
    WorkerObservationProgressProbe,
    WorkerObservationScope,
)


@dataclass(frozen=True, slots=True)
class NativeNepExecutor:
    requires_native_dataset: ClassVar[bool] = False
    train_context: OperationContext
    gmm_reference_context: OperationContext | None = None
    gmm_score_context: OperationContext | None = None

    def train(
        self,
        recipe: Path,
        member: Path,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> tuple[Path, Path]:
        root = Path(member)
        if not Path(recipe).is_file():
            raise ValueError(f"NEP lifecycle native train recipe is missing: {recipe}")
        probe = None
        if progress_reporter is not None:
            try:
                probe = NepTrainingProgressProbe.from_nep_input(
                    Path(recipe),
                    member_dir=root,
                    member_id=progress_reporter.operation_id,
                    member_index=member_index,
                    total_members=total_members,
                )
            except Exception:
                probe = None
        run_operation(
            self.train_context,
            "nep",
            (),
            cwd=root,
            diagnostic_name="nep_train",
            progress_reporter=progress_reporter,
            progress_probe=probe,
            probe_interval_seconds=float(
                self.train_context.runtime.get("progress_probe_interval_seconds", 1.0)
            ),
        )
        model = root / "nep.txt"
        restart = root / "nep.restart"
        if not model.is_file() or model.stat().st_size <= 0:
            raise ValueError("NEP lifecycle native train did not produce nep.txt")
        if not restart.is_file() or restart.stat().st_size <= 0:
            raise ValueError("NEP lifecycle native train did not produce nep.restart")
        return model, restart

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
            raise ValueError("NEP gmm_reference execution requires its selected operation context")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime_path = work / "gmm_reference_runtime.json"
        type_map_path = work / "gmm_reference_type_map.json"
        runtime_path.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        type_map_path.write_text(json.dumps(list(type_map)), encoding="utf-8")
        try:
            with WorkerObservationScope(work) as observations:
                arguments = [
                    "-m", "workflow_backends.student.nep.descriptor_worker",
                    "--operation", "gmm_reference",
                    "--model", str(model),
                    "--reference", str(reference_path),
                    "--router", str(router_dir),
                    "--runtime", str(runtime_path),
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
                    diagnostic_name="nep_student_gmm_reference",
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                )
        finally:
            runtime_path.unlink(missing_ok=True)
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

    def structure_vectors(
        self,
        *,
        model: Path,
        input_path: Path,
        work_dir: Path,
        include_element_vectors: bool = False,
    ):
        context = self.gmm_score_context
        if context is None:
            raise ValueError(
                "NEP structural-vector readout requires its selected score context"
            )
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime_path = work / "structure_vectors_runtime.json"
        runtime_path.write_text(
            json.dumps(context.runtime, sort_keys=True), encoding="utf-8"
        )
        try:
            result = run_operation(
                context,
                "python",
                (
                    "-m",
                    "workflow_backends.student.nep.descriptor_worker",
                    "--operation",
                    "structure_vectors",
                    "--model",
                    str(model),
                    "--runtime",
                    str(runtime_path),
                    "--observations",
                    str(input_path),
                ),
                cwd=work,
                diagnostic_name="nep_student_structure_vectors",
                structured_result_prefix=RESULT_PREFIX,
                structured_result_limit_bytes=64 * 1024 * 1024,
            )
        finally:
            runtime_path.unlink(missing_ok=True)
        payload = json.loads(structured_result_stdout(result, prefix=RESULT_PREFIX))
        if not isinstance(payload, dict) or set(payload) != {
            "descriptor_schema",
            "frame_ids",
            "frame_means",
            "element_means",
        }:
            raise ValueError("NEP structural-vector worker returned invalid evidence")
        schema = validated_readout_input_schema(payload["descriptor_schema"])
        identities = tuple(str(item) for item in payload["frame_ids"])
        rows = np.asarray(payload["frame_means"], dtype=np.float32)
        if rows.shape != (len(identities), int(schema["width"])) or not np.isfinite(rows).all():
            raise ValueError("NEP structural-vector worker returned invalid vectors")
        if not include_element_vectors:
            return dict(schema), identities, rows
        raw_elements = payload["element_means"]
        if not isinstance(raw_elements, dict):
            raise ValueError("NEP structural-vector worker returned invalid elements")
        elements: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for element, raw_values in raw_elements.items():
            if not isinstance(raw_values, dict) or set(raw_values) != {
                "frame_ids",
                "vectors",
            }:
                raise ValueError("NEP structural-vector worker returned invalid elements")
            element_ids = tuple(str(item) for item in raw_values["frame_ids"])
            element_rows = np.asarray(raw_values["vectors"], dtype=np.float32)
            if (
                any(identity not in identities for identity in element_ids)
                or len(set(element_ids)) != len(element_ids)
                or element_rows.shape != (len(element_ids), int(schema["width"]))
                or not np.isfinite(element_rows).all()
            ):
                raise ValueError("NEP structural-vector worker returned invalid elements")
            elements[str(element)] = (element_ids, element_rows)
        return dict(schema), identities, rows, elements

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
                "NEP geometry gmm_score requires its selected operation context"
            )
        identities = tuple(str(item) for item in ordered_identities)
        if not identities or any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("NEP geometry gmm_score requires unique canonical identities")
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
            "-m", "workflow_backends.student.nep.descriptor_worker",
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
                    diagnostic_name="nep_student_gmm_score",
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
        from workflow_backends.compact_score_store import compact_score_view_from_stdout
        from workflow_artifacts.lifecycle.sampling_models import file_sha256
        return compact_score_view_from_stdout(
            structured_result_stdout(result, prefix="DICE_GMM_SCORE_RESULT="),
            ordered_identities=identities,
            model_digest=file_sha256(Path(model).resolve()),
            reference_digest=file_sha256(Path(reference_path).resolve()),
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
            raise ValueError("NEP gmm_score execution requires its selected operation context")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        runtime_path = work / "gmm_score_runtime.json"
        identity_path = work / "gmm_score_identity_filter.json"
        runtime_path.write_text(json.dumps(context.runtime, sort_keys=True), encoding="utf-8")
        identity_path.write_text(json.dumps(list(ordered_identities)), encoding="utf-8")
        arguments = [
            "-m", "workflow_backends.student.nep.descriptor_worker",
            "--operation", "gmm_score",
            "--model", str(model),
            "--reference", str(reference_path),
            "--router", str(router_dir),
            "--runtime", str(runtime_path),
            "--observations", str(observation_path),
            "--identity-filter", str(identity_path),
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
                    diagnostic_name="nep_student_gmm_score",
                    structured_result_prefix=GMM_RESULT_PREFIX,
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                    close_progress_reporter=False,
                )
        finally:
            runtime_path.unlink(missing_ok=True)
            identity_path.unlink(missing_ok=True)
        return _parse_score_result(
            structured_result_stdout(result, prefix=GMM_RESULT_PREFIX),
            ordered_identities=ordered_identities,
            emit_frame_means=emit_frame_means,
        )


def _parse_score_result(
    stdout: str,
    *,
    ordered_identities: tuple[str, ...],
    emit_frame_means: bool,
) -> GMMScoreResult:
    lines = [
        line[len(GMM_RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(GMM_RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise ValueError("NEP gmm_score worker did not return exactly one frame result")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("NEP gmm_score worker returned malformed frame JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("NEP gmm_score worker returned an unsupported frame result schema")
    try:
        payload = normalize_gmm_frame_score_result(payload)
    except ValueError as exc:
        raise ValueError("NEP gmm_score worker returned an unsupported frame result schema") from exc
    if set(payload) - {"element_means"} != {
        "schema",
        "schema_revision",
        "frame_ids",
        "frame_p",
        "frame_statistics",
        "frame_means",
        "descriptor_schema",
    }:
        raise ValueError("NEP gmm_score worker returned an unsupported frame result schema")
    descriptor_schema = validated_readout_input_schema(payload["descriptor_schema"])
    frame_ids = tuple(str(item) for item in payload["frame_ids"])
    if frame_ids != tuple(ordered_identities):
        raise ValueError("NEP gmm_score worker changed the ordered identity filter")
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
        raise ValueError("NEP gmm_score worker returned invalid finite frame values")
    raw_means = payload["frame_means"]
    if emit_frame_means:
        means = np.asarray(raw_means, dtype=np.float32)
        if (
            means.shape != (len(frame_ids), int(descriptor_schema["width"]))
            or not np.all(np.isfinite(means))
        ):
            raise ValueError("NEP gmm_score worker returned invalid readout-input frame means")
        means = np.ascontiguousarray(means)
    else:
        if raw_means is not None:
            raise ValueError("NEP random gmm_score unexpectedly returned readout-input frame means")
        means = None
    return GMMScoreResult(
        frame_ids=np.asarray(frame_ids, dtype=np.str_),
        frame_p=frame_p,
        frame_statistics=statistics,
        frame_means=means,
        descriptor_schema=descriptor_schema,
        element_means=parse_element_means(payload.get("element_means"), frame_ids=frame_ids, width=int(descriptor_schema["width"])),
    )


__all__ = ["NativeNepExecutor"]
