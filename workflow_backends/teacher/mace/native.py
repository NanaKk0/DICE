from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path
import json
import os
import hashlib

import yaml
import numpy as np

from workflow_backends.common.inference_pool import (
    inference_pool_enabled, shard_labels, shard_stream_score, shard_file_score,
    single_gpu_context,
)

from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from workflow_artifacts.gmm_worker_evidence import (
    gmm_score_chunks_payload,
    normalize_gmm_frame_score_result,
)
from workflow_backends.descriptor_stream import validated_readout_input_schema
from workflow_backends.execution import run_operation, structured_result_stdout
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.execution.distributed_receipts import (
    load_and_validate_single_node_rank_receipts,
)
from workflow_runtime.execution.task_scheduler import DistributedAssignment
from workflow_runtime.execution.torchrun import build_single_node_torchrun_arguments
from workflow_runtime.lifecycle.progress.operation import OperationProgressReporter
from workflow_runtime.lifecycle.progress.worker_observations import (
    LabelObservationProjector,
    DescriptorObservationProjector,
    ScoreObservationProjector,
    WorkerObservationCursor,
    WorkerObservationProgressProbe,
    WorkerObservationScope,
)
from workflow_backends.teacher.mace.progress import MaceTrainingProgressProbe
from workflow_artifacts.lifecycle.gmm_io import (
    GMMReferenceFitResult,
    GMMScoreResult,
    POLICY_ID,
    POLICY_REVISION,
    validate_gmm_router,
)


_EXTXYZ_GEOMETRY_QUANTIZATION_ATOL = 5.0e-9 + 1.0e-12


def _prepare_primary_rank_log_views(receipt_root: Path, work_dir: Path) -> None:
    """Expose the live rank-zero streams through the canonical MACE log names."""
    for channel in ("stdout", "stderr"):
        rank_log = receipt_root / f"rank-0.{channel}.log"
        primary_log = work_dir / f"mace_train.{channel}.log"
        rank_log.touch(exist_ok=False)
        os.link(rank_log, primary_log)


def _restore_extxyz_transport_geometry(source: Atoms, labeled: Atoms) -> Atoms:
    """Restore canonical geometry after a label-only extxyz round trip."""
    if not np.array_equal(labeled.numbers, source.numbers):
        raise ValueError("MACE lifecycle label worker changed atomic numbers")
    if not np.array_equal(labeled.pbc, source.pbc):
        raise ValueError("MACE lifecycle label worker changed periodic boundary flags")

    position_delta = float(np.max(np.abs(labeled.positions - source.positions), initial=0.0))
    cell_delta = float(np.max(np.abs(labeled.cell.array - source.cell.array), initial=0.0))
    if position_delta > _EXTXYZ_GEOMETRY_QUANTIZATION_ATOL or cell_delta > _EXTXYZ_GEOMETRY_QUANTIZATION_ATOL:
        raise ValueError(
            "MACE lifecycle label worker changed canonical geometry beyond extxyz transport tolerance: "
            f"positions max_abs_delta={position_delta:.17g}, cell max_abs_delta={cell_delta:.17g}"
        )

    results = getattr(getattr(labeled, "calc", None), "results", None)
    preserved = (
        {key: deepcopy(value) for key, value in results.items() if key in all_properties}
        if isinstance(results, dict)
        else {}
    )
    restored = labeled.copy()
    restored.set_cell(source.cell.array, scale_atoms=False)
    restored.set_positions(source.positions)
    restored.set_pbc(source.pbc)
    if preserved:
        restored.calc = SinglePointCalculator(restored, **preserved)
    return restored


@dataclass(frozen=True, slots=True)
class NativeMaceExecutor:
    train_context: OperationContext
    label_context: OperationContext | None = None
    gmm_reference_context: OperationContext | None = None
    gmm_score_context: OperationContext | None = None

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
            raise ValueError("MACE teacher structural readout requires score context")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        type_map_path = work / "teacher_structure_type_map.json"
        compact_root = work / "compact_teacher_structure_vectors"
        type_map_path.write_text(json.dumps(list(type_map)), encoding="utf-8")
        from workflow_backends.teacher.structure_vector_worker import RESULT_PREFIX

        arguments = [
            "-m",
            "workflow_backends.teacher.structure_vector_worker",
            "--backend",
            "mace",
            "--model",
            str(model),
            "--input",
            str(input_path),
            "--type-map",
            str(type_map_path),
            "--output",
            str(compact_root),
            "--workers",
            str(int(context.runtime.get("mace_workers", 1))),
            "--batch-size",
            str(int(context.runtime.get("mace_batch_size", 1))),
            "--torch-threads",
            str(int(context.runtime.get("mace_threads", 1))),
        ]
        result = run_operation(
            context,
            "python",
            arguments,
            cwd=work,
            diagnostic_name="mace_teacher_structure_vectors",
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
            raise ValueError("MACE teacher structural readout returned no receipt")
        digest = hashlib.sha256()
        with Path(model).resolve().open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        from workflow_backends.compact_structure_vector_store import (
            read_compact_structure_vector_store,
        )

        return read_compact_structure_vector_store(
            json.loads(lines[0]),
            expected_root=compact_root,
            expected_model_digest=digest.hexdigest(),
            include_elements=True,
        )

    def train(
        self,
        rendered_yaml: Path,
        work_dir: Path,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path:
        rendered = yaml.safe_load(rendered_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(rendered, dict):
            raise ValueError("MACE native train requires a rendered YAML object")
        name = rendered.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MACE native train requires an explicit non-empty name")
        raw_model_dir = rendered.get("model_dir")
        if not isinstance(raw_model_dir, str) or not raw_model_dir.strip():
            raise ValueError("MACE native train requires an explicit model_dir")
        model_dir = Path(raw_model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(work_dir) / model_dir
        swa = rendered.get("swa", False)
        if not isinstance(swa, bool):
            raise ValueError("MACE native train recipe swa must be boolean")
        progress_probe = None
        if progress_reporter is not None:
            try:
                progress_probe = MaceTrainingProgressProbe.from_rendered_recipe(
                    rendered,
                    work_dir=Path(work_dir),
                    member_id=progress_reporter.operation_id,
                    member_index=member_index,
                    total_members=total_members,
                )
            except Exception:
                progress_probe = None
        run_operation(
            self.train_context,
            "mace_run_train",
            ["--config", str(rendered_yaml)],
            cwd=work_dir,
            diagnostic_name="mace_train",
            progress_reporter=progress_reporter,
            progress_probe=progress_probe,
            probe_interval_seconds=float(
                self.train_context.runtime.get("progress_probe_interval_seconds", 1.0)
            ),
        )
        suffix = "_stagetwo.model" if swa else ".model"
        model = model_dir / f"{name}{suffix}"
        if not model.is_file() or model.stat().st_size <= 0:
            phase = "stage-two" if swa else "stage-one"
            raise ValueError(f"MACE native train did not produce the required {phase} terminal model: {model}")
        return model.resolve()

    def train_distributed(
        self,
        rendered_yaml: Path,
        work_dir: Path,
        assignment: DistributedAssignment,
        *,
        progress_reporter: OperationProgressReporter | None = None,
        member_index: int = 1,
        total_members: int = 1,
    ) -> Path:
        allocation = self.train_context.allocation
        if (
            allocation.mode != "distributed"
            or allocation.launcher != "torchrun"
            or allocation.nodes != 1
            or allocation.gpus_per_node != len(assignment.visible_devices)
            or total_members != 1
        ):
            raise ValueError("MACE distributed training allocation/member shape is invalid")
        rendered = yaml.safe_load(rendered_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(rendered, dict):
            raise ValueError("MACE native train requires a rendered YAML object")
        name = rendered.get("name")
        raw_model_dir = rendered.get("model_dir")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MACE native train requires an explicit non-empty name")
        if not isinstance(raw_model_dir, str) or not raw_model_dir.strip():
            raise ValueError("MACE native train requires an explicit model_dir")
        model_dir = Path(raw_model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(work_dir) / model_dir
        swa = rendered.get("swa", False)
        if not isinstance(swa, bool):
            raise ValueError("MACE native train recipe swa must be boolean")
        receipt_root = Path(work_dir) / "distributed_receipts"
        if receipt_root.exists():
            raise ValueError("MACE distributed rank receipt root already exists")
        receipt_root.mkdir(parents=True)
        _prepare_primary_rank_log_views(receipt_root, Path(work_dir))
        native_command = (
            self.train_context.executable("mace_run_train"),
            "--config",
            str(Path(rendered_yaml).resolve()),
            "--distributed",
            "--launcher",
            "torchrun",
        )
        arguments = build_single_node_torchrun_arguments(
            nproc_per_node=allocation.gpus_per_node,
            worker_module="workflow_backends.teacher.mace.distributed_worker",
            worker_arguments=(
                "--receipt-root",
                str(receipt_root.resolve()),
                "--expected-world-size",
                str(allocation.gpus_per_node),
                "--visible-devices-json",
                json.dumps(list(assignment.visible_devices), separators=(",", ":")),
                "--assignment-id",
                assignment.assignment_id,
                "--",
                *native_command,
            ),
        )
        run_operation(
            self.train_context,
            "python",
            arguments,
            cwd=work_dir,
            diagnostic_name="mace_train_distributed",
            progress_reporter=progress_reporter,
            timeout_seconds=24.0 * 60.0 * 60.0,
        )
        load_and_validate_single_node_rank_receipts(
            receipt_root,
            expected_world_size=allocation.gpus_per_node,
            expected_visible_devices=assignment.visible_devices,
            expected_assignment_id=assignment.assignment_id,
            expected_command=native_command,
        )
        suffix = "_stagetwo.model" if swa else ".model"
        model = model_dir / f"{name}{suffix}"
        if not model.is_file() or model.stat().st_size <= 0:
            phase = "stage-two" if swa else "stage-one"
            raise ValueError(
                f"MACE distributed train did not produce the required {phase} terminal model: {model}"
            )
        return model.resolve()

    def label(self, models: tuple[Path, ...], frames: list[Atoms], work_dir: Path, *, progress_reporter: OperationProgressReporter | None = None, trusted_below: float | None = None) -> list[Atoms]:
        if self.label_context is None:
            raise ValueError("MACE lifecycle label execution requires its selected operation context")
        if frames and inference_pool_enabled(self.label_context):
            return shard_labels(
                self, context=self.label_context, models=models, frames=frames,
                work_dir=work_dir, trusted_below=trusted_below,
                progress_reporter=progress_reporter,
            )
        task = Path(work_dir) / "mace_label"
        task.mkdir(parents=True, exist_ok=True)
        input_path, output_path = task / "input.xyz", task / "labeled.xyz"
        uncertainty_path, config_path = task / "uncertainties.json", task / "worker_config.json"
        write_extxyz_structures(input_path, frames, allow_empty=True)
        config = {"model_paths": [str(path.resolve()) for path in models], "device": "cuda"}
        if len(models) == 1:
            config["committee_evaluation_mode"] = "separate"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with WorkerObservationScope(task) as observations:
            probe = None
            if progress_reporter is not None:
                projector = LabelObservationProjector(
                    {str(frame.info.get("structure_id", "")): str(frame.info.get("system_id", "unknown")) for frame in frames},
                    total=len(frames),
                    trusted_below=trusted_below,
                )
                probe = WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), projector)
            run_operation(self.label_context, "python", ["-m", "workflow_backends.teacher.mace.relabel_worker", "--input", str(input_path), "--output", str(output_path), "--uncertainties", str(uncertainty_path), "--config", str(config_path), "--observation-channel", str(observations.path)], cwd=task, diagnostic_name="mace_label", progress_reporter=progress_reporter, progress_probe=probe)
        labeled = read_extxyz_structures(output_path, allow_empty=True)
        values = json.loads(uncertainty_path.read_text(encoding="utf-8"))
        if len(labeled) != len(frames) or len(values) != len(frames):
            raise ValueError("MACE lifecycle label worker returned mismatched outputs")
        labeled = [
            _restore_extxyz_transport_geometry(source, atoms)
            for source, atoms in zip(frames, labeled)
        ]
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
            raise ValueError("MACE lifecycle gmm_reference requires its selected operation context")
        task = Path(work_dir) / "mace_gmm_reference"
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
        self._append_descriptor_runtime(arguments, context)
        with WorkerObservationScope(task) as observations:
            arguments.extend(("--observation-channel", str(observations.path)))
            probe = None if progress_reporter is None else WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), DescriptorObservationProjector(label="reference"))
            run_operation(context, "python", arguments, cwd=task, diagnostic_name="mace_gmm_reference", progress_reporter=progress_reporter, progress_probe=probe)
        manifest = validate_gmm_router(
            router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        return GMMReferenceFitResult(
            router_dir=Path(router_dir),
            policy_id=POLICY_ID,
            policy_revision=POLICY_REVISION,
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
            raise ValueError("MACE lifecycle gmm_score requires its selected operation context")
        task = Path(work_dir) / "mace_gmm_score"
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
        self._append_descriptor_runtime(arguments, context)
        try:
            with WorkerObservationScope(task) as observations:
                arguments.extend(("--observation-channel", str(observations.path)))
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), ScoreObservationProjector(progress_systems or {}, total=len(frame_ids), role="teacher"))
                operation = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=task,
                    diagnostic_name="mace_gmm_score",
                    structured_result_prefix="DICE_GMM_RESULT=",
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                )
                return self._score_result(
                    structured_result_stdout(operation, prefix="DICE_GMM_RESULT="),
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
            raise ValueError("MACE lifecycle gmm_score requires its selected operation context")
        if not candidate_paths or len(candidate_paths) != len(frame_ids_by_chunk):
            raise ValueError("MACE lifecycle chunked gmm_score requires aligned non-empty input chunks")
        frame_ids = tuple(identity for chunk in frame_ids_by_chunk for identity in chunk)
        if not frame_ids or any(not identity for identity in frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError("MACE lifecycle chunked gmm_score requires unique canonical frame identities")
        if any(not chunk for chunk in frame_ids_by_chunk):
            raise ValueError("MACE lifecycle chunked gmm_score does not accept empty input chunks")

        task = Path(work_dir) / "mace_gmm_score"
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
                        for path, ids in zip(candidate_paths, frame_ids_by_chunk)
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
                probe = None if progress_reporter is None else WorkerObservationProgressProbe(WorkerObservationCursor(observations.path), ScoreObservationProjector(progress_systems or {}, total=len(frame_ids), role="teacher"))
                operation = run_operation(
                    context,
                    "python",
                    arguments,
                    cwd=task,
                    diagnostic_name="mace_gmm_score",
                    structured_result_prefix="DICE_GMM_RESULT=",
                    structured_result_limit_bytes=64 * 1024 * 1024,
                    progress_reporter=progress_reporter,
                    progress_probe=probe,
                )
                return self._score_result(
                    structured_result_stdout(operation, prefix="DICE_GMM_RESULT="),
                    frame_ids,
                    emit_frame_means,
                )
        finally:
            chunk_manifest.unlink(missing_ok=True)

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
        arguments = [
            "-m", "workflow_backends.teacher.mace.descriptor_worker",
            "--operation", operation,
            "--model", str(Path(model).resolve()),
            "--router-dir", str(Path(router_dir).resolve()),
            "--model-digest", str(model_digest),
            "--reference-digest", str(reference_digest),
        ]
        if input_path is not None:
            arguments.extend(("--input", str(Path(input_path).resolve())))
        return arguments

    @staticmethod
    def _append_descriptor_runtime(arguments: list[str], context: OperationContext) -> None:
        for runtime_key, option in (
            ("mace_workers", "--workers"),
            ("mace_batch_size", "--batch-size"),
            ("mace_threads", "--torch-threads"),
        ):
            if runtime_key in context.runtime:
                arguments.extend((option, str(context.runtime[runtime_key])))

    @staticmethod
    def _score_result(stdout: str, expected_ids: tuple[str, ...], emit_frame_means: bool) -> GMMScoreResult:
        marker = "DICE_GMM_RESULT="
        position = stdout.rfind(marker)
        if position < 0:
            raise ValueError("MACE lifecycle gmm_score worker did not return a frame-level result")
        try:
            payload = json.loads(stdout[position + len(marker):].splitlines()[0])
            if not isinstance(payload, dict):
                raise ValueError("frame result is not an object")
            payload = normalize_gmm_frame_score_result(payload)
            if set(payload) != {
                "schema",
                "schema_revision",
                "frame_ids",
                "frame_p",
                "frame_statistics",
                "frame_means",
                "descriptor_schema",
            }:
                raise ValueError("frame result has unsupported fields")
            frame_ids = np.asarray(payload["frame_ids"], dtype=np.str_)
            frame_p = np.asarray(payload["frame_p"], dtype=np.float64)
            statistics = np.asarray(payload["frame_statistics"], dtype=np.float64)
            raw_means = payload.get("frame_means")
            descriptor_schema = validated_readout_input_schema(payload["descriptor_schema"])
            means = None if raw_means is None else np.asarray(raw_means, dtype=np.float32)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("MACE lifecycle gmm_score worker returned malformed frame-level JSON") from exc
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
            raise ValueError("MACE lifecycle gmm_score worker changed canonical frame results")
        if emit_frame_means:
            if (
                means is None
                or means.shape != (count, int(descriptor_schema["width"]))
                or not np.all(np.isfinite(means))
            ):
                raise ValueError("MACE lifecycle gmm_score worker omitted finite dynamic-width readout frame means")
        elif means is not None:
            raise ValueError("MACE lifecycle random gmm_score unexpectedly returned frame means")
        return GMMScoreResult(frame_ids, frame_p, statistics, means, descriptor_schema)
