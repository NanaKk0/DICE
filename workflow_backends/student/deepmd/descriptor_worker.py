from __future__ import annotations

import argparse
import os
import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from workflow_artifacts.gmm_worker_evidence import (
    element_means_payload,
    gmm_frame_score_result_payload,
    normalize_gmm_score_chunks,
)

from workflow_backends.descriptor_stream import (
    DescriptorStreamHeader,
    validated_descriptor_chunks,
)
from workflow_backends.student.deepmd.descriptors import (
    open_deepmd_readout_input_descriptor_chunk_stream,
)
from workflow_backends.teacher.deepmd.descriptors import (
    DeepMDPTReadoutInputExporter,
    deepmd_pt_readout_input_descriptor_chunk_stream,
)
from workflow_core.structures.formats import read_extxyz_structures
from workflow_artifacts.lifecycle.gmm_io import (
    GMMReferenceStreamFitRequest,
    GMMReferenceFitResult,
    GMMStreamScoreRequest,
    fit_gmm_router_stream,
    score_gmm_router_stream,
    validate_gmm_router,
    GMMScoreResult,
)
from workflow_runtime.lifecycle.progress.worker_observations import (
    WorkerObservationWriter,
    emit_gmm_score_chunks,
    observed_descriptor_chunks,
    optional_observation_writer,
)


STRUCTURE_VECTOR_RESULT_PREFIX = "DICE_STRUCTURE_VECTORS="


def export_deepmd_readout_input_frame_means(
    model: str | Path,
    frames: Sequence[object],
    *,
    type_map: Sequence[str],
    chunk_frames: int = 50,
    batch_size: int = 9472,
) -> tuple[dict[str, object], tuple[str, ...], np.ndarray]:
    schema, identities, rows, _elements = (
        export_deepmd_readout_input_frame_and_element_means(
            model,
            frames,
            type_map=type_map,
            chunk_frames=chunk_frames,
            batch_size=batch_size,
        )
    )
    return schema, identities, rows


def export_deepmd_readout_input_frame_and_element_means(
    model: str | Path,
    frames: Sequence[object],
    *,
    type_map: Sequence[str],
    chunk_frames: int = 50,
    batch_size: int = 9472,
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    np.ndarray,
    dict[str, tuple[tuple[str, ...], np.ndarray]],
]:
    ordered_frames = tuple(frames)
    _require_canonical_identities(
        ordered_frames, label="DeepMD structural-vector readout"
    )
    with _open_descriptor_chunk_stream(
        Path(model).resolve(),
        (ordered_frames,),
        frame_count=len(ordered_frames),
        active_type_map=tuple(str(item) for item in type_map),
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    ) as (header, chunks):
        identities: list[str | None] = [None] * len(ordered_frames)
        means: list[np.ndarray | None] = [None] * len(ordered_frames)
        element_means: dict[str, list[tuple[int, str, np.ndarray]]] = {
            str(element): [] for element in header.type_map
        }
        for chunk in validated_descriptor_chunks(header, chunks):
            for local_index, ordinal_value in enumerate(chunk.frame_ordinals):
                ordinal = int(ordinal_value)
                start = int(chunk.frame_ptr[local_index])
                stop = int(chunk.frame_ptr[local_index + 1])
                if stop <= start:
                    raise ValueError(
                        "DeepMD structural-vector readout requires atoms per frame"
                    )
                identities[ordinal] = str(chunk.frame_ids[local_index])
                means[ordinal] = np.asarray(
                    chunk.descriptors[start:stop].mean(axis=0),
                    dtype=np.float32,
                )
                frame_types = chunk.type_indices[start:stop]
                for type_index in dict.fromkeys(int(item) for item in frame_types):
                    element_means[header.type_map[type_index]].append(
                        (
                            ordinal,
                            str(chunk.frame_ids[local_index]),
                            np.asarray(
                                chunk.descriptors[start:stop][
                                    frame_types == type_index
                                ].mean(axis=0),
                                dtype=np.float32,
                            ),
                        )
                    )
    expected = tuple(str(frame.info["structure_id"]) for frame in ordered_frames)
    if any(identity is None for identity in identities) or any(
        mean is None for mean in means
    ):
        raise ValueError("DeepMD structural-vector readout changed canonical evidence")
    actual = tuple(str(identity) for identity in identities)
    rows = np.ascontiguousarray(
        np.stack([mean for mean in means if mean is not None]),
        dtype=np.float32,
    )
    if (
        actual != expected
        or rows.shape != (len(expected), int(header.schema["width"]))
        or not np.isfinite(rows).all()
    ):
        raise ValueError("DeepMD structural-vector readout changed canonical evidence")
    elements: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for element, values in element_means.items():
        if not values:
            continue
        values.sort(key=lambda item: item[0])
        element_rows = np.ascontiguousarray(
            np.stack([item[2] for item in values]), dtype=np.float32
        )
        if (
            element_rows.shape[1] != int(header.schema["width"])
            or not np.isfinite(element_rows).all()
        ):
            raise ValueError("DeepMD element structural vectors are invalid")
        elements[element] = (tuple(item[1] for item in values), element_rows)
    return dict(header.schema), actual, rows, elements


def run_gmm_reference_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    router_dir: str | Path,
    type_map: Sequence[str],
    chunk_frames: int = 50,
    batch_size: int = 9472,
    spill: bool = False,
    work_dir: str | Path | None = None,
    observation_writer: WorkerObservationWriter | None = None,
) -> None:
    """Fit the portable student router inside one model-owning native operation."""
    model_path = Path(model).resolve()
    reference = Path(reference_path).resolve()
    frames = read_extxyz_structures(reference, allow_empty=False)
    _require_canonical_identities(frames, label="DeepMD-TF GMM reference")
    expected_type_map = tuple(str(item) for item in type_map)
    with _open_descriptor_chunk_stream(
        model_path,
        (frames,),
        frame_count=len(frames),
        active_type_map=expected_type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    ) as (header, chunks):
        if header.type_map != expected_type_map:
            raise ValueError("DeepMD-TF GMM reference model type_map does not match student type_map")
        fit_gmm_router_stream(
            GMMReferenceStreamFitRequest(
                router_dir=Path(router_dir),
                header=header,
                chunks=observed_descriptor_chunks(
                    chunks,
                    writer=observation_writer,
                    total=header.frame_count,
                    width=int(header.schema["width"]),
                ),
                model_digest=_file_digest(model_path),
                reference_digest=_file_digest(reference),
                spill=bool(spill),
                work_dir=None if work_dir is None else Path(work_dir),
            )
        )
        if observation_writer is not None:
            manifest = validate_gmm_router(router_dir)
            element_count = len(dict(manifest.get("elements", {})))
            observation_writer.descriptor_progress(
                completed=header.frame_count,
                total=header.frame_count,
                width=int(header.schema["width"]),
                elements_completed=element_count,
                elements_total=element_count,
            )


def run_gmm_score_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    observation_path: str | Path,
    router_dir: str | Path,
    ordered_identities: Sequence[str],
    emit_frame_means: bool,
    chunk_frames: int = 50,
    batch_size: int = 9472,
    observation_writer: WorkerObservationWriter | None = None,
) -> GMMScoreResult:
    """Filter the canonical observation source and stream one persisted-router replay."""
    model_path = Path(model).resolve()
    reference = Path(reference_path).resolve()
    identities = tuple(str(item) for item in ordered_identities)
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("DeepMD-TF GMM score identity filter must be unique and non-empty")
    frames = _ordered_observation_filter(Path(observation_path).resolve(), identities)
    model_digest = _file_digest(model_path)
    reference_digest = _file_digest(reference)
    if frames:
        manifest = validate_gmm_router(
            router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        active_type_map = tuple(str(item) for item in manifest["type_map"])
        with _open_descriptor_chunk_stream(
            model_path,
            (frames,),
            frame_count=len(frames),
            active_type_map=active_type_map,
            chunk_frames=chunk_frames,
            batch_size=batch_size,
        ) as (header, chunks):
            scored = score_gmm_router_stream(
                GMMStreamScoreRequest(
                    router_dir=Path(router_dir),
                    header=header,
                    chunks=observed_descriptor_chunks(
                        chunks,
                        writer=observation_writer,
                        total=header.frame_count,
                        width=int(header.schema["width"]),
                    ),
                    model_digest=model_digest,
                    reference_digest=reference_digest,
                    emit_frame_means=emit_frame_means,
                )
            )
    else:
        manifest = validate_gmm_router(
            router_dir,
            model_digest=model_digest,
            reference_digest=reference_digest,
        )
        header = DescriptorStreamHeader(
            schema=dict(manifest["descriptor_schema"]),
            type_map=tuple(str(item) for item in manifest["type_map"]),
            frame_count=0,
        )
        scored = score_gmm_router_stream(
            GMMStreamScoreRequest(
                router_dir=Path(router_dir),
                header=header,
                chunks=(),
                model_digest=model_digest,
                reference_digest=reference_digest,
                emit_frame_means=emit_frame_means,
            )
        )
    actual = tuple(str(item) for item in scored.frame_ids.tolist())
    if actual != identities:
        raise ValueError("DeepMD-TF GMM score result does not match the ordered identity filter")
    emit_gmm_score_chunks(
        observation_writer,
        role="student",
        frame_ids=scored.frame_ids,
        p_values=scored.frame_p,
    )
    return scored


def run_gmm_score_chunks_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    observation_paths: Sequence[str | Path],
    ordered_identity_chunks: Sequence[Sequence[str]],
    router_dir: str | Path,
    emit_frame_means: bool,
    chunk_frames: int = 50,
    batch_size: int = 9472,
    observation_writer: WorkerObservationWriter | None = None,
) -> GMMScoreResult:
    """Score all bounded observation inputs within one loaded DeepPot operation."""

    if not observation_paths or len(observation_paths) != len(ordered_identity_chunks):
        raise ValueError("DeepMD-TF chunked GMM score requires aligned non-empty input chunks")
    identity_chunks = tuple(tuple(str(item) for item in chunk) for chunk in ordered_identity_chunks)
    identities = tuple(identity for chunk in identity_chunks for identity in chunk)
    if (
        not identities
        or any(not identity for identity in identities)
        or len(set(identities)) != len(identities)
        or any(not chunk for chunk in identity_chunks)
    ):
        raise ValueError("DeepMD-TF chunked GMM score identities must be unique and non-empty")
    paths = tuple(Path(path).resolve() for path in observation_paths)
    model_path = Path(model).resolve()
    reference = Path(reference_path).resolve()
    model_digest = _file_digest(model_path)
    reference_digest = _file_digest(reference)

    def replay_chunks():
        for path, chunk_identities in zip(paths, identity_chunks):
            yield _ordered_observation_filter(path, chunk_identities)

    manifest = validate_gmm_router(
        router_dir,
        model_digest=model_digest,
        reference_digest=reference_digest,
    )
    active_type_map = tuple(str(item) for item in manifest["type_map"])
    with _open_descriptor_chunk_stream(
        model_path,
        replay_chunks(),
        frame_count=len(identities),
        active_type_map=active_type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    ) as (header, chunks):
        scored = score_gmm_router_stream(
            GMMStreamScoreRequest(
                router_dir=Path(router_dir),
                header=header,
                chunks=observed_descriptor_chunks(
                    chunks,
                    writer=observation_writer,
                    total=header.frame_count,
                    width=int(header.schema["width"]),
                ),
                model_digest=model_digest,
                reference_digest=reference_digest,
                emit_frame_means=emit_frame_means,
            )
        )
    actual = tuple(str(item) for item in scored.frame_ids.tolist())
    if actual != identities:
        raise ValueError("DeepMD-TF GMM score result does not match the ordered chunk identities")
    emit_gmm_score_chunks(
        observation_writer,
        role="student",
        frame_ids=scored.frame_ids,
        p_values=scored.frame_p,
    )
    return scored


def run_gmm_score_rolling_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    rolling_root: str | Path,
    ordered_identities: Sequence[str],
    slot_count: int,
    router_dir: str | Path,
    emit_frame_means: bool,
    chunk_frames: int = 50,
    batch_size: int = 9472,
    observation_writer: WorkerObservationWriter | None = None,
) -> GMMScoreResult:
    """Consume acknowledged extxyz slots within one loaded DeepPot operation."""
    from workflow_backends.rolling_extxyz import RollingExtxyzChannel

    identities = tuple(str(item) for item in ordered_identities)
    channel = RollingExtxyzChannel.open(
        rolling_root,
        ordered_identities=identities,
        slot_count=slot_count,
    )
    model_path = Path(model).resolve()
    reference = Path(reference_path).resolve()
    model_digest = _file_digest(model_path)
    reference_digest = _file_digest(reference)
    manifest = validate_gmm_router(
        router_dir,
        model_digest=model_digest,
        reference_digest=reference_digest,
    )
    active_type_map = tuple(str(item) for item in manifest["type_map"])
    frame_chunks = channel.consume(producer_alive=lambda: os.getppid() != 1)
    with _open_descriptor_chunk_stream(
        model_path,
        frame_chunks,
        frame_count=len(identities),
        active_type_map=active_type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    ) as (header, chunks):
        scored = score_gmm_router_stream(
            GMMStreamScoreRequest(
                router_dir=Path(router_dir),
                header=header,
                chunks=observed_descriptor_chunks(
                    chunks,
                    writer=observation_writer,
                    total=header.frame_count,
                    width=int(header.schema["width"]),
                ),
                model_digest=model_digest,
                reference_digest=reference_digest,
                emit_frame_means=emit_frame_means,
            )
        )
    actual = tuple(str(item) for item in scored.frame_ids.tolist())
    if actual != identities:
        raise ValueError(
            "DeepMD-TF rolling GMM score result does not match canonical identities"
        )
    emit_gmm_score_chunks(
        observation_writer,
        role="student",
        frame_ids=scored.frame_ids,
        p_values=scored.frame_p,
    )
    return scored


def run_gmm_score_geometry_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    geometry_socket: str | Path,
    ordered_identities: Sequence[str],
    byte_budget: int,
    atom_budget: int,
    router_dir: str | Path,
    emit_frame_means: bool,
    chunk_frames: int = 50,
    batch_size: int = 9472,
    observation_writer: WorkerObservationWriter | None = None,
) -> GMMScoreResult:
    """Consume the T2 framed geometry protocol in one model-owning operation."""
    from workflow_backends.geometry_stream import consume_geometry_stream

    identities = tuple(str(item) for item in ordered_identities)
    model_path = Path(model).resolve()
    reference = Path(reference_path).resolve()
    model_digest = _file_digest(model_path)
    reference_digest = _file_digest(reference)
    manifest = validate_gmm_router(
        router_dir,
        model_digest=model_digest,
        reference_digest=reference_digest,
    )
    active_type_map = tuple(str(item) for item in manifest["type_map"])
    frame_chunks = consume_geometry_stream(
        geometry_socket,
        ordered_identities=identities,
        byte_budget=byte_budget,
        atom_budget=atom_budget,
    )
    with _open_descriptor_chunk_stream(
        model_path,
        frame_chunks,
        frame_count=len(identities),
        active_type_map=active_type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    ) as (header, chunks):
        scored = score_gmm_router_stream(
            GMMStreamScoreRequest(
                router_dir=Path(router_dir),
                header=header,
                chunks=observed_descriptor_chunks(
                    chunks,
                    writer=observation_writer,
                    total=header.frame_count,
                    width=int(header.schema["width"]),
                ),
                model_digest=model_digest,
                reference_digest=reference_digest,
                emit_frame_means=emit_frame_means,
            )
        )
    actual = tuple(str(item) for item in scored.frame_ids.tolist())
    if actual != identities:
        raise ValueError(
            "DeepMD-TF geometry GMM score result does not match canonical identities"
        )
    emit_gmm_score_chunks(
        observation_writer,
        role="student",
        frame_ids=scored.frame_ids,
        p_values=scored.frame_p,
    )
    return scored


def run_gmm_screen_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    geometry_socket: str | Path,
    ordered_identities: Sequence[str],
    byte_budget: int,
    atom_budget: int,
    router_dir: str | Path,
    type_map: Sequence[str],
    emit_frame_means: bool,
    chunk_frames: int,
    batch_size: int,
    observation_writer: WorkerObservationWriter | None = None,
) -> tuple[GMMReferenceFitResult, GMMScoreResult, dict[str, float]]:
    """Fit and score with one DPA4C model load inside one screen worker."""
    import time

    from workflow_backends.deepmd_pt_runtime import bounded_prefetch
    from workflow_backends.geometry_stream import consume_geometry_stream

    model_path = Path(model).resolve()
    reference = Path(reference_path).resolve()
    identities = tuple(str(item) for item in ordered_identities)
    active_type_map = tuple(str(item) for item in type_map)
    model_load_started = time.perf_counter()
    exporter = DeepMDPTReadoutInputExporter(
        model_path, device="cuda", model_family="dpa4c"
    )
    model_load_seconds = time.perf_counter() - model_load_started
    candidate_frames = bounded_prefetch(
        consume_geometry_stream(
            geometry_socket,
            ordered_identities=identities,
            byte_budget=byte_budget,
            atom_budget=atom_budget,
        ),
        slots=2,
        diagnostic_name="candidate_prefetch",
    )
    reference_started = time.perf_counter()
    reference_frames = read_extxyz_structures(reference, allow_empty=False)
    _require_canonical_identities(reference_frames, label="DPA4C fused GMM reference")
    reference_header, reference_chunks = deepmd_pt_readout_input_descriptor_chunk_stream(
        (reference_frames,),
        frame_count=len(reference_frames),
        exporter=exporter,
        type_map=active_type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    )
    fit = fit_gmm_router_stream(
        GMMReferenceStreamFitRequest(
            router_dir=Path(router_dir),
            header=reference_header,
            chunks=observed_descriptor_chunks(
                reference_chunks,
                writer=observation_writer,
                total=reference_header.frame_count,
                width=int(reference_header.schema["width"]),
            ),
            model_digest=_file_digest(model_path),
            reference_digest=_file_digest(reference),
        )
    )
    reference_seconds = time.perf_counter() - reference_started
    # Router publication/validation is complete before the first score forward.
    manifest = validate_gmm_router(
        router_dir,
        model_digest=_file_digest(model_path),
        reference_digest=_file_digest(reference),
    )
    router_type_map = tuple(str(item) for item in manifest["type_map"])
    fitted_type_map = tuple(str(item) for item in fit.element_pca_dimensions)
    filtered_active_type_map = tuple(
        element for element in active_type_map if element in set(router_type_map)
    )
    if (
        not router_type_map
        or router_type_map != fitted_type_map
        or router_type_map != filtered_active_type_map
    ):
        raise ValueError(
            "DPA4C fused router type map must be the fitted declared-order active subset"
        )
    score_started = time.perf_counter()
    score_header, score_chunks = deepmd_pt_readout_input_descriptor_chunk_stream(
        candidate_frames,
        frame_count=len(identities),
        exporter=exporter,
        type_map=active_type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    )
    score_chunks = bounded_prefetch(score_chunks, slots=2)
    scored = score_gmm_router_stream(
        GMMStreamScoreRequest(
            router_dir=Path(router_dir),
            header=score_header,
            chunks=observed_descriptor_chunks(
                score_chunks,
                writer=observation_writer,
                total=score_header.frame_count,
                width=int(score_header.schema["width"]),
            ),
            model_digest=_file_digest(model_path),
            reference_digest=_file_digest(reference),
            emit_frame_means=emit_frame_means,
        )
    )
    actual = tuple(str(item) for item in scored.frame_ids.tolist())
    if actual != identities:
        raise ValueError("DPA4C fused score changed canonical candidate order")
    score_seconds = time.perf_counter() - score_started
    return fit, scored, {
        "model_load_seconds": model_load_seconds,
        "reference_seconds": reference_seconds,
        "score_seconds": score_seconds,
    }


@contextmanager
def _open_descriptor_chunk_stream(
    model_path: Path,
    frame_chunks: Iterable[Sequence[object]],
    *,
    frame_count: int,
    active_type_map: Sequence[str],
    chunk_frames: int,
    batch_size: int = 9472,
) -> Iterator[tuple[DescriptorStreamHeader, Iterable[object]]]:
    """Select the native graph owner while retaining one common stream ABI."""

    if model_path.suffix == ".pt2":
        exporter = DeepMDPTReadoutInputExporter(
            model_path, device="cuda", model_family="dpa4c"
        )
        header, chunks = deepmd_pt_readout_input_descriptor_chunk_stream(
            frame_chunks,
            frame_count=frame_count,
            exporter=exporter,
            type_map=active_type_map,
            chunk_frames=chunk_frames,
            batch_size=batch_size,
        )
        from workflow_backends.deepmd_pt_runtime import bounded_prefetch

        yield header, bounded_prefetch(chunks, slots=2)
        return
    if model_path.suffix != ".pb":
        raise ValueError("DeepMD GMM descriptor model must use .pb or .pt2")
    with open_deepmd_readout_input_descriptor_chunk_stream(
        model_path,
        frame_chunks,
        frame_count=frame_count,
        chunk_frames=chunk_frames,
    ) as stream:
        header, _chunks = stream
        expected = tuple(str(item) for item in active_type_map)
        if header.type_map != expected:
            raise ValueError(
                "DeepMD-TF GMM model type_map does not match the active workflow map"
            )
        yield stream


def _ordered_observation_filter(path: Path, identities: tuple[str, ...]):
    frames = read_extxyz_structures(path, allow_empty=False)
    _require_canonical_identities(frames, label="DeepMD-TF GMM observation source")
    by_identity = {str(frame.info["structure_id"]): frame for frame in frames}
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError(
            "DeepMD-TF GMM score identity filter is absent from the observation source: "
            + ", ".join(missing)
        )
    return [by_identity[identity] for identity in identities]


def _require_canonical_identities(frames, *, label: str) -> None:
    identities = tuple(str(frame.info.get("structure_id", "")) for frame in frames)
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError(f"{label} requires unique non-empty canonical structure_id values")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a streaming DeepMD-TF student GMM operation.")
    parser.add_argument(
        "--operation", required=True,
        choices=("gmm_reference", "gmm_score", "gmm_screen", "structure_vectors"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--router")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--type-map")
    parser.add_argument("--observations")
    parser.add_argument("--identity-filter")
    parser.add_argument("--chunk-manifest")
    parser.add_argument("--rolling-root")
    parser.add_argument("--rolling-slots", type=int)
    parser.add_argument("--geometry-socket")
    parser.add_argument("--geometry-byte-budget", type=int)
    parser.add_argument("--geometry-atom-budget", type=int)
    parser.add_argument("--compact-result-root")
    parser.add_argument("--emit-frame-means", action="store_true")
    parser.add_argument("--observation-channel")
    args = parser.parse_args(argv)
    observation_writer = optional_observation_writer(args.observation_channel)
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    chunk_frames = int(runtime.get("descriptor_chunk_frames", 50))
    batch_size = int(runtime.get("batch_size", 9472))
    if args.operation == "structure_vectors":
        if (
            args.observations is None
            or args.type_map is None
            or args.compact_result_root is None
        ):
            raise ValueError(
                "DeepMD structure_vectors requires observations, type map, and compact result root"
            )
        type_map = json.loads(Path(args.type_map).read_text(encoding="utf-8"))
        if not isinstance(type_map, list):
            raise ValueError("DeepMD structure_vectors type map must be a JSON array")
        frames = read_extxyz_structures(args.observations, allow_empty=False)
        schema, identities, means, element_means = (
            export_deepmd_readout_input_frame_and_element_means(
            args.model,
            frames,
            type_map=tuple(str(item) for item in type_map),
            chunk_frames=chunk_frames,
            batch_size=batch_size,
            )
        )
        from workflow_backends.compact_structure_vector_store import (
            write_compact_structure_vector_store,
        )

        payload = write_compact_structure_vector_store(
            args.compact_result_root,
            descriptor_schema=schema,
            frame_ids=identities,
            frame_means=means,
            element_means=element_means,
            model_digest=_file_digest(Path(args.model).resolve()),
        )
        print(
            STRUCTURE_VECTOR_RESULT_PREFIX
            + json.dumps(payload, separators=(",", ":"))
        )
    elif args.operation == "gmm_screen":
        if any(
            value is None
            for value in (
                args.type_map,
                args.identity_filter,
                args.geometry_socket,
                args.geometry_byte_budget,
                args.geometry_atom_budget,
                args.compact_result_root,
                args.reference,
                args.router,
            )
        ):
            raise ValueError("DPA4C fused gmm_screen requires reference, geometry, and result inputs")
        identities = json.loads(Path(args.identity_filter).read_text(encoding="utf-8"))
        type_map = json.loads(Path(args.type_map).read_text(encoding="utf-8"))
        if not isinstance(identities, list) or not isinstance(type_map, list):
            raise ValueError("DPA4C fused gmm_screen manifests must be JSON arrays")
        fit, scored, phase_timings = run_gmm_screen_operation(
            model=args.model,
            reference_path=args.reference,
            geometry_socket=args.geometry_socket,
            ordered_identities=tuple(str(item) for item in identities),
            byte_budget=args.geometry_byte_budget,
            atom_budget=args.geometry_atom_budget,
            router_dir=args.router,
            type_map=tuple(str(item) for item in type_map),
            emit_frame_means=bool(args.emit_frame_means),
            chunk_frames=chunk_frames,
            batch_size=batch_size,
            observation_writer=observation_writer,
        )
        from workflow_backends.compact_score_store import write_compact_score_store
        from workflow_backends.deepmd_pt_runtime import pop_runtime_diagnostics

        payload = write_compact_score_store(
            args.compact_result_root,
            scored,
            model_digest=_file_digest(Path(args.model).resolve()),
            reference_digest=_file_digest(Path(args.reference).resolve()),
            runtime_diagnostics=pop_runtime_diagnostics(),
        )
        payload["fit"] = {
            "policy_id": fit.policy_id,
            "policy_revision": fit.policy_revision,
            "element_pca_dimensions": dict(fit.element_pca_dimensions),
            "descriptor_schema": dict(fit.descriptor_schema or {}),
        }
        payload["phase_timings"] = phase_timings
        payload["operation_evidence"] = {
            "operation_role": "student.deepmd.pt.gmm_screen",
            "worker_count": 1,
            "model_load_count": 1,
            "candidate_prefetch_slots": 2,
        }
        print("DICE_GMM_SCORE_RESULT=" + json.dumps(payload, separators=(",", ":")))
    elif args.operation == "gmm_reference":
        if args.type_map is None or args.reference is None or args.router is None:
            raise ValueError("DeepMD-TF gmm_reference requires --type-map")
        type_map = json.loads(Path(args.type_map).read_text(encoding="utf-8"))
        if not isinstance(type_map, list):
            raise ValueError("DeepMD-TF gmm_reference type map must be a JSON array")
        run_gmm_reference_operation(
            model=args.model,
            reference_path=args.reference,
            router_dir=args.router,
            type_map=tuple(str(item) for item in type_map),
            chunk_frames=chunk_frames,
            batch_size=batch_size,
            spill=bool(runtime.get("gmm_reference_spill", False)),
            work_dir=Path(args.runtime).resolve().parent,
            observation_writer=observation_writer,
        )
    else:
        if args.reference is None or args.router is None:
            raise ValueError("DeepMD-TF gmm_score requires reference and router")
        if args.geometry_socket is not None:
            if (
                args.identity_filter is None
                or args.geometry_byte_budget is None
                or args.geometry_atom_budget is None
                or args.compact_result_root is None
            ):
                raise ValueError(
                    "DeepMD-TF geometry gmm_score requires identity filter and occupancy budgets"
                )
            if any(
                value is not None
                for value in (args.observations, args.chunk_manifest, args.rolling_root)
            ):
                raise ValueError(
                    "DeepMD-TF geometry gmm_score forbids extxyz and rolling inputs"
                )
            identities = json.loads(Path(args.identity_filter).read_text(encoding="utf-8"))
            if not isinstance(identities, list):
                raise ValueError("DeepMD-TF geometry identity filter must be a JSON array")
            scored = run_gmm_score_geometry_operation(
                model=args.model,
                reference_path=args.reference,
                geometry_socket=args.geometry_socket,
                ordered_identities=tuple(str(item) for item in identities),
                byte_budget=args.geometry_byte_budget,
                atom_budget=args.geometry_atom_budget,
                router_dir=args.router,
                emit_frame_means=bool(args.emit_frame_means),
                chunk_frames=chunk_frames,
                batch_size=batch_size,
                observation_writer=observation_writer,
            )
        elif args.rolling_root is not None:
            if args.identity_filter is None or args.rolling_slots is None:
                raise ValueError(
                    "DeepMD-TF rolling gmm_score requires identity filter and slot count"
                )
            if args.observations is not None or args.chunk_manifest is not None:
                raise ValueError(
                    "DeepMD-TF rolling gmm_score forbids static observation inputs"
                )
            identities = json.loads(Path(args.identity_filter).read_text(encoding="utf-8"))
            if not isinstance(identities, list):
                raise ValueError("DeepMD-TF rolling identity filter must be a JSON array")
            scored = run_gmm_score_rolling_operation(
                model=args.model,
                reference_path=args.reference,
                rolling_root=args.rolling_root,
                ordered_identities=tuple(str(item) for item in identities),
                slot_count=args.rolling_slots,
                router_dir=args.router,
                emit_frame_means=bool(args.emit_frame_means),
                chunk_frames=chunk_frames,
                batch_size=batch_size,
                observation_writer=observation_writer,
            )
        elif args.chunk_manifest is not None:
            if args.observations is not None or args.identity_filter is not None:
                raise ValueError(
                    "DeepMD-TF chunked gmm_score does not accept observations or identity filter"
                )
            observation_paths, identity_chunks = _read_chunk_manifest(args.chunk_manifest)
            scored = run_gmm_score_chunks_operation(
                model=args.model,
                reference_path=args.reference,
                observation_paths=observation_paths,
                ordered_identity_chunks=identity_chunks,
                router_dir=args.router,
                emit_frame_means=bool(args.emit_frame_means),
                chunk_frames=chunk_frames,
                batch_size=batch_size,
                observation_writer=observation_writer,
            )
        else:
            if args.observations is None or args.identity_filter is None:
                raise ValueError("DeepMD-TF gmm_score requires observations and identity filter")
            identities = json.loads(Path(args.identity_filter).read_text(encoding="utf-8"))
            if not isinstance(identities, list):
                raise ValueError("DeepMD-TF gmm_score identity filter must be a JSON array")
            scored = run_gmm_score_operation(
                model=args.model,
                reference_path=args.reference,
                observation_path=args.observations,
                router_dir=args.router,
                ordered_identities=tuple(str(item) for item in identities),
                emit_frame_means=bool(args.emit_frame_means),
                chunk_frames=chunk_frames,
                batch_size=batch_size,
                observation_writer=observation_writer,
            )
        if args.geometry_socket is not None:
            from workflow_backends.compact_score_store import write_compact_score_store
            from workflow_backends.deepmd_pt_runtime import pop_runtime_diagnostics

            payload = write_compact_score_store(
                args.compact_result_root,
                scored,
                model_digest=_file_digest(Path(args.model).resolve()),
                reference_digest=_file_digest(Path(args.reference).resolve()),
                runtime_diagnostics=pop_runtime_diagnostics(),
            )
        else:
            payload = gmm_frame_score_result_payload(
                frame_ids=[str(item) for item in scored.frame_ids.tolist()],
                frame_p=[float(item) for item in scored.frame_p.tolist()],
                frame_statistics=[float(item) for item in scored.frame_statistics.tolist()],
                frame_means=None
                if scored.frame_means is None
                else np.asarray(scored.frame_means, dtype=np.float32).tolist(),
                descriptor_schema=dict(scored.descriptor_schema or {}),
                element_means=element_means_payload(scored.element_means),
            )
        print("DICE_GMM_SCORE_RESULT=" + json.dumps(payload, separators=(",", ":")))
    return 0


def _read_chunk_manifest(
    path: str | Path,
) -> tuple[tuple[Path, ...], tuple[tuple[str, ...], ...]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("DeepMD-TF GMM score chunk manifest must be readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("DeepMD-TF GMM score chunk manifest has an invalid schema")
    payload = normalize_gmm_score_chunks(payload)
    raw_chunks = payload.get("chunks")
    frame_count = payload.get("frame_count")
    if (
        not isinstance(frame_count, int)
        or frame_count <= 0
        or not isinstance(raw_chunks, list)
        or not raw_chunks
    ):
        raise ValueError("DeepMD-TF GMM score chunk manifest has an invalid schema")
    paths: list[Path] = []
    identity_chunks: list[tuple[str, ...]] = []
    flattened: list[str] = []
    for item in raw_chunks:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("DeepMD-TF GMM score chunk manifest contains an invalid path")
        raw_ids = item.get("frame_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(identity, str) or not identity for identity in raw_ids)
        ):
            raise ValueError("DeepMD-TF GMM score chunk manifest contains invalid frame identities")
        identities = tuple(raw_ids)
        paths.append(Path(item["path"]).resolve())
        identity_chunks.append(identities)
        flattened.extend(identities)
    if len(flattened) != frame_count or len(set(flattened)) != frame_count:
        raise ValueError("DeepMD-TF GMM score chunk manifest changed canonical frame membership")
    return tuple(paths), tuple(identity_chunks)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "main",
    "run_gmm_reference_operation",
    "run_gmm_score_operation",
    "run_gmm_score_chunks_operation",
    "run_gmm_score_rolling_operation",
    "run_gmm_score_geometry_operation",
    "run_gmm_screen_operation",
]
