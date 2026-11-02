from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from workflow_artifacts.gmm_worker_evidence import gmm_frame_score_result_payload, element_means_payload
from ase import Atoms

from workflow_backends.descriptor_stream import (
    DescriptorStreamHeader,
    validated_descriptor_chunks,
)
from workflow_backends.student.nep.descriptors import (
    NEPReadoutInputExporter,
    nep_readout_input_descriptor_stream,
)
from workflow_core.structures.formats import read_extxyz_structures
from workflow_artifacts.lifecycle.gmm_io import (
    GMMReferenceStreamFitRequest,
    GMMScoreResult,
    GMMStreamScoreRequest,
    fit_gmm_router_stream,
    score_gmm_router_stream,
    validate_gmm_router,
)
from workflow_runtime.lifecycle.progress.worker_observations import (
    WorkerObservationWriter,
    emit_gmm_score_chunks,
    observed_descriptor_chunks,
    optional_observation_writer,
)


RESULT_PREFIX = "DICE_NEP_READOUT_MEANS="
GMM_RESULT_PREFIX = "DICE_GMM_SCORE_RESULT="


def export_nep_readout_input_frame_means(
    model: str | Path,
    frames: Sequence[Atoms],
    *,
    chunk_frames: int = 16,
    runtime_factory: Callable[[str], Any] | None = None,
) -> tuple[dict[str, object], tuple[str, ...], np.ndarray]:
    schema, identities, rows, _elements = (
        export_nep_readout_input_frame_and_element_means(
            model,
            frames,
            chunk_frames=chunk_frames,
            runtime_factory=runtime_factory,
        )
    )
    return schema, identities, rows


def export_nep_readout_input_frame_and_element_means(
    model: str | Path,
    frames: Sequence[Atoms],
    *,
    chunk_frames: int = 16,
    runtime_factory: Callable[[str], Any] | None = None,
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    np.ndarray,
    dict[str, tuple[tuple[str, ...], np.ndarray]],
]:
    """Consume native atomic rows in memory and retain only ordered frame means."""
    exporter = NEPReadoutInputExporter(model, runtime_factory=runtime_factory)
    header, raw_chunks = nep_readout_input_descriptor_stream(
        frames,
        exporter=exporter,
        chunk_frames=chunk_frames,
    )
    identities: list[str] = []
    means: list[np.ndarray] = []
    element_values: dict[str, list[tuple[str, np.ndarray]]] = {
        str(element): [] for element in header.type_map
    }
    for chunk in validated_descriptor_chunks(header, raw_chunks):
        for local_index, identity in enumerate(chunk.frame_ids):
            start = int(chunk.frame_ptr[local_index])
            stop = int(chunk.frame_ptr[local_index + 1])
            if stop <= start:
                raise ValueError("NEP readout-input frame means require at least one atom per frame")
            identities.append(str(identity))
            means.append(np.asarray(chunk.descriptors[start:stop].mean(axis=0), dtype=np.float32))
            frame_types = chunk.type_indices[start:stop]
            for type_index in dict.fromkeys(int(item) for item in frame_types):
                element_values[header.type_map[type_index]].append(
                    (
                        str(identity),
                        np.asarray(
                            chunk.descriptors[start:stop][
                                frame_types == type_index
                            ].mean(axis=0),
                            dtype=np.float32,
                        ),
                    )
                )
    ordered = tuple(identities)
    expected = tuple(str(frame.info["structure_id"]) for frame in frames)
    if ordered != expected:
        raise ValueError("NEP readout-input worker changed canonical frame order")
    rows = np.ascontiguousarray(np.stack(means), dtype=np.float32)
    if rows.shape != (len(frames), int(header.schema["width"])) or not np.isfinite(rows).all():
        raise ValueError("NEP readout-input worker returned invalid frame means")
    elements = {
        element: (
            tuple(identity for identity, _row in values),
            np.ascontiguousarray(np.stack([row for _identity, row in values])),
        )
        for element, values in element_values.items()
        if values
    }
    return dict(header.schema), ordered, rows, elements


def run_gmm_reference_operation(
    *,
    model: str | Path,
    reference_path: str | Path,
    router_dir: str | Path,
    type_map: Sequence[str],
    chunk_frames: int = 16,
    spill: bool = False,
    work_dir: str | Path | None = None,
    runtime_factory: Callable[[str], Any] | None = None,
    observation_writer: WorkerObservationWriter | None = None,
) -> None:
    """Fit a portable router while the selected NEP runtime owns atomic rows."""
    model_path = _canonical_model_path(model)
    reference = Path(reference_path).resolve()
    frames = read_extxyz_structures(reference, allow_empty=False)
    _require_canonical_identities(frames, label="NEP GMM reference")
    exporter = NEPReadoutInputExporter(model_path, runtime_factory=runtime_factory)
    header, chunks = nep_readout_input_descriptor_stream(
        frames,
        exporter=exporter,
        chunk_frames=chunk_frames,
    )
    expected_type_map = tuple(str(item) for item in type_map)
    if header.type_map != expected_type_map:
        raise ValueError("NEP GMM reference model type_map does not match student type_map")
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
    chunk_frames: int = 16,
    runtime_factory: Callable[[str], Any] | None = None,
    observation_writer: WorkerObservationWriter | None = None,
) -> GMMScoreResult:
    """Replay one identity filter and consume descriptors without a full cache."""
    model_path = _canonical_model_path(model)
    reference = Path(reference_path).resolve()
    identities = tuple(str(item) for item in ordered_identities)
    if any(not item for item in identities) or len(set(identities)) != len(identities):
        raise ValueError("NEP GMM score identity filter must be unique and non-empty")
    frames = _ordered_observation_filter(Path(observation_path).resolve(), identities)
    model_digest = _file_digest(model_path)
    reference_digest = _file_digest(reference)
    if frames:
        exporter = NEPReadoutInputExporter(model_path, runtime_factory=runtime_factory)
        header, chunks = nep_readout_input_descriptor_stream(
            frames,
            exporter=exporter,
            chunk_frames=chunk_frames,
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
        chunks = ()
    observed_chunks = observed_descriptor_chunks(
        chunks,
        writer=observation_writer,
        total=header.frame_count,
        width=int(header.schema["width"]),
    )
    scored = score_gmm_router_stream(
        GMMStreamScoreRequest(
            router_dir=Path(router_dir),
            header=header,
            chunks=observed_chunks,
            model_digest=model_digest,
            reference_digest=reference_digest,
            emit_frame_means=bool(emit_frame_means),
        )
    )
    if tuple(str(item) for item in scored.frame_ids.tolist()) != identities:
        raise ValueError("NEP GMM score result does not match the ordered identity filter")
    emit_gmm_score_chunks(
        observation_writer,
        role="student",
        frame_ids=scored.frame_ids,
        p_values=scored.frame_p,
    )
    return scored


def _canonical_model_path(model: str | Path) -> Path:
    path = Path(model)
    if path.is_symlink() or path.suffix != ".txt" or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"NEP GMM requires one non-empty regular native .txt model: {path}")
    return path.resolve()


def run_gmm_geometry_score_operation(*, model, reference_path, router_dir, ordered_identities,
                                     emit_frame_means, geometry_socket, byte_budget, atom_budget,
                                     chunk_frames=16, runtime_factory=None, observation_writer=None):
    from workflow_backends.geometry_stream import consume_geometry_stream

    model_path = _canonical_model_path(model)
    identities = tuple(ordered_identities)
    exporter = NEPReadoutInputExporter(model_path, runtime_factory=runtime_factory)
    header = DescriptorStreamHeader(exporter.schema(), exporter.metadata.type_map, len(identities))

    def chunks():
        consumed = 0
        for frames in consume_geometry_stream(
            geometry_socket, ordered_identities=identities,
            byte_budget=byte_budget, atom_budget=atom_budget,
        ):
            for start in range(0, len(frames), chunk_frames):
                batch = frames[start:start + chunk_frames]
                yield exporter.export_frames(tuple(enumerate(batch, start=consumed)))
                consumed += len(batch)
        if consumed != len(identities):
            raise ValueError("NEP geometry score omitted canonical frames")

    scored = score_gmm_router_stream(GMMStreamScoreRequest(
        router_dir=Path(router_dir), header=header,
        chunks=observed_descriptor_chunks(chunks(), writer=observation_writer,
                                         total=len(identities), width=int(header.schema["width"])),
        model_digest=_file_digest(model_path), reference_digest=_file_digest(Path(reference_path)),
        emit_frame_means=bool(emit_frame_means),
    ))
    if tuple(scored.frame_ids) != identities:
        raise ValueError("NEP geometry score changed canonical order")
    emit_gmm_score_chunks(observation_writer, role="student", frame_ids=scored.frame_ids, p_values=scored.frame_p)
    return scored


def _ordered_observation_filter(path: Path, identities: tuple[str, ...]) -> list[Atoms]:
    frames = read_extxyz_structures(path, allow_empty=False)
    _require_canonical_identities(frames, label="NEP GMM observation source")
    by_identity = {str(frame.info["structure_id"]): frame for frame in frames}
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError(
            "NEP GMM score identity filter is absent from the observation source: "
            + ", ".join(missing)
        )
    return [by_identity[identity] for identity in identities]


def _require_canonical_identities(frames: Sequence[Atoms], *, label: str) -> None:
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
    parser = argparse.ArgumentParser(description="Run one streaming NEP student GMM operation.")
    parser.add_argument(
        "--operation",
        required=True,
        choices=("gmm_reference", "gmm_score", "structure_vectors"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--router")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--type-map")
    parser.add_argument("--observations")
    parser.add_argument("--identity-filter")
    parser.add_argument("--emit-frame-means", action="store_true")
    parser.add_argument("--observation-channel")
    parser.add_argument("--geometry-socket")
    parser.add_argument("--geometry-byte-budget", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--geometry-atom-budget", type=int, default=1_000_000)
    parser.add_argument("--compact-result-root")
    args = parser.parse_args(argv)
    observation_writer = optional_observation_writer(args.observation_channel)
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8"))
    chunk_frames = int(runtime.get("descriptor_chunk_frames", 16))
    if args.operation == "structure_vectors":
        if args.observations is None:
            raise ValueError("NEP structure_vectors requires observations")
        frames = read_extxyz_structures(args.observations, allow_empty=False)
        schema, identities, means, element_means = (
            export_nep_readout_input_frame_and_element_means(
            args.model, frames, chunk_frames=chunk_frames
            )
        )
        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "descriptor_schema": schema,
                    "frame_ids": list(identities),
                    "frame_means": means.tolist(),
                    "element_means": {
                        element: {
                            "frame_ids": list(values[0]),
                            "vectors": values[1].tolist(),
                        }
                        for element, values in element_means.items()
                    },
                },
                separators=(",", ":"),
            )
        )
    elif args.operation == "gmm_reference":
        if args.type_map is None:
            raise ValueError("NEP gmm_reference requires --type-map")
        if args.reference is None or args.router is None:
            raise ValueError("NEP gmm_reference requires reference and router")
        type_map = json.loads(Path(args.type_map).read_text(encoding="utf-8"))
        if not isinstance(type_map, list):
            raise ValueError("NEP gmm_reference type map must be a JSON array")
        run_gmm_reference_operation(
            model=args.model,
            reference_path=args.reference,
            router_dir=args.router,
            type_map=tuple(str(item) for item in type_map),
            chunk_frames=chunk_frames,
            spill=bool(runtime.get("gmm_reference_spill", False)),
            work_dir=Path(args.runtime).resolve().parent,
            observation_writer=observation_writer,
        )
    else:
        if (
            args.reference is None
            or args.router is None
            or (args.observations is None and args.geometry_socket is None)
            or args.identity_filter is None
        ):
            raise ValueError("NEP gmm_score requires observations and identity filter")
        identities = json.loads(Path(args.identity_filter).read_text(encoding="utf-8"))
        if not isinstance(identities, list):
            raise ValueError("NEP gmm_score identity filter must be a JSON array")
        score_operation = run_gmm_geometry_score_operation if args.geometry_socket else run_gmm_score_operation
        source_options = (
            {"geometry_socket": Path(args.geometry_socket), "byte_budget": args.geometry_byte_budget,
             "atom_budget": args.geometry_atom_budget}
            if args.geometry_socket else {"observation_path": args.observations}
        )
        scored = score_operation(
            model=args.model,
            reference_path=args.reference,
            **source_options,
            router_dir=args.router,
            ordered_identities=tuple(str(item) for item in identities),
            emit_frame_means=bool(args.emit_frame_means),
            chunk_frames=chunk_frames,
            observation_writer=observation_writer,
        )
        if args.compact_result_root:
            from workflow_backends.compact_score_store import write_compact_score_store
            receipt = write_compact_score_store(
                args.compact_result_root, scored, model_digest=_file_digest(Path(args.model)),
                reference_digest=_file_digest(Path(args.reference)),
            )
            print(GMM_RESULT_PREFIX + json.dumps(receipt, separators=(",", ":")))
            return 0
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
        print(GMM_RESULT_PREFIX + json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = [
    "GMM_RESULT_PREFIX",
    "RESULT_PREFIX",
    "export_nep_readout_input_frame_means",
    "main",
    "run_gmm_reference_operation",
    "run_gmm_score_operation",
]
