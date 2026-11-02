"""Consume DPA4 fitting-last streams inside one native GMM operation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from workflow_artifacts.gmm_worker_evidence import normalize_gmm_score_chunks
from workflow_backends.teacher.deepmd.descriptors import (
    DPA4ReadoutInputExporter,
    dpa4_readout_input_descriptor_chunk_stream,
    dpa4_readout_input_descriptor_stream,
)
from workflow_core.structures.formats import read_extxyz_structures
from workflow_artifacts.lifecycle.gmm_io import (
    GMMReferenceStreamFitRequest,
    GMMStreamScoreRequest,
    fit_gmm_router_stream,
    score_gmm_router_stream,
)
from workflow_runtime.lifecycle.progress.worker_observations import (
    emit_gmm_score_chunks,
    observed_descriptor_chunks,
    optional_observation_writer,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DPA4 readout-input GMM extraction")
    parser.add_argument(
        "--operation", required=True, choices=("reference", "score", "score_stream")
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint")
    source_group.add_argument("--terminal")
    parser.add_argument("--input")
    parser.add_argument("--chunk-manifest")
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--type-map-json")
    parser.add_argument("--identity-filter")
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--reference-digest", required=True)
    parser.add_argument("--emit-frame-means", action="store_true")
    parser.add_argument("--spill-work-dir")
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=9472)
    parser.add_argument("--geometry-socket")
    parser.add_argument("--geometry-byte-budget", type=int)
    parser.add_argument("--geometry-atom-budget", type=int)
    parser.add_argument("--compact-result-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--observation-channel")
    args = parser.parse_args(argv)
    observation_writer = optional_observation_writer(args.observation_channel)
    if args.chunk_frames <= 0:
        raise ValueError("DPA4 descriptor chunk frames must be positive")
    router_dir = Path(args.router_dir).resolve()
    if args.operation == "reference":
        if args.chunk_manifest is not None:
            raise ValueError("DPA4 GMM reference does not accept a score chunk manifest")
        frames = read_extxyz_structures(
            _required_path(args.input, "--input"), allow_empty=True
        )
        if not frames:
            raise ValueError("DPA4 readout-input reference fit requires non-empty canonical input")
        type_map = _type_map_from_json(args.type_map_json)
        for ordinal, frame in enumerate(frames):
            if not str(frame.info.get("structure_id", "")):
                frame.info["structure_id"] = f"reference-frame:{ordinal:08d}"
    elif args.operation == "score_stream":
        type_map = _router_type_map(router_dir)
        if any(
            value is None
            for value in (
                args.geometry_socket,
                args.geometry_byte_budget,
                args.geometry_atom_budget,
                args.identity_filter,
                args.compact_result_root,
            )
        ):
            raise ValueError("DPA4 streamed score requires geometry, identity, and result inputs")
        identities = json.loads(Path(args.identity_filter).read_text(encoding="utf-8"))
        if not isinstance(identities, list) or any(not isinstance(item, str) or not item for item in identities):
            raise ValueError("DPA4 streamed score identities must be a JSON array")
        from workflow_backends.geometry_stream import consume_geometry_stream

        frame_chunks = consume_geometry_stream(
            args.geometry_socket,
            ordered_identities=tuple(identities),
            byte_budget=args.geometry_byte_budget,
            atom_budget=args.geometry_atom_budget,
        )
        score_frame_count = len(identities)
    else:
        type_map = _router_type_map(router_dir)
        if args.chunk_manifest is not None:
            if args.input is not None or args.identity_filter is not None:
                raise ValueError(
                    "DPA4 GMM chunked score does not accept --input or --identity-filter"
                )
            frame_chunks, score_frame_count = _chunk_manifest_frames(
                args.chunk_manifest
            )
        else:
            frames = read_extxyz_structures(
                _required_path(args.input, "--input"), allow_empty=True
            )
            frames = _ordered_identity_filter(frames, args.identity_filter)
            frame_chunks = (frames,)
            score_frame_count = len(frames)

    model_load_started = time.perf_counter()
    exporter = DPA4ReadoutInputExporter(
        args.checkpoint or args.terminal,
        device=args.device,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    if args.operation == "reference":
        header, chunks = dpa4_readout_input_descriptor_stream(
            frames,
            exporter=exporter,
            type_map=type_map,
            chunk_frames=args.chunk_frames,
            batch_size=args.batch_size,
        )
    else:
        header, chunks = dpa4_readout_input_descriptor_chunk_stream(
            frame_chunks,
            frame_count=score_frame_count,
            exporter=exporter,
            type_map=type_map,
            chunk_frames=args.chunk_frames,
            batch_size=args.batch_size,
        )
        if args.operation == "score_stream":
            from workflow_backends.deepmd_pt_runtime import bounded_prefetch

            chunks = bounded_prefetch(chunks, slots=2)
    if args.operation == "reference":
        result = fit_gmm_router_stream(GMMReferenceStreamFitRequest(
            router_dir=router_dir,
            header=header,
            chunks=observed_descriptor_chunks(chunks, writer=observation_writer, total=header.frame_count, width=int(header.schema["width"])),
            model_digest=args.model_digest,
            reference_digest=args.reference_digest,
            spill=args.spill_work_dir is not None,
            work_dir=None if args.spill_work_dir is None else Path(args.spill_work_dir),
        ))
        if observation_writer is not None:
            observation_writer.descriptor_progress(completed=header.frame_count, total=header.frame_count, width=int(header.schema["width"]), elements_completed=len(result.element_pca_dimensions), elements_total=len(result.element_pca_dimensions))
    else:
        result = score_gmm_router_stream(GMMStreamScoreRequest(
            router_dir=router_dir,
            header=header,
            chunks=observed_descriptor_chunks(chunks, writer=observation_writer, total=header.frame_count, width=int(header.schema["width"])),
            model_digest=args.model_digest,
            reference_digest=args.reference_digest,
            emit_frame_means=bool(args.emit_frame_means),
        ))
        emit_gmm_score_chunks(observation_writer, role="teacher", frame_ids=result.frame_ids, p_values=result.frame_p)
        if args.operation == "score_stream":
            from workflow_backends.compact_score_store import write_compact_score_store
            from workflow_backends.deepmd_pt_runtime import pop_runtime_diagnostics

            payload = write_compact_score_store(
                args.compact_result_root,
                result,
                model_digest=args.model_digest,
                reference_digest=args.reference_digest,
                runtime_diagnostics=pop_runtime_diagnostics(),
            )
            payload["operation_evidence"] = {
                "operation_role": "teacher.deepmd.pt.gmm_score",
                "worker_count": 1,
                "model_load_count": 1,
                "model_load_seconds": model_load_seconds,
            }
        else:
            payload = {
                "frame_ids": result.frame_ids.tolist(),
                "frame_p": result.frame_p.tolist(),
                "frame_statistics": result.frame_statistics.tolist(),
                "frame_means": None if result.frame_means is None else result.frame_means.tolist(),
                "descriptor_schema": dict(result.descriptor_schema or header.schema),
            }
        print("DICE_DPA4_GMM_RESULT=" + json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        ))
    return 0


def _type_map_from_json(raw: str | None) -> tuple[str, ...]:
    try:
        values = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ValueError("DPA4 GMM reference type map must be a JSON list") from exc
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("DPA4 GMM reference type map must contain unique elements")
    return tuple(values)


def _router_type_map(router_dir: Path) -> tuple[str, ...]:
    try:
        manifest = json.loads((router_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("DPA4 GMM score requires a readable router manifest") from exc
    values = manifest.get("type_map")
    if not isinstance(values, list) or not values:
        raise ValueError("DPA4 GMM score router has an invalid type_map")
    return tuple(str(item) for item in values)


def _ordered_identity_filter(frames: list[object], filter_path: str | None) -> list[object]:
    if filter_path is None:
        return frames
    try:
        identities = json.loads(Path(filter_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("DPA4 GMM score identity filter must be readable JSON") from exc
    if (
        not isinstance(identities, list)
        or any(not isinstance(item, str) or not item for item in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError("DPA4 GMM score identity filter must contain unique identities")
    by_identity = {
        str(frame.info.get("structure_id", "")): frame
        for frame in frames
        if str(frame.info.get("structure_id", ""))
    }
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError(f"DPA4 GMM score identity filter is missing frames: {missing}")
    return [by_identity[identity] for identity in identities]


def _required_path(value: str | None, option: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DPA4 GMM operation requires {option}")
    return Path(value).resolve()


def _chunk_manifest_frames(manifest_path: str) -> tuple[object, int]:
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("DPA4 GMM score chunk manifest must be readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("DPA4 GMM score chunk manifest has an invalid schema")
    payload = normalize_gmm_score_chunks(payload)
    chunks = payload.get("chunks")
    frame_count = payload.get("frame_count")
    if (
        not isinstance(frame_count, int)
        or frame_count <= 0
        or not isinstance(chunks, list)
        or not chunks
    ):
        raise ValueError("DPA4 GMM score chunk manifest has an invalid schema")

    parsed: list[tuple[Path, tuple[str, ...]]] = []
    flattened: list[str] = []
    for item in chunks:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("DPA4 GMM score chunk manifest contains an invalid path")
        raw_ids = item.get("frame_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(identity, str) or not identity for identity in raw_ids)
        ):
            raise ValueError(
                "DPA4 GMM score chunk manifest contains invalid frame identities"
            )
        identities = tuple(raw_ids)
        parsed.append((Path(item["path"]).resolve(), identities))
        flattened.extend(identities)
    if len(flattened) != frame_count or len(set(flattened)) != frame_count:
        raise ValueError("DPA4 GMM score chunk manifest changed canonical membership")

    def replay_chunks():
        for path, identities in parsed:
            frames = read_extxyz_structures(path, allow_empty=True)
            if len(frames) != len(identities):
                raise ValueError(
                    "DPA4 GMM score chunk size does not match its identity list"
                )
            yield _ordered_frames(frames, identities)

    return replay_chunks(), frame_count


def _ordered_frames(frames: list[object], identities: tuple[str, ...]) -> list[object]:
    by_identity = {
        str(frame.info.get("structure_id", "")): frame
        for frame in frames
        if str(frame.info.get("structure_id", ""))
    }
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError(f"DPA4 GMM score chunk is missing frames: {missing}")
    return [by_identity[identity] for identity in identities]


if __name__ == "__main__":
    raise SystemExit(main())
