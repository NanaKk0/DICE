from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow_artifacts.gmm_worker_evidence import (
    gmm_frame_score_result_payload,
    normalize_gmm_score_chunks,
)
from workflow_backends.teacher.mace.descriptors import (
    MaceNodeFeatureWorkerPool,
    mace_readout_input_descriptor_chunk_stream,
    mace_readout_input_descriptor_stream,
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
    parser = argparse.ArgumentParser(description="Run MACE readout-input GMM extraction and policy consumption")
    parser.add_argument("--operation", required=True, choices=("reference", "score"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--input")
    parser.add_argument("--chunk-manifest")
    parser.add_argument("--router-dir")
    parser.add_argument("--type-map-json")
    parser.add_argument("--identity-filter")
    parser.add_argument("--model-digest")
    parser.add_argument("--reference-digest")
    parser.add_argument("--emit-frame-means", action="store_true")
    parser.add_argument("--spill-work-dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--observation-channel")
    args = parser.parse_args(argv)
    observation_writer = optional_observation_writer(args.observation_channel)
    if args.workers <= 0 or args.batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("MACE descriptor workers, batch size, and torch threads must be positive")
    router_dir = _required_path(args.router_dir, "--router-dir")
    model_digest = _required_text(args.model_digest, "--model-digest")
    reference_digest = _required_text(args.reference_digest, "--reference-digest")
    if args.operation == "reference":
        if args.chunk_manifest is not None:
            raise ValueError("MACE GMM reference does not accept a score chunk manifest")
        frames = read_extxyz_structures(_required_path(args.input, "--input"), allow_empty=True)
        if not frames:
            raise ValueError("MACE readout-input reference fit requires non-empty canonical input")
        type_map = _type_map_from_json(args.type_map_json)
        for ordinal, frame in enumerate(frames):
            if not str(frame.info.get("structure_id", "")):
                frame.info["structure_id"] = f"reference-frame:{ordinal:08d}"
    else:
        type_map = _router_type_map(router_dir)
        if args.chunk_manifest is not None:
            if args.input is not None or args.identity_filter is not None:
                raise ValueError("MACE GMM chunked score does not accept --input or --identity-filter")
            frame_chunks, score_frame_count = _chunk_manifest_frames(args.chunk_manifest)
        else:
            frames = read_extxyz_structures(_required_path(args.input, "--input"), allow_empty=True)
            frames = _ordered_identity_filter(frames, args.identity_filter)
            frame_chunks = (frames,)
            score_frame_count = len(frames)

    with _worker_pool(args) as exporter:
        if args.operation == "reference":
            header, chunks = mace_readout_input_descriptor_stream(
                frames,
                type_map=type_map,
                exporter=exporter,
            )
            if observation_writer is not None:
                observation_writer.descriptor_progress(completed=0, total=header.frame_count, width=int(header.schema["width"]))
            result = fit_gmm_router_stream(
                GMMReferenceStreamFitRequest(
                    router_dir=router_dir,
                    header=header,
                    chunks=observed_descriptor_chunks(chunks, writer=observation_writer, total=header.frame_count, width=int(header.schema["width"])),
                    model_digest=model_digest,
                    reference_digest=reference_digest,
                    spill=args.spill_work_dir is not None,
                    work_dir=None if args.spill_work_dir is None else Path(args.spill_work_dir),
                )
            )
            if observation_writer is not None:
                observation_writer.descriptor_progress(completed=header.frame_count, total=header.frame_count, width=int(header.schema["width"]), elements_completed=len(result.element_pca_dimensions), elements_total=len(result.element_pca_dimensions))
        else:
            header, chunks = mace_readout_input_descriptor_chunk_stream(
                frame_chunks,
                frame_count=score_frame_count,
                type_map=type_map,
                exporter=exporter,
            )
            result = score_gmm_router_stream(
                GMMStreamScoreRequest(
                    router_dir=router_dir,
                    header=header,
                    chunks=observed_descriptor_chunks(chunks, writer=observation_writer, total=header.frame_count, width=int(header.schema["width"])),
                    model_digest=model_digest,
                    reference_digest=reference_digest,
                    emit_frame_means=bool(args.emit_frame_means),
                )
            )
            emit_gmm_score_chunks(observation_writer, role="teacher", frame_ids=result.frame_ids, p_values=result.frame_p)
            payload = gmm_frame_score_result_payload(
                frame_ids=result.frame_ids.tolist(),
                frame_p=result.frame_p.tolist(),
                frame_statistics=result.frame_statistics.tolist(),
                frame_means=None if result.frame_means is None else result.frame_means.tolist(),
                descriptor_schema=dict(result.descriptor_schema or header.schema),
            )
            print("DICE_GMM_RESULT=" + json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return 0


def _worker_pool(args: argparse.Namespace) -> MaceNodeFeatureWorkerPool:
    return MaceNodeFeatureWorkerPool(
        args.model,
        workers=args.workers,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        device="cuda",
        default_dtype="float32",
    )


def _required_path(value: str | None, option: str) -> Path:
    text = _required_text(value, option)
    return Path(text).resolve()


def _required_text(value: str | None, option: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MACE GMM operation requires {option}")
    return value.strip()


def _type_map_from_json(raw: str | None) -> tuple[str, ...]:
    try:
        values = json.loads(_required_text(raw, "--type-map-json"))
    except json.JSONDecodeError as exc:
        raise ValueError("MACE GMM reference type map must be a JSON list") from exc
    if not isinstance(values, list):
        raise ValueError("MACE GMM reference type map must be a JSON list")
    return tuple(str(item) for item in values)


def _router_type_map(router_dir: Path) -> tuple[str, ...]:
    try:
        manifest = json.loads((router_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("MACE GMM score requires a readable durable router manifest") from exc
    values = manifest.get("type_map")
    if not isinstance(values, list):
        raise ValueError("MACE GMM score router has an invalid type_map")
    return tuple(str(item) for item in values)


def _ordered_identity_filter(frames: list[object], filter_path: str | None) -> list[object]:
    if filter_path is None:
        return frames
    try:
        raw = json.loads(Path(filter_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("MACE GMM score identity filter must be readable JSON") from exc
    if (
        not isinstance(raw, list)
        or any(not isinstance(item, str) or not item for item in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError("MACE GMM score identity filter must contain unique canonical identities")
    return _ordered_frames(frames, tuple(raw))


def _ordered_frames(frames: list[object], identities: tuple[str, ...]) -> list[object]:
    by_identity: dict[str, object] = {}
    for frame in frames:
        identity = str(frame.info.get("structure_id", ""))
        if identity and identity not in by_identity:
            by_identity[identity] = frame
    missing = [identity for identity in identities if identity not in by_identity]
    if missing:
        raise ValueError(f"MACE GMM score identity filter is missing source frames: {missing}")
    return [by_identity[identity] for identity in identities]


def _chunk_manifest_frames(
    manifest_path: str,
) -> tuple[object, int]:
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("MACE GMM score chunk manifest must be readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("MACE GMM score chunk manifest has an invalid schema")
    payload = normalize_gmm_score_chunks(payload)
    chunks = payload.get("chunks")
    frame_count = payload.get("frame_count")
    if (
        not isinstance(frame_count, int)
        or frame_count <= 0
        or not isinstance(chunks, list)
        or not chunks
    ):
        raise ValueError("MACE GMM score chunk manifest has an invalid schema")

    parsed: list[tuple[Path, tuple[str, ...]]] = []
    flattened: list[str] = []
    for item in chunks:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("MACE GMM score chunk manifest contains an invalid path")
        raw_ids = item.get("frame_ids")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(identity, str) or not identity for identity in raw_ids)
        ):
            raise ValueError("MACE GMM score chunk manifest contains invalid frame identities")
        identities = tuple(raw_ids)
        parsed.append((Path(item["path"]).resolve(), identities))
        flattened.extend(identities)
    if len(flattened) != frame_count or len(set(flattened)) != frame_count:
        raise ValueError("MACE GMM score chunk manifest changed canonical frame membership")

    def replay_chunks():
        for path, identities in parsed:
            frames = read_extxyz_structures(path, allow_empty=True)
            if len(frames) != len(identities):
                raise ValueError("MACE GMM score chunk size does not match its identity list")
            yield _ordered_frames(frames, identities)

    return replay_chunks(), frame_count


if __name__ == "__main__":
    raise SystemExit(main())
