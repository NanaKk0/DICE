from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.structures.formats import write_extxyz_structures


def validated_observation_chunk_size(value: int) -> int:
    size = int(value)
    if size <= 0:
        raise ValueError("lifecycle observation replay chunk size must be positive")
    return size


def replay_with_metadata(
    source: Any,
    frames: list[Any],
    *,
    chunk_size: int,
) -> list[Any]:
    identities = tuple(str(frame.info["structure_id"]) for frame in frames)
    metadata = {str(frame.info["structure_id"]): dict(frame.info) for frame in frames}
    result = []
    for chunk in source.replay(
        identities,
        chunk_size=validated_observation_chunk_size(chunk_size),
    ):
        for frame in chunk:
            identity = str(frame.info.get("structure_id", ""))
            if identity not in metadata:
                raise ValueError(
                    "lifecycle observation replay returned an unexpected identity"
                )
            frame.info.update(metadata[identity])
            result.append(frame)
    if tuple(str(frame.info["structure_id"]) for frame in result) != identities:
        raise ValueError(
            "lifecycle observation replay identities do not match requested order"
        )
    return result


def materialize_gmm_score_chunks(
    *,
    source: Any,
    frames: list[Any],
    chunks_root: Path,
    chunk_size: int,
) -> tuple[tuple[Path, ...], tuple[tuple[str, ...], ...]]:
    size = validated_observation_chunk_size(chunk_size)
    paths: list[Path] = []
    identities_by_chunk: list[tuple[str, ...]] = []
    for start in range(0, len(frames), size):
        chunk_metadata = frames[start : start + size]
        chunk = replay_with_metadata(source, chunk_metadata, chunk_size=size)
        chunk_root = chunks_root / f"chunk_{start // size:06d}"
        chunk_root.mkdir(parents=True, exist_ok=True)
        paths.append(
            write_extxyz_structures(
                chunk_root / "candidates.xyz", chunk, allow_empty=False
            )
        )
        identities_by_chunk.append(
            tuple(str(frame.info["structure_id"]) for frame in chunk)
        )
    return tuple(paths), tuple(identities_by_chunk)


__all__ = [
    "materialize_gmm_score_chunks",
    "replay_with_metadata",
    "validated_observation_chunk_size",
]
