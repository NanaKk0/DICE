from __future__ import annotations

import hashlib
import random


ALGORITHM = "sha256-python-random-v1"


def select_frame_indices(
    *,
    source_sha256: str,
    frame_count: int,
    max_frames: int,
    iter_index: int,
    round_index: int,
    sampling_role_id: str,
) -> tuple[tuple[int, ...], str]:
    """Return one stable, round-local sample without replacement."""

    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("seed source SHA-256 must be 64 lowercase hex characters")
    if isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError("seed source frame count must be positive")
    if isinstance(max_frames, bool) or max_frames <= 0:
        raise ValueError("seed source max_frames must be positive")
    if isinstance(iter_index, bool) or iter_index < 0:
        raise ValueError("sampling iter index must be non-negative")
    if isinstance(round_index, bool) or round_index < 0:
        raise ValueError("sampling round index must be non-negative")
    if not sampling_role_id:
        raise ValueError("sampling role id must be non-empty")

    material = "\0".join(
        (
            ALGORITHM,
            source_sha256,
            str(iter_index),
            str(round_index),
            sampling_role_id,
            str(max_frames),
        )
    ).encode("utf-8")
    seed_hex = hashlib.sha256(material).hexdigest()
    count = min(frame_count, max_frames)
    indices = tuple(
        sorted(random.Random(int(seed_hex, 16)).sample(range(frame_count), count))
    )
    return indices, seed_hex


__all__ = ["ALGORITHM", "select_frame_indices"]
