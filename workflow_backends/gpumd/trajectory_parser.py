from __future__ import annotations

from pathlib import Path

from workflow_core.structures.formats import read_extxyz_structures


def read_gpumd_movie_frames(path: str | Path):
    return read_extxyz_structures(path, allow_empty=True)
