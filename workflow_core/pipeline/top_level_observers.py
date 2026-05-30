from __future__ import annotations

from typing import Callable

from workflow_runtime.paths.path_manager import PathManager


def _count_extxyz_frames(path) -> int | None:
    try:
        from workflow_core.structures.formats import read_extxyz_structures

        return len(read_extxyz_structures(path))
    except Exception:
        return None


def build_top_level_step_observer(step_name: str, paths: PathManager, round_id: int) -> Callable[[], None] | None:
    if step_name == "first_principles":
        return lambda: _count_extxyz_frames(paths.first_principles_labeled_xyz(round_id))
    return None
