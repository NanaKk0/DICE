from __future__ import annotations

from pathlib import Path

from workflow_backends.common.model_files import link_model_file
from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures


def write_task_model_xyz(seed_path: str | Path, output_path: str | Path) -> Path:
    atoms = read_extxyz_structures(seed_path, allow_empty=False)[0]
    return write_extxyz_structures(output_path, [atoms], allow_empty=False).resolve()


def link_committee_models(model_paths: list[str], task_dir: str | Path) -> list[Path]:
    task_dir = Path(task_dir).resolve()
    linked: list[Path] = []
    for index, model_path in enumerate(model_paths):
        target = task_dir / f"nep_{index}.txt"
        linked.append(link_model_file(model_path, target))
    return linked


def copy_committee_models(model_paths: list[str], task_dir: str | Path) -> list[Path]:
    return link_committee_models(model_paths, task_dir)
