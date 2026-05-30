from __future__ import annotations

from pathlib import Path

from ase import Atoms

from workflow_core.common.io import load_atoms_list, write_candidate_xyz
from workflow_core.seed_pool.metadata import (
    ensure_minimal_seed_metadata,
    sanitize_system_id,
)


def load_system_extxyz_dir(root_dir: str | Path) -> dict[str, list[Atoms]]:
    root = Path(root_dir).resolve()
    grouped: dict[str, list[Atoms]] = {}
    if not root.exists():
        return grouped
    for path in sorted(root.glob("*.extxyz")):
        if not path.exists() or path.stat().st_size == 0:
            grouped[path.stem] = []
            continue
        atoms_list = [ensure_minimal_seed_metadata(item) for item in load_atoms_list(path)]
        system_id = str(atoms_list[0].info["system_id"]) if atoms_list else path.stem
        grouped[system_id] = atoms_list
    return grouped


def write_system_extxyz_dir(root_dir: str | Path, grouped_atoms: dict[str, list[Atoms]]) -> dict[str, Path]:
    root = Path(root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for existing in root.glob("*.extxyz"):
        existing.unlink()
    for system_id in sorted(grouped_atoms):
        path = root / f"{sanitize_system_id(system_id)}.extxyz"
        atoms_list = [ensure_minimal_seed_metadata(item, system_id=system_id) for item in grouped_atoms[system_id]]
        write_candidate_xyz(path, atoms_list)
        written[system_id] = path.resolve()
    return written


class SeedPoolRepository:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("formal seed-pool repository is retired; use student pipeline live seed-pool artifacts")
