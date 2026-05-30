from pathlib import Path
from typing import Any

from ase.io import read, write


def split_dft_xyz(xyz: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Split an extxyz file into numbered structure.xyz task directories."""
    output_root = Path(out_dir)
    atoms_list = read(str(xyz), ":")
    output_root.mkdir(parents=True, exist_ok=True)
    for index, atoms in enumerate(atoms_list):
        task_dir = output_root / str(index)
        task_dir.mkdir(exist_ok=True)
        write(task_dir / "structure.xyz", atoms)
    return {"structure_count": len(atoms_list), "output_dir": str(output_root.resolve())}
