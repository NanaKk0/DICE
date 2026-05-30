from __future__ import annotations

from pathlib import Path
from typing import Any


def write_deepmd_dataset(atoms_list: list[Any], output_dir: str | Path, element_order: list[str]) -> Path:
    from workflow_core.labeling.export import xyz_to_deepmd

    resolved = Path(output_dir).resolve()
    xyz_to_deepmd(atoms_list, resolved, [str(item) for item in element_order])
    return resolved
