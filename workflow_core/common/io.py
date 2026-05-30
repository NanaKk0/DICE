from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_atoms_list(path: str | Path) -> list[Any]:
    from ase.io import read

    return list(read(str(path), index=":"))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_candidate_xyz(path: str | Path, atoms_list: list[Any]) -> Path:
    from workflow_core.structures.formats import write_extxyz_structures

    return write_extxyz_structures(path, atoms_list, allow_empty=True)


def append_candidate_xyz(path: str | Path, atoms_list: list[Any]) -> Path:
    from ase.io import write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(atoms_list) == 0:
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return path
    write(path, atoms_list, format="extxyz", append=path.exists() and path.stat().st_size > 0)
    return path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
