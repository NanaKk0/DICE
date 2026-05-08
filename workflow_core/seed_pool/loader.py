from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read


@dataclass
class SeedRecord:
    seed_id: int
    atoms: Atoms
    source_file: str
    source_frame: int
    source_round: int | None
    layer_key: str | None


def load_seed_records(seed_xyz: str | Path) -> list[SeedRecord]:
    atoms_list = read(str(seed_xyz), index=":")
    records: list[SeedRecord] = []

    for index, atoms in enumerate(atoms_list):
        info = atoms.info
        seed_id = int(info.get("seed_id", index))
        source_round = info.get("source_round")
        if source_round == -1:
            source_round = None

        records.append(
            SeedRecord(
                seed_id=seed_id,
                atoms=atoms,
                source_file=str(info.get("source_file", "")),
                source_frame=int(info.get("source_frame", index)),
                source_round=None if source_round is None else int(source_round),
                layer_key=info.get("layer_key"),
            )
        )

    return records
