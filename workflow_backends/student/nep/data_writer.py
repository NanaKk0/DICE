from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from workflow_core.common.io import write_candidate_xyz
from workflow_core.structures.formats import (
    active_extxyz_to_canonical_extxyz,
    active_extxyz_to_canonical_structures,
    read_extxyz_structures,
    write_extxyz_structures,
)


def read_extxyz_frames(path: str | Path) -> list[Any]:
    return read_extxyz_structures(path, allow_empty=True)


def split_canonical_extxyz_to_train_test(
    dataset_path: str | Path,
    train_xyz_path: str | Path,
    test_xyz_path: str | Path,
    *,
    train_fraction: float,
    random_seed: int,
) -> dict[str, Any]:
    frames = read_extxyz_frames(dataset_path)
    if not frames:
        raise RuntimeError(f"No structures found in dataset: {dataset_path}")
    indices = list(range(len(frames)))
    rng = random.Random(int(random_seed))
    rng.shuffle(indices)
    train_count = max(1, min(len(frames), int(round(len(frames) * float(train_fraction)))))
    if len(frames) > 1:
        train_count = min(train_count, len(frames) - 1)
    train_indices = set(indices[:train_count])
    train_frames = [_to_nep_training_frame(frames[idx]) for idx in range(len(frames)) if idx in train_indices]
    test_frames = [_to_nep_training_frame(frames[idx]) for idx in range(len(frames)) if idx not in train_indices]
    if not test_frames:
        test_frames = [_copy_atoms_with_singlepoint_results(train_frames[-1])]
    write_candidate_xyz(train_xyz_path, train_frames)
    write_candidate_xyz(test_xyz_path, test_frames)
    return {
        "total_count": len(frames),
        "train_count": len(train_frames),
        "test_count": len(test_frames),
    }


def _to_nep_training_frame(atoms):
    item = _copy_atoms_with_singlepoint_results(atoms)
    for key in ("stress", "virial"):
        item.info.pop(key, None)
    return item


def _copy_atoms_with_singlepoint_results(atoms):
    from ase.calculators.singlepoint import SinglePointCalculator

    item = atoms.copy()
    results = {}
    try:
        results["energy"] = float(atoms.get_potential_energy())
    except Exception:
        if atoms.info.get("energy") is not None:
            results["energy"] = float(atoms.info["energy"])
    try:
        results["forces"] = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        if atoms.arrays.get("forces") is not None:
            results["forces"] = np.asarray(atoms.arrays["forces"], dtype=float)
    if results:
        item.calc = SinglePointCalculator(item, **results)
    return item


def write_prediction_extxyz(
    dataset_path: str | Path,
    output_path: str | Path,
) -> Path:
    frames = read_extxyz_frames(dataset_path)
    prepared = []
    for atoms in frames:
        item = atoms.copy()
        if "energy" not in item.info:
            item.info["energy"] = 0.0
        if "forces" not in item.arrays:
            item.arrays["forces"] = np.zeros((len(item), 3), dtype=float)
        prepared.append(item)
    write_candidate_xyz(output_path, prepared)
    return Path(output_path).resolve()


def write_seed_atoms_to_model_xyz(atoms, output_path: str | Path) -> Path:
    return write_extxyz_structures(output_path, [atoms], allow_empty=False).resolve()


def write_seed_extxyz_to_model_xyz(seed_path: str | Path, output_path: str | Path) -> Path:
    atoms = read_extxyz_structures(seed_path, allow_empty=False)[0]
    return write_seed_atoms_to_model_xyz(atoms, output_path)


def active_xyz_to_canonical_atoms(
    active_xyz_path: str | Path,
    *,
    force_std_values: list[float] | None = None,
    routes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Any]:
    return active_extxyz_to_canonical_structures(
        active_xyz_path,
        force_std_values=force_std_values,
        routes=routes,
        metadata=metadata,
    )


def active_xyz_to_canonical_extxyz(
    active_xyz_path: str | Path,
    output_path: str | Path,
    *,
    force_std_values: list[float] | None = None,
    routes: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    return active_extxyz_to_canonical_extxyz(
        active_xyz_path,
        output_path,
        force_std_values=force_std_values,
        routes=routes,
        metadata=metadata,
    )


def movie_xyz_to_canonical_extxyz(movie_xyz_path: str | Path, output_path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
    return active_extxyz_to_canonical_extxyz(movie_xyz_path, output_path, metadata=metadata)
