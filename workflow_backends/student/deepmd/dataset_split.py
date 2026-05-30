from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_ARRAY_FILES = ("coord.npy", "force.npy", "energy.npy", "box.npy")
OPTIONAL_ARRAY_FILES = ("virial.npy",)


@dataclass(slots=True)
class SystemArrays:
    coord: np.ndarray
    force: np.ndarray
    energy: np.ndarray
    box: np.ndarray
    virial: np.ndarray | None = None

    @property
    def nframes(self) -> int:
        return int(self.energy.shape[0])


def is_deepmd_system_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "type.raw").exists():
        return False
    set_dirs = [item for item in path.iterdir() if item.is_dir() and item.name.startswith("set.")]
    if not set_dirs:
        return False
    first_set = sorted(set_dirs)[0]
    return all((first_set / name).exists() for name in REQUIRED_ARRAY_FILES)


def discover_deepmd_systems(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root).resolve()
    systems: list[Path] = []

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_dir():
            continue
        if any(part in {"train", "val", "valid"} for part in path.parts):
            continue
        if is_deepmd_system_dir(path):
            systems.append(path.resolve())

    return systems


def _load_system_arrays(system_dir: Path) -> SystemArrays:
    set_dir = sorted(item for item in system_dir.iterdir() if item.is_dir() and item.name.startswith("set."))[0]
    return SystemArrays(
        coord=np.load(set_dir / "coord.npy"),
        force=np.load(set_dir / "force.npy"),
        energy=np.load(set_dir / "energy.npy"),
        box=np.load(set_dir / "box.npy"),
        virial=np.load(set_dir / "virial.npy") if (set_dir / "virial.npy").exists() else None,
    )


def _write_system_subset(
    src_system_dir: Path,
    dest_system_dir: Path,
    frame_indices: np.ndarray,
) -> None:
    arrays = _load_system_arrays(src_system_dir)
    frame_indices = np.asarray(frame_indices, dtype=int)

    if frame_indices.size == 0:
        raise ValueError(f"Cannot write empty subset for system {src_system_dir}")

    dest_set_dir = dest_system_dir / "set.000"
    dest_set_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_system_dir / "type.raw", dest_system_dir / "type.raw")
    type_map = src_system_dir / "type_map.raw"
    if type_map.exists():
        shutil.copy2(type_map, dest_system_dir / "type_map.raw")

    np.save(dest_set_dir / "coord.npy", arrays.coord[frame_indices])
    np.save(dest_set_dir / "force.npy", arrays.force[frame_indices])
    np.save(dest_set_dir / "energy.npy", arrays.energy[frame_indices])
    np.save(dest_set_dir / "box.npy", arrays.box[frame_indices])
    if arrays.virial is not None:
        np.save(dest_set_dir / "virial.npy", arrays.virial[frame_indices])


def _collect_systems_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path.resolve() for path in sorted(root.rglob("*")) if is_deepmd_system_dir(path)]


def make_deepmd_train_valid_split(
    dataset_root: str | Path,
    split_root: str | Path,
    valid_fraction: float,
    split_seed: int,
    mode: str,
    overwrite: bool = False,
) -> tuple[list[str], list[str]]:
    dataset_root = Path(dataset_root).resolve()
    split_root = Path(split_root).resolve()
    train_root = split_root / "train"
    valid_root = split_root / "val"
    meta_path = split_root / "split_meta.json"

    if mode != "frame_within_system":
        raise ValueError(f"Unsupported split mode: {mode}")

    if overwrite and split_root.exists():
        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)

    systems = discover_deepmd_systems(dataset_root)
    if not systems:
        raise RuntimeError(f"No DeepMD systems found under {dataset_root}")

    rng = np.random.default_rng(int(split_seed))
    metadata: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "split_root": str(split_root),
        "valid_fraction": float(valid_fraction),
        "split_seed": int(split_seed),
        "mode": mode,
        "systems": [],
    }

    train_systems: list[str] = []
    valid_systems: list[str] = []

    for system_dir in systems:
        arrays = _load_system_arrays(system_dir)
        nframes = arrays.nframes
        if nframes < 2:
            raise RuntimeError(f"System has fewer than 2 frames and cannot be split: {system_dir}")

        indices = np.arange(nframes, dtype=int)
        rng.shuffle(indices)

        n_valid = max(1, int(round(nframes * valid_fraction)))
        n_valid = min(n_valid, nframes - 1)
        valid_idx = np.sort(indices[:n_valid])
        train_idx = np.sort(indices[n_valid:])

        rel_path = system_dir.relative_to(dataset_root)
        train_system_dir = (train_root / rel_path).resolve()
        valid_system_dir = (valid_root / rel_path).resolve()

        _write_system_subset(system_dir, train_system_dir, train_idx)
        _write_system_subset(system_dir, valid_system_dir, valid_idx)

        train_systems.append(str(train_system_dir))
        valid_systems.append(str(valid_system_dir))
        metadata["systems"].append(
            {
                "source_system": str(system_dir),
                "relative_path": str(rel_path),
                "nframes_total": int(nframes),
                "nframes_train": int(train_idx.size),
                "nframes_valid": int(valid_idx.size),
                "train_system": str(train_system_dir),
                "valid_system": str(valid_system_dir),
            }
        )

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return train_systems, valid_systems


def collect_existing_split(split_root: str | Path) -> tuple[list[str], list[str]]:
    split_root = Path(split_root).resolve()
    train_systems = [str(path) for path in _collect_systems_under(split_root / "train")]
    valid_systems = [str(path) for path in _collect_systems_under(split_root / "val")]
    if not valid_systems:
        valid_systems = [str(path) for path in _collect_systems_under(split_root / "valid")]
    return train_systems, valid_systems
