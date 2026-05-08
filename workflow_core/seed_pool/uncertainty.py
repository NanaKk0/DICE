from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from workflow_backends.deepmd.committee_infer import evaluate_frames_with_committee
from workflow_backends.deepmd.uncertainty import compute_frame_uncertainty
from workflow_core.common.io import load_atoms_list, write_candidate_xyz


def infer_seed_system_id(atoms: Atoms, field_name: str = "seed_system_id") -> str:
    info = atoms.info or {}
    for key in (field_name, "seed_system_id", "system_id", "source_system"):
        value = info.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return atoms.get_chemical_formula(mode="hill")


def sanitize_system_id(system_id: str) -> str:
    text = str(system_id or "unknown").strip()
    if not text:
        text = "unknown"
    safe = []
    for ch in text:
        safe.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(safe)


def seed_structure_id(atoms: Atoms, fallback_prefix: str = "seed") -> str:
    info = atoms.info or {}
    for key in ("structure_id", "seed_structure_id", "source_structure_id"):
        value = info.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    payload = np.asarray(atoms.get_positions(), dtype=float).round(6).tobytes()
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"{fallback_prefix}_{digest}"


def ensure_minimal_seed_metadata(atoms: Atoms, *, system_id: str | None = None, structure_id: str | None = None) -> Atoms:
    item = atoms.copy()
    resolved_system = str(system_id or infer_seed_system_id(item))
    resolved_structure = str(structure_id or seed_structure_id(item))
    times_used = int(item.info.get("times_used", item.info.get("seed_use_count", 0)) or 0)
    item.info["system_id"] = resolved_system
    item.info["seed_system_id"] = resolved_system
    item.info["structure_id"] = resolved_structure
    item.info["seed_structure_id"] = resolved_structure
    item.info["force_std_max"] = float(item.info.get("force_std_max", item.info.get("decision_metric", 0.0)) or 0.0)
    item.info["times_used"] = max(0, times_used)
    item.info["seed_use_count"] = max(0, times_used)
    return item


def group_atoms_by_system(atoms_list: list[Atoms]) -> dict[str, list[Atoms]]:
    grouped: dict[str, list[Atoms]] = defaultdict(list)
    for atoms in atoms_list:
        item = ensure_minimal_seed_metadata(atoms)
        grouped[str(item.info["system_id"])].append(item)
    return dict(grouped)


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


def combine_grouped_atoms(grouped_atoms: dict[str, list[Atoms]]) -> list[Atoms]:
    combined: list[Atoms] = []
    for system_id in sorted(grouped_atoms):
        combined.extend(ensure_minimal_seed_metadata(item, system_id=system_id) for item in grouped_atoms[system_id])
    return combined


def deduplicate_atoms_by_structure_id(atoms_list: list[Atoms]) -> list[Atoms]:
    deduped: dict[str, Atoms] = {}
    for atoms in atoms_list:
        item = ensure_minimal_seed_metadata(atoms)
        key = str(item.info["structure_id"])
        previous = deduped.get(key)
        if previous is None or float(item.info.get("force_std_max", 0.0)) > float(previous.info.get("force_std_max", 0.0)):
            deduped[key] = item
    return list(deduped.values())


def sort_atoms_by_uncertainty_desc(atoms_list: list[Atoms]) -> list[Atoms]:
    return sorted(
        (ensure_minimal_seed_metadata(item) for item in atoms_list),
        key=lambda atoms: (
            -float(atoms.info.get("force_std_max", 0.0)),
            str(atoms.info.get("structure_id", "")),
        ),
    )


def rescore_atoms_with_committee(atoms_list: list[Atoms], committee_graphs: list[str], temp_root: str | Path, prefix: str) -> list[Atoms]:
    atoms_items = [ensure_minimal_seed_metadata(item) for item in atoms_list]
    if not atoms_items:
        return []
    temp_root = Path(temp_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    input_xyz = temp_root / f"{prefix}.extxyz"
    write_candidate_xyz(input_xyz, atoms_items)
    infer_result = evaluate_frames_with_committee(input_xyz, committee_graphs)
    npz = np.load(infer_result["npz_path"])
    forces_array = npz["forces"]
    rescored: list[Atoms] = []
    for frame_index, atoms in enumerate(atoms_items):
        metrics = compute_frame_uncertainty(forces_array[frame_index])
        item = atoms.copy()
        item.info["force_std_max"] = float(metrics["force_std_max"])
        item.info["decision_metric"] = float(metrics["force_std_max"])
        rescored.append(item)
    return rescored


