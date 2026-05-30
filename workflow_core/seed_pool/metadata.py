from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from ase import Atoms


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


def times_used(atoms: Atoms) -> int:
    return max(0, int(atoms.info.get("times_used", atoms.info.get("seed_use_count", 0)) or 0))


def set_times_used(atoms: Atoms, value: int) -> None:
    safe_value = max(0, int(value))
    atoms.info["times_used"] = safe_value
    atoms.info["seed_use_count"] = safe_value


def ensure_minimal_seed_metadata(atoms: Atoms, *, system_id: str | None = None, structure_id: str | None = None) -> Atoms:
    item = atoms.copy()
    resolved_system = str(system_id or infer_seed_system_id(item))
    resolved_structure = str(structure_id or seed_structure_id(item))
    current_times_used = times_used(item)
    item.info["system_id"] = resolved_system
    item.info["seed_system_id"] = resolved_system
    item.info["structure_id"] = resolved_structure
    item.info["seed_structure_id"] = resolved_structure
    item.info["force_std_max"] = float(item.info.get("force_std_max", item.info.get("decision_metric", 0.0)) or 0.0)
    set_times_used(item, current_times_used)
    return item


def group_atoms_by_system(atoms_list: list[Atoms]) -> dict[str, list[Atoms]]:
    grouped: dict[str, list[Atoms]] = defaultdict(list)
    for atoms in atoms_list:
        item = ensure_minimal_seed_metadata(atoms)
        grouped[str(item.info["system_id"])].append(item)
    return dict(grouped)


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


def sort_atoms_by_usage_then_uncertainty_desc(atoms_list: list[Atoms]) -> list[Atoms]:
    return sorted(
        (ensure_minimal_seed_metadata(item) for item in atoms_list),
        key=lambda atoms: (
            times_used(atoms),
            -float(atoms.info.get("force_std_max", 0.0)),
            str(atoms.info.get("structure_id", "")),
        ),
    )
