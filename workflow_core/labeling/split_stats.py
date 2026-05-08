from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read


def system_key(atoms: Atoms) -> str:
    return str(atoms.info.get("formula") or atoms.get_chemical_formula(mode="hill"))


def _empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "accurate": 0,
        "candidate": 0,
        "failed": 0,
        "selected": 0,
    }


def _with_ratios(counts: dict[str, int]) -> dict[str, Any]:
    total = int(counts.get("total", 0) or 0)
    payload: dict[str, Any] = {key: int(value) for key, value in counts.items()}
    denom = max(1, total)
    payload["accurate_ratio"] = float(payload.get("accurate", 0) / denom)
    payload["candidate_ratio"] = float(payload.get("candidate", 0) / denom)
    payload["failed_ratio"] = float(payload.get("failed", 0) / denom)
    payload["selected_ratio"] = float(payload.get("selected", 0) / denom)
    return payload


def summarize_sampling_records_by_system(frame_records: list[dict[str, Any]], selected_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = defaultdict(_empty_counts)

    for record in frame_records:
        atoms = record.get("atoms")
        if atoms is None:
            continue
        key = system_key(atoms)
        frame_class = str(record.get("frame_class", "unknown"))
        grouped[key]["total"] += 1
        if frame_class == "accurate":
            grouped[key]["accurate"] += 1
        elif frame_class in {"candidate", "suspicious_candidate"}:
            grouped[key]["candidate"] += 1
        elif frame_class in {"failed_uncertainty", "failed_geometry"}:
            grouped[key]["failed"] += 1

    for record in selected_records or []:
        atoms = record.get("atoms")
        if atoms is None:
            continue
        grouped[system_key(atoms)]["selected"] += 1

    return {
        "group_by": "formula",
        "systems": {key: _with_ratios(grouped[key]) for key in sorted(grouped)},
        "totals": _with_ratios(_sum_grouped_counts(grouped.values())),
    }


def summarize_label_split_by_system(train_xyz: str | Path, dft_xyz: str | Path, discard_xyz: str | Path) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    _accumulate_xyz(grouped, train_xyz, "accurate")
    _accumulate_xyz(grouped, dft_xyz, "candidate")
    _accumulate_xyz(grouped, discard_xyz, "failed")
    for counts in grouped.values():
        counts["total"] = int(counts.get("accurate", 0)) + int(counts.get("candidate", 0)) + int(counts.get("failed", 0))
    return {
        "group_by": "formula",
        "meaning": {
            "accurate": "label split train.xyz",
            "candidate": "label split dft.xyz before optional DFT screening",
            "failed": "label split discard.xyz; disabled for teacher labeling, normally zero",
        },
        "systems": {key: _with_ratios(grouped[key]) for key in sorted(grouped)},
        "totals": _with_ratios(_sum_grouped_counts(grouped.values())),
    }


def _accumulate_xyz(grouped: dict[str, dict[str, int]], xyz_path: str | Path, field: str) -> None:
    path = Path(xyz_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    for atoms in read(str(path), index=":"):
        grouped[system_key(atoms)][field] += 1


def _sum_grouped_counts(items) -> dict[str, int]:
    total = Counter()
    for counts in items:
        total.update({key: int(value) for key, value in counts.items()})
    payload = _empty_counts()
    payload.update({key: int(value) for key, value in total.items()})
    return payload
