from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from workflow_core.common.io import ensure_dir, load_atoms_list, write_candidate_xyz, write_json


ENERGY_KEYS = ("energy", "REF_energy")
FORCES_KEYS = ("forces", "REF_forces")
STRESS_KEYS = ("stress", "REF_stress")


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _calculator_value(atoms: Atoms, key: str) -> Any | None:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None
    results = getattr(calc, "results", {}) or {}
    return results.get(key)


def _extract_energy(atoms: Atoms) -> float | None:
    for key in ENERGY_KEYS:
        value = _calculator_value(atoms, key)
        if value is None:
            value = atoms.info.get(key)
        parsed = _maybe_float(value)
        if parsed is not None:
            return parsed
    return None


def _extract_forces(atoms: Atoms):
    for key in FORCES_KEYS:
        value = _calculator_value(atoms, key)
        if value is None and key in atoms.arrays:
            value = atoms.arrays[key]
        if value is not None:
            return value
    return None


def _extract_stress(atoms: Atoms):
    for key in STRESS_KEYS:
        value = _calculator_value(atoms, key)
        if value is None and key in atoms.arrays:
            value = atoms.arrays[key]
        if value is None:
            value = atoms.info.get(key)
        if value is not None:
            return value
    return None


def _normalize_labeled_atoms(
    atoms_list: list[Atoms],
    *,
    require_stress: bool = True,
) -> tuple[list[Atoms], list[dict[str, Any]]]:
    normalized: list[Atoms] = []
    diagnostics: list[dict[str, Any]] = []
    for index, atoms in enumerate(atoms_list):
        copied = atoms.copy()
        energy = _extract_energy(atoms)
        forces = _extract_forces(atoms)
        stress = _extract_stress(atoms)
        missing: list[str] = []
        if energy is None:
            missing.append("energy")
        if forces is None:
            missing.append("forces")
        if require_stress and stress is None:
            missing.append("stress")
        if missing:
            raise ValueError(
                f"Frame {index} is missing required labeled properties: {', '.join(missing)}"
            )
        calc_kwargs = {"energy": energy, "forces": forces}
        if stress is not None:
            calc_kwargs["stress"] = stress
        copied.calc = SinglePointCalculator(copied, **calc_kwargs)
        normalized.append(copied)
        diagnostics.append(
            {
                "frame_index": int(index),
                "natoms": int(len(copied)),
                "formula": copied.get_chemical_formula(mode="hill"),
                "has_stress": bool(stress is not None),
            }
        )
    return normalized, diagnostics


def discover_external_dft_xyz_files(paths, outer_round: int) -> list[Path]:
    round_dir = paths.user_external_dft_round_dir(outer_round)
    if not round_dir.exists():
        return []

    preferred = paths.user_external_dft_labeled_xyz(outer_round)
    if preferred.exists() and preferred.stat().st_size > 0:
        return [preferred.resolve()]

    xyz_files: list[Path] = []
    for path in sorted(round_dir.rglob("*.xyz")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.name.lower() == "failed.xyz":
            continue
        xyz_files.append(path.resolve())
    return xyz_files


def resolve_historical_labeled_dft_sources(paths, outer_round: int) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    max_prev_round = int(outer_round) - 1
    if max_prev_round < 0:
        return resolved

    for prev_round in range(0, max_prev_round + 1):
        external_files = discover_external_dft_xyz_files(paths, prev_round)
        if external_files:
            for path in external_files:
                resolved.append(
                    {
                        "round": int(prev_round),
                        "source_kind": "user_assets_external_dft",
                        "path": str(path),
                    }
                )
            continue

        runtime_labeled = paths.dft_submit_labeled_xyz(prev_round)
        if runtime_labeled.exists() and runtime_labeled.stat().st_size > 0:
            resolved.append(
                {
                    "round": int(prev_round),
                    "source_kind": "runtime_dft_submit",
                    "path": str(runtime_labeled.resolve()),
                }
            )
    return resolved


def import_external_dft_results(
    *,
    paths,
    outer_round: int,
    labeled_xyz: str | Path,
    failed_xyz: str | Path | None = None,
    note: str | None = None,
    require_stress: bool = True,
    copy_source_snapshot: bool = True,
) -> dict[str, Any]:
    input_xyz = Path(labeled_xyz).resolve()
    if not input_xyz.exists() or not input_xyz.is_file():
        raise FileNotFoundError(f"External DFT labeled xyz not found: {input_xyz}")

    round_dir = ensure_dir(paths.user_external_dft_round_dir(outer_round))
    target_labeled = paths.user_external_dft_labeled_xyz(outer_round)
    target_failed = paths.user_external_dft_failed_xyz(outer_round)
    target_summary = paths.user_external_dft_summary_json(outer_round)
    target_manifest = paths.user_external_dft_manifest_json(outer_round)

    atoms_list = load_atoms_list(input_xyz)
    normalized_atoms, diagnostics = _normalize_labeled_atoms(
        atoms_list,
        require_stress=require_stress,
    )
    write_candidate_xyz(target_labeled, normalized_atoms)

    failed_count = 0
    if failed_xyz is not None:
        failed_source = Path(failed_xyz).resolve()
        if not failed_source.exists():
            raise FileNotFoundError(f"External DFT failed xyz not found: {failed_source}")
        shutil.copy2(failed_source, target_failed)
        try:
            failed_count = len(load_atoms_list(failed_source))
        except Exception:
            failed_count = 0
    else:
        target_failed.write_text("", encoding="utf-8")

    if copy_source_snapshot:
        snapshot_dir = ensure_dir(round_dir / "source_snapshot")
        shutil.copy2(input_xyz, snapshot_dir / input_xyz.name)
        if failed_xyz is not None:
            failed_source = Path(failed_xyz).resolve()
            shutil.copy2(failed_source, snapshot_dir / failed_source.name)

    summary = {
        "outer_round": int(outer_round),
        "status": "completed",
        "source_type": "external_dft_import",
        "imported_labeled_xyz": str(input_xyz),
        "imported_failed_xyz": str(Path(failed_xyz).resolve()) if failed_xyz is not None else None,
        "stored_labeled_xyz": str(target_labeled.resolve()),
        "stored_failed_xyz": str(target_failed.resolve()),
        "n_success": len(normalized_atoms),
        "n_failed": int(failed_count),
        "n_total": int(len(normalized_atoms) + failed_count),
        "require_stress": bool(require_stress),
        "note": note,
    }
    manifest = {
        **summary,
        "frame_diagnostics": diagnostics,
    }
    write_json(target_summary, summary)
    write_json(target_manifest, manifest)
    return manifest

