from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms


def compute_frame_uncertainty(forces_by_model: np.ndarray) -> dict[str, Any]:
    forces = np.asarray(forces_by_model, dtype=float)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise ValueError(f"forces_by_model must have shape (n_models, natoms, 3), got {forces.shape}")

    mean_force = np.mean(forces, axis=0)
    diff = forces - mean_force
    sigma_i = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))

    return {
        "force_std_max": float(np.max(sigma_i)) if sigma_i.size else 0.0,
        "force_std_avg": float(np.mean(sigma_i)) if sigma_i.size else 0.0,
        "force_std_p90": float(np.percentile(sigma_i, 90)) if sigma_i.size else 0.0,
        "per_atom_force_std": sigma_i.astype(float).tolist(),
    }


def inspect_geometry(atoms: Atoms, geometry_cfg: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    has_nan_positions = bool(np.isnan(positions).any() or np.isinf(positions).any())

    natoms = len(atoms)
    if natoms < 2 or has_nan_positions:
        min_pair_distance = float("inf") if natoms < 2 else float("nan")
    else:
        distances = atoms.get_all_distances(mic=True)
        iu = np.triu_indices(natoms, k=1)
        min_pair_distance = float(np.min(distances[iu])) if iu[0].size else float("inf")

    try:
        cell_volume = float(atoms.cell.volume)
    except Exception:
        cell_volume = 0.0

    hard_thr = float(geometry_cfg.get("hard_min_pair_distance", 0.65))
    soft_thr = float(geometry_cfg.get("soft_min_pair_distance", 0.85))

    if has_nan_positions:
        geometry_ok: bool | str = False
    elif natoms < 2:
        geometry_ok = True
    elif min_pair_distance < hard_thr:
        geometry_ok = False
    elif min_pair_distance < soft_thr:
        geometry_ok = "suspicious"
    else:
        geometry_ok = True

    return {
        "min_pair_distance": min_pair_distance,
        "has_nan_positions": has_nan_positions,
        "natoms": natoms,
        "cell_volume": cell_volume,
        "geometry_ok": geometry_ok,
    }


def classify_frame(
    frame_metrics: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    geometry_cfg: dict[str, Any],
) -> str:
    del geometry_cfg

    geometry_ok = frame_metrics.get("geometry_ok", True)
    if geometry_ok is False:
        return "failed_geometry"

    decision_metric = str(uncertainty_cfg.get("decision_metric", "force_std_max"))
    metric = float(frame_metrics[decision_metric])
    level_f_lo = float(uncertainty_cfg["level_f_lo"])
    level_f_hi = float(uncertainty_cfg["level_f_hi"])

    if metric < level_f_lo:
        return "accurate"
    if metric >= level_f_hi:
        return "failed_uncertainty"
    if geometry_ok == "suspicious":
        return "suspicious_candidate"
    return "candidate"
