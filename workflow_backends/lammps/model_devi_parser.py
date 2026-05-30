from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.seed_pool.uncertainty import ensure_minimal_seed_metadata


def parse_model_devi_out(path: Path, *, natoms: int | None, atomic_enabled: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            atomic_values = []
            if atomic_enabled and natoms is not None and len(parts) >= 7 + natoms:
                atomic_values = [float(item) for item in parts[7 : 7 + natoms]]
            rows.append(
                {
                    "step": int(float(parts[0])),
                    "max_devi_v": float(parts[1]),
                    "min_devi_v": float(parts[2]),
                    "avg_devi_v": float(parts[3]),
                    "max_devi_f": float(parts[4]),
                    "min_devi_f": float(parts[5]),
                    "avg_devi_f": float(parts[6]),
                    "atomic_devi_f": atomic_values,
                }
            )
    return rows


def route_frame_class_uncertainty_only(max_devi_f: float, *, f_trust_lo: float, f_trust_hi: float) -> str:
    metric = float(max_devi_f)
    if metric < float(f_trust_lo):
        return "accurate"
    if metric >= float(f_trust_hi):
        return "failed_uncertainty"
    return "candidate"


def build_fast_selected_record(*, atoms, job: dict[str, Any], frame_index: int, model_devi_row: dict[str, Any]) -> dict[str, Any]:
    atoms = atoms.copy()
    system_id = str(job.get("source_system_id") or atoms.info.get("system_id") or atoms.get_chemical_formula(mode="hill"))
    structure_id = str(
        job.get("source_structure_id")
        or atoms.info.get("structure_id")
        or atoms.info.get("seed_structure_id")
        or f"{Path(job['initial_structure_path']).stem}_step_{int(model_devi_row['step'])}"
    )
    atoms.info["student_backend"] = "deepmd"
    atoms.info["sampling_backend"] = "lammps"
    atoms.info["uncertainty_source"] = "model_devi"
    atoms.info["student_uncertainty"] = float(model_devi_row["max_devi_f"])
    atoms.info["uncertainty_metric"] = "force_std_max"
    atoms.info["source_task"] = str(job["job_id"])
    atoms.info["source_seed"] = str(Path(job["initial_structure_path"]).name)
    atoms.info["md_step"] = int(model_devi_row["step"])
    atoms.info["max_devi_f"] = float(model_devi_row["max_devi_f"])
    atoms.info["avg_devi_f"] = float(model_devi_row["avg_devi_f"])
    atoms.info["route"] = "candidate"
    atoms = ensure_minimal_seed_metadata(atoms, system_id=system_id, structure_id=structure_id)
    atoms.info["force_std_max"] = float(model_devi_row["max_devi_f"])
    return {
        "atoms": atoms,
        "source_job_id": str(job["job_id"]),
        "source_frame_index": int(frame_index),
        "driver_model_index": int(job["driver_model_index"]),
        "decision_metric": float(model_devi_row["max_devi_f"]),
        "metrics": {
            "force_std_max": float(model_devi_row["max_devi_f"]),
            "force_std_avg": float(model_devi_row["avg_devi_f"]),
            "force_std_p90": float(model_devi_row["max_devi_f"]),
        },
        "frame_class": "candidate",
    }


def build_fast_pool_atoms(*, atoms, job: dict[str, Any], model_devi_row: dict[str, Any], route: str):
    item = atoms.copy()
    system_id = str(job.get("source_system_id") or item.info.get("system_id") or item.get_chemical_formula(mode="hill"))
    structure_id = str(
        item.info.get("structure_id")
        or item.info.get("seed_structure_id")
        or f"{job['job_id']}_step_{int(model_devi_row['step'])}"
    )
    item = ensure_minimal_seed_metadata(item, system_id=system_id, structure_id=structure_id)
    item.info["student_backend"] = "deepmd"
    item.info["sampling_backend"] = "lammps"
    item.info["uncertainty_source"] = "model_devi"
    item.info["student_uncertainty"] = float(model_devi_row["max_devi_f"])
    item.info["uncertainty_metric"] = "force_std_max"
    item.info["force_std_max"] = float(model_devi_row["max_devi_f"])
    item.info["source_task"] = str(job["job_id"])
    item.info["source_job_id"] = str(job["job_id"])
    item.info["source_frame_index"] = int(model_devi_row.get("frame_index", 0))
    item.info["md_step"] = int(model_devi_row["step"])
    item.info["route"] = str(route)
    return item
