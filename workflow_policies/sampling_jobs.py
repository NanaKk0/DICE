from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read

from workflow_core.seed_pool.uncertainty import infer_seed_system_id


def build_sampling_jobs(
    initial_structure_paths: list[str],
    committee_pool: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not committee_pool:
        raise RuntimeError("committee_pool is empty")

    md_cfg = cfg["md"]
    driver_policy = str(cfg.get("driver_policy", "round_robin"))

    jobs: list[dict[str, Any]] = []
    driver_cursor = 0

    for structure_path in initial_structure_paths:
        structure_path = str(Path(structure_path).resolve())
        seed_atoms = read(structure_path, index=0)
        source_system_id = str(infer_seed_system_id(seed_atoms))
        source_structure_id = str(
            seed_atoms.info.get("structure_id")
            or seed_atoms.info.get("seed_structure_id")
            or Path(structure_path).stem
        )
        for temperature in md_cfg["temperature_schedule"]:
            if driver_policy != "round_robin":
                raise ValueError(f"Unsupported driver_policy: {driver_policy}")
            driver = committee_pool[driver_cursor % len(committee_pool)]
            driver_cursor += 1
            job_index = len(jobs) + 1

            jobs.append(
                {
                    "job_id": f"traj_{job_index:06d}",
                    "initial_structure_path": structure_path,
                    "source_system_id": source_system_id,
                    "source_structure_id": source_structure_id,
                    "temperature": float(temperature),
                    "steps": int(md_cfg["steps"]),
                    "timestep_fs": float(md_cfg["timestep_fs"]),
                    "driver_model_index": int(driver["model_index"]),
                    "driver_graph_path": str(Path(driver["graph_pb"]).resolve()),
                    "job_seed": int(md_cfg.get("seed", 20260319)) + job_index * 97,
                }
            )

    return jobs
