from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from workflow_core.dft.status import cp2k_output_looks_successful
from workflow_core.dft.defaults import DEFAULT_DFT_OUTPUT_NAME, DEFAULT_DFT_STRUCTURE_NAME
from workflow_core.dft.runtime import resolve_dft_runtime
from workflow_core.common.io import load_atoms_list, read_json, write_candidate_xyz, write_json


TASK_DIR_PATTERN = re.compile(r"^(?:task\.)?(\d+)$")
ENERGY_PATTERNS = [
    re.compile(r"ENERGY\| Total FORCE_EVAL.*?:\s*([-+0-9Ee\.]+)"),
    re.compile(r"Total energy:\s*([-+0-9Ee\.]+)", re.IGNORECASE),
]
def load_dft_collect_config(config_path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(config_path).resolve())
    payload.setdefault("required_outputs", [DEFAULT_DFT_OUTPUT_NAME])
    payload["cp2k_output_name"] = DEFAULT_DFT_OUTPUT_NAME
    payload["structure_name"] = DEFAULT_DFT_STRUCTURE_NAME
    return payload


def _iter_task_dirs(job_root: Path) -> list[tuple[int, Path]]:
    tasks = []
    for child in sorted(job_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        match = TASK_DIR_PATTERN.match(child.name)
        if not match:
            continue
        tasks.append((int(match.group(1)), child))
    return tasks


def _parse_cp2k_energy(output_text: str) -> float:
    for pattern in ENERGY_PATTERNS:
        matches = pattern.findall(output_text)
        if matches:
            return float(matches[-1])
    raise RuntimeError("Failed to parse CP2K total energy")


def _parse_cp2k_forces(output_text: str, natoms: int) -> np.ndarray:
    lines = output_text.splitlines()
    force_rows: list[list[float]] = []
    capture = False
    for line in lines:
        normalized = line.strip().upper()
        if normalized.startswith("ATOMIC FORCES"):
            capture = True
            force_rows = []
            continue
        if capture:
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                try:
                    fx, fy, fz = float(parts[-3]), float(parts[-2]), float(parts[-1])
                except ValueError:
                    fx = fy = fz = None
                if fx is not None:
                    force_rows.append([fx, fy, fz])
                    continue
            if force_rows:
                break
    if len(force_rows) != natoms:
        raise RuntimeError(f"Failed to parse CP2K forces: expected {natoms}, got {len(force_rows)}")
    return np.asarray(force_rows, dtype=float)


def _parse_cp2k_stress(output_text: str) -> np.ndarray | None:
    upper = output_text.upper()
    marker = "STRESS TENSOR"
    if marker not in upper:
        return None
    lines = output_text.splitlines()
    for idx, line in enumerate(lines):
        if marker in line.upper():
            rows: list[list[float]] = []
            for candidate in lines[idx + 1 : idx + 8]:
                parts = candidate.split()
                numbers = []
                for part in parts:
                    try:
                        numbers.append(float(part))
                    except ValueError:
                        continue
                if len(numbers) >= 3:
                    rows.append(numbers[-3:])
                if len(rows) == 3:
                    return np.asarray(rows, dtype=float)
    return None


def parse_cp2k_task(task_dir: Path, cfg: dict[str, Any]) -> Atoms:
    atoms = read(str(task_dir / DEFAULT_DFT_STRUCTURE_NAME), index=0)
    output_text = (task_dir / DEFAULT_DFT_OUTPUT_NAME).read_text(encoding="utf-8", errors="replace")
    energy = _parse_cp2k_energy(output_text)
    forces = _parse_cp2k_forces(output_text, len(atoms))
    stress = _parse_cp2k_stress(output_text)

    calc_kwargs: dict[str, Any] = {"energy": energy, "forces": forces}
    if stress is not None:
        calc_kwargs["stress"] = stress
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms


def _task_success(task_dir: Path, cfg: dict[str, Any]) -> bool:
    status_path = task_dir / "task_status.json"
    if status_path.exists():
        try:
            payload = read_json(status_path)
            if payload.get("status") == "success":
                return True
        except Exception:
            pass
    return cp2k_output_looks_successful(task_dir, cfg)


def collect_dft_results(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_dft_runtime(cfg)
    job_root = Path(cfg["task_root"]).resolve() / cfg["jobid"]
    output_root = Path(cfg.get("output_root", Path(cfg["task_root"]).resolve().parent)).resolve()
    labeled_xyz = Path(cfg.get("labeled_xyz", output_root / "labeled.xyz")).resolve()
    failed_xyz = Path(cfg.get("failed_xyz", output_root / "failed.xyz")).resolve()
    summary_path = Path(cfg.get("summary_path", output_root / "summary.json")).resolve()
    task_map_path = Path(cfg.get("task_map_path", output_root / "task_map.json")).resolve()
    task_map = read_json(task_map_path) if task_map_path.exists() else []
    task_map_by_idx = {int(item["task_idx"]): item for item in task_map}

    labeled_atoms: list[Atoms] = []
    failed_atoms: list[Atoms] = []
    failed_tasks: list[str] = []

    for task_idx, task_dir in _iter_task_dirs(job_root):
        structure_path = task_dir / DEFAULT_DFT_STRUCTURE_NAME
        base_atoms = read(str(structure_path), index=0)
        if not _task_success(task_dir, cfg):
            base_atoms.info["dft_collect_error"] = "task_not_successful"
            failed_atoms.append(base_atoms)
            failed_tasks.append(task_dir.name)
            continue
        try:
            atoms = parse_cp2k_task(task_dir, cfg)
        except Exception as exc:
            failed = base_atoms.copy()
            failed.info["dft_collect_error"] = str(exc)
            failed_atoms.append(failed)
            failed_tasks.append(task_dir.name)
            continue
        meta_path = task_dir / "meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            for key, value in meta.items():
                if key == "cell_resolution":
                    atoms.info[key] = value
                    continue
                atoms.info[key] = value
        atoms.info["dft_jobid"] = cfg["jobid"]
        atoms.info["dft_task_idx"] = int(task_idx)
        mapped = task_map_by_idx.get(task_idx, {})
        for key in ("source_round", "source_traj", "source_frame", "selection_score", "committee_uncertainty"):
            if key in mapped and key not in atoms.info:
                atoms.info[key] = mapped[key]
        labeled_atoms.append(atoms)

    write_candidate_xyz(labeled_xyz, labeled_atoms)
    write_candidate_xyz(failed_xyz, failed_atoms)
    summary = {
        "jobid": cfg["jobid"],
        "n_total": len(task_map) if task_map else len(labeled_atoms) + len(failed_atoms),
        "n_success": len(labeled_atoms),
        "n_failed": len(failed_atoms),
        "success_ratio": (len(labeled_atoms) / max(1, len(labeled_atoms) + len(failed_atoms))),
        "failed_tasks": failed_tasks,
        "labeled_xyz": str(labeled_xyz),
        "failed_xyz": str(failed_xyz),
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    write_json(summary_path, summary)
    return summary

