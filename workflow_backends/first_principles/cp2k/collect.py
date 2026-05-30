from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

from workflow_backends.first_principles.cp2k.defaults import DEFAULT_DFT_OUTPUT_NAME, DEFAULT_DFT_STRUCTURE_NAME


ENERGY_PATTERNS = [
    re.compile(r"ENERGY\| Total FORCE_EVAL.*?:\s*([-+0-9Ee\.]+)"),
    re.compile(r"Total energy:\s*([-+0-9Ee\.]+)", re.IGNORECASE),
]


def parse_cp2k_energy(output_text: str) -> float:
    for pattern in ENERGY_PATTERNS:
        matches = pattern.findall(output_text)
        if matches:
            return float(matches[-1])
    raise RuntimeError("Failed to parse CP2K total energy")


def parse_cp2k_forces(output_text: str, natoms: int) -> np.ndarray:
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
            if len(parts) >= 5 and parts[0].isdigit():
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


def parse_cp2k_stress(output_text: str) -> np.ndarray | None:
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
    structure_name = str(cfg.get("structure_name", DEFAULT_DFT_STRUCTURE_NAME))
    output_name = str(cfg.get("cp2k_output_name", DEFAULT_DFT_OUTPUT_NAME))
    atoms = read(str(task_dir / structure_name), index=0)
    output_text = (task_dir / output_name).read_text(encoding="utf-8", errors="replace")
    energy = parse_cp2k_energy(output_text)
    forces = parse_cp2k_forces(output_text, len(atoms))
    stress = parse_cp2k_stress(output_text)

    calc_kwargs: dict[str, Any] = {"energy": energy, "forces": forces}
    if stress is not None:
        calc_kwargs["stress"] = stress
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms
