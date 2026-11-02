from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from ase.units import GPa


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?|[-+]?(?:NaN|Inf)"


@dataclass(frozen=True, slots=True)
class ABACUSParsedLabels:
    energy_ev: float
    native_forces_ev_per_angstrom: np.ndarray
    stress_ev_per_angstrom3: np.ndarray | None
    virial_ev: np.ndarray | None
    scf_converged: bool | None


def parse_abacus_output(
    path: str | Path,
    *,
    native_symbols: tuple[str, ...],
    volume_angstrom3: float,
    require_stress: bool,
) -> ABACUSParsedLabels:
    output_path = Path(path)
    try:
        text = output_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read ABACUS output {output_path}") from exc
    if not re.search(r"^\s*Total\s+Time\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
        raise ValueError("ABACUS output is missing normal termination evidence")
    energy_matches = re.findall(
        rf"(?:final\s+etot\s+is|!FINAL_ETOT_IS|TOTAL\s+ENERGY)\s+({_FLOAT})\s+eV",
        text,
        flags=re.IGNORECASE,
    )
    if not energy_matches:
        raise ValueError("ABACUS output is missing finite energy")
    energy = _number(energy_matches[-1])
    if not math.isfinite(energy):
        raise ValueError("ABACUS output is missing finite energy")
    forces, labels = _parse_forces(text)
    expected_labels = _native_atom_labels(native_symbols)
    if labels != expected_labels:
        raise ValueError("ABACUS output forces do not preserve native atom order")
    if forces.shape != (len(native_symbols), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("ABACUS output is missing complete finite forces")
    stress = None
    virial = None
    stress_kbar = _parse_stress_kbar(text)
    if require_stress and stress_kbar is None:
        raise ValueError("ABACUS output is missing required stress")
    if stress_kbar is not None:
        if not np.all(np.isfinite(stress_kbar)):
            raise ValueError("ABACUS output stress is non-finite")
        if not np.allclose(stress_kbar, stress_kbar.T, atol=1.0e-8, rtol=1.0e-8):
            raise ValueError("ABACUS output stress tensor is not symmetric")
        # ABACUS prints kbar with positive compression; ASE uses positive tension.
        stress_matrix = -stress_kbar * 0.1 * GPa
        stress = np.asarray(
            [
                stress_matrix[0, 0],
                stress_matrix[1, 1],
                stress_matrix[2, 2],
                stress_matrix[1, 2],
                stress_matrix[0, 2],
                stress_matrix[0, 1],
            ]
        )
        virial = -stress_matrix * abs(float(volume_angstrom3))
    if "charge density convergence is achieved" in text:
        scf_converged: bool | None = True
    elif re.search(r"convergence\s+(?:is\s+)?not\s+achieved", text, re.IGNORECASE):
        scf_converged = False
    else:
        scf_converged = None
    return ABACUSParsedLabels(
        energy_ev=energy,
        native_forces_ev_per_angstrom=forces,
        stress_ev_per_angstrom3=stress,
        virial_ev=virial,
        scf_converged=scf_converged,
    )


def _parse_forces(text: str) -> tuple[np.ndarray, tuple[str, ...]]:
    headers = list(
        re.finditer(
            r"^\s*TOTAL-FORCE\s*\(eV/Angstrom\)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    row = re.compile(
        rf"^\s*([A-Z][a-z]?\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
        flags=re.MULTILINE,
    )
    for header in reversed(headers):
        body = text[header.end() : header.end() + 200_000]
        matches = list(row.finditer(body))
        if matches:
            labels = tuple(match.group(1) for match in matches)
            values = np.asarray(
                [[_number(match.group(index)) for index in range(2, 5)] for match in matches],
                dtype=float,
            )
            return values, labels
    raise ValueError("ABACUS output is missing complete finite forces")


def _parse_stress_kbar(text: str) -> np.ndarray | None:
    headers = list(
        re.finditer(
            r"^\s*TOTAL-STRESS\s*\(KBAR\)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    row = re.compile(
        rf"^\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
        flags=re.MULTILINE,
    )
    for header in reversed(headers):
        rows = [
            [_number(match.group(index)) for index in range(1, 4)]
            for match in row.finditer(text[header.end() : header.end() + 4000])
        ]
        if len(rows) >= 3:
            return np.asarray(rows[:3], dtype=float)
    return None


def _native_atom_labels(symbols: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return tuple(labels)


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = ["ABACUSParsedLabels", "parse_abacus_output"]
