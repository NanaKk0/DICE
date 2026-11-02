from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from ase.units import Bohr, GPa, Hartree


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?|[-+]?(?:NaN|Inf)"


@dataclass(frozen=True, slots=True)
class CP2KParsedLabels:
    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    stress_ev_per_angstrom3: np.ndarray | None
    virial_ev: np.ndarray | None
    scf_converged: bool | None


def parse_cp2k_output(
    path: str | Path,
    *,
    symbols: tuple[str, ...],
    volume_angstrom3: float,
    require_stress: bool,
) -> CP2KParsedLabels:
    output_path = Path(path)
    try:
        text = output_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read CP2K output {output_path}") from exc
    if "PROGRAM ENDED AT" not in text:
        raise ValueError("CP2K output is missing normal termination evidence")
    energy_matches = re.findall(
        rf"^\s*ENERGY\|.*?energy\s*\[(?:a\.u\.|hartree)\]\s*:?\s*({_FLOAT})",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not energy_matches:
        raise ValueError("CP2K output is missing finite energy")
    energy_au = _number(energy_matches[-1])
    if not math.isfinite(energy_au):
        raise ValueError("CP2K output is missing finite energy")
    forces_au, parsed_symbols = _parse_forces(text)
    if (
        (parsed_symbols is not None and parsed_symbols != symbols)
        or forces_au.shape != (len(symbols), 3)
    ):
        raise ValueError("CP2K output forces do not preserve selected atom order")
    if not np.all(np.isfinite(forces_au)):
        raise ValueError("CP2K output forces are non-finite")
    stress = None
    virial = None
    stress_gpa = _parse_stress_gpa(text)
    if require_stress and stress_gpa is None:
        raise ValueError("CP2K output is missing required stress")
    if stress_gpa is not None:
        if not np.all(np.isfinite(stress_gpa)):
            raise ValueError("CP2K output stress is non-finite")
        # CP2K's printed stress uses positive compression; ASE uses positive tension.
        stress_matrix = -stress_gpa * GPa
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
    if "SCF run NOT converged" in text:
        scf_converged: bool | None = False
    elif "SCF run converged" in text:
        scf_converged = True
    else:
        scf_converged = None
    return CP2KParsedLabels(
        energy_ev=energy_au * Hartree,
        forces_ev_per_angstrom=forces_au * Hartree / Bohr,
        stress_ev_per_angstrom3=stress,
        virial_ev=virial,
        scf_converged=scf_converged,
    )


def _parse_forces(text: str) -> tuple[np.ndarray, tuple[str, ...] | None]:
    starts = list(
        re.finditer(r"^\s*ATOMIC FORCES in \[a\.u\.\]", text, flags=re.MULTILINE)
    )
    for start in reversed(starts):
        terminal = re.search(
            r"^\s*SUM OF ATOMIC FORCES", text[start.end() :], flags=re.MULTILINE
        )
        if terminal is None:
            continue
        body = text[start.end() : start.end() + terminal.start()]
        rows: list[list[float]] = []
        symbols: list[str] = []
        pattern = re.compile(
            rf"^\s*\d+\s+\d+\s+([A-Za-z][A-Za-z]?)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
            flags=re.MULTILINE,
        )
        for match in pattern.finditer(body):
            symbols.append(match.group(1))
            rows.append([_number(match.group(i)) for i in range(2, 5)])
        if rows:
            return np.asarray(rows, dtype=float), tuple(symbols)
    modern_starts = list(
        re.finditer(
            r"^\s*FORCES\|\s*Atomic forces\s*\[hartree/bohr\]",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    modern_row = re.compile(
        rf"^\s*FORCES\|\s*(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s+{_FLOAT}\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for start in reversed(modern_starts):
        terminal = re.search(
            r"^\s*FORCES\|\s*Sum\b",
            text[start.end() :],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if terminal is None:
            continue
        body = text[start.end() : start.end() + terminal.start()]
        indexed = [
            (
                int(match.group(1)),
                [_number(match.group(i)) for i in range(2, 5)],
            )
            for match in modern_row.finditer(body)
        ]
        if indexed and [index for index, _ in indexed] == list(
            range(1, len(indexed) + 1)
        ):
            return np.asarray([row for _, row in indexed], dtype=float), None
    raise ValueError("CP2K output is missing complete atomic forces")


def _parse_stress_gpa(text: str) -> np.ndarray | None:
    headers = list(
        re.finditer(
            r"^(?:\s*STRESS\|\s*Analytical stress tensor|\s*STRESS TENSOR)\s*\[(GPa|bar)\]\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    row_pattern = re.compile(
        rf"^(?:\s*STRESS\|)?\s*([XYZ])\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    for header in reversed(headers):
        rows: dict[str, list[float]] = {}
        for match in row_pattern.finditer(text[header.end() : header.end() + 3000]):
            rows[match.group(1).upper()] = [
                _number(match.group(i)) for i in range(2, 5)
            ]
            if set(rows) == {"X", "Y", "Z"}:
                matrix = np.asarray([rows[axis] for axis in ("X", "Y", "Z")])
                if not np.allclose(matrix, matrix.T, atol=1.0e-8, rtol=1.0e-8):
                    raise ValueError("CP2K output stress tensor is not symmetric")
                unit = header.group(1).lower()
                return matrix * (1.0e-4 if unit == "bar" else 1.0)
    return None


def _number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = ["CP2KParsedLabels", "parse_cp2k_output"]
