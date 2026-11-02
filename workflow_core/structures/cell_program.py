from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
import numpy as np

from workflow_core.common.sampling_plan import TemperatureProgram


_AMU_PER_ANGSTROM3_TO_G_CM3 = 1.66053906660
_STRAIN_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
_Matrix3 = tuple[tuple[float, float, float], ...]
_Positions = tuple[tuple[float, float, float], ...]


def _matrix(value: np.ndarray) -> _Matrix3:
    array = np.asarray(value, dtype=float)
    return tuple(tuple(float(item) for item in row) for row in array)


def _positions(value: np.ndarray) -> _Positions:
    array = np.asarray(value, dtype=float)
    return tuple(tuple(float(item) for item in row) for row in array)


def _density(total_mass_amu: float, cell_rows: np.ndarray) -> float:
    return (
        float(total_mass_amu)
        * _AMU_PER_ANGSTROM3_TO_G_CM3
        / abs(float(np.linalg.det(cell_rows)))
    )


def _deformation_from_declared(
    current_density: float,
    declared: dict[str, object] | None,
) -> np.ndarray:
    if declared is None:
        return np.eye(3)
    if declared.get("mode") == "isotropic_density":
        if set(declared) != {"mode", "stop_density_g_cm3"}:
            raise ValueError("cell program density endpoint is malformed")
        target_density = declared["stop_density_g_cm3"]
        if (
            isinstance(target_density, bool)
            or not isinstance(target_density, (int, float))
            or not math.isfinite(float(target_density))
            or float(target_density) <= 0.0
        ):
            raise ValueError("cell program density endpoint is malformed")
        return np.eye(3) * (
            current_density / float(target_density)
        ) ** (1.0 / 3.0)
    if declared.get("mode") != "symmetric_strain" or set(declared) != {
        "mode",
        "stop",
    }:
        raise ValueError("cell program endpoint mode is invalid")
    stop = declared["stop"]
    if not isinstance(stop, dict) or set(stop) != set(_STRAIN_COMPONENTS) or not all(
        isinstance(stop[name], (int, float))
        and not isinstance(stop[name], bool)
        and math.isfinite(float(stop[name]))
        for name in _STRAIN_COMPONENTS
    ):
        raise ValueError("cell program strain endpoint is malformed")
    values = {name: float(stop[name]) for name in _STRAIN_COMPONENTS}
    deformation = np.asarray(
        (
            (1.0 + values["xx"], values["xy"], values["xz"]),
            (values["xy"], 1.0 + values["yy"], values["yz"]),
            (values["xz"], values["yz"], 1.0 + values["zz"]),
        )
    )
    if np.any(np.linalg.eigvalsh(deformation) <= 0.0):
        raise ValueError("cell program strain endpoint must be positive definite")
    return deformation


@dataclass(frozen=True, slots=True)
class ResolvedCellProgramSegment:
    segment_index: int
    start_step: int
    end_step: int
    start_temperature_k: float
    stop_temperature_k: float
    start_cell_rows_angstrom: _Matrix3
    end_cell_rows_angstrom: _Matrix3
    start_density_g_cm3: float
    end_density_g_cm3: float
    deformation_gradient: _Matrix3
    declared_cell: dict[str, object] | None

    @property
    def steps(self) -> int:
        return self.end_step - self.start_step


@dataclass(frozen=True, slots=True)
class ResolvedCellProgramHistory:
    input_cell_rows_angstrom: _Matrix3
    input_positions_angstrom: _Positions
    atom_symbols: tuple[str, ...]
    composition: tuple[tuple[str, int], ...]
    standard_mass_amu: float
    start_density_g_cm3: float
    segments: tuple[ResolvedCellProgramSegment, ...]
    expected_dump_steps: tuple[int, ...]

    def cell_at(self, segment_index: int, fraction: float) -> np.ndarray:
        segment = self.segments[int(segment_index)]
        progress = _fraction(fraction)
        deformation = np.asarray(segment.deformation_gradient, dtype=float)
        affine = (1.0 - progress) * np.eye(3) + progress * deformation
        return np.asarray(segment.start_cell_rows_angstrom, dtype=float) @ affine.T

    def positions_at(self, segment_index: int, fraction: float) -> np.ndarray:
        target = int(segment_index)
        positions = np.asarray(self.input_positions_angstrom, dtype=float)
        for segment in self.segments[:target]:
            positions = positions @ np.asarray(
                segment.deformation_gradient, dtype=float
            ).T
        deformation = np.asarray(
            self.segments[target].deformation_gradient, dtype=float
        )
        progress = _fraction(fraction)
        affine = (1.0 - progress) * np.eye(3) + progress * deformation
        return positions @ affine.T

    def context_payload(self) -> dict[str, object]:
        return {
            "input_cell_rows_angstrom": self.input_cell_rows_angstrom,
            "input_positions_angstrom": self.input_positions_angstrom,
            "atom_symbols": self.atom_symbols,
            "composition": self.composition,
            "standard_mass_amu": self.standard_mass_amu,
            "start_density_g_cm3": self.start_density_g_cm3,
            "segments": tuple(
                {
                    "segment_index": segment.segment_index,
                    "start_step": segment.start_step,
                    "end_step": segment.end_step,
                    "start_temperature_k": segment.start_temperature_k,
                    "stop_temperature_k": segment.stop_temperature_k,
                    "start_cell_rows_angstrom": segment.start_cell_rows_angstrom,
                    "end_cell_rows_angstrom": segment.end_cell_rows_angstrom,
                    "start_density_g_cm3": segment.start_density_g_cm3,
                    "end_density_g_cm3": segment.end_density_g_cm3,
                    "deformation_gradient": segment.deformation_gradient,
                    "declared_cell": segment.declared_cell,
                }
                for segment in self.segments
            ),
            "expected_dump_steps": self.expected_dump_steps,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.context_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _fraction(value: float) -> float:
    progress = float(value)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("cell program interpolation progress must be within [0, 1]")
    return progress


def resolve_cell_program_history(
    seed: Atoms,
    program: TemperatureProgram,
    *,
    trajectory_every_steps: int,
) -> ResolvedCellProgramHistory:
    """Resolve one path-free program against one physical periodic seed."""

    if not isinstance(seed, Atoms) or len(seed) <= 0:
        raise ValueError("cell program requires one non-empty ASE seed")
    cell = np.asarray(seed.cell.array, dtype=float)
    positions = np.asarray(seed.positions, dtype=float)
    pbc = np.asarray(seed.pbc)
    volume = abs(float(np.linalg.det(cell))) if np.all(np.isfinite(cell)) else math.nan
    if (
        pbc.shape != (3,)
        or not np.all(pbc)
        or cell.shape != (3, 3)
        or not np.all(np.isfinite(cell))
        or not math.isfinite(volume)
        or volume <= 1.0e-12
    ):
        raise ValueError(
            "cell program seed requires a finite nonsingular fully periodic cell"
        )
    if positions.shape != (len(seed), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("cell program seed positions must be finite atom-by-three coordinates")
    numbers = np.asarray(seed.numbers, dtype=int)
    if np.any(numbers <= 0):
        raise ValueError("cell program seed contains an unresolved standard atomic mass")
    symbols = tuple(seed.get_chemical_symbols())
    masses = np.asarray(
        [atomic_masses[atomic_numbers[symbol]] for symbol in symbols], dtype=float
    )
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("cell program seed contains an unresolved standard atomic mass")
    total_mass = float(np.sum(masses))
    cadence = trajectory_every_steps
    if (
        isinstance(cadence, bool)
        or not isinstance(cadence, int)
        or cadence <= 0
        or program.total_steps % cadence
    ):
        raise ValueError(
            "cell program trajectory cadence must emit the exact final endpoint"
        )
    start_density = _density(total_mass, cell)
    current_cell = cell.copy()
    current_density = start_density
    resolved: list[ResolvedCellProgramSegment] = []
    start_step = 0
    for index, segment in enumerate(program.segments):
        endpoint = segment.cell
        declared_cell: dict[str, object] | None = None
        if endpoint is not None and endpoint.mode == "isotropic_density":
            target = float(endpoint.stop_density_g_cm3)
            declared_cell = {
                "mode": endpoint.mode,
                "stop_density_g_cm3": target,
            }
        elif endpoint is not None:
            declared_cell = {
                "mode": endpoint.mode,
                "stop": dict(endpoint.symmetric_strain),
            }
        deformation = _deformation_from_declared(current_density, declared_cell)
        end_cell = current_cell @ deformation.T
        end_density = _density(total_mass, end_cell)
        resolved.append(
            ResolvedCellProgramSegment(
                segment_index=index,
                start_step=start_step,
                end_step=segment.end_step,
                start_temperature_k=segment.start_temperature_k,
                stop_temperature_k=segment.stop_temperature_k,
                start_cell_rows_angstrom=_matrix(current_cell),
                end_cell_rows_angstrom=_matrix(end_cell),
                start_density_g_cm3=current_density,
                end_density_g_cm3=end_density,
                deformation_gradient=_matrix(deformation),
                declared_cell=declared_cell,
            )
        )
        current_cell = end_cell
        current_density = end_density
        start_step = segment.end_step
    expected_steps = tuple(range(cadence, program.total_steps + 1, cadence))
    counts = Counter(symbols)
    composition = tuple(
        (symbol, int(counts[symbol])) for symbol in dict.fromkeys(symbols)
    )
    return ResolvedCellProgramHistory(
        input_cell_rows_angstrom=_matrix(cell),
        input_positions_angstrom=_positions(positions),
        atom_symbols=symbols,
        composition=composition,
        standard_mass_amu=total_mass,
        start_density_g_cm3=start_density,
        segments=tuple(resolved),
        expected_dump_steps=expected_steps,
    )


def restore_cell_program_history(
    payload: Mapping[str, object],
) -> ResolvedCellProgramHistory:
    """Restore one JSON-shaped resolved history for deterministic replay."""

    expected = {
        "input_cell_rows_angstrom",
        "input_positions_angstrom",
        "atom_symbols",
        "composition",
        "standard_mass_amu",
        "start_density_g_cm3",
        "segments",
        "expected_dump_steps",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("cell program history payload fields are incompatible")
    try:
        segments = tuple(
            ResolvedCellProgramSegment(
                segment_index=int(item["segment_index"]),
                start_step=int(item["start_step"]),
                end_step=int(item["end_step"]),
                start_temperature_k=float(item["start_temperature_k"]),
                stop_temperature_k=float(item["stop_temperature_k"]),
                start_cell_rows_angstrom=_matrix(
                    np.asarray(item["start_cell_rows_angstrom"], dtype=float)
                ),
                end_cell_rows_angstrom=_matrix(
                    np.asarray(item["end_cell_rows_angstrom"], dtype=float)
                ),
                start_density_g_cm3=float(item["start_density_g_cm3"]),
                end_density_g_cm3=float(item["end_density_g_cm3"]),
                deformation_gradient=_matrix(
                    np.asarray(item["deformation_gradient"], dtype=float)
                ),
                declared_cell=(
                    None
                    if item["declared_cell"] is None
                    else dict(item["declared_cell"])
                ),
            )
            for item in payload["segments"]
        )
        history = ResolvedCellProgramHistory(
            input_cell_rows_angstrom=_matrix(
                np.asarray(payload["input_cell_rows_angstrom"], dtype=float)
            ),
            input_positions_angstrom=_positions(
                np.asarray(payload["input_positions_angstrom"], dtype=float)
            ),
            atom_symbols=tuple(str(item) for item in payload["atom_symbols"]),
            composition=tuple(
                (str(item[0]), int(item[1])) for item in payload["composition"]
            ),
            standard_mass_amu=float(payload["standard_mass_amu"]),
            start_density_g_cm3=float(payload["start_density_g_cm3"]),
            segments=segments,
            expected_dump_steps=tuple(
                int(item) for item in payload["expected_dump_steps"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("cell program history payload is malformed") from exc
    _validate_restored_history(history)
    return history


def _validate_restored_history(history: ResolvedCellProgramHistory) -> None:
    cell = np.asarray(history.input_cell_rows_angstrom, dtype=float)
    positions = np.asarray(history.input_positions_angstrom, dtype=float)
    if (
        cell.shape != (3, 3)
        or not np.all(np.isfinite(cell))
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
        or positions.shape != (len(history.atom_symbols), 3)
        or not np.all(np.isfinite(positions))
        or not history.atom_symbols
    ):
        raise ValueError("cell program history input geometry is invalid")
    try:
        masses = np.asarray(
            [atomic_masses[atomic_numbers[symbol]] for symbol in history.atom_symbols],
            dtype=float,
        )
    except KeyError as exc:
        raise ValueError("cell program history atom symbols are invalid") from exc
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("cell program history standard masses are invalid")
    counts = Counter(history.atom_symbols)
    expected_composition = tuple(
        (symbol, int(counts[symbol]))
        for symbol in dict.fromkeys(history.atom_symbols)
    )
    expected_mass = float(np.sum(masses))
    expected_density = _density(expected_mass, cell)
    if (
        history.composition != expected_composition
        or not math.isclose(
            history.standard_mass_amu, expected_mass, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
        or not math.isclose(
            history.start_density_g_cm3,
            expected_density,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not history.segments
    ):
        raise ValueError("cell program history composition or density is inconsistent")
    current_cell = cell
    current_density = expected_density
    current_step = 0
    for index, segment in enumerate(history.segments):
        start_cell = np.asarray(segment.start_cell_rows_angstrom, dtype=float)
        end_cell = np.asarray(segment.end_cell_rows_angstrom, dtype=float)
        deformation = np.asarray(segment.deformation_gradient, dtype=float)
        if (
            segment.segment_index != index
            or segment.start_step != current_step
            or segment.end_step <= segment.start_step
            or not math.isfinite(segment.start_temperature_k)
            or segment.start_temperature_k <= 0.0
            or not math.isfinite(segment.stop_temperature_k)
            or segment.stop_temperature_k <= 0.0
            or start_cell.shape != (3, 3)
            or end_cell.shape != (3, 3)
            or deformation.shape != (3, 3)
            or not np.all(np.isfinite(start_cell))
            or not np.all(np.isfinite(end_cell))
            or not np.all(np.isfinite(deformation))
            or abs(float(np.linalg.det(end_cell))) <= 1.0e-12
            or not np.allclose(start_cell, current_cell, rtol=1.0e-12, atol=1.0e-12)
            or not math.isclose(
                segment.start_density_g_cm3,
                current_density,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or not np.allclose(
                end_cell,
                start_cell @ deformation.T,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError("cell program history segment chain is inconsistent")
        end_density = _density(expected_mass, end_cell)
        if not math.isclose(
            segment.end_density_g_cm3,
            end_density,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("cell program history segment density is inconsistent")
        expected_deformation = _deformation_from_declared(
            current_density, segment.declared_cell
        )
        if not np.allclose(
            deformation,
            expected_deformation,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("cell program history declared endpoint is inconsistent")
        current_cell = end_cell
        current_density = end_density
        current_step = segment.end_step
    steps = history.expected_dump_steps
    if (
        not steps
        or any(isinstance(step, bool) or step <= 0 for step in steps)
        or steps[-1] != current_step
    ):
        raise ValueError("cell program history dump schedule is inconsistent")
    cadence = steps[0]
    if steps != tuple(range(cadence, current_step + 1, cadence)):
        raise ValueError("cell program history dump schedule is inconsistent")


__all__ = [
    "ResolvedCellProgramHistory",
    "ResolvedCellProgramSegment",
    "resolve_cell_program_history",
    "restore_cell_program_history",
]
