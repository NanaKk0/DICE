from __future__ import annotations

import json
import math
from pathlib import Path

from ase.calculators.lammps.coordinatetransform import Prism
from ase.io import iread
import numpy as np

from workflow_backends.sample_artifacts import CellProgramSampleTaskContext
from workflow_core.structures.cell_program import ResolvedCellProgramHistory


_CELL_COMPONENT_RTOL = 1.0e-8
_CELL_COMPONENT_ATOL_ANGSTROM = 1.0e-10
_DENSITY_RTOL = 1.0e-8
_AMU_PER_ANGSTROM3_TO_G_CM3 = 1.66053906660


def _prism(cell_rows: object) -> tuple[float, ...]:
    values = Prism(
        np.asarray(cell_rows, dtype=float),
        pbc=True,
        reduce_cell=False,
    ).get_lammps_prism()
    return tuple(float(item) for item in values)


def build_lammps_cell_schedule(
    history: ResolvedCellProgramHistory,
) -> dict[str, object]:
    intervals: list[dict[str, object]] = []
    emitted = set(history.expected_dump_steps)
    for segment in history.segments:
        points = sorted(
            {
                step
                for step in emitted
                if segment.start_step < step <= segment.end_step
            }
            | {segment.end_step}
        )
        start = segment.start_step
        for end in points:
            start_fraction = (start - segment.start_step) / segment.steps
            end_fraction = (end - segment.start_step) / segment.steps
            start_temperature = (
                (1.0 - start_fraction) * segment.start_temperature_k
                + start_fraction * segment.stop_temperature_k
            )
            stop_temperature = (
                (1.0 - end_fraction) * segment.start_temperature_k
                + end_fraction * segment.stop_temperature_k
            )
            start_prism = _prism(
                history.cell_at(segment.segment_index, start_fraction)
            )
            end_prism = _prism(
                history.cell_at(segment.segment_index, end_fraction)
            )
            native_fraction = (end - start) / (segment.end_step - start)
            deform_final = tuple(
                start_value + (end_value - start_value) / native_fraction
                for start_value, end_value in zip(
                    start_prism, end_prism, strict=True
                )
            )
            intervals.append(
                {
                    "segment_index": segment.segment_index,
                    "segment_start_step": segment.start_step,
                    "segment_end_step": segment.end_step,
                    "start_step": start,
                    "end_step": end,
                    "steps": end - start,
                    "start_temperature_k": start_temperature,
                    "stop_temperature_k": stop_temperature,
                    "start_prism_angstrom": start_prism,
                    "end_prism_angstrom": end_prism,
                    "deform_final_prism_angstrom": deform_final,
                }
            )
            start = end
    initial_prism = _prism(history.input_cell_rows_angstrom)
    target_tilts = tuple(
        value
        for interval in intervals
        for value in tuple(interval["end_prism_angstrom"])[3:]
    )
    requires_triclinic_conversion = all(
        abs(float(value)) <= 1.0e-14 for value in initial_prism[3:]
    ) and any(abs(float(value)) > 1.0e-14 for value in target_tilts)
    return {
        "schema": "dice.lammps-cell-program-schedule",
        "schema_revision": 1,
        "cell_history_sha256": history.sha256,
        "expected_dump_steps": history.expected_dump_steps,
        "atom_symbols": history.atom_symbols,
        "initial_prism_angstrom": initial_prism,
        "requires_triclinic_conversion": requires_triclinic_conversion,
        "intervals": tuple(intervals),
    }


def write_lammps_cell_schedule(
    path: str | Path,
    history: ResolvedCellProgramHistory,
) -> dict[str, object]:
    payload = build_lammps_cell_schedule(history)
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return payload


def render_lammps_cell_program_input(
    recipe_text: str,
    schedule: dict[str, object],
) -> str:
    lines = [recipe_text.rstrip(), "# DICE_BEGIN_CELL_PROGRAM"]
    intervals = tuple(schedule["intervals"])
    initial_prism = tuple(schedule["initial_prism_angstrom"])
    uses_tilt_factors = any(
        abs(float(value)) > 1.0e-14
        for value in (
            *initial_prism[3:],
            *(
                tilt
                for interval in intervals
                for tilt in tuple(interval["end_prism_angstrom"])[3:]
            ),
        )
    )
    active_segment: int | None = None
    for interval in intervals:
        segment_index = int(interval["segment_index"])
        if segment_index != active_segment:
            if active_segment is not None:
                lines.append(f"unfix dice_cell_deform_{active_segment}")
                lines.append(f"unfix dice_cell_nvt_{active_segment}")
            segment_intervals = tuple(
                item
                for item in intervals
                if int(item["segment_index"]) == segment_index
            )
            lines.append(
                f"fix dice_cell_nvt_{segment_index} all nvt temp "
                f"{segment_intervals[0]['start_temperature_k']} "
                f"{segment_intervals[-1]['stop_temperature_k']} 0.1"
            )
            x, y, z, xy, xz, yz = segment_intervals[-1][
                "end_prism_angstrom"
            ]
            tilt_targets = (
                f" xy final {xy} xz final {xz} yz final {yz}"
                if uses_tilt_factors
                else ""
            )
            lines.append(
                f"fix dice_cell_deform_{segment_index} all deform 1 "
                f"x final 0.0 {x} y final 0.0 {y} z final 0.0 {z} "
                f"{tilt_targets} "
                "remap x flip no units box"
            )
            active_segment = segment_index
        lines.append(
            f"run {interval['steps']} start {interval['segment_start_step']} "
            f"stop {interval['segment_end_step']}"
        )
    if active_segment is not None:
        lines.append(f"unfix dice_cell_deform_{active_segment}")
        lines.append(f"unfix dice_cell_nvt_{active_segment}")
    lines.append("# DICE_END_CELL_PROGRAM")
    return "\n".join(lines) + "\n"


def validate_lammps_cell_program_trajectory(
    *,
    task_dir: str | Path,
    context: CellProgramSampleTaskContext,
    type_map: tuple[str, ...],
) -> None:
    history = context.resolved_history
    schedule_path = Path(task_dir) / "dice_cell_schedule.json"
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("LAMMPS cell-program schedule evidence is malformed") from exc
    if schedule != json.loads(
        json.dumps(build_lammps_cell_schedule(history))
    ):
        raise ValueError("LAMMPS cell-program schedule disagrees with task history")
    raw_frames = tuple(
        iread(
            Path(task_dir) / "traj.lammpstrj",
            index=":",
            format="lammps-dump-text",
            specorder=list(type_map),
        )
    )
    if any(int(frame.info.get("timestep", -1)) < 0 for frame in raw_frames):
        raise ValueError("LAMMPS cell-program trajectory step is malformed")
    frames = tuple(
        frame
        for frame in raw_frames
        if int(frame.info.get("timestep", -1)) > 0
    )
    if len(frames) != len(context.expected_dump_steps):
        raise ValueError("LAMMPS cell-program trajectory inventory is incomplete")
    intervals_by_step = {
        int(item["end_step"]): item for item in schedule["intervals"]
    }
    for frame, step in zip(frames, context.expected_dump_steps, strict=True):
        if int(frame.info.get("timestep", -1)) != step:
            raise ValueError("LAMMPS cell-program trajectory steps are reordered")
        if tuple(frame.get_chemical_symbols()) != history.atom_symbols:
            raise ValueError("LAMMPS cell-program atom identity/order changed")
        interval = intervals_by_step[step]
        x, y, z, xy, xz, yz = interval["end_prism_angstrom"]
        expected_cell = np.asarray(
            ((x, 0.0, 0.0), (xy, y, 0.0), (xz, yz, z)), dtype=float
        )
        actual_cell = np.asarray(frame.cell.array, dtype=float)
        if (
            actual_cell.shape != (3, 3)
            or not np.all(np.isfinite(actual_cell))
            or abs(float(np.linalg.det(actual_cell))) <= 1.0e-12
            or not np.allclose(
                actual_cell,
                expected_cell,
                rtol=_CELL_COMPONENT_RTOL,
                atol=_CELL_COMPONENT_ATOL_ANGSTROM,
            )
        ):
            raise ValueError("LAMMPS cell-program trajectory cell endpoint mismatch")
        expected_density = (
            history.standard_mass_amu
            * _AMU_PER_ANGSTROM3_TO_G_CM3
            / abs(float(np.linalg.det(expected_cell)))
        )
        actual_density = (
            history.standard_mass_amu
            * _AMU_PER_ANGSTROM3_TO_G_CM3
            / abs(float(np.linalg.det(actual_cell)))
        )
        if not math.isclose(
            actual_density,
            expected_density,
            rel_tol=_DENSITY_RTOL,
            abs_tol=0.0,
        ):
            raise ValueError("LAMMPS cell-program trajectory density endpoint mismatch")


__all__ = [
    "build_lammps_cell_schedule",
    "render_lammps_cell_program_input",
    "validate_lammps_cell_program_trajectory",
    "write_lammps_cell_schedule",
]
