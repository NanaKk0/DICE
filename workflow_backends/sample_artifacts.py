from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from ase import Atoms

from workflow_core.structures.canonical_identity import stamp_canonical_identity
from workflow_core.structures.formats import write_extxyz_structures
from workflow_artifacts.identities import RevisionedIdentity


TASK_CONTEXT_IDENTITY = RevisionedIdentity("dice.lifecycle.sample-task-context", 1)
TASK_CONTEXT_SCHEMA = TASK_CONTEXT_IDENTITY.name
TASK_CONTEXT_SCHEMA_REVISION = TASK_CONTEXT_IDENTITY.revision
EXPLICIT_TASK_CONTEXT_SCHEMA_REVISION = 2
CELL_PROGRAM_TASK_CONTEXT_SCHEMA_REVISION = 3
_HISTORICAL_TASK_CONTEXT_SCHEMA = "dice.v2.sample-task-context.v1"


@dataclass(frozen=True, slots=True)
class SampleTaskContext:
    schema: str
    task_index: int
    system_id: str
    ensemble: str
    temperature_k: float
    pressure_bar: float | None
    steps: int
    cadence: int
    selected_seed_id: str
    schema_revision: int = TASK_CONTEXT_SCHEMA_REVISION

    @classmethod
    def create(
        cls,
        *,
        task_index: int,
        system_id: str,
        ensemble: str,
        temperature_k: float,
        pressure_bar: float | None,
        steps: int,
        cadence: int,
        selected_seed_id: str,
    ) -> "SampleTaskContext":
        return cls(
            schema=TASK_CONTEXT_SCHEMA,
            task_index=int(task_index),
            system_id=str(system_id),
            ensemble=str(ensemble),
            temperature_k=float(temperature_k),
            pressure_bar=None if pressure_bar is None else float(pressure_bar),
            steps=int(steps),
            cadence=int(cadence),
            selected_seed_id=str(selected_seed_id),
        )


@dataclass(frozen=True, slots=True)
class ExplicitSampleTaskContext:
    schema: str
    schema_revision: int
    task_index: int
    task_id: str
    condition_group_id: str
    kind: str
    sampling_role_id: str
    selection_group_id: str
    system_id: str
    source_frame_index: int
    selected_structure_id: str
    temperature_k: float | None
    pressure_bar: float | None
    total_steps: int
    trajectory_every_steps: int
    native_timestep: float
    recipe_binding: str
    recipe_sha256: str
    rendered_input_sha256: str
    setup_sha256: str
    native_seed: int
    ordered_models: tuple[str, ...]
    model_digests: tuple[str, ...]
    temperature_program: tuple[dict[str, object], ...] = ()
    segment_boundaries: tuple[int, ...] = ()
    schedule_digests: tuple[tuple[str, str], ...] = ()

    @property
    def cadence(self) -> int:
        return self.trajectory_every_steps

    @property
    def steps(self) -> int:
        return self.total_steps

    @property
    def ensemble(self) -> str:
        if self.kind == 'md_mc':
            return 'npt' if self.pressure_bar is not None else 'nvt'
        return (
            "nvt"
            if self.kind in {"nvt_program", "nvt_cell_program", "md_mc"}
            else "npt" if self.kind == "npt_program" else self.kind
        )

    @property
    def selected_seed_id(self) -> str:
        return self.selected_structure_id


@dataclass(frozen=True, slots=True)
class CellProgramSampleTaskContext(ExplicitSampleTaskContext):
    resolved_cell_history: dict[str, object] = field(default_factory=dict)
    cell_history_sha256: str = ""
    expected_dump_steps: tuple[int, ...] = ()

    @property
    def ensemble(self) -> str:
        return "nvt"

    @property
    def resolved_history(self):
        from workflow_core.structures.cell_program import (
            restore_cell_program_history,
        )

        return restore_cell_program_history(self.resolved_cell_history)


@dataclass(frozen=True, slots=True)
class MdMcSampleTaskContext(ExplicitSampleTaskContext):
    md_mc: dict[str, object] = field(default_factory=dict)


def explicit_task_seed(
    *, iter_index: int, round_index: int, sampling_role_id: str
) -> int:
    raw = int.from_bytes(
        hashlib.sha256(
            f"explicit-seed\0{int(iter_index)}\0{int(round_index)}\0{sampling_role_id}".encode()
        ).digest()[:4],
        "big",
    )
    return raw % 900_000_000 + 1


def canonical_selected_seed(seed: Atoms, *, system_id: str) -> Atoms:
    return stamp_canonical_identity(seed, system_id=system_id, structure_source="external")


def task_seed(
    *,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    system_id: str,
) -> int:
    raw = int.from_bytes(
        hashlib.sha256(
            f"{Path(run_json_path).resolve()}\0{int(iter_index)}\0{int(round_index)}\0{system_id}".encode()
        ).digest()[:4],
        "big",
    )
    return raw % 900_000_000 + 1


def selected_seed_index(
    *,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    system_id: str,
    seed_count: int,
) -> int:
    if int(seed_count) <= 0:
        raise ValueError("lifecycle selected seed source must be non-empty")
    return task_seed(
        run_json_path=run_json_path,
        iter_index=iter_index,
        round_index=round_index,
        system_id=system_id,
    ) % int(seed_count)


def publish_selected_seeds(path: str | Path, seeds: Sequence[Atoms]) -> Path:
    return write_extxyz_structures(path, list(seeds), allow_empty=False)


def write_task_context(
    path: str | Path,
    context: (
        SampleTaskContext
        | ExplicitSampleTaskContext
        | CellProgramSampleTaskContext
    ),
) -> Path:
    target = Path(path)
    target.write_text(json.dumps(asdict(context), sort_keys=True), encoding="utf-8")
    return target


def load_task_context(path: str | Path) -> SampleTaskContext | ExplicitSampleTaskContext:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed lifecycle sample task context: {target}") from exc
    if isinstance(payload, dict) and payload.get('schema_revision') == 4:
        from workflow_core.common.md_mc import IdentitySwapParameters
        expected = set(MdMcSampleTaskContext.__dataclass_fields__)
        if set(payload) != expected or payload.get('schema') != TASK_CONTEXT_SCHEMA or payload.get('kind') != 'md_mc':
            raise ValueError('invalid MDMC task context')
        for key in ('task_index', 'source_frame_index', 'total_steps', 'trajectory_every_steps', 'native_seed'):
            if type(payload[key]) is not int or payload[key] < (0 if key in {'task_index', 'source_frame_index'} else 1):
                raise ValueError('invalid MDMC integer task context field')
        for key in ('temperature_k', 'native_timestep'):
            if isinstance(payload[key], bool) or not isinstance(payload[key], (int, float)):
                raise ValueError('invalid MDMC numeric task context field')
        for key in ('ordered_models', 'model_digests', 'temperature_program', 'segment_boundaries', 'schedule_digests'):
            payload[key] = tuple(payload[key])
        context = MdMcSampleTaskContext(**payload)
        required = {'parameters', 'seed_cell', 'seed_numbers', 'type_map', 'member_ids', 'execution_mode'}
        if set(context.md_mc) not in (required, required | {'mc_seed'}):
            raise ValueError('invalid MDMC protocol context fields')
        parameters = context.md_mc['parameters']
        IdentitySwapParameters(**parameters)
        import numpy as np
        from ase.data import atomic_numbers
        cell = np.asarray(context.md_mc['seed_cell'], dtype=float)
        numbers = context.md_mc['seed_numbers']
        symbols = context.md_mc['type_map']
        members = context.md_mc['member_ids']
        if (cell.shape != (3, 3) or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0
            or not numbers or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in numbers)
            or not symbols or len(set(symbols)) != len(symbols) or any(s not in atomic_numbers for s in symbols)
            or not set(numbers) <= {atomic_numbers[s] for s in symbols}
            or not {atomic_numbers[s] for s in parameters['elements']} <= set(numbers)
            or len(members) != len(context.ordered_models) or len(set(members)) != len(members)
            or any(not isinstance(m, str) or not m for m in members)
            or context.md_mc['execution_mode'] not in {'ordinary', 'kokkos'}
            or ('mc_seed' in context.md_mc and (isinstance(context.md_mc['mc_seed'], bool) or not isinstance(context.md_mc['mc_seed'], int) or context.md_mc['mc_seed'] <= 0))):
            raise ValueError('invalid MDMC seed, mapping or execution context')
        if (context.task_id != f'task_{context.task_index:06d}' or context.task_index < 0
            or context.system_id != context.selection_group_id or not context.sampling_role_id
            or not context.condition_group_id
            or (context.pressure_bar is not None and (isinstance(context.pressure_bar, bool) or not isinstance(context.pressure_bar, (int, float)) or not math.isfinite(context.pressure_bar)))
            or context.temperature_k is None or not math.isfinite(context.temperature_k) or context.temperature_k <= 0
            or context.steps < parameters['every_md_steps'] or context.cadence <= 0
            or not math.isfinite(context.native_timestep) or context.native_timestep <= 0
            or not context.ordered_models or len(context.ordered_models) != len(context.model_digests)
            or any(len(d) != 64 for d in (*context.model_digests, context.recipe_sha256, context.setup_sha256, context.rendered_input_sha256))):
            raise ValueError('invalid MDMC task context semantics')
        return context
    if (
        isinstance(payload, dict)
        and payload.get("schema") == TASK_CONTEXT_SCHEMA
        and payload.get("schema_revision")
        == CELL_PROGRAM_TASK_CONTEXT_SCHEMA_REVISION
    ):
        expected = {
            field.name
            for field in CellProgramSampleTaskContext.__dataclass_fields__.values()
        }
        if set(payload) != expected:
            raise ValueError(
                f"malformed lifecycle cell-program task context fields: {target}"
            )
        payload["ordered_models"] = tuple(payload["ordered_models"])
        payload["model_digests"] = tuple(payload["model_digests"])
        payload["temperature_program"] = tuple(payload["temperature_program"])
        payload["segment_boundaries"] = tuple(payload["segment_boundaries"])
        payload["schedule_digests"] = tuple(
            tuple(item) for item in payload["schedule_digests"]
        )
        payload["expected_dump_steps"] = tuple(payload["expected_dump_steps"])
        context = CellProgramSampleTaskContext(**payload)
        try:
            history = context.resolved_history
        except ValueError as exc:
            raise ValueError(
                f"invalid lifecycle cell-program task context: {target}"
            ) from exc
        if (
            context.task_index < 0
            or context.task_id != f"task_{context.task_index:06d}"
            or not context.condition_group_id
            or context.kind != "nvt_cell_program"
            or not context.sampling_role_id
            or not context.selection_group_id
            or context.system_id != context.selection_group_id
            or isinstance(context.source_frame_index, bool)
            or not isinstance(context.source_frame_index, int)
            or context.source_frame_index < 0
            or not context.selected_structure_id
            or context.temperature_k is not None
            or context.pressure_bar is not None
            or context.total_steps <= 0
            or context.trajectory_every_steps <= 0
            or not math.isfinite(context.native_timestep)
            or context.native_timestep <= 0
            or context.native_seed <= 0
            or not context.ordered_models
            or len(context.ordered_models) != len(context.model_digests)
            or any(
                len(digest) != 64
                for digest in (
                    context.recipe_sha256,
                    context.rendered_input_sha256,
                    context.setup_sha256,
                    context.cell_history_sha256,
                    *context.model_digests,
                    *(digest for _, digest in context.schedule_digests),
                )
            )
            or not context.temperature_program
            or context.segment_boundaries
            != tuple(segment.end_step for segment in history.segments)
            or context.total_steps != history.segments[-1].end_step
            or context.expected_dump_steps != history.expected_dump_steps
            or context.cell_history_sha256 != history.sha256
            or context.temperature_program
            != tuple(
                {
                    "start_temperature_k": segment.start_temperature_k,
                    "stop_temperature_k": segment.stop_temperature_k,
                    "steps": segment.steps,
                    "end_step": segment.end_step,
                    **(
                        {"cell": segment.declared_cell}
                        if segment.declared_cell is not None
                        else {}
                    ),
                }
                for segment in history.segments
            )
        ):
            raise ValueError(
                f"invalid lifecycle cell-program task context: {target}"
            )
        return context
    if (
        isinstance(payload, dict)
        and payload.get("schema") == TASK_CONTEXT_SCHEMA
        and payload.get("schema_revision") == EXPLICIT_TASK_CONTEXT_SCHEMA_REVISION
    ):
        expected = {field.name for field in ExplicitSampleTaskContext.__dataclass_fields__.values()}
        if set(payload) != expected:
            raise ValueError(f"malformed lifecycle explicit sample task context fields: {target}")
        payload["ordered_models"] = tuple(payload["ordered_models"])
        payload["model_digests"] = tuple(payload["model_digests"])
        payload["temperature_program"] = tuple(payload["temperature_program"])
        payload["segment_boundaries"] = tuple(payload["segment_boundaries"])
        payload["schedule_digests"] = tuple(tuple(item) for item in payload["schedule_digests"])
        context = ExplicitSampleTaskContext(**payload)
        program_valid = True
        if context.kind in {"nvt_program", "npt_program"}:
            expected_end = 0
            if (
                context.temperature_k is not None
                or (
                    (context.kind == "npt_program")
                    != (context.pressure_bar is not None)
                )
                or not context.temperature_program
                or len(context.schedule_digests) != 3
                or tuple(name for name, _ in context.schedule_digests)
                != (
                    "dice_temperature_start.dat",
                    "dice_temperature_stop.dat",
                    "dice_temperature_steps.dat",
                )
            ):
                program_valid = False
            for segment in context.temperature_program:
                if not isinstance(segment, dict) or set(segment) != {
                    "start_temperature_k",
                    "stop_temperature_k",
                    "steps",
                    "end_step",
                }:
                    program_valid = False
                    break
                try:
                    start = float(segment["start_temperature_k"])
                    stop = float(segment["stop_temperature_k"])
                    segment_steps = int(segment["steps"])
                    end_step = int(segment["end_step"])
                except (TypeError, ValueError):
                    program_valid = False
                    break
                expected_end += segment_steps
                if (
                    not math.isfinite(start)
                    or start <= 0
                    or not math.isfinite(stop)
                    or stop <= 0
                    or segment_steps <= 0
                    or end_step != expected_end
                ):
                    program_valid = False
                    break
            if (
                expected_end != context.total_steps
                or context.segment_boundaries
                != tuple(item["end_step"] for item in context.temperature_program)
                or any(len(digest) != 64 for _, digest in context.schedule_digests)
            ):
                program_valid = False
        elif (
            context.temperature_program
            or context.segment_boundaries
            or context.schedule_digests
        ):
            program_valid = False
        if (
            context.task_index < 0
            or context.task_id != f"task_{context.task_index:06d}"
            or not context.condition_group_id
            or context.kind not in {"nvt", "npt", "nvt_program", "npt_program"}
            or not context.sampling_role_id
            or not context.selection_group_id
            or context.system_id != context.selection_group_id
            or isinstance(context.source_frame_index, bool)
            or not isinstance(context.source_frame_index, int)
            or context.source_frame_index < 0
            or not context.selected_structure_id
            or context.total_steps <= 0
            or context.trajectory_every_steps <= 0
            or not math.isfinite(context.native_timestep)
            or context.native_timestep <= 0
            or context.native_seed <= 0
            or not context.ordered_models
            or len(context.ordered_models) != len(context.model_digests)
            or any(len(digest) != 64 for digest in (
                context.recipe_sha256,
                context.rendered_input_sha256,
                context.setup_sha256,
                *context.model_digests,
            ))
            or (context.kind in {"npt", "npt_program"}) != (context.pressure_bar is not None)
            or (context.kind not in {"nvt_program", "npt_program"} and context.temperature_k is None)
            or not program_valid
        ):
            raise ValueError(f"invalid lifecycle explicit sample task context: {target}")
        return context
    canonical_fields = {
        "schema", "schema_revision",
        "task_index",
        "system_id",
        "ensemble",
        "temperature_k",
        "pressure_bar",
        "steps",
        "cadence",
        "selected_seed_id",
    }
    historical_fields = canonical_fields - {"schema_revision"}
    if not isinstance(payload, dict) or set(payload) not in {
        frozenset(canonical_fields), frozenset(historical_fields)
    }:
        raise ValueError(f"malformed lifecycle sample task context fields: {target}")
    if payload.get("schema") == _HISTORICAL_TASK_CONTEXT_SCHEMA and set(payload) == historical_fields:
        payload = {
            **payload,
            "schema": TASK_CONTEXT_SCHEMA,
            "schema_revision": TASK_CONTEXT_SCHEMA_REVISION,
        }
    context = SampleTaskContext(**payload)
    temperature_valid = (
        isinstance(context.temperature_k, (int, float))
        and not isinstance(context.temperature_k, bool)
        and math.isfinite(float(context.temperature_k))
        and float(context.temperature_k) > 0
    )
    pressure_valid = (
        context.pressure_bar is not None
        and isinstance(context.pressure_bar, (int, float))
        and not isinstance(context.pressure_bar, bool)
        and math.isfinite(float(context.pressure_bar))
    )
    if (
        context.schema != TASK_CONTEXT_SCHEMA
        or context.schema_revision != TASK_CONTEXT_SCHEMA_REVISION
        or isinstance(context.schema_revision, bool)
        or context.task_index < 0
        or not context.system_id
        or context.ensemble not in {"nvt", "npt"}
        or not temperature_valid
        or (context.ensemble == "nvt" and context.pressure_bar is not None)
        or (context.ensemble == "npt" and not pressure_valid)
        or context.steps <= 0
        or context.cadence <= 0
        or not context.selected_seed_id
    ):
        raise ValueError(f"invalid lifecycle sample task context: {target}")
    return context


__all__ = [
    "CELL_PROGRAM_TASK_CONTEXT_SCHEMA_REVISION",
    "TASK_CONTEXT_SCHEMA",
    "TASK_CONTEXT_SCHEMA_REVISION",
    "TASK_CONTEXT_IDENTITY",
    "EXPLICIT_TASK_CONTEXT_SCHEMA_REVISION",
    "ExplicitSampleTaskContext",
    "CellProgramSampleTaskContext",
    "SampleTaskContext",
    "canonical_selected_seed",
    "explicit_task_seed",
    "load_task_context",
    "publish_selected_seeds",
    "selected_seed_index",
    "task_seed",
    "write_task_context",
]
