from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

from ase import Atoms
from ase.io import iread

from workflow_backends.observation_source import (
    ObservationLocator,
    ObservationRecord,
    observation_identity,
)
from workflow_backends.sample_artifacts import (
    CellProgramSampleTaskContext,
    ExplicitSampleTaskContext,
    SampleTaskContext,
    canonical_selected_seed,
    load_task_context,
    selected_seed_index,
    explicit_task_seed,
    task_seed,
)
from workflow_backends.lammps.cell_program import (
    validate_lammps_cell_program_trajectory,
)
from workflow_backends.lammps.execution_mode import (
    KOKKOS_COMMAND,
    ORDINARY_COMMAND,
    select_lammps_execution_mode,
)
from workflow_artifacts.lifecycle.sample_conditions import SamplingPlan, SamplingTaskCondition
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.cell_program import resolve_cell_program_history
from workflow_core.structures.formats import (
    extract_lammpstrj_selected_frames,
    read_extxyz_structures,
)
from workflow_backends.sampling import (
    SamplingSeedSelection,
    seed_selection_manifest_payload,
)

_UNSPECIFIED_PROFILE = object()


class LammpsObservationSource:
    """Screen-owned catalog and bounded replay over committed LAMMPS tasks."""

    def __init__(
        self,
        *,
        sample_dir: str | Path,
        run_json_path: str | Path,
        iter_index: int,
        round_index: int,
        type_map: Sequence[str],
        model_count: int,
        contexts: Sequence[SampleTaskContext | ExplicitSampleTaskContext] | None = None,
        plan: ExplicitSamplingPlan | None = None,
        accepted_kinds: tuple[str, ...] | None = None,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.run_json_path = Path(run_json_path)
        self.iter_index = int(iter_index)
        self.round_index = int(round_index)
        self.type_map = tuple(str(item) for item in type_map)
        self.model_count = int(model_count)
        self.contexts = None if contexts is None else tuple(contexts)
        self.plan = plan
        self.accepted_kinds = accepted_kinds
        self._catalog: tuple[ObservationRecord, ...] | None = None

    @property
    def failed_task_count(self):
        count = 0
        root = self.sample_dir/'tasks'
        for task in (root.iterdir() if root.is_dir() else ()):
            if not task.is_dir() or not re.fullmatch(r'task_\d{6}', task.name):
                continue
            path = task/'md_mc_result.json'
            if path.is_file():
                count += json.loads(path.read_text()).get('status') == 'failed'
            else:
                count += (task/'sampling_failure.json').is_file()
        return count

    def catalog(self) -> tuple[ObservationRecord, ...]:
        if self._catalog is not None:
            return self._catalog
        records: list[ObservationRecord] = []
        task_dirs = _ordered_task_dirs(self.sample_dir)
        planned_tasks = (
            self.plan.tasks_for_round(self.round_index)
            if self.plan is not None
            else None
        )
        if planned_tasks is not None and len(task_dirs) != len(planned_tasks):
            raise ValueError(
                "LAMMPS explicit sample task inventory disagrees with the resolved plan; rebuild rN/sample"
            )
        for task_index, task_dir in enumerate(task_dirs):
            if (
                planned_tasks is not None
                and self.accepted_kinds is not None
                and planned_tasks[task_index].kind not in self.accepted_kinds
            ):
                continue
            context = (
                load_task_context(task_dir / "task_context.json")
                if self.contexts is None
                else self.contexts[task_index]
            )
            if context.task_index != task_index:
                raise ValueError("LAMMPS sample task_context task index mismatch; rebuild rN/sample")
            explicit_context = isinstance(context, ExplicitSampleTaskContext)
            if planned_tasks is not None:
                condition = planned_tasks[task_index]
                expected_program = (
                    _program_payload(condition.program)
                    if condition.program is not None
                    else ()
                )
                if not explicit_context or (
                    context.task_id != condition.task_id
                    or context.condition_group_id != condition.condition_group_id
                    or context.sampling_role_id != condition.sampling_role_id
                    or context.selection_group_id != condition.selection_group_id
                    or context.system_id != condition.selection_group_id
                    or context.kind != condition.kind
                    or context.ensemble != condition.ensemble
                    or context.temperature_k != condition.temperature_k
                    or context.pressure_bar != condition.pressure_bar
                    or context.steps != condition.steps
                    or context.recipe_binding != condition.recipe_binding
                    or context.temperature_program != expected_program
                ):
                    raise ValueError(
                        "LAMMPS explicit task provenance disagrees with the resolved plan; rebuild rN/sample"
                    )
            failure_path = task_dir / "sampling_failure.json"
            if getattr(context, 'kind', None) == 'md_mc':
                from workflow_backends.lammps.identity_swap import validate_result
                _ctx, result = validate_result(task_dir)
                if planned_tasks is not None and context.md_mc['parameters'] != json.loads(json.dumps(asdict(condition.md_mc))):
                    raise ValueError('MDMC parameters disagree with resolved task')
                for frame_index, row in enumerate(result['observations']):
                    step = row['step']
                    identity = observation_identity(run_json_path=self.run_json_path,
                        iter_index=self.iter_index, round_index=self.round_index,
                        system_id=context.system_id, temperature_k=context.temperature_k,
                        pressure_bar=context.pressure_bar, md_step=step, sampling_role_id=context.sampling_role_id,
                        condition_group_id=context.condition_group_id, task_id=context.task_id)
                    records.append(ObservationRecord(structure_id=identity, system_id=context.system_id,
                        task_index=task_index, frame_index=frame_index, md_step=step,
                        student_uncertainty=row['uncertainty'],
                        locator=ObservationLocator('LAMMPS', task_index, frame_index, step),
                        sampling_role_id=context.sampling_role_id, selection_group_id=context.selection_group_id,
                        condition_group_id=context.condition_group_id, task_id=context.task_id))
                continue
            if failure_path.exists() or failure_path.is_symlink():
                terminals = validate_lammps_sampling_failure(
                    task_dir=task_dir,
                    cadence=context.cadence,
                    steps=context.steps,
                    require_deviation=self.model_count > 1,
                    expected_system_id=context.system_id,
                )
                if terminals is None:
                    continue
                trajectory_path = _committed_trajectory_path(task_dir)
                trajectory_rows = _scan_lammpstrj(trajectory_path)
            else:
                trajectory_path = _committed_trajectory_path(task_dir)
                trajectory_rows = _scan_lammpstrj(trajectory_path)
                terminals = _validate_lammps_task_terminal_rows(
                    root=task_dir,
                    cadence=context.cadence,
                    steps=context.steps,
                    require_deviation=self.model_count > 1,
                    require_exact_cadence=explicit_context,
                    trajectory_rows=trajectory_rows,
                )
            if isinstance(context, CellProgramSampleTaskContext):
                if failure_path.exists() or failure_path.is_symlink():
                    raise ValueError(
                        "LAMMPS cell-program tasks may not publish partial failure evidence"
                    )
                validate_lammps_cell_program_trajectory(
                    task_dir=task_dir,
                    context=context,
                    type_map=self.type_map,
                )
            deviations = {
                step: value
                for step, value in _max_deviation_rows(task_dir / "model_devi.out").items()
                if step > 0
            } if self.model_count > 1 else {}
            trajectory_steps = {step for _, step in trajectory_rows if step > 0}
            if self.model_count > 1 and set(deviations) != trajectory_steps:
                raise ValueError("LAMMPS trajectory/model_devi pairing mismatch; rebuild rN/sample")
            for frame_index, step in trajectory_rows:
                if step <= 0:
                    continue
                if step % context.cadence:
                    raise ValueError("LAMMPS observation cadence mismatch; rebuild rN/sample")
                uncertainty = 0.0 if self.model_count == 1 else deviations[step]
                temperature_k = (
                    _program_temperature_at_step(context.temperature_program, step)
                    if explicit_context and context.kind == "npt_program"
                    else (
                        context.temperature_k
                        if context.temperature_k is not None
                        else context.temperature_program[0]["start_temperature_k"]
                    )
                )
                identity = observation_identity(
                    run_json_path=self.run_json_path,
                    iter_index=self.iter_index,
                    round_index=self.round_index,
                    system_id=context.system_id,
                    temperature_k=float(temperature_k),
                    pressure_bar=context.pressure_bar,
                    md_step=step,
                    sampling_role_id=(context.sampling_role_id if explicit_context else None),
                    condition_group_id=(context.condition_group_id if explicit_context else None),
                    task_id=(context.task_id if explicit_context else None),
                )
                locator = ObservationLocator("LAMMPS", task_index, frame_index, step)
                records.append(
                    ObservationRecord(
                        structure_id=identity,
                        system_id=context.system_id,
                        task_index=task_index,
                        frame_index=frame_index,
                        md_step=step,
                        student_uncertainty=uncertainty,
                        locator=locator,
                        sampling_role_id=(context.sampling_role_id if explicit_context else None),
                        selection_group_id=(context.selection_group_id if explicit_context else None),
                        condition_group_id=(context.condition_group_id if explicit_context else None),
                        task_id=(context.task_id if explicit_context else None),
                    )
                )
        if not records:
            raise ValueError("LAMMPS native source has no positive-step observations; rebuild rN/sample")
        self._catalog = tuple(records)
        return self._catalog

    def replay(
        self,
        ordered_identities: Sequence[str],
        *,
        chunk_size: int,
    ) -> Iterator[tuple[Atoms, ...]]:
        size = int(chunk_size)
        if size <= 0:
            raise ValueError("LAMMPS observation replay chunk_size must be positive")
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("LAMMPS replay identities must be unique and non-empty")
        by_id = {record.structure_id: record for record in self.catalog()}
        missing = [identity for identity in identities if identity not in by_id]
        if missing:
            raise ValueError(f"LAMMPS replay identity is absent from the catalog: {missing[0]}")
        tasks = _ordered_task_dirs(self.sample_dir)
        catalog_order = {record.structure_id: index for index, record in enumerate(self.catalog())}
        requested_positions = tuple(catalog_order[identity] for identity in identities)
        if all(left < right for left, right in zip(requested_positions, requested_positions[1:])):
            yield from self._replay_ordered_subset(identities, by_id=by_id, tasks=tasks, chunk_size=size)
            return
        for start in range(0, len(identities), size):
            chunk_ids = identities[start : start + size]
            requested = [by_id[identity] for identity in chunk_ids]
            loaded: dict[str, Atoms] = {}
            for task_index in sorted({record.task_index for record in requested}):
                task_records = [record for record in requested if record.task_index == task_index]
                if (tasks[task_index]/'md_mc_result.json').is_file():
                    from workflow_backends.lammps.identity_swap import committed_frames
                    wanted = {record.md_step for record in task_records}
                    frames = {f.info['md_step']: f for f in committed_frames(tasks[task_index]) if f.info['md_step'] in wanted}
                else:
                    frames = extract_lammpstrj_selected_frames(
                    _committed_trajectory_path(tasks[task_index]),
                    type_map=list(self.type_map),
                    target_steps={record.md_step for record in task_records},
                    include_forces=False,
                )
                for record in task_records:
                    try:
                        atoms = frames[record.md_step]
                    except KeyError as exc:
                        raise ValueError(
                            f"LAMMPS replay frame is missing at task {task_index} step {record.md_step}; rebuild rN/sample"
                        ) from exc
                    atoms.info.update(
                        system_id=record.system_id,
                        structure_id=record.structure_id,
                        structure_source="workflow",
                        md_step=record.md_step,
                        **({
                            "student_uncertainty": record.student_uncertainty,
                            "force_uncertainty": record.student_uncertainty,
                            "force_std_max": record.student_uncertainty,
                        } if record.student_uncertainty is not None else {}),
                        **(
                            {
                                "sampling_role_id": record.sampling_role_id,
                                "selection_group_id": record.selection_group_id,
                                "condition_group_id": record.condition_group_id,
                            }
                            if record.sampling_role_id is not None
                            else {}
                        ),
                    )
                    loaded[record.structure_id] = atoms
            yield tuple(loaded[identity] for identity in chunk_ids)

    def _replay_ordered_subset(
        self,
        identities: tuple[str, ...],
        *,
        by_id: dict[str, ObservationRecord],
        tasks: tuple[Path, ...],
        chunk_size: int,
    ) -> Iterator[tuple[Atoms, ...]]:
        records = [by_id[identity] for identity in identities]
        output: list[Atoms] = []
        produced: list[str] = []
        for task_index in sorted({record.task_index for record in records}):
            task_records = [record for record in records if record.task_index == task_index]
            wanted = {record.md_step: record for record in task_records}
            seen: set[int] = set()
            if (tasks[task_index]/'md_mc_result.json').is_file():
                from workflow_backends.lammps.identity_swap import committed_frames
                native_frames = committed_frames(tasks[task_index])
            else:
                native_frames = iread(
                _committed_trajectory_path(tasks[task_index]),
                index=":",
                format="lammps-dump-text",
                specorder=list(self.type_map),
                )
            for atoms in native_frames:
                step = int(atoms.info.get("timestep", -1))
                record = wanted.get(step)
                if record is None:
                    continue
                atoms.info.update(
                    system_id=record.system_id,
                    structure_id=record.structure_id,
                    structure_source="workflow",
                    md_step=record.md_step,
                    **({
                        "student_uncertainty": record.student_uncertainty,
                        "force_uncertainty": record.student_uncertainty,
                        "force_std_max": record.student_uncertainty,
                    } if record.student_uncertainty is not None else {}),
                    **(
                        {
                            "sampling_role_id": record.sampling_role_id,
                            "selection_group_id": record.selection_group_id,
                            "condition_group_id": record.condition_group_id,
                        }
                        if record.sampling_role_id is not None
                        else {}
                    ),
                )
                output.append(atoms)
                produced.append(record.structure_id)
                seen.add(step)
                if len(output) == chunk_size:
                    yield tuple(output)
                    output = []
            if seen != set(wanted):
                raise ValueError(f"LAMMPS ordered replay is missing task {task_index} frames; rebuild rN/sample")
        if output:
            yield tuple(output)
        if tuple(produced) != identities:
            raise ValueError("LAMMPS ordered replay identities do not match requested order")


def _ordered_task_dirs(sample_dir: Path) -> tuple[Path, ...]:
    root = Path(sample_dir) / "tasks"
    tasks = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    expected = [f"task_{index:06d}" for index in range(len(tasks))]
    if [path.name for path in tasks] != expected:
        raise ValueError("LAMMPS sample tasks must be an exact contiguous task set; rebuild rN/sample")
    if not tasks:
        raise ValueError("LAMMPS sample has no task directories; rebuild rN/sample")
    return tuple(tasks)


def _committed_trajectory_path(task_dir: Path) -> Path:
    failure = task_dir / "sampling_failure.json"
    if not failure.exists() and not failure.is_symlink():
        return task_dir / "traj.lammpstrj"
    payload = json.loads(failure.read_text(encoding="utf-8"))
    trajectory = payload.get("trajectory")
    if trajectory is None:
        return task_dir / "traj.lammpstrj"
    name = trajectory.get("path") if isinstance(trajectory, dict) else None
    if name not in {"traj.lammpstrj", "committed_prefix.lammpstrj"}:
        raise ValueError("LAMMPS failed trajectory path is incompatible")
    return task_dir / name


def _scan_lammpstrj(
    path: Path,
    *,
    expected_atom_count: int | None = None,
    expected_type_count: int | None = None,
    validate_frame_geometry: bool = False,
) -> tuple[tuple[int, int], ...]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("LAMMPS native source is missing traj.lammpstrj; rebuild rN/sample")
    frames: list[tuple[int, int]] = []
    identity: tuple[tuple[int, int], ...] | None = None
    box_flags: tuple[bool, bool, bool] | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            try:
                step = int(handle.readline().strip())
                if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError
                natoms = int(handle.readline().strip())
                if natoms <= 0 or (
                    expected_atom_count is not None and natoms != expected_atom_count
                ):
                    raise ValueError
                box_header = handle.readline()
                if not box_header.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError
                if validate_frame_geometry:
                    tokens = box_header.split()[3:]
                    if len(tokens) < 3:
                        raise ValueError
                    raw_flags = tuple(tokens[-3:])
                    if any(
                        flag not in {
                            "pp",
                            "ff",
                            "fs",
                            "sf",
                            "fm",
                            "mf",
                            "mm",
                            "ms",
                            "sm",
                            "ss",
                        }
                        for flag in raw_flags
                    ):
                        raise ValueError
                    current_flags = tuple(flag == "pp" for flag in raw_flags)
                    if box_flags is None:
                        box_flags = current_flags
                    elif current_flags != box_flags:
                        raise ValueError
                for _ in range(3):
                    bounds = handle.readline().split()
                    if len(bounds) not in {2, 3} or not all(
                        math.isfinite(float(item)) for item in bounds
                    ):
                        raise ValueError
                columns = handle.readline().split()
                if columns[:2] != ["ITEM:", "ATOMS"]:
                    raise ValueError
                names = columns[2:]
                required = {"id", "type", "x", "y", "z"}
                if not required.issubset(names):
                    raise ValueError
                indices = {name: names.index(name) for name in required}
                current: list[tuple[int, int]] = []
                for _ in range(natoms):
                    row = handle.readline().split()
                    if len(row) != len(names):
                        raise ValueError
                    atom_id = int(row[indices["id"]])
                    atom_type = int(row[indices["type"]])
                    if (
                        atom_id <= 0
                        or atom_type <= 0
                        or (
                            expected_type_count is not None
                            and atom_type > expected_type_count
                        )
                        or not all(
                        math.isfinite(float(row[indices[name]]))
                        for name in ("x", "y", "z")
                        )
                    ):
                        raise ValueError
                    current.append((atom_id, atom_type))
                current_identity = tuple(current)
                if len(set(current_identity)) != natoms:
                    raise ValueError
                if identity is None:
                    identity = current_identity
                elif current_identity != identity:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError("malformed LAMMPS trajectory; rebuild rN/sample") from exc
            frames.append((len(frames), step))
    return tuple(frames)


def _max_deviation_rows(path: Path) -> dict[int, float]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("LAMMPS committee source is missing model_devi.out; rebuild rN/sample")
    rows: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 5:
            raise ValueError("malformed LAMMPS model_devi.out; rebuild rN/sample")
        try:
            step = int(float(parts[0]))
            maximum = float(parts[4])
        except ValueError as exc:
            raise ValueError("malformed LAMMPS model_devi.out; rebuild rN/sample") from exc
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError(
                "non-finite LAMMPS model_devi.out; rebuild rN/sample"
            )
        if step in rows:
            raise ValueError("duplicate LAMMPS model_devi step; rebuild rN/sample")
        rows[step] = maximum
    return rows


def validate_lammps_task_terminals(
    *,
    task_dir: str | Path,
    cadence: int,
    steps: int,
    require_deviation: bool,
    require_exact_cadence: bool = False,
    expected_atom_count: int | None = None,
    expected_type_count: int | None = None,
    validate_frame_geometry: bool = False,
    trajectory_path: str | Path | None = None,
) -> dict[str, tuple[int, ...]]:
    """Validate one command's trajectory/deviation products before terminal."""

    root = Path(task_dir)
    trajectory_source = (
        root / "traj.lammpstrj"
        if trajectory_path is None
        else Path(trajectory_path)
    )
    trajectory_rows = _scan_lammpstrj(
        trajectory_source,
        expected_atom_count=expected_atom_count,
        expected_type_count=expected_type_count,
        validate_frame_geometry=validate_frame_geometry,
    )
    return _validate_lammps_task_terminal_rows(
        root=root,
        cadence=cadence,
        steps=steps,
        require_deviation=require_deviation,
        require_exact_cadence=require_exact_cadence,
        trajectory_rows=trajectory_rows,
    )


def _validate_lammps_task_terminal_rows(
    *,
    root: Path,
    cadence: int,
    steps: int,
    require_deviation: bool,
    require_exact_cadence: bool,
    trajectory_rows: tuple[tuple[int, int], ...],
) -> dict[str, tuple[int, ...]]:
    stride = int(cadence)
    run_steps = int(steps)
    if stride <= 0 or run_steps <= 0:
        raise ValueError("LAMMPS task cadence and steps must be positive")
    trajectory_steps = tuple(step for _, step in trajectory_rows if step > 0)
    if not trajectory_steps:
        raise ValueError("LAMMPS task has no positive-step trajectory")
    if (
        any(step % stride or step > run_steps for step in trajectory_steps)
        or any(
            left >= right
            for left, right in zip(trajectory_steps, trajectory_steps[1:])
        )
    ):
        raise ValueError("LAMMPS trajectory cadence/order mismatch")
    if require_exact_cadence and trajectory_steps != tuple(
        range(stride, run_steps + 1, stride)
    ):
        raise ValueError("LAMMPS trajectory cadence inventory mismatch")

    deviation_steps: tuple[int, ...] = ()
    if require_deviation:
        deviations = _max_deviation_rows(root / "model_devi.out")
        deviation_steps = tuple(step for step in deviations if step > 0)
        if deviation_steps != trajectory_steps:
            raise ValueError("LAMMPS trajectory/model_devi pairing mismatch")
    elif (root / "model_devi.out").exists():
        _max_deviation_rows(root / "model_devi.out")
    return {
        "trajectory_timesteps": trajectory_steps,
        "deviation_timesteps": deviation_steps,
    }


def validate_lammps_sampling_failure(
    *,
    task_dir: str | Path,
    cadence: int,
    steps: int,
    require_deviation: bool,
    expected_atom_count: int | None = None,
    expected_type_count: int | None = None,
    validate_frame_geometry: bool = False,
    expected_system_id: str | None = None,
) -> dict[str, tuple[int, ...]] | None:
    """Validate an audited failed task and return its committed frame prefix."""

    root = Path(task_dir)
    path = root / "sampling_failure.json"
    if path.is_symlink() or not _nonempty(path):
        raise ValueError("LAMMPS sampling failure evidence is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("LAMMPS sampling failure evidence is invalid JSON") from exc
    keys = {
        "schema",
        "schema_revision",
        "task_id",
        "task_index",
        "system_id",
        "error_type",
        "message",
        "trajectory",
        "deviation",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != keys
        or payload["schema"] != "dice.lammps-sampling-failure"
        or payload["schema_revision"] != 1
        or not isinstance(payload["task_index"], int)
        or isinstance(payload["task_index"], bool)
        or payload["task_index"] < 0
        or payload["task_id"] != f"task_{payload['task_index']:06d}"
        or root.name != payload["task_id"]
        or not isinstance(payload["system_id"], str)
        or not payload["system_id"]
        or (
            expected_system_id is not None
            and payload["system_id"] != expected_system_id
        )
        or not isinstance(payload["error_type"], str)
        or not payload["error_type"]
        or not isinstance(payload["message"], str)
        or not payload["message"]
    ):
        raise ValueError("LAMMPS sampling failure evidence is incompatible")
    if (root / "native_execution.json").exists() or (
        root / "native_execution.json"
    ).is_symlink():
        raise ValueError("failed LAMMPS task must not have success execution evidence")
    trajectory = payload["trajectory"]
    deviation = payload["deviation"]
    if trajectory is None:
        if deviation is not None:
            raise ValueError("LAMMPS empty failed task has deviation evidence")
        return None
    if (
        not isinstance(trajectory, dict)
        or set(trajectory) != {"path", "sha256", "timesteps"}
        or trajectory["path"]
        not in {"traj.lammpstrj", "committed_prefix.lammpstrj"}
        or not isinstance(trajectory["sha256"], str)
        or len(trajectory["sha256"]) != 64
        or not isinstance(trajectory["timesteps"], list)
    ):
        raise ValueError("LAMMPS sampling failure trajectory evidence is incompatible")
    if require_deviation:
        if (
            not isinstance(deviation, dict)
            or set(deviation) != {"path", "sha256", "timesteps"}
            or deviation["path"] != "model_devi.out"
        ):
            raise ValueError("LAMMPS sampling failure deviation evidence is incompatible")
    elif deviation is not None:
        raise ValueError("single-model failed task cannot have deviation evidence")
    terminals = validate_lammps_task_terminals(
        task_dir=root,
        cadence=cadence,
        steps=steps,
        require_deviation=require_deviation,
        require_exact_cadence=False,
        expected_atom_count=expected_atom_count,
        expected_type_count=expected_type_count,
        validate_frame_geometry=validate_frame_geometry,
        trajectory_path=root / trajectory["path"],
    )
    expected_trajectory = {
        "path": trajectory["path"],
        "sha256": _file_digest(root / trajectory["path"]),
        "timesteps": list(terminals["trajectory_timesteps"]),
    }
    expected_deviation = (
        {
            "path": "model_devi.out",
            "sha256": _file_digest(root / "model_devi.out"),
            "timesteps": list(terminals["deviation_timesteps"]),
        }
        if require_deviation
        else None
    )
    if trajectory != expected_trajectory or deviation != expected_deviation:
        raise ValueError("LAMMPS sampling failure evidence changed")
    return terminals


def validate_lammps_native_execution(
    *,
    task_dir: str | Path,
    models: Sequence[str | Path],
    terminals: dict[str, tuple[int, ...]],
    deepmd_recipe_kind: str | None,
    expected_environment_profile_id: object = _UNSPECIFIED_PROFILE,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> None:
    """Attest the single ordinary-DeepMD command and its bound products."""

    root = Path(task_dir)
    path = root / "native_execution.json"
    if path.is_symlink() or not _nonempty(path):
        raise ValueError("LAMMPS native execution evidence is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("LAMMPS native execution evidence is invalid JSON") from exc
    ordinary_keys = {
        "schema",
        "schema_revision",
        "command",
        "sampler_role",
        "environment_profile_id",
        "deepmd_recipe_kind",
        "pair_style",
        "ordered_models",
        "trajectory",
        "deviation",
    }
    explicit_keys = ordinary_keys | {
        "task_id",
        "task_index",
        "condition_group_id",
        "sampling_role_id",
        "selection_group_id",
        "system_id",
        "source_frame_index",
        "selected_structure_id",
        "temperature_k",
        "pressure_bar",
        "kind",
        "steps",
        "cadence",
        "native_seed",
        "recipe_binding",
        "recipe_sha256",
        "rendered_input_sha256",
        "setup_sha256",
    }
    program_explicit_keys = explicit_keys | {
        "temperature_program",
        "segment_boundaries",
        "schedule_digests",
    }
    cell_program_explicit_keys = program_explicit_keys | {
        "resolved_cell_history",
        "cell_history_sha256",
        "expected_dump_steps",
    }
    kokkos_keys = ordinary_keys | {"execution_mode"}
    explicit_kokkos_keys = explicit_keys | {"execution_mode"}
    program_explicit_kokkos_keys = program_explicit_keys | {"execution_mode"}
    cell_program_explicit_kokkos_keys = cell_program_explicit_keys | {
        "execution_mode"
    }
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset(ordinary_keys),
        frozenset(explicit_keys),
        frozenset(program_explicit_keys),
        frozenset(cell_program_explicit_keys),
        frozenset(kokkos_keys),
        frozenset(explicit_kokkos_keys),
        frozenset(program_explicit_kokkos_keys),
        frozenset(cell_program_explicit_kokkos_keys),
    }:
        raise ValueError("LAMMPS native execution evidence shape is invalid")
    model_paths = tuple(Path(model).resolve() for model in models)
    suffixes = {model.suffix for model in model_paths}
    if len(suffixes) != 1 or next(iter(suffixes)) not in {".pb", ".pt2"}:
        raise ValueError("LAMMPS native execution model ABI is invalid")
    cli = "tf" if next(iter(suffixes)) == ".pb" else "pt"
    expected_models = [
        {
            "presentation": f"graph_{index:03d}{model.suffix}",
            "sha256": _file_digest(model, cache=digest_cache),
        }
        for index, model in enumerate(model_paths)
    ]
    recipe_kind = payload["deepmd_recipe_kind"]
    common_incompatible = (
        payload["schema"] != "dice.lammps-native-execution"
        or payload["sampler_role"] != f"sampler.lammps.{cli}"
        or recipe_kind not in {"standard", "dpa4c"}
        or (
            deepmd_recipe_kind is not None
            and recipe_kind != deepmd_recipe_kind
        )
        or payload["ordered_models"] != expected_models
    )
    revision = payload["schema_revision"]
    if isinstance(revision, bool):
        raise ValueError("LAMMPS native execution evidence is incompatible")
    if revision == 1:
        mode_incompatible = (
            set(payload) not in {
                frozenset(ordinary_keys),
                frozenset(explicit_keys),
                frozenset(program_explicit_keys),
                frozenset(cell_program_explicit_keys),
            }
            or payload["command"] != list(ORDINARY_COMMAND)
            or payload["pair_style"] != "deepmd"
        )
    elif revision == 2:
        try:
            selected_mode = select_lammps_execution_mode(
                sampler_role=f"sampler.lammps.{cli}",
                deepmd_recipe_kind=str(recipe_kind),
                model_count=len(model_paths),
            )
        except ValueError:
            selected_mode = "invalid"
        mode_incompatible = (
            set(payload) not in {
                frozenset(kokkos_keys),
                frozenset(explicit_kokkos_keys),
                frozenset(program_explicit_kokkos_keys),
                frozenset(cell_program_explicit_kokkos_keys),
            }
            or selected_mode != "kokkos"
            or payload["execution_mode"] != "kokkos"
            or payload["command"] != list(KOKKOS_COMMAND)
            or payload["pair_style"] != "deepmd/kk"
            or payload["deviation"] is not None
        )
    else:
        mode_incompatible = True
    if common_incompatible or mode_incompatible:
        raise ValueError("LAMMPS native execution evidence is incompatible")
    if set(payload).issuperset(explicit_keys):
        kind = payload["kind"]
        temperature = payload["temperature_k"]
        pressure = payload["pressure_bar"]
        explicit_incompatible = (
            not isinstance(payload["task_index"], int)
            or isinstance(payload["task_index"], bool)
            or payload["task_index"] < 0
            or payload["task_id"] != f"task_{payload['task_index']:06d}"
            or not isinstance(payload["sampling_role_id"], str)
            or not payload["sampling_role_id"]
            or not isinstance(payload["selection_group_id"], str)
            or not payload["selection_group_id"]
            or payload["system_id"] != payload["selection_group_id"]
            or not isinstance(payload["condition_group_id"], str)
            or not payload["condition_group_id"]
            or payload["recipe_binding"] != payload["condition_group_id"]
            or isinstance(payload["source_frame_index"], bool)
            or not isinstance(payload["source_frame_index"], int)
            or payload["source_frame_index"] < 0
            or not isinstance(payload["selected_structure_id"], str)
            or not payload["selected_structure_id"]
            or kind not in {"nvt", "npt", "nvt_program", "nvt_cell_program", "npt_program"}
            or not isinstance(payload["steps"], int)
            or isinstance(payload["steps"], bool)
            or payload["steps"] <= 0
            or not isinstance(payload["cadence"], int)
            or isinstance(payload["cadence"], bool)
            or payload["cadence"] <= 0
            or not isinstance(payload["native_seed"], int)
            or isinstance(payload["native_seed"], bool)
            or payload["native_seed"] <= 0
            or (
                kind != "nvt_cell_program"
                and payload["recipe_sha256"]
                != _file_digest(root / "input.lammps", cache=digest_cache)
            )
            or payload["rendered_input_sha256"]
            != _file_digest(root / "input.lammps", cache=digest_cache)
            or payload["setup_sha256"]
            != _file_digest(root / "dice_task_setup.lammps", cache=digest_cache)
            or (
                kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                and temperature is not None
            )
            or (
                kind not in {"nvt_program", "nvt_cell_program", "npt_program"}
                and (
                    not isinstance(temperature, (int, float))
                    or isinstance(temperature, bool)
                    or not math.isfinite(float(temperature))
                    or float(temperature) <= 0
                )
            )
            or (kind in {"npt", "npt_program"}) != (pressure is not None)
            or (
                pressure is not None
                and (
                    not isinstance(pressure, (int, float))
                    or isinstance(pressure, bool)
                    or not math.isfinite(float(pressure))
                )
            )
        )
        if explicit_incompatible:
            raise ValueError("LAMMPS explicit native execution evidence is incompatible")
        if kind == "nvt_cell_program":
            context = load_task_context(root / "task_context.json")
            if not isinstance(context, CellProgramSampleTaskContext) or any(
                payload.get(key) != value
                for key, value in {
                    "resolved_cell_history": context.resolved_cell_history,
                    "cell_history_sha256": context.cell_history_sha256,
                    "expected_dump_steps": list(context.expected_dump_steps),
                }.items()
            ):
                raise ValueError(
                    "LAMMPS cell-program native execution evidence is incompatible"
                )
            validate_lammps_cell_program_trajectory(
                task_dir=root,
                context=context,
                type_map=_type_map_from_setup(root / "dice_task_setup.lammps"),
            )
    profile_id = payload["environment_profile_id"]
    if profile_id is not None and (
        not isinstance(profile_id, str) or not profile_id.strip()
    ):
        raise ValueError("LAMMPS native execution profile id is invalid")
    if (
        expected_environment_profile_id is not _UNSPECIFIED_PROFILE
        and profile_id != expected_environment_profile_id
    ):
        raise ValueError("LAMMPS native execution profile id changed")
    expected_trajectory = {
        "path": "traj.lammpstrj",
        "sha256": _file_digest(root / "traj.lammpstrj", cache=digest_cache),
        "timesteps": list(terminals["trajectory_timesteps"]),
    }
    deviation_path = root / "model_devi.out"
    expected_deviation = (
        None
        if not deviation_path.is_file()
        else {
            "path": "model_devi.out",
            "sha256": _file_digest(deviation_path, cache=digest_cache),
            "timesteps": list(terminals["deviation_timesteps"]),
        }
    )
    if (
        payload["trajectory"] != expected_trajectory
        or payload["deviation"] != expected_deviation
    ):
        raise ValueError("LAMMPS native execution terminal binding changed")


def validate_lammps_sample_block(
    *,
    sample_dir: str | Path,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    recipe: str | Path | None = None,
    recipes: dict[str, str | Path] | None = None,
    group_recipes: dict[str, str | Path] | None = None,
    models: Sequence[str | Path],
    systems: dict[str, Sequence[Atoms]],
    steps: int | None,
    temperatures_k: Sequence[float] | None = None,
    ensemble: str | None = None,
    pressure_bar: float | None = None,
    plan: SamplingPlan | ExplicitSamplingPlan | None = None,
    plumed_sources: dict[str, str | Path],
    seed_selections: tuple[SamplingSeedSelection, ...] = (),
    deepmd_recipe_kind: str | None = None,
    require_native_execution: bool = False,
    expected_environment_profile_id: object = _UNSPECIFIED_PROFILE,
    accepted_kinds: tuple[str, ...] | None = None,
) -> bool:
    try:
        root = Path(sample_dir)
        explicit = isinstance(plan, ExplicitSamplingPlan)
        selection_manifest = root / "seed_selection.json"
        if seed_selections:
            if selection_manifest.is_symlink() or not selection_manifest.is_file():
                return False
            if json.loads(selection_manifest.read_text(encoding="utf-8")) != seed_selection_manifest_payload(
                iter_index=iter_index,
                round_index=round_index,
                selections=seed_selections,
            ):
                return False
        elif selection_manifest.exists() or selection_manifest.is_symlink():
            return False
        if any(path.name == "observations.xyz" for path in root.rglob("observations.xyz")):
            return False
        selected: list[Atoms] = []
        selected_by_system: dict[str, Atoms] = {}
        for system_id, source_frames in systems.items():
            index = (
                0
                if explicit
                else selected_seed_index(
                    run_json_path=run_json_path,
                    iter_index=iter_index,
                    round_index=round_index,
                    system_id=system_id,
                    seed_count=len(source_frames),
                )
            )
            seed = canonical_selected_seed(source_frames[index], system_id=system_id)
            selected.append(seed)
            selected_by_system[system_id] = seed
        actual_seeds = read_extxyz_structures(root / "seeds.xyz", allow_empty=False)
        if len(actual_seeds) != len(selected):
            return False
        for actual, expected in zip(actual_seeds, selected, strict=True):
            if (
                actual.info.get("system_id") != expected.info.get("system_id")
                or actual.info.get("structure_id") != expected.info.get("structure_id")
                or exact_structure_hash(actual) != exact_structure_hash(expected)
            ):
                return False
        conditions = _expected_conditions(
            plan=plan,
            systems=systems,
            temperatures_k=temperatures_k,
            ensemble=ensemble,
            pressure_bar=pressure_bar,
            round_index=round_index,
        )
        recipe_texts = _recipe_texts(
            recipe=recipe,
            recipes=recipes,
            group_recipes=group_recipes,
            ensembles=tuple(
                dict.fromkeys(
                    item.ensemble
                    for item in conditions
                    if accepted_kinds is None
                    or getattr(item, "kind", None) in accepted_kinds
                )
            ),
            condition_group_ids=(
                tuple(
                    group.condition_group_id
                    for group in plan.condition_groups
                    if accepted_kinds is None or group.kind in accepted_kinds
                )
                if explicit
                else ()
            ),
        )
        tasks = _ordered_task_dirs(root)
        if len(tasks) != len(conditions):
            return False
        model_paths = tuple(Path(model).resolve() for model in models)
        model_digests = tuple(_file_digest(path) for path in model_paths)
        usable_observations = 0
        for task_index, condition in enumerate(conditions):
                if (
                    accepted_kinds is not None
                    and getattr(condition, "kind", None) not in accepted_kinds
                ):
                    continue
                system_id = condition.system_id
                seed = selected_by_system[system_id]
                if getattr(condition, 'kind', None) == 'md_mc':
                    from workflow_backends.lammps.identity_swap import validate_result
                    ctx, result = validate_result(tasks[task_index])
                    if (ctx.task_id != condition.task_id or ctx.sampling_role_id != condition.sampling_role_id
                        or ctx.condition_group_id != condition.condition_group_id or ctx.selection_group_id != condition.selection_group_id
                        or ctx.temperature_k != condition.temperature_k or ctx.steps != condition.steps
                        or ctx.ensemble != condition.ensemble or ctx.pressure_bar != condition.pressure_bar
                        or ctx.md_mc['parameters'] != json.loads(json.dumps(asdict(condition.md_mc)))
                        or ctx.model_digests != model_digests
                        or ctx.selected_structure_id != seed.info['structure_id']):
                        return False
                    usable_observations += len(result['observations'])
                    continue
                temperature = (
                    condition.program.segments[0].start_temperature_k
                    if explicit
                    and condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                    else condition.temperature_k
                )
                task_dir = tasks[task_index]
                context = load_task_context(task_dir / "task_context.json")
                recipe_key = (
                    condition.condition_group_id if explicit else condition.ensemble
                )
                recipe_text = recipe_texts[recipe_key]
                task_steps = int(condition.steps if explicit else steps)
                expected_seed = (
                    explicit_task_seed(
                        iter_index=iter_index,
                        round_index=round_index,
                        sampling_role_id=condition.sampling_role_id,
                    )
                    if explicit
                    else task_seed(
                        run_json_path=run_json_path,
                        iter_index=iter_index,
                        round_index=round_index,
                        system_id=system_id,
                    )
                )
                if explicit:
                    if not isinstance(context, ExplicitSampleTaskContext):
                        return False
                    if (
                        context.task_id != condition.task_id
                        or context.condition_group_id != condition.condition_group_id
                        or context.kind != condition.kind
                        or context.sampling_role_id != condition.sampling_role_id
                        or context.selection_group_id != condition.selection_group_id
                        or context.system_id != condition.selection_group_id
                        or context.source_frame_index != condition.source_frame_index
                        or context.selected_structure_id != str(seed.info["structure_id"])
                        or context.temperature_k
                        != (
                            None
                            if condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                            else float(temperature)
                        )
                        or context.pressure_bar != condition.pressure_bar
                        or context.total_steps != task_steps
                        or context.trajectory_every_steps != _recipe_cadence(recipe_text)
                        or context.recipe_binding != condition.recipe_binding
                        or context.recipe_sha256 != hashlib.sha256(recipe_text.encode()).hexdigest()
                        or context.rendered_input_sha256 != _file_digest(task_dir / "input.lammps")
                        or context.setup_sha256 != _file_digest(task_dir / "dice_task_setup.lammps")
                        or context.native_seed != expected_seed
                    ):
                        return False
                    if condition.kind in {"nvt_program", "npt_program"}:
                        expected_program = _program_payload(condition.program)
                        schedule_names = (
                            "dice_temperature_start.dat",
                            "dice_temperature_stop.dat",
                            "dice_temperature_steps.dat",
                        )
                        if (
                            context.temperature_program != expected_program
                            or context.segment_boundaries != condition.program.boundaries
                            or context.schedule_digests
                            != tuple(
                                (name, _file_digest(task_dir / name))
                                for name in schedule_names
                            )
                            or tuple(
                                (task_dir / name).read_text(encoding="utf-8")
                                for name in schedule_names
                            )
                            != (
                                "".join(
                                    f"{item.start_temperature_k}\n"
                                    for item in condition.program.segments
                                ),
                                "".join(
                                    f"{item.stop_temperature_k}\n"
                                    for item in condition.program.segments
                                ),
                                "".join(
                                    f"{item.steps}\n"
                                    for item in condition.program.segments
                                ),
                            )
                        ):
                            return False
                    elif condition.kind == "nvt_cell_program":
                        if not isinstance(context, CellProgramSampleTaskContext):
                            return False
                        history = resolve_cell_program_history(
                            seed,
                            condition.program,
                            trajectory_every_steps=context.cadence,
                        )
                        if (
                            context.temperature_program
                            != _program_payload(condition.program)
                            or context.segment_boundaries
                            != condition.program.boundaries
                            or context.resolved_history != history
                            or context.cell_history_sha256 != history.sha256
                            or context.expected_dump_steps
                            != history.expected_dump_steps
                            or context.schedule_digests
                            != ((
                                "dice_cell_schedule.json",
                                _file_digest(task_dir / "dice_cell_schedule.json"),
                            ),)
                        ):
                            return False
                else:
                    expected_context = SampleTaskContext.create(
                        task_index=task_index,
                        system_id=system_id,
                        ensemble=condition.ensemble,
                        temperature_k=float(temperature),
                        pressure_bar=condition.pressure_bar,
                        steps=task_steps,
                        cadence=_recipe_cadence(recipe_text),
                        selected_seed_id=str(seed.info["structure_id"]),
                    )
                    if context != expected_context:
                        return False
                task_seed_frames = read_extxyz_structures(task_dir / "seed.xyz", allow_empty=False)
                if len(task_seed_frames) != 1 or exact_structure_hash(task_seed_frames[0]) != exact_structure_hash(seed):
                    return False
                failure_evidence = task_dir / "sampling_failure.json"
                failed = failure_evidence.exists() or failure_evidence.is_symlink()
                required = ("conf.lmp", "dice_task_setup.lammps", "input.lammps")
                if any(not _nonempty(task_dir / name) for name in required):
                    return False
                if (
                    getattr(condition, "kind", None) != "nvt_cell_program"
                    and (task_dir / "input.lammps").read_text(encoding="utf-8")
                    != recipe_text
                ):
                    return False
                if not _setup_matches(
                    task_dir / "dice_task_setup.lammps",
                    condition=condition,
                    steps=task_steps,
                    expected_seed=expected_seed,
                ):
                    return False
                presentations = sorted(task_dir.glob("graph_*"))
                expected_names = [
                    f"graph_{index:03d}{model.suffix}"
                    for index, model in enumerate(model_paths)
                ]
                if [path.name for path in presentations] != expected_names:
                    return False
                if any(path.is_symlink() or not path.is_file() for path in presentations):
                    return False
                if tuple(_file_digest(path) for path in presentations) != model_digests:
                    return False
                if explicit and (
                    context.ordered_models != tuple(path.name for path in presentations)
                    or context.model_digests != model_digests
                ):
                    return False
                deviation = task_dir / "model_devi.out"
                if len(model_paths) > 1 and not failed and not _nonempty(deviation):
                    return False
                if failed:
                    terminals = validate_lammps_sampling_failure(
                        task_dir=task_dir,
                        cadence=context.cadence,
                        steps=context.steps,
                        require_deviation=len(model_paths) > 1,
                        expected_atom_count=len(seed),
                        expected_system_id=context.system_id,
                    )
                else:
                    terminals = validate_lammps_task_terminals(
                        task_dir=task_dir,
                        cadence=context.cadence,
                        steps=context.steps,
                        require_deviation=len(model_paths) > 1,
                        require_exact_cadence=explicit,
                        expected_atom_count=len(seed),
                    )
                    if getattr(condition, "kind", None) == "nvt_cell_program":
                        validate_lammps_cell_program_trajectory(
                            task_dir=task_dir,
                            context=context,
                            type_map=_type_map_from_setup(
                                task_dir / "dice_task_setup.lammps"
                            ),
                        )
                if terminals is not None:
                    usable_observations += len(terminals["trajectory_timesteps"])
                native_evidence = task_dir / "native_execution.json"
                if not failed and (
                    explicit or require_native_execution or native_evidence.exists()
                ):
                    validate_lammps_native_execution(
                        task_dir=task_dir,
                        models=model_paths,
                        terminals=terminals,
                        deepmd_recipe_kind=deepmd_recipe_kind,
                        expected_environment_profile_id=(
                            expected_environment_profile_id
                        ),
                    )
                    if explicit:
                        native_payload = json.loads(native_evidence.read_text(encoding="utf-8"))
                        if any(
                            native_payload.get(key) != value
                            for key, value in {
                                "task_id": condition.task_id,
                                "task_index": task_index,
                                "condition_group_id": condition.condition_group_id,
                                "sampling_role_id": condition.sampling_role_id,
                                "selection_group_id": condition.selection_group_id,
                                "system_id": condition.selection_group_id,
                                "source_frame_index": condition.source_frame_index,
                                "selected_structure_id": context.selected_structure_id,
                                "temperature_k": context.temperature_k,
                                "pressure_bar": context.pressure_bar,
                                "kind": condition.kind,
                                "steps": task_steps,
                                "cadence": context.cadence,
                                "native_seed": context.native_seed,
                                "recipe_binding": context.recipe_binding,
                                "recipe_sha256": context.recipe_sha256,
                                "rendered_input_sha256": context.rendered_input_sha256,
                                "setup_sha256": context.setup_sha256,
                            }.items()
                        ):
                            return False
                        if condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"} and (
                            tuple(native_payload.get("temperature_program", ()))
                            != context.temperature_program
                            or tuple(native_payload.get("segment_boundaries", ()))
                            != context.segment_boundaries
                            or tuple(
                                tuple(item)
                                for item in native_payload.get("schedule_digests", ())
                            )
                            != context.schedule_digests
                        ):
                            return False
                        if condition.kind == "nvt_cell_program" and any(
                            native_payload.get(key) != value
                            for key, value in {
                                "resolved_cell_history": context.resolved_cell_history,
                                "cell_history_sha256": context.cell_history_sha256,
                                "expected_dump_steps": list(context.expected_dump_steps),
                            }.items()
                        ):
                            return False
                plumed = plumed_sources.get(system_id)
                if plumed is None:
                    if (task_dir / "plumed.dat").exists():
                        return False
                elif (
                    not _nonempty(task_dir / "plumed.dat")
                    or (task_dir / "plumed.dat").read_text(encoding="utf-8")
                    != Path(plumed).read_text(encoding="utf-8")
                ):
                    return False
        return usable_observations > 0
    except (OSError, ValueError, KeyError, TypeError):
        return False


def reconstruct_legacy_lammps_contexts(
    *,
    sample_dir: str | Path,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    recipe: str | Path | None = None,
    recipes: dict[str, str | Path] | None = None,
    models: Sequence[str | Path],
    type_map: Sequence[str],
    systems: dict[str, Sequence[Atoms]],
    steps: int,
    temperatures_k: Sequence[float] | None = None,
    ensemble: str | None = None,
    pressure_bar: float | None = None,
    plan: SamplingPlan | None = None,
    plumed_sources: dict[str, str | Path],
) -> tuple[SampleTaskContext, ...]:
    root = Path(sample_dir)
    tasks = _ordered_task_dirs(root)
    context_paths = tuple(task / "task_context.json" for task in tasks)
    if (root / "seeds.xyz").exists() or any(path.exists() for path in context_paths):
        raise ValueError("legacy LAMMPS sample evidence is partial or ambiguous; rebuild from rN/sample")
    conditions = _expected_conditions(
        plan=plan,
        systems=systems,
        temperatures_k=temperatures_k,
        ensemble=ensemble,
        pressure_bar=pressure_bar,
    )
    recipe_texts = _recipe_texts(
        recipe=recipe,
        recipes=recipes,
        ensembles=tuple(dict.fromkeys(item.ensemble for item in conditions)),
    )
    if len(tasks) != len(conditions):
        raise ValueError("legacy LAMMPS task set does not match run inputs; rebuild from rN/sample")
    model_paths = tuple(Path(model).resolve() for model in models)
    model_digests = tuple(_file_digest(path) for path in model_paths)
    contexts: list[SampleTaskContext] = []
    selected_by_system: dict[str, Atoms] = {}
    for system_id, source_frames in systems.items():
        selected_index = selected_seed_index(
            run_json_path=run_json_path,
            iter_index=iter_index,
            round_index=round_index,
            system_id=system_id,
            seed_count=len(source_frames),
        )
        seed = canonical_selected_seed(source_frames[selected_index], system_id=system_id)
        selected_by_system[system_id] = seed
    for task_index, condition in enumerate(conditions):
            system_id = condition.system_id
            seed = selected_by_system[system_id]
            temperature = condition.temperature_k
            recipe_text = recipe_texts[condition.ensemble]
            cadence = _recipe_cadence(recipe_text)
            task_dir = tasks[task_index]
            task_seed_frames = read_extxyz_structures(task_dir / "seed.xyz", allow_empty=False)
            if len(task_seed_frames) != 1 or exact_structure_hash(task_seed_frames[0]) != exact_structure_hash(seed):
                raise ValueError("legacy LAMMPS selected seed mismatch; rebuild from rN/sample")
            required = ("conf.lmp", "dice_task_setup.lammps", "input.lammps", "traj.lammpstrj")
            if any(not _nonempty(task_dir / name) for name in required):
                raise ValueError("legacy LAMMPS task input/output is missing; rebuild from rN/sample")
            if (task_dir / "input.lammps").read_text(encoding="utf-8") != recipe_text:
                raise ValueError("legacy LAMMPS rendered recipe mismatch; rebuild from rN/sample")
            if not _setup_matches(
                task_dir / "dice_task_setup.lammps",
                condition=condition,
                steps=steps,
                expected_seed=task_seed(
                    run_json_path=run_json_path,
                    iter_index=iter_index,
                    round_index=round_index,
                    system_id=system_id,
                ),
            ):
                raise ValueError("legacy LAMMPS setup thermodynamics mismatch; rebuild from rN/sample")
            presentations = sorted(task_dir.glob("graph_*"))
            expected_names = [f"graph_{index:03d}{model.suffix}" for index, model in enumerate(model_paths)]
            if [path.name for path in presentations] != expected_names:
                raise ValueError("legacy LAMMPS model presentation mismatch; rebuild from rN/sample")
            if any(path.is_symlink() or not path.is_file() for path in presentations):
                raise ValueError("legacy LAMMPS model presentation is invalid; rebuild from rN/sample")
            if tuple(_file_digest(path) for path in presentations) != model_digests:
                raise ValueError("legacy LAMMPS model bytes mismatch; rebuild from rN/sample")
            if len(model_paths) > 1 and not _nonempty(task_dir / "model_devi.out"):
                raise ValueError("legacy LAMMPS model_devi.out is missing; rebuild from rN/sample")
            plumed = plumed_sources.get(system_id)
            if plumed is None and (task_dir / "plumed.dat").exists():
                raise ValueError("legacy LAMMPS PLUMED evidence is unexpected; rebuild from rN/sample")
            if plumed is not None and (
                not _nonempty(task_dir / "plumed.dat")
                or (task_dir / "plumed.dat").read_text(encoding="utf-8")
                != Path(plumed).read_text(encoding="utf-8")
            ):
                raise ValueError("legacy LAMMPS PLUMED evidence mismatch; rebuild from rN/sample")
            contexts.append(
                SampleTaskContext.create(
                    task_index=task_index,
                    system_id=system_id,
                    ensemble=condition.ensemble,
                    temperature_k=temperature,
                    pressure_bar=condition.pressure_bar,
                    steps=steps,
                    cadence=cadence,
                    selected_seed_id=str(seed.info["structure_id"]),
                )
            )
    source = LammpsObservationSource(
        sample_dir=root,
        run_json_path=run_json_path,
        iter_index=iter_index,
        round_index=round_index,
        type_map=type_map,
        model_count=len(model_paths),
        contexts=contexts,
    )
    source.catalog()
    return tuple(contexts)


def _recipe_cadence(recipe_text: str) -> int:
    declared = re.search(
        r"(?m)^\s*variable\s+dice_trajectory_every_steps\s+equal\s+([1-9]\d*)\s*$",
        recipe_text,
    )
    if declared:
        return int(declared.group(1))
    match = re.search(
        r"(?m)^\s*dump\s+\S+\s+\S+\s+\S+\s+(\d+)", recipe_text
    )
    return int(match.group(1)) if match else 20


def _type_map_from_setup(path: Path) -> tuple[str, ...]:
    matches = re.findall(
        r"(?m)^\s*pair_coeff\s+\*\s+\*\s+(.+?)\s*$",
        path.read_text(encoding="utf-8"),
    )
    if len(matches) != 1:
        raise ValueError("LAMMPS setup lacks one explicit pair_coeff type map")
    values = tuple(matches[0].split())
    if not values or any(not value for value in values):
        raise ValueError("LAMMPS setup type map is empty")
    return values


def _program_temperature_at_step(
    segments: tuple[dict[str, object], ...], step: int
) -> float:
    previous_end = 0
    for segment in segments:
        end = int(segment["end_step"])
        if step <= end:
            start_temperature = float(segment["start_temperature_k"])
            stop_temperature = float(segment["stop_temperature_k"])
            return start_temperature + (stop_temperature - start_temperature) * (
                (step - previous_end) / (end - previous_end)
            )
        previous_end = end
    raise ValueError("LAMMPS program observation exceeds the declared schedule")


def _program_payload(program: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "start_temperature_k": segment.start_temperature_k,
            "stop_temperature_k": segment.stop_temperature_k,
            "steps": segment.steps,
            "end_step": segment.end_step,
            **(
                {
                    "cell": (
                        {
                            "mode": "isotropic_density",
                            "stop_density_g_cm3": segment.cell.stop_density_g_cm3,
                        }
                        if segment.cell.mode == "isotropic_density"
                        else {
                            "mode": "symmetric_strain",
                            "stop": dict(segment.cell.symmetric_strain),
                        }
                    )
                }
                if getattr(segment, "cell", None) is not None
                else {}
            ),
        }
        for segment in program.segments
    )


def _expected_conditions(
    *,
    plan: SamplingPlan | ExplicitSamplingPlan | None,
    systems: dict[str, Sequence[Atoms]],
    temperatures_k: Sequence[float] | None,
    ensemble: str | None,
    pressure_bar: float | None,
    round_index: int | None = None,
) -> tuple[object, ...]:
    if plan is not None:
        plan_system_ids = (
            tuple(seed.sampling_role_id for seed in plan.seeds)
            if isinstance(plan, ExplicitSamplingPlan)
            else tuple(system.system_id for system in plan.systems)
        )
        if plan_system_ids != tuple(systems):
            raise ValueError("LAMMPS planned systems do not match configured systems")
        if isinstance(plan, ExplicitSamplingPlan):
            if round_index is None:
                raise ValueError("explicit LAMMPS validation requires a round index")
            return plan.tasks_for_round(round_index)
        return plan.tasks
    if ensemble not in {"nvt", "npt"} or temperatures_k is None:
        raise ValueError("LAMMPS validation requires a resolved sampling plan")
    if ensemble == "npt" and pressure_bar is None:
        raise ValueError("LAMMPS validation requires NPT pressure")
    return tuple(
        SamplingTaskCondition(
            system_id,
            ensemble,
            float(temperature),
            None if ensemble == "nvt" else float(pressure_bar),
        )
        for system_id in systems
        for temperature in temperatures_k
    )


def _recipe_texts(
    *,
    recipe: str | Path | None,
    recipes: dict[str, str | Path] | None,
    ensembles: tuple[str, ...],
    group_recipes: dict[str, str | Path] | None = None,
    condition_group_ids: tuple[str, ...] = (),
) -> dict[str, str]:
    if condition_group_ids:
        if recipe is not None or recipes is not None or group_recipes is None:
            raise ValueError("explicit LAMMPS validation requires group recipes only")
        if set(group_recipes) != set(condition_group_ids):
            raise ValueError("LAMMPS validation group recipes do not match condition groups")
        return {
            group_id: Path(group_recipes[group_id]).read_text(encoding="utf-8")
            for group_id in condition_group_ids
        }
    if group_recipes is not None:
        raise ValueError("legacy LAMMPS validation rejects group recipes")
    if (recipe is None) == (recipes is None):
        raise ValueError("LAMMPS validation requires exactly one recipe declaration")
    if recipes is None:
        if len(ensembles) > 1:
            raise ValueError("heterogeneous LAMMPS validation requires ensemble recipes")
        return {ensembles[0]: Path(recipe).read_text(encoding="utf-8")}
    if set(recipes) != set(ensembles):
        raise ValueError("LAMMPS validation recipe keys do not match effective ensembles")
    return {
        ensemble: Path(path).read_text(encoding="utf-8")
        for ensemble, path in recipes.items()
    }


def _setup_matches(
    path: Path,
    *,
    condition: SamplingTaskCondition,
    steps: int,
    expected_seed: int,
) -> bool:
    if not _nonempty(path):
        return False
    variables: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.split("#", 1)[0].split()
        if len(row) == 4 and row[:1] == ["variable"] and row[2] == "equal":
            if row[1] in variables:
                return False
            variables[row[1]] = row[3]
    expected_keys = {"dice_run_steps", "dice_temperature", "dice_seed"}
    if condition.ensemble == "npt":
        expected_keys.add("dice_pressure")
    if set(variables) & {"dice_run_steps", "dice_temperature", "dice_seed", "dice_pressure"} != expected_keys:
        return False
    try:
        expected_temperature = (
            condition.program.segments[0].start_temperature_k
            if getattr(condition, "kind", None)
            in {"nvt_program", "nvt_cell_program", "npt_program"}
            else condition.temperature_k
        )
        return (
            int(variables["dice_run_steps"]) == int(steps)
            and float(variables["dice_temperature"]) == expected_temperature
            and int(variables["dice_seed"]) == int(expected_seed)
            and (
                condition.ensemble == "nvt"
                or float(variables["dice_pressure"]) == condition.pressure_bar
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _file_digest(
    path: Path,
    *,
    cache: dict[tuple[int, int, int, int], str] | None = None,
) -> str:
    if cache is not None:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = cache.get(key)
        if cached is not None:
            return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    if cache is not None:
        cache[key] = result
    return result


__all__ = [
    "LammpsObservationSource",
    "reconstruct_legacy_lammps_contexts",
    "validate_lammps_native_execution",
    "validate_lammps_sampling_failure",
    "validate_lammps_task_terminals",
    "validate_lammps_sample_block",
]
