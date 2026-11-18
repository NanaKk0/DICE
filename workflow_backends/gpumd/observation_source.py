from __future__ import annotations

from collections.abc import Iterator, Sequence
from decimal import Decimal, InvalidOperation
from io import StringIO
import hashlib
import math
from pathlib import Path
import re

from ase import Atoms
from ase.io import iread, read
from workflow_backends.common.frame_index import index_extxyz_frames

from workflow_backends.observation_source import (
    ObservationLocator,
    ObservationRecord,
    observation_identity,
)
from workflow_backends.sample_artifacts import (
    ExplicitSampleTaskContext,
    SampleTaskContext,
    canonical_selected_seed,
    load_task_context,
)
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures
from workflow_artifacts.lifecycle.sample_conditions import (
    SamplingPlan,
    SamplingTaskCondition,
)


class GpumdObservationSource:
    """Screen-owned catalog and bounded replay over committed GPUMD tasks."""

    def __init__(
        self,
        *,
        sample_dir: str | Path,
        run_json_path: str | Path,
        iter_index: int,
        round_index: int,
        contexts: Sequence[SampleTaskContext] | None = None,
        plan: ExplicitSamplingPlan | None = None,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.run_json_path = Path(run_json_path)
        self.iter_index = int(iter_index)
        self.round_index = int(round_index)
        self.contexts = None if contexts is None else tuple(contexts)
        self.plan = plan
        self._catalog: tuple[ObservationRecord, ...] | None = None
        self._frame_indices = {}

    def catalog(self) -> tuple[ObservationRecord, ...]:
        if self._catalog is not None:
            return self._catalog
        records: list[ObservationRecord] = []
        task_dirs = _ordered_task_dirs(self.sample_dir)
        planned = (
            self.plan.tasks_for_round(self.round_index)
            if self.plan is not None
            else None
        )
        if planned is not None and len(task_dirs) != len(planned):
            raise ValueError("GPUMD explicit sample inventory disagrees with its plan")
        for task_index, task_dir in enumerate(task_dirs):
            if planned is not None and (planned[task_index].kind in {'neb', 'relax'} or planned[task_index].protocol_request is not None):
                continue
            context = (
                load_task_context(task_dir / "task_context.json")
                if self.contexts is None
                else self.contexts[task_index]
            )
            if context.task_index != task_index:
                raise ValueError("GPUMD sample task_context task index mismatch; rebuild rN/sample")
            explicit = isinstance(context, ExplicitSampleTaskContext)
            if planned is not None:
                condition = planned[task_index]
                if not explicit or (
                    context.task_id != condition.task_id
                    or context.condition_group_id != condition.condition_group_id
                    or context.sampling_role_id != condition.sampling_role_id
                    or context.selection_group_id != condition.selection_group_id
                    or context.kind != condition.kind
                    or context.temperature_k != condition.temperature_k
                    or context.pressure_bar != condition.pressure_bar
                    or context.steps != condition.steps
                ):
                    raise ValueError("GPUMD explicit task provenance disagrees with its plan")
            active_rows = _active_rows(
                task_dir / "active.out",
                run_input_path=task_dir / "run.in",
            )
            index = index_extxyz_frames(task_dir/"active.xyz", expected_count=len(active_rows))
            self._frame_indices[task_index] = index
            frame_count = len(index.spans)
            if len(active_rows) != frame_count:
                raise ValueError("GPUMD active.out/active.xyz length mismatch; rebuild rN/sample")
            for frame_index, (step, uncertainty) in enumerate(active_rows):
                if step <= 0 or step % context.cadence:
                    raise ValueError("GPUMD active observation cadence mismatch; rebuild rN/sample")
                identity = observation_identity(
                    run_json_path=self.run_json_path,
                    iter_index=self.iter_index,
                    round_index=self.round_index,
                    system_id=context.system_id,
                    temperature_k=context.temperature_k,
                    pressure_bar=context.pressure_bar,
                    md_step=step,
                    sampling_role_id=(context.sampling_role_id if explicit else None),
                    condition_group_id=(context.condition_group_id if explicit else None),
                    task_id=(context.task_id if explicit else None),
                )
                locator = ObservationLocator("GPUMD", task_index, frame_index, step)
                records.append(
                    ObservationRecord(
                        structure_id=identity,
                        system_id=context.system_id,
                        task_index=task_index,
                        frame_index=frame_index,
                        md_step=step,
                        student_uncertainty=uncertainty,
                        locator=locator,
                        sampling_role_id=(context.sampling_role_id if explicit else None),
                        selection_group_id=(context.selection_group_id if explicit else None),
                        condition_group_id=(context.condition_group_id if explicit else None),
                        task_id=(context.task_id if explicit else None),
                    )
                )
        if not records:
            raise ValueError("GPUMD native source has no positive-step observations; rebuild rN/sample")
        self._catalog = tuple(records)
        self._records_by_id = {record.structure_id: record for record in records}
        return self._catalog

    def replay(
        self,
        ordered_identities: Sequence[str],
        *,
        chunk_size: int,
    ) -> Iterator[tuple[Atoms, ...]]:
        size = int(chunk_size)
        if size <= 0:
            raise ValueError("GPUMD observation replay chunk_size must be positive")
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("GPUMD replay identities must be unique and non-empty")
        self.catalog()
        by_id = self._records_by_id
        missing = [identity for identity in identities if identity not in by_id]
        if missing:
            raise ValueError(f"GPUMD replay identity is absent from the catalog: {missing[0]}")
        for start in range(0, len(identities), size):
            wanted, grouped, loaded = identities[start:start + size], {}, {}
            for identity in wanted:
                record = by_id[identity]
                grouped.setdefault(record.task_index, []).append(record)
            for task_index, records in grouped.items():
                for record, raw in zip(records, self._frame_indices[task_index].read(
                    record.frame_index for record in records), strict=True):
                    atoms = read(StringIO(raw.decode('utf-8')), format='extxyz')
                    atoms.info.update(system_id=record.system_id, structure_id=record.structure_id,
                        structure_source='workflow', md_step=record.md_step,
                        student_uncertainty=record.student_uncertainty,
                        force_uncertainty=record.student_uncertainty, force_std_max=record.student_uncertainty)
                    if record.sampling_role_id is not None:
                        atoms.info.update(sampling_role_id=record.sampling_role_id,
                            selection_group_id=record.selection_group_id,
                            condition_group_id=record.condition_group_id, task_id=record.task_id)
                    loaded[record.structure_id] = atoms
            yield tuple(loaded[identity] for identity in wanted)


def _ordered_task_dirs(sample_dir: Path) -> tuple[Path, ...]:
    root = Path(sample_dir) / "tasks"
    tasks = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if [path.name for path in tasks] != [f"task_{index:06d}" for index in range(len(tasks))]:
        raise ValueError("GPUMD sample tasks must be an exact contiguous task set; rebuild rN/sample")
    if not tasks:
        raise ValueError("GPUMD sample has no task directories; rebuild rN/sample")
    return tuple(tasks)


def _active_rows(path: Path, *, run_input_path: Path) -> tuple[tuple[int, float], ...]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("GPUMD native source is missing active.out; rebuild rN/sample")
    time_step = _rendered_time_step(run_input_path)
    rows: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) < 2:
            raise ValueError("malformed GPUMD active.out; rebuild rN/sample")
        try:
            native_time = Decimal(parts[0])
            uncertainty = float(parts[1])
            step_ratio = native_time / time_step
        except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
            raise ValueError("malformed GPUMD active.out; rebuild rN/sample") from exc
        if (
            not native_time.is_finite()
            or not step_ratio.is_finite()
            or not math.isfinite(uncertainty)
            or uncertainty < 0.0
        ):
            raise ValueError("malformed GPUMD active.out; rebuild rN/sample")
        integral_step = step_ratio.to_integral_value()
        if step_ratio != integral_step:
            raise ValueError(
                "GPUMD active.out time is not an exact MD step; rebuild rN/sample"
            )
        rows.append((int(integral_step), uncertainty))
    if not rows:
        raise ValueError("GPUMD active.out has no observations; rebuild rN/sample")
    return tuple(rows)


def _rendered_time_step(path: Path) -> Decimal:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("GPUMD native source is missing run.in; rebuild rN/sample")
    rows = [
        line.split("#", 1)[0].split()
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
    ]
    run_indices = [index for index, row in enumerate(rows) if row[:1] == ["run"]]
    if len(run_indices) != 1:
        raise ValueError("GPUMD run.in main run is ambiguous; rebuild rN/sample")
    time_step_rows = [
        row for row in rows[: run_indices[0]] if row[:1] == ["time_step"]
    ]
    if not time_step_rows:
        return Decimal(1)
    if len(time_step_rows) != 1 or len(time_step_rows[0]) != 2:
        raise ValueError("GPUMD run.in time_step is ambiguous; rebuild rN/sample")
    try:
        time_step = Decimal(time_step_rows[0][1])
    except InvalidOperation as exc:
        raise ValueError("GPUMD run.in time_step is malformed; rebuild rN/sample") from exc
    if not time_step.is_finite() or time_step <= 0:
        raise ValueError("GPUMD run.in time_step must be positive; rebuild rN/sample")
    return time_step


def _extxyz_frame_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("GPUMD native source is missing active.xyz; rebuild rN/sample")
    count = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            try:
                natoms = int(first.strip())
            except ValueError as exc:
                raise ValueError("malformed GPUMD active.xyz; rebuild rN/sample") from exc
            if natoms <= 0 or not handle.readline():
                raise ValueError("malformed GPUMD active.xyz; rebuild rN/sample")
            for _ in range(natoms):
                if not handle.readline():
                    raise ValueError("malformed GPUMD active.xyz; rebuild rN/sample")
            count += 1
    return count


def validate_gpumd_sample_block(
    *,
    sample_dir: str | Path,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    models: Sequence[str | Path],
    systems: dict[str, Sequence[Atoms]],
    steps: int | None,
    temperatures_k: Sequence[float] | None = None,
    ensemble: str | None = None,
    pressure_bar: float | None = None,
    plan: SamplingPlan | ExplicitSamplingPlan | None = None,
    group_recipes: dict[str, str | Path] | None = None,
    accepted_kinds: tuple[str, ...] | None = None,
) -> bool:
    try:
        root = Path(sample_dir)
        if any(path.name == "observations.xyz" for path in root.rglob("observations.xyz")):
            return False
        selected: list[Atoms] = []
        selected_by_system: dict[str, Atoms] = {}
        for system_id, source_frames in systems.items():
            if not source_frames:
                return False
            raw = int.from_bytes(
                hashlib.sha256(
                    f"{Path(run_json_path).resolve()}\0{int(iter_index)}\0{int(round_index)}\0{system_id}".encode()
                ).digest()[:4],
                "big",
            )
            seed = canonical_selected_seed(
                source_frames[0 if isinstance(plan, ExplicitSamplingPlan) else raw % len(source_frames)],
                system_id=system_id,
            )
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
        tasks = _ordered_task_dirs(root)
        if len(tasks) != len(conditions):
            return False
        model_paths = tuple(Path(model).resolve() for model in models)
        model_digests = tuple(_file_digest(path) for path in model_paths)
        for task_index, condition in enumerate(conditions):
                if getattr(condition, 'protocol_request', None) is not None:
                    continue
                if accepted_kinds is not None and getattr(condition, "kind", None) not in accepted_kinds:
                    continue
                system_id = condition.system_id
                seed = selected_by_system[system_id]
                temperature = condition.temperature_k
                task_dir = tasks[task_index]
                run_in = task_dir / "run.in"
                if any(
                    not _nonempty(task_dir / name)
                    for name in ("model.xyz", "run.in", "active.xyz", "active.out")
                ):
                    return False
                rendered = run_in.read_text(encoding="utf-8")
                match = re.search(r"(?m)^active\s+(\d+)\b", rendered)
                if match is None:
                    return False
                cadence = int(match.group(1))
                context = load_task_context(task_dir / "task_context.json")
                task_steps = int(condition.steps if isinstance(plan, ExplicitSamplingPlan) else steps)
                if isinstance(plan, ExplicitSamplingPlan):
                    if not isinstance(context, ExplicitSampleTaskContext) or (
                        context.task_index != task_index
                        or context.task_id != condition.task_id
                        or context.condition_group_id != condition.condition_group_id
                        or context.kind != condition.kind
                        or context.sampling_role_id != condition.sampling_role_id
                        or context.selection_group_id != condition.selection_group_id
                        or context.system_id != condition.selection_group_id
                        or context.selected_structure_id != str(seed.info["structure_id"])
                        or context.temperature_k != float(temperature)
                        or context.pressure_bar != condition.pressure_bar
                        or context.total_steps != task_steps
                        or context.trajectory_every_steps != cadence
                        or context.recipe_binding != condition.recipe_binding
                    ):
                        return False
                    if group_recipes is None:
                        return False
                    recipe_path = Path(group_recipes[condition.condition_group_id])
                    if context.recipe_sha256 != _file_digest(recipe_path):
                        return False
                    if context.rendered_input_sha256 != _file_digest(run_in):
                        return False
                else:
                    expected_context = SampleTaskContext.create(
                        task_index=task_index,
                        system_id=system_id,
                        ensemble=condition.ensemble,
                        temperature_k=float(temperature),
                        pressure_bar=condition.pressure_bar,
                        steps=task_steps,
                        cadence=cadence,
                        selected_seed_id=str(seed.info["structure_id"]),
                    )
                    if context != expected_context:
                        return False
                if not _rendered_condition_matches(rendered, condition=condition, steps=task_steps):
                    return False
                task_seed = read_extxyz_structures(task_dir / "model.xyz", allow_empty=False)
                if len(task_seed) != 1 or exact_structure_hash(task_seed[0]) != exact_structure_hash(seed):
                    return False
                presentations = sorted(task_dir.glob("nep_*.txt"))
                if [path.name for path in presentations] != [
                    f"nep_{index}.txt" for index in range(len(model_paths))
                ]:
                    return False
                if any(path.is_symlink() or not path.is_file() for path in presentations):
                    return False
                if tuple(_file_digest(path) for path in presentations) != model_digests:
                    return False
                potential_lines = [
                    line.strip()
                    for line in rendered.splitlines()
                    if line.strip().startswith("potential ")
                ]
                if potential_lines != [f"potential nep_{index}.txt" for index in range(len(model_paths))]:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def reconstruct_legacy_gpumd_contexts(
    *,
    sample_dir: str | Path,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    models: Sequence[str | Path],
    systems: dict[str, Sequence[Atoms]],
    steps: int,
    temperatures_k: Sequence[float] | None = None,
    ensemble: str | None = None,
    pressure_bar: float | None = None,
    plan: SamplingPlan | None = None,
) -> tuple[SampleTaskContext, ...]:
    root = Path(sample_dir)
    tasks = _ordered_task_dirs(root)
    context_paths = tuple(task / "task_context.json" for task in tasks)
    if (root / "seeds.xyz").exists() or any(path.exists() for path in context_paths):
        raise ValueError("legacy GPUMD sample evidence is partial or ambiguous; rebuild from rN/sample")
    conditions = _expected_conditions(
        plan=plan,
        systems=systems,
        temperatures_k=temperatures_k,
        ensemble=ensemble,
        pressure_bar=pressure_bar,
    )
    if len(tasks) != len(conditions):
        raise ValueError("legacy GPUMD task set does not match run inputs; rebuild from rN/sample")
    model_paths = tuple(Path(model).resolve() for model in models)
    model_digests = tuple(_file_digest(path) for path in model_paths)
    contexts: list[SampleTaskContext] = []
    selected_by_system: dict[str, Atoms] = {}
    for system_id, source_frames in systems.items():
        if not source_frames:
            raise ValueError("legacy GPUMD selected seed source is empty; rebuild from rN/sample")
        raw = int.from_bytes(
            hashlib.sha256(
                f"{Path(run_json_path).resolve()}\0{int(iter_index)}\0{int(round_index)}\0{system_id}".encode()
            ).digest()[:4],
            "big",
        )
        seed = canonical_selected_seed(source_frames[raw % len(source_frames)], system_id=system_id)
        selected_by_system[system_id] = seed
    for task_index, condition in enumerate(conditions):
            system_id = condition.system_id
            seed = selected_by_system[system_id]
            temperature = condition.temperature_k
            task_dir = tasks[task_index]
            required = ("model.xyz", "run.in", "active.xyz", "active.out")
            if any(not _nonempty(task_dir / name) for name in required):
                raise ValueError("legacy GPUMD task input/output is missing; rebuild from rN/sample")
            task_seed = read_extxyz_structures(task_dir / "model.xyz", allow_empty=False)
            if len(task_seed) != 1 or exact_structure_hash(task_seed[0]) != exact_structure_hash(seed):
                raise ValueError("legacy GPUMD selected seed mismatch; rebuild from rN/sample")
            presentations = sorted(task_dir.glob("nep_*.txt"))
            if [path.name for path in presentations] != [
                f"nep_{index}.txt" for index in range(len(model_paths))
            ]:
                raise ValueError("legacy GPUMD model presentation mismatch; rebuild from rN/sample")
            if any(path.is_symlink() or not path.is_file() for path in presentations):
                raise ValueError("legacy GPUMD model presentation is invalid; rebuild from rN/sample")
            if tuple(_file_digest(path) for path in presentations) != model_digests:
                raise ValueError("legacy GPUMD model bytes mismatch; rebuild from rN/sample")
            rendered = (task_dir / "run.in").read_text(encoding="utf-8")
            active_match = re.search(r"(?m)^active\s+(\d+)\b", rendered)
            if active_match is None:
                raise ValueError("legacy GPUMD active cadence is missing; rebuild from rN/sample")
            cadence = int(active_match.group(1))
            potentials = [
                line.strip()
                for line in rendered.splitlines()
                if line.strip().startswith("potential ")
            ]
            if potentials != [f"potential nep_{index}.txt" for index in range(len(model_paths))]:
                raise ValueError("legacy GPUMD rendered models mismatch; rebuild from rN/sample")
            if not _rendered_condition_matches(rendered, condition=condition, steps=steps):
                raise ValueError("legacy GPUMD rendered thermodynamics mismatch; rebuild from rN/sample")
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
    source = GpumdObservationSource(
        sample_dir=root,
        run_json_path=run_json_path,
        iter_index=iter_index,
        round_index=round_index,
        contexts=contexts,
    )
    source.catalog()
    return tuple(contexts)


def _expected_conditions(
    *,
    plan: SamplingPlan | ExplicitSamplingPlan | None,
    systems: dict[str, Sequence[Atoms]],
    temperatures_k: Sequence[float] | None,
    ensemble: str | None,
    pressure_bar: float | None,
    round_index: int | None = None,
) -> tuple[SamplingTaskCondition, ...]:
    if plan is not None:
        plan_ids = (
            tuple(role for role, _group in plan.selection_plan_for_round(int(round_index)).role_to_group)
            if isinstance(plan, ExplicitSamplingPlan)
            else tuple(system.system_id for system in plan.systems)
        )
        if plan_ids != tuple(systems):
            raise ValueError("GPUMD planned systems do not match configured systems")
        return (
            plan.tasks_for_round(int(round_index))
            if isinstance(plan, ExplicitSamplingPlan)
            else plan.tasks
        )
    if ensemble not in {"nvt", "npt"} or temperatures_k is None:
        raise ValueError("GPUMD validation requires a resolved sampling plan")
    if ensemble == "npt" and pressure_bar is None:
        raise ValueError("GPUMD validation requires NPT pressure")
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


def _rendered_condition_matches(
    rendered: str, *, condition: SamplingTaskCondition, steps: int,
) -> bool:
    rows = [line.split("#", 1)[0].split() for line in rendered.splitlines()]
    velocity = [row for row in rows if row[:1] == ["velocity"]]
    ensembles = [row for row in rows if row[:1] == ["ensemble"]]
    runs = [row for row in rows if row[:1] == ["run"]]
    try:
        if len(velocity) != 1:
            return False
        if len(runs) != 1 or int(runs[0][1]) != int(steps) or len(ensembles) != 1:
            return False
        row = ensembles[0]
        if condition.ensemble == "nvt":
            return (
                len(row) == 5
                and row[1].startswith('nvt_')
                and float(row[2]) == condition.temperature_k
                and float(row[3]) == condition.temperature_k
            )
        return (
            len(row) in {6, 8, 12, 18}
            and row[1].startswith('npt_')
            and float(row[2]) == condition.temperature_k
            and float(row[3]) == condition.temperature_k
            and math.isclose(float(row[5]), condition.pressure_bar if len(row)==6 else condition.pressure_bar*1e-4)
        )
    except (IndexError, TypeError, ValueError):
        return False


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "GpumdObservationSource",
    "reconstruct_legacy_gpumd_contexts",
    "validate_gpumd_sample_block",
]
