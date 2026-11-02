from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Protocol, Sequence

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
import numpy as np
from workflow_backends.common.model_files import link_model_file
from workflow_backends.execution import operation_execution_scope
from workflow_backends.common.task_outcomes import TaskOutcome, execute_task
from workflow_backends.common.cancellation import handled_sigterm as _handled_sigterm
from workflow_backends.common.sampling_pool import (
    execute_sampling_tasks,
    sampling_visible_devices,
)
from workflow_backends.sample_artifacts import (
    CELL_PROGRAM_TASK_CONTEXT_SCHEMA_REVISION,
    CellProgramSampleTaskContext,
    MdMcSampleTaskContext,
    ExplicitSampleTaskContext,
    SampleTaskContext,
    canonical_selected_seed,
    explicit_task_seed,
    load_task_context,
    publish_selected_seeds,
    selected_seed_index,
    task_seed,
    write_task_context,
)
from workflow_backends.lammps.cell_program import (
    build_lammps_cell_schedule,
    render_lammps_cell_program_input,
    validate_lammps_cell_program_trajectory,
    write_lammps_cell_schedule,
)
from workflow_core.structures.formats import atoms_to_lammps_data, write_extxyz_structures
from workflow_backends.lammps.observation_source import (
    LammpsObservationSource,
    validate_lammps_task_terminals,
)
from workflow_backends.lammps.execution_mode import (
    command_for_mode,
    select_lammps_execution_mode,
)
from workflow_artifacts.lifecycle.sampling_models import EffectiveSamplingModelPlan
from workflow_backends.sampling import (
    SamplingSeedSelection,
    seed_selection_manifest_payload,
)
from workflow_artifacts.lifecycle.sample_conditions import SamplingPlan, SystemSamplingCondition
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
from workflow_core.structures.cell_program import resolve_cell_program_history
from workflow_runtime.lifecycle.progress import (
    OperationProgressReporter,
    SamplingSnapshot,
    SamplingSystemProgress,
    SamplingTaskDetail,
)


class LammpsExecutor(Protocol):
    def run(self, task_dir: Path, task: dict) -> object: ...


@dataclass(frozen=True, slots=True)
class LammpsTask:
    task_index: int
    task_dir: Path
    payload: dict


@dataclass(slots=True)
class LammpsSampler:
    run_json_path: Path
    iter_index: int
    round_index: int
    executor: LammpsExecutor

    def sample(
        self,
        *,
        recipe: str | Path | None = None,
        recipes: dict[str, str | Path] | None = None,
        group_recipes: dict[str, str | Path] | None = None,
        sample_dir: str | Path,
        models: Sequence[str | Path] | EffectiveSamplingModelPlan,
        type_map: tuple[str, ...],
        systems: dict[str, list[Atoms]],
        steps: int | None,
        temperatures_k: Sequence[float] | None = None,
        ensemble: str | None = None,
        pressure_bar: float | None = None,
        plan: SamplingPlan | ExplicitSamplingPlan | None = None,
        plumed_sources: dict[str, str | Path],
        seed_selections: tuple[SamplingSeedSelection, ...] = (),
        progress_callback: Callable[[SamplingSnapshot], None] | None = None,
        progress_reporter_factory: Callable[
            [str], OperationProgressReporter
        ]
        | None = None,
    ) -> None:
        effective_plan = _effective_plan(
            plan=plan,
            systems=systems,
            round_index=self.round_index,
            temperatures_k=temperatures_k,
            ensemble=ensemble,
            pressure_bar=pressure_bar,
        )
        explicit = isinstance(effective_plan, ExplicitSamplingPlan)
        all_round_tasks = (
            effective_plan.tasks_for_round(self.round_index)
            if explicit
            else effective_plan.tasks
        )
        indexed_round_tasks = tuple(
            (task_index, task)
            for task_index, task in enumerate(all_round_tasks)
            if not explicit or task.kind not in {"neb", "relax", "gcmc_md", "widom"}
        )
        round_tasks = tuple(task for _task_index, task in indexed_round_tasks)
        if not round_tasks:
            raise ValueError("LAMMPS sampling requires at least one active MD task")
        active_condition_group_ids = tuple(
            dict.fromkeys(
                task.condition_group_id for task in round_tasks
            )
        ) if explicit else ()
        recipe_sources = _recipe_sources(
            recipe=recipe,
            recipes=recipes,
            group_recipes=group_recipes,
            ensembles=effective_plan.ensembles,
            condition_group_ids=(
                active_condition_group_ids
            ),
        )
        if explicit:
            groups = {
                group.condition_group_id: group
                for group in effective_plan.condition_groups
            }
            active_condition_groups = tuple(
                groups[group_id] for group_id in active_condition_group_ids
            )
            for group_id in active_condition_group_ids:
                group = groups[group_id]
                self._validate_recipe(
                    recipe_sources[group.condition_group_id], kind=group.kind,
                    ensemble=('npt' if group.pressures_bar else 'nvt') if group.kind == 'md_mc' else None,
                )
        else:
            for source in dict.fromkeys(recipe_sources.values()):
                self._validate_recipe(source)
        model_paths = (
            models.paths
            if isinstance(models, EffectiveSamplingModelPlan)
            else tuple(Path(model) for model in models)
        )
        if not model_paths:
            raise ValueError("lifecycle LAMMPS sampling requires at least one DeepMD model")
        suffixes = {Path(model).suffix for model in model_paths}
        if len(suffixes) != 1 or next(iter(suffixes)) not in {".pb", ".pt2"}:
            raise ValueError(
                "lifecycle LAMMPS models must use one consistent DeepMD ABI suffix"
            )
        cli = "tf" if next(iter(suffixes)) == ".pb" else "pt"
        if isinstance(models, EffectiveSamplingModelPlan):
            recipe_kinds = {
                model.deepmd_recipe_kind for model in models.models
            }
            declared_clis = {model.cli for model in models.models}
            if len(recipe_kinds) != 1 or declared_clis != {cli}:
                raise ValueError(
                    "lifecycle LAMMPS effective plan has mixed recipe kind or ABI"
                )
            recipe_kind = next(iter(recipe_kinds))
            require_strict_native_evidence = (
                models.compressed or recipe_kind == "dpa4c"
            )
        else:
            recipe_kind = "standard"
            require_strict_native_evidence = False
        if isinstance(effective_plan, ExplicitSamplingPlan):
            require_strict_native_evidence = True
        execution_mode = select_lammps_execution_mode(
            sampler_role=f"sampler.lammps.{cli}",
            deepmd_recipe_kind=recipe_kind,
            model_count=len(model_paths),
        )
        sampler_role = f"sampler.lammps.{cli}"
        executor_context = getattr(self.executor, "context", None)
        context_role = getattr(executor_context, "role", None)
        if context_role is not None and context_role != sampler_role:
            raise ValueError(
                f"lifecycle LAMMPS executor requires operation role {sampler_role}"
            )
        profile_id = getattr(executor_context, "profile_id", None)
        if plumed_sources and set(plumed_sources) != set(systems):
            raise ValueError("lifecycle LAMMPS PLUMED must be supplied for all systems or none")
        root = Path(sample_dir); task_root = root / "tasks"
        if seed_selections:
            _publish_seed_selection_manifest(
                root / "seed_selection.json",
                iter_index=self.iter_index,
                round_index=self.round_index,
                selections=seed_selections,
            )
        selected_seeds: dict[str, Atoms] = {}
        for system_id, seeds in systems.items():
            if not seeds:
                raise ValueError(f"lifecycle LAMMPS system {system_id} has no seed frames")
            selected_seeds[system_id] = canonical_selected_seed(
                seeds[0 if explicit else self._seed_index(system_id, len(seeds))],
                system_id=system_id,
            )
        publish_selected_seeds(root / "seeds.xyz", tuple(selected_seeds.values()))
        total_tasks = len(round_tasks)
        active_system_ids = tuple(
            system_id
            for system_id in selected_seeds
            if any(task.system_id == system_id for task in round_tasks)
        )
        task_counts = {
            system_id: sum(1 for task in round_tasks if task.system_id == system_id)
            for system_id in active_system_ids
        }
        task_step_totals = tuple(
            int(getattr(task, "steps", steps)) for task in round_tasks
        )
        sampling_role_totals = (
            tuple(
                (
                    role_id,
                    sum(1 for task in round_tasks if task.system_id == role_id),
                    sum(
                        int(getattr(task, "steps", steps))
                        for task in round_tasks
                        if task.system_id == role_id
                    ),
                )
                for role_id in active_system_ids
            )
            if explicit
            else ()
        )
        condition_group_totals = (
            tuple(
                (
                    group.condition_group_id,
                    sum(
                        1
                        for task in round_tasks
                        if task.condition_group_id == group.condition_group_id
                    ),
                    sum(
                        int(task.steps)
                        for task in round_tasks
                        if task.condition_group_id == group.condition_group_id
                    ),
                )
                for group in active_condition_groups
            )
            if explicit
            else ()
        )
        ensembles = tuple(dict.fromkeys(task.ensemble for task in round_tasks))
        temperature_values = (
            sum(
                len(group.temperatures_k)
                if group.kind not in {"nvt_program", "nvt_cell_program", "npt_program"}
                else 1
                for group in active_condition_groups
            )
            if explicit
            else effective_plan.total_temperature_values
        )
        completed_by_system = {system_id: 0 for system_id in active_system_ids}
        started_at = time.monotonic()
        progress_lock = threading.RLock()
        task_details: dict[str, SamplingTaskDetail] = {}
        mps_phase: str | None = None
        detail_limit = 4
        successful_tasks = 0
        usable_failed_frames = 0
        failure_summaries: list[tuple[int, str]] = []

        def progress(*, active_task_id=None, active_system_id=None, active_temperature_k=None, active_pressure_bar=None):
            completed = sum(completed_by_system.values())
            elapsed = time.monotonic() - started_at
            eta = None if completed == 0 or completed >= total_tasks else elapsed * (total_tasks - completed) / completed
            ordered_details = tuple(task_details[key] for key in sorted(task_details))
            return SamplingSnapshot(
                backend="LAMMPS",
                model_shape="single" if len(model_paths) == 1 else "committee",
                completed_tasks=completed,
                total_tasks=total_tasks,
                systems=len(active_system_ids),
                temperatures=temperature_values,
                ensemble=ensembles[0].upper() if len(ensembles) == 1 else "MIXED",
                pressure_bar=active_pressure_bar,
                steps_per_task=(
                    max(int(getattr(task, "steps", steps)) for task in round_tasks)
                ),
                plumed=bool(plumed_sources),
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                rows=(
                    SamplingSystemProgress("ALL", completed, total_tasks),
                    *(SamplingSystemProgress(system_id, completed_by_system[system_id], task_counts[system_id]) for system_id in active_system_ids),
                ),
                active_task_id=active_task_id,
                active_system_id=active_system_id,
                active_temperature_k=active_temperature_k,
                task_details=ordered_details[:detail_limit],
                omitted_task_details=max(0, len(ordered_details) - detail_limit),
                mps_phase=mps_phase,
                task_step_totals=task_step_totals if explicit else (),
                sampling_role_totals=sampling_role_totals,
                condition_group_totals=condition_group_totals,
            )

        def publish_progress() -> None:
            if progress_callback is None:
                return
            with progress_lock:
                progress_callback(progress())

        def update_task(task_id: str, snapshot: SamplingSnapshot) -> None:
            with progress_lock:
                current = task_details.get(task_id)
                if current is None:
                    raise ValueError(f"lifecycle LAMMPS progress update belongs to closed task {task_id!r}")
                task_details[task_id] = SamplingTaskDetail(
                    task_id=task_id,
                    system_id=current.system_id,
                    temperature_k=current.temperature_k,
                    step=snapshot.active_step,
                    speed=snapshot.speed,
                    speed_unit=snapshot.speed_unit,
                    diagnostic_locator=current.diagnostic_locator,
                    state=current.state,
                    message=current.message,
                )
                if progress_callback is not None:
                    progress_callback(progress())

        publish_progress()
        prepared_tasks: list[LammpsTask] = []
        for task_index, condition in indexed_round_tasks:
            system_id = condition.system_id
            selected = selected_seeds[system_id]
            temperature = (
                condition.program.segments[0].start_temperature_k
                if explicit
                and condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                else condition.temperature_k
            )
            pressure = condition.pressure_bar
            source = recipe_sources[
                condition.condition_group_id if explicit else condition.ensemble
            ]
            cadence = self._cadence(source)
            cell_history = (
                resolve_cell_program_history(
                    selected,
                    condition.program,
                    trajectory_every_steps=cadence,
                )
                if explicit and condition.kind == "nvt_cell_program"
                else None
            )
            task_dir = task_root / f"task_{task_index:06d}"
            task_dir.mkdir(parents=True, exist_ok=True)
            native_seed = (
                explicit_task_seed(
                    iter_index=self.iter_index,
                    round_index=self.round_index,
                    sampling_role_id=condition.sampling_role_id,
                )
                if explicit
                else self._task_seed(system_id)
            )
            task_steps = int(condition.steps if explicit else steps)
            md_mc = getattr(condition, 'md_mc', None)
            if md_mc is not None:
                from workflow_core.structures.md_mc import validate_seed
                validate_seed(selected, md_mc, type_map)
            self._write_task(
                task_dir,
                source,
                selected,
                model_paths,
                type_map,
                task_steps,
                temperature,
                pressure,
                native_seed,
                plumed_sources.get(system_id),
                execution_mode,
                program=(condition.program if explicit else None),
                cell_history=cell_history,
                md_mc=md_mc,
            )
            schedule_digests = (
                tuple(
                    (name, _file_digest(task_dir / name))
                    for name in (
                        "dice_temperature_start.dat",
                        "dice_temperature_stop.dat",
                        "dice_temperature_steps.dat",
                    )
                )
                if explicit and condition.kind in {"nvt_program", "npt_program"}
                else (
                    (("dice_cell_schedule.json", _file_digest(
                        task_dir / "dice_cell_schedule.json"
                    )),)
                    if explicit and condition.kind == "nvt_cell_program"
                    else ()
                )
            )
            temperature_program = (
                tuple(
                    {
                        "start_temperature_k": segment.start_temperature_k,
                        "stop_temperature_k": segment.stop_temperature_k,
                        "steps": segment.steps,
                        "end_step": segment.end_step,
                        **(
                            {"cell": cell_history.segments[index].declared_cell}
                            if cell_history is not None
                            and cell_history.segments[index].declared_cell is not None
                            else {}
                        ),
                    }
                    for index, segment in enumerate(condition.program.segments)
                )
                if explicit
                and condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                else ()
            )
            ordered_models = tuple(
                {
                    "presentation": presentation.name,
                    "sha256": _file_digest(presentation),
                }
                for presentation in sorted(task_dir.glob("graph_*"))
            )
            task = {
                "system_id": system_id,
                "ensemble": condition.ensemble,
                "temperature_k": float(temperature),
                "pressure_bar": pressure,
                "steps": task_steps,
                "cadence": cadence,
                "type_map": tuple(type_map),
                "model_count": len(model_paths),
                "atom_count": len(selected),
                "deepmd_recipe_kind": recipe_kind,
                "sampler_role": sampler_role,
                "environment_profile_id": profile_id,
                "execution_mode": execution_mode,
                "ordered_models": ordered_models,
                **(
                    {
                        "task_id": condition.task_id,
                        "task_index": task_index,
                        "condition_group_id": condition.condition_group_id,
                        "sampling_role_id": condition.sampling_role_id,
                        "selection_group_id": condition.selection_group_id,
                        "context_system_id": condition.selection_group_id,
                        "source_frame_index": condition.source_frame_index,
                        "selected_structure_id": str(selected.info["structure_id"]),
                        "context_temperature_k": (
                            None
                            if condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                            else float(temperature)
                        ),
                        "kind": condition.kind,
                        "native_seed": native_seed,
                        "recipe_binding": condition.recipe_binding,
                        "recipe_sha256": _file_digest(source),
                        "rendered_input_sha256": _file_digest(task_dir / "input.lammps"),
                        "setup_sha256": _file_digest(task_dir / "dice_task_setup.lammps"),
                        "temperature_program": temperature_program,
                        "segment_boundaries": (
                            condition.program.boundaries
                            if condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                            else ()
                        ),
                        "schedule_digests": schedule_digests,
                        **(
                            {
                                "resolved_cell_history": cell_history.context_payload(),
                                "cell_history_sha256": cell_history.sha256,
                                "expected_dump_steps": cell_history.expected_dump_steps,
                            }
                            if cell_history is not None
                            else {}
                        ),
                    }
                    if explicit
                    else {}
                ),
            }
            if md_mc is not None:
                task['md_mc'] = asdict(md_mc)
            if explicit:
                if md_mc is not None:
                    from workflow_backends.lammps.identity_swap import mc_seed
                context_class = (
                    MdMcSampleTaskContext if md_mc is not None else CellProgramSampleTaskContext
                    if condition.kind == "nvt_cell_program"
                    else ExplicitSampleTaskContext
                )
                write_task_context(
                    task_dir / "task_context.json",
                    context_class(
                        schema="dice.lifecycle.sample-task-context",
                        schema_revision=(
                            CELL_PROGRAM_TASK_CONTEXT_SCHEMA_REVISION
                            if condition.kind == "nvt_cell_program"
                            else 4 if md_mc is not None
                            else 2
                        ),
                        task_index=task_index,
                        task_id=condition.task_id,
                        condition_group_id=condition.condition_group_id,
                        kind=condition.kind,
                        sampling_role_id=condition.sampling_role_id,
                        selection_group_id=condition.selection_group_id,
                        system_id=condition.selection_group_id,
                        source_frame_index=condition.source_frame_index,
                        selected_structure_id=str(selected.info["structure_id"]),
                        temperature_k=(
                            None
                            if condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                            else float(temperature)
                        ),
                        pressure_bar=pressure,
                        total_steps=task_steps,
                        trajectory_every_steps=task["cadence"],
                        native_timestep=self._timestep(source),
                        recipe_binding=condition.recipe_binding,
                        recipe_sha256=_file_digest(source),
                        rendered_input_sha256=_file_digest(task_dir / "input.lammps"),
                        setup_sha256=_file_digest(task_dir / "dice_task_setup.lammps"),
                        native_seed=native_seed,
                        ordered_models=tuple(item["presentation"] for item in ordered_models),
                        model_digests=tuple(item["sha256"] for item in ordered_models),
                        temperature_program=temperature_program,
                        segment_boundaries=(
                            condition.program.boundaries
                            if condition.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                            else ()
                        ),
                        schedule_digests=schedule_digests,
                        **({'md_mc': {
                            'parameters': asdict(md_mc),
                            'seed_cell': selected.cell.array.tolist(),
                            'seed_numbers': selected.numbers.tolist(),
                            'type_map': list(type_map),
                            'member_ids': list(models.member_ids) if isinstance(models, EffectiveSamplingModelPlan) else (['student'] if len(model_paths) == 1 else [f'student_{i+1:02d}' for i in range(len(model_paths))]),
                            'execution_mode': execution_mode,
                            'mc_seed': mc_seed(iter_index=self.iter_index, round_index=self.round_index, native_seed=native_seed),
                        }} if md_mc is not None else {}),
                        **(
                            {
                                "resolved_cell_history": cell_history.context_payload(),
                                "cell_history_sha256": cell_history.sha256,
                                "expected_dump_steps": cell_history.expected_dump_steps,
                            }
                            if cell_history is not None
                            else {}
                        ),
                    ),
                )
            else:
                write_task_context(
                    task_dir / "task_context.json",
                    SampleTaskContext.create(
                    task_index=task_index,
                    system_id=system_id,
                    ensemble=condition.ensemble,
                    temperature_k=float(temperature),
                    pressure_bar=pressure,
                    steps=task_steps,
                    cadence=task["cadence"],
                    selected_seed_id=str(selected.info["structure_id"]),
                    ),
                )
            prepared_tasks.append(LammpsTask(task_index, task_dir, task))

        def run_prepared(prepared: LammpsTask, execution_environment: dict[str, str]) -> None:
            task_index = prepared.task_index
            task_dir = prepared.task_dir
            task = prepared.payload
            system_id = str(task["system_id"])
            task_id = f"task_{task_index:06d}"
            with progress_lock:
                active = progress(
                    active_task_id=task_id,
                    active_system_id=system_id,
                    active_temperature_k=float(task["temperature_k"]),
                    active_pressure_bar=task["pressure_bar"],
                )
                task_details[task_id] = SamplingTaskDetail(
                    task_id=task_id,
                    system_id=system_id,
                    temperature_k=float(task["temperature_k"]),
                    diagnostic_locator=(
                        f"tasks/{task_id}/lammps_sample.stderr.log"
                    ),
                )
                if progress_callback is not None:
                    active = progress(
                        active_task_id=task_id,
                        active_system_id=system_id,
                        active_temperature_k=float(task["temperature_k"]),
                        active_pressure_bar=task["pressure_bar"],
                    )
                    progress_callback(active)
            if progress_callback is not None:
                reporter = OperationProgressReporter(
                    operation_id=task_id,
                    on_snapshot=lambda snapshot, task_id=task_id: update_task(task_id, snapshot),
                )
            else:
                reporter = None if progress_reporter_factory is None else progress_reporter_factory(task_id)
            try:
                _run_with_optional_progress(
                    self.executor.run,
                    task_dir,
                    task,
                    progress_reporter=reporter,
                    sampling_snapshot=active,
                    execution_environment=execution_environment,
                )
            except BaseException as exc:
                with progress_lock:
                    current = task_details.get(task_id)
                    if current is not None:
                        task_details[task_id] = SamplingTaskDetail(
                            task_id=current.task_id,
                            system_id=current.system_id,
                            temperature_k=current.temperature_k,
                            step=current.step,
                            speed=current.speed,
                            speed_unit=current.speed_unit,
                            diagnostic_locator=current.diagnostic_locator,
                            state="failed",
                            message=str(exc)[-400:] or type(exc).__name__,
                        )
                        if progress_callback is not None:
                            progress_callback(progress())
                raise
            finally:
                if reporter is not None and not reporter.closed:
                    reporter.close()

        def complete(prepared: LammpsTask) -> None:
            if prepared.payload.get('kind') == 'md_mc':
                from workflow_backends.lammps.identity_swap import finalize, validate_result
                if int(prepared.payload['model_count']) > 1:
                    self.executor.finalize_md_mc(prepared.task_dir)
                else:
                    finalize(prepared.task_dir)
                _context, result = validate_result(prepared.task_dir)
                if result['status'] != 'completed':
                    raise ValueError(result['failure'])
                with progress_lock:
                    task_details.pop(f'task_{prepared.task_index:06d}', None)
                    completed_by_system[str(prepared.payload['system_id'])] += 1
                    if progress_callback is not None:
                        progress_callback(progress())
                return
            if require_strict_native_evidence:
                terminals = self._validate_terminal_outputs(
                    prepared.task_dir,
                    cadence=int(prepared.payload["cadence"]),
                    steps=int(prepared.payload["steps"]),
                    require_deviation=len(model_paths) > 1,
                    require_exact_cadence=(
                        "condition_group_id" in prepared.payload
                    ),
                    expected_atom_count=int(prepared.payload["atom_count"]),
                )
                if prepared.payload.get("kind") == "nvt_cell_program":
                    context = load_task_context(
                        prepared.task_dir / "task_context.json"
                    )
                    if not isinstance(context, CellProgramSampleTaskContext):
                        raise ValueError(
                            "LAMMPS cell program requires revision-3 task context"
                        )
                    validate_lammps_cell_program_trajectory(
                        task_dir=prepared.task_dir,
                        context=context,
                        type_map=tuple(prepared.payload["type_map"]),
                    )
                self._publish_native_execution(
                    prepared.task_dir,
                    prepared.payload,
                    terminals=terminals,
                )
            else:
                self._validate_legacy_terminal_outputs(
                    prepared.task_dir,
                    require_deviation=len(model_paths) > 1,
                )
            with progress_lock:
                task_details.pop(f"task_{prepared.task_index:06d}", None)
                completed_by_system[str(prepared.payload["system_id"])] += 1
                if progress_callback is not None:
                    progress_callback(progress())

        policy = getattr(self.executor, "execution_policy", None)
        parallel_tasks = 1 if policy is None else int(policy.parallel_tasks)
        allocation = getattr(executor_context, "allocation", None)
        use_allocation_pool = (
            parallel_tasks == 1
            or (
                allocation is not None
                and (
                    allocation.gpus_per_node > 1
                    or allocation.mode != "pool"
                    or allocation.per_gpu != 1
                )
            )
        )

        def execute_tolerant(
            prepared: LammpsTask,
            execution_environment: dict[str, str],
        ) -> TaskOutcome:
            registry = getattr(execution_environment, 'process_registry', None)
            if registry is not None:
                with operation_execution_scope(environment_overrides=execution_environment, process_registry=registry):
                    return execute_tolerant_scoped(prepared, execution_environment)
            return execute_tolerant_scoped(prepared, execution_environment)

        def execute_tolerant_scoped(prepared, execution_environment) -> TaskOutcome:
            nonlocal successful_tasks, usable_failed_frames
            identity = f"task_{prepared.task_index:06d}"

            def run() -> TaskOutcome:
                run_prepared(prepared, execution_environment)
                complete(prepared)
                return TaskOutcome(identity, "completed")

            def retain(exc: Exception) -> TaskOutcome:
                nonlocal usable_failed_frames
                if prepared.payload.get('kind') == 'md_mc':
                    from workflow_backends.lammps.identity_swap import finalize, validate_result
                    existing = prepared.task_dir/'md_mc_result.json'
                    retained = None
                    if existing.is_file():
                        try:
                            retained = validate_result(prepared.task_dir)[1]
                        except (OSError, ValueError, KeyError, TypeError):
                            retained = None
                    if retained is None or retained['status'] != 'failed':
                        if int(prepared.payload['model_count']) > 1:
                            try:
                                self.executor.finalize_md_mc(prepared.task_dir, failure=str(exc))
                            except Exception as evaluation_error:
                                finalize(prepared.task_dir, failure=f'{exc}; evaluation child failed: {evaluation_error}')
                        else:
                            finalize(prepared.task_dir, failure=str(exc))
                    _context, result = validate_result(prepared.task_dir)
                    usable = len(result['observations'])
                    with progress_lock:
                        usable_failed_frames += usable
                        completed_by_system[str(prepared.payload['system_id'])] += 1
                        failure_summaries.append((prepared.task_index, str(exc)[-400:]))
                    return TaskOutcome(identity, 'failed', (str(prepared.task_dir/'md_mc_result.json'),) if usable else (), str(exc)[-400:])
                if prepared.payload.get("kind") == "nvt_cell_program":
                    for name in (
                        "traj.lammpstrj", "model_devi.out",
                        "native_execution.json", "sampling_failure.json",
                    ):
                        artifact = prepared.task_dir / name
                        if artifact.is_file() or artifact.is_symlink():
                            artifact.unlink()
                    raise exc
                usable = self._publish_sampling_failure(prepared, exc)
                with progress_lock:
                    usable_failed_frames += usable
                    completed_by_system[str(prepared.payload["system_id"])] += 1
                    failure_summaries.append((prepared.task_index,
                        str(exc)[-400:] or type(exc).__name__))
                references = (str(prepared.task_dir / "sampling_failure.json"),) if usable else ()
                return TaskOutcome(identity, "failed", references, str(exc)[-400:])

            outcome = execute_task(identity, run=run, retain_failure=retain)
            if outcome.status == "completed":
                with progress_lock:
                    successful_tasks += 1
            return outcome

        def require_usable_pool() -> None:
            if successful_tasks == 0 and usable_failed_frames == 0:
                summary = "; ".join(
                    f"task {index}: {message}"
                    for index, message in sorted(failure_summaries)[:8]
                )
                raise RuntimeError(
                    "lifecycle LAMMPS sample block has no usable positive-step frames"
                    + (f"; failures: {summary}" if summary else "")
                )

        if use_allocation_pool:
            def execute_one(prepared: LammpsTask) -> TaskOutcome:
                return execute_tolerant(prepared, {})

            try:
                if allocation is not None and allocation.per_gpu != 1:
                    open_device_sessions = getattr(
                        self.executor, "open_mps_device_sessions", None
                    )
                    if not callable(open_device_sessions):
                        raise ValueError(
                            "LAMMPS pool per_gpu > 1 requires an MPS device-session capability"
                        )
                    devices = sampling_visible_devices(allocation.gpus_per_node)
                    with open_device_sessions(root, devices) as sessions:
                        if set(sessions) != set(devices):
                            raise ValueError(
                                "LAMMPS MPS device sessions do not match allocated devices"
                            )
                        execute_sampling_tasks(
                            tuple(prepared_tasks),
                            context=executor_context,
                            task_id=lambda prepared: f"task_{prepared.task_index:06d}",
                            execute=execute_one,
                            allow_per_gpu=True,
                            assignment_environment=lambda assignment: sessions[
                                assignment.visible_device
                            ],
                        )
                else:
                    execute_sampling_tasks(
                        tuple(prepared_tasks),
                        context=executor_context,
                        task_id=lambda prepared: f"task_{prepared.task_index:06d}",
                        execute=execute_one,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"lifecycle LAMMPS sample block failed atomically: {exc}"
                ) from exc
            require_usable_pool()
            return

        open_mps_session = getattr(self.executor, "open_mps_session", None)
        if not callable(open_mps_session):
            raise ValueError(
                "lifecycle LAMMPS parallel execution requires a CUDA MPS session; "
                "ordinary CUDA process sharing is unsupported"
            )
        try:
            with _handled_sigterm():
                with progress_lock:
                    mps_phase = "starting"
                    if progress_callback is not None:
                        progress_callback(progress())
                with open_mps_session(root) as raw_environment:
                    with progress_lock:
                        mps_phase = "active"
                        if progress_callback is not None:
                            progress_callback(progress())
                    execution_environment = {} if raw_environment is None else raw_environment
                    self._run_worker_pool(
                        prepared_tasks,
                        parallel_tasks=parallel_tasks,
                        run_task=lambda prepared: execute_tolerant(
                            prepared, execution_environment
                        ),
                        complete_task=lambda _prepared: None,
                        cancel_active=getattr(execution_environment, "cancel_active", lambda: None),
                    )
                    with progress_lock:
                        mps_phase = "stopping"
                        if progress_callback is not None:
                            progress_callback(progress())
        except Exception as exc:
            task_index = getattr(exc, "task_index", None)
            location = "" if task_index is None else f" at task {task_index}"
            secondary = getattr(exc, "secondary_failures", ())
            secondary_text = ""
            if secondary:
                secondary_text = "; secondary peer diagnostics: " + "; ".join(
                    f"task {index}: {message[:400]}" for index, message in secondary[:8]
                )
            raise RuntimeError(
                f"lifecycle LAMMPS sample block failed atomically{location}: {exc}{secondary_text}"
            ) from exc
        require_usable_pool()

    @staticmethod
    def _run_worker_pool(
        tasks: Sequence[LammpsTask],
        *,
        parallel_tasks: int,
        run_task: Callable[[LammpsTask], None],
        complete_task: Callable[[LammpsTask], None],
        cancel_active: Callable[[], None],
    ) -> None:
        pending = iter(tasks)
        active: dict[Future[None], LammpsTask] = {}
        with ThreadPoolExecutor(max_workers=parallel_tasks, thread_name_prefix="dice-lammps-mps") as pool:
            try:
                for _ in range(min(parallel_tasks, len(tasks))):
                    prepared = next(pending)
                    active[pool.submit(run_task, prepared)] = prepared
                while active:
                    done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: active[item].task_index):
                        prepared = active.pop(future)
                        try:
                            future.result()
                            complete_task(prepared)
                        except BaseException as exc:
                            try:
                                setattr(exc, "task_index", prepared.task_index)
                            except Exception:
                                pass
                            raise
                        try:
                            next_task = next(pending)
                        except StopIteration:
                            continue
                        active[pool.submit(run_task, next_task)] = next_task
            except BaseException as primary:
                cancel_active()
                secondary: list[tuple[int, str]] = []
                for future, prepared in sorted(
                    active.items(), key=lambda item: item[1].task_index
                ):
                    try:
                        future.result()
                    except BaseException as peer_error:
                        secondary.append((prepared.task_index, str(peer_error)))
                if secondary:
                    try:
                        setattr(primary, "secondary_failures", tuple(secondary[:8]))
                    except Exception:
                        pass
                raise

    def _validate_recipe(self, recipe: Path, *, kind: str | None = None, ensemble: str | None = None) -> None:
        text = recipe.read_text(encoding="utf-8")
        if kind == 'md_mc':
            from workflow_backends.lammps.identity_swap import validate_recipe
            validate_recipe(text, ensemble=ensemble)
            return
        if "DICE_BEGIN" in text or "DICE_END" in text:
            raise ValueError("lifecycle LAMMPS source recipe must not contain managed markers")
        includes = re.findall(r"(?m)^\s*include\s+dice_task_setup\.lammps\s*$", text)
        if len(includes) != 1: raise ValueError("lifecycle LAMMPS recipe requires exactly one include dice_task_setup.lammps")
        capability = re.findall(
            r"(?m)^\s*#\s*DICE_CAPABILITY:\s*nvt_cell_program\s*$", text
        )
        if kind == "nvt_cell_program":
            cadence = re.findall(
                r"(?m)^\s*variable\s+dice_trajectory_every_steps\s+equal\s+([1-9]\d*)\s*$",
                text,
            )
            if len(capability) != 1 or len(cadence) != 1:
                raise ValueError(
                    "lifecycle LAMMPS nvt_cell_program recipe requires its explicit capability and cadence"
                )
            if re.search(
                r"(?mi)^\s*(?:run|clear|reset_timestep|read_data|read_restart|restart|write_restart)\b",
                text,
            ) or re.search(
                r"(?mi)^\s*fix\s+\S+\s+\S+\s+(?:nvt|npt|deform)\b", text
            ):
                raise ValueError(
                    "lifecycle LAMMPS nvt_cell_program recipe may not own managed dynamics"
                )
            return
        if capability:
            raise ValueError(
                "lifecycle LAMMPS cell-program capability is valid only for nvt_cell_program"
            )
        if kind not in {"nvt_program", "npt_program"}:
            runs = re.findall(r"(?m)^\s*run\s+\$\{dice_run_steps\}\s*$", text)
            if len(runs) != 1: raise ValueError("lifecycle LAMMPS recipe requires exactly one main run ${dice_run_steps}")
            return
        cadence = re.findall(
            r"(?m)^\s*variable\s+dice_trajectory_every_steps\s+equal\s+([1-9]\d*)\s*$",
            text,
        )
        fix_pattern = (
            r"(?m)^\s*fix\s+dice_npt\s+all\s+npt\s+temp\s+\$\{dice_tstart\}\s+\$\{dice_tstop\}\s+\S+\s+iso\s+\$\{dice_pressure\}\s+\$\{dice_pressure\}\s+\S+\s*$"
            if kind == "npt_program"
            else r"(?m)^\s*fix\s+dice_nvt\s+all\s+nvt\s+temp\s+\$\{dice_tstart\}\s+\$\{dice_tstop\}\s+\S+\s*$"
        )
        unfix_pattern = (
            r"(?m)^\s*unfix\s+dice_npt\s*$"
            if kind == "npt_program"
            else r"(?m)^\s*unfix\s+dice_nvt\s*$"
        )
        required = (
            r"(?m)^\s*variable\s+dice_tstart\s+file\s+dice_temperature_start\.dat\s*$",
            r"(?m)^\s*variable\s+dice_tstop\s+file\s+dice_temperature_stop\.dat\s*$",
            r"(?m)^\s*variable\s+dice_segment_steps\s+file\s+dice_temperature_steps\.dat\s*$",
            r"(?m)^\s*label\s+dice_segment_loop\s*$",
            fix_pattern,
            r"(?m)^\s*run\s+\$\{dice_segment_steps\}\s*$",
            unfix_pattern,
            r"(?m)^\s*next\s+dice_tstart\s+dice_tstop\s+dice_segment_steps\s*$",
            r"(?m)^\s*jump\s+SELF\s+dice_segment_loop\s*$",
        )
        matches = [re.findall(pattern, text) for pattern in required]
        if len(cadence) != 1 or any(len(items) != 1 for items in matches):
            raise ValueError(f"lifecycle LAMMPS {kind} recipe requires the canonical schedule loop")
        include_at = text.index(re.search(r"(?m)^\s*include\s+dice_task_setup\.lammps\s*$", text).group())
        cadence_at = text.index(re.search(r"(?m)^\s*variable\s+dice_trajectory_every_steps", text).group())
        positions = [re.search(pattern, text).start() for pattern in required]
        if cadence_at >= include_at or positions != sorted(positions):
            raise ValueError(f"lifecycle LAMMPS {kind} recipe loop order is invalid")
        loop_at = positions[3]
        if re.search(r"(?mi)^\s*(?:clear|reset_timestep|read_data|read_restart)\b", text):
            raise ValueError(f"lifecycle LAMMPS {kind} recipe may not restart task state")
        velocities = tuple(re.finditer(r"(?mi)^\s*velocity\b", text))
        if len(velocities) > 1 or any(match.start() >= loop_at for match in velocities):
            raise ValueError(f"lifecycle LAMMPS {kind} recipe velocity creation is allowed only once before the loop")

    def _write_task(self, task_dir: Path, recipe: Path, seed: Atoms, models: Sequence[str | Path], type_map: tuple[str, ...], steps: int, temperature: float, pressure: float | None, task_seed: int, plumed: str | Path | None, execution_mode: str, *, program: object | None = None, cell_history: object | None = None, md_mc=None) -> None:
        write_extxyz_structures(task_dir / "seed.xyz", [seed], allow_empty=False)
        atoms_to_lammps_data(_lammps_seed(seed), task_dir / "conf.lmp", type_map=list(type_map))
        names=[]
        for index, model in enumerate(models):
            model_path = Path(model)
            if not model_path.is_file() or model_path.stat().st_size <= 0:
                raise ValueError(f"lifecycle LAMMPS DeepMD model is missing or empty: {model_path}")
            suffix = model_path.suffix
            if suffix not in {".pb", ".pt2"}:
                raise ValueError(
                    f"lifecycle LAMMPS DeepMD model must preserve a .pb or .pt2 suffix: {model}"
                )
            target=task_dir/f"graph_{index:03d}{suffix}"; link_model_file(model_path,target); names.append(target.name)
        cadence=self._cadence(recipe)
        masses = [f"mass {index} {float(atomic_masses[atomic_numbers[symbol]]):.8g}" for index, symbol in enumerate(type_map, start=1)]
        pair_style = f"pair_style deepmd {' '.join(names)}"
        if len(names) > 1:
            pair_style += f" out_file model_devi.out out_freq {cadence}"
        if md_mc is not None:
            pair_style = f'pair_style {"deepmd/kk" if execution_mode == "kokkos" else "deepmd"} {names[0]}'
        setup=[f"variable dice_run_steps equal {int(steps)}", f"variable dice_temperature equal {float(temperature)}", f"variable dice_seed equal {task_seed}"]
        if execution_mode == "kokkos":
            setup.append("atom_modify map yes")
        setup.append("read_data conf.lmp")
        if md_mc is not None and pressure is not None:
            from ase.calculators.lammps import Prism
            from workflow_backends.lammps.identity_swap import validate_recipe
            protocol = validate_recipe(recipe.read_text(), ensemble='npt')
            if protocol['pressure_mode'] == 'tri' and not Prism(seed.cell).is_skewed():
                setup.append('change_box all triclinic')
        if cell_history is not None and bool(
            build_lammps_cell_schedule(cell_history)[
                "requires_triclinic_conversion"
            ]
        ):
            setup.append("change_box all triclinic")
        setup.extend(
            [
                *masses,
                pair_style,
                f"pair_coeff * * {' '.join(type_map)}",
                f"dump 1 all custom {cadence} traj.lammpstrj id type x y z" + (' fx fy fz' if md_mc is not None else ''),
                "dump_modify 1 sort id format float %.16g",
                "thermo_modify flush yes",
            ]
        )
        if pressure is not None: setup.append(f"variable dice_pressure equal {float(pressure)}")
        (task_dir/"dice_task_setup.lammps").write_text("\n".join(setup)+"\n",encoding="utf-8")
        recipe_text = recipe.read_text(encoding="utf-8")
        if cell_history is not None:
            schedule = write_lammps_cell_schedule(
                task_dir / "dice_cell_schedule.json", cell_history
            )
            recipe_text = render_lammps_cell_program_input(recipe_text, schedule)
        (task_dir/"input.lammps").write_text(recipe_text,encoding="utf-8")
        if md_mc is not None:
            from workflow_backends.lammps.identity_swap import render, mc_seed
            native_mc_seed = mc_seed(iter_index=self.iter_index, round_index=self.round_index, native_seed=task_seed)
            (task_dir/'dice_md_mc.lammps').write_text(render(md_mc, steps=steps, cadence=cadence,
                temperature=temperature, type_map=type_map, mc_seed=native_mc_seed))
        if program is not None and cell_history is None:
            segments = tuple(program.segments)
            for name, values in (
                ("dice_temperature_start.dat", (item.start_temperature_k for item in segments)),
                ("dice_temperature_stop.dat", (item.stop_temperature_k for item in segments)),
                ("dice_temperature_steps.dat", (item.steps for item in segments)),
            ):
                (task_dir / name).write_text(
                    "".join(f"{value}\n" for value in values), encoding="utf-8"
                )
        if plumed is not None:
            text=Path(plumed).read_text(encoding="utf-8")
            if re.search(r"(?m)^\s*INCLUDE\b",text,re.I): raise ValueError("lifecycle LAMMPS PLUMED inputs must be self-contained")
            (task_dir/"plumed.dat").write_text(text,encoding="utf-8")

    def _validate_terminal_outputs(
        self,
        task_dir: Path,
        *,
        cadence: int,
        steps: int,
        require_deviation: bool,
        require_exact_cadence: bool = False,
        expected_atom_count: int | None = None,
    ) -> dict[str, tuple[int, ...]]:
        return validate_lammps_task_terminals(
            task_dir=task_dir,
            cadence=cadence,
            steps=steps,
            require_deviation=require_deviation,
            require_exact_cadence=require_exact_cadence,
            expected_atom_count=expected_atom_count,
        )

    @staticmethod
    def _validate_legacy_terminal_outputs(
        task_dir: Path,
        *,
        require_deviation: bool,
    ) -> None:
        required = [task_dir / "traj.lammpstrj"]
        if require_deviation:
            required.append(task_dir / "model_devi.out")
        for path in required:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size <= 0
            ):
                raise ValueError(
                    f"LAMMPS task terminal is missing or invalid: {path.name}"
                )

    @staticmethod
    def _publish_sampling_failure(
        prepared: LammpsTask,
        error: BaseException,
    ) -> int:
        task_dir = prepared.task_dir
        task = prepared.payload
        evidence = task_dir / "sampling_failure.json"
        if evidence.exists() or evidence.is_symlink():
            raise ValueError("LAMMPS sampling failure evidence already exists")
        trajectory = None
        deviation = None
        trajectory_name = "traj.lammpstrj"
        try:
            terminals = validate_lammps_task_terminals(
                task_dir=task_dir,
                cadence=int(task["cadence"]),
                steps=int(task["steps"]),
                require_deviation=int(task["model_count"]) > 1,
                require_exact_cadence=False,
                expected_atom_count=int(task["atom_count"]),
            )
        except ValueError:
            terminals = None
            if int(task["model_count"]) == 1:
                prefix = task_dir / "committed_prefix.lammpstrj"
                try:
                    _write_complete_lammpstrj_prefix(
                        task_dir / "traj.lammpstrj",
                        prefix,
                        expected_atom_count=int(task["atom_count"]),
                    )
                    terminals = validate_lammps_task_terminals(
                        task_dir=task_dir,
                        cadence=int(task["cadence"]),
                        steps=int(task["steps"]),
                        require_deviation=False,
                        require_exact_cadence=False,
                        expected_atom_count=int(task["atom_count"]),
                        trajectory_path=prefix,
                    )
                    trajectory_name = prefix.name
                except ValueError:
                    if prefix.exists():
                        prefix.unlink()
        if terminals is not None:
            trajectory = {
                "path": trajectory_name,
                "sha256": _file_digest(task_dir / trajectory_name),
                "timesteps": list(terminals["trajectory_timesteps"]),
            }
            deviation_path = task_dir / "model_devi.out"
            if int(task["model_count"]) > 1:
                deviation = {
                    "path": "model_devi.out",
                    "sha256": _file_digest(deviation_path),
                    "timesteps": list(terminals["deviation_timesteps"]),
                }
        evidence.write_text(
            json.dumps(
                {
                    "schema": "dice.lammps-sampling-failure",
                    "schema_revision": 1,
                    "task_id": task.get(
                        "task_id", f"task_{prepared.task_index:06d}"
                    ),
                    "task_index": prepared.task_index,
                    "system_id": str(
                        task.get("context_system_id", task["system_id"])
                    ),
                    "error_type": type(error).__name__,
                    "message": str(error)[-400:] or type(error).__name__,
                    "trajectory": trajectory,
                    "deviation": deviation,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return (
            0
            if terminals is None
            else len(terminals["trajectory_timesteps"])
        )

    @staticmethod
    def _publish_native_execution(
        task_dir: Path,
        task: dict,
        *,
        terminals: dict[str, tuple[int, ...]],
    ) -> None:
        evidence = task_dir / "native_execution.json"
        if evidence.exists() or evidence.is_symlink():
            raise ValueError("LAMMPS native execution evidence already exists")
        deviation = task_dir / "model_devi.out"
        execution_mode = task["execution_mode"]
        evidence.write_text(
            json.dumps(
                {
                    "schema": "dice.lammps-native-execution",
                    "schema_revision": 2 if execution_mode == "kokkos" else 1,
                    "command": list(command_for_mode(execution_mode)),
                    "sampler_role": task["sampler_role"],
                    "environment_profile_id": task["environment_profile_id"],
                    "deepmd_recipe_kind": task["deepmd_recipe_kind"],
                    **({"execution_mode": execution_mode} if execution_mode == "kokkos" else {}),
                    "pair_style": "deepmd/kk" if execution_mode == "kokkos" else "deepmd",
                    "ordered_models": list(task["ordered_models"]),
                    **(
                        {
                            "task_id": task["task_id"],
                            "task_index": task["task_index"],
                            "condition_group_id": task["condition_group_id"],
                            "sampling_role_id": task["sampling_role_id"],
                            "selection_group_id": task["selection_group_id"],
                            "system_id": task["context_system_id"],
                            "source_frame_index": task["source_frame_index"],
                            "selected_structure_id": task["selected_structure_id"],
                            "temperature_k": task["context_temperature_k"],
                            "pressure_bar": task["pressure_bar"],
                            "kind": task["kind"],
                            "steps": task["steps"],
                            "cadence": task["cadence"],
                            "native_seed": task["native_seed"],
                            "recipe_binding": task["recipe_binding"],
                            "recipe_sha256": task["recipe_sha256"],
                            "rendered_input_sha256": task["rendered_input_sha256"],
                            "setup_sha256": task["setup_sha256"],
                        }
                        if "condition_group_id" in task
                        else {}
                    ),
                    **(
                        {
                            "temperature_program": list(task["temperature_program"]),
                            "segment_boundaries": list(task["segment_boundaries"]),
                            "schedule_digests": [list(item) for item in task["schedule_digests"]],
                        }
                        if task.get("kind") in {"nvt_program", "nvt_cell_program", "npt_program"}
                        else {}
                    ),
                    **(
                        {
                            "resolved_cell_history": task["resolved_cell_history"],
                            "cell_history_sha256": task["cell_history_sha256"],
                            "expected_dump_steps": list(task["expected_dump_steps"]),
                        }
                        if task.get("kind") == "nvt_cell_program"
                        else {}
                    ),
                    "trajectory": {
                        "path": "traj.lammpstrj",
                        "sha256": _file_digest(task_dir / "traj.lammpstrj"),
                        "timesteps": list(terminals["trajectory_timesteps"]),
                    },
                    "deviation": (
                        None
                        if not deviation.is_file()
                        else {
                            "path": "model_devi.out",
                            "sha256": _file_digest(deviation),
                            "timesteps": list(
                                terminals["deviation_timesteps"]
                            ),
                        }
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def _cadence(self, recipe: Path) -> int:
        text = recipe.read_text(encoding="utf-8")
        declared = re.search(
            r"(?m)^\s*variable\s+dice_trajectory_every_steps\s+equal\s+([1-9]\d*)\s*(?:#.*)?$",
            text,
        )
        if declared:
            return int(declared.group(1))
        match=re.search(r"(?m)^\s*dump\s+\S+\s+\S+\s+\S+\s+(\d+)",text)
        return int(match.group(1)) if match else 20
    def _timestep(self, recipe: Path) -> float:
        match = re.search(
            r"(?m)^\s*timestep\s+([0-9.+\-eE]+)\s*(?:#.*)?$",
            recipe.read_text(encoding="utf-8"),
        )
        return float(match.group(1)) if match else 0.001
    def _seed_index(self, system: str, count: int) -> int:
        return selected_seed_index(
            run_json_path=self.run_json_path,
            iter_index=self.iter_index,
            round_index=self.round_index,
            system_id=system,
            seed_count=count,
        )
    def _task_seed(self, system: str) -> int:
        return task_seed(
            run_json_path=self.run_json_path,
            iter_index=self.iter_index,
            round_index=self.round_index,
            system_id=system,
        )
    def _identity(self, system: str, temp: float, pressure: float | None, step: int) -> str: return "sha256:"+hashlib.sha256(f"{self.run_json_path.resolve()}\0{self.iter_index}\0{self.round_index}\0{system}\0{temp}\0{pressure}\0{step}".encode()).hexdigest()


def _write_complete_lammpstrj_prefix(
    source: Path,
    target: Path,
    *,
    expected_atom_count: int,
) -> None:
    """Copy only complete leading dump frames, retaining the raw failed output."""

    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise ValueError("LAMMPS failed trajectory has no recoverable prefix")
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    cursor = 0
    committed_end = 0
    identity: tuple[tuple[int, int], ...] | None = None
    while cursor < len(lines):
        start = cursor
        if lines[cursor].strip() != "ITEM: TIMESTEP":
            raise ValueError("LAMMPS failed trajectory is malformed before its tail")
        if cursor + 4 >= len(lines):
            break
        try:
            step = int(lines[cursor + 1].strip())
            natoms_header = lines[cursor + 2].strip()
            natoms = int(lines[cursor + 3].strip())
        except ValueError as exc:
            raise ValueError("LAMMPS failed trajectory header is malformed") from exc
        if (
            step < 0
            or natoms_header != "ITEM: NUMBER OF ATOMS"
            or natoms != expected_atom_count
            or natoms <= 0
        ):
            raise ValueError("LAMMPS failed trajectory header is incompatible")
        end = start + 9 + natoms
        if end > len(lines):
            break
        if not lines[start + 4].startswith("ITEM: BOX BOUNDS"):
            raise ValueError("LAMMPS failed trajectory box header is malformed")
        try:
            for row in lines[start + 5 : start + 8]:
                bounds = row.split()
                if len(bounds) < 2 or not np.isfinite(
                    [float(bounds[0]), float(bounds[1])]
                ).all():
                    raise ValueError
        except ValueError as exc:
            raise ValueError("LAMMPS failed trajectory box is malformed") from exc
        columns = lines[start + 8].split()
        if columns[:2] != ["ITEM:", "ATOMS"]:
            raise ValueError("LAMMPS failed trajectory atom header is malformed")
        names = columns[2:]
        required = {"id", "type", "x", "y", "z"}
        if not required.issubset(names):
            raise ValueError("LAMMPS failed trajectory columns are incompatible")
        indices = {name: names.index(name) for name in required}
        current: list[tuple[int, int]] = []
        try:
            for row_text in lines[start + 9 : end]:
                row = row_text.split()
                if len(row) != len(names):
                    raise ValueError
                atom_id = int(row[indices["id"]])
                atom_type = int(row[indices["type"]])
                coordinates = [float(row[indices[name]]) for name in ("x", "y", "z")]
                if atom_id <= 0 or atom_type <= 0 or not np.isfinite(coordinates).all():
                    raise ValueError
                current.append((atom_id, atom_type))
        except (IndexError, ValueError) as exc:
            raise ValueError("LAMMPS failed trajectory atoms are malformed") from exc
        current_identity = tuple(current)
        if len(set(current_identity)) != natoms:
            raise ValueError("LAMMPS failed trajectory atom identity is malformed")
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("LAMMPS failed trajectory atom identity changed")
        committed_end = end
        cursor = end
    if committed_end <= 0:
        raise ValueError("LAMMPS failed trajectory has no complete frame prefix")
    if target.exists() or target.is_symlink():
        raise ValueError("LAMMPS committed trajectory prefix already exists")
    target.write_text("".join(lines[:committed_end]), encoding="utf-8")


def _lammps_seed(seed: Atoms) -> Atoms:
    """Give isolated lifecycle seed frames an explicit non-singular LAMMPS box."""
    item = seed.copy()
    if abs(float(item.get_cell().volume)) > 1e-12:
        return item
    positions = item.get_positions()
    span = positions.max(axis=0) - positions.min(axis=0) if len(item) else np.zeros(3)
    item.set_cell(np.diag(np.maximum(span + 10.0, 10.0)))
    item.set_pbc(False)
    return item


def _effective_plan(
    *,
    plan: SamplingPlan | ExplicitSamplingPlan | None,
    systems: dict[str, list[Atoms]],
    round_index: int,
    temperatures_k: Sequence[float] | None,
    ensemble: str | None,
    pressure_bar: float | None,
) -> SamplingPlan | ExplicitSamplingPlan:
    if plan is None:
        if ensemble not in {"nvt", "npt"} or not temperatures_k:
            raise ValueError("lifecycle LAMMPS sampling requires a resolved sampling plan")
        if ensemble == "npt" and pressure_bar is None:
            raise ValueError("lifecycle LAMMPS NPT sampling requires pressure")
        plan = SamplingPlan.create(tuple(
            SystemSamplingCondition(
                system_id,
                ensemble,
                tuple(float(value) for value in temperatures_k),
                () if ensemble == "nvt" else (float(pressure_bar),),
            )
            for system_id in systems
        ))
    plan_system_ids = (
        tuple(
            role_id
            for role_id, _group_id in plan.selection_plan_for_round(
                round_index
            ).role_to_group
        )
        if isinstance(plan, ExplicitSamplingPlan)
        else tuple(system.system_id for system in plan.systems)
    )
    if plan_system_ids != tuple(systems):
        raise ValueError("LAMMPS resolved sampling plan must exactly match configured system order")
    return plan


def _recipe_sources(
    *,
    recipe: str | Path | None,
    recipes: dict[str, str | Path] | None,
    group_recipes: dict[str, str | Path] | None,
    ensembles: tuple[str, ...],
    condition_group_ids: tuple[str, ...],
) -> dict[str, Path]:
    if condition_group_ids:
        if recipe is not None or recipes is not None or group_recipes is None:
            raise ValueError("explicit LAMMPS sampling requires group recipes only")
        missing = set(condition_group_ids) - set(group_recipes)
        if missing:
            raise ValueError("LAMMPS group recipes must cover active condition groups")
        return {
            group_id: Path(group_recipes[group_id]).resolve()
            for group_id in condition_group_ids
        }
    if group_recipes is not None:
        raise ValueError("legacy LAMMPS sampling does not accept group recipes")
    if (recipe is None) == (recipes is None):
        raise ValueError("LAMMPS sampling requires exactly one of recipe or recipes")
    if recipes is None:
        if len(ensembles) > 1:
            raise ValueError("heterogeneous LAMMPS sampling requires ensemble recipes")
        source = Path(recipe).resolve()
        return {ensembles[0]: source}
    if set(recipes) != set(ensembles):
        raise ValueError("LAMMPS recipe keys must exactly match effective ensembles")
    return {ensemble: Path(path).resolve() for ensemble, path in recipes.items()}


def _run_with_optional_progress(operation, task_dir: Path, task: dict, **kwargs):
    parameters = inspect.signature(operation).parameters
    for name in tuple(kwargs):
        if name not in parameters:
            kwargs.pop(name)
    return operation(task_dir, task, **kwargs)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_seed_selection_manifest(
    path: Path,
    *,
    iter_index: int,
    round_index: int,
    selections: tuple[SamplingSeedSelection, ...],
) -> None:
    payload = seed_selection_manifest_payload(
        iter_index=iter_index,
        round_index=round_index,
        selections=selections,
    )
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError("existing seed-frame selection manifest disagrees with resolved sampling")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("stale seed-frame selection manifest temporary path")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
