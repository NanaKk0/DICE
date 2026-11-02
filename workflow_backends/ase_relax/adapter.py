from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Mapping, Protocol

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import iread
from ase.optimize import FIRE

from workflow_backends.execution import run_operation
from workflow_backends.observation_source import ObservationLocator, ObservationRecord
from workflow_backends.sampling import (
    SamplingBackendBinding,
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingExecutionResult,
)
from workflow_core.common.sampling_plan import (
    relax_algorithm_identity,
    relax_constraint_identity,
    relax_parameters_payload,
)
from workflow_core.lifecycle.sampling import (
    SamplingEvidenceFailure,
    SamplingEvidenceFacts,
    SamplingEvidenceRebuildRequired,
    SamplingObservationResult,
    SamplingValidationResult,
)
from workflow_core.structures.canonical_identity import structure_id_for_atoms
from workflow_core.structures.canonical_identity import stamp_canonical_identity
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_runtime.lifecycle.progress import SamplingSnapshot, SamplingSystemProgress


_CONTEXT_SCHEMA = "dice.lifecycle.relax-task-context"
_RESULT_SCHEMA = "dice.lifecycle.relax-task-result"
_COMPLETION_RECEIPT_SCHEMA = "dice.lifecycle.relax-task-completion-receipt"
_SCHEMA_REVISION = 1


@dataclass(frozen=True, slots=True)
class _RelaxSnapshot:
    frame: Atoms
    energy_ev: float
    forces_ev_a: np.ndarray


class RelaxCalculatorProvider(Protocol):
    policy_identity: dict[str, object]

    def preflight(self, model_paths: tuple[Path, ...]) -> None: ...

    def create(self, model_path: Path) -> object: ...


class _RelaxObservationSource:
    def __init__(self, sample_dir: Path, rows: tuple[dict[str, object], ...]) -> None:
        self._sample_dir = sample_dir
        self._rows = rows
        self._records = tuple(self._record(row) for row in rows)
        identities = tuple(record.structure_id for record in self._records)
        if len(identities) != len(set(identities)):
            raise ValueError("relax observation identities must be unique")
        self._record_by_identity = dict(zip(identities, self._records, strict=True))
        self._ordinal_by_identity = {
            identity: ordinal for ordinal, identity in enumerate(identities)
        }

    @staticmethod
    def _record(row: dict[str, object]) -> ObservationRecord:
        task_index = int(row["task_index"])
        optimizer_step = int(row["optimizer_step"])
        locator = ObservationLocator(
            "ASE-RELAX",
            task_index,
            int(row["frame_index"]),
            optimizer_step,
            optimizer_step=optimizer_step,
        )
        return ObservationRecord(
            structure_id=str(row["structure_id"]),
            system_id=str(row["selection_group_id"]),
            task_index=task_index,
            frame_index=int(row["frame_index"]),
            md_step=optimizer_step,
            student_uncertainty=float(row["force_deviation"]),
            locator=locator,
            sampling_role_id=str(row["sampling_role_id"]),
            selection_group_id=str(row["selection_group_id"]),
            condition_group_id=str(row["condition_group_id"]),
            task_id=str(row["task_id"]),
            optimizer_step=optimizer_step,
        )

    def catalog(self) -> tuple[ObservationRecord, ...]:
        return self._records

    def replay(
        self,
        ordered_identities: Sequence[str],
        *,
        chunk_size: int,
    ) -> Iterator[tuple[Atoms, ...]]:
        size = int(chunk_size)
        identities = tuple(str(item) for item in ordered_identities)
        if size <= 0:
            raise ValueError("relax observation replay chunk_size must be positive")
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("relax replay identities must be unique and non-empty")
        missing = tuple(
            item for item in identities if item not in self._record_by_identity
        )
        if missing:
            raise ValueError(
                f"relax replay identity is absent from the catalog: {missing[0]}"
            )
        wanted = set(identities)
        loaded: dict[str, Atoms] = {}
        task_indices = sorted(
            {self._record_by_identity[item].task_index for item in identities}
        )
        for task_index in task_indices:
            path = (
                self._sample_dir
                / "tasks"
                / f"task_{task_index:06d}"
                / "observations.extxyz"
            )
            for frame in iread(path, index=":"):
                identity = str(frame.info.get("structure_id", ""))
                if identity in wanted:
                    loaded[identity] = frame
        if set(loaded) != wanted:
            raise ValueError("relax observation replay evidence is incomplete")
        for start in range(0, len(identities), size):
            yield tuple(loaded[item] for item in identities[start : start + size])


class AseRelaxSamplingNativeOps:
    """Execute and strictly replay fixed-cell ASE FIRE relaxation tasks."""

    def __init__(
        self,
        *,
        calculator_provider: RelaxCalculatorProvider,
        model_family: str,
        operation_role: str,
        operation_profile_id: str,
        task_id_overrides: Mapping[tuple[str, str, int], str] | None = None,
        allow_empty_observations: bool = False,
        max_workers: int = 1,
        retain_prefix: bool = False,
    ) -> None:
        self.calculator_provider = calculator_provider
        self.model_family = str(model_family)
        self.operation_role = str(operation_role)
        self.operation_profile_id = str(operation_profile_id)
        self.task_id_overrides = dict(task_id_overrides or {})
        self.allow_empty_observations = bool(allow_empty_observations)
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError("ASE relax max_workers must be a positive integer")
        self.max_workers = max_workers
        self.retain_prefix = retain_prefix
        if not self.model_family or not self.operation_role or not self.operation_profile_id:
            raise ValueError("ASE relax requires closed model-family and operation identity")
        self._calculator_cache = threading.local()

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        evidence = request.evidence
        tasks = self._tasks(evidence)
        models = tuple(Path(path).resolve() for path in evidence.models.paths)
        self.calculator_provider.preflight(models)
        systems = evidence.seed_mapping
        probe = systems[tasks[0].sampling_role_id][0]
        for model in models:
            frame = probe.copy()
            frame.calc = self._calculator(model)
            self._finite_evaluation(frame)
        task_root = evidence.sample_dir / "tasks"
        task_root.mkdir(parents=True, exist_ok=True)
        seed_inventory = tuple(
            stamp_canonical_identity(
                systems[role_id][0],
                system_id=role_id,
                structure_source="workflow",
            )
            for role_id, _selection_group in evidence.plan.selection_plan_for_round(
                evidence.round_index
            ).role_to_group
        )
        self._write_frames(evidence.sample_dir / "seeds.xyz", seed_inventory)

        def execute_task(task: object) -> None:
            task_index = int(str(task.task_id).removeprefix("task_"))
            frames = systems[task.sampling_role_id]
            if len(frames) != 1:
                raise ValueError("relax task requires exactly one selected seed frame")
            self.execute_individual(evidence, task)


        worker_count = min(self.max_workers, len(tasks))
        if worker_count == 1:
            for task in tasks:
                execute_task(task)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="dice-relax",
            ) as pool:
                futures = tuple(pool.submit(execute_task, task) for task in tasks)
                for future in futures:
                    future.result()
        source = self._open_source(evidence)
        if not source.catalog() and not self._allows_empty_observations(evidence):
            raise ValueError("ASE relax sample has no screen-visible observations")
        self._report_progress(request, tasks)
        return SamplingExecutionResult(evidence.sample_dir, len(tasks), source.failed_task_count)

    @staticmethod
    def _report_progress(
        request: SamplingExecutionRequest, tasks: tuple[object, ...]
    ) -> None:
        if request.progress_callback is None:
            return
        ordered_roles = tuple(
            dict.fromkeys(task.sampling_role_id for task in tasks)
        )
        request.progress_callback(
            SamplingSnapshot(
                backend="ASE",
                model_shape=(
                    "single" if len(request.evidence.models.paths) == 1 else "committee"
                ),
                completed_tasks=len(tasks),
                total_tasks=len(tasks),
                systems=len(ordered_roles),
                temperatures=0,
                ensemble="relax",
                steps_per_task=max(task.relax.max_steps for task in tasks),
                rows=tuple(
                    SamplingSystemProgress(
                        role_id,
                        sum(task.sampling_role_id == role_id for task in tasks),
                        sum(task.sampling_role_id == role_id for task in tasks),
                    )
                    for role_id in ordered_roles
                ),
            )
        )

    def validate_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingValidationResult:
        try:
            source = self._open_source(request)
            return SamplingValidationResult(
                bool(source.catalog()) or self._allows_empty_observations(request)
            )
        except (FileNotFoundError, OSError, ValueError):
            return SamplingValidationResult(False)

    def open_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        return SamplingObservationResult(self._open_source(request))

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        raise SamplingEvidenceRebuildRequired(
            "historical sample evidence has no relax interpretation; "
            f"rebuild from r{request.round_index}/sample",
            failure=SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE,
            round_index=request.round_index,
        )

    def observe_evidence(self, request: SamplingEvidenceRequest) -> SamplingEvidenceFacts:
        task_root = request.sample_dir / "tasks"
        tasks = self._tasks(request)
        return SamplingEvidenceFacts(
            seeds_present=all(
                (task_root / task.task_id / "raw_seed.extxyz").is_file()
                for task in tasks
            ),
            task_context_presence=tuple(
                (task_root / task.task_id / "task_context.json").is_file()
                for task in tasks
            ),
        )

    def _tasks(self, evidence: SamplingEvidenceRequest) -> tuple[object, ...]:
        if not hasattr(evidence.plan, "tasks_for_round"):
            raise ValueError("relax sampling requires canonical condition groups")
        tasks = tuple(
            task
            for task in evidence.plan.tasks_for_round(evidence.round_index)
            if task.kind == "relax"
        )
        if not tasks or any(task.relax is None for task in tasks):
            raise ValueError("ASE relax adapter requires closed relax tasks")
        if self.task_id_overrides:
            tasks = tuple(
                replace(
                    task,
                    task_id=self.task_id_overrides.get(
                        (
                            task.condition_group_id,
                            task.sampling_role_id,
                            task.source_frame_index,
                        ),
                        task.task_id,
                    ),
                )
                for task in tasks
            )
        return tasks

    def _allows_empty_observations(self, evidence: SamplingEvidenceRequest) -> bool:
        if self.allow_empty_observations:
            return True
        return any(
            task.kind != "relax"
            for task in evidence.plan.tasks_for_round(evidence.round_index)
        )

    def _execute_task(
        self,
        evidence: SamplingEvidenceRequest,
        *,
        task_index: int,
        task: object,
        seed: Atoms,
        models: tuple[Path, ...],
    ) -> None:
        parameters = task.relax
        self._validate_seed(seed, parameters.fixed_atoms)
        task_dir = evidence.sample_dir / "tasks" / task.task_id
        if task_dir.exists() or task_dir.is_symlink():
            raise ValueError(f"relax task destination already exists: {task_dir}")
        staging = evidence.sample_dir / "tasks" / f".{task.task_id}.staging"
        if staging.exists() or staging.is_symlink():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            presented_models = self._present_models(staging, models)
            raw_seed = seed.copy()
            raw_seed.set_constraint(FixAtoms(indices=parameters.fixed_atoms))
            atoms = raw_seed.copy()
            atoms.calc = self._calculator(models[0])
            if self.retain_prefix:
                self._write_frames(staging / "raw_seed.extxyz", (raw_seed,))
                self._write_json(staging / "task_context.json", self._context_payload(
                    evidence, task_index=task_index, task=task, seed=raw_seed, models=presented_models))
            initial_snapshot = self._snapshot(atoms)
            self._timing(evidence.sample_dir, task.task_id, 'started')
            history: list[tuple[int, _RelaxSnapshot]] = [(0, initial_snapshot)]
            fire_kwargs: dict[str, object] = {"logfile": str(staging / "optimizer.log")}
            if parameters.max_step_a is not None:
                fire_kwargs["maxstep"] = parameters.max_step_a
            optimizer = FIRE(atoms, **fire_kwargs)

            def commit(step, snapshot):
                from workflow_backends.ase_relax.prefix import commit_step
                commit_step(self, evidence, task, staging, step, snapshot, models)

            def capture() -> None:
                step = int(optimizer.nsteps)
                if step == history[-1][0]:
                    return
                history.append((step, self._snapshot(atoms)))
                if self.retain_prefix:
                    commit(step, history[-1][1])

            if self.retain_prefix:
                commit(0, initial_snapshot)
            optimizer.attach(capture, interval=1)
            converged = bool(
                optimizer.run(
                    fmax=parameters.fmax_ev_a,
                    steps=parameters.max_steps,
                )
            )
            optimizer_steps = int(optimizer.nsteps)
            if history[-1][0] != optimizer_steps:
                capture()
            if not converged and not parameters.accept_unconverged:
                raise ValueError(
                    "relaxation exhausted max_steps without satisfying fmax_ev_a"
                )
            termination = "converged" if converged else "max_steps"
            final_snapshot = history[-1][1]
            if parameters.include_history:
                visible = [
                    item
                    for item in history
                    if item[0] == 0 or item[0] % parameters.history_stride == 0
                ]
                if visible[-1][0] != optimizer_steps:
                    visible.append((optimizer_steps, final_snapshot))
            else:
                visible = [(optimizer_steps, final_snapshot)]
            self._write_frames(staging / "raw_seed.extxyz", (raw_seed,))
            if parameters.include_history:
                self._write_frames(
                    staging / "trajectory.extxyz",
                    tuple(snapshot.frame for _step, snapshot in history),
                )
            self._write_frames(
                staging / "final_structure.extxyz", (final_snapshot.frame,)
            )
            context = self._context_payload(
                evidence,
                task_index=task_index,
                task=task,
                seed=raw_seed,
                models=presented_models,
            )
            self._write_json(staging / "task_context.json", context)
            rows, frames = self._evaluate_observations(
                evidence,
                task_index=task_index,
                task=task,
                observations=tuple(visible),
                seed_id=str(context["seed_structure_id"]),
                models=models,
                relaxation_status=(converged, termination)
                if parameters.accept_unconverged
                else None,
            )
            self._write_frames(staging / "observations.extxyz", frames)
            required = [
                "task_context.json",
                "raw_seed.extxyz",
                "final_structure.extxyz",
                "observations.extxyz",
                "optimizer.log",
            ]
            if parameters.include_history:
                required.append("trajectory.extxyz")
            result = {
                "schema": _RESULT_SCHEMA,
                "schema_revision": _SCHEMA_REVISION,
                "task_id": task.task_id,
                "termination": termination,
                "converged": converged,
                "optimizer_steps": optimizer_steps,
                "initial_energy_ev": initial_snapshot.energy_ev,
                "final_energy_ev": final_snapshot.energy_ev,
                "final_max_force_ev_a": self._max_unconstrained_force(
                    final_snapshot.forces_ev_a,
                    parameters.fixed_atoms,
                ),
                "algorithm": relax_algorithm_identity(parameters),
                "constraint": relax_constraint_identity(parameters),
                "observations": rows,
                "files": {
                    name: self._digest(staging / name) for name in required
                },
            }
            self._write_json(staging / "relax_result.json", result)
            self._validate_task(staging, context, require_success=False)
            self._write_json(
                staging / "task_completion_receipt.json",
                {
                    "schema": _COMPLETION_RECEIPT_SCHEMA,
                    "schema_revision": _SCHEMA_REVISION,
                    "task_id": task.task_id,
                    "result_sha256": self._digest(staging / "relax_result.json"),
                },
            )
            os.replace(staging, task_dir)
            self._timing(evidence.sample_dir, task.task_id, 'finished')
        except BaseException:
            if not self.retain_prefix and staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise

    @staticmethod
    def _timing(root, task_id, event):
        directory = root / 'backend_logs'
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dict(task_id=task_id, event=event, time=time.monotonic(), pid=os.getpid())) + '\n'
        descriptor = os.open(directory / 'relax_timing.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, payload.encode())
        finally:
            os.close(descriptor)

    def execute_individual(self, evidence, task):
        from workflow_backends.common.task_outcomes import TaskOutcome, execute_task
        from workflow_backends.ase_relax.prefix import retain_failure
        models = tuple(Path(path).resolve() for path in evidence.models.paths)
        def run():
            self._execute_task(evidence, task_index=int(task.task_id.removeprefix("task_")),
                task=task, seed=evidence.seed_mapping[task.sampling_role_id][0], models=models)
            return TaskOutcome(task.task_id, "completed")
        if not self.retain_prefix:
            return run()
        return execute_task(task.task_id, run=run,
            retain_failure=lambda error: retain_failure(self, evidence, task, error))

    def _calculator(self, model_path: Path) -> object:
        cache = getattr(self._calculator_cache, "items", None)
        if cache is None:
            cache = {}
            self._calculator_cache.items = cache
        key = (str(model_path), self._digest(model_path))
        if key not in cache:
            cache[key] = self.calculator_provider.create(model_path)
        return cache[key]

    @staticmethod
    def _validate_seed(seed: Atoms, fixed_atoms: tuple[int, ...]) -> None:
        if not isinstance(seed, Atoms) or len(seed) <= 0:
            raise ValueError("relax seed must be one non-empty structure")
        if any(index >= len(seed) for index in fixed_atoms):
            raise ValueError("relax fixed atom index is outside the seed atom order")
        if not np.isfinite(seed.positions).all() or not np.isfinite(seed.cell.array).all():
            raise ValueError("relax seed geometry must be finite")
        if seed.constraints:
            if (
                len(seed.constraints) != 1
                or not isinstance(seed.constraints[0], FixAtoms)
                or tuple(int(index) for index in seed.constraints[0].get_indices())
                != fixed_atoms
            ):
                raise ValueError("relax seed constraints are incompatible with fixed_atoms")

    @staticmethod
    def _finite_evaluation(atoms: Atoms) -> tuple[float, np.ndarray]:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if (
            not math.isfinite(energy)
            or forces.shape != (len(atoms), 3)
            or not np.isfinite(forces).all()
        ):
            raise ValueError("relax calculator requires finite energy and atomic forces")
        return energy, forces

    def _snapshot(self, atoms: Atoms) -> _RelaxSnapshot:
        energy, forces = self._finite_evaluation(atoms)
        frame = atoms.copy()
        frame.calc = None
        if not np.isfinite(frame.positions).all():
            raise ValueError("relax optimizer produced non-finite positions")
        return _RelaxSnapshot(frame, energy, forces.copy())

    def _evaluate_observations(
        self,
        evidence: SamplingEvidenceRequest,
        *,
        task_index: int,
        task: object,
        observations: tuple[tuple[int, _RelaxSnapshot], ...],
        seed_id: str,
        models: tuple[Path, ...],
        relaxation_status: tuple[bool, str] | None = None,
    ) -> tuple[list[dict[str, object]], tuple[Atoms, ...]]:
        rows: list[dict[str, object]] = []
        frames: list[Atoms] = []
        committee = tuple(self._calculator(model) for model in models[1:])
        for frame_index, (optimizer_step, snapshot) in enumerate(observations):
            energies = [snapshot.energy_ev]
            forces = [snapshot.forces_ev_a]
            for calculator in committee:
                evaluated = snapshot.frame.copy()
                evaluated.calc = calculator
                energy, force = self._finite_evaluation(evaluated)
                energies.append(energy)
                forces.append(force)
            deviation = self._force_deviation(np.asarray(forces, dtype=float))
            identity = self._observation_identity(
                evidence,
                task_id=task.task_id,
                optimizer_step=optimizer_step,
                seed_id=seed_id,
            )
            frame = snapshot.frame.copy()
            frame.info.update(
                system_id=task.selection_group_id,
                structure_id=identity,
                structure_source="workflow",
                sampling_role_id=task.sampling_role_id,
                selection_group_id=task.selection_group_id,
                condition_group_id=task.condition_group_id,
                task_id=task.task_id,
                optimizer_step=optimizer_step,
                student_uncertainty=deviation,
                force_uncertainty=deviation,
                force_std_max=deviation,
            )
            if relaxation_status is not None:
                frame.info.update(
                    relax_converged=relaxation_status[0],
                    relax_termination=relaxation_status[1],
                )
            rows.append(
                {
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "sampling_role_id": task.sampling_role_id,
                    "selection_group_id": task.selection_group_id,
                    "condition_group_id": task.condition_group_id,
                    "source_frame_index": task.source_frame_index,
                    "frame_index": frame_index,
                    "optimizer_step": optimizer_step,
                    "structure_id": identity,
                    "member_energies_ev": energies,
                    "member_forces_ev_a": [force.tolist() for force in forces],
                    "force_deviation": deviation,
                    "policy_identity": dict(self.calculator_provider.policy_identity),
                }
            )
            frames.append(frame)
        return rows, tuple(frames)

    def _context_payload(
        self,
        evidence: SamplingEvidenceRequest,
        *,
        task_index: int,
        task: object,
        seed: Atoms,
        models: tuple[Path, ...],
    ) -> dict[str, object]:
        return {
            "schema": _CONTEXT_SCHEMA,
            "schema_revision": _SCHEMA_REVISION,
            "run_json_path": str(evidence.run_json_path.resolve()),
            "iter": evidence.iter_index,
            "round": evidence.round_index,
            "task_index": task_index,
            "task_id": task.task_id,
            "condition_group_id": task.condition_group_id,
            "sampling_role_id": task.sampling_role_id,
            "selection_group_id": task.selection_group_id,
            "source_frame_index": task.source_frame_index,
            "seed_structure_id": structure_id_for_atoms(seed),
            "parameters": relax_parameters_payload(task.relax),
            "algorithm": relax_algorithm_identity(task.relax),
            "constraint": relax_constraint_identity(task.relax),
            "model_family": self.model_family,
            "operation": {
                "role": self.operation_role,
                "profile_id": self.operation_profile_id,
            },
            "models": [
                {
                    "member_id": member_id,
                    "path": model.name,
                    "sha256": self._digest(model),
                }
                for member_id, model in zip(
                    evidence.models.member_ids, models, strict=True
                )
            ],
        }

    def _open_source(self, evidence: SamplingEvidenceRequest) -> _RelaxObservationSource:
        tasks = self._tasks(evidence)
        task_root = evidence.sample_dir / "tasks"
        if not task_root.is_dir():
            raise FileNotFoundError(task_root)
        systems = evidence.seed_mapping
        rows: list[dict[str, object]] = []
        failed = 0
        for task in tasks:
            task_dir = task_root / task.task_id
            if not task_dir.is_dir() or task_dir.is_symlink():
                raise ValueError("relax task inventory is missing or malformed")
            source_frames = systems[task.sampling_role_id]
            if len(source_frames) != 1:
                raise ValueError("relax task requires exactly one selected seed frame")
            seed = source_frames[0].copy()
            seed.set_constraint(FixAtoms(indices=task.relax.fixed_atoms))
            models = tuple(
                sorted(path for path in task_dir.iterdir() if path.name.startswith("model_"))
            )
            context = self._context_payload(
                evidence,
                task_index=int(str(task.task_id).removeprefix("task_")),
                task=task,
                seed=seed,
                models=models,
            )
            rows.extend(self._validate_task(task_dir, context, require_success=True))
            failed += self._read_json(task_dir / "relax_result.json").get("termination") == "failed"
        rows.sort(key=lambda row: (int(row["task_index"]), int(row["frame_index"])))
        source = _RelaxObservationSource(evidence.sample_dir, tuple(rows))
        source.failed_task_count = failed
        return source

    def _validate_task(
        self,
        task_dir: Path,
        expected_context: dict[str, object],
        *,
        require_success: bool,
    ) -> list[dict[str, object]]:
        context = self._read_json(task_dir / "task_context.json")
        result = self._read_json(task_dir / "relax_result.json")
        if result.get("schema_revision") == 2:
            from workflow_backends.ase_relax.prefix import validate_failure
            return validate_failure(self, task_dir, expected_context, result)
        success_path = task_dir / "task_completion_receipt.json"
        success = self._read_json(success_path) if success_path.is_file() else None
        if context != expected_context:
            raise ValueError("relax task context does not match the current task intent")
        if require_success and success is None:
            raise ValueError("relax task completion receipt is missing")
        converged = result.get("converged")
        termination = result.get("termination")
        parameters = expected_context["parameters"]
        accepted_termination = (
            converged is True and termination == "converged"
        ) or (
            parameters.get("accept_unconverged") is True
            and converged is False
            and termination == "max_steps"
        )
        if (
            result.get("schema") != _RESULT_SCHEMA
            or result.get("schema_revision") != _SCHEMA_REVISION
            or result.get("task_id") != expected_context["task_id"]
            or not accepted_termination
        ):
            raise ValueError("relax task result is malformed")
        files = result.get("files")
        include_history = bool(expected_context["parameters"]["include_history"])
        required = {
            "task_context.json",
            "raw_seed.extxyz",
            "final_structure.extxyz",
            "observations.extxyz",
            "optimizer.log",
        }
        if include_history:
            required.add("trajectory.extxyz")
        if not isinstance(files, dict) or set(files) != required:
            raise ValueError("relax task result file inventory is malformed")
        if any(
            not isinstance(digest, str) or self._digest(task_dir / name) != digest
            for name, digest in files.items()
        ):
            raise ValueError("relax task file digest mismatch")
        if success is not None and success != {
            "schema": _COMPLETION_RECEIPT_SCHEMA,
            "schema_revision": _SCHEMA_REVISION,
            "task_id": expected_context["task_id"],
            "result_sha256": self._digest(task_dir / "relax_result.json"),
        }:
            raise ValueError("relax task completion receipt is malformed")
        raw = read_extxyz_structures(task_dir / "raw_seed.extxyz", allow_empty=False)
        final = read_extxyz_structures(
            task_dir / "final_structure.extxyz", allow_empty=False
        )
        if len(raw) != 1 or len(final) != 1:
            raise ValueError("relax seed or final structure evidence is malformed")
        fixed_atoms = tuple(int(index) for index in parameters["fixed_atoms"])
        self._validate_frame_family(raw[0], final, fixed_atoms)
        observations = result.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError("relax task observations are malformed")
        frames = read_extxyz_structures(
            task_dir / "observations.extxyz", allow_empty=False
        )
        if len(frames) != len(observations):
            raise ValueError("relax observation evidence is incomplete")
        self._validate_frame_family(raw[0], frames, fixed_atoms)
        steps = [row.get("optimizer_step") for row in observations if isinstance(row, dict)]
        optimizer_steps = result.get("optimizer_steps")
        if (
            converged is False
            and int(optimizer_steps) != int(parameters["max_steps"])
        ):
            raise ValueError("relax max-step termination evidence drifted")
        stride = int(parameters["history_stride"])
        expected_steps = (
            [
                *range(0, int(optimizer_steps) + 1, stride),
                *([] if int(optimizer_steps) % stride == 0 else [int(optimizer_steps)]),
            ]
            if include_history
            else [int(optimizer_steps)]
        )
        if steps != expected_steps:
            raise ValueError("relax observation order evidence drifted")
        for row, frame in zip(observations, frames, strict=True):
            if (
                not isinstance(row, dict)
                or str(frame.info.get("structure_id", "")) != row.get("structure_id")
                or not math.isfinite(float(row.get("force_deviation", math.nan)))
            ):
                raise ValueError("relax observation identity or deviation is invalid")
        if include_history:
            trajectory = read_extxyz_structures(
                task_dir / "trajectory.extxyz", allow_empty=False
            )
            self._validate_frame_family(raw[0], trajectory, fixed_atoms)
            if len(trajectory) != int(optimizer_steps) + 1:
                raise ValueError("relax trajectory cadence evidence is incomplete")
        return observations

    @staticmethod
    def _validate_frame_family(
        seed: Atoms,
        frames: Sequence[Atoms],
        fixed_atoms: tuple[int, ...],
    ) -> None:
        symbols = seed.get_chemical_symbols()
        pbc = seed.pbc
        cell = seed.cell.array
        fixed_positions = seed.positions[list(fixed_atoms)]
        for frame in frames:
            if (
                frame.get_chemical_symbols() != symbols
                or not np.array_equal(frame.pbc, pbc)
                or not np.array_equal(frame.cell.array, cell)
                or not np.isfinite(frame.positions).all()
                or not np.isfinite(frame.cell.array).all()
            ):
                raise ValueError("relax atom or fixed-cell evidence drifted")
            indices = AseRelaxSamplingNativeOps._fixed_atom_indices(frame)
            if indices != fixed_atoms:
                raise ValueError("relax FixAtoms constraint evidence drifted")
            if fixed_atoms and not np.array_equal(
                frame.positions[list(fixed_atoms)], fixed_positions
            ):
                raise ValueError("relax fixed coordinate evidence drifted")

    @staticmethod
    def _fixed_atom_indices(frame: Atoms) -> tuple[int, ...]:
        if not frame.constraints:
            return ()
        if len(frame.constraints) != 1 or not isinstance(frame.constraints[0], FixAtoms):
            raise ValueError("relax constraint evidence is not one ASE FixAtoms mask")
        return tuple(int(index) for index in frame.constraints[0].get_indices())

    @staticmethod
    def _present_models(task_dir: Path, models: tuple[Path, ...]) -> tuple[Path, ...]:
        presented: list[Path] = []
        for index, source in enumerate(models):
            if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
                raise ValueError("relax model handle must be a non-empty regular file")
            suffix = "".join(source.suffixes) or ".model"
            target = task_dir / f"model_{index:03d}{suffix}"
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            presented.append(target)
        return tuple(presented)

    @staticmethod
    def _max_unconstrained_force(
        forces: np.ndarray, fixed_atoms: tuple[int, ...]
    ) -> float:
        if len(forces) == len(fixed_atoms):
            return 0.0
        mask = np.ones(len(forces), dtype=bool)
        if fixed_atoms:
            mask[list(fixed_atoms)] = False
        norms = np.linalg.norm(forces[mask], axis=1)
        return float(np.max(norms)) if norms.size else 0.0

    @staticmethod
    def _force_deviation(forces: np.ndarray) -> float:
        if forces.shape[0] == 1:
            return 0.0
        centered = forces - forces.mean(axis=0)
        per_atom = np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=0))
        return float(np.max(per_atom)) if per_atom.size else 0.0

    @staticmethod
    def _observation_identity(
        evidence: SamplingEvidenceRequest,
        *,
        task_id: str,
        optimizer_step: int,
        seed_id: str,
    ) -> str:
        payload = (
            f"{evidence.run_json_path.resolve()}\0{evidence.iter_index}\0"
            f"{evidence.round_index}\0{task_id}\0{optimizer_step}\0{seed_id}"
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _write_frames(path: Path, frames: Sequence[Atoms]) -> None:
        write_extxyz_structures(path, tuple(frame.copy() for frame in frames))

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"relax canonical evidence is unreadable: {path.name}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"relax canonical evidence is malformed: {path.name}")
        return payload

    @staticmethod
    def _digest(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"relax required regular file is missing: {path.name}")
        return hashlib.sha256(path.read_bytes()).hexdigest()


class _ValidationOnlyCalculatorProvider:
    policy_identity = {"name": "committee-force-deviation", "revision": 1}

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        del model_paths

    def create(self, model_path: Path) -> object:
        raise AssertionError(
            f"read-only relax validation attempted calculator creation for {model_path}"
        )


class SubprocessAseRelaxSamplingNativeOps:
    """Run ASE relaxation inside the selected student software profile."""

    def __init__(self, context: OperationContext, *, model_family: str,
                 persistent_without_mps: bool = False) -> None:
        expected_role = {
            "deepmd-tf": "student.deepmd.tf.relax",
            "deepmd-pt": "student.deepmd.pt.relax",
            "nep": "student.nep.relax",
        }.get(str(model_family))
        if expected_role is None or context.role != expected_role:
            raise ValueError("ASE relax operation context does not match its model family")
        self.context = context
        self.persistent_without_mps = persistent_without_mps
        self.model_family = str(model_family)
        self._validation = AseRelaxSamplingNativeOps(
            calculator_provider=_ValidationOnlyCalculatorProvider(),
            model_family=self.model_family,
            operation_role=context.role,
            operation_profile_id=context.profile_id,
        )

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        evidence = request.evidence
        tasks = self._validation._tasks(evidence)
        worker_root = evidence.sample_dir / ".relax_worker_inputs"
        if worker_root.exists() or worker_root.is_symlink():
            raise ValueError("ASE relax worker input destination is not fresh")
        worker_root.mkdir(parents=True)
        seed_paths: dict[str, Path] = {}
        for role_id, frames in evidence.seed_mapping.items():
            path = worker_root / f"{role_id}.extxyz"
            self._validation._write_frames(path, frames)
            seed_paths[role_id] = path
        role_to_group = dict(
            evidence.plan.selection_plan_for_round(evidence.round_index).role_to_group
        )
        manifest = {
            "schema": "dice.native.ase-relax-request",
            "schema_revision": 1,
            "run_json_path": str(evidence.run_json_path),
            "iter": evidence.iter_index,
            "round": evidence.round_index,
            "sample_dir": str(evidence.sample_dir),
            "model_family": self.model_family,
            "operation_role": self.context.role,
            "operation_profile_id": self.context.profile_id,
            "allow_empty_observations": any(
                task.kind != "relax"
                for task in evidence.plan.tasks_for_round(evidence.round_index)
            ),
            "max_workers": min(len(tasks), self.context.allocation.cpus_per_task, 4),
            "models": [
                {"member_id": member_id, "path": str(path)}
                for member_id, path in zip(
                    evidence.models.member_ids,
                    evidence.models.paths,
                    strict=True,
                )
            ],
            "seeds": [
                {
                    "sampling_role_id": role_id,
                    "selection_group_id": role_to_group[role_id],
                    "path": str(seed_paths[role_id]),
                }
                for role_id in role_to_group
            ],
            "groups": [
                {
                    "condition_group_id": task.condition_group_id,
                    "task_id": task.task_id,
                    "sampling_role_id": task.sampling_role_id,
                    "source_frame_index": task.source_frame_index,
                    **relax_parameters_payload(task.relax),
                }
                for task in tasks
            ],
        }
        manifest_path = evidence.sample_dir / ".relax_worker_request.json"
        self._validation._write_json(manifest_path, manifest)
        if self.context.allocation.per_gpu > 1 or self.persistent_without_mps:
            from workflow_backends.ase_relax.pool import execute_relax_pool
            from workflow_runtime.execution.allocation import validate_operation_allocation
            validate_operation_allocation(self.context.role, self.context.allocation)
            if self.context.allocation.cpus_per_task < self.context.allocation.per_gpu:
                raise ValueError("relax MPS requires at least one CPU per worker")
            self._validation._write_frames(evidence.sample_dir / "seeds.xyz", tuple(
                stamp_canonical_identity(evidence.seed_mapping[role][0],
                    system_id=role, structure_source="workflow") for role in role_to_group))
            from workflow_backends.ase_relax.prefix import retain_failure
            execute_relax_pool(self.context, manifest_path, tasks,
                retain_failure=lambda task, error: retain_failure(self._validation, evidence, task, error),
                managed_mps=not self.persistent_without_mps)
        else:
            run_operation(
                self.context,
                "python",
                (
                    "-m",
                    "workflow_backends.ase_relax.worker",
                    "--manifest",
                    str(manifest_path),
                ),
                cwd=evidence.sample_dir,
                diagnostic_name="ase_relax",
            )
        source = self._validation.open_current(evidence).source
        if not source.catalog() and not self._validation._allows_empty_observations(evidence):
            raise ValueError("ASE relax sample has no screen-visible observations")
        self._validation._report_progress(request, tasks)
        manifest_path.unlink()
        shutil.rmtree(worker_root)
        return SamplingExecutionResult(evidence.sample_dir, len(tasks), source.failed_task_count)

    def validate_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingValidationResult:
        return self._validation.validate_current(request)

    def open_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        return self._validation.open_current(request)

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        return self._validation.reconstruct_compatible_legacy(request)

    def observe_evidence(self, request: SamplingEvidenceRequest) -> SamplingEvidenceFacts:
        return self._validation.observe_evidence(request)


class AseRelaxSamplingBackendFactory:
    def __init__(self, context: OperationContext, *, model_family: str) -> None:
        self.context = context
        self.model_family = str(model_family)

    def bind(self, binding: SamplingBackendBinding) -> SubprocessAseRelaxSamplingNativeOps:
        del binding
        return SubprocessAseRelaxSamplingNativeOps(
            self.context, model_family=self.model_family
        )


__all__ = [
    "AseRelaxSamplingBackendFactory",
    "AseRelaxSamplingNativeOps",
    "RelaxCalculatorProvider",
    "SubprocessAseRelaxSamplingNativeOps",
]
