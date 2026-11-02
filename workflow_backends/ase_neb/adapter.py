from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Protocol
from dataclasses import dataclass, replace

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import iread
from ase.mep import DyNEB, NEB
from ase.optimize import FIRE

from workflow_backends.observation_source import (
    ObservationLocator,
    ObservationRecord,
)
from workflow_backends.sampling import (
    SamplingBackendBinding,
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingExecutionResult,
)
from workflow_backends.execution import run_operation
from workflow_runtime.lifecycle.environment import OperationContext
from workflow_core.common.sampling_plan import (
    neb_algorithm_identity,
    neb_constraint_identity,
    neb_parameters_payload,
)
from workflow_core.lifecycle.sampling import (
    SamplingEvidenceFacts,
    SamplingEvidenceRebuildRequired,
    SamplingEvidenceFailure,
    SamplingObservationResult,
    SamplingValidationResult,
)
from workflow_core.structures.canonical_identity import structure_id_for_atoms
from workflow_core.structures.canonical_identity import stamp_canonical_identity
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)


_CONTEXT_SCHEMA = "dice.lifecycle.neb-task-context"
_RESULT_SCHEMA = "dice.lifecycle.neb-task-result"
_COMPLETION_RECEIPT_SCHEMA = "dice.lifecycle.neb-task-completion-receipt"
_SCHEMA_REVISION = 3


@dataclass(frozen=True, slots=True)
class _NebDriverSnapshot:
    frame: Atoms
    energy_ev: float
    forces_ev_a: np.ndarray


class NebCalculatorProvider(Protocol):
    policy_identity: dict[str, object]

    def preflight(self, model_paths: tuple[Path, ...]) -> None: ...

    def create(self, model_path: Path) -> object: ...


class _NebObservationSource:
    def __init__(self, sample_dir: Path, rows: tuple[dict[str, object], ...]) -> None:
        self._sample_dir = sample_dir
        self._rows = rows
        self._records = tuple(self._record(row) for row in rows)
        identities = tuple(record.structure_id for record in self._records)
        if len(identities) != len(set(identities)):
            raise ValueError("NEB observation identities must be unique")
        self._record_by_identity = dict(zip(identities, self._records, strict=True))
        self._ordinal_by_identity = {
            identity: ordinal for ordinal, identity in enumerate(identities)
        }

    @staticmethod
    def _record(row: dict[str, object]) -> ObservationRecord:
        task_index = int(row["task_index"])
        optimizer_step = int(row["optimizer_step"])
        image_index = int(row["image_index"])
        locator = ObservationLocator(
            "ASE-NEB",
            task_index,
            int(row["frame_index"]),
            optimizer_step,
            optimizer_step=optimizer_step,
            image_index=image_index,
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
            image_index=image_index,
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
        if size <= 0:
            raise ValueError("NEB observation replay chunk_size must be positive")
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("NEB replay identities must be unique and non-empty")
        missing = [
            identity for identity in identities if identity not in self._record_by_identity
        ]
        if missing:
            raise ValueError(
                f"NEB replay identity is absent from the catalog: {missing[0]}"
            )
        ordinals = tuple(self._ordinal_by_identity[identity] for identity in identities)
        source_ordered = all(
            left < right for left, right in zip(ordinals, ordinals[1:])
        )
        scanned = self._scan_requested(identities)
        if source_ordered:
            chunk: list[Atoms] = []
            actual: list[str] = []
            for frame in scanned:
                actual.append(str(frame.info.get("structure_id", "")))
                chunk.append(frame)
                if len(chunk) == size:
                    yield tuple(chunk)
                    chunk.clear()
            if tuple(actual) != identities:
                raise ValueError("NEB observation replay evidence is incomplete")
            if chunk:
                yield tuple(chunk)
            return

        loaded = {
            str(frame.info.get("structure_id", "")): frame for frame in scanned
        }
        if set(loaded) != set(identities):
            raise ValueError("NEB observation replay evidence is incomplete")
        for start in range(0, len(identities), size):
            chunk_ids = identities[start : start + size]
            yield tuple(loaded[identity] for identity in chunk_ids)

    def _scan_requested(self, identities: tuple[str, ...]) -> Iterator[Atoms]:
        wanted = set(identities)
        task_indices = sorted(
            {
                self._record_by_identity[identity].task_index
                for identity in identities
            }
        )
        for task_index in task_indices:
            path = (
                self._sample_dir
                / "tasks"
                / f"task_{task_index:06d}"
                / "observations.extxyz"
            )
            for frame in iread(path, index=":"):
                if str(frame.info.get("structure_id", "")) in wanted:
                    yield frame


class AseNebSamplingNativeOps:
    """Execute and strictly replay closed ASE NEB tasks."""

    def __init__(
        self,
        *,
        calculator_provider: NebCalculatorProvider,
        model_family: str,
        operation_role: str,
        operation_profile_id: str,
        task_id_overrides: Mapping[str, str] | None = None,
        allow_empty_observations: bool = False,
        max_workers: int = 1,
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
            raise ValueError("ASE NEB max_workers must be a positive integer")
        self.max_workers = max_workers
        if not self.model_family or not self.operation_role or not self.operation_profile_id:
            raise ValueError("ASE NEB requires closed model-family and operation identity")

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        evidence = request.evidence
        tasks = self._tasks(evidence)
        models = tuple(Path(path).resolve() for path in evidence.models.paths)
        self.calculator_provider.preflight(models)
        systems = evidence.seed_mapping
        neb_roles = tuple(dict.fromkeys(task.sampling_role_id for task in tasks))
        probe = systems[neb_roles[0]][0]
        for model in models:
            frame = probe.copy()
            frame.calc = self.calculator_provider.create(model)
            energy = float(frame.get_potential_energy())
            forces = np.asarray(frame.get_forces(), dtype=float)
            if (
                not math.isfinite(energy)
                or forces.shape != (len(frame), 3)
                or not np.isfinite(forces).all()
            ):
                raise ValueError(
                    "NEB calculator preflight requires finite energy and atomic forces"
                )
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
            endpoints = systems[task.sampling_role_id]
            self._validate_endpoints(
                endpoints, fixed_atoms=task.neb.fixed_atoms
            )
            self._execute_task(
                evidence,
                task_index=task_index,
                task=task,
                endpoints=endpoints,
                models=models,
            )

        worker_count = min(self.max_workers, len(tasks))
        if worker_count == 1:
            for task in tasks:
                execute_task(task)
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="dice-neb",
            ) as pool:
                futures = tuple(pool.submit(execute_task, task) for task in tasks)
                for future in futures:
                    future.result()
        source = self._open_source(evidence)
        if not source.catalog() and not self._allows_empty_observations(evidence):
            raise ValueError("ASE NEB sample has no screen-visible observations")
        return SamplingExecutionResult(evidence.sample_dir, len(tasks))

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
            "historical sample evidence has no NEB interpretation; "
            f"rebuild from r{request.round_index}/sample",
            failure=SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE,
            round_index=request.round_index,
        )

    def observe_evidence(self, request: SamplingEvidenceRequest) -> SamplingEvidenceFacts:
        task_root = request.sample_dir / "tasks"
        tasks = self._tasks(request)
        return SamplingEvidenceFacts(
            seeds_present=all(
                (task_root / task.task_id / "endpoints.extxyz").is_file()
                for task in tasks
            ),
            task_context_presence=tuple(
                (task_root / task.task_id / "task_context.json").is_file()
                for task in tasks
            ),
        )

    def _tasks(self, evidence: SamplingEvidenceRequest) -> tuple[object, ...]:
        if not hasattr(evidence.plan, "tasks_for_round"):
            raise ValueError("NEB sampling requires canonical condition groups")
        tasks = tuple(
            task
            for task in evidence.plan.tasks_for_round(evidence.round_index)
            if task.kind == "neb"
        )
        if not tasks or any(task.neb is None for task in tasks):
            raise ValueError("ASE NEB adapter requires closed NEB tasks")
        if self.task_id_overrides:
            tasks = tuple(
                replace(
                    task,
                    task_id=self.task_id_overrides.get(
                        task.condition_group_id, task.task_id
                    ),
                )
                for task in tasks
            )
        return tasks

    def _allows_empty_observations(self, evidence: SamplingEvidenceRequest) -> bool:
        if self.allow_empty_observations:
            return True
        return any(
            task.kind != "neb"
            for task in evidence.plan.tasks_for_round(evidence.round_index)
        )

    @staticmethod
    def _validate_endpoints(
        endpoints: tuple[Atoms, ...], fixed_atoms: tuple[int, ...] = ()
    ) -> None:
        if len(endpoints) != 2:
            raise ValueError("NEB endpoint bundle must contain exactly two frames")
        initial, final = endpoints
        if len(initial) != len(final):
            raise ValueError("NEB endpoints must have identical atom counts")
        if initial.get_chemical_symbols() != final.get_chemical_symbols():
            raise ValueError("NEB endpoints must have identical element order")
        if not np.array_equal(initial.pbc, final.pbc):
            raise ValueError("NEB endpoints must have identical PBC")
        if not np.array_equal(initial.cell.array, final.cell.array):
            raise ValueError("variable-cell NEB is unsupported")
        if not all(
            np.isfinite(values).all()
            for values in (
                initial.positions,
                final.positions,
                initial.cell.array,
                final.cell.array,
            )
        ):
            raise ValueError("NEB endpoint geometry must be finite")
        if any(index >= len(initial) for index in fixed_atoms):
            raise ValueError("NEB fixed atom index is outside the endpoint atom order")
        constraint_masks: list[tuple[int, ...] | None] = []
        for endpoint in endpoints:
            if not endpoint.constraints:
                constraint_masks.append(None)
                continue
            if (
                len(endpoint.constraints) != 1
                or not isinstance(endpoint.constraints[0], FixAtoms)
            ):
                raise ValueError(
                    "NEB endpoint constraints are incompatible with fixed_atoms"
                )
            constraint_masks.append(
                tuple(int(index) for index in endpoint.constraints[0].get_indices())
            )
        if any(mask is not None for mask in constraint_masks) and any(
            mask != fixed_atoms for mask in constraint_masks
        ):
            raise ValueError(
                "NEB endpoint constraints are incompatible with fixed_atoms"
            )
        if fixed_atoms and not np.array_equal(
            initial.positions[list(fixed_atoms)],
            final.positions[list(fixed_atoms)],
        ):
            raise ValueError("NEB fixed endpoint coordinates must be identical")

    def _execute_task(
        self,
        evidence: SamplingEvidenceRequest,
        *,
        task_index: int,
        task: object,
        endpoints: tuple[Atoms, ...],
        models: tuple[Path, ...],
    ) -> None:
        task_dir = evidence.sample_dir / "tasks" / task.task_id
        if task_dir.exists():
            raise ValueError(f"NEB task destination already exists: {task_dir}")
        staging = evidence.sample_dir / "tasks" / f".{task.task_id}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            presented_models = self._present_models(staging, models)
            raw_endpoints = tuple(frame.copy() for frame in endpoints)
            for endpoint in raw_endpoints:
                endpoint.set_constraint(FixAtoms(indices=task.neb.fixed_atoms))
            endpoint_rows: list[dict[str, object]] = []
            if task.neb.relax:
                authoritative_endpoints, endpoint_rows = self._relax_endpoints(
                    staging,
                    task=task,
                    endpoints=raw_endpoints,
                    driver_model=presented_models[0],
                )
            else:
                authoritative_endpoints = tuple(
                    endpoint.copy() for endpoint in raw_endpoints
                )
            initial, final = (
                endpoint.copy() for endpoint in authoritative_endpoints
            )
            images = [initial]
            images.extend(
                endpoints[0].copy() for _ in range(task.neb.images)
            )
            images.append(final)
            for image in images:
                image.set_constraint(FixAtoms(indices=task.neb.fixed_atoms))
                image.calc = self.calculator_provider.create(
                    presented_models[0]
                )
            algorithm = neb_algorithm_identity(task.neb)
            constraint = neb_constraint_identity(task.neb)
            if task.neb.relax:
                self._write_frames(
                    staging / "raw_endpoints.extxyz", raw_endpoints
                )
            self._write_frames(
                staging / "endpoints.extxyz", authoritative_endpoints
            )
            endpoint_preparation = None
            if task.neb.relax:
                endpoint_preparation = self._endpoint_preparation_payload(
                    staging,
                    evidence=evidence,
                    task=task,
                    raw_endpoints=raw_endpoints,
                    optimized_endpoints=authoritative_endpoints,
                    endpoint_rows=endpoint_rows,
                )
                self._write_json(
                    staging / "endpoint_relaxation.json",
                    endpoint_preparation,
                )
            band_kwargs = {
                "k": float(algorithm["spring_ev_a2"]),
                "climb": (
                    False
                    if task.neb.climb_preconverge_fmax_ev_a is not None
                    else task.neb.climb
                ),
                "parallel": False,
                "method": str(algorithm["method"]),
                "allow_shared_calculator": False,
            }
            if task.neb.method == "dyneb":
                band = DyNEB(
                    images,
                    fmax=task.neb.fmax_ev_a,
                    dynamic_relaxation=True,
                    scale_fmax=0.0,
                    **band_kwargs,
                )
            else:
                band = NEB(images, **band_kwargs)
            band.interpolate(
                method=task.neb.init,
                mic=task.neb.init == "idpp",
                apply_constraint=True,
            )
            initial_interior = tuple(image.copy() for image in images[1:-1])

            def validated_driver_band_state() -> tuple[_NebDriverSnapshot, ...]:
                snapshots: list[_NebDriverSnapshot] = []
                for image_index, image in enumerate(images):
                    energy = float(image.get_potential_energy())
                    forces = np.asarray(image.get_forces(), dtype=float)
                    if (
                        not math.isfinite(energy)
                        or forces.shape != (len(image), 3)
                        or not np.isfinite(forces).all()
                    ):
                        raise ValueError(
                            "NEB driver model produced non-finite energy or forces"
                        )
                    snapshots.append(
                        _NebDriverSnapshot(
                            frame=image.copy(),
                            energy_ev=energy,
                            forces_ev_a=forces.copy(),
                        )
                    )
                return tuple(snapshots)

            latest_driver_state = validated_driver_band_state()[1:-1]
            history: list[tuple[int, tuple[_NebDriverSnapshot, ...]]] = [
                (0, latest_driver_state)
            ]
            optimizer: Any = None
            step_offset = 0

            def refresh_driver_state() -> None:
                nonlocal latest_driver_state
                latest_driver_state = validated_driver_band_state()[1:-1]

            def capture_completed_step() -> None:
                step = step_offset + int(optimizer.nsteps)
                if step == history[-1][0]:
                    return
                history.append(
                    (
                        step,
                        latest_driver_state,
                    )
                )

            optimization_stages: list[dict[str, object]] = []

            def run_stage(
                *,
                name: str,
                climb: bool,
                fmax_ev_a: float,
                step_budget: int,
                logfile: str,
            ) -> bool:
                nonlocal optimizer, step_offset
                band.climb = climb
                optimizer = FIRE(band, logfile=str(staging / logfile))
                optimizer.attach(refresh_driver_state, interval=1)
                if task.neb.include_history:
                    optimizer.attach(capture_completed_step, interval=1)
                stage_converged = bool(
                    optimizer.run(fmax=fmax_ev_a, steps=step_budget)
                )
                stage_steps = int(optimizer.nsteps)
                optimization_stages.append(
                    {
                        "name": name,
                        "climb": climb,
                        "fmax_ev_a": fmax_ev_a,
                        "step_budget": step_budget,
                        "optimizer_steps": stage_steps,
                        "converged": stage_converged,
                    }
                )
                step_offset += stage_steps
                return stage_converged

            preclimb_fmax = task.neb.climb_preconverge_fmax_ev_a
            preclimb_converged = True
            if preclimb_fmax is not None:
                preclimb_converged = run_stage(
                    name="preclimb",
                    climb=False,
                    fmax_ev_a=preclimb_fmax,
                    step_budget=task.neb.max_steps,
                    logfile="optimizer_preclimb.log",
                )
                remaining_steps = task.neb.max_steps - step_offset
                if preclimb_converged and remaining_steps > 0:
                    converged = run_stage(
                        name="climb",
                        climb=True,
                        fmax_ev_a=task.neb.fmax_ev_a,
                        step_budget=remaining_steps,
                        logfile="optimizer.log",
                    )
                else:
                    converged = False
            else:
                converged = run_stage(
                    name="climb" if task.neb.climb else "neb",
                    climb=task.neb.climb,
                    fmax_ev_a=task.neb.fmax_ev_a,
                    step_budget=task.neb.max_steps,
                    logfile="optimizer.log",
                )
            optimizer_steps = step_offset
            final_interior = tuple(image.copy() for image in images[1:-1])
            final_driver_band_state = validated_driver_band_state()
            if not converged and not task.neb.include_history:
                observations: tuple[
                    tuple[int, int, _NebDriverSnapshot], ...
                ] = ()
            elif task.neb.include_history:
                if history[-1][0] != optimizer_steps:
                    raise ValueError(
                        "NEB optimizer history is incomplete at its terminal step"
                    )
                observations = tuple(
                    (optimizer_step, image_index, snapshot)
                    for optimizer_step, interior in history
                    for image_index, snapshot in enumerate(interior, start=1)
                )
            else:
                observations = tuple(
                    (optimizer_steps, image_index, snapshot)
                    for image_index, snapshot in enumerate(
                        final_driver_band_state
                    )
                )
            endpoint_ids = tuple(
                structure_id_for_atoms(frame) for frame in authoritative_endpoints
            )
            context = self._context_payload(
                evidence,
                task_index=task_index,
                task=task,
                endpoint_ids=endpoint_ids,
                models=presented_models,
                endpoint_preparation=endpoint_preparation,
            )
            self._write_json(staging / "task_context.json", context)
            self._write_frames(staging / "initial_band.extxyz", initial_interior)
            self._write_frames(staging / "final_band.extxyz", final_interior)
            rows, frames = self._evaluate_observations(
                evidence,
                task_index=task_index,
                task=task,
                observations=observations,
                endpoint_ids=endpoint_ids,
                models=presented_models,
            )
            self._write_frames(staging / "observations.extxyz", frames)
            result = {
                "schema": _RESULT_SCHEMA,
                "schema_revision": _SCHEMA_REVISION,
                "task_id": task.task_id,
                "converged": converged,
                "termination": (
                    "converged"
                    if converged
                    else "max_steps"
                ),
                "optimizer_steps": optimizer_steps,
                "algorithm": algorithm,
                "constraint": constraint,
                "observations": rows,
                "files": {
                    name: self._digest(staging / name)
                    for name in (
                        "task_context.json",
                        "endpoints.extxyz",
                        "initial_band.extxyz",
                        "final_band.extxyz",
                        "observations.extxyz",
                        *(
                            (
                                "raw_endpoints.extxyz",
                                "endpoint_relaxation.json",
                            )
                            if task.neb.relax
                            else ()
                        ),
                    )
                },
            }
            if preclimb_fmax is not None:
                result["optimization_stages"] = optimization_stages
            if endpoint_preparation is not None:
                result["endpoint_preparation"] = endpoint_preparation
            self._write_json(staging / "neb_result.json", result)
            self._validate_task(staging, context, require_success=False)
            self._write_json(
                staging / "task_completion_receipt.json",
                {
                    "schema": _COMPLETION_RECEIPT_SCHEMA,
                    "schema_revision": _SCHEMA_REVISION,
                    "task_id": task.task_id,
                    "result_sha256": self._digest(staging / "neb_result.json"),
                },
            )
            os.replace(staging, task_dir)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    @staticmethod
    def _present_models(task_dir: Path, models: tuple[Path, ...]) -> tuple[Path, ...]:
        presented: list[Path] = []
        for index, source in enumerate(models):
            if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
                raise ValueError("NEB model handle must be a non-empty regular file")
            suffix = "".join(source.suffixes) or ".model"
            target = task_dir / f"model_{index:03d}{suffix}"
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            presented.append(target)
        return tuple(presented)

    def _context_payload(
        self,
        evidence: SamplingEvidenceRequest,
        *,
        task_index: int,
        task: object,
        endpoint_ids: tuple[str, str],
        models: tuple[Path, ...],
        endpoint_preparation: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = {
            "schema": _CONTEXT_SCHEMA,
            "schema_revision": _SCHEMA_REVISION,
            "iter": evidence.iter_index,
            "round": evidence.round_index,
            "task_index": task_index,
            "task_id": task.task_id,
            "condition_group_id": task.condition_group_id,
            "sampling_role_id": task.sampling_role_id,
            "selection_group_id": task.selection_group_id,
            "endpoint_structure_ids": list(endpoint_ids),
            "parameters": neb_parameters_payload(task.neb),
            "algorithm": neb_algorithm_identity(task.neb),
            "constraint": neb_constraint_identity(task.neb),
            "models": [
                {
                    "member_id": member_id,
                    "presentation": model.name,
                    "sha256": self._digest(model),
                }
                for member_id, model in zip(
                    evidence.models.member_ids, models, strict=True
                )
            ],
            "operation": {
                "role": self.operation_role,
                "profile_id": self.operation_profile_id,
                "model_family": self.model_family,
            },
        }
        if endpoint_preparation is not None:
            payload["endpoint_preparation"] = endpoint_preparation
        return payload

    def _relax_endpoints(
        self,
        staging: Path,
        *,
        task: object,
        endpoints: tuple[Atoms, ...],
        driver_model: Path,
    ) -> tuple[tuple[Atoms, ...], list[dict[str, object]]]:
        optimized: list[Atoms] = []
        rows: list[dict[str, object]] = []
        for role, source in zip(("initial", "final"), endpoints, strict=True):
            endpoint = source.copy()
            endpoint.set_constraint(FixAtoms(indices=task.neb.fixed_atoms))
            endpoint.calc = self.calculator_provider.create(driver_model)
            symbols = endpoint.get_chemical_symbols()
            pbc = endpoint.pbc.copy()
            cell = endpoint.cell.array.copy()
            fixed_positions = endpoint.positions[list(task.neb.fixed_atoms)].copy()

            def validate_state() -> None:
                energy = float(endpoint.get_potential_energy())
                forces = np.asarray(endpoint.get_forces(), dtype=float)
                if (
                    endpoint.get_chemical_symbols() != symbols
                    or not np.array_equal(endpoint.pbc, pbc)
                    or not np.array_equal(endpoint.cell.array, cell)
                    or not np.isfinite(endpoint.positions).all()
                    or not math.isfinite(energy)
                    or forces.shape != (len(endpoint), 3)
                    or not np.isfinite(forces).all()
                    or self._fixed_atom_indices(endpoint)
                    != tuple(task.neb.fixed_atoms)
                    or (
                        bool(task.neb.fixed_atoms)
                        and not np.array_equal(
                            endpoint.positions[list(task.neb.fixed_atoms)],
                            fixed_positions,
                        )
                    )
                ):
                    raise ValueError(
                        f"NEB {role} endpoint relaxation produced atom/cell/constraint drift or non-finite evidence"
                    )

            validate_state()
            optimizer = FIRE(
                endpoint,
                logfile=str(staging / f"endpoint_{role}_optimizer.log"),
            )
            optimizer.attach(validate_state, interval=1)
            converged = bool(
                optimizer.run(
                    fmax=task.neb.fmax_ev_a,
                    steps=task.neb.max_steps,
                )
            )
            validate_state()
            if not converged:
                raise ValueError(
                    f"NEB {role} endpoint relaxation exhausted max_steps without convergence"
                )
            frame = endpoint.copy()
            frame.calc = None
            optimized.append(frame)
            rows.append(
                {
                    "role": role,
                    "converged": True,
                    "optimizer_steps": int(optimizer.nsteps),
                    "raw_structure_id": structure_id_for_atoms(source),
                    "optimized_structure_id": structure_id_for_atoms(frame),
                }
            )
        return tuple(optimized), rows

    def _endpoint_preparation_payload(
        self,
        staging: Path,
        *,
        evidence: SamplingEvidenceRequest,
        task: object,
        raw_endpoints: tuple[Atoms, ...],
        optimized_endpoints: tuple[Atoms, ...],
        endpoint_rows: list[dict[str, object]],
    ) -> dict[str, object]:
        raw_ids = [structure_id_for_atoms(frame) for frame in raw_endpoints]
        optimized_ids = [
            structure_id_for_atoms(frame) for frame in optimized_endpoints
        ]
        if [row["raw_structure_id"] for row in endpoint_rows] != raw_ids or [
            row["optimized_structure_id"] for row in endpoint_rows
        ] != optimized_ids:
            raise ValueError("NEB endpoint relaxation identity evidence is inconsistent")
        return {
            "schema": "dice.lifecycle.neb-endpoint-relaxation",
            "schema_revision": 1,
            "algorithm": {
                "name": "ase-endpoint-fire",
                "revision": 1,
                "optimizer": "FIRE",
                "fixed_cell": True,
            },
            "driver_member_id": evidence.models.member_ids[0],
            "fmax_ev_a": task.neb.fmax_ev_a,
            "max_steps": task.neb.max_steps,
            "constraint": neb_constraint_identity(task.neb),
            "raw_structure_ids": raw_ids,
            "optimized_structure_ids": optimized_ids,
            "raw_endpoints_sha256": self._digest(
                staging / "raw_endpoints.extxyz"
            ),
            "optimized_endpoints_sha256": self._digest(
                staging / "endpoints.extxyz"
            ),
            "endpoints": endpoint_rows,
        }

    def _evaluate_observations(
        self,
        evidence: SamplingEvidenceRequest,
        *,
        task_index: int,
        task: object,
        observations: tuple[tuple[int, int, _NebDriverSnapshot], ...],
        endpoint_ids: tuple[str, str],
        models: tuple[Path, ...],
    ) -> tuple[list[dict[str, object]], tuple[Atoms, ...]]:
        rows: list[dict[str, object]] = []
        frames: list[Atoms] = []
        endpoint_provenance = "\0".join(endpoint_ids)
        committee_calculators = tuple(
            self.calculator_provider.create(model) for model in models[1:]
        )
        for frame_index, (optimizer_step, image_index, snapshot) in enumerate(
            observations
        ):
            source = snapshot.frame
            energies = [snapshot.energy_ev]
            forces = [snapshot.forces_ev_a]
            for calculator in committee_calculators:
                frame = source.copy()
                frame.calc = calculator
                energy = float(frame.get_potential_energy())
                force = np.asarray(frame.get_forces(), dtype=float)
                if (
                    not math.isfinite(energy)
                    or force.shape != (len(frame), 3)
                    or not np.isfinite(force).all()
                ):
                    raise ValueError("NEB observation model evidence must be finite")
                energies.append(energy)
                forces.append(force)
            deviation = self._force_deviation(np.asarray(forces, dtype=float))
            identity = self._observation_identity(
                evidence,
                task_id=task.task_id,
                optimizer_step=optimizer_step,
                image_index=image_index,
                endpoint_provenance=endpoint_provenance,
            )
            frame = source.copy()
            frame.calc = None
            frame.info.update(
                system_id=task.selection_group_id,
                structure_id=identity,
                structure_source="workflow",
                sampling_role_id=task.sampling_role_id,
                selection_group_id=task.selection_group_id,
                condition_group_id=task.condition_group_id,
                task_id=task.task_id,
                optimizer_step=optimizer_step,
                image_index=image_index,
                student_uncertainty=deviation,
                force_uncertainty=deviation,
                force_std_max=deviation,
            )
            rows.append(
                {
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "sampling_role_id": task.sampling_role_id,
                    "selection_group_id": task.selection_group_id,
                    "condition_group_id": task.condition_group_id,
                    "frame_index": frame_index,
                    "optimizer_step": optimizer_step,
                    "image_index": image_index,
                    "structure_id": identity,
                    "member_energies_ev": energies,
                    "member_forces_ev_a": [force.tolist() for force in forces],
                    "force_deviation": deviation,
                    "policy_identity": dict(
                        self.calculator_provider.policy_identity
                    ),
                }
            )
            frames.append(frame)
        return rows, tuple(frames)

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
        image_index: int,
        endpoint_provenance: str,
    ) -> str:
        payload = (
            f"{evidence.run_json_path}\0{evidence.iter_index}\0{evidence.round_index}"
            f"\0{task_id}\0{optimizer_step}\0{image_index}\0{endpoint_provenance}"
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def _open_source(self, evidence: SamplingEvidenceRequest) -> _NebObservationSource:
        tasks = self._tasks(evidence)
        task_root = evidence.sample_dir / "tasks"
        if not task_root.is_dir():
            raise FileNotFoundError(task_root)
        expected_dirs = tuple(task_root / task.task_id for task in tasks)
        actual_dirs = tuple(sorted(path for path in task_root.iterdir() if path.is_dir()))
        known_names = {
            task.task_id
            for task in evidence.plan.tasks_for_round(evidence.round_index)
        }
        if any(path.name not in known_names or path.is_symlink() for path in actual_dirs) or any(
            task_dir not in actual_dirs for task_dir in expected_dirs
        ):
            raise ValueError("NEB task inventory is missing, reordered, or ambiguous")
        rows: list[dict[str, object]] = []
        systems = evidence.seed_mapping
        for task, task_dir in zip(tasks, expected_dirs, strict=True):
            task_index = int(str(task.task_id).removeprefix("task_"))
            raw_endpoints = systems[task.sampling_role_id]
            endpoint_preparation = None
            if task.neb.relax:
                endpoint_preparation = self._validate_endpoint_preparation(
                    task_dir,
                    evidence=evidence,
                    task=task,
                    expected_raw_endpoints=raw_endpoints,
                )
                endpoint_ids = tuple(
                    endpoint_preparation["optimized_structure_ids"]
                )
            else:
                endpoint_ids = tuple(
                    structure_id_for_atoms(frame) for frame in raw_endpoints
                )
            models = tuple(
                sorted(path for path in task_dir.iterdir() if path.name.startswith("model_"))
            )
            context = self._context_payload(
                evidence,
                task_index=task_index,
                task=task,
                endpoint_ids=endpoint_ids,
                models=models,
                endpoint_preparation=endpoint_preparation,
            )
            rows.extend(self._validate_task(task_dir, context, require_success=True))
        return _NebObservationSource(evidence.sample_dir, tuple(rows))

    def _validate_task(
        self,
        task_dir: Path,
        expected_context: dict[str, object],
        *,
        require_success: bool,
    ) -> list[dict[str, object]]:
        context = self._read_json(task_dir / "task_context.json")
        if context != expected_context:
            raise ValueError("NEB task context does not match the current task intent")
        result = self._read_json(task_dir / "neb_result.json")
        success = self._read_json(task_dir / "task_completion_receipt.json") if (task_dir / "task_completion_receipt.json").is_file() else None
        if require_success and success is None:
            raise ValueError("NEB task completion receipt is missing")
        if (
            result.get("schema") != _RESULT_SCHEMA
            or result.get("schema_revision") != _SCHEMA_REVISION
            or result.get("task_id") != expected_context["task_id"]
            or result.get("algorithm") != expected_context["algorithm"]
            or result.get("constraint") != expected_context["constraint"]
            or not isinstance(result.get("converged"), bool)
            or result.get("termination")
            != ("converged" if result.get("converged") else "max_steps")
            or not isinstance(result.get("optimizer_steps"), int)
            or result["optimizer_steps"] < 0
            or result["optimizer_steps"]
            > int(expected_context["parameters"]["max_steps"])
        ):
            raise ValueError("NEB task result is malformed")
        required_files = {
            "task_context.json",
            "endpoints.extxyz",
            "initial_band.extxyz",
            "final_band.extxyz",
            "observations.extxyz",
        }
        if "endpoint_preparation" in expected_context:
            required_files.update(
                {"raw_endpoints.extxyz", "endpoint_relaxation.json"}
            )
            if result.get("endpoint_preparation") != expected_context[
                "endpoint_preparation"
            ]:
                raise ValueError("NEB endpoint preparation result evidence drifted")
        elif "endpoint_preparation" in result:
            raise ValueError("disabled NEB endpoint preparation evidence is present")
        files = result.get("files")
        if not isinstance(files, dict) or set(files) != required_files:
            raise ValueError("NEB task result file inventory is malformed")
        if any(self._digest(task_dir / name) != digest for name, digest in files.items()):
            raise ValueError("NEB task file digest mismatch")
        if success is not None and success != {
            "schema": _COMPLETION_RECEIPT_SCHEMA,
            "schema_revision": _SCHEMA_REVISION,
            "task_id": expected_context["task_id"],
            "result_sha256": self._digest(task_dir / "neb_result.json"),
        }:
            raise ValueError("NEB task completion receipt is malformed")
        parameters = expected_context["parameters"]
        image_count = int(parameters["images"])
        fixed_atoms = tuple(int(index) for index in parameters["fixed_atoms"])
        endpoints = read_extxyz_structures(
            task_dir / "endpoints.extxyz", allow_empty=False
        )
        initial_band = read_extxyz_structures(
            task_dir / "initial_band.extxyz", allow_empty=False
        )
        final_band = read_extxyz_structures(
            task_dir / "final_band.extxyz", allow_empty=False
        )
        if (
            len(endpoints) != 2
            or len(initial_band) != image_count
            or len(final_band) != image_count
            or [structure_id_for_atoms(frame) for frame in endpoints]
            != expected_context["endpoint_structure_ids"]
        ):
            raise ValueError("NEB endpoint or band evidence is incomplete")
        symbols = endpoints[0].get_chemical_symbols()
        pbc = endpoints[0].pbc
        cell = endpoints[0].cell.array
        fixed_positions = endpoints[0].positions[list(fixed_atoms)]
        for frame in (*endpoints, *initial_band, *final_band):
            if (
                frame.get_chemical_symbols() != symbols
                or not np.array_equal(frame.pbc, pbc)
                or not np.array_equal(frame.cell.array, cell)
                or not np.isfinite(frame.positions).all()
                or not np.isfinite(frame.cell.array).all()
            ):
                raise ValueError("NEB atom or fixed-cell evidence drifted")
            if self._fixed_atom_indices(frame) != fixed_atoms:
                raise ValueError("NEB FixAtoms constraint evidence drifted")
            if fixed_atoms and not np.array_equal(
                frame.positions[list(fixed_atoms)], fixed_positions
            ):
                raise ValueError("NEB fixed coordinate evidence drifted")
        observations = result.get("observations")
        if not isinstance(observations, list):
            raise ValueError("NEB task observations are malformed")
        frames = read_extxyz_structures(
            task_dir / "observations.extxyz", allow_empty=True
        )
        if len(frames) != len(observations):
            raise ValueError("NEB observation evidence is incomplete")
        expected_pairs = (
            [
                (step, image_index)
                for step in range(result["optimizer_steps"] + 1)
                for image_index in range(1, image_count + 1)
            ]
            if parameters["include_history"]
            else (
                [
                    (result["optimizer_steps"], image_index)
                    for image_index in range(image_count + 2)
                ]
                if result["converged"]
                else []
            )
        )
        actual_pairs = [
            (row.get("optimizer_step"), row.get("image_index"))
            if isinstance(row, dict)
            else (None, None)
            for row in observations
        ]
        if actual_pairs != expected_pairs:
            raise ValueError("NEB observation order evidence drifted")
        if (
            result["converged"]
            and not parameters["include_history"]
            and (
                not np.array_equal(frames[0].positions, endpoints[0].positions)
                or not np.array_equal(frames[-1].positions, endpoints[1].positions)
            )
        ):
            raise ValueError("NEB final-only endpoint observation evidence drifted")
        for row, frame in zip(observations, frames, strict=True):
            if not isinstance(row, dict) or str(frame.info.get("structure_id", "")) != row.get("structure_id"):
                raise ValueError("NEB observation identity evidence is mismatched")
            if (
                frame.get_chemical_symbols() != symbols
                or not np.array_equal(frame.pbc, pbc)
                or not np.array_equal(frame.cell.array, cell)
                or self._fixed_atom_indices(frame) != fixed_atoms
                or (
                    bool(fixed_atoms)
                    and not np.array_equal(
                        frame.positions[list(fixed_atoms)], fixed_positions
                    )
                )
            ):
                raise ValueError("NEB observation constraint or cell evidence drifted")
            deviation = row.get("force_deviation")
            if (
                isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(float(deviation))
                or float(deviation) < 0.0
            ):
                raise ValueError("NEB observation deviation is invalid")
        return observations

    def _validate_endpoint_preparation(
        self,
        task_dir: Path,
        *,
        evidence: SamplingEvidenceRequest,
        task: object,
        expected_raw_endpoints: tuple[Atoms, ...],
    ) -> dict[str, object]:
        payload = self._read_json(task_dir / "endpoint_relaxation.json")
        raw = read_extxyz_structures(
            task_dir / "raw_endpoints.extxyz", allow_empty=False
        )
        optimized = read_extxyz_structures(
            task_dir / "endpoints.extxyz", allow_empty=False
        )
        raw_ids = [structure_id_for_atoms(frame) for frame in raw]
        expected_raw_ids = [
            structure_id_for_atoms(frame) for frame in expected_raw_endpoints
        ]
        optimized_ids = [structure_id_for_atoms(frame) for frame in optimized]
        rows = payload.get("endpoints")
        if (
            len(raw) != 2
            or len(optimized) != 2
            or raw_ids != expected_raw_ids
            or set(payload)
            != {
                "schema",
                "schema_revision",
                "algorithm",
                "driver_member_id",
                "fmax_ev_a",
                "max_steps",
                "constraint",
                "raw_structure_ids",
                "optimized_structure_ids",
                "raw_endpoints_sha256",
                "optimized_endpoints_sha256",
                "endpoints",
            }
            or payload["schema"] != "dice.lifecycle.neb-endpoint-relaxation"
            or payload["schema_revision"] != 1
            or payload["algorithm"]
            != {
                "name": "ase-endpoint-fire",
                "revision": 1,
                "optimizer": "FIRE",
                "fixed_cell": True,
            }
            or payload["driver_member_id"] != evidence.models.member_ids[0]
            or payload["fmax_ev_a"] != task.neb.fmax_ev_a
            or payload["max_steps"] != task.neb.max_steps
            or payload["constraint"] != neb_constraint_identity(task.neb)
            or payload["raw_structure_ids"] != raw_ids
            or payload["optimized_structure_ids"] != optimized_ids
            or payload["raw_endpoints_sha256"]
            != self._digest(task_dir / "raw_endpoints.extxyz")
            or payload["optimized_endpoints_sha256"]
            != self._digest(task_dir / "endpoints.extxyz")
            or not isinstance(rows, list)
            or len(rows) != 2
        ):
            raise ValueError("NEB endpoint preparation evidence is malformed or drifted")
        expected_roles = ("initial", "final")
        for index, (row, role) in enumerate(zip(rows, expected_roles, strict=True)):
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "role",
                    "converged",
                    "optimizer_steps",
                    "raw_structure_id",
                    "optimized_structure_id",
                }
                or row["role"] != role
                or row["converged"] is not True
                or isinstance(row["optimizer_steps"], bool)
                or not isinstance(row["optimizer_steps"], int)
                or not 0 <= row["optimizer_steps"] <= task.neb.max_steps
                or row["raw_structure_id"] != raw_ids[index]
                or row["optimized_structure_id"] != optimized_ids[index]
            ):
                raise ValueError("NEB endpoint convergence evidence is malformed")
        symbols = raw[0].get_chemical_symbols()
        pbc = raw[0].pbc
        cell = raw[0].cell.array
        fixed_atoms = tuple(task.neb.fixed_atoms)
        fixed_positions = raw[0].positions[list(fixed_atoms)]
        for frame in (*raw, *optimized):
            if (
                frame.get_chemical_symbols() != symbols
                or not np.array_equal(frame.pbc, pbc)
                or not np.array_equal(frame.cell.array, cell)
                or not np.isfinite(frame.positions).all()
                or self._fixed_atom_indices(frame) != fixed_atoms
                or (
                    bool(fixed_atoms)
                    and not np.array_equal(
                        frame.positions[list(fixed_atoms)], fixed_positions
                    )
                )
            ):
                raise ValueError("NEB endpoint preparation atom/cell evidence drifted")
        return payload

    @staticmethod
    def _fixed_atom_indices(frame: Atoms) -> tuple[int, ...]:
        if not frame.constraints:
            return ()
        if len(frame.constraints) != 1 or not isinstance(
            frame.constraints[0], FixAtoms
        ):
            raise ValueError("NEB constraint evidence is not one ASE FixAtoms mask")
        return tuple(int(index) for index in frame.constraints[0].get_indices())

    @staticmethod
    def _write_frames(path: Path, frames: Sequence[Atoms]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        write_extxyz_structures(temporary, frames, allow_empty=True)
        os.replace(temporary, path)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"NEB canonical evidence is unreadable: {path.name}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"NEB canonical evidence is malformed: {path.name}")
        return payload

    @staticmethod
    def _digest(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"NEB required regular file is missing: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class _ValidationOnlyCalculatorProvider:
    policy_identity = {"name": "committee-force-deviation", "revision": 1}

    def preflight(self, model_paths: tuple[Path, ...]) -> None:
        del model_paths

    def create(self, model_path: Path) -> object:
        raise AssertionError(
            f"read-only NEB validation attempted calculator creation for {model_path}"
        )


class SubprocessAseNebSamplingNativeOps:
    """Run the closed ASE/Calculator implementation in its selected role env."""

    def __init__(self, context: OperationContext, *, model_family: str) -> None:
        expected_role = {
            "deepmd-tf": "student.deepmd.tf.neb",
            "deepmd-pt": "student.deepmd.pt.neb",
            "nep": "student.nep.neb",
        }.get(str(model_family))
        if expected_role is None or context.role != expected_role:
            raise ValueError("ASE NEB operation context does not match its model family")
        self.context = context
        self.model_family = str(model_family)
        self._validation = AseNebSamplingNativeOps(
            calculator_provider=_ValidationOnlyCalculatorProvider(),
            model_family=self.model_family,
            operation_role=context.role,
            operation_profile_id=context.profile_id,
        )

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        evidence = request.evidence
        tasks = self._validation._tasks(evidence)
        worker_root = evidence.sample_dir / ".neb_worker_inputs"
        if worker_root.exists() or worker_root.is_symlink():
            raise ValueError("ASE NEB worker input destination is not fresh")
        worker_root.mkdir(parents=True)
        try:
            seed_paths: dict[str, Path] = {}
            for role_id, frames in evidence.seed_mapping.items():
                path = worker_root / f"{role_id}.extxyz"
                self._validation._write_frames(path, frames)
                seed_paths[role_id] = path
            role_to_group = dict(
                evidence.plan.selection_plan_for_round(
                    evidence.round_index
                ).role_to_group
            )
            manifest = {
                "schema": "dice.native.ase-neb-request",
                "schema_revision": 4,
                "run_json_path": str(evidence.run_json_path),
                "iter": evidence.iter_index,
                "round": evidence.round_index,
                "sample_dir": str(evidence.sample_dir),
                "model_family": self.model_family,
                "operation_role": self.context.role,
                "operation_profile_id": self.context.profile_id,
                "allow_empty_observations": any(
                    task.kind != "neb"
                    for task in evidence.plan.tasks_for_round(evidence.round_index)
                ),
                "max_workers": min(
                    len(tasks),
                    self.context.allocation.cpus_per_task,
                    4,
                ),
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
                        **neb_parameters_payload(task.neb),
                    }
                    for task in tasks
                ],
            }
            manifest_path = evidence.sample_dir / ".neb_worker_request.json"
            self._validation._write_json(manifest_path, manifest)
            run_operation(
                self.context,
                "python",
                (
                    "-m",
                    "workflow_backends.ase_neb.worker",
                    "--manifest",
                    str(manifest_path),
                ),
                cwd=evidence.sample_dir,
                diagnostic_name="ase_neb",
            )
            source = self._validation.open_current(evidence).source
            result = SamplingExecutionResult(
                evidence.sample_dir, len(tasks)
            )
            source.catalog()
            manifest_path.unlink()
            shutil.rmtree(worker_root)
            return result
        except BaseException:
            raise

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


class AseNebSamplingBackendFactory:
    def __init__(self, context: OperationContext, *, model_family: str) -> None:
        self.context = context
        self.model_family = str(model_family)

    def bind(self, binding: SamplingBackendBinding) -> SubprocessAseNebSamplingNativeOps:
        del binding
        return SubprocessAseNebSamplingNativeOps(
            self.context, model_family=self.model_family
        )


__all__ = [
    "AseNebSamplingBackendFactory",
    "AseNebSamplingNativeOps",
    "NebCalculatorProvider",
    "SubprocessAseNebSamplingNativeOps",
]
