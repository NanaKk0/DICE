"""Backend-owned bounded recovery of committed sample blocks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from ase.constraints import FixAtoms
from ase.io import iread

from workflow_backends.sample_artifacts import (
    CellProgramSampleTaskContext,
    ExplicitSampleTaskContext,
    explicit_task_seed,
    load_task_context,
)
from workflow_core.structures.canonical_identity import (
    structure_id_for_atoms,
    validate_canonical_identity,
)
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures
from workflow_core.common.sampling_plan import (
    neb_algorithm_identity,
    neb_constraint_identity,
    neb_parameters_payload,
    relax_algorithm_identity,
    relax_constraint_identity,
    relax_parameters_payload,
)


def validate_sample_artifacts(root: str | Path, facts: object) -> bool:
    try:
        block = _directory(root)
        seeds = _seeds(block, facts)
        tasks = _tasks(block, len(facts.tasks))
        # Sampling model presentations are normally hard-linked into every
        # task directory.  Keep one digest per inode for this validation pass;
        # this preserves byte-level checks while avoiding hashing the same
        # multi-megabyte model once per task.
        digest_cache: dict[tuple[int, int, int, int], str] = {}
        usable_observations = 0
        for index, (task_dir, intent) in enumerate(
            zip(tasks, facts.tasks, strict=True)
        ):
            seed = seeds[facts.system_ids.index(intent.system_id)]
            if intent.kind == 'md_mc':
                from dataclasses import asdict
                from workflow_backends.lammps.identity_swap import validate_result
                context, result = validate_result(task_dir)
                if (context.task_index != index or context.task_id != intent.task_id
                    or context.sampling_role_id != intent.sampling_role_id or context.condition_group_id != intent.condition_group_id
                    or context.selection_group_id != intent.selection_group_id or context.temperature_k != intent.temperature_k
                    or context.ensemble != intent.ensemble or context.pressure_bar != intent.pressure_bar
                    or context.steps != intent.steps or context.source_frame_index != intent.source_frame_index
                    or context.selected_seed_id != seed.info['structure_id']
                    or context.md_mc['parameters'] != json.loads(json.dumps(asdict(intent.md_mc)))
                    or tuple(context.md_mc['member_ids']) != facts.member_ids):
                    return False
                usable_observations += len(result['observations'])
                continue
            if intent.kind in {'gcmc_md', 'widom'}:
                if intent.kind == 'widom':
                    from workflow_backends.widom.adapter import validate_task
                else:
                    from workflow_backends.gcmc.adapter import validate_task

                result = validate_task(task_dir, task_id=intent.task_id)
                if tuple(result['member_ids']) != facts.member_ids or any(
                    result[key] != getattr(intent, key)
                    for key in ('sampling_role_id', 'selection_group_id', 'condition_group_id')
                ):
                    return False
                if intent.kind == 'widom':
                    from dataclasses import asdict
                    if (result['parameters'] != json.loads(json.dumps(asdict(intent.widom)))
                        or result['source_frame_index'] != intent.source_frame_index
                        or result['host']['numbers'] != seed.numbers.tolist()
                        or not np.allclose(result['host']['positions'], seed.positions, rtol=0, atol=1e-8)
                        or not np.allclose(result['host']['cell'], seed.cell.array, rtol=0, atol=1e-10)):
                        return False
                for name in result['files']:
                    if not (task_dir/name).stat().st_size:
                        continue
                    for frame in iread(task_dir/name, index=':'):
                        if not set(frame.get_chemical_symbols()) <= set(facts.type_map):
                            return False
                        _finite_atoms(frame)
                usable_observations += len(result['observations'])
                continue
            if intent.kind == "neb":
                usable_observations += _validate_neb_task(
                    task_dir,
                    intent=intent,
                    task_index=index,
                    seed=seed,
                    facts=facts,
                    digest_cache=digest_cache,
                )
                continue
            if intent.kind == "relax":
                usable_observations += _validate_relax_task(
                    task_dir,
                    intent=intent,
                    task_index=index,
                    seed=seed,
                    facts=facts,
                    digest_cache=digest_cache,
                )
                continue
            context = load_task_context(task_dir / "task_context.json")
            if intent.condition_group_id is not None:
                expected_program = (
                    tuple(
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
                        for segment in intent.program.segments
                    )
                    if intent.program is not None
                    else ()
                )
                if not isinstance(context, ExplicitSampleTaskContext) or (
                    context.task_index != index
                    or context.task_id != intent.task_id
                    or intent.task_id != task_dir.name
                    or context.condition_group_id != intent.condition_group_id
                    or context.sampling_role_id != intent.sampling_role_id
                    or context.selection_group_id != intent.selection_group_id
                    or context.system_id != intent.selection_group_id
                    or context.kind != intent.kind
                    or context.ensemble != intent.ensemble
                    or context.temperature_k != intent.temperature_k
                    or context.pressure_bar != intent.pressure_bar
                    or context.steps != intent.steps
                    or context.source_frame_index != intent.source_frame_index
                    or context.recipe_binding != intent.condition_group_id
                    or context.temperature_program != expected_program
                    or context.segment_boundaries
                    != tuple(item["end_step"] for item in expected_program)
                    or context.native_seed
                    != explicit_task_seed(
                        iter_index=facts.iter_index,
                        round_index=facts.round_index,
                        sampling_role_id=str(intent.sampling_role_id),
                    )
                    or context.selected_seed_id != seed.info["structure_id"]
                ):
                    return False
            elif (
                isinstance(context, ExplicitSampleTaskContext)
                or context.task_index != index
                or intent.task_id != task_dir.name
                or context.system_id != intent.system_id
                or context.ensemble != intent.ensemble
                or context.temperature_k != intent.temperature_k
                or context.pressure_bar != intent.pressure_bar
                or context.steps != facts.steps
                or context.selected_seed_id != seed.info["structure_id"]
            ):
                return False
            if facts.backend == "LAMMPS":
                usable_observations += _validate_lammps_task(
                    task_dir, context, seed, facts, digest_cache=digest_cache
                )
            else:
                _validate_gpumd_task(
                    task_dir, context, seed, facts, digest_cache=digest_cache
                )
                usable_observations += 1
        if usable_observations <= 0 and not all(task.kind in {'gcmc_md', 'widom'} for task in facts.tasks):
            return False
        _validate_compression(block, facts, digest_cache=digest_cache)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _validate_relax_task(
    task: Path,
    *,
    intent: object,
    task_index: int,
    seed: object,
    facts: object,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> int:
    if digest_cache is None:
        digest_cache = {}
    context = _json_object(task / "task_context.json")
    result = _json_object(task / "relax_result.json")
    success = _json_object(task / "task_completion_receipt.json")
    parameters = relax_parameters_payload(intent.relax)
    algorithm = relax_algorithm_identity(intent.relax)
    constraint = relax_constraint_identity(intent.relax)
    expected_role = (
        f"student.deepmd.{facts.cli}.relax"
        if facts.backend == "LAMMPS"
        else "student.nep.relax"
    )
    if (
        context.get("schema") != "dice.lifecycle.relax-task-context"
        or context.get("schema_revision") != 1
        or context.get("iter") != facts.iter_index
        or context.get("round") != facts.round_index
        or context.get("task_index") != task_index
        or context.get("task_id") != intent.task_id
        or intent.task_id != task.name
        or context.get("condition_group_id") != intent.condition_group_id
        or context.get("sampling_role_id") != intent.sampling_role_id
        or context.get("selection_group_id") != intent.selection_group_id
        or context.get("source_frame_index") != intent.source_frame_index
        or context.get("seed_structure_id") != structure_id_for_atoms(seed)
        or context.get("parameters") != parameters
        or context.get("algorithm") != algorithm
        or context.get("constraint") != constraint
        or not isinstance(context.get("operation"), dict)
        or context["operation"].get("role") != expected_role
    ):
        raise ValueError("relax task context changed")
    expected_family = f"deepmd-{facts.cli}" if facts.backend == "LAMMPS" else "nep"
    if context.get("model_family") != expected_family:
        raise ValueError("relax task model family changed")
    model_rows = context.get("models")
    if not isinstance(model_rows, list) or len(model_rows) != len(facts.member_ids):
        raise ValueError("relax task model inventory changed")
    for index, (row, member_id) in enumerate(
        zip(model_rows, facts.member_ids, strict=True)
    ):
        if not isinstance(row, dict) or row.get("member_id") != member_id:
            raise ValueError("relax task model member order changed")
        model_name = row.get("path")
        if not isinstance(model_name, str) or not model_name.startswith(
            f"model_{index:03d}"
        ):
            raise ValueError("relax task model presentation changed")
        if row.get("sha256") != _sha256(
            task / model_name, cache=digest_cache
        ):
            raise ValueError("relax task model digest changed")
    converged = result.get("converged")
    termination = result.get("termination")
    accepted_termination = (
        converged is True and termination == "converged"
    ) or (
        intent.relax.accept_unconverged
        and converged is False
        and termination == "max_steps"
    )
    if (
        result.get("schema") != "dice.lifecycle.relax-task-result"
        or result.get("schema_revision") != 1
        or result.get("task_id") != intent.task_id
        or not accepted_termination
        or isinstance(result.get("optimizer_steps"), bool)
        or not isinstance(result.get("optimizer_steps"), int)
        or not 0 <= result["optimizer_steps"] <= intent.relax.max_steps
        or result.get("algorithm") != algorithm
        or result.get("constraint") != constraint
    ):
        raise ValueError("relax task result changed")
    for field in (
        "initial_energy_ev",
        "final_energy_ev",
        "final_max_force_ev_a",
    ):
        value = result.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("relax task scalar result changed")
    if float(result["final_max_force_ev_a"]) < 0.0 or (
        converged is True
        and float(result["final_max_force_ev_a"]) > intent.relax.fmax_ev_a
    ):
        raise ValueError("relax task convergence evidence changed")
    if converged is False and result["optimizer_steps"] != intent.relax.max_steps:
        raise ValueError("relax max-step termination evidence changed")
    required_files = {
        "task_context.json",
        "raw_seed.extxyz",
        "final_structure.extxyz",
        "observations.extxyz",
        "optimizer.log",
    }
    if intent.relax.include_history:
        required_files.add("trajectory.extxyz")
    files = result.get("files")
    if not isinstance(files, dict) or set(files) != required_files:
        raise ValueError("relax task file inventory changed")
    if any(
        digest != _sha256(task / name, cache=digest_cache)
        for name, digest in files.items()
    ):
        raise ValueError("relax task file digest changed")
    if success != {
        "schema": "dice.lifecycle.relax-task-completion-receipt",
        "schema_revision": 1,
        "task_id": intent.task_id,
        "result_sha256": _sha256(task / "relax_result.json", cache=digest_cache),
    }:
        raise ValueError("relax task completion receipt changed")
    raw = read_extxyz_structures(task / "raw_seed.extxyz", allow_empty=False)
    final = read_extxyz_structures(
        task / "final_structure.extxyz", allow_empty=False
    )
    if (
        len(raw) != 1
        or len(final) != 1
        or structure_id_for_atoms(raw[0]) != structure_id_for_atoms(seed)
    ):
        raise ValueError("relax seed or final evidence changed")
    fixed_atoms = tuple(intent.relax.fixed_atoms)
    fixed_positions = raw[0].positions[list(fixed_atoms)]
    observations = read_extxyz_structures(
        task / "observations.extxyz", allow_empty=False
    )
    trajectory = (
        read_extxyz_structures(task / "trajectory.extxyz", allow_empty=False)
        if intent.relax.include_history
        else ()
    )
    for frame in (*raw, *final, *observations, *trajectory):
        _finite_atoms(frame)
        if (
            frame.get_chemical_symbols() != raw[0].get_chemical_symbols()
            or not np.array_equal(frame.pbc, raw[0].pbc)
            or not np.array_equal(frame.cell.array, raw[0].cell.array)
            or _fixed_atom_indices(frame) != fixed_atoms
            or (
                bool(fixed_atoms)
                and not np.array_equal(
                    frame.positions[list(fixed_atoms)], fixed_positions
                )
            )
        ):
            raise ValueError("relax fixed structure evidence changed")
    rows = result.get("observations")
    if not isinstance(rows, list) or len(rows) != len(observations):
        raise ValueError("relax observation inventory changed")
    optimizer_steps = result["optimizer_steps"]
    expected_steps = (
        [
            *range(0, optimizer_steps + 1, intent.relax.history_stride),
            *(
                []
                if optimizer_steps % intent.relax.history_stride == 0
                else [optimizer_steps]
            ),
        ]
        if intent.relax.include_history
        else [optimizer_steps]
    )
    if [row.get("optimizer_step") for row in rows] != expected_steps:
        raise ValueError("relax observation order changed")
    for row, frame in zip(rows, observations, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("task_index") != task_index
            or row.get("task_id") != intent.task_id
            or row.get("sampling_role_id") != intent.sampling_role_id
            or row.get("selection_group_id") != intent.selection_group_id
            or row.get("condition_group_id") != intent.condition_group_id
            or row.get("source_frame_index") != intent.source_frame_index
            or row.get("structure_id") != frame.info.get("structure_id")
            or not isinstance(row.get("member_energies_ev"), list)
            or len(row["member_energies_ev"]) != len(facts.member_ids)
            or not isinstance(row.get("member_forces_ev_a"), list)
            or len(row["member_forces_ev_a"]) != len(facts.member_ids)
            or not math.isfinite(float(row.get("force_deviation", math.nan)))
            or float(row["force_deviation"]) < 0.0
        ):
            raise ValueError("relax observation evidence changed")
        if any(
            not math.isfinite(float(energy))
            for energy in row["member_energies_ev"]
        ):
            raise ValueError("relax member energy evidence changed")
        member_forces = np.asarray(row["member_forces_ev_a"], dtype=float)
        if (
            member_forces.shape != (len(facts.member_ids), len(frame), 3)
            or not np.isfinite(member_forces).all()
        ):
            raise ValueError("relax member force evidence changed")
    if intent.relax.include_history and len(trajectory) != optimizer_steps + 1:
        raise ValueError("relax trajectory evidence changed")
    return len(rows)


def _validate_neb_task(
    task: Path,
    *,
    intent: object,
    task_index: int,
    seed: object,
    facts: object,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> int:
    if digest_cache is None:
        digest_cache = {}
    context = _json_object(task / "task_context.json")
    endpoints = read_extxyz_structures(task / "endpoints.extxyz", allow_empty=False)
    raw_endpoints = (
        read_extxyz_structures(task / "raw_endpoints.extxyz", allow_empty=False)
        if intent.neb.relax
        else endpoints
    )
    if (
        len(endpoints) != 2
        or len(raw_endpoints) != 2
        or exact_structure_hash(raw_endpoints[0]) != exact_structure_hash(seed)
        or len(endpoints[0]) != len(endpoints[1])
        or len(raw_endpoints[0]) != len(raw_endpoints[1])
        or endpoints[0].get_chemical_symbols()
        != endpoints[1].get_chemical_symbols()
        or raw_endpoints[0].get_chemical_symbols()
        != raw_endpoints[1].get_chemical_symbols()
        or endpoints[0].get_chemical_symbols()
        != raw_endpoints[0].get_chemical_symbols()
        or not np.array_equal(endpoints[0].pbc, endpoints[1].pbc)
        or not np.array_equal(raw_endpoints[0].pbc, raw_endpoints[1].pbc)
        or not np.array_equal(endpoints[0].cell.array, endpoints[1].cell.array)
        or not np.array_equal(
            raw_endpoints[0].cell.array, raw_endpoints[1].cell.array
        )
        or not np.array_equal(endpoints[0].cell.array, raw_endpoints[0].cell.array)
    ):
        raise ValueError("NEB endpoint evidence changed")
    for endpoint in (*raw_endpoints, *endpoints):
        _finite_atoms(endpoint)
    fixed_atoms = tuple(intent.neb.fixed_atoms)
    fixed_positions = raw_endpoints[0].positions[list(fixed_atoms)]
    for endpoint in (*raw_endpoints, *endpoints):
        if _fixed_atom_indices(endpoint) != fixed_atoms:
            raise ValueError("NEB endpoint FixAtoms evidence changed")
        if fixed_atoms and not np.array_equal(
            endpoint.positions[list(fixed_atoms)], fixed_positions
        ):
            raise ValueError("NEB fixed endpoint coordinates changed")
    endpoint_ids = [
        f"sha256:{exact_structure_hash(endpoint)}" for endpoint in endpoints
    ]
    parameters = neb_parameters_payload(intent.neb)
    algorithm = neb_algorithm_identity(intent.neb)
    constraint = neb_constraint_identity(intent.neb)
    operation = context.get("operation")
    models = context.get("models")
    context_fields = {
        "schema",
        "schema_revision",
        "iter",
        "round",
        "task_index",
        "task_id",
        "condition_group_id",
        "sampling_role_id",
        "selection_group_id",
        "endpoint_structure_ids",
        "parameters",
        "algorithm",
        "constraint",
        "models",
        "operation",
    }
    endpoint_preparation = None
    if intent.neb.relax:
        context_fields.add("endpoint_preparation")
        endpoint_preparation = _validate_neb_endpoint_preparation(
            task,
            intent=intent,
            member_ids=facts.member_ids,
            raw_endpoints=raw_endpoints,
            optimized_endpoints=endpoints,
            digest_cache=digest_cache,
        )
    if (
        set(context) != context_fields
        or context["schema"] != "dice.lifecycle.neb-task-context"
        or context["schema_revision"] != 3
        or context["iter"] != facts.iter_index
        or context["round"] != facts.round_index
        or context["task_index"] != task_index
        or context["task_id"] != intent.task_id
        or intent.task_id != task.name
        or context["condition_group_id"] != intent.condition_group_id
        or context["sampling_role_id"] != intent.sampling_role_id
        or context["selection_group_id"] != intent.selection_group_id
        or context["endpoint_structure_ids"] != endpoint_ids
        or context["parameters"] != parameters
        or context["algorithm"] != algorithm
        or context["constraint"] != constraint
        or (
            intent.neb.relax
            and context["endpoint_preparation"] != endpoint_preparation
        )
        or not isinstance(operation, dict)
        or operation.get("role")
        != (
            f"student.deepmd.{facts.cli}.neb"
            if facts.backend == "LAMMPS"
            else "student.nep.neb"
        )
        or not isinstance(operation.get("profile_id"), str)
        or not operation["profile_id"]
        or not isinstance(operation.get("model_family"), str)
        or not operation["model_family"]
        or not isinstance(models, list)
        or len(models) != len(facts.member_ids)
    ):
        raise ValueError("NEB task context changed")
    presentations: list[Path] = []
    for index, (model, member_id) in enumerate(
        zip(models, facts.member_ids, strict=True)
    ):
        if (
            not isinstance(model, dict)
            or set(model) != {"member_id", "presentation", "sha256"}
            or model["member_id"] != member_id
            or not isinstance(model["presentation"], str)
            or not model["presentation"].startswith(f"model_{index:03d}")
        ):
            raise ValueError("NEB ordered model evidence changed")
        presentation = task / model["presentation"]
        if _sha256(presentation, cache=digest_cache) != model["sha256"]:
            raise ValueError("NEB model presentation digest changed")
        presentations.append(presentation)
    result = _json_object(task / "neb_result.json")
    success = _json_object(task / "task_completion_receipt.json")
    if success != {
        "schema": "dice.lifecycle.neb-task-completion-receipt",
        "schema_revision": 3,
        "task_id": intent.task_id,
        "result_sha256": _sha256(task / "neb_result.json", cache=digest_cache),
    }:
        raise ValueError("NEB task completion receipt changed")
    if (
        result.get("schema") != "dice.lifecycle.neb-task-result"
        or result.get("schema_revision") != 3
        or result.get("task_id") != intent.task_id
        or result.get("algorithm") != algorithm
        or result.get("constraint") != constraint
        or not isinstance(result.get("converged"), bool)
        or result.get("termination")
        != ("converged" if result.get("converged") else "max_steps")
        or isinstance(result.get("optimizer_steps"), bool)
        or not isinstance(result.get("optimizer_steps"), int)
        or not 0 <= result["optimizer_steps"] <= intent.neb.max_steps
    ):
        raise ValueError("NEB task result changed")
    files = result.get("files")
    required_files = {
        "task_context.json",
        "endpoints.extxyz",
        "initial_band.extxyz",
        "final_band.extxyz",
        "observations.extxyz",
    }
    if intent.neb.relax:
        required_files.update(
            {"raw_endpoints.extxyz", "endpoint_relaxation.json"}
        )
        if result.get("endpoint_preparation") != endpoint_preparation:
            raise ValueError("NEB endpoint preparation result evidence changed")
    elif "endpoint_preparation" in result:
        raise ValueError("disabled NEB endpoint preparation evidence is present")
    if (
        not isinstance(files, dict)
        or set(files) != required_files
        or any(
            _sha256(task / name, cache=digest_cache) != digest
            for name, digest in files.items()
        )
    ):
        raise ValueError("NEB canonical file evidence changed")
    initial_band = read_extxyz_structures(
        task / "initial_band.extxyz", allow_empty=False
    )
    final_band = read_extxyz_structures(
        task / "final_band.extxyz", allow_empty=False
    )
    if (
        len(initial_band) != intent.neb.images
        or len(final_band) != intent.neb.images
    ):
        raise ValueError("NEB band evidence is incomplete")
    for frame in (*initial_band, *final_band):
        _finite_atoms(frame)
        if (
            frame.get_chemical_symbols() != endpoints[0].get_chemical_symbols()
            or not np.array_equal(frame.pbc, endpoints[0].pbc)
            or not np.array_equal(frame.cell.array, endpoints[0].cell.array)
            or _fixed_atom_indices(frame) != fixed_atoms
            or (
                bool(fixed_atoms)
                and not np.array_equal(
                    frame.positions[list(fixed_atoms)], fixed_positions
                )
            )
        ):
            raise ValueError("NEB band atom/cell/constraint evidence drifted")
    rows = result.get("observations")
    observations = read_extxyz_structures(
        task / "observations.extxyz", allow_empty=True
    )
    if not isinstance(rows, list) or len(rows) != len(observations):
        raise ValueError("NEB observation evidence is incomplete")
    if not result["converged"] and not intent.neb.include_history and rows:
        raise ValueError("unconverged final-only NEB published observations")
    expected_pairs = (
        [
            (step, image_index)
            for step in range(result["optimizer_steps"] + 1)
            for image_index in range(1, intent.neb.images + 1)
        ]
        if intent.neb.include_history
        else (
            [
                (result["optimizer_steps"], image_index)
                for image_index in range(intent.neb.images + 2)
            ]
            if result["converged"]
            else []
        )
    )
    actual_pairs: list[tuple[int, int]] = []
    identities: set[str] = set()
    for frame_index, (row, frame) in enumerate(zip(rows, observations, strict=True)):
        if not isinstance(row, dict):
            raise ValueError("NEB observation row is malformed")
        actual_pairs.append((row.get("optimizer_step"), row.get("image_index")))
        identity = row.get("structure_id")
        energies = row.get("member_energies_ev")
        forces = row.get("member_forces_ev_a")
        deviation = row.get("force_deviation")
        if (
            row.get("task_index") != task_index
            or row.get("task_id") != intent.task_id
            or row.get("sampling_role_id") != intent.sampling_role_id
            or row.get("selection_group_id") != intent.selection_group_id
            or row.get("condition_group_id") != intent.condition_group_id
            or row.get("frame_index") != frame_index
            or not isinstance(identity, str)
            or identity in identities
            or frame.info.get("structure_id") != identity
            or not isinstance(energies, list)
            or len(energies) != len(facts.member_ids)
            or not np.isfinite(np.asarray(energies, dtype=float)).all()
            or not isinstance(forces, list)
            or np.asarray(forces, dtype=float).shape
            != (len(facts.member_ids), len(frame), 3)
            or not np.isfinite(np.asarray(forces, dtype=float)).all()
            or isinstance(deviation, bool)
            or not isinstance(deviation, (int, float))
            or not math.isfinite(float(deviation))
            or deviation < 0
        ):
            raise ValueError("NEB observation row changed")
        identities.add(identity)
        _finite_atoms(frame)
        if (
            frame.get_chemical_symbols() != endpoints[0].get_chemical_symbols()
            or not np.array_equal(frame.pbc, endpoints[0].pbc)
            or not np.array_equal(frame.cell.array, endpoints[0].cell.array)
            or _fixed_atom_indices(frame) != fixed_atoms
            or (
                bool(fixed_atoms)
                and not np.array_equal(
                    frame.positions[list(fixed_atoms)], fixed_positions
                )
            )
        ):
            raise ValueError("NEB observation atom/cell/constraint evidence changed")
    if actual_pairs != expected_pairs:
        raise ValueError("NEB observation history order changed")
    if (
        result["converged"]
        and not intent.neb.include_history
        and (
            not np.array_equal(observations[0].positions, endpoints[0].positions)
            or not np.array_equal(observations[-1].positions, endpoints[1].positions)
        )
    ):
        raise ValueError("NEB final-only endpoint observations changed")
    return len(rows)


def _validate_neb_endpoint_preparation(
    task: Path,
    *,
    intent: object,
    member_ids: tuple[str, ...],
    raw_endpoints: list[object],
    optimized_endpoints: list[object],
    digest_cache: dict[tuple[int, int, int, int], str],
) -> dict[str, object]:
    payload = _json_object(task / "endpoint_relaxation.json")
    raw_ids = [
        f"sha256:{exact_structure_hash(frame)}" for frame in raw_endpoints
    ]
    optimized_ids = [
        f"sha256:{exact_structure_hash(frame)}" for frame in optimized_endpoints
    ]
    rows = payload.get("endpoints")
    if (
        set(payload)
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
        or payload["driver_member_id"] != member_ids[0]
        or payload["fmax_ev_a"] != intent.neb.fmax_ev_a
        or payload["max_steps"] != intent.neb.max_steps
        or payload["constraint"] != neb_constraint_identity(intent.neb)
        or payload["raw_structure_ids"] != raw_ids
        or payload["optimized_structure_ids"] != optimized_ids
        or payload["raw_endpoints_sha256"]
        != _sha256(task / "raw_endpoints.extxyz", cache=digest_cache)
        or payload["optimized_endpoints_sha256"]
        != _sha256(task / "endpoints.extxyz", cache=digest_cache)
        or not isinstance(rows, list)
        or len(rows) != 2
    ):
        raise ValueError("NEB endpoint preparation evidence changed")
    for index, role in enumerate(("initial", "final")):
        row = rows[index]
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
            or not 0 <= row["optimizer_steps"] <= intent.neb.max_steps
            or row["raw_structure_id"] != raw_ids[index]
            or row["optimized_structure_id"] != optimized_ids[index]
        ):
            raise ValueError("NEB endpoint convergence evidence changed")
    return payload


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sample JSON evidence must be an object")
    return payload


def _seeds(root: Path, facts: object):
    path = root / "seeds.xyz"
    if path.is_symlink():
        raise ValueError("sample seeds must be a regular extxyz")
    seeds = read_extxyz_structures(path, allow_empty=False)
    if len(seeds) != len(facts.system_ids):
        raise ValueError("sample seed coverage changed")
    for seed, system_id in zip(seeds, facts.system_ids, strict=True):
        identity = validate_canonical_identity(seed)
        if identity["system_id"] != system_id:
            raise ValueError("sample seed system order changed")
        if not set(seed.get_chemical_symbols()).issubset(set(facts.type_map)):
            raise ValueError("sample seed contains unsupported elements")
        _finite_atoms(seed)
    return tuple(seeds)


def _tasks(root: Path, count: int) -> tuple[Path, ...]:
    task_root = _directory(root / "tasks")
    entries = tuple(sorted(task_root.iterdir(), key=lambda path: path.name))
    expected = tuple(f"task_{index:06d}" for index in range(count))
    if (
        tuple(path.name for path in entries) != expected
        or any(path.is_symlink() or not path.is_dir() for path in entries)
    ):
        raise ValueError("sample task inventory is not exact")
    return entries


def _validate_lammps_task(
    task: Path,
    context,
    seed,
    facts: object,
    *,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> int:
    if digest_cache is None:
        digest_cache = {}
    from workflow_backends.lammps.observation_source import (
        validate_lammps_sampling_failure,
        validate_lammps_task_terminals,
    )
    from workflow_backends.lammps.cell_program import (
        validate_lammps_cell_program_trajectory,
    )

    task_seed = read_extxyz_structures(task / "seed.xyz", allow_empty=False)
    if (
        len(task_seed) != 1
        or task_seed[0].info.get("structure_id") != seed.info["structure_id"]
        or exact_structure_hash(task_seed[0]) != exact_structure_hash(seed)
    ):
        raise ValueError("LAMMPS task seed disagrees with block-local seed")
    for name in ("conf.lmp", "dice_task_setup.lammps", "input.lammps"):
        _nonempty_regular(task / name)
    _validate_lammps_rendered(task, context)
    suffix = ".pb" if facts.cli == "tf" else ".pt2"
    presentations = tuple(sorted(task.glob("graph_*"), key=lambda path: path.name))
    expected = tuple(
        f"graph_{index:03d}{suffix}" for index in range(len(facts.member_ids))
    )
    if tuple(path.name for path in presentations) != expected:
        raise ValueError("LAMMPS model presentation inventory changed")
    for path in presentations:
        _nonempty_regular(path)
    if isinstance(context, ExplicitSampleTaskContext):
        if context.kind in {"nvt_program", "npt_program"}:
            schedule_names = (
                "dice_temperature_start.dat",
                "dice_temperature_stop.dat",
                "dice_temperature_steps.dat",
            )
            for name in schedule_names:
                _nonempty_regular(task / name)
            if context.schedule_digests != tuple(
                (name, _sha256(task / name, cache=digest_cache))
                for name in schedule_names
            ):
                raise ValueError("explicit LAMMPS schedule evidence changed")
        elif context.kind == "nvt_cell_program":
            if not isinstance(context, CellProgramSampleTaskContext):
                raise ValueError("cell program requires revision-3 task context")
            name = "dice_cell_schedule.json"
            _nonempty_regular(task / name)
            if context.schedule_digests != (
                (name, _sha256(task / name, cache=digest_cache)),
            ):
                raise ValueError("explicit LAMMPS cell schedule evidence changed")
        if (
            (
                context.kind != "nvt_cell_program"
                and context.recipe_sha256
                != _sha256(task / "input.lammps", cache=digest_cache)
            )
            or context.rendered_input_sha256
            != _sha256(task / "input.lammps", cache=digest_cache)
            or context.setup_sha256
            != _sha256(task / "dice_task_setup.lammps", cache=digest_cache)
            or context.ordered_models != tuple(path.name for path in presentations)
            or context.model_digests
            != tuple(_sha256(path, cache=digest_cache) for path in presentations)
        ):
            raise ValueError("explicit LAMMPS task evidence changed")
    deviation = task / "model_devi.out"
    committee = len(facts.member_ids) > 1
    if (
        not committee
        and not isinstance(context, ExplicitSampleTaskContext)
        and (deviation.exists() or deviation.is_symlink())
    ):
        raise ValueError("single-model LAMMPS sample forbids deviation evidence")
    failure = task / "sampling_failure.json"
    failed = failure.exists() or failure.is_symlink()
    if failed and isinstance(context, CellProgramSampleTaskContext):
        raise ValueError("cell program may not publish partial failure evidence")
    if failed:
        terminals = validate_lammps_sampling_failure(
            task_dir=task,
            cadence=context.cadence,
            steps=context.steps,
            require_deviation=committee,
            expected_atom_count=len(seed),
            expected_type_count=len(facts.type_map),
            validate_frame_geometry=True,
            expected_system_id=context.system_id,
        )
    else:
        terminals = validate_lammps_task_terminals(
            task_dir=task,
            cadence=context.cadence,
            steps=context.steps,
            require_deviation=committee,
            require_exact_cadence=isinstance(context, ExplicitSampleTaskContext),
            expected_atom_count=len(seed),
            expected_type_count=len(facts.type_map),
            validate_frame_geometry=True,
        )
        if isinstance(context, CellProgramSampleTaskContext):
            validate_lammps_cell_program_trajectory(
                task_dir=task,
                context=context,
                type_map=tuple(facts.type_map),
            )
    if terminals is not None:
        expected_terminal = (
            (context.steps // context.cadence) * context.cadence
            if isinstance(context, ExplicitSampleTaskContext)
            else context.steps
        )
        if not failed and terminals["trajectory_timesteps"][-1] != expected_terminal:
            raise ValueError("LAMMPS trajectory is not terminal at configured steps")
    plumed = task / "plumed.dat"
    if facts.plumed_enabled:
        _nonempty_regular(plumed)
    elif plumed.exists() or plumed.is_symlink():
        raise ValueError("disabled PLUMED mode has block-local PLUMED evidence")
    receipt = task / "native_execution.json"
    if not failed and terminals is not None and (
        isinstance(context, ExplicitSampleTaskContext)
        or facts.compress
        or receipt.exists()
        or receipt.is_symlink()
    ):
        _validate_native_receipt(
            task,
            context,
            facts,
            presentations,
            terminals,
            digest_cache=digest_cache,
        )
    return 0 if terminals is None else len(terminals["trajectory_timesteps"])


def _validate_lammps_rendered(task: Path, context) -> None:
    setup = (task / "dice_task_setup.lammps").read_text(encoding="utf-8")
    variables: dict[str, str] = {}
    for line in setup.splitlines():
        row = line.split("#", 1)[0].split()
        if len(row) == 4 and row[0] == "variable" and row[2] == "equal":
            if row[1] in variables:
                raise ValueError("LAMMPS setup duplicates a managed variable")
            variables[row[1]] = row[3]
    required = {"dice_run_steps", "dice_temperature", "dice_seed"}
    if context.ensemble == "npt":
        required.add("dice_pressure")
    if not required.issubset(variables):
        raise ValueError("LAMMPS setup lacks normalized task variables")
    expected_temperature = (
        context.temperature_program[0]["start_temperature_k"]
        if isinstance(context, ExplicitSampleTaskContext)
        and context.kind in {"nvt_program", "nvt_cell_program", "npt_program"}
        else context.temperature_k
    )
    if (
        int(variables["dice_run_steps"]) != context.steps
        or float(variables["dice_temperature"]) != expected_temperature
        or int(variables["dice_seed"]) <= 0
        or (
            context.ensemble == "npt"
            and float(variables["dice_pressure"]) != context.pressure_bar
        )
    ):
        raise ValueError("LAMMPS setup disagrees with task context")
    rendered = (task / "input.lammps").read_text(encoding="utf-8")
    expected_run = (
        r"(?m)^\s*run\s+\$\{dice_segment_steps\}\s*$"
        if isinstance(context, ExplicitSampleTaskContext)
        and context.kind in {"nvt_program", "npt_program"}
        else (
            r"(?m)^\s*run\s+\d+\s+start\s+\d+\s+stop\s+\d+\s*$"
            if isinstance(context, CellProgramSampleTaskContext)
            else r"(?m)^\s*run\s+\$\{dice_run_steps\}\s*$"
        )
    )
    if (
        len(re.findall(r"(?m)^\s*include\s+dice_task_setup\.lammps\s*$", rendered)) != 1
        or (
            len(re.findall(expected_run, rendered)) < 1
            if isinstance(context, CellProgramSampleTaskContext)
            else len(re.findall(expected_run, rendered)) != 1
        )
    ):
        raise ValueError("LAMMPS rendered input lacks managed task seams")


def _finite_lammps_trajectory(path: Path, type_map: tuple[str, ...]) -> None:
    frames = list(
        iread(path, index=":", format="lammps-dump-text", specorder=list(type_map))
    )
    if not frames:
        raise ValueError("LAMMPS trajectory is empty")
    signature = tuple(frames[0].get_chemical_symbols())
    pbc = tuple(bool(item) for item in frames[0].pbc)
    for frame in frames:
        if tuple(frame.get_chemical_symbols()) != signature or tuple(frame.pbc) != pbc:
            raise ValueError("LAMMPS trajectory atom/PBC order changed")
        _finite_atoms(frame)


def _validate_gpumd_task(
    task: Path,
    context,
    seed,
    facts: object,
    *,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> None:
    if digest_cache is None:
        digest_cache = {}
    task_seed = read_extxyz_structures(task / "model.xyz", allow_empty=False)
    if (
        len(task_seed) != 1
        or task_seed[0].info.get("structure_id") != seed.info["structure_id"]
        or exact_structure_hash(task_seed[0]) != exact_structure_hash(seed)
    ):
        raise ValueError("GPUMD task seed disagrees with block-local seed")
    presentations = tuple(sorted(task.glob("nep_*.txt"), key=lambda path: path.name))
    expected = tuple(f"nep_{index}.txt" for index in range(len(facts.member_ids)))
    if tuple(path.name for path in presentations) != expected:
        raise ValueError("GPUMD model presentation inventory changed")
    for path in presentations:
        _nonempty_regular(path)
    if isinstance(context, ExplicitSampleTaskContext) and (
        context.ordered_models != tuple(path.name for path in presentations)
        or context.model_digests
        != tuple(_sha256(path, cache=digest_cache) for path in presentations)
        or context.rendered_input_sha256
        != _sha256(task / "run.in", cache=digest_cache)
        or context.setup_sha256
        != _sha256(task / "model.xyz", cache=digest_cache)
    ):
        raise ValueError("explicit GPUMD task evidence changed")
    from workflow_backends.gpumd.observation_source import (
        _active_rows,
        _extxyz_frame_count,
    )

    rows = _active_rows(task / "active.out", run_input_path=task / "run.in")
    if not rows or rows[-1][0] != context.steps:
        raise ValueError("GPUMD active stream does not reach configured steps")
    active = read_extxyz_structures(task / "active.xyz", allow_empty=False)
    if len(active) != len(rows) or _extxyz_frame_count(task / "active.xyz") != len(rows):
        raise ValueError("GPUMD active geometry/evidence count changed")
    signature = tuple(seed.get_chemical_symbols())
    for frame in active:
        if tuple(frame.get_chemical_symbols()) != signature:
            raise ValueError("GPUMD active atom/species order changed")
        _finite_atoms(frame)


def _validate_native_receipt(
    task,
    context,
    facts,
    presentations,
    terminals,
    *,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> None:
    from workflow_backends.lammps.observation_source import (
        validate_lammps_native_execution,
    )

    validate_lammps_native_execution(
        task_dir=task,
        models=presentations,
        terminals=terminals,
        deepmd_recipe_kind=None,
        digest_cache=digest_cache,
    )
    if isinstance(context, ExplicitSampleTaskContext):
        payload = json.loads((task / "native_execution.json").read_text(encoding="utf-8"))
        expected = {
            "task_id": context.task_id,
            "task_index": context.task_index,
            "condition_group_id": context.condition_group_id,
            "sampling_role_id": context.sampling_role_id,
            "selection_group_id": context.selection_group_id,
            "system_id": context.system_id,
            "source_frame_index": context.source_frame_index,
            "selected_structure_id": context.selected_structure_id,
            "temperature_k": context.temperature_k,
            "pressure_bar": context.pressure_bar,
            "kind": context.kind,
            "steps": context.steps,
            "cadence": context.cadence,
            "native_seed": context.native_seed,
            "recipe_binding": context.recipe_binding,
            "recipe_sha256": context.recipe_sha256,
            "rendered_input_sha256": context.rendered_input_sha256,
            "setup_sha256": context.setup_sha256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("explicit LAMMPS native receipt changed")
        if context.kind in {"nvt_program", "nvt_cell_program", "npt_program"} and (
            tuple(payload.get("temperature_program", ()))
            != context.temperature_program
            or tuple(payload.get("segment_boundaries", ()))
            != context.segment_boundaries
            or tuple(
                tuple(item) for item in payload.get("schedule_digests", ())
            )
            != context.schedule_digests
        ):
            raise ValueError("explicit LAMMPS program receipt changed")
        if isinstance(context, CellProgramSampleTaskContext) and any(
            payload.get(key) != value
            for key, value in {
                "resolved_cell_history": context.resolved_cell_history,
                "cell_history_sha256": context.cell_history_sha256,
                "expected_dump_steps": list(context.expected_dump_steps),
            }.items()
        ):
            raise ValueError("explicit LAMMPS cell-program receipt changed")


def _validate_compression(
    root: Path,
    facts: object,
    *,
    digest_cache: dict[tuple[int, int, int, int], str] | None = None,
) -> None:
    if digest_cache is None:
        digest_cache = {}
    manifest = root / "sampling_model_assets.json"
    derived = root / "sampling_models"
    if not facts.compress:
        if manifest.exists() or manifest.is_symlink() or derived.exists() or derived.is_symlink():
            raise ValueError("uncompressed sample has compression artifacts")
        return
    from workflow_artifacts.lifecycle.sampling_models import (
        validate_local_sampling_model_assets,
    )

    records = validate_local_sampling_model_assets(
        sample_dir=root,
        member_ids=facts.member_ids,
        cli=facts.cli,
    )
    expected = tuple(record["derived_sha256"] for record in records)
    suffix = ".pb" if facts.cli == "tf" else ".pt2"
    for task, intent in zip(
        _tasks(root, len(facts.tasks)), facts.tasks, strict=True
    ):
        presentation_stem = (
            "model" if intent.kind in {"neb", "relax"} else "graph"
        )
        presentations = tuple(
            task / f"{presentation_stem}_{index:03d}{suffix}"
            for index in range(len(expected))
        )
        if (
            tuple(_sha256(path, cache=digest_cache) for path in presentations)
            != expected
        ):
            raise ValueError("compressed task presentations changed from manifest")


def _finite_atoms(atoms) -> None:
    if (
        not np.isfinite(np.asarray(atoms.positions, dtype=float)).all()
        or not np.isfinite(np.asarray(atoms.cell.array, dtype=float)).all()
    ):
        raise ValueError("sample geometry contains non-finite values")


def _fixed_atom_indices(atoms) -> tuple[int, ...]:
    if not atoms.constraints:
        return ()
    if len(atoms.constraints) != 1 or not isinstance(
        atoms.constraints[0], FixAtoms
    ):
        raise ValueError("NEB constraint evidence is not one ASE FixAtoms mask")
    return tuple(int(index) for index in atoms.constraints[0].get_indices())


def _nonempty_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"sample artifact is missing or non-regular: {path.name}")


def _directory(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("sample recovery requires a regular directory")
    return raw.resolve()


def _sha256(
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    if cache is not None:
        cache[key] = result
    return result


__all__ = ["validate_sample_artifacts"]
