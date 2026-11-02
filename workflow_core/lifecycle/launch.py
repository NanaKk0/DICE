from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import numpy as np
import os
from pathlib import Path

from workflow_core.lifecycle.config import (
    RunConfig,
    load_run_config,
    require_external_config_path,
    validate_role_recipe_feasibility,
    validate_sampler_recipe_feasibility,
)
from workflow_core.lifecycle.routes import RouteSpec
from workflow_runtime.lifecycle.environment import EnvironmentPreflight, load_environment_profile, preflight_environment
from workflow_runtime.execution.profile import (
    ExecutionBinding,
    ExecutionProfile,
    load_execution_profile,
)
from workflow_runtime.execution.planning import PlannedExecution, plan_execution_units


@dataclass(frozen=True, slots=True)
class LifecycleBlock:
    block_id: str
    artifact_root: Path


@dataclass(frozen=True, slots=True)
class LifecyclePlan:
    blocks: tuple[LifecycleBlock, ...]


@dataclass(frozen=True, slots=True)
class LifecycleLaunch:
    config: RunConfig
    run_root: Path
    entropy_root: str
    plan: LifecyclePlan
    environment_path: Path | None
    preflight: EnvironmentPreflight | None
    resume_target: str | None
    execution_path: Path | None = None
    execution_profile: ExecutionProfile | None = None
    execution_plans: dict[str, tuple[PlannedExecution, ...]] = field(
        default_factory=dict
    )
    route: RouteSpec | None = None
    block_operation_role_map: dict[str, tuple[str, ...]] | None = None
    block_policy_evidence: dict[str, dict[str, object]] | None = None


def lifecycle_preflight_evidence(launch: LifecycleLaunch) -> dict[str, object]:
    """Project one prepared launch into stable, read-only compile-time evidence."""
    if launch.preflight is None:
        raise ValueError("lifecycle preflight evidence requires an environment preflight")
    if launch.route is None:
        raise ValueError("lifecycle preflight evidence requires a resolved route")
    blocks = [block.block_id for block in launch.plan.blocks]
    return {
        "schema": "dice.lifecycle.preflight",
        "schema_revision": 1,
        "iter": launch.config.iter_index,
        "route": launch.route.kind.value,
        "blocks": blocks,
        "terminal_block": blocks[-1],
        "environment": str(launch.environment_path),
        "operation_roles": list(launch.preflight.operation_roles),
        "operation_environment": launch.preflight.evidence,
        "execution": None if launch.execution_path is None else str(launch.execution_path),
        "execution_profile_digest": (
            None
            if launch.execution_profile is None
            else launch.execution_profile.digest
        ),
    }


def prepare_lifecycle_launch(
    run_json: str | Path,
    *,
    environment_path: str | Path | None = None,
    execution_path: str | Path | None = None,
    preflight: bool = False,
    resume: str | None = None,
) -> LifecycleLaunch:
    config = load_run_config(run_json)
    route = _route(config)
    plan = build_lifecycle_plan(config)
    if resume is not None and resume not in {block.block_id for block in plan.blocks}:
        raise ValueError(f"invalid lifecycle lifecycle resume target {resume!r}")
    selected_environment = _select_environment(environment_path)
    selected_execution = _select_execution(execution_path)
    block_roles = {block.block_id: block_operation_roles(config, block.block_id) for block in plan.blocks}
    block_policies = {
        block.block_id: route.policy_evidence(config.raw, block.block_id)
        for block in plan.blocks
    }
    launch = LifecycleLaunch(
        config=config,
        run_root=config.root,
        entropy_root=str(config.path),
        plan=plan,
        environment_path=selected_environment,
        preflight=None,
        resume_target=resume,
        execution_path=selected_execution,
        execution_profile=None,
        execution_plans={},
        route=route,
        block_operation_role_map=block_roles,
        block_policy_evidence=block_policies,
    )
    if not preflight:
        return launch
    first_block = resume if resume is not None else plan.blocks[0].block_id
    return prepare_execution_suffix(
        launch,
        first_block=first_block,
        require_environment=preflight,
    )


def prepare_execution_suffix(
    launch: LifecycleLaunch,
    *,
    first_block: str,
    require_environment: bool = True,
) -> LifecycleLaunch:
    """Validate external feasibility only for blocks that will execute."""

    block_ids = tuple(block.block_id for block in launch.plan.blocks)
    if first_block not in block_ids:
        raise ValueError(f"invalid lifecycle execution block {first_block!r}")
    suffix = block_ids[block_ids.index(first_block) :]
    _validate_suffix_external_inputs(launch.config, suffix)

    selected_environment = launch.environment_path
    environment_preflight = None
    ordered_roles: list[str] = []
    for block in suffix:
        for role in block_operation_roles(launch.config, block):
            if role not in ordered_roles:
                ordered_roles.append(role)
    suffix_roles = tuple(ordered_roles)
    if require_environment:
        if selected_environment is None:
            raise ValueError(
                f"lifecycle execution suffix from {first_block} requires "
                "--environment or DICE_ENVIRONMENT"
            )
        profile = load_environment_profile(selected_environment)
        environment_preflight = preflight_environment(profile, suffix_roles)

    execution_profile = None
    if launch.execution_path is not None:
        execution_profile = load_execution_profile(launch.execution_path)
        _validate_execution_profile(
            launch.config,
            execution_profile,
            operation_roles=frozenset(suffix_roles),
        )
    suffix_role_map = {
        block: block_operation_roles(launch.config, block) for block in suffix
    }
    execution_plans = _build_execution_plans(
        launch.config,
        block_roles=suffix_role_map,
        profile=execution_profile,
    )
    return replace(
        launch,
        preflight=environment_preflight,
        execution_profile=execution_profile,
        execution_plans=execution_plans,
    )


def prepare_execution_block(
    launch: LifecycleLaunch,
    *,
    block_id: str,
    require_environment: bool = True,
) -> LifecycleLaunch:
    """Validate external feasibility for exactly one scheduled public block."""

    block_ids = tuple(block.block_id for block in launch.plan.blocks)
    if block_id not in block_ids:
        raise ValueError(f"invalid lifecycle execution block {block_id!r}")
    _validate_suffix_external_inputs(launch.config, (block_id,))
    roles = block_operation_roles(launch.config, block_id)
    environment_preflight = None
    if require_environment:
        if launch.environment_path is None:
            raise ValueError(
                f"lifecycle execution block {block_id} requires "
                "--environment or DICE_ENVIRONMENT"
            )
        profile = load_environment_profile(launch.environment_path)
        environment_preflight = preflight_environment(profile, roles)
    execution_profile = None
    if launch.execution_path is not None:
        execution_profile = load_execution_profile(launch.execution_path)
        _validate_execution_profile(
            launch.config,
            execution_profile,
            operation_roles=frozenset(roles),
        )
    return replace(
        launch,
        preflight=environment_preflight,
        execution_profile=execution_profile,
        execution_plans=_build_execution_plans(
            launch.config,
            block_roles={block_id: roles},
            profile=execution_profile,
        ),
    )


def _validate_suffix_external_inputs(
    config: RunConfig, suffix: tuple[str, ...]
) -> None:
    suffix_set = set(suffix)

    def require(block: str, field: str, raw_path: str) -> Path:
        try:
            return require_external_config_path(config, raw_path, field_name=field)
        except ValueError as exc:
            raise ValueError(f"{block} external input {field}: {exc}") from exc

    if "teach/train" in suffix_set:
        for index, path in enumerate(config.raw["teacher_train_bootstrap"]):
            require("teach/train", f"teacher_train_bootstrap[{index}]", path)
        for index, path in enumerate(config.raw["fp_data"]):
            require("teach/train", f"fp_data[{index}]", path)
        require("teach/train", "teacher.recipe", config.teacher["recipe"])
        if "foundation_model" in config.teacher:
            require(
                "teach/train",
                "teacher.foundation_model",
                config.teacher["foundation_model"],
            )
        try:
            validate_role_recipe_feasibility(config, "teacher")
            from workflow_backends.teacher.training import (
                preflight_teacher_train_capability,
            )

            preflight_teacher_train_capability(
                backend=config.teacher["backend"],
                cli=config.teacher.get("cli"),
                recipe_path=config.root / config.teacher["recipe"],
                model_shape=config.teacher["model"],
                policy=config.teacher["policy"],
                foundation_asset_present="foundation_model" in config.teacher,
            )
        except ValueError as exc:
            raise ValueError(f"teach/train external input teacher.recipe: {exc}") from exc

    if "teach/screen" in suffix_set:
        field = "student_train_bootstrap" if config.iter_index == 0 else "fp_data"
        for index, path in enumerate(config.raw[field]):
            require("teach/screen", f"{field}[{index}]", path)

    student_train_blocks = tuple(
        block for block in suffix if block.startswith("r") and block.endswith("/train")
    )
    if student_train_blocks:
        owner = student_train_blocks[0]
        require(owner, "student.recipe", config.student["recipe"])
        for field in ("foundation_model", "foundation_restart"):
            if field in config.student:
                require(owner, f"student.{field}", config.student[field])
        try:
            validate_role_recipe_feasibility(config, "student")
        except ValueError as exc:
            raise ValueError(f"{owner} external input student.recipe: {exc}") from exc

    sample_blocks = tuple(
        block for block in suffix if block.startswith("r") and block.endswith("/sample")
    )
    if sample_blocks:
        owner = sample_blocks[0]
        sampler = config.sampler
        sample = config.raw["sample"]
        if "condition_groups" in sample:
            plan = config.sampling_plan
            if plan is None or not hasattr(plan, "tasks_for_round"):
                raise ValueError("explicit sampling preflight requires its resolved task plan")
            round_indices = tuple(
                int(block.split("/", 1)[0][1:]) for block in sample_blocks
            )
            active_group_ids = {
                task.condition_group_id
                for round_index in round_indices
                for task in plan.tasks_for_round(round_index)
            }
            active_role_ids = {
                role_id
                for round_index in round_indices
                for role_id, _group_id in plan.selection_plan_for_round(
                    round_index
                ).role_to_group
            }
            recipe_digests: dict[str, str] = {}
            guest_digests: dict[str, str] = {}
            guest_templates = {}
            from workflow_core.structures.formats import read_extxyz_structures

            for group_id, group in sample["condition_groups"].items():
                if group_id not in active_group_ids or group.get("kind") in {"neb", "relax"}:
                    continue
                if group.get('kind') == 'widom':
                    field = f'sample.condition_groups.{group_id}.guest'
                    guest_path = require(owner, field, group['guest'])
                    guests = read_extxyz_structures(guest_path, allow_empty=False)
                    guest_templates[group_id] = guests
                    guest_digests[group_id] = hashlib.sha256(guest_path.read_bytes()).hexdigest()
                    continue
                if group.get('kind') == 'gcmc_md':
                    field = f'sample.condition_groups.{group_id}.mc.guest'
                    guest_path = require(owner, field, group['mc']['guest'])
                    guests = read_extxyz_structures(guest_path, allow_empty=False)
                    if (len(guests) != 1 or not len(guests[0]) or guests[0].constraints
                        or not np.isfinite(guests[0].positions).all()
                        or any(key in guests[0].arrays for key in ('dice_atom_id', 'dice_molecule_id'))):
                        raise ValueError(f'{owner} external input {field}: requires one finite unconstrained insertion template without persistent IDs')
                    if not set(guests[0].get_chemical_symbols()) <= set(config.student['type_map']):
                        raise ValueError(f'{owner} external input {field}: guest species outside student.type_map')
                    if not np.isfinite(guests[0].get_masses()).all() or np.any(guests[0].get_masses() <= 0):
                        raise ValueError(f'{owner} external input {field}: guest masses must be finite and positive')
                    guest_digests[group_id] = hashlib.sha256(guest_path.read_bytes()).hexdigest()
                    guest_templates[group_id] = guests[0]
                    if 'md' in group:
                        recipe_path = require(owner, f'sample.condition_groups.{group_id}.md.recipe', group['md']['recipe'])
                        recipe_digests[group_id] = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
                    continue
                recipe_path = require(
                    owner,
                    f"sample.condition_groups.{group_id}.recipe",
                    group["recipe"],
                )
                recipe_digests[group_id] = hashlib.sha256(
                    recipe_path.read_bytes()
                ).hexdigest()
            from workflow_core.structures.formats import read_extxyz_structures

            neb_tasks_by_role: dict[str, list[object]] = {}
            relax_tasks_by_role: dict[str, list[object]] = {}
            for round_index in round_indices:
                for task in plan.tasks_for_round(round_index):
                    if task.kind == "neb":
                        neb_tasks_by_role.setdefault(
                            task.sampling_role_id, []
                        ).append(task)
                    elif task.kind == "relax":
                        relax_tasks_by_role.setdefault(
                            task.sampling_role_id, []
                        ).append(task)

            for role_id, seed in sample["seeds"].items():
                if role_id not in active_role_ids:
                    continue
                field = f"sample.seeds.{role_id}.source"
                seed_path = require(owner, field, seed["source"])
                try:
                    frames = read_extxyz_structures(seed_path, allow_empty=False)
                except (OSError, TypeError, ValueError) as exc:
                    raise ValueError(f"{owner} external input {field}: invalid first frame") from exc
                if role_id in neb_tasks_by_role:
                    if len(frames) != 2:
                        raise ValueError(
                            f"{owner} external input {field}: NEB requires exactly two canonical endpoint frames"
                        )
                    initial, final = frames
                    if len(initial) != len(final):
                        raise ValueError(
                            f"{owner} external input {field}: NEB endpoints must have identical atom counts"
                        )
                    if initial.get_chemical_symbols() != final.get_chemical_symbols():
                        raise ValueError(
                            f"{owner} external input {field}: NEB endpoints must have identical element order"
                        )
                    if not np.array_equal(initial.pbc, final.pbc):
                        raise ValueError(
                            f"{owner} external input {field}: NEB endpoints must have identical PBC"
                        )
                    if not np.array_equal(initial.cell.array, final.cell.array):
                        raise ValueError(
                            f"{owner} external input {field}: variable-cell NEB is unsupported"
                        )
                    if not all(
                        np.isfinite(values).all()
                        for values in (
                            initial.positions,
                            final.positions,
                            initial.cell.array,
                            final.cell.array,
                        )
                    ):
                        raise ValueError(
                            f"{owner} external input {field}: NEB endpoint geometry must be finite"
                        )
                    for task in neb_tasks_by_role[role_id]:
                        if any(index >= len(initial) for index in task.neb.fixed_atoms):
                            raise ValueError(
                                f"sample.condition_groups.{task.condition_group_id}.fixed_atoms "
                                "contains an index outside the shared endpoint atom order"
                            )
                        fixed_atoms = tuple(task.neb.fixed_atoms)
                        if fixed_atoms and not np.array_equal(
                            initial.positions[list(fixed_atoms)],
                            final.positions[list(fixed_atoms)],
                        ):
                            raise ValueError(
                                f"sample.condition_groups.{task.condition_group_id}.fixed_atoms "
                                "requires identical endpoint coordinates"
                            )
                        from ase.constraints import FixAtoms

                        masks: list[tuple[int, ...] | None] = []
                        for endpoint in (initial, final):
                            if not endpoint.constraints:
                                masks.append(None)
                            elif (
                                len(endpoint.constraints) == 1
                                and isinstance(endpoint.constraints[0], FixAtoms)
                            ):
                                masks.append(
                                    tuple(
                                        int(index)
                                        for index in endpoint.constraints[0].get_indices()
                                    )
                                )
                            else:
                                raise ValueError(
                                    f"sample.condition_groups.{task.condition_group_id}.fixed_atoms "
                                    "is incompatible with endpoint constraints"
                                )
                        if any(mask is not None for mask in masks) and any(
                            mask != fixed_atoms for mask in masks
                        ):
                            raise ValueError(
                                f"sample.condition_groups.{task.condition_group_id}.fixed_atoms "
                                "is incompatible with endpoint constraints"
                            )
                if role_id in relax_tasks_by_role:
                    from ase.constraints import FixAtoms

                    for task in relax_tasks_by_role[role_id]:
                        for frame in frames:
                            if any(index >= len(frame) for index in task.relax.fixed_atoms):
                                raise ValueError(
                                    f"sample.condition_groups.{task.condition_group_id}.fixed_atoms "
                                    "contains an index outside the seed atom order"
                                )
                            if not np.isfinite(frame.positions).all() or not np.isfinite(frame.cell.array).all():
                                raise ValueError(
                                    f"{owner} external input {field}: relax geometry must be finite"
                                )
                            if frame.constraints and (
                                len(frame.constraints) != 1
                                or not isinstance(frame.constraints[0], FixAtoms)
                                or tuple(int(index) for index in frame.constraints[0].get_indices())
                                != tuple(task.relax.fixed_atoms)
                            ):
                                raise ValueError(
                                    f"sample.condition_groups.{task.condition_group_id}.fixed_atoms "
                                    "is incompatible with seed constraints"
                                )
                    checked_frames = frames
                else:
                    checked_frames = frames if "max_frames" in seed else frames[:1]
                widom_tasks = tuple(task for round_index in round_indices for task in plan.tasks_for_round(round_index)
                                    if task.kind == 'widom' and task.sampling_role_id == role_id)
                swap_tasks = tuple(task for round_index in round_indices for task in plan.tasks_for_round(round_index)
                                   if task.kind == 'md_mc' and task.sampling_role_id == role_id)
                if swap_tasks:
                    from workflow_core.structures.md_mc import validate_seed
                    for frame in checked_frames:
                        for task in swap_tasks:
                            validate_seed(frame, task.md_mc, config.student['type_map'])
                if widom_tasks:
                    from workflow_core.structures.widom import validate_inputs
                    for frame in checked_frames:
                        for task in widom_tasks:
                            validate_inputs(frame, guest_templates[task.condition_group_id], task.widom, config.student['type_map'])
                gcmc_tasks = tuple(task for round_index in round_indices for task in plan.tasks_for_round(round_index)
                                   if task.kind == 'gcmc_md' and task.sampling_role_id == role_id)
                if gcmc_tasks:
                    from workflow_core.structures.gcmc_membership import validate_membership
                    for frame in checked_frames:
                        if not np.isfinite(frame.get_masses()).all() or np.any(frame.get_masses() <= 0):
                            raise ValueError(f'{owner} external input {field}: GCMC masses must be finite and positive')
                        if (not len(frame) or frame.constraints or not np.all(frame.pbc)
                            or not np.isfinite(frame.cell.array).all() or np.linalg.matrix_rank(frame.cell.array) != 3
                            or not np.isfinite(frame.positions).all()
                            or not np.isfinite(frame.get_momenta()).all()):
                            raise ValueError(f'{owner} external input {field}: GCMC requires a finite unconstrained full-rank periodic seed')
                        if any(key in frame.arrays for key in ('dice_atom_id', 'dice_molecule_id')):
                            for task in gcmc_tasks:
                                validate_membership(frame, guest_templates[task.condition_group_id].numbers)
                atom_counts = {len(frame) for frame in checked_frames}
                if len(atom_counts) != 1:
                    raise ValueError(
                        f"{owner} external input {field}: frame-sampled source must have one atom count"
                    )
                unknown = sorted(
                    {
                        symbol
                        for frame in checked_frames
                        for symbol in frame.get_chemical_symbols()
                    }
                    - set(config.student["type_map"])
                )
                if unknown:
                    raise ValueError(
                        f"{owner} external input {field}: sampled source contains symbols "
                        f"outside student.type_map: {unknown}"
                    )
            for round_index in round_indices:
                seen: set[tuple[object, ...]] = set()
                for task in plan.tasks_for_round(round_index):
                    program = (
                        None
                        if task.program is None
                        else task.program.segments
                    )
                    key = (
                        task.sampling_role_id,
                        recipe_digests.get(task.condition_group_id),
                        task.kind,
                        task.temperature_k,
                        task.pressure_bar,
                        program,
                        task.steps,
                        task.gcmc,
                        task.widom,
                        task.md_mc,
                        task.cycles,
                        guest_digests.get(task.condition_group_id),
                    )
                    if key in seen:
                        raise ValueError(
                            f"{owner} explicit plan contains a duplicate effective native task"
                        )
                    seen.add(key)
        elif "recipe" in sampler:
            require(owner, "sampler.recipe", sampler["recipe"])
        else:
            for ensemble, path in sampler["recipes"].items():
                require(owner, f"sampler.recipes.{ensemble}", path)
        if "systems" in sample:
            for system_id, system in sample["systems"].items():
                require(owner, f"sample.systems.{system_id}.seed", system["seed"])
                if "plumed" in system:
                    require(owner, f"sample.systems.{system_id}.plumed", system["plumed"])
        try:
            validate_sampler_recipe_feasibility(
                config,
                condition_group_ids=(
                    active_group_ids if "condition_groups" in sample else None
                ),
            )
        except ValueError as exc:
            raise ValueError(f"{owner} external input sampler.recipe: {exc}") from exc

    if "fp/label" in suffix_set:
        label = config.raw["fp"]["label"]
        require("fp/label", "fp.label.recipe", label["recipe"])
        from workflow_backends.first_principles.recipe_evidence import (
            fp_recipe_input_paths,
        )

        try:
            paths = fp_recipe_input_paths(
                run_root=config.root,
                calculator=label["calculator"],
                recipe=label["recipe"],
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"fp/label external input fp.label.recipe: {exc}") from exc
        for path in paths:
            value = Path(path)
            if value.is_absolute():
                try:
                    value = value.relative_to(config.root)
                except ValueError as exc:
                    raise ValueError(
                        "fp/label external input fp.label.recipe escapes run root"
                    ) from exc
            require("fp/label", "fp.label.recipe/assets", str(value))


def build_lifecycle_plan(config: RunConfig) -> LifecyclePlan:
    block_ids = ["teach/train", "teach/screen"]
    if _route(config).has_teach_scan:
        block_ids.append("teach/scan")
    for index in range(config.round_count):
        block_ids.extend((f"r{index}/train", f"r{index}/sample", f"r{index}/screen"))
    block_ids.append("fp")
    if "label" in config.raw["fp"]:
        block_ids.append("fp/label")
    iter_root = config.root / f"iter{config.iter_index}"
    return LifecyclePlan(tuple(LifecycleBlock(block_id, iter_root / block_id) for block_id in block_ids))


def required_operation_roles(config: RunConfig) -> tuple[str, ...]:
    """Return only operation roles selected by this concrete lifecycle model shape.

    Supported single/single routes use phase-specific model-owning GMM
    operations. Committee routes retain only backend-native train/label/sample roles.
    """
    ordered: list[str] = []
    for block in build_lifecycle_plan(config).blocks:
        for role in block_operation_roles(config, block.block_id):
            if role not in ordered:
                ordered.append(role)
    return tuple(ordered)


def block_operation_roles(config: RunConfig, block: str) -> tuple[str, ...]:
    """Map one public lifecycle block to its exact selected operation roles."""
    if block.startswith("r") and block.endswith("/sample") and hasattr(
        config.sampling_plan, "tasks_for_round"
    ):
        round_index = int(block.split("/", 1)[0][1:])
        tasks = config.sampling_plan.tasks_for_round(round_index)
        has_neb = any(task.kind == "neb" for task in tasks)
        has_relax = any(task.kind == "relax" for task in tasks)
        if has_neb or has_relax:
            route = _route(config)
            roles: list[str] = []
            if route.sample_compression_role is not None:
                roles.append(route.sample_compression_role)
            if any(task.kind not in {"neb", "relax"} for task in tasks):
                roles.append(route.sampler_operation_role)
            if has_neb:
                roles.append(f"{route.student_operation_prefix}.neb")
            if has_relax:
                roles.append(f"{route.student_operation_prefix}.relax")
            return tuple(roles)
    if block == "fp/label":
        calculator = str(config.raw["fp"]["label"]["calculator"]).lower()
        return (f"first_principles.{calculator}.label",)
    if block == "fp" and config.raw["fp"]["policy"] in {
        "JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
    }:
        route = _route(config)
        if (
            config.raw["fp"]["policy"] == "ELEMENT_DIRECT"
            and config.teacher["model"] == "committee"
        ):
            return route.operation_roles(block)
        return (f"{route.teacher_operation_prefix}.gmm_score",)
    return _route(config).operation_roles(block)


def _route(config: RunConfig) -> RouteSpec:
    if config.route is None:
        raise ValueError("lifecycle launch requires a validated configuration with a resolved route")
    return config.route


def _select_environment(explicit: str | Path | None) -> Path | None:
    candidate = explicit if explicit is not None else os.environ.get("DICE_ENVIRONMENT")
    return None if candidate in (None, "") else Path(candidate).resolve()


def _select_execution(explicit: str | Path | None) -> Path | None:
    candidate = explicit if explicit is not None else os.environ.get("DICE_EXECUTION")
    return None if candidate in (None, "") else Path(candidate).resolve()


def _validate_execution_profile(
    config: RunConfig,
    profile: ExecutionProfile,
    *,
    operation_roles: frozenset[str] | None = None,
) -> None:
    import json

    for binding in profile.bindings:
        if (
            operation_roles is not None
            and binding.operation_role not in operation_roles
        ):
            continue
        resource = profile.resource_classes[binding.resource_class]
        if not profile.historical:
            continue
        if resource.launcher.kind != "torchrun":
            continue
        student = config.student
        if student.get("backend") != "DeepMD" or student.get("cli") != "pt":
            raise ValueError("torchrun execution requires the DeepMD-PT DPA4C student backend")
        recipe_path = config.root / str(student["recipe"])
        try:
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("torchrun execution cannot validate the DPA4C recipe") from exc
        model = recipe.get("model") if isinstance(recipe, dict) else None
        descriptor = model.get("descriptor") if isinstance(model, dict) else None
        if not isinstance(descriptor, dict) or descriptor.get("type") != "dpa4c":
            raise ValueError(
                "torchrun execution is supported only for an actual DPA4C student recipe"
            )
        pool = profile.target_pools[binding.target_pool]
        if len(pool.targets) != 2:
            raise ValueError("torchrun execution requires exactly two bound targets")


def _build_execution_plans(
    config: RunConfig,
    *,
    block_roles: dict[str, tuple[str, ...]],
    profile: ExecutionProfile | None,
) -> dict[str, tuple[PlannedExecution, ...]]:
    if profile is None:
        return {}
    result: dict[str, tuple[PlannedExecution, ...]] = {}
    for block_id, roles in block_roles.items():
        selected: list[tuple[str, ExecutionBinding]] = []
        for role in roles:
            matching = tuple(
                binding for binding in profile.bindings if binding.operation_role == role
            )
            if not matching:
                continue
            if len(matching) != 1:
                raise ValueError(
                    f"execution profile requires one selected resource class for role {role!r}"
                )
            selected.append((role, matching[0]))
        if not selected:
            continue
        coordinates = {
            (binding.resource_class, binding.target_pool)
            for _role, binding in selected
        }
        if len(selected) > 1:
            if len(coordinates) != 1:
                raise ValueError(
                    "public lifecycle block requires one homogeneous execution "
                    f"binding: {block_id}"
                )
            _resource_class, target_pool = next(iter(coordinates))
            pool = profile.target_pools[target_pool]
            if any(profile.targets[target].adapter != "slurm" for target in pool.targets):
                raise ValueError(
                    "multiple operation roles may share one execution binding only "
                    f"for a public Slurm block: {block_id}"
                )
        role, binding = selected[0]
        task_ids = _execution_task_inventory(config, block_id=block_id, role=role)
        role_identity = "\0".join(selected_role for selected_role, _binding in selected)
        attempt_id = "attempt-" + hashlib.sha256(
            f"{profile.digest}\0{config.iter_index}\0{block_id}\0{role_identity}".encode(
                "utf-8"
            )
        ).hexdigest()[:20]
        result[block_id] = plan_execution_units(
            block_id=block_id,
            operation_role=role,
            resource_class=binding.resource_class,
            task_ids=task_ids,
            profile=profile,
            attempt_id=attempt_id,
        )
    return result


def _execution_task_inventory(
    config: RunConfig,
    *,
    block_id: str,
    role: str,
) -> tuple[str, ...]:
    if role.endswith(".train") and block_id == "teach/train":
        count = int(config.teacher.get("members", 1))
        return tuple(
            "teacher" if count == 1 else f"teacher_{index:02d}"
            for index in range(1, count + 1)
        )
    if role.endswith(".train") and block_id.endswith("/train"):
        count = int(config.student.get("members", 1))
        return tuple(
            "student" if count == 1 else f"student_{index:02d}"
            for index in range(1, count + 1)
        )
    if block_id.endswith("/sample"):
        plan = config.sampling_plan
        if plan is None:
            raise ValueError("sample execution inventory requires a resolved plan")
        round_index = int(block_id.split("/", 1)[0][1:])
        tasks = (
            plan.tasks_for_round(round_index)
            if bool(getattr(plan, "explicit", False))
            else plan.systems
        )
        return tuple(f"task_{index:06d}" for index, _ in enumerate(tasks))
    return ("operation",)
