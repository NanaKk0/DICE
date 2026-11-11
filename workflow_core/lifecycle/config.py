from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Any

from ase.data import atomic_numbers

from workflow_core.common.widom import WidomSamplingParameters
from workflow_core.common.md_mc import IdentitySwapParameters
from workflow_core.common.gcmc import GcmcMdParameters, GcmcSamplingParameters

from workflow_core.lifecycle.routes import RouteSpec, resolve_route
from workflow_core.model_families import deepmd_recipe_model_family
from workflow_artifacts.lifecycle.sample_conditions import (
    SamplingPlan,
    SystemSamplingCondition,
)
from workflow_core.lifecycle.sampling_plan import (
    CellProgramEndpoint,
    ExplicitSamplingPlan,
    NebSamplingParameters,
    RelaxSamplingParameters,
    SamplingConditionGroup,
    SamplingRound,
    SeedRole,
    SelectionPlan,
    TemperatureProgram,
    TemperatureProgramSegment,
)


_REQUIRED_TOP_LEVEL_FIELDS = frozenset({"iter", "teacher_train_bootstrap", "student_train_bootstrap", "fp_data", "teacher", "student", "sampler", "sample", "screen", "fp"})
_TOP_LEVEL_FIELDS = _REQUIRED_TOP_LEVEL_FIELDS | {"visualize"}
_SYSTEM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SYMMETRIC_STRAIN_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass(frozen=True, slots=True)
class RunConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    route: RouteSpec | None = None
    sampling_plan: SamplingPlan | ExplicitSamplingPlan | None = None

    @property
    def iter_index(self) -> int:
        return int(self.raw["iter"])

    @property
    def visualize_enabled(self) -> bool:
        return self.raw.get("visualize") is not False

    @property
    def teacher(self) -> dict[str, Any]:
        return self.raw["teacher"]

    @property
    def student(self) -> dict[str, Any]:
        return self.raw["student"]

    @property
    def sampler(self) -> dict[str, Any]:
        return self.raw["sampler"]

    @property
    def round_count(self) -> int:
        if isinstance(self.sampling_plan, ExplicitSamplingPlan):
            return int(self.sampling_plan.round_count)
        sample = self.raw["sample"]
        return (
            int(sample["round_count"])
            if "round_count" in sample
            else len(sample["steps"])
        )

    @property
    def selection_plan(self) -> SelectionPlan:
        if isinstance(self.sampling_plan, ExplicitSamplingPlan):
            return self.sampling_plan.selection_plan
        systems = tuple(self.raw["sample"]["systems"])
        return SelectionPlan.from_seeds(
            tuple(SeedRole(system_id, system_id) for system_id in systems)
        )


def load_run_config(path: str | Path) -> RunConfig:
    run_path = Path(path).resolve()
    try:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read run.json at {run_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"run.json at {run_path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("run.json must be a JSON object")
    unknown = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    missing = sorted(_REQUIRED_TOP_LEVEL_FIELDS - set(payload))
    if unknown:
        raise ValueError(f"run.json contains unsupported field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"run.json is missing required field(s): {', '.join(missing)}")
    visualize = payload.get("visualize", True)
    if not isinstance(visualize, bool):
        raise ValueError("visualize must be a boolean")
    if visualize:
        payload.pop("visualize", None)
    config = RunConfig(path=run_path, root=run_path.parent, raw=payload)
    route, sampling_plan = _validate_config(config)
    return replace(config, route=route, sampling_plan=sampling_plan)


def _validate_config(config: RunConfig) -> tuple[RouteSpec, SamplingPlan | ExplicitSamplingPlan]:
    raw = config.raw
    if not isinstance(raw["iter"], int) or raw["iter"] < 0:
        raise ValueError("iter must be a non-negative integer")
    _validate_path_list(config, raw["teacher_train_bootstrap"], "teacher_train_bootstrap", nonempty=True)
    _validate_path_list(config, raw["student_train_bootstrap"], "student_train_bootstrap", nonempty=raw["iter"] == 0)
    _validate_path_list(config, raw["fp_data"], "fp_data", nonempty=raw["iter"] > 0)
    if raw["iter"] > 0 and raw["student_train_bootstrap"]:
        raise ValueError("student_train_bootstrap is valid only for iter: 0")
    _validate_teacher(config, raw["teacher"])
    _validate_student(config, raw["student"])
    _validate_sampler(config, raw["sampler"])
    sampling_plan = _validate_sample(config, raw["sample"], sampler_backend=raw["sampler"]["backend"])
    _validate_sampler_plan(raw["sampler"], sampling_plan)
    _validate_screen(
        raw["screen"],
        rounds=(
            sampling_plan.round_count
            if isinstance(sampling_plan, ExplicitSamplingPlan)
            else len(raw["sample"]["steps"])
        ),
        teacher=raw["teacher"],
        student=raw["student"],
    )
    _validate_fp(config, raw["fp"])
    teacher_types = set(raw["teacher"]["type_map"])
    student_types = set(raw["student"]["type_map"])
    if not student_types <= teacher_types:
        raise ValueError("teacher type_map must cover the student type_map")
    expected_sampler = "LAMMPS" if raw["student"]["backend"] == "DeepMD" else "GPUMD"
    if raw["sampler"]["backend"] != expected_sampler:
        raise ValueError(f"student backend {raw['student']['backend']} requires sampler backend {expected_sampler}")
    return _validate_model_shape_route(config), sampling_plan


def _validate_teacher(config: RunConfig, value: Any) -> None:
    if isinstance(value, dict):
        retired = [field for field in ("trust", "cutoff") if field in value]
        if retired:
            fields = ", ".join(f"teacher.{field}" for field in retired)
            raise ValueError(
                f"{fields} is retired; configure teacher confidence with screen.teacher_p.trust"
            )
    allowed = {"backend", "model", "members", "type_map", "recipe", "policy", "valid_frac", "lo", "hi", "cli", "foundation_model", "train_options"}
    _mapping_fields(value, allowed, "teacher")
    _enum(value, "backend", {"MACE", "DeepMD"}, "teacher")
    _normalize_role_train_options(value, role="teacher")
    _enum(value, "model", {"single", "committee"}, "teacher")
    _type_map(value, "teacher")
    _relative_path(config, value.get("recipe"), "teacher.recipe")
    _enum(value, "policy", {"from_scratch", "foundation_finetune"}, "teacher")
    _fraction(value.get("valid_frac"), "teacher.valid_frac")
    if value["backend"] == "DeepMD":
        _enum(value, "cli", {"tf", "pt"}, "teacher")
    elif "cli" in value:
        raise ValueError("teacher.cli is only valid for DeepMD")
    _foundation_path(config, value, "teacher")
    if value["model"] == "single":
        if "members" in value or "lo" in value or "hi" in value:
            raise ValueError("single teacher forbids members, lo, and hi")
    else:
        _members(value, "teacher")
        _band(value, "teacher")


def _validate_student(config: RunConfig, value: Any) -> None:
    allowed = {
        "backend",
        "model",
        "members",
        "type_map",
        "recipe",
        "policy",
        "valid_frac",
        "cli",
        "compress",
        "foundation_model",
        "foundation_restart",
        "train_options",
        "include_teacher_train_data",
    }
    _mapping_fields(value, allowed, "student")
    _enum(value, "backend", {"DeepMD", "NEP"}, "student")
    _normalize_role_train_options(value, role="student")
    include_teacher_train_data = value.get("include_teacher_train_data", False)
    if not isinstance(include_teacher_train_data, bool):
        raise ValueError("student.include_teacher_train_data must be a boolean")
    if include_teacher_train_data:
        value["include_teacher_train_data"] = True
    else:
        value.pop("include_teacher_train_data", None)
    _enum(value, "model", {"single", "committee"}, "student")
    _type_map(value, "student")
    _relative_path(config, value.get("recipe"), "student.recipe")
    _enum(value, "policy", {"from_scratch", "foundation_finetune"}, "student")
    _fraction(value.get("valid_frac"), "student.valid_frac")
    if value["backend"] == "DeepMD":
        _enum(value, "cli", {"tf", "pt"}, "student")
        compress = value.get("compress", False)
        if not isinstance(compress, bool):
            raise ValueError("student.compress must be a boolean")
        value["compress"] = compress
    elif "cli" in value:
        raise ValueError("student.cli is forbidden for NEP")
    elif "compress" in value:
        raise ValueError("student.compress is valid only for DeepMD")
    _foundation_path(config, value, "student")
    if value["backend"] == "NEP" and value["policy"] == "foundation_finetune":
        _relative_path(config, value.get("foundation_restart"), "student.foundation_restart")
    elif "foundation_restart" in value:
        raise ValueError("student.foundation_restart is valid only for NEP foundation_finetune")
    if value["model"] == "committee":
        _members(value, "student")
    elif "members" in value:
        raise ValueError("single student forbids members")


def _validate_sampler(config: RunConfig, value: Any) -> None:
    _mapping_fields(value, {"backend", "recipe", "recipes"}, "sampler")
    _enum(value, "backend", {"LAMMPS", "GPUMD"}, "sampler")
    if "recipe" in value and "recipes" in value:
        raise ValueError("sampler requires exactly one of recipe or recipes in legacy mode")
    if "recipe" not in value and "recipes" not in value:
        return
    if "recipe" in value:
        _relative_path(config, value["recipe"], "sampler.recipe")
        return
    if value["backend"] != "LAMMPS":
        raise ValueError("sampler.recipes is valid only for LAMMPS")
    recipes = value["recipes"]
    _mapping_fields(recipes, {"nvt", "npt"}, "sampler.recipes")
    if not recipes:
        raise ValueError("sampler.recipes must not be empty")
    for ensemble, recipe in recipes.items():
        _relative_path(config, recipe, f"sampler.recipes.{ensemble}")


def _validate_sampler_plan(
    value: dict[str, Any], plan: SamplingPlan | ExplicitSamplingPlan
) -> None:
    if isinstance(plan, ExplicitSamplingPlan):
        if "recipe" in value or "recipes" in value:
            raise ValueError(
                "explicit sampling requires sampler.backend only and group-local recipes"
            )
        if value["backend"] == "GPUMD":
            unsupported = tuple(
                group.kind
                for group in plan.condition_groups
                if group.kind not in {"neb", "relax", "nvt", "npt"}
            )
            if unsupported:
                raise ValueError(
                    "GPUMD explicit sampling supports only neb, relax, nvt, and npt "
                    "condition groups"
                )
        return
    if "recipe" not in value and "recipes" not in value:
        raise ValueError("legacy sampling requires exactly one of sampler.recipe or sampler.recipes")
    ensembles = set(plan.ensembles)
    if "recipes" in value:
        keys = set(value["recipes"])
        if keys != ensembles:
            raise ValueError(
                "sampler.recipes keys must exactly match effective ensembles "
                f"{sorted(ensembles)}"
            )
    elif value["backend"] == "LAMMPS" and len(ensembles) > 1:
        raise ValueError("heterogeneous LAMMPS ensemble plans require sampler.recipes")


def _validate_sample(
    config: RunConfig, value: Any, *, sampler_backend: Any
) -> SamplingPlan | ExplicitSamplingPlan:
    if not isinstance(value, dict):
        raise ValueError("sample must be an object")
    explicit_fields = {"seeds", "condition_groups", "round_count", "rounds"}
    legacy_fields = {"systems", "steps", "ensemble", "temperatures_k", "pressures_bar"}
    has_explicit = bool(set(value) & explicit_fields)
    has_legacy = bool(set(value) & legacy_fields)
    if has_explicit and has_legacy:
        raise ValueError("sample cannot mix explicit seeds/condition_groups with legacy systems/steps")
    if has_explicit:
        return _resolve_explicit_sampling_plan(config, value)
    _mapping_fields(value, legacy_fields, "sample")
    systems = value.get("systems")
    if not isinstance(systems, dict) or not systems:
        raise ValueError("sample.systems must be a non-empty object")
    plumed_systems: list[str] = []
    for system_id, system in systems.items():
        if not isinstance(system_id, str) or not _SYSTEM_ID.fullmatch(system_id):
            raise ValueError(f"invalid system id {system_id!r}")
        _mapping_fields(
            system,
            {"seed", "plumed", "ensemble", "temperatures_k", "pressures_bar"},
            f"sample.systems.{system_id}",
        )
        _relative_path(config, system.get("seed"), f"sample.systems.{system_id}.seed")
        if "plumed" in system:
            _relative_path(config, system["plumed"], f"sample.systems.{system_id}.plumed")
            plumed_systems.append(system_id)
    if plumed_systems and sampler_backend != "LAMMPS":
        raise ValueError("PLUMED is valid only for the LAMMPS sampler")
    if plumed_systems and len(plumed_systems) != len(systems):
        raise ValueError("PLUMED must be supplied for all systems or none")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps or not all(isinstance(step, int) and step > 0 for step in steps):
        raise ValueError("sample.steps must be a non-empty positive-integer array")
    return _resolve_sampling_plan(value)


def _resolve_explicit_sampling_plan(
    config: RunConfig, value: dict[str, Any]
) -> ExplicitSamplingPlan:
    _mapping_fields(value, {"seeds", "condition_groups", "round_count", "rounds"}, "sample")
    if "round_count" in value and "rounds" in value:
        raise ValueError("sample.round_count and sample.rounds are mutually exclusive")
    raw_rounds = value.get("rounds")
    if raw_rounds is not None:
        if not isinstance(raw_rounds, list) or not raw_rounds:
            raise ValueError("sample.rounds must be a non-empty array")
        round_count = len(raw_rounds)
    else:
        round_count = value.get("round_count")
        if isinstance(round_count, bool) or not isinstance(round_count, int) or round_count <= 0:
            raise ValueError("sample.round_count must be a positive integer")
    raw_seeds = value.get("seeds")
    if not isinstance(raw_seeds, dict) or not raw_seeds:
        raise ValueError("sample.seeds must be a non-empty object")
    seeds: list[SeedRole] = []
    for role_id, declaration in raw_seeds.items():
        if not isinstance(role_id, str) or not _SYSTEM_ID.fullmatch(role_id):
            raise ValueError(f"invalid sampling seed id {role_id!r}")
        _mapping_fields(
            declaration,
            {"source", "selection_group", "max_frames"},
            f"sample.seeds.{role_id}",
        )
        _relative_path(config, declaration.get("source"), f"sample.seeds.{role_id}.source")
        if "max_frames" in declaration:
            max_frames = declaration["max_frames"]
            if (
                isinstance(max_frames, bool)
                or not isinstance(max_frames, int)
                or max_frames <= 0
            ):
                raise ValueError(
                    f"sample.seeds.{role_id}.max_frames must be a positive integer"
                )
        selection_group = declaration.get("selection_group", role_id)
        if (
            not isinstance(selection_group, str)
            or not _SYSTEM_ID.fullmatch(selection_group)
        ):
            raise ValueError(
                f"sample.seeds.{role_id}.selection_group must be a filename-safe identifier"
            )
        seeds.append(SeedRole(role_id, selection_group))

    raw_groups = value.get("condition_groups")
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError("sample.condition_groups must be a non-empty object")
    seed_ids = tuple(seed.sampling_role_id for seed in seeds)
    assigned_roles: set[str] = set()
    groups: list[SamplingConditionGroup] = []
    duplicate_keys: set[tuple[object, ...]] = set()
    for group_id, declaration in raw_groups.items():
        prefix = f"sample.condition_groups.{group_id}"
        if not isinstance(group_id, str) or not _SYSTEM_ID.fullmatch(group_id):
            raise ValueError(f"invalid condition group id {group_id!r}")
        if not isinstance(declaration, dict):
            raise ValueError(f"{prefix} must be an object")
        kind = declaration.get("kind")
        if kind not in {"neb", "relax", "nvt", "npt", "nvt_program", "nvt_cell_program", "npt_program", "gcmc_md", "widom", "md_mc"}:
            raise ValueError(
                f"{prefix}.kind must be one of "
                "['gcmc_md', 'md_mc', 'neb', 'npt', 'npt_program', 'nvt', 'nvt_cell_program', 'nvt_program', 'relax', 'widom']"
            )
        allowed = {"kind", "seeds"}
        if kind == "widom":
            allowed |= {"guest", "frames", "max_attempts", "random_seed", "overlap"}
        elif kind == "gcmc_md":
            allowed |= {"temperatures_k", "cycles", "mc", "md", "random_seed", "limits"}
        elif kind == "neb":
            allowed |= {
                "images",
                "fmax_ev_a",
                "max_steps",
                "include_history",
                "init",
                "method",
                "climb",
                "climb_preconverge_fmax_ev_a",
                "fixed_atoms",
                "relax",
            }
        elif kind == "relax":
            allowed |= {
                "fmax_ev_a",
                "max_steps",
                "max_step_a",
                "include_history",
                "history_stride",
                "accept_unconverged",
                "fixed_atoms",
            }
        else:
            allowed.add("recipe")
            allowed |= (
                {"segments"}
                if kind in {"nvt_program", "nvt_cell_program", "npt_program"}
                else {"temperatures_k", "steps"}
            )
        if kind in {"npt", "npt_program"}:
            allowed.add("pressures_bar")
        if kind == 'md_mc':
            allowed |= {'ensemble', 'mc'}
            if declaration.get('ensemble') == 'npt':
                allowed.add('pressures_bar')
        _mapping_fields(declaration, allowed, prefix)
        if kind not in {"neb", "relax", "gcmc_md", "widom"}:
            _relative_path(config, declaration.get("recipe"), f"{prefix}.recipe")
        raw_assignment = declaration.get("seeds")
        if raw_assignment is None:
            role_ids = seed_ids
        else:
            if (
                not isinstance(raw_assignment, list)
                or not raw_assignment
                or any(not isinstance(item, str) for item in raw_assignment)
                or len(set(raw_assignment)) != len(raw_assignment)
            ):
                raise ValueError(f"{prefix}.seeds must be a non-empty unique seed-id array")
            unknown = sorted(set(raw_assignment) - set(seed_ids))
            if unknown:
                raise ValueError(f"{prefix}.seeds contains unknown seed id(s): {', '.join(unknown)}")
            selected = set(raw_assignment)
            role_ids = tuple(seed_id for seed_id in seed_ids if seed_id in selected)
        if kind == "neb" and len(role_ids) != 1:
            raise ValueError(f"{prefix}.seeds must bind exactly one endpoint role")
        assigned_roles.update(role_ids)

        temperatures: tuple[float, ...] = ()
        pressures: tuple[float, ...] = ()
        program: TemperatureProgram | None = None
        neb: NebSamplingParameters | None = None
        relax_parameters: RelaxSamplingParameters | None = None
        gcmc_parameters: GcmcSamplingParameters | None = None
        widom_parameters = None
        md_mc_parameters = None
        if kind == 'md_mc':
            if declaration.get('ensemble') not in {'nvt', 'npt'}:
                raise ValueError(f'{prefix}.ensemble must be nvt or npt')
            mc = declaration.get('mc')
            _mapping_fields(mc, {'move', 'elements', 'every_md_steps', 'attempts_per_event'}, f'{prefix}.mc')
            elements = mc.get('elements')
            if not isinstance(elements, list) or any(s not in config.student['type_map'] for s in elements):
                raise ValueError(f'{prefix}.mc.elements must be covered by student.type_map')
            try:
                md_mc_parameters = IdentitySwapParameters(
                    elements=elements, every_md_steps=mc.get('every_md_steps'),
                    attempts_per_event=mc.get('attempts_per_event'), move=mc.get('move'),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{prefix}.mc: {exc}') from exc
        if kind == "widom":
            _relative_path(config, declaration.get('guest'), f'{prefix}.guest')
            overlap = declaration.get('overlap', {})
            radii = {}
            if overlap is not None:
                _mapping_fields(overlap, {'scale', 'radii_a'}, f'{prefix}.overlap')
                if overlap.get('scale', .6) is None:
                    raise ValueError(f'{prefix}.overlap.scale must be positive; use overlap=null to disable')
                radii = overlap.get('radii_a', {})
                if 'radii_a' in overlap and not radii:
                    raise ValueError(f'{prefix}.overlap.radii_a must be a complete nonempty override')
                if not isinstance(radii, dict) or any(k not in atomic_numbers for k in radii):
                    raise ValueError(f'{prefix}.overlap.radii_a must map element symbols to radii')
            widom_parameters = WidomSamplingParameters(
                declaration.get('frames'), declaration.get('max_attempts'),
                declaration.get('random_seed', 0),
                None if overlap is None else overlap.get('scale', .6), tuple(sorted(radii.items())))
            steps_by_round = ()
            normalized_conditions = ((declaration['guest'], widom_parameters),)
        elif kind == "gcmc_md":
            md_parameters = None
            if 'md' in declaration:
                md = declaration['md']
                _mapping_fields(md, {'ensemble', 'recipe', 'steps', 'every', 'pressure_bar'}, f'{prefix}.md')
                _relative_path(config, md.get('recipe'), f'{prefix}.md.recipe')
                if md.get('ensemble') != 'npt' and 'pressure_bar' in md:
                    raise ValueError(f'{prefix}.md.pressure_bar is valid only for npt')
                md_parameters = GcmcMdParameters(md.get('ensemble'), md.get('steps'), md.get('every'), md.get('pressure_bar'))
            mc = declaration.get("mc")
            _mapping_fields(mc, {"guest", "pressures_bar", "moves", "every", "eos", "translate_a", "rotate_deg", "weights", "overlap", "de_max_ev"}, f"{prefix}.mc")
            _relative_path(config, mc.get("guest"), f"{prefix}.mc.guest")
            eos = mc.get("eos")
            _mapping_fields(eos, {"kind", "phi", "tc_k", "pc_bar", "omega"}, f"{prefix}.mc.eos")
            if eos.get("kind") not in {"ideal", "fixed_phi", "pr"}:
                raise ValueError(f"{prefix}.mc.eos.kind is not yet qualified")
            if (eos['kind'] != 'fixed_phi' and 'phi' in eos) or (eos['kind'] == 'fixed_phi' and 'phi' not in eos):
                raise ValueError(f'{prefix}.mc.eos.phi is required only for fixed_phi')
            overlap = mc.get('overlap', {})
            _mapping_fields(overlap, {'scale', 'radii_a'}, f'{prefix}.mc.overlap')
            radii = overlap.get('radii_a', {})
            if not isinstance(radii, dict) or any(k not in atomic_numbers for k in radii):
                raise ValueError(f'{prefix}.mc.overlap.radii_a must map element symbols to radii')
            limits = declaration.get('limits', {})
            _mapping_fields(limits, {'max_guests'}, f'{prefix}.limits')
            names = ('insertion', 'deletion', 'translation', 'rotation', 'reinsertion')
            weights = mc.get('weights', dict(zip(names, (.25, .25, .25, .25, 0.))))
            if not isinstance(weights, dict) or set(weights) != set(names):
                raise ValueError(f'{prefix}.mc.weights requires exactly the five move probabilities')
            for field, value in (("moves", mc.get("moves")), ("every", mc.get("every", 100))):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{prefix}.mc.{field} must be a positive integer")
            temperatures = _temperature_group(declaration.get("temperatures_k"), f"{prefix}.temperatures_k")
            pressures = _pressure_group(mc.get("pressures_bar"), f"{prefix}.mc.pressures_bar")
            if not isinstance(mc.get('pressures_bar'), list) or any(p <= 0 for p in pressures):
                raise ValueError(f"{prefix}.mc.pressures_bar must be positive")
            gcmc_parameters = GcmcSamplingParameters(
                cycles_by_round=_explicit_round_steps(declaration.get("cycles"), round_count, f"{prefix}.cycles"),
                moves=mc["moves"], every=mc.get("every", 100),
                translate_a=mc.get('translate_a', .75), rotate_deg=mc.get('rotate_deg', 90.),
                eos_kind=eos['kind'], phi=eos.get('phi', 1.), weights=tuple(weights[k] for k in names),
                overlap_scale=overlap.get('scale', .6), radii_a=tuple(radii.items()),
                de_max_ev=mc.get('de_max_ev'), max_guests=limits.get('max_guests'),
                random_seed=declaration.get('random_seed'),
                md=md_parameters,
                tc_k=eos.get('tc_k'), pc_bar=eos.get('pc_bar'), omega=eos.get('omega'),
            )
            steps_by_round = ()
            normalized_conditions = tuple((t, p, mc["guest"], gcmc_parameters)
                                          for t in temperatures for p in pressures)
        elif kind == "neb":
            images = declaration.get("images")
            if (
                isinstance(images, bool)
                or not isinstance(images, int)
                or images <= 0
            ):
                raise ValueError(f"{prefix}.images must be a positive integer")
            fmax_ev_a = declaration.get("fmax_ev_a")
            if (
                isinstance(fmax_ev_a, bool)
                or not isinstance(fmax_ev_a, (int, float))
                or not math.isfinite(float(fmax_ev_a))
                or float(fmax_ev_a) <= 0.0
            ):
                raise ValueError(f"{prefix}.fmax_ev_a must be a finite positive number")
            max_steps = declaration.get("max_steps")
            if (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or max_steps <= 0
            ):
                raise ValueError(f"{prefix}.max_steps must be a positive integer")
            include_history = declaration.get("include_history", False)
            if not isinstance(include_history, bool):
                raise ValueError(f"{prefix}.include_history must be a boolean")
            init = declaration.get("init", "linear")
            if init not in {"linear", "idpp"}:
                raise ValueError(f"{prefix}.init must be one of ['idpp', 'linear']")
            method = declaration.get("method", "neb")
            if "method" in declaration and method not in {"neb", "dyneb"}:
                raise ValueError(
                    f"{prefix}.method must be one of ['dyneb', 'neb']"
                )
            climb = declaration.get("climb", False)
            if not isinstance(climb, bool):
                raise ValueError(f"{prefix}.climb must be a boolean")
            climb_preconverge_fmax_ev_a = declaration.get(
                "climb_preconverge_fmax_ev_a"
            )
            if climb_preconverge_fmax_ev_a is not None:
                if (
                    isinstance(climb_preconverge_fmax_ev_a, bool)
                    or not isinstance(climb_preconverge_fmax_ev_a, (int, float))
                    or not math.isfinite(float(climb_preconverge_fmax_ev_a))
                    or float(climb_preconverge_fmax_ev_a) <= float(fmax_ev_a)
                ):
                    raise ValueError(
                        f"{prefix}.climb_preconverge_fmax_ev_a must be finite and greater than fmax_ev_a"
                    )
                if not climb:
                    raise ValueError(
                        f"{prefix}.climb_preconverge_fmax_ev_a requires climb=true"
                    )
                if method != "neb":
                    raise ValueError(
                        f"{prefix}.climb_preconverge_fmax_ev_a requires method='neb'"
                    )
            raw_fixed_atoms = declaration.get("fixed_atoms", [])
            if (
                not isinstance(raw_fixed_atoms, list)
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    for index in raw_fixed_atoms
                )
                or len(set(raw_fixed_atoms)) != len(raw_fixed_atoms)
            ):
                raise ValueError(
                    f"{prefix}.fixed_atoms must be an ordered unique array of zero-based atom indices"
                )
            relax = declaration.get("relax", False)
            if not isinstance(relax, bool):
                raise ValueError(f"{prefix}.relax must be a boolean")
            neb = NebSamplingParameters(
                images=images,
                fmax_ev_a=float(fmax_ev_a),
                max_steps=max_steps,
                include_history=include_history,
                init=init,
                method=method,
                climb=climb,
                climb_preconverge_fmax_ev_a=climb_preconverge_fmax_ev_a,
                fixed_atoms=tuple(raw_fixed_atoms),
                relax=relax,
            )
            steps_by_round = ()
            normalized_conditions = (neb,)
        elif kind == "relax":
            fmax_ev_a = declaration.get("fmax_ev_a")
            max_steps = declaration.get("max_steps")
            max_step_a = declaration.get("max_step_a")
            include_history = declaration.get("include_history", True)
            history_stride = declaration.get("history_stride", 1)
            accept_unconverged = declaration.get("accept_unconverged", False)
            raw_fixed_atoms = declaration.get("fixed_atoms", [])
            if not include_history and "history_stride" in declaration:
                raise ValueError(
                    f"{prefix}.history_stride is valid only when include_history=true"
                )
            try:
                relax_parameters = RelaxSamplingParameters(
                    fmax_ev_a=fmax_ev_a,
                    max_steps=max_steps,
                    max_step_a=max_step_a,
                    include_history=include_history,
                    history_stride=history_stride,
                    accept_unconverged=accept_unconverged,
                    fixed_atoms=tuple(raw_fixed_atoms) if isinstance(raw_fixed_atoms, list) else raw_fixed_atoms,
                )
            except (TypeError, ValueError) as exc:
                field = next(
                    (
                        name
                        for name in (
                            "fmax_ev_a", "max_steps", "max_step_a",
                            "include_history", "history_stride", "accept_unconverged",
                            "fixed_atoms",
                        )
                        if name in str(exc)
                    ),
                    "configuration",
                )
                detail = str(exc).removeprefix(f"relax {field} ")
                raise ValueError(f"{prefix}.{field} {detail}") from exc
            steps_by_round = ()
            normalized_conditions = (relax_parameters,)
        elif kind in {"nvt_program", "nvt_cell_program", "npt_program"}:
            program = _temperature_program(
                declaration.get("segments"),
                f"{prefix}.segments",
                allow_cell=kind == "nvt_cell_program",
            )
            if kind == "npt_program":
                pressures = _pressure_group(
                    declaration.get("pressures_bar"), f"{prefix}.pressures_bar"
                )
            steps_by_round = (program.total_steps,) * round_count
            normalized_conditions: tuple[object, ...] = (
                tuple((program.segments, pressure) for pressure in pressures)
                if kind == "npt_program"
                else (program.segments,)
            )
        else:
            temperatures = _temperature_group(
                declaration.get("temperatures_k"), f"{prefix}.temperatures_k"
            )
            pressures = (
                _pressure_group(declaration.get("pressures_bar"), f"{prefix}.pressures_bar")
                if kind == "npt" or (kind == 'md_mc' and declaration['ensemble'] == 'npt')
                else ()
            )
            steps_by_round = _explicit_round_steps(
                declaration.get("steps"), round_count, f"{prefix}.steps"
            )
            if md_mc_parameters is not None and any(s < md_mc_parameters.every_md_steps for s in steps_by_round):
                raise ValueError(f'{prefix}.steps must cover at least one MC event')
            normalized_conditions = tuple(
                (temperature, pressure)
                for temperature in temperatures
                for pressure in (pressures if pressures else (None,))
            )
        recipe = None if kind in {"neb", "relax", "gcmc_md", "widom"} else str(declaration["recipe"])
        for role_id in role_ids:
            for condition in normalized_conditions:
                key = (role_id, recipe, kind, condition, steps_by_round, md_mc_parameters)
                if key in duplicate_keys:
                    raise ValueError(f"{prefix} creates a duplicate effective native task")
                duplicate_keys.add(key)
        groups.append(
            SamplingConditionGroup(
                condition_group_id=group_id,
                kind=kind,
                sampling_role_ids=role_ids,
                steps_by_round=steps_by_round,
                temperatures_k=temperatures,
                pressures_bar=pressures,
                program=program,
                neb=neb,
                relax=relax_parameters,
                gcmc=gcmc_parameters,
                widom=widom_parameters,
                md_mc=md_mc_parameters,
            )
        )
    neb_role_ids = {
        role_id
        for group in groups
        if group.kind == "neb"
        for role_id in group.sampling_role_ids
    }
    md_role_ids = {
        role_id
        for group in groups
        if group.kind != "neb"
        for role_id in group.sampling_role_ids
    }
    for role_id in seed_ids:
        if role_id in neb_role_ids and role_id in md_role_ids:
            raise ValueError(
                f"sample.seeds.{role_id} cannot be assigned to both NEB and MD groups"
            )
        if role_id in neb_role_ids and "max_frames" in raw_seeds[role_id]:
            raise ValueError(
                f"sample.seeds.{role_id}.max_frames is forbidden for NEB endpoint roles"
            )
    if raw_rounds is None:
        unused = [seed_id for seed_id in seed_ids if seed_id not in assigned_roles]
        if unused:
            raise ValueError(f"sample.seeds contains unused seed role(s): {', '.join(unused)}")
        rounds = None
    else:
        group_ids = tuple(group.condition_group_id for group in groups)
        rounds = tuple(
            _sampling_round_activation(
                raw,
                round_index=index,
                seed_ids=seed_ids,
                group_ids=group_ids,
            )
            for index, raw in enumerate(raw_rounds)
        )
    return ExplicitSamplingPlan(
        tuple(seeds),
        tuple(groups),
        round_count,
        rounds=rounds,
    )


def _sampling_round_activation(
    value: Any,
    *,
    round_index: int,
    seed_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
) -> SamplingRound:
    prefix = f"sample.rounds[{round_index}]"
    _mapping_fields(value, {"seeds", "condition_groups"}, prefix)

    def active_ids(field: str, declared: tuple[str, ...]) -> tuple[str, ...]:
        raw = value.get(field)
        if raw is None:
            return declared
        if (
            not isinstance(raw, list)
            or not raw
            or any(not isinstance(item, str) for item in raw)
            or len(set(raw)) != len(raw)
        ):
            raise ValueError(f"{prefix}.{field} must be a non-empty unique ID array")
        unknown = sorted(set(raw) - set(declared))
        if unknown:
            raise ValueError(
                f"{prefix}.{field} contains unknown ID(s): {', '.join(unknown)}"
            )
        selected = set(raw)
        return tuple(identity for identity in declared if identity in selected)

    return SamplingRound(
        active_ids("seeds", seed_ids),
        active_ids("condition_groups", group_ids),
    )


def _explicit_round_steps(value: Any, rounds: int, field_name: str) -> tuple[int, ...]:
    values = [value] * rounds if isinstance(value, int) and not isinstance(value, bool) else value
    if (
        not isinstance(values, list)
        or len(values) != rounds
        or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in values)
    ):
        raise ValueError(f"{field_name} must be a positive scalar or an array matching sample.round_count")
    return tuple(values)


def _temperature_program(
    value: Any,
    field_name: str,
    *,
    allow_cell: bool = False,
) -> TemperatureProgram:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty segment array")
    segments: list[TemperatureProgramSegment] = []
    boundary = 0
    for index, raw in enumerate(value):
        prefix = f"{field_name}[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{prefix} must be an object")
        fields = set(raw)
        if "cell" in fields:
            if not allow_cell:
                raise ValueError(
                    f"{prefix}.cell is valid only for nvt_cell_program"
                )
            fields.remove("cell")
        hold = fields == {"temperature_k", "steps"}
        ramp = fields == {"start_temperature_k", "stop_temperature_k", "steps"}
        if not hold and not ramp:
            raise ValueError(f"{prefix} must be exactly one hold or linear-ramp segment")
        steps = raw["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"{prefix}.steps must be a positive integer")
        start = raw["temperature_k"] if hold else raw["start_temperature_k"]
        stop = raw["temperature_k"] if hold else raw["stop_temperature_k"]
        if not _finite_positive(start) or not _finite_positive(stop):
            raise ValueError(f"{prefix} temperatures must be finite and positive")
        cell = (
            _cell_program_endpoint(raw["cell"], f"{prefix}.cell")
            if "cell" in raw
            else None
        )
        boundary += steps
        segments.append(
            TemperatureProgramSegment(
                float(start), float(stop), steps, boundary, cell
            )
        )
    return TemperatureProgram(tuple(segments))


def _cell_program_endpoint(value: Any, field_name: str) -> CellProgramEndpoint:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    mode = value.get("mode")
    if mode == "isotropic_density":
        _mapping_fields(value, {"mode", "stop_density_g_cm3"}, field_name)
        density = value.get("stop_density_g_cm3")
        if not _finite_positive(density):
            raise ValueError(
                f"{field_name}.stop_density_g_cm3 must be finite and positive"
            )
        return CellProgramEndpoint(mode, float(density))
    if mode == "symmetric_strain":
        _mapping_fields(value, {"mode", "stop"}, field_name)
        stop = value.get("stop")
        if not isinstance(stop, dict) or set(stop) != set(
            _SYMMETRIC_STRAIN_COMPONENTS
        ):
            raise ValueError(
                f"{field_name}.stop must contain exactly "
                f"{list(_SYMMETRIC_STRAIN_COMPONENTS)}"
            )
        if not all(_finite(stop[name]) for name in _SYMMETRIC_STRAIN_COMPONENTS):
            raise ValueError(
                f"{field_name}.stop components must be finite non-boolean numbers"
            )
        components = tuple(
            (name, float(stop[name])) for name in _SYMMETRIC_STRAIN_COMPONENTS
        )
        strain = dict(components)
        xx = 1.0 + strain["xx"]
        yy = 1.0 + strain["yy"]
        zz = 1.0 + strain["zz"]
        yz = strain["yz"]
        xz = strain["xz"]
        xy = strain["xy"]
        leading_two = xx * yy - xy * xy
        determinant = (
            xx * yy * zz
            + 2.0 * xy * xz * yz
            - xx * yz * yz
            - yy * xz * xz
            - zz * xy * xy
        )
        if xx <= 0.0 or leading_two <= 0.0 or determinant <= 0.0:
            raise ValueError(f"{field_name}.stop must define positive definite I + E")
        return CellProgramEndpoint(mode, symmetric_strain=components)
    raise ValueError(
        f"{field_name}.mode must be one of "
        "['isotropic_density', 'symmetric_strain']"
    )


def _resolve_sampling_plan(sample: dict[str, Any]) -> SamplingPlan:
    if "ensemble" in sample:
        _enum(sample, "ensemble", {"nvt", "npt"}, "sample")
    common_temperatures = (
        _temperature_group(sample["temperatures_k"], "sample.temperatures_k")
        if "temperatures_k" in sample
        else None
    )
    common_pressures = (
        _pressure_group(sample["pressures_bar"], "sample.pressures_bar")
        if "pressures_bar" in sample
        else None
    )

    resolved: list[SystemSamplingCondition] = []
    for system_id, system in sample["systems"].items():
        prefix = f"sample.systems.{system_id}"
        if "ensemble" in system:
            _enum(system, "ensemble", {"nvt", "npt"}, prefix)
        system_temperatures = (
            _temperature_group(system["temperatures_k"], f"{prefix}.temperatures_k")
            if "temperatures_k" in system
            else None
        )
        system_pressures = (
            _pressure_group(system["pressures_bar"], f"{prefix}.pressures_bar")
            if "pressures_bar" in system
            else None
        )

        ensemble = system.get("ensemble", sample.get("ensemble"))
        if ensemble is None:
            raise ValueError(f"{prefix}.ensemble cannot resolve from system or sample default")
        temperatures = system_temperatures if system_temperatures is not None else common_temperatures
        if temperatures is None:
            raise ValueError(f"{prefix}.temperatures_k cannot resolve for {ensemble}")
        if ensemble == "nvt":
            if "pressures_bar" in system:
                raise ValueError(f"{prefix}.pressures_bar is forbidden for effective nvt")
            pressures: list[float] = []
        else:
            pressures = system_pressures if system_pressures is not None else common_pressures
            if pressures is None:
                raise ValueError(f"{prefix}.pressures_bar cannot resolve for npt")
        resolved.append(
            SystemSamplingCondition(
                system_id=system_id,
                ensemble=ensemble,
                temperatures_k=tuple(float(item) for item in temperatures),
                pressures_bar=tuple(float(item) for item in pressures),
            )
        )
    return SamplingPlan.create(tuple(resolved))


def _temperature_group(value: Any, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or not all(_finite_positive(item) for item in value):
        raise ValueError(f"{field_name} must be a non-empty finite-positive array")
    normalized = tuple(float(item) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


def _pressure_group(value: Any, field_name: str) -> tuple[float, ...]:
    values = value if isinstance(value, list) else [value]
    if not values or not all(_finite(item) for item in values):
        raise ValueError(f"{field_name} must be one finite scalar or a non-empty finite array")
    normalized = tuple(float(item) for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


def _validate_screen(value: Any, *, rounds: int, teacher: dict[str, Any], student: dict[str, Any]) -> None:
    _mapping_fields(
        value,
        {
            "pool", "keep", "fp_pool", "policy", "lo", "hi", "teacher_p",
            "student_p", "pair_guard", "fill_underfilled_from_safe_region",
            "prescreen",
        },
        "screen",
    )
    single_teacher = teacher["model"] == "single"
    single_student = student["model"] == "single"
    prescreen = value.get("prescreen", "GMM")
    if not single_student and "prescreen" in value:
        raise ValueError("screen.prescreen is valid only for a single student")
    if single_student and prescreen not in {
        "GMM",
        "JOINT_DIRECT",
        "ELEMENT_JOINT_DIRECT",
    }:
        raise ValueError(
            "screen.prescreen must be GMM or JOINT_DIRECT or ELEMENT_JOINT_DIRECT"
        )
    joint_direct = single_student and prescreen in {
        "JOINT_DIRECT",
        "ELEMENT_JOINT_DIRECT",
    }
    if prescreen == "GMM":
        value.pop("prescreen", None)
    fill_safe_region = value.get(
        "fill_underfilled_from_safe_region",
        False if joint_direct else True,
    )
    if not isinstance(fill_safe_region, bool):
        raise ValueError("screen.fill_underfilled_from_safe_region must be a boolean")
    if joint_direct and fill_safe_region:
        raise ValueError(
            "screen.fill_underfilled_from_safe_region must be false for joint DIRECT prescreen"
        )
    value["fill_underfilled_from_safe_region"] = fill_safe_region
    if "pair_guard" in value:
        value["pair_guard"] = _pair_guard_config(
            value["pair_guard"],
            type_map=tuple(student["type_map"]),
        )
    if isinstance(value.get("policy"), str) and "coverage" in value["policy"].lower():
        raise ValueError(
            "retired lifecycle coverage policy is unsupported; use the fixed "
            "gmm-reference-tail revision 4 single-model route"
        )
    if single_teacher:
        value["teacher_p"] = _p_band_config(
            value.get("teacher_p"),
            field_name="screen.teacher_p",
            include_trust=True,
        )
        fp_pool = value.get("fp_pool")
        if not isinstance(fp_pool, int) or isinstance(fp_pool, bool) or fp_pool == 0 or fp_pool < -1:
            raise ValueError("single teacher requires positive scalar screen.fp_pool or conditional -1")
        if fp_pool == -1 and "ceiling" not in value["teacher_p"]:
            raise ValueError("screen.fp_pool: -1 requires screen.teacher_p.ceiling")
    if single_student:
        if joint_direct:
            if "student_p" in value:
                raise ValueError(
                    "screen.student_p is invalid for joint DIRECT prescreen"
                )
        else:
            value["student_p"] = _p_band_config(
                value.get("student_p"),
                field_name="screen.student_p",
                include_trust=False,
            )
            raw_pools = [value.get("pool")] * rounds if isinstance(value.get("pool"), int) else value.get("pool")
            if isinstance(raw_pools, list) and any(item == -1 for item in raw_pools) and "ceiling" not in value["student_p"]:
                raise ValueError("screen.pool: -1 requires screen.student_p.ceiling")
    elif "student_p" in value:
        raise ValueError("screen.student_p is valid only for a single student")
    pools = _round_quota(
        value.get("pool"),
        rounds,
        "screen.pool",
        allow_all=student["model"] == "committee" or (
            single_student
            and not joint_direct
            and "ceiling" in value["student_p"]
        ),
    )
    keeps = _round_quota(value.get("keep"), rounds, "screen.keep", allow_all=False)
    for pool, keep in zip(pools, keeps):
        if pool > 0 and keep > pool:
            raise ValueError("screen.keep must not exceed a positive screen.pool")
    _enum(
        value,
        "policy",
        {"DIRECT", "ELEMENT_DIRECT", "FPS", "random"},
        "screen",
    )
    if student["model"] == "committee":
        _band(value, "screen")
    elif "lo" in value or "hi" in value:
        raise ValueError("single student forbids screen.lo and screen.hi")
    if not single_teacher:
        if "fp_pool" in value:
            raise ValueError("committee teacher forbids screen.fp_pool")
        if "teacher_p" in value:
            raise ValueError("screen.teacher_p is valid only for a single teacher")


def _validate_fp(config: RunConfig, value: Any) -> None:
    _mapping_fields(value, {"keep", "policy", "all_rn", "label"}, "fp")
    if not isinstance(value.get("keep"), int) or value["keep"] <= 0:
        raise ValueError("fp.keep must be a positive integer")
    _enum(
        value,
        "policy",
        {
            "DIRECT",
            "ELEMENT_DIRECT",
            "FPS",
            "random",
            "JOINT_DIRECT",
            "ELEMENT_JOINT_DIRECT",
        },
        "fp",
    )
    if (
        value["policy"] in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}
        and config.teacher["model"] != "single"
    ):
        raise ValueError(
            f"fp.policy {value['policy']} is valid only for a single teacher"
        )
    all_rn = value.get("all_rn", False)
    if not isinstance(all_rn, bool):
        raise ValueError("fp.all_rn must be a boolean")
    value["all_rn"] = all_rn
    if "label" in value:
        label = value["label"]
        _mapping_fields(label, {"calculator", "recipe"}, "fp.label")
        _enum(label, "calculator", {"ABACUS", "CP2K"}, "fp.label")
        _relative_path(config, label.get("recipe"), "fp.label.recipe")


def _validate_model_shape_route(config: RunConfig) -> RouteSpec:
    teacher = config.teacher
    student = config.student
    route = resolve_route(
        teacher=teacher,
        student=student,
        sampler=config.sampler,
    )
    return route


def validate_role_recipe_feasibility(config: RunConfig, role: str) -> None:
    """Validate one executing train role's external native recipe."""

    if role not in {"teacher", "student"}:
        raise ValueError(f"unsupported lifecycle recipe role {role!r}")
    spec = config.raw[role]
    recipe_path = _resolved_path(config, spec["recipe"])
    _require_external_file(recipe_path, f"{role}.recipe")
    if spec["backend"] == "DeepMD":
        try:
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{role}.recipe must be native DeepMD JSON") from exc
        if not isinstance(recipe, dict):
            raise ValueError(f"{role}.recipe must be a JSON object")
        if "workflow" in recipe:
            raise ValueError(f"{role}.recipe contains workflow-owned declarations")
        training = recipe.get("training", {})
        if isinstance(training, dict):
            for field in ("training_data", "validation_data"):
                section = training.get(field, {})
                if isinstance(section, dict) and section.get("systems"):
                    raise ValueError(
                        f"{role}.recipe declares workflow-owned {field}.systems"
                    )
        if role == "student":
            if (
                deepmd_recipe_model_family(recipe) == "dpa4c"
                and spec.get("cli") != "pt"
            ):
                raise ValueError("DPA4C student recipes require a DeepMD/PT declaration")
        route = config.route
        if role == "teacher" and route is not None and route.teacher_requires_reference_tail:
            if deepmd_recipe_model_family(recipe) != "dpa4":
                raise ValueError(
                    "the supported DeepMD-PT GMM teacher recipe must select native DPA4"
                )
        if role == "student" and route is not None and route.student_requires_reference_tail:
            descriptor = recipe.get("model", {}).get("descriptor", {})
            expected = "se_e2_a" if spec.get("cli") == "tf" else "dpa4c"
            selected = (
                deepmd_recipe_model_family(recipe) == "dpa4c"
                if expected == "dpa4c"
                else isinstance(descriptor, dict) and descriptor.get("type") == expected
            )
            if not selected:
                raise ValueError(
                    "the supported DeepMD GMM student recipe requires "
                    f"model.descriptor.type={expected!r} for cli={spec.get('cli')!r}"
                )
    elif role == "student" and config.route is not None and config.route.student_requires_reference_tail:
        _validate_nep4_train_recipe(recipe_path)


def validate_sampler_recipe_feasibility(
    config: RunConfig,
    *,
    condition_group_ids: set[str] | None = None,
) -> None:
    """Validate only sampler recipes owned by an executing sample suffix."""

    sampler = config.raw["sampler"]
    sample = config.raw["sample"]
    if "condition_groups" in sample:
        recipes = tuple(
            (
                f"sample.condition_groups.{group_id}.recipe",
                declaration["recipe"],
            )
            for group_id, declaration in sample["condition_groups"].items()
            if condition_group_ids is None or group_id in condition_group_ids
            if "recipe" in declaration
        )
    else:
        recipes = (
            (("sampler.recipe", sampler["recipe"]),)
            if "recipe" in sampler
            else tuple(
                (f"sampler.recipes.{ensemble}", recipe)
                for ensemble, recipe in sampler["recipes"].items()
            )
        )
    for field, raw_recipe in recipes:
        path = _resolved_path(config, raw_recipe)
        _require_external_file(path, field)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"{field} must be readable text") from exc
        if "DICE_BEGIN" in text or "DICE_END" in text or "{{" in text:
            raise ValueError(f"{field} contains workflow-owned managed declarations")


def require_external_config_path(
    config: RunConfig, raw_path: str, *, field_name: str
) -> Path:
    """Resolve and require one already syntax-validated external input."""

    path = _resolved_path(config, raw_path)
    _require_external_file(path, field_name)
    return path


def _require_external_file(path: Path, field_name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field_name} path does not exist or is not a regular file")


def _validate_nep4_train_recipe(path: Path) -> None:
    try:
        rows = [
            line.split("#", 1)[0].split()
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("the supported NEP4 GMM student recipe must be readable text") from exc
    versions = [row for row in rows if row[:1] == ["version"]]
    if len(versions) != 1 or versions[0] != ["version", "4"]:
        raise ValueError("the supported NEP4 GMM student recipe requires exactly one 'version 4' line")


def _mapping_fields(value: Any, allowed: set[str], field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unsupported field(s): {', '.join(unknown)}")


def _normalize_role_train_options(value: dict[str, Any], *, role: str) -> None:
    if "train_options" not in value:
        return
    if value["backend"] != "DeepMD":
        raise ValueError(f"{role}.train_options is valid only for DeepMD")
    raw = value["train_options"]
    _mapping_fields(raw, {"skip_neighbor_stat"}, f"{role}.train_options")
    skip_neighbor_stat = raw.get("skip_neighbor_stat", False)
    if not isinstance(skip_neighbor_stat, bool):
        raise ValueError(f"{role}.train_options.skip_neighbor_stat must be a boolean")
    if skip_neighbor_stat:
        value["train_options"] = {"skip_neighbor_stat": True}
    else:
        del value["train_options"]


def _enum(value: dict[str, Any], field: str, allowed: set[str], prefix: str) -> None:
    if value.get(field) not in allowed:
        raise ValueError(f"{prefix}.{field} must be one of {sorted(allowed)}")


def _type_map(value: dict[str, Any], prefix: str) -> None:
    types = value.get("type_map")
    if not isinstance(types, list) or not types or not all(isinstance(item, str) and item and item.isalpha() for item in types) or len(set(types)) != len(types):
        raise ValueError(f"{prefix}.type_map must be a non-empty unique chemical-symbol list")


def _validate_path_list(config: RunConfig, value: Any, field_name: str, *, nonempty: bool) -> None:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{field_name} must be {'a non-empty ' if nonempty else 'an '}array")
    for index, item in enumerate(value):
        _relative_path(config, item, f"{field_name}[{index}]")


def _relative_path(config: RunConfig, value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a relative path")
    candidate = Path(value)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts) or any(char in value for char in "*?["):
        raise ValueError(f"{field_name} must be a relative non-glob path")


def _resolved_path(config: RunConfig, value: str) -> Path:
    return (config.root / value).resolve()


def _fraction(value: Any, field_name: str) -> None:
    if not _finite(value) or not 0 <= float(value) < 1:
        raise ValueError(f"{field_name} must be in [0, 1)")


def _p_band_config(value: Any, *, field_name: str, include_trust: bool) -> dict[str, float]:
    if value is None:
        raw: dict[str, Any] = {}
    else:
        allowed = {"floor", "ceiling"}
        if include_trust:
            allowed.add("trust")
        _mapping_fields(value, allowed, field_name)
        raw = value
    floor = raw.get("floor", 0.0)
    if not _finite(floor) or not 0 <= float(floor) < 1:
        raise ValueError(f"{field_name}.floor must be a finite scalar in [0, 1)")
    normalized = {"floor": float(floor)}
    if include_trust:
        trust = raw.get("trust", 0.1)
        if not _finite(trust) or not 0 <= float(trust) <= 1:
            raise ValueError(f"{field_name}.trust must be a finite scalar in [0, 1]")
        normalized["trust"] = float(trust)
    if "ceiling" in raw:
        ceiling = raw["ceiling"]
        if not _finite(ceiling) or not float(floor) < float(ceiling) <= 1:
            raise ValueError(f"{field_name}.ceiling must be a finite scalar in (floor, 1]")
        normalized["ceiling"] = float(ceiling)
    return normalized


def _pair_guard_config(value: Any, *, type_map: tuple[str, ...]) -> dict[str, Any]:
    _mapping_fields(value, {"min", "pairs"}, "screen.pair_guard")
    if "min" not in value or not _finite_positive(value["min"]):
        raise ValueError("screen.pair_guard.min must be a finite positive scalar in angstrom")
    minimum = float(value["min"])
    raw_pairs = value.get("pairs", {})
    if not isinstance(raw_pairs, dict):
        raise ValueError("screen.pair_guard.pairs must be an object")
    configured = set(type_map)
    normalized: dict[str, float] = {}
    for raw_key, raw_threshold in raw_pairs.items():
        if not isinstance(raw_key, str) or raw_key != raw_key.strip():
            raise ValueError(f"invalid screen.pair_guard pair key {raw_key!r}; expected A-B")
        parts = raw_key.split("-")
        if len(parts) != 2 or any(not part or not part.isalpha() for part in parts):
            raise ValueError(f"invalid screen.pair_guard pair key {raw_key!r}; expected A-B")
        if any(part not in atomic_numbers for part in parts):
            raise ValueError(f"screen.pair_guard pair key {raw_key!r} contains an invalid chemical symbol")
        if any(part not in configured for part in parts):
            raise ValueError(
                f"screen.pair_guard pair key {raw_key!r} contains an element absent from student.type_map"
            )
        canonical = "-".join(sorted(parts))
        if canonical in normalized:
            raise ValueError(f"duplicate canonical screen.pair_guard pair rule: {canonical}")
        if not _finite_positive(raw_threshold) or float(raw_threshold) < minimum:
            raise ValueError(
                f"screen.pair_guard pair threshold for {canonical} must be finite, positive, and at least min"
            )
        normalized[canonical] = float(raw_threshold)
    return {"min": minimum, "pairs": dict(sorted(normalized.items()))}


def _members(value: dict[str, Any], prefix: str) -> None:
    if not isinstance(value.get("members"), int) or value["members"] < 2:
        raise ValueError(f"{prefix}.members must be at least 2 for a committee")


def _band(value: dict[str, Any], prefix: str) -> None:
    lo, hi = value.get("lo"), value.get("hi")
    if not _finite(lo) or not _finite(hi) or not 0 <= float(lo) < float(hi):
        raise ValueError(f"{prefix} requires 0 <= lo < hi")


def _foundation_path(config: RunConfig, value: dict[str, Any], prefix: str) -> None:
    if value["policy"] == "foundation_finetune":
        _relative_path(config, value.get("foundation_model"), f"{prefix}.foundation_model")
    elif "foundation_model" in value:
        raise ValueError(f"{prefix}.foundation_model is only valid for foundation_finetune")


def _round_quota(value: Any, rounds: int, field_name: str, *, allow_all: bool) -> list[int]:
    values = [value] * rounds if isinstance(value, int) else value
    if not isinstance(values, list) or len(values) != rounds or not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        raise ValueError(f"{field_name} must be a scalar or an array matching sample.steps")
    if any(item == 0 or item < -1 for item in values):
        raise ValueError(f"{field_name} must contain positive integers or conditional -1")
    if any(item == -1 for item in values):
        if not allow_all:
            raise ValueError(f"{field_name}: -1 is not valid for this route")
    return values


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0
