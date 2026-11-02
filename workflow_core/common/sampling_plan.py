from __future__ import annotations

from dataclasses import dataclass
import math

from workflow_core.common.gcmc import GcmcSamplingParameters
from workflow_core.common.widom import WidomSamplingParameters
from workflow_core.common.md_mc import IdentitySwapParameters


@dataclass(frozen=True, slots=True)
class NebSamplingParameters:
    """Closed backend-neutral parameters for one independent NEB path."""

    images: int
    fmax_ev_a: float
    max_steps: int
    include_history: bool = False
    init: str = "linear"
    method: str = "neb"
    climb: bool = False
    climb_preconverge_fmax_ev_a: float | None = None
    fixed_atoms: tuple[int, ...] = ()
    relax: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.images, bool)
            or not isinstance(self.images, int)
            or self.images <= 0
        ):
            raise ValueError("NEB images must be a positive integer")
        if (
            isinstance(self.fmax_ev_a, bool)
            or not isinstance(self.fmax_ev_a, (int, float))
            or not math.isfinite(float(self.fmax_ev_a))
            or float(self.fmax_ev_a) <= 0.0
        ):
            raise ValueError("NEB fmax_ev_a must be a finite positive number")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("NEB max_steps must be a positive integer")
        if not isinstance(self.include_history, bool):
            raise ValueError("NEB include_history must be a boolean")
        if self.init not in {"linear", "idpp"}:
            raise ValueError("NEB init must be linear or idpp")
        if self.method not in {"neb", "dyneb"}:
            raise ValueError("NEB normalized method must be neb or dyneb")
        if not isinstance(self.climb, bool):
            raise ValueError("NEB climb must be a boolean")
        if self.climb_preconverge_fmax_ev_a is not None:
            threshold = self.climb_preconverge_fmax_ev_a
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or float(threshold) <= float(self.fmax_ev_a)
            ):
                raise ValueError(
                    "NEB climb_preconverge_fmax_ev_a must be finite and greater than fmax_ev_a"
                )
            if not self.climb:
                raise ValueError(
                    "NEB climb_preconverge_fmax_ev_a requires climb=true"
                )
            if self.method != "neb":
                raise ValueError(
                    "NEB climb_preconverge_fmax_ev_a requires method=neb"
                )
            object.__setattr__(
                self,
                "climb_preconverge_fmax_ev_a",
                float(threshold),
            )
        if not isinstance(self.relax, bool):
            raise ValueError("NEB relax must be a boolean")
        fixed_atoms = tuple(self.fixed_atoms)
        if (
            any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in fixed_atoms
            )
            or len(set(fixed_atoms)) != len(fixed_atoms)
        ):
            raise ValueError(
                "NEB fixed_atoms must be ordered unique zero-based indices"
            )
        object.__setattr__(self, "fmax_ev_a", float(self.fmax_ev_a))
        object.__setattr__(self, "fixed_atoms", fixed_atoms)


@dataclass(frozen=True, slots=True)
class RelaxSamplingParameters:
    """Closed backend-neutral parameters for one fixed-cell relaxation."""

    fmax_ev_a: float
    max_steps: int
    max_step_a: float | None = None
    include_history: bool = True
    history_stride: int = 1
    accept_unconverged: bool = False
    fixed_atoms: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.fmax_ev_a, bool)
            or not isinstance(self.fmax_ev_a, (int, float))
            or not math.isfinite(float(self.fmax_ev_a))
            or float(self.fmax_ev_a) <= 0.0
        ):
            raise ValueError("relax fmax_ev_a must be a finite positive number")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("relax max_steps must be a positive integer")
        if self.max_step_a is not None and (
            isinstance(self.max_step_a, bool)
            or not isinstance(self.max_step_a, (int, float))
            or not math.isfinite(float(self.max_step_a))
            or float(self.max_step_a) <= 0.0
        ):
            raise ValueError("relax max_step_a must be a finite positive number")
        if not isinstance(self.include_history, bool):
            raise ValueError("relax include_history must be a boolean")
        if not isinstance(self.accept_unconverged, bool):
            raise ValueError("relax accept_unconverged must be a boolean")
        if (
            isinstance(self.history_stride, bool)
            or not isinstance(self.history_stride, int)
            or self.history_stride <= 0
        ):
            raise ValueError("relax history_stride must be a positive integer")
        if not self.include_history and self.history_stride != 1:
            raise ValueError("relax history_stride is valid only when include_history=true")
        fixed_atoms = tuple(self.fixed_atoms)
        if (
            any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in fixed_atoms
            )
            or len(set(fixed_atoms)) != len(fixed_atoms)
        ):
            raise ValueError(
                "relax fixed_atoms must be ordered unique zero-based indices"
            )
        object.__setattr__(self, "fmax_ev_a", float(self.fmax_ev_a))
        if self.max_step_a is not None:
            object.__setattr__(self, "max_step_a", float(self.max_step_a))
        object.__setattr__(self, "fixed_atoms", fixed_atoms)


def relax_parameters_payload(
    parameters: RelaxSamplingParameters,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "fmax_ev_a": parameters.fmax_ev_a,
        "max_steps": parameters.max_steps,
        "max_step_a": parameters.max_step_a,
        "include_history": parameters.include_history,
        "history_stride": parameters.history_stride,
        "fixed_atoms": list(parameters.fixed_atoms),
    }
    if parameters.accept_unconverged:
        payload["accept_unconverged"] = True
    return payload


def relax_algorithm_identity(
    parameters: RelaxSamplingParameters,
) -> dict[str, object]:
    return {
        "name": "ase-fire-fixed-cell-relax",
        "revision": 1,
        "optimizer": "FIRE",
        "fixed_cell": True,
        "max_step_a": parameters.max_step_a,
    }


def relax_constraint_identity(
    parameters: RelaxSamplingParameters,
) -> dict[str, object]:
    return {
        "name": "ase-fix-atoms",
        "revision": 1,
        "indices": list(parameters.fixed_atoms),
        "zero_based": True,
    }


def neb_parameters_payload(parameters: NebSamplingParameters) -> dict[str, object]:
    payload: dict[str, object] = {
        "images": parameters.images,
        "fmax_ev_a": parameters.fmax_ev_a,
        "max_steps": parameters.max_steps,
        "include_history": parameters.include_history,
        "init": parameters.init,
        "method": parameters.method,
        "climb": parameters.climb,
        "fixed_atoms": list(parameters.fixed_atoms),
    }
    if parameters.relax:
        payload["relax"] = True
    if parameters.climb_preconverge_fmax_ev_a is not None:
        payload["climb_preconverge_fmax_ev_a"] = (
            parameters.climb_preconverge_fmax_ev_a
        )
    return payload


def neb_algorithm_identity(parameters: NebSamplingParameters) -> dict[str, object]:
    if parameters.method == "neb":
        ase_method = "improvedtangent"
        band = "NEB"
    elif parameters.method == "dyneb":
        ase_method = "improvedtangent"
        band = "DyNEB"
    else:
        raise ValueError(f"unsupported normalized NEB method {parameters.method!r}")
    identity: dict[str, object] = {
        "name": "ase-neb-fire",
        "revision": 4,
        "band": band,
        "interpolation": parameters.init,
        "interpolation_mic": parameters.init == "idpp",
        "method": ase_method,
        "spring_ev_a2": 0.1,
        "optimizer": "FIRE",
        "climb": parameters.climb,
        "fixed_cell": True,
        "fixed_endpoints": True,
    }
    if parameters.climb_preconverge_fmax_ev_a is not None:
        identity["climb_schedule"] = {
            "mode": "preconverge_then_climb",
            "preconverge_fmax_ev_a": parameters.climb_preconverge_fmax_ev_a,
            "max_steps_scope": "shared_total",
        }
    if parameters.method == "dyneb":
        identity.update(
            dynamic_relaxation=True,
            scale_fmax=0.0,
            fmax_ev_a=parameters.fmax_ev_a,
        )
    return identity


def neb_constraint_identity(parameters: NebSamplingParameters) -> dict[str, object]:
    return {
        "name": "ase-fix-atoms",
        "revision": 1,
        "indices": list(parameters.fixed_atoms),
        "zero_based": True,
    }


@dataclass(frozen=True, slots=True)
class SeedRole:
    sampling_role_id: str
    selection_group_id: str
    source_frame_index: int = 0


@dataclass(frozen=True, slots=True)
class SelectionGroup:
    selection_group_id: str
    sampling_role_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionPlan:
    groups: tuple[SelectionGroup, ...]
    role_to_group: tuple[tuple[str, str], ...]

    @classmethod
    def from_seeds(cls, seeds: tuple[SeedRole, ...]) -> "SelectionPlan":
        ordered: dict[str, list[str]] = {}
        mapping: list[tuple[str, str]] = []
        for seed in seeds:
            mapping.append((seed.sampling_role_id, seed.selection_group_id))
            ordered.setdefault(seed.selection_group_id, []).append(
                seed.sampling_role_id
            )
        return cls(
            tuple(
                SelectionGroup(group_id, tuple(role_ids))
                for group_id, role_ids in ordered.items()
            ),
            tuple(mapping),
        )

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(group.selection_group_id for group in self.groups)

    def group_for_role(self, sampling_role_id: str) -> str:
        try:
            return dict(self.role_to_group)[sampling_role_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown sampling role {sampling_role_id!r} in selection plan"
            ) from exc

    def uniform(self, value: object) -> tuple[tuple[str, object], ...]:
        return tuple((group_id, value) for group_id in self.group_ids)


@dataclass(frozen=True, slots=True)
class CellProgramEndpoint:
    mode: str
    stop_density_g_cm3: float | None = None
    symmetric_strain: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class TemperatureProgramSegment:
    start_temperature_k: float
    stop_temperature_k: float
    steps: int
    end_step: int
    cell: CellProgramEndpoint | None = None


@dataclass(frozen=True, slots=True)
class TemperatureProgram:
    segments: tuple[TemperatureProgramSegment, ...]

    @property
    def total_steps(self) -> int:
        return self.segments[-1].end_step

    @property
    def boundaries(self) -> tuple[int, ...]:
        return tuple(segment.end_step for segment in self.segments)


@dataclass(frozen=True, slots=True)
class SamplingConditionGroup:
    condition_group_id: str
    kind: str
    sampling_role_ids: tuple[str, ...]
    steps_by_round: tuple[int, ...] = ()
    temperatures_k: tuple[float, ...] = ()
    pressures_bar: tuple[float, ...] = ()
    program: TemperatureProgram | None = None
    neb: NebSamplingParameters | None = None
    relax: RelaxSamplingParameters | None = None
    gcmc: GcmcSamplingParameters | None = None
    widom: WidomSamplingParameters | None = None
    md_mc: IdentitySwapParameters | None = None


@dataclass(frozen=True, slots=True)
class SamplingRound:
    sampling_role_ids: tuple[str, ...]
    condition_group_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedSamplingTask:
    task_id: str
    condition_group_id: str
    sampling_role_id: str
    selection_group_id: str
    kind: str
    recipe_binding: str | None
    steps: int | None
    temperature_k: float | None = None
    pressure_bar: float | None = None
    program: TemperatureProgram | None = None
    source_frame_index: int = 0
    neb: NebSamplingParameters | None = None
    relax: RelaxSamplingParameters | None = None
    gcmc: GcmcSamplingParameters | None = None
    widom: WidomSamplingParameters | None = None
    md_mc: IdentitySwapParameters | None = None
    cycles: int | None = None

    @property
    def system_id(self) -> str:
        """Physical seed role that owns this native sampling task."""

        return self.sampling_role_id

    @property
    def ensemble(self) -> str:
        if self.kind == 'md_mc':
            return 'npt' if self.pressure_bar is not None else 'nvt'
        if self.kind in {"neb", "relax"}:
            return self.kind
        return (
            "nvt"
            if self.kind in {"nvt_program", "nvt_cell_program", "md_mc"}
            else "npt" if self.kind == "npt_program" else self.kind
        )


@dataclass(frozen=True, slots=True)
class ExplicitSamplingPlan:
    seeds: tuple[SeedRole, ...]
    condition_groups: tuple[SamplingConditionGroup, ...]
    round_count: int
    selection_plan: SelectionPlan | None = None
    rounds: tuple[SamplingRound, ...] | None = None

    explicit: bool = True

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if not seeds or len({seed.sampling_role_id for seed in seeds}) != len(seeds):
            raise ValueError("explicit sampling plan requires ordered unique seed roles")
        resolved = SelectionPlan.from_seeds(seeds)
        if self.selection_plan is not None and self.selection_plan != resolved:
            raise ValueError("explicit sampling and selection plans disagree")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "condition_groups", tuple(self.condition_groups))
        object.__setattr__(self, "round_count", int(self.round_count))
        object.__setattr__(self, "selection_plan", resolved)
        if self.round_count <= 0:
            raise ValueError("explicit sampling plan requires at least one round")
        rounds = self.rounds
        explicit_round_activation = rounds is not None
        if rounds is None:
            rounds = tuple(
                SamplingRound(
                    tuple(seed.sampling_role_id for seed in seeds),
                    tuple(group.condition_group_id for group in self.condition_groups),
                )
                for _ in range(self.round_count)
            )
        else:
            rounds = tuple(rounds)
        if len(rounds) != self.round_count:
            raise ValueError("explicit sampling rounds disagree with round count")
        seed_ids = {seed.sampling_role_id for seed in seeds}
        group_index = {
            group.condition_group_id: group for group in self.condition_groups
        }
        for round_index, activation in enumerate(rounds):
            if not isinstance(activation, SamplingRound):
                raise TypeError("explicit sampling rounds require SamplingRound values")
            active_roles = set(activation.sampling_role_ids)
            active_groups = set(activation.condition_group_ids)
            if not active_roles or not active_groups:
                raise ValueError(f"sampling round {round_index} must activate roles and groups")
            if not active_roles <= seed_ids or not active_groups <= set(group_index):
                raise ValueError(f"sampling round {round_index} contains unknown activation")
            effective_roles: set[str] = set()
            for group_id in activation.condition_group_ids:
                assigned = set(group_index[group_id].sampling_role_ids) & active_roles
                if not assigned:
                    raise ValueError(
                        f"sampling round {round_index} condition group {group_id!r} has no effective seed"
                    )
                effective_roles.update(assigned)
            unused = active_roles - effective_roles
            if explicit_round_activation and unused:
                raise ValueError(
                    f"sampling round {round_index} has inactive seed role(s): {', '.join(sorted(unused))}"
                )
        object.__setattr__(self, "rounds", rounds)

    def selection_plan_for_round(self, round_index: int) -> SelectionPlan:
        activation = self._round(round_index)
        active_roles = set(activation.sampling_role_ids)
        return SelectionPlan.from_seeds(
            tuple(
                seed for seed in self.seeds
                if seed.sampling_role_id in active_roles
            )
        )

    def selection_plan_for_rounds(
        self, round_indices: tuple[int, ...]
    ) -> SelectionPlan:
        active_roles = {
            role_id
            for round_index in round_indices
            for role_id, _group_id in self.selection_plan_for_round(
                round_index
            ).role_to_group
        }
        return SelectionPlan.from_seeds(
            tuple(
                seed for seed in self.seeds
                if seed.sampling_role_id in active_roles
            )
        )

    def _round(self, round_index: int) -> SamplingRound:
        if isinstance(round_index, bool) or not 0 <= int(round_index) < self.round_count:
            raise ValueError("sampling round index is outside the resolved plan")
        return self.rounds[int(round_index)]

    def tasks_for_round(self, round_index: int) -> tuple[ResolvedSamplingTask, ...]:
        activation = self._round(round_index)
        active_roles = set(activation.sampling_role_ids)
        active_groups = set(activation.condition_group_ids)
        tasks: list[ResolvedSamplingTask] = []
        group_by_role = dict(self.selection_plan.role_to_group)
        seed_by_role = {seed.sampling_role_id: seed for seed in self.seeds}
        for group in self.condition_groups:
            if group.condition_group_id not in active_groups:
                continue
            for role_id in group.sampling_role_ids:
                if role_id not in active_roles:
                    continue
                if group.kind == "nvt" or (group.kind == 'md_mc' and not group.pressures_bar):
                    conditions = ((temperature, None) for temperature in group.temperatures_k)
                elif group.kind in {"npt", "gcmc_md", "md_mc"}:
                    conditions = (
                        (temperature, pressure)
                        for temperature in group.temperatures_k
                        for pressure in group.pressures_bar
                    )
                elif group.kind == "npt_program":
                    conditions = ((None, pressure) for pressure in group.pressures_bar)
                else:
                    conditions = ((None, None),)
                for temperature, pressure in conditions:
                    neb = group.neb if group.kind == "neb" else None
                    relax = group.relax if group.kind == "relax" else None
                    tasks.append(
                        ResolvedSamplingTask(
                            task_id=f"task_{len(tasks):06d}",
                            condition_group_id=group.condition_group_id,
                            sampling_role_id=role_id,
                            selection_group_id=group_by_role[role_id],
                            kind=group.kind,
                            recipe_binding=(
                                None
                                if group.kind in {"neb", "relax", "widom"} or (
                                    group.kind == 'gcmc_md' and group.gcmc.md is None)
                                else group.condition_group_id
                            ),
                            steps=(
                                None
                                if group.kind in {"neb", "relax", "gcmc_md", "widom"}
                                else group.steps_by_round[int(round_index)]
                            ),
                            temperature_k=temperature,
                            pressure_bar=pressure,
                            program=group.program,
                            source_frame_index=seed_by_role[role_id].source_frame_index,
                            neb=neb,
                            relax=relax,
                            gcmc=group.gcmc,
                            widom=group.widom,
                            md_mc=group.md_mc,
                            cycles=(group.gcmc.cycles_by_round[int(round_index)]
                                    if group.gcmc is not None else None),
                        )
                    )
        return tuple(tasks)

    def expand_source_frames(
        self,
        frame_indices_by_role: dict[str, tuple[int, ...]],
    ) -> "ExplicitSamplingPlan":
        """Expand selected source frames into deterministic sampling roles."""

        unknown = sorted(set(frame_indices_by_role) - {seed.sampling_role_id for seed in self.seeds})
        if unknown:
            raise ValueError(f"unknown frame-sampled role(s): {', '.join(unknown)}")
        expanded_by_role: dict[str, tuple[str, ...]] = {}
        expanded_seeds: list[SeedRole] = []
        for seed in self.seeds:
            indices = frame_indices_by_role.get(seed.sampling_role_id)
            if indices is None:
                expanded_by_role[seed.sampling_role_id] = (seed.sampling_role_id,)
                expanded_seeds.append(seed)
                continue
            if not indices or len(set(indices)) != len(indices) or any(
                isinstance(index, bool) or index < 0 for index in indices
            ):
                raise ValueError(
                    f"selected source frames for {seed.sampling_role_id!r} must be unique non-negative indices"
                )
            role_ids = tuple(
                f"{seed.sampling_role_id}__frame_{index:06d}" for index in indices
            )
            expanded_by_role[seed.sampling_role_id] = role_ids
            expanded_seeds.extend(
                SeedRole(role_id, seed.selection_group_id, index)
                for role_id, index in zip(role_ids, indices, strict=True)
            )
        groups = tuple(
            SamplingConditionGroup(
                condition_group_id=group.condition_group_id,
                kind=group.kind,
                sampling_role_ids=tuple(
                    expanded_role
                    for role_id in group.sampling_role_ids
                    for expanded_role in expanded_by_role[role_id]
                ),
                steps_by_round=group.steps_by_round,
                temperatures_k=group.temperatures_k,
                pressures_bar=group.pressures_bar,
                program=group.program,
                neb=group.neb,
                relax=group.relax,
                gcmc=group.gcmc,
                widom=group.widom,
                md_mc=group.md_mc,
            )
            for group in self.condition_groups
        )
        rounds = tuple(
            SamplingRound(
                tuple(
                    expanded_role
                    for role_id in activation.sampling_role_ids
                    for expanded_role in expanded_by_role[role_id]
                ),
                activation.condition_group_ids,
            )
            for activation in self.rounds
        )
        return ExplicitSamplingPlan(
            tuple(expanded_seeds),
            groups,
            self.round_count,
            rounds=rounds,
        )

    @property
    def tasks(self) -> tuple[ResolvedSamplingTask, ...]:
        return self.tasks_for_round(0)

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def total_temperature_values(self) -> int:
        return sum(
            (
                len(group.temperatures_k)
                if group.kind not in {"nvt_program", "nvt_cell_program", "npt_program"}
                else 1
            )
            for group in self.condition_groups
        )

    @property
    def ensembles(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    ('npt' if group.pressures_bar else 'nvt')
                    if group.kind == 'md_mc' else "nvt"
                    if group.kind in {"nvt_program", "nvt_cell_program"}
                    else "npt" if group.kind == "npt_program" else group.kind
                )
                for group in self.condition_groups
            )
        )


__all__ = [
    "CellProgramEndpoint",
    "ExplicitSamplingPlan",
    "NebSamplingParameters",
    "neb_algorithm_identity",
    "neb_constraint_identity",
    "neb_parameters_payload",
    "RelaxSamplingParameters",
    "relax_algorithm_identity",
    "relax_constraint_identity",
    "relax_parameters_payload",
    "ResolvedSamplingTask",
    "SamplingConditionGroup",
    "SamplingRound",
    "SeedRole",
    "SelectionGroup",
    "SelectionPlan",
    "TemperatureProgram",
    "TemperatureProgramSegment",
]
