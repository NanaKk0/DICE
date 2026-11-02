from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from workflow_artifacts.lifecycle.sampling_models import EffectiveSamplingModelPlan
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
from workflow_backends.lammps.observation_source import (
    LammpsObservationSource,
    reconstruct_legacy_lammps_contexts,
    validate_lammps_sample_block,
)
from workflow_backends.lammps.sampler import LammpsSampler
from workflow_backends.sampling import (
    SamplingBackendBinding,
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingExecutionResult,
    SamplingObservationResult,
    SamplingValidationResult,
    observe_path_backed_sampling_evidence,
)
from workflow_core.lifecycle.sampling import SamplingEvidenceFacts


@dataclass(frozen=True, slots=True)
class LammpsSamplingBackendFactory:
    """Bind the selected LAMMPS executor to route-resolved sampling assets."""

    executor: Any

    def bind(
        self,
        binding: SamplingBackendBinding,
    ) -> LammpsSamplingNativeOps:
        if binding.widom_group_ids:
            from workflow_backends.widom.runtime import SubprocessWidomSamplingNativeOps
            from workflow_backends.hybrid_sampling import WidomMixedNativeOps
            widom = SubprocessWidomSamplingNativeOps(getattr(self.executor, 'context', None),
                guest_sources={k: v for k, v in binding.guest_inputs.items() if k in binding.widom_group_ids},
                type_map=binding.type_map, executor=self.executor)
            remaining = {k: v for k, v in binding.guest_inputs.items() if k not in binding.widom_group_ids}
            ordinary = self.bind(replace(binding, guest_inputs=remaining, widom_group_ids=())) if remaining or binding.recipes else None
            return WidomMixedNativeOps(widom=widom, md=ordinary)
        if binding.guest_inputs:
            from workflow_backends.gcmc.runtime import SubprocessGcmcSamplingNativeOps
            ordinary = tuple(r for r in binding.recipes if r.condition_group_id not in binding.guest_inputs)
            gcmc = SubprocessGcmcSamplingNativeOps(
                getattr(self.executor, 'context', None), guest_sources=binding.guest_inputs,
                type_map=binding.type_map,
                md_recipes={r.condition_group_id: r.path for r in binding.recipes if r not in ordinary},
                executor=self.executor,
            )
            if ordinary:
                from workflow_backends.hybrid_sampling import GcmcMixedNativeOps
                return GcmcMixedNativeOps(gcmc=gcmc, md=self.bind(replace(binding, recipes=ordinary, guest_inputs={})))
            return gcmc
        grouped = any(
            recipe.condition_group_id is not None for recipe in binding.recipes
        )
        if grouped:
            if any(recipe.condition_group_id is None for recipe in binding.recipes):
                raise ValueError("LAMMPS explicit recipes must all bind condition groups")
            return LammpsSamplingNativeOps(
                executor=self.executor,
                type_map=binding.type_map,
                group_recipes={
                    str(recipe.condition_group_id): recipe.path
                    for recipe in binding.recipes
                },
                plumed_sources=binding.system_inputs,
            )
        single = len(binding.recipes) == 1 and binding.recipes[0].ensemble is None
        if not single and any(recipe.ensemble is None for recipe in binding.recipes):
            raise ValueError(
                "LAMMPS sampling requires either one common recipe or ensemble recipes"
            )
        recipe_arguments: dict[str, object]
        if single:
            recipe_arguments = {"recipe": binding.recipes[0].path}
        else:
            recipe_arguments = {
                "recipes": {
                    str(recipe.ensemble): recipe.path for recipe in binding.recipes
                }
            }
        return LammpsSamplingNativeOps(
            executor=self.executor,
            type_map=binding.type_map,
            plumed_sources=binding.system_inputs,
            **recipe_arguments,
        )


@dataclass(slots=True)
class LammpsSamplingNativeOps:
    """LAMMPS-native operations over coordinator-resolved model inputs."""

    executor: Any
    type_map: tuple[str, ...]
    recipe: Path | None = None
    recipes: Mapping[str, Path] | None = None
    group_recipes: Mapping[str, Path] | None = None
    plumed_sources: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        declarations = sum(
            item is not None for item in (self.recipe, self.recipes, self.group_recipes)
        )
        if declarations != 1:
            raise ValueError("LAMMPS sampling backend requires exactly one recipe declaration")
        self.type_map = tuple(str(symbol) for symbol in self.type_map)
        if not self.type_map:
            raise ValueError("LAMMPS sampling backend requires a non-empty type map")
        if self.recipe is not None:
            self.recipe = Path(self.recipe).resolve()
        if self.recipes is not None:
            self.recipes = MappingProxyType(
                {str(ensemble): Path(path).resolve() for ensemble, path in self.recipes.items()}
            )
        if self.group_recipes is not None:
            self.group_recipes = MappingProxyType(
                {
                    str(group_id): Path(path).resolve()
                    for group_id, path in self.group_recipes.items()
                }
            )
        self.plumed_sources = MappingProxyType(
            {str(system_id): Path(path).resolve() for system_id, path in self.plumed_sources.items()}
        )

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        evidence = request.evidence
        native_models = (
            evidence.models
            if isinstance(evidence.models, EffectiveSamplingModelPlan)
            else evidence.model_paths
        )
        LammpsSampler(
            run_json_path=evidence.run_json_path,
            iter_index=evidence.iter_index,
            round_index=evidence.round_index,
            executor=self.executor,
        ).sample(
            **self._recipe_arguments(),
            sample_dir=evidence.sample_dir,
            models=native_models,
            type_map=self.type_map,
            systems=evidence.seed_mapping,
            steps=evidence.steps,
            plan=evidence.plan,
            plumed_sources=dict(self.plumed_sources),
            seed_selections=evidence.seed_selections,
            progress_callback=request.progress_callback,
            progress_reporter_factory=request.progress_reporter_factory,
        )
        return SamplingExecutionResult(
            sample_dir=evidence.sample_dir,
            failed_task_count=LammpsObservationSource(
                sample_dir=evidence.sample_dir, run_json_path=evidence.run_json_path,
                iter_index=evidence.iter_index, round_index=evidence.round_index,
                type_map=self.type_map, model_count=len(evidence.model_paths),
            ).failed_task_count,
            task_count=(
                sum(
                    task.kind not in {"neb", "relax", "gcmc_md", "widom"}
                    for task in evidence.plan.tasks_for_round(evidence.round_index)
                )
                if isinstance(evidence.plan, ExplicitSamplingPlan)
                else evidence.plan.total_tasks
            ),
        )

    def observe_evidence(
        self, request: SamplingEvidenceRequest
    ) -> SamplingEvidenceFacts:
        return observe_path_backed_sampling_evidence(request)

    def open_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        return SamplingObservationResult(
            LammpsObservationSource(
                sample_dir=request.sample_dir,
                run_json_path=request.run_json_path,
                iter_index=request.iter_index,
                round_index=request.round_index,
                type_map=self.type_map,
                model_count=len(request.model_paths),
                plan=(
                    request.plan
                    if isinstance(request.plan, ExplicitSamplingPlan)
                    else None
                ),
                accepted_kinds=(
                    ("nvt", "npt", "nvt_program", "nvt_cell_program", "npt_program", "md_mc")
                    if isinstance(request.plan, ExplicitSamplingPlan)
                    else None
                ),
            )
        )

    def validate_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingValidationResult:
        model_evidence = self._model_evidence(request)
        arguments: dict[str, Any] = {
            **self._recipe_arguments(),
            "sample_dir": request.sample_dir,
            "run_json_path": request.run_json_path,
            "iter_index": request.iter_index,
            "round_index": request.round_index,
            "models": request.model_paths,
            "systems": request.seed_mapping,
            "steps": request.steps,
            "plan": request.plan,
            "accepted_kinds": (
                ("nvt", "npt", "nvt_program", "nvt_cell_program", "npt_program", "md_mc")
                if isinstance(request.plan, ExplicitSamplingPlan)
                else None
            ),
            "plumed_sources": dict(self.plumed_sources),
            "seed_selections": request.seed_selections,
            **model_evidence,
        }
        context = getattr(self.executor, "context", None)
        if model_evidence["require_native_execution"] and context is not None:
            arguments["expected_environment_profile_id"] = getattr(
                context, "profile_id", None
            )
        return SamplingValidationResult(validate_lammps_sample_block(**arguments))

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        contexts = reconstruct_legacy_lammps_contexts(
            **self._recipe_arguments(),
            sample_dir=request.sample_dir,
            run_json_path=request.run_json_path,
            iter_index=request.iter_index,
            round_index=request.round_index,
            models=request.model_paths,
            type_map=self.type_map,
            systems=request.seed_mapping,
            steps=request.steps,
            plan=request.plan,
            plumed_sources=dict(self.plumed_sources),
        )
        return SamplingObservationResult(
            LammpsObservationSource(
                sample_dir=request.sample_dir,
                run_json_path=request.run_json_path,
                iter_index=request.iter_index,
                round_index=request.round_index,
                type_map=self.type_map,
                model_count=len(request.model_paths),
                contexts=contexts,
            ),
            historical=True,
        )

    def _recipe_arguments(self) -> dict[str, object]:
        if self.recipe is not None:
            return {"recipe": self.recipe}
        if self.group_recipes is not None:
            return {"group_recipes": dict(self.group_recipes)}
        return {"recipes": dict(self.recipes or {})}

    @staticmethod
    def _model_evidence(request: SamplingEvidenceRequest) -> dict[str, object]:
        if not isinstance(request.models, EffectiveSamplingModelPlan):
            return {
                "deepmd_recipe_kind": None,
                "require_native_execution": False,
            }
        recipe_kinds = {model.deepmd_recipe_kind for model in request.models.models}
        if len(recipe_kinds) != 1:
            return {
                "deepmd_recipe_kind": None,
                "require_native_execution": True,
            }
        recipe_kind = next(iter(recipe_kinds))
        return {
            "deepmd_recipe_kind": recipe_kind,
            "require_native_execution": request.models.compressed or recipe_kind == "dpa4c",
        }


__all__ = ["LammpsSamplingBackendFactory", "LammpsSamplingNativeOps"]
