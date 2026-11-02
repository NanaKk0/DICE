from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow_backends.gpumd.observation_source import (
    GpumdObservationSource,
    reconstruct_legacy_gpumd_contexts,
    validate_gpumd_sample_block,
)
from workflow_backends.gpumd.sampler import GpumdSampler
from workflow_core.common.sampling_plan import ExplicitSamplingPlan
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
class GpumdSamplingBackendFactory:
    """Bind the selected GPUMD executor to its single native recipe."""

    executor: Any

    def bind(
        self,
        binding: SamplingBackendBinding,
    ) -> GpumdSamplingNativeOps:
        if any(recipe.condition_group_id is not None for recipe in binding.recipes):
            if any(recipe.condition_group_id is None for recipe in binding.recipes):
                raise ValueError("GPUMD explicit recipes must all bind condition groups")
            return GpumdSamplingNativeOps(
                executor=self.executor,
                group_recipes={
                    str(recipe.condition_group_id): recipe.path
                    for recipe in binding.recipes
                },
            )
        if len(binding.recipes) != 1 or binding.recipes[0].ensemble is not None:
            raise ValueError("GPUMD sampling requires exactly one common recipe")
        if binding.system_inputs:
            raise ValueError("GPUMD sampling does not accept auxiliary system inputs")
        return GpumdSamplingNativeOps(
            executor=self.executor,
            recipe=binding.recipes[0].path,
        )


@dataclass(slots=True)
class GpumdSamplingNativeOps:
    """GPUMD-native operations over coordinator-resolved model inputs."""

    executor: Any
    recipe: Path | None = None
    group_recipes: dict[str, Path] | None = None

    def __post_init__(self) -> None:
        if (self.recipe is None) == (self.group_recipes is None):
            raise ValueError("GPUMD sampling requires one recipe declaration")
        if self.recipe is not None:
            self.recipe = Path(self.recipe).resolve()
        if self.group_recipes is not None:
            self.group_recipes = {
                str(key): Path(value).resolve()
                for key, value in self.group_recipes.items()
            }

    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        evidence = request.evidence
        GpumdSampler(
            run_json_path=evidence.run_json_path,
            iter_index=evidence.iter_index,
            round_index=evidence.round_index,
            executor=self.executor,
        ).sample(
            recipe=self.recipe,
            group_recipes=self.group_recipes,
            sample_dir=evidence.sample_dir,
            models=evidence.model_paths,
            systems=evidence.seed_mapping,
            steps=evidence.steps,
            plan=evidence.plan,
            plumed_sources={},
            progress_callback=request.progress_callback,
            progress_reporter_factory=request.progress_reporter_factory,
        )
        return SamplingExecutionResult(
            sample_dir=evidence.sample_dir,
            task_count=(
                sum(
                    task.kind != "neb"
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
            GpumdObservationSource(
                sample_dir=request.sample_dir,
                run_json_path=request.run_json_path,
                iter_index=request.iter_index,
                round_index=request.round_index,
                plan=(request.plan if isinstance(request.plan, ExplicitSamplingPlan) else None),
            )
        )

    def validate_current(
        self, request: SamplingEvidenceRequest
    ) -> SamplingValidationResult:
        return SamplingValidationResult(
            validate_gpumd_sample_block(
                sample_dir=request.sample_dir,
                run_json_path=request.run_json_path,
                iter_index=request.iter_index,
                round_index=request.round_index,
                models=request.model_paths,
                systems=request.seed_mapping,
                steps=request.steps,
                plan=request.plan,
                group_recipes=self.group_recipes,
                accepted_kinds=(
                    ("nvt", "npt")
                    if isinstance(request.plan, ExplicitSamplingPlan)
                    else None
                ),
            )
        )

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        contexts = reconstruct_legacy_gpumd_contexts(
            sample_dir=request.sample_dir,
            run_json_path=request.run_json_path,
            iter_index=request.iter_index,
            round_index=request.round_index,
            models=request.model_paths,
            systems=request.seed_mapping,
            steps=request.steps,
            plan=request.plan,
        )
        return SamplingObservationResult(
            GpumdObservationSource(
                sample_dir=request.sample_dir,
                run_json_path=request.run_json_path,
                iter_index=request.iter_index,
                round_index=request.round_index,
                contexts=contexts,
            ),
            historical=True,
        )

__all__ = ["GpumdSamplingBackendFactory", "GpumdSamplingNativeOps"]
