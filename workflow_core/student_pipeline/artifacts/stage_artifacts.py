from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from workflow_core.student_pipeline.config.stage_config import canonical_stage_type
from workflow_core.student_pipeline.config.train_regime import resolve_train_decision
from workflow_core.student_pipeline.artifacts.successful_stage_resolver import SuccessfulStageArtifactResolver
from workflow_artifacts.first_principles.handoff import first_principles_handoff_input_path


@dataclass(frozen=True, slots=True)
class ArtifactResolutionContext:
    teacher_round: int
    student_round: int
    stage_run_index: int
    stage_type: str
    canonical_stage_type: str
    template_name: str
    stage_dir: Path
    stage_results_dir: Path
    config_path: Path
    stage_cfg: dict[str, Any]
    cfg: dict[str, Any]
    status: dict[str, Any]
    paths: Any
    current_stage: dict[str, Any]
    current_invocation: dict[str, Any]
    invocations: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class RelabelStageInputs:
    primary_input_xyz: Path
    failed_backflow_input_xyz: Path | None


@dataclass(frozen=True, slots=True)
class RelabelStagePublishedOutputs:
    student_training_labeled_xyz: Path
    student_training_dataset_dir: Path
    teacher_dft_candidate_xyz: Path
    label_results_dir: Path
    active_learning_dir: Path


@dataclass(frozen=True, slots=True)
class RelabelArtifactContext:
    primary_input_xyz: Path
    failed_backflow_input_xyz: Path | None
    label_results_dir: Path
    active_learning_dir: Path


@dataclass(frozen=True, slots=True)
class RelabelStageArtifacts:
    stage_type: str
    inputs: RelabelStageInputs
    published_outputs: RelabelStagePublishedOutputs
    adapter_context: RelabelArtifactContext


@dataclass(frozen=True, slots=True)
class TrainStageInputs:
    dataset_path: Path
    dataset_roots: tuple[Path, ...]
    previous_committee_path: Path | None


@dataclass(frozen=True, slots=True)
class TrainStagePublishedOutputs:
    committee_dir: Path
    committee_summary_path: Path
    train_results_dir: Path


@dataclass(frozen=True, slots=True)
class TrainArtifactContext:
    dataset_path: Path
    dataset_roots: tuple[Path, ...]
    previous_committee_path: Path | None
    committee_summary_path: Path
    source_dataset_path: Path | None


@dataclass(frozen=True, slots=True)
class TrainStageArtifacts:
    stage_type: str
    inputs: TrainStageInputs
    published_outputs: TrainStagePublishedOutputs
    adapter_context: TrainArtifactContext


@dataclass(frozen=True, slots=True)
class SamplingStageInputs:
    committee_summary_path: Path | None
    seed_pool_path: Path


@dataclass(frozen=True, slots=True)
class SamplingStagePublishedOutputs:
    selected_seed_xyz: Path
    student_candidate_xyz: Path
    student_failed_xyz: Path
    seed_pool_after_sampling_xyz: Path


@dataclass(frozen=True, slots=True)
class SamplingArtifactContext:
    committee_summary_path: Path | None
    selected_seed_xyz: Path
    source_seed_pool_xyz: Path
    sampling_work_dir: Path


@dataclass(frozen=True, slots=True)
class SamplingStageArtifacts:
    stage_type: str
    inputs: SamplingStageInputs
    published_outputs: SamplingStagePublishedOutputs
    adapter_context: SamplingArtifactContext


@dataclass(frozen=True, slots=True)
class FilteringInputs:
    candidate_xyz: Path
    failed_xyz: Path


@dataclass(frozen=True, slots=True)
class FilteringPublishedOutputs:
    filtered_candidate_xyz: Path
    filtered_failed_xyz: Path


@dataclass(frozen=True, slots=True)
class StudentRelabelInputFilteringStageArtifacts:
    stage_type: str
    inputs: FilteringInputs
    published_outputs: FilteringPublishedOutputs
    adapter_context: None


@dataclass(frozen=True, slots=True)
class SeedPoolRefreshInputs:
    seed_pool_xyz: Path
    candidate_increment_xyz: Path
    failed_increment_xyz: Path


@dataclass(frozen=True, slots=True)
class SeedPoolRefreshPublishedOutputs:
    seed_pool_xyz: Path


@dataclass(frozen=True, slots=True)
class SeedPoolRefreshStageArtifacts:
    stage_type: str
    inputs: SeedPoolRefreshInputs
    published_outputs: SeedPoolRefreshPublishedOutputs
    adapter_context: None


@dataclass(frozen=True, slots=True)
class TeacherDftCandidateFilteringInputs:
    teacher_dft_candidates_xyz: Path


@dataclass(frozen=True, slots=True)
class TeacherDftCandidateFilteringPublishedOutputs:
    screened_dft_candidates_xyz: Path
    first_principles_input_xyz: Path


@dataclass(frozen=True, slots=True)
class TeacherDftCandidateFilteringStageArtifacts:
    stage_type: str
    inputs: TeacherDftCandidateFilteringInputs
    published_outputs: TeacherDftCandidateFilteringPublishedOutputs
    adapter_context: None


ResolvedStageArtifacts = (
    RelabelStageArtifacts
    | TrainStageArtifacts
    | SamplingStageArtifacts
    | StudentRelabelInputFilteringStageArtifacts
    | SeedPoolRefreshStageArtifacts
    | TeacherDftCandidateFilteringStageArtifacts
)


def build_artifact_resolution_context(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
    stage_cfg: dict[str, Any],
) -> ArtifactResolutionContext:
    student_round = int(stage_context["student_round"])
    stage_run_index = int(stage_context["stage_run_index"])
    current_invocation = next(
        invocation
        for invocation in list(runtime_state.get("invocations", []) or [])
        if int(invocation.get("student_round", 0)) == student_round
    )
    current_stage = next(
        stage
        for stage in list(current_invocation.get("stages", []) or [])
        if int(stage.get("stage_run_index", 0)) == stage_run_index
    )
    stage_type = str(stage_context["stage"])
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("student pipeline runtime_state is missing paths")
    stage_dir = paths.student_stage_artifacts_dir(
        int(stage_context["teacher_round"]),
        student_round,
        stage_run_index,
        canonical_stage_type(stage_type),
    ).resolve()
    return ArtifactResolutionContext(
        teacher_round=int(stage_context["teacher_round"]),
        student_round=student_round,
        stage_run_index=stage_run_index,
        stage_type=stage_type,
        canonical_stage_type=canonical_stage_type(stage_type),
        template_name=str(current_invocation.get("template_name", stage_context.get("template_name", ""))),
        stage_dir=stage_dir,
        stage_results_dir=stage_dir,
        config_path=Path(str(stage_context["config"])).resolve(),
        stage_cfg=dict(stage_cfg or {}),
        cfg=dict(runtime_state.get("cfg", {}) or {}),
        status=dict(runtime_state.get("status", {}) or {}),
        paths=paths,
        current_stage=dict(current_stage),
        current_invocation=dict(current_invocation),
        invocations=tuple(dict(invocation) for invocation in list(runtime_state.get("invocations", []) or [])),
    )


def resolve_stage_artifacts(context: ArtifactResolutionContext) -> ResolvedStageArtifacts:
    stage_type = canonical_stage_type(context.stage_type)
    if stage_type == "relabel":
        return _resolve_relabel_artifacts(context)
    if stage_type == "train":
        return _resolve_train_artifacts(context)
    if stage_type == "sampling":
        return _resolve_sampling_artifacts(context)
    if stage_type == "student_relabel_input_filtering":
        return _resolve_filtering_artifacts(context)
    if stage_type == "seed_pool_refresh":
        return _resolve_seed_pool_refresh_artifacts(context)
    if stage_type == "teacher_dft_candidate_filtering":
        return _resolve_teacher_dft_candidate_filtering_artifacts(context)
    raise ValueError(f"unsupported canonical student pipeline stage type: {context.stage_type!r}")


def published_output_paths(artifacts: ResolvedStageArtifacts) -> list[Path]:
    return _collect_paths(getattr(artifacts, "published_outputs", None), include_dirs=False)


def _resolve_relabel_artifacts(context: ArtifactResolutionContext) -> RelabelStageArtifacts:
    published_outputs = RelabelStagePublishedOutputs(
        student_training_labeled_xyz=(context.stage_dir / "label" / "train.xyz").resolve(),
        student_training_dataset_dir=(context.stage_dir / "label").resolve(),
        teacher_dft_candidate_xyz=(context.stage_dir / "active_learning" / "dft.xyz").resolve(),
        label_results_dir=(context.stage_dir / "label").resolve(),
        active_learning_dir=(context.stage_dir / "active_learning").resolve(),
    )
    primary_input = _resolve_relabel_primary_input(context)
    failed_input = _resolve_relabel_failed_backflow_input(context)
    inputs = RelabelStageInputs(
        primary_input_xyz=primary_input,
        failed_backflow_input_xyz=failed_input,
    )
    return RelabelStageArtifacts(
        stage_type="relabel",
        inputs=inputs,
        published_outputs=published_outputs,
        adapter_context=RelabelArtifactContext(
            primary_input_xyz=primary_input,
            failed_backflow_input_xyz=failed_input,
            label_results_dir=published_outputs.label_results_dir,
            active_learning_dir=published_outputs.active_learning_dir,
        ),
    )


def _resolve_train_artifacts(context: ArtifactResolutionContext) -> TrainStageArtifacts:
    decision = resolve_train_decision(
        train_regime=dict(context.cfg.get("train_regime", {}) or {}),
        is_entry_train=bool(context.teacher_round == 0 and context.student_round == 1),
        train_recipe_ref=str(context.stage_cfg["train_recipe_ref"]),
    )
    if decision.dataset_mode == "incremental":
        previous_relabel = _find_previous_stage(context, "relabel")
        if previous_relabel is None:
            raise ValueError("train artifact resolution could not find previous relabel stage for incremental mode")
        dataset_path = (context.stage_dir / "input" / "train.xyz").resolve()
        previous_relabel_label_dir = (_stage_artifact_dir_for_entry(context, previous_relabel) / "label").resolve()
        source_dataset_path = (previous_relabel_label_dir / "train.xyz").resolve()
        dataset_roots = (previous_relabel_label_dir,)
    else:
        dataset_path = (context.stage_dir / "input" / "train.xyz").resolve()
        source_dataset_path = dataset_path
        dataset_roots = (dataset_path,)
    latest_committee = SuccessfulStageArtifactResolver(_resolver_runtime_state(context)).latest_train_committee_before(
        context.current_stage
    )
    previous_committee_path = latest_committee.model_root if decision.requires_previous_committee and latest_committee is not None else None
    published_outputs = TrainStagePublishedOutputs(
        committee_dir=(context.stage_results_dir / "student_committee").resolve(),
        committee_summary_path=(context.stage_results_dir / "student_committee_summary.json").resolve(),
        train_results_dir=context.stage_results_dir.resolve(),
    )
    return TrainStageArtifacts(
        stage_type="train",
        inputs=TrainStageInputs(
            dataset_path=dataset_path,
            dataset_roots=dataset_roots,
            previous_committee_path=previous_committee_path,
        ),
        published_outputs=published_outputs,
        adapter_context=TrainArtifactContext(
            dataset_path=dataset_path,
            dataset_roots=dataset_roots,
            previous_committee_path=previous_committee_path,
            committee_summary_path=published_outputs.committee_summary_path,
            source_dataset_path=source_dataset_path,
        ),
    )


def _resolve_sampling_artifacts(context: ArtifactResolutionContext) -> SamplingStageArtifacts:
    latest_committee = SuccessfulStageArtifactResolver(_resolver_runtime_state(context)).latest_train_committee_before(
        context.current_stage
    )
    committee_summary_path = latest_committee.committee_summary_path if latest_committee is not None else None
    published_outputs = SamplingStagePublishedOutputs(
        selected_seed_xyz=(context.stage_results_dir / "seed_pool.xyz").resolve(),
        student_candidate_xyz=(context.stage_results_dir / "candidate.xyz").resolve(),
        student_failed_xyz=(context.stage_results_dir / "failed.xyz").resolve(),
        seed_pool_after_sampling_xyz=(context.stage_results_dir / "seed_pool_after_sampling.xyz").resolve(),
    )
    return SamplingStageArtifacts(
        stage_type="sampling",
        inputs=SamplingStageInputs(
            committee_summary_path=committee_summary_path,
            seed_pool_path=(context.stage_dir / "input" / "seed_pool.xyz").resolve(),
        ),
        published_outputs=published_outputs,
        adapter_context=SamplingArtifactContext(
            committee_summary_path=committee_summary_path,
            selected_seed_xyz=published_outputs.selected_seed_xyz,
            source_seed_pool_xyz=(context.stage_dir / "input" / "seed_pool.xyz").resolve(),
            sampling_work_dir=context.stage_results_dir.resolve(),
        ),
    )


def _resolve_filtering_artifacts(context: ArtifactResolutionContext) -> StudentRelabelInputFilteringStageArtifacts:
    return StudentRelabelInputFilteringStageArtifacts(
        stage_type="student_relabel_input_filtering",
        inputs=FilteringInputs(
            candidate_xyz=(context.stage_dir / "input" / "candidate.xyz").resolve(),
            failed_xyz=(context.stage_dir / "input" / "failed.xyz").resolve(),
        ),
        published_outputs=FilteringPublishedOutputs(
            filtered_candidate_xyz=(context.stage_results_dir / "filtered_candidate.xyz").resolve(),
            filtered_failed_xyz=(context.stage_results_dir / "filtered_failed.xyz").resolve(),
        ),
        adapter_context=None,
    )


def _resolve_seed_pool_refresh_artifacts(context: ArtifactResolutionContext) -> SeedPoolRefreshStageArtifacts:
    return SeedPoolRefreshStageArtifacts(
        stage_type="seed_pool_refresh",
        inputs=SeedPoolRefreshInputs(
            seed_pool_xyz=(context.stage_dir / "input" / "seed_pool.xyz").resolve(),
            candidate_increment_xyz=(context.stage_dir / "input" / "candidate_increment.xyz").resolve(),
            failed_increment_xyz=(context.stage_dir / "input" / "failed_increment.xyz").resolve(),
        ),
        published_outputs=SeedPoolRefreshPublishedOutputs(
            seed_pool_xyz=(context.stage_results_dir / "seed_pool.xyz").resolve(),
        ),
        adapter_context=None,
    )


def _resolve_teacher_dft_candidate_filtering_artifacts(
    context: ArtifactResolutionContext,
) -> TeacherDftCandidateFilteringStageArtifacts:
    return TeacherDftCandidateFilteringStageArtifacts(
        stage_type="teacher_dft_candidate_filtering",
        inputs=TeacherDftCandidateFilteringInputs(
            teacher_dft_candidates_xyz=(context.stage_dir / "input" / "teacher_dft_candidates.xyz").resolve(),
        ),
        published_outputs=TeacherDftCandidateFilteringPublishedOutputs(
            screened_dft_candidates_xyz=(context.stage_results_dir / "screened_dft_candidates.xyz").resolve(),
            first_principles_input_xyz=first_principles_handoff_input_path(context.paths, context.teacher_round),
        ),
        adapter_context=None,
    )


def _resolve_relabel_primary_input(context: ArtifactResolutionContext) -> Path:
    return (context.stage_dir / "input" / "relabel_input.xyz").resolve()


def _resolve_relabel_failed_backflow_input(context: ArtifactResolutionContext) -> Path | None:
    failed_backflow = dict(context.stage_cfg.get("failed_backflow", {}) or {})
    if not bool(failed_backflow.get("enabled", False)):
        return None
    return (context.stage_dir / "input" / "failed_backflow_input.xyz").resolve()


def _find_previous_stage(context: ArtifactResolutionContext, canonical_type: str) -> dict[str, Any] | None:
    current_index = int(context.stage_run_index)
    previous: dict[str, Any] | None = None
    for stage in list((context.current_invocation.get("stages") or [])):
        stage_index = int(stage.get("stage_run_index", 0))
        if stage_index >= current_index:
            continue
        if canonical_stage_type(str(stage.get("stage", ""))) != canonical_type:
            continue
        previous = dict(stage)
    return previous


def _stage_artifact_dir_for_entry(context: ArtifactResolutionContext, stage: dict[str, Any]) -> Path:
    return context.paths.student_stage_artifacts_dir(
        context.teacher_round,
        context.student_round,
        int(stage.get("stage_run_index", 0)),
        canonical_stage_type(str(stage.get("stage", ""))),
    ).resolve()


def _resolver_runtime_state(context: ArtifactResolutionContext) -> dict[str, Any]:
    return {
        "paths": context.paths,
        "invocations": [dict(invocation) for invocation in context.invocations],
    }


def _collect_paths(value: Any, *, include_dirs: bool, field_name: str | None = None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        if not include_dirs and field_name is not None and field_name.endswith("_dir"):
            return []
        return [value.resolve()]
    if is_dataclass(value):
        collected: list[Path] = []
        for field in fields(value):
            collected.extend(_collect_paths(getattr(value, field.name), include_dirs=include_dirs, field_name=field.name))
        return collected
    if isinstance(value, (list, tuple)):
        collected: list[Path] = []
        for item in value:
            collected.extend(_collect_paths(item, include_dirs=include_dirs))
        return collected
    return []
