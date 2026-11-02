from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import warnings
from typing import Any, TextIO

from workflow_core.structures.formats import (
    read_extxyz_structures,
    validate_extxyz_file,
    write_extxyz_structures,
)
from workflow_core.lifecycle.data import (
    DataService,
    LabelContract,
    TeacherDatasetMaterialization,
)
from workflow_core.lifecycle.screen import publish_fp

from workflow_core.lifecycle.launch import LifecycleLaunch
from workflow_core.lifecycle.composition import BackendComposition, bind_lifecycle_screen
from workflow_core.lifecycle.process_manager import ProcessManager
from workflow_core.lifecycle.process_manager import ResumeFeasibility
from workflow_core.lifecycle.state_model import (
    CompletionEvidence,
    CompletionObservation,
    CompletionStatus,
    PublicBlockId,
    PublicLifecyclePlan,
    PublicResultDependencies,
)
from workflow_core.lifecycle.teacher_facade import LifecycleTeacherFacade
from workflow_core.lifecycle.use_cases import (
    PublicBlockOutcome,
    PublicBlockUseCases,
)
from workflow_core.lifecycle.routes import (
    EvidenceKind,
    RouteKind,
    RouteSpec,
    ScreenArtifactContract,
)
from workflow_runtime.lifecycle.state import (
    ArtifactCheck,
    LifecycleBlockObservation,
)


BlockHandler = Callable[[Path], object]


def archive_public_lifecycle_higher_round_evidence(
    launch: LifecycleLaunch,
    *,
    after_round: int,
) -> Path:
    """Archive an explicitly retired round suffix after public preflight."""

    blocks = tuple(block.block_id for block in launch.plan.blocks)
    artifact_checks = _artifact_checks(launch.config, blocks=blocks)
    state = _adapt_compatibility_state(launch, blocks, artifact_checks)
    return state.archive_higher_round_evidence(after_round=after_round)


def execute_public_lifecycle(
    launch: LifecycleLaunch,
    *,
    backend_composition: BackendComposition,
) -> None:
    """Compose the iter-local public blocks without entering the legacy route."""
    _execute_public_lifecycle(
        launch,
        backend_composition=backend_composition,
        scheduled_block=None,
    )


def execute_public_lifecycle_block(
    launch: LifecycleLaunch,
    *,
    block_id: str,
    backend_composition: BackendComposition,
) -> None:
    """Execute exactly one validated public block for a scheduled worker."""

    _execute_public_lifecycle(
        launch,
        backend_composition=backend_composition,
        scheduled_block=PublicBlockId(block_id),
    )


def _execute_public_lifecycle(
    launch: LifecycleLaunch,
    *,
    backend_composition: BackendComposition,
    scheduled_block: PublicBlockId | None,
) -> None:
    lifecycle_launch = launch
    config = lifecycle_launch.config
    backend_composition = bind_lifecycle_screen(config, backend_composition)
    teacher_contract = LabelContract()
    student_contract = LabelContract()
    data = DataService(
        run_json_path=config.path,
        iter_index=config.iter_index,
        teacher_type_map=tuple(config.teacher["type_map"]),
        student_type_map=tuple(config.student["type_map"]),
        teacher_contract=teacher_contract,
        student_contract=student_contract,
    )
    teacher = backend_composition.teacher
    sampling = backend_composition.sampling
    student_training_validators = _StudentTrainingValidatorMap(
        config,
        backend_composition.student,
    )
    block_ids = tuple(block.block_id for block in lifecycle_launch.plan.blocks)
    artifact_checks = _artifact_checks(
        config,
        blocks=block_ids,
        composition=backend_composition,
        data=data,
        student=backend_composition.student,
        sampling_backend=sampling,
        student_training_validators=student_training_validators,
    )
    handlers = _handlers(
        lifecycle_launch,
        data,
        teacher,
        backend_composition,
        sampling,
        student_training_validators,
    )
    visualization_worker = None
    if config.visualize_enabled:
        from workflow_runtime.lifecycle.visualization import VisualizationWorker

        visualization_worker = VisualizationWorker()
        # Scientific handlers only retain a borrowed descriptor draft. Dispatch
        # happens after canonical publication, and preparation runs in the child.
        for block_id, handler in tuple(handlers.items()):
            if block_id.startswith('r') and block_id.endswith('/screen'):
                def with_draft(stage, report, handler=handler):
                    result = handler(stage, report)
                    draft = getattr(result, 'visualization_draft', None)
                    if draft is not None:
                        try:
                            visualization_worker.defer(Path(stage).parent, draft)
                        except Exception as exc:
                            warnings.warn(f'selection visualization skipped: {exc}', RuntimeWarning)
                    return result
                handlers[block_id] = with_draft
    try:
        public_plan = PublicLifecyclePlan(
            tuple(PublicBlockId(block_id) for block_id in block_ids)
        )
    except ValueError:
        # A few long-standing injected tests deliberately construct a one-block
        # launch that is not a legal public lifecycle.  Preserve that private
        # compatibility seam until Child G; every complete public plan takes
        # the strict Process Manager path below.
        if block_ids not in {
            ("teach/train",),
            ("teach/train", "teach/scan"),
        }:
            raise
        try:
            execute_lifecycle(
                lifecycle_launch,
                handlers={block_id: handlers[block_id] for block_id in block_ids},
                artifact_checks={
                    block_id: artifact_checks[block_id] for block_id in block_ids
                },
                visualization_worker=visualization_worker,
            )
        except BaseException:
            if visualization_worker is not None:
                visualization_worker.cancel()
            raise
        else:
            if visualization_worker is not None:
                for message in visualization_worker.join():
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    state_controller = _public_state_binding_controller(
        lifecycle_launch,
        public_plan,
        artifact_checks=artifact_checks,
        composition=backend_composition,
        data=data,
        student_training_validators=student_training_validators,
    )
    try:
        execute_lifecycle(
            lifecycle_launch,
            handlers=handlers,
            artifact_checks=artifact_checks,
            process_manager=ProcessManager(
                public_plan,
                _public_result_dependencies(lifecycle_launch, public_plan),
            ),
            use_cases=PublicBlockUseCases.for_plan(public_plan),
            runtime_factory=backend_composition.lifecycle_runtime_factory,
            state_controller=state_controller,
            scheduled_block=scheduled_block,
            visualization_worker=visualization_worker,
        )
    except BaseException:
        if visualization_worker is not None:
            visualization_worker.cancel()
        raise
    else:
        if visualization_worker is not None:
            for message in visualization_worker.join():
                warnings.warn(message, RuntimeWarning, stacklevel=2)


def observe_public_lifecycle_block(
    launch: LifecycleLaunch,
    block: str,
    *,
    backend_composition: BackendComposition | None = None,
) -> LifecycleBlockObservation:
    """Return DICE-owned completion evidence for one public lifecycle block."""
    lifecycle_launch = launch
    artifact_checks = _artifact_checks(
        launch.config,
        blocks=tuple(item.block_id for item in launch.plan.blocks),
        composition=backend_composition,
    )
    public_plan = PublicLifecyclePlan(
        tuple(PublicBlockId(item.block_id) for item in launch.plan.blocks)
    )
    controller = _public_state_binding_controller(
        lifecycle_launch,
        public_plan,
        artifact_checks=artifact_checks,
        composition=backend_composition or BackendComposition,
        data=None,
        student_training_validators=None,
    )
    return controller.refresh_for(PublicBlockId(block)).observe(block)


def reconcile_public_lifecycle_read_only(launch: LifecycleLaunch) -> str | None:
    """Resolve the required terminal-result suffix without mutation."""

    blocks = tuple(block.block_id for block in launch.plan.blocks)
    artifact_checks = _artifact_checks(launch.config, blocks=blocks)
    state = _adapt_compatibility_state(launch, blocks, artifact_checks)
    plan = PublicLifecyclePlan(tuple(PublicBlockId(block) for block in blocks))
    manager = ProcessManager(plan, _public_result_dependencies(launch, plan))
    observations = _completion_observations(state, blocks, manager=manager,
        resume_target=None if launch.resume_target is None else PublicBlockId(launch.resume_target))
    reconciliation = manager.reconcile(
        observations,
        resume_target=(
            None
            if launch.resume_target is None
            else PublicBlockId(launch.resume_target)
        ),
    )
    if reconciliation.blocked:
        raise ValueError(
            next(
                decision.reason
                for decision in reconciliation.decisions
                if decision.disposition.value == "blocked"
            )
        )
    return (
        None
        if not reconciliation.blocks_to_run
        else str(reconciliation.blocks_to_run[0])
    )


def probe_public_lifecycle_resume(
    launch: LifecycleLaunch,
    target: str,
) -> ResumeFeasibility:
    """Evaluate exact-resume result dependencies without mutation or launch."""

    blocks = tuple(block.block_id for block in launch.plan.blocks)
    plan = PublicLifecyclePlan(tuple(PublicBlockId(block) for block in blocks))
    target_block = PublicBlockId(target)
    plan.index(target_block)
    state = _adapt_compatibility_state(
        launch,
        blocks,
        _artifact_checks(launch.config, blocks=blocks),
    )
    manager = ProcessManager(plan, _public_result_dependencies(launch, plan))
    return manager.probe(
        _completion_observations(state, blocks, manager=manager, resume_target=target_block),
        resume_target=target_block,
    )


def _completion_observations(
    state: Any,
    blocks: tuple[str, ...],
    *, manager: ProcessManager | None = None,
    resume_target: PublicBlockId | None = None,
) -> tuple[CompletionObservation, ...]:
    if manager is not None:
        def observe(block):
            raw = state.reconciliation_observation(str(block))
            return CompletionObservation(block, CompletionStatus(raw.status), raw.reason,
                                         CompletionEvidence(raw.evidence))
        return manager.observe_required(observe, resume_target=resume_target)
    return tuple(
        CompletionObservation(
            PublicBlockId(block),
            CompletionStatus(raw.status),
            raw.reason,
            CompletionEvidence(raw.evidence),
        )
        for block in blocks
        for raw in (state.reconciliation_observation(block),)
    )


def _public_result_dependencies(
    launch: LifecycleLaunch,
    plan: PublicLifecyclePlan,
) -> PublicResultDependencies:
    policy = str(launch.config.raw["fp"]["policy"])
    route = launch.route
    return PublicResultDependencies.for_plan(
        plan,
        fp_all_rounds=bool(launch.config.raw["fp"].get("all_rn", False)),
        fp_requires_teacher_train=policy
        in {"JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"},
        fp_requires_teacher_scan=bool(
            route is not None
            and route.teacher_requires_reference_tail
            and policy in {"DIRECT", "FPS"}
        ),
    )


def _binding_specs(
    launch: LifecycleLaunch,
    blocks: tuple[PublicBlockId, ...],
    *,
    artifact_validators: Mapping[str, object],
) -> tuple[tuple[PublicBlockId, object], ...]:
    return tuple(
        (
            block,
            artifact_validators[str(block)],
        )
        for block in blocks
    )


def _compatibility_state_binding_controller(
    launch: LifecycleLaunch,
    plan: PublicLifecyclePlan,
    *,
    artifact_checks: Mapping[str, ArtifactCheck],
):
    return BackendComposition.build_lifecycle_state_controller(
        root=launch.run_root,
        iter_index=launch.config.iter_index,
        plan=plan,
        binding_specs_factory=lambda _prefix: _binding_specs(
            launch,
            plan.blocks,
            artifact_validators=artifact_checks,
        ),
    )


def _public_state_binding_controller(
    launch: LifecycleLaunch,
    plan: PublicLifecyclePlan,
    *,
    artifact_checks: Mapping[str, ArtifactCheck],
    composition: BackendComposition,
    data: DataService,
    student_training_validators: "_StudentTrainingValidatorMap",
):
    def specs(completed_prefix: tuple[PublicBlockId, ...]):
        validators: dict[str, object] = dict(artifact_checks)
        current_block = (
            plan.blocks[len(completed_prefix)]
            if len(completed_prefix) < len(plan.blocks)
            else None
        )
        label_block = PublicBlockId("fp/label")
        if label_block == current_block and "label" in launch.config.raw["fp"]:
            selected_path = (
                launch.config.root
                / f"iter{launch.config.iter_index}"
                / "fp"
                / "selected.xyz"
            )
            if selected_path.is_file():
                _selection, expectation = _fp_label_expectation(
                    launch.config,
                    selected_path,
                )
                validators["fp/label"] = composition.bind_fp_label_artifact_validator(
                    expectation,
                    selected_path=selected_path,
                    calculator=launch.config.raw["fp"]["label"]["calculator"],
                )
        return _binding_specs(
            launch,
            plan.blocks,
            artifact_validators=validators,
        )

    return BackendComposition.build_lifecycle_state_controller(
        root=launch.run_root,
        iter_index=launch.config.iter_index,
        plan=plan,
        binding_specs_factory=specs,
    )


def _adapt_compatibility_state(
    launch: LifecycleLaunch,
    blocks: tuple[str, ...],
    artifact_checks: Mapping[str, ArtifactCheck],
):
    semantic_blocks = tuple(PublicBlockId(block) for block in blocks)
    return BackendComposition.adapt_compatibility_lifecycle_state(
        root=launch.run_root,
        iter_index=launch.config.iter_index,
        binding_specs=_binding_specs(
            launch,
            semantic_blocks,
            artifact_validators=artifact_checks,
        ),
    )


def execute_lifecycle(
    launch: LifecycleLaunch,
    *,
    handlers: Mapping[str, BlockHandler],
    artifact_checks: Mapping[str, ArtifactCheck],
    progress_terminal: TextIO | None = None,
    process_manager: ProcessManager | None = None,
    use_cases: PublicBlockUseCases | None = None,
    runtime_factory: Callable[..., Any] | None = None,
    state_controller: Any | None = None,
    scheduled_block: PublicBlockId | None = None,
    visualization_worker: Any | None = None,
) -> None:
    """Execute/reconcile the public lifecycle blocks without invoking legacy planning."""
    blocks = tuple(block.block_id for block in launch.plan.blocks)
    _validate_execution_coverage(
        blocks,
        handlers=handlers,
        artifact_checks=artifact_checks,
        process_manager=process_manager,
        use_cases=use_cases,
        runtime_factory=runtime_factory,
    )
    if process_manager is not None:
        if use_cases is None or runtime_factory is None:
            raise AssertionError("validated Process Manager requires use cases and runtime")
        if state_controller is None:
            state_controller = _compatibility_state_binding_controller(
                launch,
                process_manager.plan,
                artifact_checks=artifact_checks,
            )
        _execute_managed_lifecycle(
            launch,
            state_controller=state_controller,
            blocks=blocks,
            handlers=handlers,
            process_manager=process_manager,
            use_cases=use_cases,
            runtime_factory=runtime_factory,
            progress_terminal=progress_terminal,
            scheduled_block=scheduled_block,
            visualization_worker=visualization_worker,
        )
        return
    if scheduled_block is not None:
        raise ValueError("scheduled public block execution requires Process Manager control")
    state = _adapt_compatibility_state(launch, blocks, artifact_checks)
    block = state.reconcile(resume=launch.resume_target)
    from workflow_runtime.lifecycle.progress import AttemptDiagnosticManager, ProgressCoordinator

    completed = blocks if block is None else blocks[: blocks.index(block)]
    progress = ProgressCoordinator(
        iter_root=state.iter_root,
        iter_index=launch.config.iter_index,
        plan=blocks,
        completed=completed,
        ready=block,
        terminal=progress_terminal if progress_terminal is not None else sys.stdout,
    )
    progress.start(resume_target=launch.resume_target)
    from workflow_runtime.lifecycle.screen_reporting import ScreenReportCoordinator

    route = getattr(launch.config, "route", None)
    screen_reports = ScreenReportCoordinator(
        iter_root=state.iter_root,
        iter_index=launch.config.iter_index,
        warning=progress.degradation_warning,
        expected_route=(
            None
            if route is None
            else route.report_identity()
        ),
        system_ids=_configured_screen_report_system_ids(launch),
    )
    report_blocks = _screen_report_blocks(launch, blocks)
    try:
        screen_reports.regenerate_prefix(
            blocks=blocks,
            target=block,
            report_blocks=report_blocks,
        )
    except Exception as exc:
        progress.degradation_warning(exc)
    try:
        screen_reports.ensure_header()
    except Exception as exc:
        progress.degradation_warning(exc)
    attempt_diagnostics = AttemptDiagnosticManager(iter_root=state.iter_root)
    for stale_block in blocks:
        stale_path = state.staging_path(stale_block)
        if not stale_path.exists():
            continue
        try:
            sealed_stale = attempt_diagnostics.archive_stale(
                block=stale_block,
                staging_path=stale_path,
            )
            shutil.rmtree(stale_path, ignore_errors=True)
            progress.diagnostic_archived(
                stale_block,
                attempt_id=sealed_stale.attempt_id,
                diagnostic_ref=sealed_stale.diagnostic_ref,
            )
            for pruned in attempt_diagnostics.rotate(
                block=stale_block,
                newest_attempt_id=sealed_stale.attempt_id,
            ):
                progress.diagnostic_pruned(stale_block, attempt_id=pruned)
        except Exception as stale_error:
            progress.degradation_warning(stale_error)
            shutil.rmtree(stale_path, ignore_errors=True)
    while block is not None:
        progress.block_started(block)
        progress.phase_entered(block, "executing")
        from workflow_core.lifecycle.progress import CoreBlockProgress

        core_progress = CoreBlockProgress(coordinator=progress, block=block)
        diagnostic_ref: str | None = None
        terminal_event_emitted = False
        sealed_attempt_id: str | None = None

        def seal_failure(staging: Path, exc: BaseException, *, block: str = block) -> None:
            nonlocal diagnostic_ref, sealed_attempt_id, terminal_event_emitted
            try:
                sealed = attempt_diagnostics.seal(
                    block=block,
                    staging_path=staging,
                    attempt_id=progress.attempt_id,
                    outcome="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=exc,
                    apply_retention=False,
                )
                diagnostic_ref = sealed.diagnostic_ref
                sealed_attempt_id = sealed.attempt_id
            except Exception as seal_error:
                progress.degradation_warning(seal_error)
            if isinstance(exc, KeyboardInterrupt):
                progress.block_interrupted(
                    block, message="interrupted", diagnostic_ref=diagnostic_ref
                )
            else:
                progress.block_failed(block, message=str(exc), diagnostic_ref=diagnostic_ref)
            terminal_event_emitted = True

        try:
            state.publish(
                block,
                lambda staging, handler=handlers[block], reporter=core_progress: _invoke_block_handler(
                    handler, staging, reporter
                ),
                phase_callback=lambda phase, block=block: progress.phase_entered(block, phase),
                failure_callback=seal_failure,
            )
        except KeyboardInterrupt:
            if not terminal_event_emitted:
                progress.block_interrupted(block, message="interrupted", diagnostic_ref=diagnostic_ref)
            if sealed_attempt_id is not None:
                for pruned in attempt_diagnostics.rotate(block=block, newest_attempt_id=sealed_attempt_id):
                    progress.diagnostic_pruned(block, attempt_id=pruned)
            raise
        except Exception as exc:
            if not terminal_event_emitted:
                progress.block_failed(block, message=str(exc), diagnostic_ref=diagnostic_ref)
            if sealed_attempt_id is not None:
                for pruned in attempt_diagnostics.rotate(block=block, newest_attempt_id=sealed_attempt_id):
                    progress.diagnostic_pruned(block, attempt_id=pruned)
            raise
        if block == "teach/screen" or (block.startswith("r") and block.endswith("/screen")):
            snapshot_path = state.block_path(block) / "screen_report.json"
            if snapshot_path.is_file():
                try:
                    screen_reports.append_snapshot(snapshot_path, block=block)
                except Exception as exc:
                    progress.degradation_warning(exc)
        _enqueue_visualization_after_success(
            visualization_worker, state=state, block=block
        )
        progress.block_succeeded(block, summary=core_progress.final_summary)
        if block == "teach/scan" or block.endswith("/train") or block == "fp":
            state = _adapt_compatibility_state(launch, blocks, artifact_checks)
        if block.endswith("/screen") and block.startswith("r"):
            state.remove_retired_sample_observation_index(block)
        block = state.reconcile()


def _execute_managed_lifecycle(
    launch: LifecycleLaunch,
    *,
    state_controller: Any,
    blocks: tuple[str, ...],
    handlers: Mapping[str, BlockHandler],
    process_manager: ProcessManager,
    use_cases: PublicBlockUseCases,
    runtime_factory: Callable[..., Any],
    progress_terminal: TextIO | None,
    scheduled_block: PublicBlockId | None,
    visualization_worker: Any | None,
) -> None:
    """Keep effects in the shell while Process Manager owns the public loop."""

    from workflow_core.lifecycle.progress import CoreBlockProgress
    from workflow_runtime.lifecycle.progress import (
        AttemptDiagnosticManager,
        ProgressCoordinator,
    )
    from workflow_runtime.lifecycle.screen_reporting import ScreenReportCoordinator

    runtime_context: dict[str, Any] = {}
    reporters: dict[str, CoreBlockProgress] = {}

    def reporter_factory(block_id: PublicBlockId) -> CoreBlockProgress:
        block = str(block_id)
        progress: ProgressCoordinator = runtime_context["progress"]
        progress.block_started(block)
        progress.phase_entered(block, "executing")
        reporter = CoreBlockProgress(coordinator=progress, block=block)
        reporters[block] = reporter
        return reporter

    def phase_callback(block_id: PublicBlockId, phase: str) -> None:
        progress: ProgressCoordinator = runtime_context["progress"]
        progress.phase_entered(str(block_id), phase)

    def failure_callback(
        block_id: PublicBlockId,
        staging: Path,
        exc: BaseException,
    ) -> None:
        block = str(block_id)
        progress: ProgressCoordinator = runtime_context["progress"]
        attempts: AttemptDiagnosticManager = runtime_context["attempts"]
        diagnostic_ref: str | None = None
        sealed_attempt_id: str | None = None
        try:
            sealed = attempts.seal(
                block=block,
                staging_path=staging,
                attempt_id=progress.attempt_id,
                outcome=(
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                ),
                error=exc,
                apply_retention=False,
            )
            diagnostic_ref = sealed.diagnostic_ref
            sealed_attempt_id = sealed.attempt_id
        except Exception as seal_error:
            progress.degradation_warning(seal_error)
        if isinstance(exc, KeyboardInterrupt):
            progress.block_interrupted(
                block,
                message="interrupted",
                diagnostic_ref=diagnostic_ref,
            )
        else:
            progress.block_failed(
                block,
                message=str(exc),
                diagnostic_ref=diagnostic_ref,
            )
        if sealed_attempt_id is not None:
            for pruned in attempts.rotate(
                block=block,
                newest_attempt_id=sealed_attempt_id,
            ):
                progress.diagnostic_pruned(block, attempt_id=pruned)

    def success_callback(outcome: PublicBlockOutcome) -> None:
        block = str(outcome.block)
        state = state_controller.current_state
        progress: ProgressCoordinator = runtime_context["progress"]
        screen_reports: ScreenReportCoordinator = runtime_context["screen_reports"]
        if block == "teach/screen" or (
            block.startswith("r") and block.endswith("/screen")
        ):
            snapshot_path = state.block_path(block) / "screen_report.json"
            if snapshot_path.is_file():
                try:
                    screen_reports.append_snapshot(snapshot_path, block=block)
                except Exception as exc:
                    progress.degradation_warning(exc)
        progress.block_succeeded(block, summary=reporters[block].final_summary)
        if block.startswith("r") and block.endswith("/screen"):
            state.remove_retired_sample_observation_index(block)
        _enqueue_visualization_after_success(
            visualization_worker, state=state, block=block
        )

    runtime = runtime_factory(
        state=None,
        state_controller=state_controller,
        handlers=handlers,
        reporter_factory=reporter_factory,
        phase_callback=phase_callback,
        failure_callback=failure_callback,
        success_callback=success_callback,
    )
    resume_target = (
        scheduled_block
        if scheduled_block is not None
        else (
            None
            if launch.resume_target is None
            else PublicBlockId(launch.resume_target)
        )
    )
    reconciliation = process_manager.prepare(
        runtime,
        resume_target=resume_target,
    )
    state = state_controller.current_state
    run_blocks = tuple(str(block) for block in reconciliation.blocks_to_run)
    ready = None if not run_blocks else run_blocks[0]
    completed = tuple(
        str(decision.block)
        for decision in reconciliation.decisions
        if decision.disposition.value == "complete"
    )
    progress = ProgressCoordinator(
        iter_root=state.iter_root,
        iter_index=launch.config.iter_index,
        plan=blocks,
        completed=completed,
        ready=ready,
        terminal=progress_terminal if progress_terminal is not None else sys.stdout,
    )
    progress.start(resume_target=launch.resume_target)

    route = getattr(launch.config, "route", None)
    screen_reports = ScreenReportCoordinator(
        iter_root=state.iter_root,
        iter_index=launch.config.iter_index,
        warning=progress.degradation_warning,
        expected_route=None if route is None else route.report_identity(),
        system_ids=_configured_screen_report_system_ids(launch),
    )
    report_blocks = _screen_report_blocks(launch, blocks)
    try:
        screen_reports.regenerate_prefix(
            blocks=blocks,
            target=ready,
            report_blocks=report_blocks,
        )
    except Exception as exc:
        progress.degradation_warning(exc)
    try:
        screen_reports.ensure_header()
    except Exception as exc:
        progress.degradation_warning(exc)
    attempts = AttemptDiagnosticManager(iter_root=state.iter_root)
    runtime_context.update(
        progress=progress,
        screen_reports=screen_reports,
        attempts=attempts,
    )
    for stale_block in blocks:
        stale_path = state.staging_path(stale_block)
        if not stale_path.exists():
            continue
        try:
            sealed_stale = attempts.archive_stale(
                block=stale_block,
                staging_path=stale_path,
            )
            shutil.rmtree(stale_path, ignore_errors=True)
            progress.diagnostic_archived(
                stale_block,
                attempt_id=sealed_stale.attempt_id,
                diagnostic_ref=sealed_stale.diagnostic_ref,
            )
            for pruned in attempts.rotate(
                block=stale_block,
                newest_attempt_id=sealed_stale.attempt_id,
            ):
                progress.diagnostic_pruned(stale_block, attempt_id=pruned)
        except Exception as stale_error:
            progress.degradation_warning(stale_error)
            shutil.rmtree(stale_path, ignore_errors=True)
    if scheduled_block is None:
        process_manager.run_prepared(
            reconciliation,
            use_cases=use_cases,
            runtime=runtime,
        )
    else:
        process_manager.run_one_prepared(
            reconciliation,
            block=scheduled_block,
            use_cases=use_cases,
            runtime=runtime,
        )


def _validate_execution_coverage(
    blocks: tuple[str, ...],
    *,
    handlers: Mapping[str, BlockHandler],
    artifact_checks: Mapping[str, ArtifactCheck],
    process_manager: ProcessManager | None,
    use_cases: PublicBlockUseCases | None,
    runtime_factory: Callable[..., Any] | None,
) -> None:
    expected = set(blocks)
    handler_keys = set(handlers)
    artifact_keys = set(artifact_checks)
    if handler_keys != expected or artifact_keys != expected:
        details: list[str] = []
        if handler_keys != expected:
            details.append("handlers")
        if artifact_keys != expected:
            details.append("artifact checks")
        raise ValueError(
            "lifecycle lifecycle requires exact unique coverage for: "
            + ", ".join(details)
        )
    if (process_manager is None) != (use_cases is None):
        raise ValueError("process manager and public block use cases must be supplied together")
    if process_manager is None:
        if runtime_factory is not None:
            raise ValueError("synthetic lifecycle compatibility cannot select a runtime factory")
        try:
            PublicLifecyclePlan(tuple(PublicBlockId(block) for block in blocks))
        except ValueError:
            return
        raise ValueError("a full public lifecycle requires Process Manager control")
    if runtime_factory is None:
        raise ValueError("Process Manager control requires a lifecycle runtime factory")
    semantic_blocks = tuple(PublicBlockId(block) for block in blocks)
    if process_manager.plan.blocks != semantic_blocks:
        raise ValueError("process manager plan does not match the prepared lifecycle plan")
    if use_cases is None or use_cases.plan != process_manager.plan:
        raise ValueError("public block use cases do not match the process manager plan")


def _invoke_block_handler(handler: BlockHandler, staging: Path, reporter: Any) -> object:
    """Call the core reporter seam while preserving one-argument injected tests."""
    import inspect

    parameters = tuple(inspect.signature(handler).parameters.values())
    if len(parameters) >= 2 and parameters[1].name == "report":
        return handler(staging, reporter)
    return handler(staging)


def _screen_report_blocks(launch: LifecycleLaunch, blocks: tuple[str, ...]) -> tuple[str, ...]:
    """Return report-bearing screen blocks without making diagnostics authoritative."""
    route = getattr(launch.config, "route", None)
    result: list[str] = []
    for block in blocks:
        if block.startswith("r") and block.endswith("/screen"):
            result.append(block)
        elif block == "teach/screen":
            if route is None or route.teacher_evidence_kind is EvidenceKind.FORCE_DEVIATION:
                result.append(block)
    return tuple(result)


def _configured_screen_report_system_ids(
    launch: LifecycleLaunch,
) -> tuple[str, ...] | Mapping[str, tuple[str, ...]] | None:
    """Resolve report rows from selection groups, falling back to declarations."""
    raw_sample = getattr(launch.config, "raw", {}).get("sample", {})
    declarations = (
        raw_sample.get("systems", raw_sample.get("seeds", {}))
        if isinstance(raw_sample, dict)
        else {}
    )
    if not declarations:
        return None
    sampling_plan = getattr(launch.config, "sampling_plan", None)
    if bool(getattr(sampling_plan, "explicit", False)):
        return {
            f"r{round_index}/screen": (
                sampling_plan.selection_plan_for_round(round_index).group_ids
            )
            for round_index in range(launch.config.round_count)
        }
    selection_plan = getattr(launch.config, "selection_plan", None)
    return (
        tuple(selection_plan.group_ids)
        if selection_plan is not None
        else tuple(declarations)
    )


def _enqueue_visualization_after_success(
    worker: Any | None,
    *,
    state: Any,
    block: str,
) -> None:
    try:
        _enqueue_visualization_after_success_unchecked(worker, state=state, block=block)
    except Exception as exc:
        warnings.warn(f'selection visualization skipped: {type(exc).__name__}: {exc}', RuntimeWarning)


def _enqueue_visualization_after_success_unchecked(worker, *, state, block):
    if not (block == "fp" or (block.startswith("r") and block.endswith("/screen"))):
        return
    block_path = state.block_path(block)
    owner = block_path if block == "fp" else block_path.parent
    if worker is not None and callable(getattr(worker, 'release_pending', None)):
        if worker.release_pending(owner):
            return
    output = owner / "visualize"
    source_cache = block_path / ".visualize_cache"
    durable_cache = owner / ".visualize_cache"
    shutil.rmtree(output, ignore_errors=True)
    (owner / "visualize.progress.jsonl").unlink(missing_ok=True)
    if durable_cache != source_cache:
        shutil.rmtree(durable_cache, ignore_errors=True)
    if worker is None or not source_cache.is_dir():
        return
    if durable_cache != source_cache:
        shutil.move(str(source_cache), str(durable_cache))
    worker.enqueue(durable_cache, output)


def _enqueue_cached_visualizations(worker: Any, *, config: Any) -> None:
    """Regenerate missing/stale diagnostics from compatible durable caches."""

    owners = [
        config.root / f"iter{config.iter_index}" / f"r{index}"
        for index in range(int(config.round_count))
    ]
    owners.append(config.root / f"iter{config.iter_index}" / "fp")
    for owner in owners:
        cache = owner / ".visualize_cache"
        if not cache.is_dir():
            continue
        try:
            from workflow_artifacts.lifecycle.visualization import (
                load_visualization_request,
                visualization_output_is_current,
            )

            request = load_visualization_request(cache)
            output = owner / "visualize"
            if visualization_output_is_current(request, output):
                continue
            (owner / "visualize.progress.jsonl").unlink(missing_ok=True)
            worker.enqueue(cache, output)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"selection visualization cache skipped: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _handlers(
    launch: LifecycleLaunch,
    data: DataService,
    teacher: LifecycleTeacherFacade,
    composition: BackendComposition,
    sampling: Any,
    student_training_validators: "_StudentTrainingValidatorMap",
) -> dict[str, BlockHandler]:
    config = launch.config
    route = _resolved_route(config)
    raw = config.raw
    teacher_spec = config.teacher
    student_spec = config.student
    teacher_readout = teacher
    teacher_label = teacher

    from workflow_runtime.lifecycle.progress import (
        BasicBlockSummary,
        DatasetContribution,
        DatasetSnapshot,
        FPBlockSummary,
        TrainBlockSummary,
    )

    def teach_train(stage: Path, report: Any) -> None:
        block_started = time.monotonic()
        bootstrap_sources = [config.root / path for path in raw["teacher_train_bootstrap"]]
        fp_sources: list[Path] = []
        if config.iter_index:
            fp_sources = [config.root / path for path in raw["fp_data"]]
        training_intent, training_expectation = _teacher_training_expectation(
            config,
            data,
        )
        dataset = data.materialize_teacher_dataset(
            bootstrap_sources,
            stage / "dataset" / "train.xyz",
            valid_frac=float(teacher_spec["valid_frac"]),
            fp_sources=fp_sources,
        )
        if _materialized_teacher_dataset_handle(
            dataset,
            dataset_id=training_intent.dataset.dataset_id,
        ) != training_intent.dataset:
            raise ValueError(
                "materialized teacher dataset changed canonical membership or order"
            )
        split = dict(dataset.split_summary)
        source_counts = dict(split["source_counts"])
        frame_count = int(split["frame_count"])
        train_count = sum(int(row["train_count"]) for row in dict(split["systems"]).values())
        valid_count = sum(
            int(row["held_out_count"]) for row in dict(split["systems"]).values()
        )
        report.snapshot(
            DatasetSnapshot(
                label="dataset",
                unique=frame_count,
                train=train_count,
                valid=valid_count,
                systems=len(dict(split["systems"])),
                contributions=(
                    DatasetContribution(
                        label="bootstrap",
                        count=int(source_counts["teacher_train_bootstrap"]["frame_count"]),
                    ),
                    DatasetContribution(
                        label="fp_data",
                        count=int(source_counts["fp_data"]["frame_count"]),
                    ),
                ),
                deduplicated=int(split["duplicate_drop_count"]),
            )
        )
        report.phase("training")
        _typed_teacher_train(
            composition,
            teacher_spec,
            data,
            dataset,
            stage,
            training_intent,
            training_expectation,
            report,
        )
        report.summary(
            TrainBlockSummary(
                backend=str(teacher_spec["backend"]),
                models=int(teacher_spec.get("members", 1)),
                dataset_unique=frame_count,
                train=train_count,
                valid=valid_count,
                elapsed_seconds=time.monotonic() - block_started,
            )
        )

    def teach_screen(stage: Path, report: Any) -> None:
        source_groups: list[list[Any]]
        if config.iter_index == 0:
            source_groups = [
                data.load_student_bootstrap(
                    [config.root / path for path in raw["student_train_bootstrap"]]
                )
            ]
        else:
            # Later iters relabel the completed prior iter's direct final student
            # predecessor. Prior external FP candidates never enter this chain.
            prior_root = config.root / f"iter{config.iter_index - 1}"
            prior_train, prior_screen = _prior_iter_student_train_sources(prior_root)
            source_groups = [
                read_extxyz_structures(prior_train, allow_empty=False),
                read_extxyz_structures(prior_screen, allow_empty=True),
            ]
        if student_spec.get("include_teacher_train_data") is True:
            source_groups.append(
                read_extxyz_structures(
                    config.root
                    / f"iter{config.iter_index}"
                    / "teach"
                    / "train"
                    / "dataset"
                    / "train.xyz",
                    allow_empty=False,
                )
            )
            frames = data.merge_student_relabel_sources(source_groups)
        else:
            frames = [frame for source in source_groups for frame in source]
        teacher_uses_force_deviation = (
            route.teacher_evidence_kind is EvidenceKind.FORCE_DEVIATION
        )
        labeled = _teacher_label(
            teacher_label,
            _teacher_models(config, teacher),
            frames,
            stage,
            report=report,
            trusted_below=(
                float(teacher_spec["lo"])
                if config.iter_index == 0 and teacher_uses_force_deviation
                else None
            ),
        )
        if config.iter_index == 0 and teacher_uses_force_deviation:
            lo = float(teacher_spec["lo"])
            output = [
                frame
                for frame in labeled
                if float(frame.info.get("force_uncertainty", 0.0)) < lo
            ]
        else:
            output = labeled
        if teacher_uses_force_deviation:
            _publish_committee_teach_report(
                stage=stage,
                report=report,
                iter_index=config.iter_index,
                route=route,
                input_frames=frames,
                labeled_frames=labeled,
                lo=float(teacher_spec["lo"]),
                hi=float(teacher_spec["hi"]),
                published_train=len(output),
                published_fp_candidates=0,
            )
        else:
            _report_screen_noop(report)
        write_extxyz_structures(stage / "train.xyz", output, allow_empty=False)
        write_extxyz_structures(stage / "fp_candidate.xyz", [], allow_empty=True)

    def teach_scan(stage: Path, report: Any) -> None:
        reference = config.root / f"iter{config.iter_index}" / "teach" / "train" / "dataset" / "train.xyz"
        from workflow_backends.teacher.readout import (
            TeacherReadoutReferenceRequest,
        )

        teacher_readout.fit_reference(
            TeacherReadoutReferenceRequest(
                model=_teacher_models(config, teacher)[0],
                reference_path=reference,
                router_dir=stage / "gmm_router",
                work_dir=stage / ".gmm_work",
                type_map=tuple(teacher_spec["type_map"]),
                progress_reporter_factory=lambda operation_id, phase: report.operation(
                    operation_id, phase=phase
                ),
            )
        )

    handlers: dict[str, BlockHandler] = {"teach/train": teach_train, "teach/screen": teach_screen}
    if route.has_teach_scan:
        handlers["teach/scan"] = teach_scan
    round_steps = raw["sample"].get("steps", [None] * config.round_count)
    for round_index, steps in enumerate(round_steps):
        train_block = f"r{round_index}/train"
        sample_block = f"r{round_index}/sample"
        screen_block = f"r{round_index}/screen"

        def train(stage: Path, report: Any, round_index: int = round_index) -> None:
            block_started = time.monotonic()
            iter_root = config.root / f"iter{config.iter_index}"
            if round_index == 0:
                sources = [
                    read_extxyz_structures(
                        iter_root / "teach" / "screen" / "train.xyz",
                        allow_empty=False,
                    )
                ]
                contributions = [
                    DatasetContribution(label="bootstrap", count=len(sources[0]))
                ]
            else:
                predecessor = round_index - 1
                sources = [
                    read_extxyz_structures(
                        iter_root
                        / f"r{predecessor}"
                        / "train"
                        / "dataset"
                        / "train.xyz",
                        allow_empty=False,
                    ),
                    read_extxyz_structures(
                        iter_root / f"r{predecessor}" / "screen" / "train.xyz",
                        allow_empty=True,
                    ),
                ]
                contributions = [
                    DatasetContribution(label="predecessor", count=len(sources[0])),
                    DatasetContribution(label=f"r{predecessor}", count=len(sources[1])),
                ]
            input_count = sum(len(source) for source in sources)
            dataset = data.materialize_student_dataset(
                sources,
                stage / "dataset" / "train.xyz",
            )
            dataset_frames = read_extxyz_structures(dataset, allow_empty=False)
            unique_count = len(dataset_frames)
            report.snapshot(
                DatasetSnapshot(
                    label="dataset",
                    unique=unique_count,
                    train=unique_count,
                    contributions=tuple(contributions),
                    deduplicated=input_count - unique_count,
                )
            )
            report.phase("training")
            from workflow_backends.student.role import StudentTrainRequest

            member_ids = _expected_member_ids("student", student_spec)
            progress_group_id = (
                "student-training"
                if len(member_ids) == 1
                else "student-committee-training"
            )
            expectation, validator = student_training_validators.bind(
                round_index,
                ordered_structure_ids=tuple(
                    str(frame.info["structure_id"]) for frame in dataset_frames
                ),
            )
            physical_request = StudentTrainRequest(
                dataset_path=dataset,
                train_dir=stage,
                member_ids=member_ids,
                progress_reporter_factory=lambda member_id: report.concurrent_operation(
                    member_id,
                    phase="training",
                    group_id=progress_group_id,
                ),
            )
            composition.student.execute_training_expectation(
                expectation,
                physical_request=physical_request,
                published_root=(
                    config.root
                    / f"iter{config.iter_index}"
                    / f"r{round_index}"
                    / "train"
                ),
            )
            if not validator(stage):
                raise ValueError(
                    "student train publication failed its canonical artifact expectation"
                )
            report.summary(
                TrainBlockSummary(
                    backend=str(student_spec["backend"]),
                    models=int(student_spec.get("members", 1)),
                    dataset_unique=unique_count,
                    train=unique_count,
                    evaluation_same_data=student_spec["backend"] == "NEP",
                    elapsed_seconds=time.monotonic() - block_started,
                )
            )

        def sample(stage: Path, report: Any, round_index: int = round_index, steps: int = steps, sample_block: str = sample_block) -> None:
            _sample(
                config,
                composition.student,
                sampling,
                round_index,
                steps,
                stage,
                report=report,
            )

        def screen(stage: Path, report: Any, round_index: int = round_index, sample_block: str = sample_block) -> None:
            source = _sample_observation_source(
                config, composition.student, sampling, round_index
            )
            resolved_sampling_plan = _sampling_plan_for_recovery(
                config, round_index
            )
            return composition.screen.execute(
                round_index=round_index,
                stage=stage,
                source=source,
                report=report,
                selection_plan=getattr(
                    resolved_sampling_plan,
                    "selection_plan",
                    None,
                ) if not hasattr(
                    resolved_sampling_plan, "selection_plan_for_round"
                ) else resolved_sampling_plan.selection_plan_for_round(
                    round_index
                ),
            )
        handlers[train_block] = train
        handlers[sample_block] = sample
        handlers[screen_block] = screen

    def fp(stage: Path, report: Any) -> None:
        from workflow_policies.resolution import resolve_element_direct_policy

        terminal_resolution = resolve_element_direct_policy(
            str(raw["fp"]["policy"]),
            descriptor_model_shape=str(config.teacher["model"]),
        )
        candidate_indices = (
            range(config.round_count)
            if raw["fp"]["all_rn"]
            else (config.round_count - 1,)
        )
        terminal_selection_plan = _terminal_selection_plan(
            config, tuple(candidate_indices)
        )
        candidates = [
            config.root / f"iter{config.iter_index}" / f"r{index}" / "screen" / "fp_candidate.xyz"
            for index in candidate_indices
        ]
        required_descriptor_schema = None
        if (
            route.teacher_requires_reference_tail
            and raw["fp"]["policy"] in {"DIRECT", "FPS"}
        ):
            from workflow_policies.gmm_reference_tail import validate_gmm_router

            required_descriptor_schema = dict(
                validate_gmm_router(
                    config.root / f"iter{config.iter_index}" / "teach" / "scan" / "gmm_router"
                )["descriptor_schema"]
            )
        report.phase("catalog")
        candidate_frames_by_round = [
            read_extxyz_structures(path, allow_empty=True) for path in candidates
        ]
        report.phase("deduplicate")
        unique_by_id = {
            str(frame.info["structure_id"]): frame
            for frames in candidate_frames_by_round
            for frame in frames
        }
        report.phase("select")
        terminal_policy = str(raw["fp"]["policy"])
        effective_terminal_policy = terminal_resolution.effective_policy
        if effective_terminal_policy in {
            "JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
        }:
            from workflow_core.lifecycle.screen import collect_final_fp_candidates
            from workflow_policies.final_fp import (
                select_final_fp_element_frames_with_outcomes,
                select_final_fp_element_joint_frames_with_outcomes,
                select_final_fp_joint_frames_with_outcomes,
            )

            merged, _batches = collect_final_fp_candidates(
                candidates, selection_plan=terminal_selection_plan
            )
            group_ids = (
                terminal_selection_plan.group_ids
                if terminal_selection_plan is not None
                else _selection_group_ids(config)
            )
            merged_frames = list(merged.values())
            candidate_vectors: dict[str, Any] = {}
            background_vectors: dict[str, Any] = {}
            candidate_element_vectors: dict[str, Any] = {}
            background_element_vectors: dict[str, Any] = {}
            descriptor_schema: dict[str, object] = {}
            outcomes: dict[str, dict[str, object]] = {}
            background: list[Any] = []
            if merged_frames:
                reference_path = (
                    config.root
                    / f"iter{config.iter_index}"
                    / "teach"
                    / "train"
                    / "dataset"
                    / "train.xyz"
                )
                if effective_terminal_policy != "ELEMENT_DIRECT":
                    background = read_extxyz_structures(
                        reference_path, allow_empty=False
                    )
                router_dir = (
                    config.root
                    / f"iter{config.iter_index}"
                    / "teach"
                    / "scan"
                    / "gmm_router"
                )
                manifest = composition.gmm_router_validator(router_dir)
                schema = manifest.get("descriptor_schema")
                if not isinstance(schema, dict):
                    raise ValueError("teacher joint DIRECT router omitted descriptor schema")
                models = _teacher_models(config, teacher)
                if len(models) != 1:
                    raise ValueError("teacher joint DIRECT requires one canonical model")
                readout = teacher.read_structure_vectors(
                    model=models[0],
                    candidate_frames=merged_frames,
                    background_frames=background,
                    reference_path=reference_path,
                    router_dir=router_dir,
                    work_dir=stage / ".joint_direct_work",
                    chunk_size=composition.gmm_score_chunk_size,
                    reference_descriptor_schema=schema,
                    report=report,
                    type_map=tuple(config.teacher["type_map"]),
                )
                candidate_vectors = dict(readout.candidate_vectors)
                background_vectors = dict(readout.background_vectors)
                descriptor_schema = dict(readout.descriptor_schema)
                candidate_element_vectors = dict(
                    readout.candidate_element_vectors or {}
                )
                background_element_vectors = dict(
                    readout.background_element_vectors or {}
                )
            if merged_frames and effective_terminal_policy == "JOINT_DIRECT":
                selected, outcomes = select_final_fp_joint_frames_with_outcomes(
                    merged_frames,
                    keep=int(raw["fp"]["keep"]),
                    candidate_vectors=candidate_vectors,
                    background_vectors=background_vectors,
                    group_ids=group_ids,
                )
            elif merged_frames and effective_terminal_policy == "ELEMENT_DIRECT":
                selected, outcomes = select_final_fp_element_frames_with_outcomes(
                    merged_frames,
                    keep=int(raw["fp"]["keep"]),
                    candidate_element_vectors=candidate_element_vectors,
                    type_map=tuple(config.teacher["type_map"]),
                    group_ids=group_ids,
                )
            elif merged_frames:
                selected, outcomes = (
                    select_final_fp_element_joint_frames_with_outcomes(
                        merged_frames,
                        keep=int(raw["fp"]["keep"]),
                        candidate_element_vectors=candidate_element_vectors,
                        background_frames=background,
                        background_element_vectors=background_element_vectors,
                        type_map=tuple(config.teacher["type_map"]),
                        group_ids=group_ids,
                    )
                )
            else:
                selected = []
                outcomes = {
                    group_id: {
                        "candidate_count": 0,
                        "background_count": 0,
                        "bypass": False,
                        "realized_clusters": 0,
                        "selected_background_count": 0,
                        "selected_candidate_count": 0,
                        "quota_completion_count": 0,
                        "underfill": (
                            int(raw["fp"]["keep"])
                            if effective_terminal_policy.startswith("ELEMENT_")
                            else 0
                        ),
                        "underfill_reason": "empty-candidates",
                        **(
                            {
                                "active_elements": (),
                                "inactive_elements": (),
                                "element_dimensions": {},
                                "presence_counts": {},
                            }
                            if effective_terminal_policy.startswith("ELEMENT_")
                            else {}
                        ),
                    }
                    for group_id in group_ids
                }
            selected_path = write_extxyz_structures(
                stage / "selected.xyz", selected, allow_empty=True
            )
            if merged_frames and effective_terminal_policy == "JOINT_DIRECT":
                ordered_candidate_vectors = {
                    str(frame.info["structure_id"]): candidate_vectors[
                        str(frame.info["structure_id"])
                    ]
                    for frame in merged_frames
                }
                composition.joint_direct_fp_publisher(
                    stage / "joint_direct_fp.npz",
                    candidates=merged_frames,
                    candidate_vectors=ordered_candidate_vectors,
                    background_vectors=background_vectors,
                    descriptor_schema=descriptor_schema,
                    model_identity=_expected_member_ids(
                        "teacher", config.teacher
                    )[0],
                    dataset_identity=f"iter{config.iter_index}/teach/train/dataset",
                    group_ids=group_ids,
                    keep=int(raw["fp"]["keep"]),
                    all_rn=bool(raw["fp"]["all_rn"]),
                    selected_ids=[str(frame.info["structure_id"]) for frame in selected],
                    outcomes=outcomes,
                )
                shutil.rmtree(stage / ".joint_direct_work", ignore_errors=True)
            elif merged_frames:
                composition.element_direct_fp_publisher(
                    stage / "element_direct_fp.npz",
                    candidates=merged_frames,
                    background_frames=background,
                    candidate_element_vectors=candidate_element_vectors,
                    background_element_vectors=background_element_vectors,
                    descriptor_schema=descriptor_schema,
                    type_map=tuple(config.teacher["type_map"]),
                    model_identity=_expected_member_ids(
                        "teacher", config.teacher
                    )[0],
                    dataset_identity=f"iter{config.iter_index}/teach/train/dataset",
                    policy=terminal_policy,
                    background_mode=(
                        "teacher-training"
                        if effective_terminal_policy == "ELEMENT_JOINT_DIRECT"
                        else "none"
                    ),
                    candidate_rounds=tuple(candidate_indices),
                    group_ids=group_ids,
                    keep=int(raw["fp"]["keep"]),
                    all_rn=bool(raw["fp"]["all_rn"]),
                    selected_ids=[
                        str(frame.info["structure_id"]) for frame in selected
                    ],
                    outcomes=outcomes,
                )
                shutil.rmtree(stage / ".joint_direct_work", ignore_errors=True)
            (stage / "selection_group_outcomes.json").write_text(
                json.dumps(
                    {
                        "schema": "dice.lifecycle.selection-group-fp-outcomes",
                        "schema_revision": (
                            4 if terminal_policy.startswith("ELEMENT_") else 3
                        ),
                        "selection_groups": list(group_ids),
                        "role_to_group": (
                            []
                            if terminal_selection_plan is None
                            else [list(item) for item in terminal_selection_plan.role_to_group]
                        ),
                        "policy": terminal_policy,
                        "keep": int(raw["fp"]["keep"]),
                        "rows": [
                            {"selection_group_id": group_id, **outcomes[group_id]}
                            for group_id in group_ids
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            selected_path = publish_fp(
                candidate_paths=candidates,
                output=stage / "selected.xyz",
                keep=int(raw["fp"]["keep"]),
                policy=effective_terminal_policy,
                entropy=f"iter{config.iter_index}/fp",
                required_frame_mean_descriptor_schema=required_descriptor_schema,
                selection_plan=terminal_selection_plan,
            )
        selected_frames = read_extxyz_structures(selected_path, allow_empty=True)
        if terminal_resolution.fell_back:
            from workflow_artifacts.lifecycle.policy_resolution import (
                publish_policy_resolution,
            )

            fallback_groups = (
                terminal_selection_plan.group_ids
                if terminal_selection_plan is not None
                else _selection_group_ids(config)
            )
            for group_id in fallback_groups:
                _report_warning(
                    report,
                    RuntimeError(
                        "fp.policy ELEMENT_DIRECT fell back to random for "
                        f"{group_id}: {terminal_resolution.fallback_reason}"
                    ),
                )
            publish_policy_resolution(
                stage / "policy_resolution.json",
                coordinate="fp.policy",
                requested_policy=terminal_resolution.requested_policy,
                effective_policy=terminal_resolution.effective_policy,
                fallback_reason=str(terminal_resolution.fallback_reason),
                group_ids=fallback_groups,
                candidates=tuple(unique_by_id.values()),
                selected=selected_frames,
                quota=int(raw["fp"]["keep"]),
                entropy=f"iter{config.iter_index}/fp",
            )
            (stage / "fp_summary.json").write_text(
                json.dumps(
                    {
                        "schema": "dice.lifecycle.fp-summary",
                        "schema_revision": 1,
                        "candidates": len(unique_by_id),
                        "unique": len(unique_by_id),
                        "selected": len(selected_frames),
                        "selector_resolution": {
                            "requested_policy": terminal_resolution.requested_policy,
                            "effective_policy": terminal_resolution.effective_policy,
                            "fallback_reason": terminal_resolution.fallback_reason,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if config.visualize_enabled:
            _prepare_fp_visualization(
                config=config,
                route=route,
                teacher=teacher,
                stage=stage,
                report=report,
                candidates=list(unique_by_id.values()),
                selected=selected_frames,
                chunk_size=composition.gmm_score_chunk_size,
                existing_readout=(
                    (
                        dict(candidate_vectors),
                        dict(background_vectors),
                        dict(descriptor_schema),
                        dict(candidate_element_vectors),
                        dict(background_element_vectors),
                    )
                    if terminal_policy in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}
                    and merged_frames
                    else None
                ),
            )
        report.phase("finalizing")
        report.snapshot(
            _fp_count_flow(
                _selection_group_ids(config),
                candidate_frames_by_round=candidate_frames_by_round,
                unique_frames=tuple(unique_by_id.values()),
                selected_frames=selected_frames,
            )
        )
        report.summary(
            FPBlockSummary(
                candidates=sum(len(frames) for frames in candidate_frames_by_round),
                unique=len(unique_by_id),
                selected=len(selected_frames),
                handoff_path="fp/selected.xyz",
                legal_empty=not selected_frames,
            )
        )

    handlers["fp"] = fp
    if "label" in raw["fp"]:
        if composition.first_principles is None:
            raise ValueError(
                "lifecycle fp/label execution requires an explicit first-principles service"
            )

        def fp_label(stage: Path, report: Any) -> None:
            from workflow_backends.first_principles.protocols import FPLabelRequest
            from workflow_core.lifecycle.fp_capabilities import (
                FirstPrinciplesLabelIntent,
            )

            label = raw["fp"]["label"]
            operation_role = f"first_principles.{label['calculator'].lower()}.label"
            selected_path = (
                config.root
                / f"iter{config.iter_index}"
                / "fp"
                / "selected.xyz"
            )
            operation_id = _fp_label_operation_id(
                selected_path,
                config.root / label["recipe"],
                calculator=str(label["calculator"]),
            )
            selection, expectation = _fp_label_expectation(config, selected_path)
            intent = FirstPrinciplesLabelIntent(
                selection=selection,
                calculator=label["calculator"],
                operation_role=operation_role,
                operation_id=operation_id,
            )
            report.phase("catalog")
            request = FPLabelRequest(
                selected_path=selected_path,
                task_root=stage / "tasks",
                recipe_path=config.root / label["recipe"],
                controller_root=config.root
                / ".dice_state"
                / "fp_runs"
                / f"iter{config.iter_index}"
                / operation_id,
                calculator=label["calculator"],
                operation_role=operation_role,
                operation_id=operation_id,
                operation_context=getattr(
                    composition.first_principles, "operation_context", None
                ),
            )
            report.phase("label")
            _outcome, summary_evidence = composition.execute_fp_label_expectation(
                expectation,
                intent=intent,
                physical_request=request,
                stage_root=stage,
            )
            report.phase("finalizing")
            report.summary(
                BasicBlockSummary(
                    text=json.dumps(dict(summary_evidence), sort_keys=True)
                )
            )

        handlers["fp/label"] = fp_label
    return handlers


def _publish_committee_teach_report(
    *,
    stage: Path,
    report: Any,
    iter_index: int,
    route: RouteSpec,
    input_frames: list[Any],
    labeled_frames: list[Any],
    lo: float,
    hi: float,
    published_train: int,
    published_fp_candidates: int,
) -> None:
    from workflow_artifacts.lifecycle.screen_report import publish_screen_report
    from workflow_core.lifecycle.screen_reporting import build_committee_teach_screen_report

    try:
        snapshot = build_committee_teach_screen_report(
            iter_index=iter_index,
            route=route,
            input_frames=input_frames,
            labeled_frames=labeled_frames,
            lo=lo,
            hi=hi,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        publish_screen_report(stage / "screen_report.json", snapshot)
    except Exception as exc:
        _report_warning(report, exc)


def _report_screen_noop(report: Any) -> None:
    callback = getattr(report, "screen_report_noop", None)
    if callable(callback):
        callback()


def _report_warning(report: Any, exc: Exception) -> None:
    callback = getattr(report, "warning", None)
    if callable(callback):
        callback(exc)


def _prepare_fp_visualization(
    *,
    config: Any,
    route: RouteSpec,
    teacher: Any,
    stage: Path,
    report: Any,
    candidates: list[Any],
    selected: list[Any],
    chunk_size: int,
    existing_readout: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, object],
        dict[str, Any],
        dict[str, Any],
    ]
    | None,
) -> None:
    policy = str(config.raw["fp"]["policy"])
    if (
        policy not in {
            "DIRECT", "FPS", "JOINT_DIRECT",
            "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
        }
        or config.teacher.get("model") != "single"
        or not route.teacher_requires_reference_tail
        or not candidates
    ):
        return
    work = stage / ".fp_visualize_work"
    try:
        from workflow_artifacts.lifecycle.visualization import (
            publish_visualization_request,
        )
        from workflow_core.lifecycle.visualization import (
            VisualizationLayer,
            build_visualization_request,
            element_rows_from_element_vectors,
            structure_rows_from_vectors,
        )

        reference_path = (
            config.root
            / f"iter{config.iter_index}"
            / "teach"
            / "train"
            / "dataset"
            / "train.xyz"
        )
        background = read_extxyz_structures(reference_path, allow_empty=False)
        if existing_readout is None:
            router_dir = (
                config.root
                / f"iter{config.iter_index}"
                / "teach"
                / "scan"
                / "gmm_router"
            )
            from workflow_artifacts.lifecycle.gmm_io import validate_gmm_router

            manifest = validate_gmm_router(router_dir)
            schema = manifest.get("descriptor_schema")
            if not isinstance(schema, dict):
                raise ValueError("teacher router omitted descriptor schema")
            models = _teacher_models(config, teacher)
            if len(models) != 1:
                raise ValueError("FP visualization requires one teacher model")
            readout = teacher.read_structure_vectors(
                model=models[0],
                candidate_frames=candidates,
                background_frames=background,
                reference_path=reference_path,
                router_dir=router_dir,
                work_dir=work,
                chunk_size=chunk_size,
                reference_descriptor_schema=schema,
                report=report,
                type_map=tuple(config.teacher["type_map"]),
            )
            candidate_vectors = dict(readout.candidate_vectors)
            background_vectors = dict(readout.background_vectors)
            descriptor_schema = dict(readout.descriptor_schema)
            candidate_element_vectors = dict(
                readout.candidate_element_vectors or {}
            )
            background_element_vectors = dict(
                readout.background_element_vectors or {}
            )
        else:
            (
                candidate_vectors,
                background_vectors,
                descriptor_schema,
                candidate_element_vectors,
                background_element_vectors,
            ) = existing_readout
        candidate_rows = element_rows_from_element_vectors(
            candidates,
            candidate_element_vectors,
            descriptor_schema=descriptor_schema,
            type_map=tuple(config.teacher["type_map"]),
        )
        background_rows = element_rows_from_element_vectors(
            background,
            background_element_vectors,
            descriptor_schema=descriptor_schema,
            type_map=tuple(config.teacher["type_map"]),
        )
        request = build_visualization_request(
            kind="fp",
            layers=(
                VisualizationLayer(
                    "pool",
                    tuple(str(frame.info["structure_id"]) for frame in candidates),
                ),
                VisualizationLayer(
                    "train",
                    tuple(str(frame.info["structure_id"]) for frame in background),
                ),
                VisualizationLayer(
                    "select",
                    tuple(str(frame.info["structure_id"]) for frame in selected),
                ),
            ),
            populations=(candidate_rows, background_rows),
            structure_populations=tuple(
                structure_rows_from_vectors(
                    tuple(str(frame.info["structure_id"]) for frame in frames),
                    vectors, width=int(descriptor_schema["width"]),
                )
                for frames, vectors in ((candidates, candidate_vectors), (background, background_vectors))
            ),
            descriptor_schema=descriptor_schema,
            type_map=tuple(config.teacher["type_map"]),
            model_identity=_expected_member_ids("teacher", config.teacher)[0],
            dataset_identity=f"iter{config.iter_index}/teach/train/dataset",
        )
        publish_visualization_request(stage / ".visualize_cache", request)
    except Exception as exc:
        _report_warning(report, exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _fp_count_flow(
    system_ids: tuple[str, ...],
    *,
    candidate_frames_by_round: list[list[Any]],
    unique_frames: tuple[Any, ...],
    selected_frames: list[Any],
):
    from workflow_runtime.lifecycle.progress import CountFlowSnapshot, SystemCountRow

    candidates = [frame for frames in candidate_frames_by_round for frame in frames]
    candidate_counts = _frame_counts(candidates)
    unique_counts = _frame_counts(unique_frames)
    selected_counts = _frame_counts(selected_frames)
    rows = (
        SystemCountRow("ALL", (len(candidates), len(unique_frames), len(selected_frames))),
        *(
            SystemCountRow(
                system_id,
                (
                    candidate_counts.get(system_id, 0),
                    unique_counts.get(system_id, 0),
                    selected_counts.get(system_id, 0),
                ),
            )
            for system_id in system_ids
        ),
    )
    return CountFlowSnapshot(
        title="FP selection",
        columns=("candidates", "unique", "selected"),
        rows=rows,
    )


def _frame_counts(frames: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        system_id = str(frame.info.get("system_id", ""))
        counts[system_id] = counts.get(system_id, 0) + 1
    return counts


def _prior_iter_student_train_sources(prior_root: Path) -> tuple[Path, ...]:
    """Resolve the completed prior iter's direct final student predecessor."""
    root = Path(prior_root)
    round_indices = sorted(
        int(path.name[1:])
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("r") and path.name[1:].isdigit()
    )
    if not round_indices or round_indices != list(range(round_indices[-1] + 1)):
        raise ValueError("prior lifecycle iter must contain contiguous rN lifecycle directories starting at r0")
    final_round = round_indices[-1]
    sources = [
        root / f"r{final_round}" / "train" / "dataset" / "train.xyz",
        root / f"r{final_round}" / "screen" / "train.xyz",
    ]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"prior lifecycle iter is missing student training source: {missing[0]}")
    return tuple(sources)


def _typed_teacher_train(
    composition: BackendComposition,
    spec: dict[str, Any],
    data: DataService,
    dataset: TeacherDatasetMaterialization,
    stage: Path,
    intent: Any,
    expectation: Any,
    report: Any | None = None,
) -> None:
    from workflow_backends.teacher.training import (
        TeacherFoundationAsset,
        TeacherTrainRequest,
    )
    recipe_path = data.run_json_path.parent / spec["recipe"]
    foundation_asset = None
    foundation_path = _optional_path(data.run_json_path.parent, spec.get("foundation_model"))
    if foundation_path is not None:
        foundation_asset = TeacherFoundationAsset(
            kind="backend-native",
            path=foundation_path,
        )
    member_ids = tuple(intent.member_ids)
    request = TeacherTrainRequest(
        iter_index=data.iter_index,
        recipe_path=recipe_path,
        train_data_path=dataset.train_data_path,
        validation_data_path=dataset.validation_data_path,
        validation_mode=dataset.validation_mode,
        model_shape=spec["model"],
        member_ids=member_ids,
        type_map=tuple(spec["type_map"]),
        policy=spec["policy"],
        foundation_asset=foundation_asset,
        train_dir=stage,
        train_options=spec.get("train_options", {}),
        progress_reporter_factory=(
            None
            if report is None
            else lambda member_id: report.concurrent_operation(
                member_id,
                phase="training",
                group_id="teacher-training",
            )
        ),
    )
    composition.execute_teacher_training_expectation(
        expectation,
        intent=intent,
        physical_request=request,
        stage_root=stage,
    )


def _teacher_models(
    config: Any, teacher: LifecycleTeacherFacade
) -> tuple[Path, ...]:
    root = config.root / f"iter{config.iter_index}" / "teach" / "train"
    handles = teacher.resolve_published_models(
        stage_root=root,
        member_ids=_expected_member_ids("teacher", config.teacher),
    )
    return tuple(handle.path for handle in handles)


def _teacher_label(
    teacher_label: LifecycleTeacherFacade,
    models: tuple[Path, ...],
    frames: list[Any],
    work_dir: Path,
    *,
    report: Any | None = None,
    trusted_below: float | None = None,
) -> list[Any]:
    from workflow_backends.teacher.label_result import TeacherLabelRequest

    return teacher_label.label(
        TeacherLabelRequest(
            models=models,
            frames=tuple(frames),
            work_dir=work_dir,
            trusted_below=trusted_below,
            progress_reporter_factory=(
                None
                if report is None
                else lambda operation_id, phase: report.operation(
                    operation_id, phase=phase
                )
            ),
        )
    )


def _single_teacher_committee_student_thresholds(
    raw: Mapping[str, Any], round_index: int, route: RouteSpec
) -> dict[str, object]:
    from workflow_policies.identities import selector_identity

    thresholds: dict[str, object] = {
        "teacher_p": dict(raw["screen"]["teacher_p"]),
        "student_force_deviation": {
            "lo": float(raw["screen"]["lo"]),
            "hi": float(raw["screen"]["hi"]),
        },
        "quota": {
            "fp_pool": int(raw["screen"]["fp_pool"]),
            "pool": _round_value(raw["screen"]["pool"], round_index),
            "keep": _round_value(raw["screen"]["keep"], round_index),
        },
        "fill_underfilled_from_safe_region": bool(
            raw["screen"]["fill_underfilled_from_safe_region"]
        ),
        "selector": selector_identity(str(raw["screen"]["policy"])).as_dict(),
    }
    if route.screen_routing_policy_id is None:
        raise ValueError("single-teacher mixed route is missing its routing identity")
    return {
        "routing_policy": route.screen_routing_policy_id.as_dict(),
        **thresholds,
    }


def _single_single_thresholds(
    raw: Mapping[str, Any], round_index: int, route: RouteSpec
) -> dict[str, object]:
    from workflow_policies.identities import selector_identity

    if route.screen_routing_policy_id is None:
        raise ValueError("single/single route thresholds require routing identity")
    if raw["screen"].get("prescreen") in {
        "JOINT_DIRECT",
        "ELEMENT_JOINT_DIRECT",
    }:
        prescreen = str(raw["screen"]["prescreen"])
        return {
            "prescreen": selector_identity(prescreen).as_dict(),
            "teacher_p": dict(raw["screen"]["teacher_p"]),
            "quota": {
                "fp_pool": int(raw["screen"]["fp_pool"]),
                "pool": _round_value(raw["screen"]["pool"], round_index),
                "keep": _round_value(raw["screen"]["keep"], round_index),
            },
            "fill_underfilled_from_safe_region": False,
            "selector": selector_identity(str(raw["screen"]["policy"])).as_dict(),
        }
    return {
        "routing_policy": route.screen_routing_policy_id.as_dict(),
        "teacher_p": dict(raw["screen"]["teacher_p"]),
        "student_p": dict(raw["screen"]["student_p"]),
        "quota": {
            "fp_pool": int(raw["screen"]["fp_pool"]),
            "pool": _round_value(raw["screen"]["pool"], round_index),
            "keep": _round_value(raw["screen"]["keep"], round_index),
        },
        "fill_underfilled_from_safe_region": bool(
            raw["screen"]["fill_underfilled_from_safe_region"]
        ),
        "selector": selector_identity(str(raw["screen"]["policy"])).as_dict(),
    }


def _committee_teacher_single_student_thresholds(
    raw: Mapping[str, Any], round_index: int
) -> dict[str, object]:
    from workflow_policies.identities import selector_identity

    if raw["screen"].get("prescreen") in {
        "JOINT_DIRECT",
        "ELEMENT_JOINT_DIRECT",
    }:
        prescreen = str(raw["screen"]["prescreen"])
        return {
            "prescreen": selector_identity(prescreen).as_dict(),
            "teacher_force_deviation": {
                "lo": float(raw["teacher"]["lo"]),
                "hi": float(raw["teacher"]["hi"]),
            },
            "quota": {
                "pool": _round_value(raw["screen"]["pool"], round_index),
                "keep": _round_value(raw["screen"]["keep"], round_index),
            },
            "fill_underfilled_from_safe_region": False,
            "selector": selector_identity(str(raw["screen"]["policy"])).as_dict(),
        }
    return {
        "student_p": dict(raw["screen"]["student_p"]),
        "teacher_force_deviation": {
            "lo": float(raw["teacher"]["lo"]),
            "hi": float(raw["teacher"]["hi"]),
        },
        "quota": {
            "pool": _round_value(raw["screen"]["pool"], round_index),
            "keep": _round_value(raw["screen"]["keep"], round_index),
        },
        "fill_underfilled_from_safe_region": bool(
            raw["screen"]["fill_underfilled_from_safe_region"]
        ),
        "selector": selector_identity(str(raw["screen"]["policy"])).as_dict(),
    }


def _committee_committee_thresholds(
    raw: Mapping[str, Any], round_index: int
) -> dict[str, object]:
    from workflow_policies.identities import selector_identity

    return {
        "student_force_deviation": {
            "lo": float(raw["screen"]["lo"]),
            "hi": float(raw["screen"]["hi"]),
        },
        "teacher_force_deviation": {
            "lo": float(raw["teacher"]["lo"]),
            "hi": float(raw["teacher"]["hi"]),
        },
        "quota": {
            "pool": _round_value(raw["screen"]["pool"], round_index),
            "keep": _round_value(raw["screen"]["keep"], round_index),
        },
        "fill_underfilled_from_safe_region": bool(
            raw["screen"]["fill_underfilled_from_safe_region"]
        ),
        "selector": selector_identity(str(raw["screen"]["policy"])).as_dict(),
    }


def _student_expected_models(config: Any, round_index: int):
    from workflow_core.lifecycle.artifact_handles import ModelHandle

    return tuple(
        ModelHandle(
            member_id,
            f"iter{config.iter_index}/r{round_index}/train/{member_id}",
        )
        for member_id in _expected_member_ids("student", config.student)
    )


def _teacher_training_expectation(config: Any, data: DataService):
    from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
    from workflow_core.lifecycle.artifact_handles import ModelHandle
    from workflow_core.lifecycle.teacher_capabilities import (
        TEACHER_TRAINING_ARTIFACT_KIND,
        TeacherTrainingArtifactFacts,
        TeacherTrainingIntent,
    )

    spec = config.teacher
    bootstrap_sources = [
        config.root / path for path in config.raw["teacher_train_bootstrap"]
    ]
    fp_sources = (
        [config.root / path for path in config.raw["fp_data"]]
        if config.iter_index
        else []
    )
    dataset = data.teacher_dataset_handle(
        bootstrap_sources,
        dataset_id=f"iter{config.iter_index}/teach/train/dataset",
        valid_frac=float(spec["valid_frac"]),
        fp_sources=fp_sources,
    )
    member_ids = _expected_member_ids("teacher", spec)
    models = tuple(
        ModelHandle(
            member_id,
            f"iter{config.iter_index}/teach/train/{member_id}",
        )
        for member_id in member_ids
    )
    foundation = None
    if spec["policy"] == "foundation_finetune":
        foundation = ModelHandle(
            "foundation",
            f"teacher-foundation:{spec['foundation_model']}",
        )
    intent = TeacherTrainingIntent(
        iter_index=config.iter_index,
        dataset=dataset,
        model_shape=str(spec["model"]),
        member_ids=member_ids,
        type_map=tuple(spec["type_map"]),
        policy=str(spec["policy"]),
        foundation_model=foundation,
        train_options=spec.get("train_options", {}),
    )
    return intent, ExpectedArtifactSemantics(
        PublicBlockId("teach/train"),
        TEACHER_TRAINING_ARTIFACT_KIND,
        TeacherTrainingArtifactFacts(dataset, models),
    )


def _materialized_teacher_dataset_handle(
    dataset: TeacherDatasetMaterialization,
    *,
    dataset_id: str,
):
    from workflow_core.lifecycle.teacher_capabilities import TeacherDatasetHandle

    train = read_extxyz_structures(dataset.train_data_path, allow_empty=False)
    validation = (
        train
        if dataset.validation_mode == "self"
        else read_extxyz_structures(dataset.validation_data_path, allow_empty=False)
    )
    return TeacherDatasetHandle(
        dataset_id,
        tuple(str(frame.info.get("structure_id", "")) for frame in train),
        tuple(str(frame.info.get("structure_id", "")) for frame in validation),
        dataset.validation_mode,
    )


def _fp_label_operation_id(
    selected_path: str | Path,
    recipe_path: str | Path,
    *,
    calculator: str,
) -> str:
    """Identify one durable native operation from its exact immutable inputs."""

    digest = hashlib.sha256()
    digest.update(str(calculator).upper().encode("utf-8"))
    digest.update(b"\0")
    digest.update(Path(selected_path).read_bytes())
    digest.update(b"\0")
    digest.update(Path(recipe_path).read_bytes())
    return digest.hexdigest()


def _fp_label_expectation(config: Any, selected_path: str | Path):
    from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics
    from workflow_core.lifecycle.artifact_handles import StructureBatchHandle
    from workflow_core.lifecycle.fp_capabilities import (
        FP_LABEL_ARTIFACT_KIND,
        FirstPrinciplesLabelArtifactFacts,
        FirstPrinciplesSelection,
        fp_task_id,
    )

    selected = read_extxyz_structures(selected_path, allow_empty=True)
    structure_ids = tuple(
        str(frame.info.get("structure_id", "")) for frame in selected
    )
    system_ids = tuple(str(frame.info.get("system_id", "")) for frame in selected)
    batch = StructureBatchHandle(
        f"iter{config.iter_index}/fp/selected",
        structure_ids,
    )
    selection = FirstPrinciplesSelection(batch, system_ids)
    calculator = str(config.raw["fp"]["label"]["calculator"])
    expectation = ExpectedArtifactSemantics(
        PublicBlockId("fp/label"),
        FP_LABEL_ARTIFACT_KIND,
        FirstPrinciplesLabelArtifactFacts(
            batch,
            system_ids,
            tuple(fp_task_id(index) for index in range(len(structure_ids))),
            calculator,
        ),
    )
    return selection, expectation


def _student_training_expectation(
    config: Any,
    round_index: int,
    ordered_structure_ids: tuple[str, ...],
):
    from workflow_core.lifecycle.artifact_handles import DatasetHandle, ModelHandle
    from workflow_core.lifecycle.student_capabilities import (
        StudentTrainingExpectation,
        StudentTrainingIntent,
    )

    spec = config.student
    foundation = None
    if spec["policy"] == "foundation_finetune":
        foundation = ModelHandle(
            "foundation",
            f"student-foundation:{spec['foundation_model']}",
        )
    models = _student_expected_models(config, round_index)
    intent = StudentTrainingIntent(
        iter_index=config.iter_index,
        round_index=round_index,
        dataset=DatasetHandle(
            f"iter{config.iter_index}/r{round_index}/train/dataset",
            ordered_structure_ids,
        ),
        model_shape=str(spec["model"]),
        member_ids=tuple(model.member_id for model in models),
        type_map=tuple(spec["type_map"]),
        policy=str(spec["policy"]),
        foundation_model=foundation,
        train_options=spec.get("train_options", {}),
    )
    return StudentTrainingExpectation(intent, models)


def _student_training_source_structure_ids(
    config: Any, round_index: int
) -> tuple[str, ...]:
    sources = [
        config.root
        / f"iter{config.iter_index}"
        / "teach"
        / "screen"
        / "train.xyz"
    ]
    sources.extend(
        config.root
        / f"iter{config.iter_index}"
        / f"r{prior}"
        / "screen"
        / "train.xyz"
        for prior in range(round_index)
    )
    ordered: dict[str, None] = {}
    for source in sources:
        for frame in read_extxyz_structures(source, allow_empty=True):
            identity = str(frame.info.get("structure_id", ""))
            if not identity:
                raise ValueError("student training source lacks canonical structure_id")
            ordered.setdefault(identity, None)
    if not ordered:
        raise ValueError("student training expectation requires source structures")
    return tuple(ordered)


class _StudentTrainingValidatorMap:
    """Bind one canonical train validator per round and reuse it for publication."""

    def __init__(self, config: Any, student: Any) -> None:
        self._config = config
        self._student = student
        self._bindings: dict[int, tuple[object, ArtifactCheck]] = {}

    def bind(
        self,
        round_index: int,
        *,
        ordered_structure_ids: tuple[str, ...] | None = None,
    ) -> tuple[object, ArtifactCheck]:
        cached = self._bindings.get(round_index)
        if cached is not None:
            if ordered_structure_ids is not None:
                expectation = _student_training_expectation(
                    self._config,
                    round_index,
                    ordered_structure_ids,
                )
                if expectation != cached[0]:
                    raise ValueError(
                        "materialized student dataset diverges from its bound expectation"
                    )
            return cached
        identities = (
            _student_training_source_structure_ids(self._config, round_index)
            if ordered_structure_ids is None
            else tuple(ordered_structure_ids)
        )
        expectation = _student_training_expectation(
            self._config,
            round_index,
            identities,
        )
        expected = expectation.for_block(PublicBlockId.round(round_index, "train"))
        validator = self._student.bind_training_artifact_validator(expectation)
        if getattr(validator, "expectation", None) != expected or not callable(validator):
            raise ValueError("student artifact validator changed its expectation")
        binding = (expectation, validator)
        self._bindings[round_index] = binding
        return binding

    def validate(self, round_index: int, path: Path) -> bool:
        try:
            _expectation, validator = self.bind(round_index)
            return validator(path)
        except (OSError, TypeError, ValueError):
            return False


def _sample(
    config: Any,
    student: Any,
    sampling: Any,
    round_index: int,
    steps: int | None,
    stage: Path,
    *,
    report: Any | None = None,
) -> None:
    sample_started = time.monotonic()
    model_count = len(_student_expected_models(config, round_index))
    plan = config.sampling_plan
    if plan is None:
        raise ValueError("lifecycle sampling requires a resolved per-system plan")
    def publish_progress(snapshot):
        if report is not None:
            report.phase("sampling")
            report.snapshot(snapshot)

    coordinator, intent, evidence = _bound_sampling_handle_coordinator(
        config,
        student,
        sampling,
        round_index,
        stage,
        progress_callback=publish_progress if report is not None else None,
        progress_reporter_factory=(
            None
            if report is None
            else lambda task_id: report.concurrent_operation(
                task_id,
                phase="sampling",
                group_id="sampling-task-group",
            )
        ),
    )
    from workflow_core.lifecycle.sampling import SamplingExecutionIntent

    outcome = coordinator.execute(SamplingExecutionIntent(intent))
    if report is not None:
        report.summary(
            _sampling_block_summary(
                config,
                plan=plan,
                round_index=round_index,
                system_count=len(evidence.systems),
                model_count=model_count,
                elapsed_seconds=time.monotonic() - sample_started,
                failed_task_count=getattr(outcome, 'failed_task_count', 0),
            )
        )
    return None


def _sampling_block_summary(
    config: Any,
    *,
    plan: object,
    round_index: int,
    system_count: int,
    model_count: int,
    elapsed_seconds: float,
    failed_task_count: int = 0,
):
    from workflow_runtime.lifecycle.progress import SamplingBlockSummary

    round_tasks = (
        plan.tasks_for_round(round_index)
        if hasattr(plan, "tasks_for_round")
        else plan.tasks
    )
    ordered_kinds = tuple(
        dict.fromkeys(str(getattr(task, "kind", "md")) for task in round_tasks)
    )
    ase_kinds = {"neb", "relax"}
    has_ase = any(kind in ase_kinds for kind in ordered_kinds)
    has_native = any(kind not in ase_kinds for kind in ordered_kinds)
    configured_backend = str(config.sampler["backend"])
    effective_backend = (
        f"{configured_backend}+ASE"
        if has_ase and has_native
        else "ASE"
        if has_ase
        else configured_backend
    )

    def task_steps(task: object) -> int:
        if getattr(task, 'widom', None) is not None:
            return 0
        if getattr(task, 'gcmc', None) is not None:
            return 0 if task.gcmc.md is None else int(task.cycles)*task.gcmc.md.steps
        if getattr(task, "steps", None) is not None:
            return int(task.steps)
        if getattr(task, "neb", None) is not None:
            return int(task.neb.max_steps)
        if getattr(task, "relax", None) is not None:
            return int(task.relax.max_steps)
        legacy_sample = getattr(config, "raw", {}).get("sample", {})
        legacy_steps = legacy_sample.get("steps", ())
        if round_index < len(legacy_steps):
            return int(legacy_steps[round_index])
        raise ValueError("sampling summary task has no bounded step count")

    temperatures = {
        float(task.temperature_k)
        for task in round_tasks
        if getattr(task, "temperature_k", None) is not None
    }
    return SamplingBlockSummary(
        backend=effective_backend,
        failed_tasks=failed_task_count,
        model_shape="single" if model_count == 1 else "committee",
        tasks=len(round_tasks),
        systems=int(system_count),
        temperatures=len(temperatures),
        total_steps=sum(task_steps(task) for task in round_tasks),
        mc_attempts=sum(int(task.cycles)*task.gcmc.moves for task in round_tasks
                        if getattr(task, 'gcmc', None) is not None)
                    + sum(task.md_mc.planned_attempts(task.steps) for task in round_tasks
                          if getattr(task, 'md_mc', None) is not None),
        elapsed_seconds=float(elapsed_seconds),
        kind_counts=tuple(
            (
                kind,
                sum(
                    str(getattr(task, "kind", "md")) == kind
                    for task in round_tasks
                ),
            )
            for kind in ordered_kinds
        ),
    )


def _sample_observation_source(
    config: Any, student: Any, sampling: Any, round_index: int
):
    sample_dir = config.root / f"iter{config.iter_index}" / f"r{round_index}" / "sample"
    coordinator, intent, _evidence = _bound_sampling_handle_coordinator(
        config,
        student,
        sampling,
        round_index,
        sample_dir,
    )
    return coordinator.open_observations(intent).source


def _resolved_sampling_inputs(config: Any, round_index: int):
    from workflow_backends.sampling import (
        SamplingSeedSelection,
        SamplingSeedSource,
    )

    plan = config.sampling_plan
    if plan is None:
        raise ValueError("lifecycle sampling requires a resolved per-system plan")
    spec = config.raw["sample"]
    explicit = "seeds" in spec
    declarations = spec["seeds"] if explicit else spec["systems"]
    seed_selections: list[SamplingSeedSelection] = []
    if explicit:
        from workflow_core.lifecycle.seed_frame_sampling import (
            ALGORITHM,
            select_frame_indices,
        )

        frame_indices_by_role: dict[str, tuple[int, ...]] = {}
        expanded_sources: list[SamplingSeedSource] = []
        active_role_ids = {
            role_id
            for role_id, _group_id in plan.selection_plan_for_round(
                round_index
            ).role_to_group
        }
        active_neb_role_ids = {
            task.sampling_role_id
            for task in plan.tasks_for_round(round_index)
            if task.kind == "neb"
        }
        for role_id, declaration in declarations.items():
            if role_id not in active_role_ids:
                continue
            source_path = config.root / declaration["source"]
            frames = tuple(read_extxyz_structures(source_path, allow_empty=False))
            if "max_frames" not in declaration:
                expanded_sources.append(
                    SamplingSeedSource(
                        role_id,
                        frames if role_id in active_neb_role_ids else frames[:1],
                    )
                )
                continue
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            indices, random_seed_sha256 = select_frame_indices(
                source_sha256=source_sha256,
                frame_count=len(frames),
                max_frames=int(declaration["max_frames"]),
                iter_index=config.iter_index,
                round_index=round_index,
                sampling_role_id=role_id,
            )
            frame_indices_by_role[role_id] = indices
            seed_selections.append(
                SamplingSeedSelection(
                    source_role_id=role_id,
                    source_sha256=source_sha256,
                    frame_count=len(frames),
                    max_frames=int(declaration["max_frames"]),
                    selected_indices=indices,
                    random_seed_sha256=random_seed_sha256,
                    algorithm=ALGORITHM,
                )
            )
            expanded_sources.extend(
                SamplingSeedSource(
                    f"{role_id}__frame_{index:06d}",
                    (frames[index],),
                )
                for index in indices
            )
        if frame_indices_by_role:
            plan = plan.expand_source_frames(frame_indices_by_role)
        systems = tuple(expanded_sources)
    else:
        systems = tuple(
            SamplingSeedSource(
                system_id,
                tuple(
                    read_extxyz_structures(
                        config.root / system["seed"], allow_empty=False
                    )
                ),
            )
            for system_id, system in declarations.items()
        )
    return plan, systems, tuple(seed_selections)


def _sampling_plan_for_recovery(config: Any, round_index: int):
    plan = config.sampling_plan
    if plan is None:
        raise ValueError("lifecycle recovery requires a resolved sampling plan")
    if not bool(getattr(plan, "explicit", False)):
        return plan
    manifest = (
        config.root
        / f"iter{config.iter_index}"
        / f"r{round_index}"
        / "sample"
        / "seed_selection.json"
    )
    if manifest.is_file() and not manifest.is_symlink():
        try:
            from workflow_backends.sampling import (
                SamplingSeedSelection,
                seed_selection_manifest_payload,
            )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            selections = tuple(
                SamplingSeedSelection(
                    source_role_id=item["sampling_role_id"],
                    source_sha256=item["source_sha256"],
                    frame_count=item["frame_count"],
                    max_frames=item["max_frames"],
                    selected_indices=tuple(item["selected_indices"]),
                    random_seed_sha256=item["random_seed_sha256"],
                    algorithm=item["algorithm"],
                )
                for item in payload["sources"]
            )
            if payload != seed_selection_manifest_payload(
                iter_index=config.iter_index,
                round_index=round_index,
                selections=selections,
            ):
                raise ValueError("sample seed selection manifest changed")
            declarations = config.raw["sample"]["seeds"]
            expected_roles = tuple(
                role_id
                for role_id, declaration in declarations.items()
                if "max_frames" in declaration
                and role_id
                in {
                    active_role_id
                    for active_role_id, _group_id in plan.selection_plan_for_round(
                        round_index
                    ).role_to_group
                }
            )
            if tuple(item.source_role_id for item in selections) != expected_roles:
                raise ValueError("sample seed selection roles changed")
            if any(
                item.max_frames
                != int(declarations[item.source_role_id]["max_frames"])
                for item in selections
            ):
                raise ValueError("sample seed selection limits changed")
            return plan.expand_source_frames(
                {
                    item.source_role_id: item.selected_indices
                    for item in selections
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return plan
    if manifest.exists() or manifest.is_symlink():
        return plan
    resolved, _systems, _selections = _resolved_sampling_inputs(
        config, round_index
    )
    return resolved


def _terminal_selection_plan(config: Any, round_indices: tuple[int, ...]):
    if not bool(getattr(config.sampling_plan, "explicit", False)):
        return None
    from workflow_core.lifecycle.sampling_plan import SeedRole, SelectionPlan

    round_mappings: list[tuple[tuple[str, str], ...]] = []
    for round_index in round_indices:
        plan = _sampling_plan_for_recovery(config, round_index)
        round_mappings.append(
            plan.selection_plan_for_round(round_index).role_to_group
        )
    roles: dict[str, SeedRole] = {}
    for declared_seed in config.sampling_plan.seeds:
        base_role_id = declared_seed.sampling_role_id
        for mapping in round_mappings:
            for role_id, group_id in mapping:
                if role_id == base_role_id or role_id.startswith(
                    f"{base_role_id}__frame_"
                ):
                    roles.setdefault(role_id, SeedRole(role_id, group_id))
    return SelectionPlan.from_seeds(tuple(roles.values()))


def _sampling_evidence_request(
    config: Any,
    round_index: int,
    sample_dir: str | Path,
    *,
    student: Any | None = None,
    models: object | None = None,
):
    from workflow_backends.sampling import SamplingEvidenceRequest
    from workflow_backends.sampling_models import OrderedSamplingModels

    plan, systems, seed_selections = _resolved_sampling_inputs(
        config, round_index
    )
    spec = config.raw["sample"]
    explicit = "seeds" in spec
    if models is None:
        if student is None:
            raise ValueError(
                "sampling evidence requires the selected student role when models are not supplied"
            )
        handles = student.resolve_published_models(
            stage_root=(
                config.root
                / f"iter{config.iter_index}"
                / f"r{round_index}"
                / "train"
            ),
            member_ids=_expected_member_ids("student", config.student),
        )
        model_inputs = OrderedSamplingModels(
            tuple(handle.path for handle in handles),
            member_ids=tuple(handle.member_id for handle in handles),
        )
    else:
        model_inputs = models
    return SamplingEvidenceRequest(
        run_json_path=config.path,
        iter_index=config.iter_index,
        round_index=round_index,
        sample_dir=Path(sample_dir),
        plan=plan,
        systems=systems,
        models=model_inputs,
        steps=None if explicit else int(spec["steps"][round_index]),
        seed_selections=seed_selections,
    )


def _sampling_model_preparation_expectation(
    config: Any, round_index: int, models: tuple[object, ...]
):
    from workflow_core.lifecycle.sampling_models import (
        SamplingModelPreparationExpectation,
    )

    spec = config.student
    attributes: dict[str, str | int | bool]
    if spec["backend"] == "DeepMD":
        recipe = json.loads((config.root / spec["recipe"]).read_text(encoding="utf-8"))
        model = recipe.get("model", {}) if isinstance(recipe, dict) else {}
        descriptor = model.get("descriptor", {}) if isinstance(model, dict) else {}
        recipe_kind = (
            "dpa4c"
            if isinstance(descriptor, dict) and descriptor.get("type") == "dpa4c"
            else "standard"
        )
        attributes = {
            "deepmd_recipe_kind": recipe_kind,
            "cli": str(spec["cli"]),
            "schema_revision": 1,
        }
        backend = f"deepmd-{spec['cli']}"
    else:
        attributes = {"format": "nep", "schema_revision": 1}
        backend = "nep"
    return SamplingModelPreparationExpectation(
        iter_index=config.iter_index,
        round_index=round_index,
        models=models,
        backend=backend,
        compress=bool(spec.get("compress", False)),
        attributes=attributes,
    )


def _sampling_evidence_intent(
    config: Any,
    round_index: int,
    evidence: object,
    models: tuple[object, ...],
):
    from workflow_core.lifecycle.artifact_handles import StructureBatchHandle
    from workflow_core.lifecycle.sampling import (
        SamplingEvidenceIntent,
        SamplingSystemIntent,
        SamplingTaskIntent,
    )
    from workflow_core.structures.canonical_identity import structure_id_for_atoms

    systems = tuple(
        SamplingSystemIntent(
            source.system_id,
            StructureBatchHandle(
                f"iter{config.iter_index}/r{round_index}/sample/{source.system_id}/seeds",
                tuple(
                    str(frame.info.get("structure_id") or structure_id_for_atoms(frame))
                    for frame in source.frames
                ),
            ),
        )
        for source in getattr(evidence, "systems")
    )
    plan = getattr(evidence, "plan")
    explicit = hasattr(plan, "tasks_for_round")
    resolved_tasks = plan.tasks_for_round(round_index) if explicit else plan.tasks
    tasks = tuple(
        SamplingTaskIntent(
            f"task_{index:06d}",
            condition.system_id,
            condition.ensemble,
            condition.temperature_k,
            condition.pressure_bar,
            **(
                {
                    "condition_group_id": condition.condition_group_id,
                    "sampling_role_id": condition.sampling_role_id,
                    "selection_group_id": condition.selection_group_id,
                    "kind": condition.kind,
                    "steps": condition.steps,
                    "program": condition.program,
                    "source_frame_index": condition.source_frame_index,
                    "neb": condition.neb,
                    "relax": condition.relax,
                    "gcmc": condition.gcmc,
                    "widom": condition.widom,
                    "md_mc": condition.md_mc,
                    "cycles": condition.cycles,
                }
                if explicit
                else {}
            ),
        )
        for index, condition in enumerate(resolved_tasks)
    )
    task_kinds = {condition.kind for condition in resolved_tasks} if explicit else set()
    inventory_items = ["seeds", "task-contexts"]
    if any(kind not in {"neb", "relax", "gcmc_md", "widom"} for kind in task_kinds):
        inventory_items.extend(
            ("trajectories", "model-deviation")
            if config.sampler["backend"] == "LAMMPS"
            else ("active-observations",)
        )
    if "neb" in task_kinds:
        inventory_items.append("neb-evidence")
    if "relax" in task_kinds:
        inventory_items.append("relax-evidence")
    if "widom" in task_kinds:
        inventory_items.append("widom-evidence")
    if "gcmc_md" in task_kinds:
        inventory_items.append("gcmc-evidence")
    inventory = tuple(inventory_items)
    return SamplingEvidenceIntent(
        config.iter_index,
        round_index,
        models,
        systems,
        tasks,
        inventory,
    )


def _bound_sampling_handle_coordinator(
    config: Any,
    student: Any,
    sampling: Any,
    round_index: int,
    sample_dir: Path,
    *,
    progress_callback: object | None = None,
    progress_reporter_factory: object | None = None,
):
    from workflow_backends.sampling import (
        BoundSamplingHandleSession,
        SamplingExecutionRequest,
    )
    from workflow_backends.sampling_models import OrderedSamplingModels
    from workflow_core.lifecycle.sampling_coordinator import LifecycleSamplingCoordinator

    canonical_models = _student_expected_models(config, round_index)
    resolved_models = student.resolve_model_expectations(
        canonical_models,
        stage_root=(
            config.root
            / f"iter{config.iter_index}"
            / f"r{round_index}"
            / "train"
        ),
    )
    physical_models = OrderedSamplingModels(
        tuple(item.path for item in resolved_models),
        member_ids=tuple(item.handle.member_id for item in resolved_models),
    )
    evidence = _sampling_evidence_request(
        config,
        round_index,
        sample_dir,
        models=physical_models,
    )
    intent = _sampling_evidence_intent(
        config, round_index, evidence, canonical_models
    )
    preparation = _sampling_model_preparation_expectation(
        config, round_index, canonical_models
    )
    expected = intent.for_block(PublicBlockId.round(round_index, "sample"))
    physical_execution = SamplingExecutionRequest(
        evidence=evidence,
        progress_callback=progress_callback,
        progress_reporter_factory=progress_reporter_factory,
    )
    physical_sampling = sampling
    session = BoundSamplingHandleSession(
        expected,
        preparation,
        evidence,
        physical_execution,
        physical_sampling.model_provider,
        physical_sampling.native_ops,
        physical_sampling.evidence_observer,
        resolved_models,
    )
    return (
        LifecycleSamplingCoordinator(session, session, session),
        intent,
        evidence,
    )


def _artifact_checks(
    config: Any,
    *,
    blocks: tuple[str, ...] | None = None,
    composition: Any | None = None,
    data: DataService | None = None,
    student: Any | None = None,
    sampling_backend: Any | None = None,
    student_training_validators: _StudentTrainingValidatorMap | None = None,
) -> dict[str, ArtifactCheck]:
    from workflow_artifacts.lifecycle.screen_validation import (
        ScreenArtifactValidationRequest,
        validate_screen_artifacts,
    )
    from workflow_core.lifecycle.teacher_capabilities import (
        TeacherScanRecoveryFacts,
        TeacherScreenRecoveryFacts,
        TeacherTrainingRecoveryFacts,
    )
    from workflow_core.lifecycle.student_capabilities import (
        StudentTrainingRecoveryFacts,
    )
    from workflow_core.lifecycle.fp_capabilities import FinalFPRecoveryFacts
    from workflow_core.lifecycle.sampling import (
        SampleRecoveryFacts,
        SamplingTaskIntent,
    )

    del data, student, sampling_backend, student_training_validators
    selected = None if blocks is None else frozenset(blocks)
    wants = lambda block: selected is None or block in selected
    route = _resolved_route(config)
    binder = BackendComposition if composition is None else composition
    screen_validator = getattr(binder, "validate_screen_artifacts", validate_screen_artifacts)
    teacher_spec = config.teacher
    teacher_training_facts = TeacherTrainingRecoveryFacts(
        backend=str(teacher_spec["backend"]),
        cli=(
            str(teacher_spec["cli"])
            if teacher_spec["backend"] == "DeepMD"
            else None
        ),
        model_shape=str(teacher_spec["model"]),
        member_ids=_expected_member_ids("teacher", teacher_spec),
        type_map=tuple(teacher_spec["type_map"]),
        validation_mode=(
            "self" if float(teacher_spec["valid_frac"]) == 0.0 else "holdout"
        ),
    )
    teacher_screen_facts = TeacherScreenRecoveryFacts(
        iter_index=config.iter_index,
        type_map=tuple(teacher_spec["type_map"]),
        committee=teacher_spec["model"] == "committee",
        teacher_lo=(
            float(teacher_spec["lo"])
            if teacher_spec["model"] == "committee"
            else None
        ),
    )
    checks: dict[str, ArtifactCheck] = {}
    if wants("teach/train"):
        checks["teach/train"] = binder.bind_teacher_training_recovery_validator(
            teacher_training_facts
        )
    if wants("teach/screen"):
        checks["teach/screen"] = binder.bind_teacher_screen_recovery_validator(
            teacher_screen_facts
        )
    if wants("fp"):
        fp_candidate_indices = (
            range(config.round_count)
            if config.raw["fp"].get("all_rn", False)
            else (config.round_count - 1,)
        )
        fp_candidate_paths = tuple(
            config.root
            / f"iter{config.iter_index}"
            / f"r{index}"
            / "screen"
            / "fp_candidate.xyz"
            for index in fp_candidate_indices
        )
        explicit_sampling = bool(getattr(config.sampling_plan, "explicit", False))
        terminal_selection_plan = _terminal_selection_plan(
            config, tuple(fp_candidate_indices)
        )
        fp_recovery_facts = FinalFPRecoveryFacts(
            keep=int(config.raw["fp"]["keep"]),
            policy=str(config.raw["fp"]["policy"]),
            all_rn=bool(config.raw["fp"].get("all_rn", False)),
            round_count=config.round_count,
            iter_index=config.iter_index,
            system_ids=(
                terminal_selection_plan.group_ids
                if explicit_sampling
                else _sampling_role_ids(config)
            ),
            requires_frame_mean_evidence=(
                route.teacher_requires_reference_tail
                and config.raw["fp"]["policy"] in {"DIRECT", "FPS"}
            ),
            selection_plan=terminal_selection_plan,
            require_selection_group_audit=explicit_sampling,
            effective_policy=(
                "random"
                if config.raw["fp"]["policy"] == "ELEMENT_DIRECT"
                and config.teacher["model"] == "committee"
                else str(config.raw["fp"]["policy"])
            ),
            fallback_reason=(
                "committee_descriptor_unavailable"
                if config.raw["fp"]["policy"] == "ELEMENT_DIRECT"
                and config.teacher["model"] == "committee"
                else None
            ),
            joint_model_identity=(
                _expected_member_ids("teacher", config.teacher)[0]
                if config.raw["fp"]["policy"] in {
                    "JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
                }
                and not (
                    config.raw["fp"]["policy"] == "ELEMENT_DIRECT"
                    and config.teacher["model"] == "committee"
                )
                else None
            ),
            joint_dataset_identity=(
                f"iter{config.iter_index}/teach/train/dataset"
                if config.raw["fp"]["policy"] in {
                    "JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
                }
                and not (
                    config.raw["fp"]["policy"] == "ELEMENT_DIRECT"
                    and config.teacher["model"] == "committee"
                )
                else None
            ),
            element_type_map=(
                tuple(config.teacher["type_map"])
                if config.raw["fp"]["policy"] in {
                    "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
                }
                and not (
                    config.raw["fp"]["policy"] == "ELEMENT_DIRECT"
                    and config.teacher["model"] == "committee"
                )
                else None
            ),
        )
        fp_recovery_kwargs = {
            "candidate_paths": fp_candidate_paths,
            "teacher_router": (
                config.root
                / f"iter{config.iter_index}"
                / "teach"
                / "scan"
                / "gmm_router"
                if route.teacher_requires_reference_tail
                else None
            ),
        }
        if config.raw["fp"]["policy"] in {
            "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT",
        }:
            fp_recovery_kwargs["teacher_training_path"] = (
                config.root
                / f"iter{config.iter_index}"
                / "teach"
                / "train"
                / "dataset"
                / "train.xyz"
            )
        checks["fp"] = binder.bind_final_fp_recovery_validator(
            fp_recovery_facts,
            **fp_recovery_kwargs,
        )
    if wants("fp/label") and "label" in config.raw["fp"]:
        selected_path = (
            config.root
            / f"iter{config.iter_index}"
            / "fp"
            / "selected.xyz"
        )

        def fp_label_complete(path: Path) -> bool:
            if not selected_path.is_file():
                return False
            _selection, expectation = _fp_label_expectation(config, selected_path)
            return binder.bind_fp_label_artifact_validator(
                expectation,
                selected_path=selected_path,
                calculator=config.raw["fp"]["label"]["calculator"],
            )(path)

        checks["fp/label"] = fp_label_complete
    if wants("teach/scan") and route.has_teach_scan:
        teacher_scan_facts = TeacherScanRecoveryFacts(
            type_map=tuple(teacher_spec["type_map"])
        )
        checks["teach/scan"] = binder.bind_teacher_scan_recovery_validator(
            teacher_scan_facts
        )
    round_blocks = tuple(
        block for block in (selected or ()) if block.startswith("r")
    )
    if selected is not None and not round_blocks:
        return checks
    sampling_plan = config.sampling_plan
    if sampling_plan is None:
        raise ValueError("lifecycle recovery requires a resolved sampling plan")
    student_spec = config.student
    student_training_facts = StudentTrainingRecoveryFacts(
        backend=str(student_spec["backend"]),
        cli=(
            str(student_spec["cli"])
            if student_spec["backend"] == "DeepMD"
            else None
        ),
        model_shape=str(student_spec["model"]),
        member_ids=_expected_member_ids("student", student_spec),
        type_map=tuple(student_spec["type_map"]),
    )
    student_training_validator = (
        binder.bind_student_training_recovery_validator(
            student_training_facts
        )
    )
    explicit_sampling = hasattr(sampling_plan, "tasks_for_round")
    sample_declarations = config.raw["sample"].get(
        "seeds", config.raw["sample"].get("systems", {})
    )
    plumed_enabled = all(
        "plumed" in system
        for system in sample_declarations.values()
    )
    for round_index in range(config.round_count):
        train_block = f"r{round_index}/train"
        sample_block = f"r{round_index}/sample"
        screen_block = f"r{round_index}/screen"
        if wants(train_block):
            checks[train_block] = student_training_validator
        if not wants(sample_block) and not wants(screen_block):
            continue
        resolved_sampling_plan = _sampling_plan_for_recovery(
            config, round_index
        )
        round_selection_plan = (
            resolved_sampling_plan.selection_plan_for_round(round_index)
            if hasattr(resolved_sampling_plan, "selection_plan_for_round")
            else getattr(resolved_sampling_plan, "selection_plan", None)
        )
        round_conditions = (
            resolved_sampling_plan.tasks_for_round(round_index)
            if explicit_sampling
            else resolved_sampling_plan.tasks
        )
        sample_tasks = tuple(
            SamplingTaskIntent(
                f"task_{index:06d}",
                (
                    condition.sampling_role_id
                    if explicit_sampling
                    else condition.system_id
                ),
                condition.ensemble,
                condition.temperature_k,
                condition.pressure_bar,
                **(
                    {
                        "condition_group_id": condition.condition_group_id,
                        "sampling_role_id": condition.sampling_role_id,
                        "selection_group_id": condition.selection_group_id,
                        "kind": condition.kind,
                        "steps": condition.steps,
                        "program": condition.program,
                        "source_frame_index": condition.source_frame_index,
                        "neb": condition.neb,
                        "relax": condition.relax,
                        "gcmc": condition.gcmc,
                        "widom": condition.widom,
                        "md_mc": condition.md_mc,
                        "cycles": condition.cycles,
                    }
                    if explicit_sampling
                    else {}
                ),
            )
            for index, condition in enumerate(round_conditions)
        )
        sample_facts = SampleRecoveryFacts(
            backend=str(config.sampler["backend"]),
            cli=(
                str(student_spec["cli"])
                if config.sampler["backend"] == "LAMMPS"
                else None
            ),
            member_ids=_expected_member_ids("student", student_spec),
            type_map=tuple(student_spec["type_map"]),
            system_ids=(
                tuple(
                    role_id
                    for role_id, _group_id in round_selection_plan.role_to_group
                )
                if explicit_sampling
                else _sampling_role_ids(config)
            ),
            tasks=sample_tasks,
            steps=(
                None
                if explicit_sampling
                else int(config.raw["sample"]["steps"][round_index])
            ),
            plumed_enabled=plumed_enabled,
            compress=bool(student_spec.get("compress", False)),
            iter_index=config.iter_index,
            round_index=round_index,
        )
        if wants(sample_block):
            checks[sample_block] = binder.bind_sample_recovery_validator(sample_facts)
        expected_thresholds = None
        if route.screen_artifact_contract is ScreenArtifactContract.GMM_REFERENCE_TAIL:
            expected_thresholds = _single_single_thresholds(
                config.raw, round_index, route
            )
        elif (
            route.screen_artifact_contract
            is ScreenArtifactContract.SINGLE_TEACHER_COMMITTEE_STUDENT
        ):
            expected_thresholds = _single_teacher_committee_student_thresholds(
                config.raw, round_index, route
            )
        elif (
            route.screen_artifact_contract
            is ScreenArtifactContract.COMMITTEE_TEACHER_SINGLE_STUDENT
        ):
            expected_thresholds = _committee_teacher_single_student_thresholds(
                config.raw, round_index
            )
        elif (
            route.screen_artifact_contract
            is ScreenArtifactContract.COMMITTEE_FORCE_DEVIATION
        ):
            expected_thresholds = _committee_committee_thresholds(
                config.raw, round_index
            )
        if not wants(screen_block):
            continue
        checks[screen_block] = (
            lambda path,
            expected_thresholds=expected_thresholds,
            round_selection_plan=round_selection_plan,
            round_index=round_index: screen_validator(
                ScreenArtifactValidationRequest(
                    contract=route.screen_artifact_contract,
                    screen_dir=path,
                    screen_policy=config.raw["screen"]["policy"],
                    fp_policy=config.raw["fp"]["policy"],
                    pair_guard=config.raw["screen"].get("pair_guard"),
                    expected_system_ids=(
                        round_selection_plan.group_ids
                        if round_selection_plan is not None
                        else _selection_group_ids(config)
                    ),
                    expected_type_map=tuple(config.student["type_map"]),
                    expected_thresholds=expected_thresholds,
                    selection_plan=round_selection_plan,
                    require_selection_group_audit=bool(
                        getattr(config.sampling_plan, "explicit", False)
                    ),
                    prescreen=config.raw["screen"].get("prescreen", "GMM"),
                    joint_model_identity=(
                        f"iter{config.iter_index}/r{round_index}/train/student"
                        if config.raw["screen"].get("prescreen", "GMM")
                        in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}
                        or config.raw["screen"]["policy"] == "ELEMENT_DIRECT"
                        else None
                    ),
                    joint_dataset_identity=(
                        f"iter{config.iter_index}/r{round_index}/train/dataset"
                        if config.raw["screen"].get("prescreen", "GMM")
                        in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}
                        or config.raw["screen"]["policy"] == "ELEMENT_DIRECT"
                        else None
                    ),
                    expected_entropy=(
                        f"{config.path}/iter{config.iter_index}/r{round_index}"
                    ),
                    teacher_router=(
                        config.root
                        / f"iter{config.iter_index}"
                        / "teach"
                        / "scan"
                        / "gmm_router"
                        if route.teacher_requires_reference_tail
                        else None
                    ),
                )
            )
        )
    return checks


def _resolved_route(config: Any) -> RouteSpec:
    route = getattr(config, "route", None)
    if not isinstance(route, RouteSpec):
        raise ValueError("lifecycle core execution requires a validated resolved route")
    return route


def _sampling_role_ids(config: Any) -> tuple[str, ...]:
    sample = config.raw["sample"]
    declarations = sample.get("systems", sample.get("seeds"))
    if not isinstance(declarations, dict) or not declarations:
        raise ValueError("lifecycle requires resolved sampling roles")
    return tuple(str(role_id) for role_id in declarations)


def _selection_group_ids(config: Any) -> tuple[str, ...]:
    plan = getattr(config, "selection_plan", None)
    if plan is None or not getattr(plan, "group_ids", ()):
        raise ValueError("lifecycle requires resolved selection groups")
    return tuple(str(group_id) for group_id in plan.group_ids)


def _valid_sample_artifacts(
    config: Any,
    round_index: int,
    path: Path,
    *,
    student: Any,
    sampling_backend: Any,
) -> bool:
    coordinator, intent, _evidence = _bound_sampling_handle_coordinator(
        config,
        student,
        sampling_backend,
        round_index,
        path,
    )
    from workflow_core.lifecycle.sampling_coordinator import (
        SamplingEvidenceRebuildRequired,
    )

    try:
        return coordinator.validate_completion(intent).complete
    except SamplingEvidenceRebuildRequired:
        raise
    except (OSError, TypeError, ValueError):
        return False


def _expected_member_ids(prefix: str, spec: dict[str, Any]) -> tuple[str, ...]:
    members = int(spec.get("members", 1))
    return tuple(
        prefix if members == 1 else f"{prefix}_{index:02d}"
        for index in range(1, members + 1)
    )


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _valid_gmm_router(path: Path) -> bool:
    from workflow_policies.gmm_reference_tail import validate_gmm_router

    try:
        validate_gmm_router(path / "gmm_router")
    except (OSError, ValueError, KeyError):
        return False
    return True


def _optional_path(root: Path, raw: object) -> Path | None:
    return None if raw is None else root / str(raw)


def _round_value(value: object, index: int) -> int:
    return int(value[index] if isinstance(value, list) else value)
