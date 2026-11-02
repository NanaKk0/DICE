from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from contextlib import nullcontext
from itertools import chain
import json
from pathlib import Path
import shutil
from typing import Any

from ase import Atoms
import numpy as np

from workflow_core.lifecycle.routes import RouteKind, RouteSpec
from workflow_core.lifecycle.sampling_plan import SelectionPlan
from workflow_core.structures.formats import read_extxyz_structures
from workflow_core.lifecycle.screen_use_case import ScreenRoundUseCase
from workflow_core.lifecycle.screen_progress import (
    CountChainSnapshot,
    CountFlowSnapshot,
    CountStep,
    GenericWorkSnapshot,
    RoutingSnapshot,
    RoutingSystemRow,
    ScreenBlockSummary,
    ScorecardRow,
    ScorecardSnapshot,
    SystemCountRow,
)
from workflow_policies.pair_guard import PairGuardThresholds
from workflow_policies.identities import selector_identity


@dataclass(frozen=True, slots=True)
class ScreenRoundInputs:
    """Route-ready policy inputs for one atomic ``rN/screen`` execution."""

    system_ids: tuple[str, ...]
    pair_guard: Mapping[str, Any] | None
    screen_policy: str
    fp_policy: str
    pool: int
    keep: int
    fp_pool: int | None
    teacher_p: Mapping[str, float] | None
    student_p: Mapping[str, float] | None
    entropy: str
    student_force_lo: float | None = None
    student_force_hi: float | None = None
    teacher_force_lo: float | None = None
    teacher_force_hi: float | None = None
    fill_underfilled_from_safe_region: bool = True
    selection_plan: SelectionPlan | None = None
    require_explicit_provenance: bool = False
    prescreen: str = "GMM"
    visualize: bool = True


@dataclass(frozen=True, slots=True)
class ScreenOperationProfileEvidence:
    teacher_gmm_score: str | None = None
    teacher_label: str | None = None
    student_gmm_reference: str | None = None
    student_gmm_score: str | None = None


@dataclass(frozen=True, slots=True)
class JointVectorReadout:
    candidate_vectors: Mapping[str, np.ndarray]
    background_vectors: Mapping[str, np.ndarray]
    descriptor_schema: Mapping[str, object]
    model_identity: str
    dataset_identity: str
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]] | None = None
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]] | None = None
    candidate_frames: Sequence[Atoms] | None = None
    background_frames: Sequence[Atoms] | None = None


@dataclass(frozen=True, slots=True)
class ScreenServices:
    """Backend-operation projection consumed by route execution."""

    score_teacher: Callable[..., Any] | None
    score_student: Callable[..., Any] | None
    label_teacher: Callable[..., list[Atoms]]
    read_student_vectors: Callable[..., Any] | None = None
    operation_profiles: ScreenOperationProfileEvidence = ScreenOperationProfileEvidence()
    cpu_pool_factory: Callable[..., Any] | None = None


@dataclass(frozen=True, slots=True)
class ScreenExecutionRequest:
    route: RouteSpec
    round_index: int
    stage: Path
    source: Any
    inputs: ScreenRoundInputs
    services: ScreenServices
    report: Any
    artifacts: Any
    iter_index: int = 0
    observation_chunk_size: int = 64
    teacher_router: Path | None = None
    student_train_path: Path | None = None
    student_type_map: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScreenExecutionResult:
    train_path: Path
    fp_candidate_path: Path
    observation_count: int
    train_count: int
    fp_candidate_count: int
    proposal_ids: tuple[str, ...] = ()
    visualization_readout: JointVectorReadout | None = None
    visualization_draft: Any | None = None


def execute_screen(request: ScreenExecutionRequest) -> ScreenExecutionResult:
    """Execute one resolved route inside the caller-owned atomic staging block."""
    factory = request.services.cpu_pool_factory
    with factory() if factory is not None else nullcontext(None) as pool:
        return _execute_screen(request, ordered_map=None if pool is None else pool.map)


def _execute_screen(request: ScreenExecutionRequest, *, ordered_map) -> ScreenExecutionResult:

    if not isinstance(request.route, RouteSpec):
        raise TypeError("lifecycle screen execution requires an already-resolved RouteSpec")
    if request.route.kind not in {
        RouteKind.SINGLE_TEACHER_SINGLE_STUDENT,
        RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
        RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
        RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
    }:
        raise NotImplementedError(
            f"lifecycle screen executor does not yet own route {request.route.kind.value!r}"
        )

    core = ScreenRoundUseCase(request.route, ordered_map=ordered_map)
    report = request.report
    report.phase("catalog")
    raw_catalog = request.source.catalog()
    _validate_catalog_selection_provenance(raw_catalog, request.inputs)
    empty_population = not raw_catalog and getattr(request.source, "allows_empty", False)
    catalog = () if empty_population else _deduplicate_observation_catalog(core, raw_catalog)
    frame_steps = (
        CountStep("trajectory", len(raw_catalog)),
        CountStep("catalog", len(catalog)),
    )
    report.snapshot(CountChainSnapshot(label="frames", steps=frame_steps))
    normalized_observations = _metadata_observations(catalog)
    if request.inputs.pair_guard is not None:
        report.phase("pair guard")
    observations, pair_guard_decisions = _apply_pair_guard_source(
        core,
        request.source,
        artifacts=request.artifacts,
        screen_dir=request.stage,
        config=request.inputs.pair_guard,
        records=catalog,
        chunk_size=request.observation_chunk_size,
        ordered_map=ordered_map,
    )
    if pair_guard_decisions is not None:
        frame_steps = (*frame_steps, CountStep("guard", len(observations)))
        report.snapshot(CountChainSnapshot(label="frames", steps=frame_steps))
    if empty_population or (pair_guard_decisions is not None and not observations):
        result = _publish_all_reject(
            request,
            catalog_count=len(catalog),
            normalized_observations=normalized_observations,
            pair_guard_decisions=pair_guard_decisions or (),
        )
    elif request.route.kind is RouteKind.SINGLE_TEACHER_SINGLE_STUDENT:
        result = _execute_single_single(
            request,
            core=core,
            catalog_count=len(catalog),
            normalized_observations=normalized_observations,
            observations=observations,
            pair_guard_decisions=pair_guard_decisions,
        )
    elif request.route.kind is RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT:
        result = _execute_single_teacher_committee_student(
            request,
            core=core,
            catalog_count=len(catalog),
            normalized_observations=normalized_observations,
            observations=observations,
            pair_guard_decisions=pair_guard_decisions,
        )
    elif request.route.kind is RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT:
        result = _execute_committee_teacher_single_student(
            request,
            core=core,
            catalog_count=len(catalog),
            normalized_observations=normalized_observations,
            observations=observations,
        )
    else:
        result = _execute_committee_committee(
            request,
            core=core,
            catalog_count=len(catalog),
            normalized_observations=normalized_observations,
            observations=observations,
        )
    if empty_population:
        request.artifacts.write_json(request.stage / 'screen_summary.json', {
            'schema': 'dice.lifecycle.empty-screen', 'schema_revision': 1,
            'route': request.route.kind.value, 'system_ids': list(request.inputs.system_ids),
            'observations': 0, 'train': 0, 'fp_candidates': 0,
            'screen_policy': request.inputs.screen_policy, 'fp_policy': request.inputs.fp_policy,
            'prescreen': request.inputs.prescreen,
        })
    _ensure_committee_fallback_resolution(request)
    _publish_selection_group_outcomes(request, observations=observations, empty_population=empty_population)
    draft = _prepare_screen_visualization(
        request,
        observations=observations,
        result=result,
    )
    return replace(result, visualization_draft=draft)


def _prepare_screen_visualization(
    request: ScreenExecutionRequest,
    *,
    observations: list[Atoms],
    result: ScreenExecutionResult,
) -> Any | None:
    if (
        not request.inputs.visualize
        or request.inputs.prescreen not in {
            "GMM",
            "JOINT_DIRECT",
            "ELEMENT_JOINT_DIRECT",
        }
        or request.route.kind
        not in {
            RouteKind.SINGLE_TEACHER_SINGLE_STUDENT,
            RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT,
        }
        or request.services.read_student_vectors is None
        or request.student_train_path is None
        or not observations
    ):
        return
    try:
        from workflow_core.lifecycle.visualization import VisualizationDraft
        readout = result.visualization_readout
        if (
            readout is None
            or readout.candidate_element_vectors is None
            or readout.background_element_vectors is None
        ):
            raise ValueError('visualization skipped: no reusable per-element readout; diagnostic-only inference is not run on screen')
        # Borrow already-produced per-element readout, without touching Atoms,
        # stacking vectors, computing digests or doing diagnostic I/O here.
        return VisualizationDraft(
            candidate_ids=tuple(str(frame.info['structure_id']) for frame in observations),
            background_ids=tuple(readout.background_vectors), proposal_ids=result.proposal_ids,
            candidate_vectors=readout.candidate_vectors,
            background_vectors=readout.background_vectors,
            candidate_elements=readout.candidate_element_vectors,
            background_elements=readout.background_element_vectors,
            descriptor_schema=readout.descriptor_schema, type_map=request.student_type_map,
            model_identity=readout.model_identity, dataset_identity=readout.dataset_identity)
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _execute_single_single(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
    pair_guard_decisions: tuple[Any, ...] | None,
) -> ScreenExecutionResult:
    if request.inputs.prescreen in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
        return _execute_joint_single_single(
            request,
            core=core,
            catalog_count=catalog_count,
            normalized_observations=normalized_observations,
            observations=observations,
            pair_guard_decisions=pair_guard_decisions,
        )
    return _execute_student_first_single_single(
        request,
        core=core,
        catalog_count=catalog_count,
        normalized_observations=normalized_observations,
        observations=observations,
        pair_guard_decisions=pair_guard_decisions,
    )
def _execute_joint_single_single(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
    pair_guard_decisions: tuple[Any, ...] | None,
) -> ScreenExecutionResult:
    del normalized_observations, pair_guard_decisions
    inputs = request.inputs
    if inputs.teacher_p is None or inputs.fp_pool is None:
        raise ValueError(
            "single/single JOINT_DIRECT screen requires teacher p and fp_pool"
        )
    if request.services.score_teacher is None:
        raise ValueError("single/single JOINT_DIRECT screen requires teacher scoring")
    report = request.report
    work = request.stage / ".screen_work"
    proposal, readout, outcomes = _joint_student_proposal(
        request, core=core, observations=observations, work=work, report=report
    )
    report.phase("teacher scoring")
    if proposal:
        teacher_score = request.services.score_teacher(
            frames=proposal,
            work_dir=work / "teacher_gmm_score",
            emit_frame_means=inputs.fp_policy in {"DIRECT", "FPS"},
            report=report,
        )
    else:
        from types import SimpleNamespace

        teacher_score = SimpleNamespace(
            frames=(), frame_means=None, descriptor_schema=None
        )
    teacher_scored = list(teacher_score.frames)
    fp_candidates, teacher_trusted, teacher_flags = core.reserve_teacher_reference(
        teacher_scored,
        fp_pool=inputs.fp_pool,
        teacher_floor=float(inputs.teacher_p["floor"]),
        teacher_trust=float(inputs.teacher_p["trust"]),
        teacher_ceiling=inputs.teacher_p.get("ceiling"),
        entropy=inputs.entropy,
    )
    report.phase("routing")
    selector_outcomes: dict[str, dict[str, object]] = {}
    if inputs.screen_policy == "ELEMENT_DIRECT":
        if readout.candidate_element_vectors is None:
            raise ValueError(
                "ELEMENT_DIRECT requires complete per-element candidate evidence"
            )
        structured_teacher_trusted = _replay_with_metadata(
            request.source,
            teacher_trusted,
            chunk_size=request.observation_chunk_size,
        )
        selected, selector_outcomes = core.select_element_per_system(
            structured_teacher_trusted,
            keep=inputs.keep,
            element_vectors=readout.candidate_element_vectors,
            type_map=request.student_type_map,
        )
    else:
        feature_vectors = (
            readout.candidate_vectors
            if inputs.screen_policy in {"DIRECT", "FPS"}
            else None
        )
        selected = core.select_per_system(
            teacher_trusted,
            pool=-1,
            keep=inputs.keep,
            policy=inputs.screen_policy,
            entropy=f"{inputs.entropy}/post-teacher",
            feature_vectors=feature_vectors,
        )
    selected = core.restore_order(observations, selected)
    selected_frames = _replay_with_metadata(
        request.source, selected, chunk_size=request.observation_chunk_size
    )
    train = (
        request.services.label_teacher(
            frames=selected_frames,
            work_dir=work / "teacher_label",
            report=report,
            trusted_below=None,
        )
        if selected_frames
        else []
    )
    fp_candidate_frames = _replay_with_metadata(
        request.source, fp_candidates, chunk_size=request.observation_chunk_size
    )
    for frame in train:
        frame.info["acquisition_route"] = "student_joint_direct_teacher_trusted"
    for frame in fp_candidate_frames:
        frame.info["acquisition_route"] = "student_joint_direct_teacher_fp_candidate"
    report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", train, allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", fp_candidate_frames, allow_empty=True
    )
    teacher_means = None
    if teacher_score.frame_means is not None:
        teacher_means = {
            str(frame.info["structure_id"]): row
            for frame, row in zip(
                teacher_scored, teacher_score.frame_means, strict=True
            )
        }
    _publish_gmm_representative_sidecars(
        artifacts=request.artifacts,
        stage=request.stage,
        fp_candidates=fp_candidate_frames,
        selected_train=selected_frames,
        teacher_means=teacher_means,
        student_means=readout.candidate_vectors,
        teacher_descriptor_schema=teacher_score.descriptor_schema,
        student_descriptor_schema=readout.descriptor_schema,
        fp_policy=inputs.fp_policy,
        screen_policy=inputs.screen_policy,
    )
    train_ids = {str(frame.info["structure_id"]) for frame in selected}
    fp_ids = {str(frame.info["structure_id"]) for frame in fp_candidates}
    route_thresholds = {
        "prescreen": selector_identity(inputs.prescreen).as_dict(),
        "teacher_p": dict(inputs.teacher_p),
        "quota": {
            "fp_pool": inputs.fp_pool,
            "pool": inputs.pool,
            "keep": inputs.keep,
        },
        "fill_underfilled_from_safe_region": False,
        "selector": selector_identity(inputs.screen_policy).as_dict(),
    }
    teacher_metrics = {
        str(frame.info["structure_id"]): float(frame.info["teacher_p"])
        for frame in teacher_scored
    }
    if inputs.prescreen == "ELEMENT_JOINT_DIRECT":
        if request.student_train_path is None:
            raise ValueError(
                "ELEMENT_JOINT_DIRECT publication requires student training membership"
            )
        readout_performed = readout.candidate_element_vectors is not None
        if readout_performed != (readout.background_element_vectors is not None):
            raise ValueError(
                "ELEMENT_JOINT_DIRECT publication has incomplete partition evidence"
            )
        if not readout_performed and any(
            not bool(outcome["bypass"]) for outcome in outcomes.values()
        ):
            raise ValueError(
                "ELEMENT_JOINT_DIRECT omitted required non-bypass descriptor evidence"
            )
        request.artifacts.publish_element_direct_screen_evidence(
            request.stage / "element_direct_screen.npz",
            observations=_joint_candidate_frames(request, readout, observations),
            background_frames=_joint_background_frames(request, readout),
            candidate_element_vectors=readout.candidate_element_vectors or {},
            background_element_vectors=readout.background_element_vectors or {},
            descriptor_schema=readout.descriptor_schema,
            type_map=request.student_type_map,
            model_identity=readout.model_identity,
            dataset_identity=readout.dataset_identity,
            prescreen=inputs.prescreen,
            pool=inputs.pool,
            keep=inputs.keep,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
            route_kind=request.route.kind.value,
            proposal_ids=[str(frame.info["structure_id"]) for frame in proposal],
            selector_candidate_ids={
                str(frame.info["structure_id"]) for frame in teacher_trusted
            },
            normal_train_ids=train_ids,
            train_ids=train_ids,
            fp_ids=fp_ids,
            teacher_metrics=teacher_metrics,
            outcomes=outcomes,
            selector_outcomes=selector_outcomes,
            route_thresholds=route_thresholds,
            readout_performed=readout_performed,
        )
    else:
        request.artifacts.publish_joint_direct_screen_evidence(
            request.stage / "joint_direct_screen.npz",
            observations=observations,
            candidate_vectors=readout.candidate_vectors,
            background_vectors=readout.background_vectors,
            descriptor_schema=readout.descriptor_schema,
            model_identity=readout.model_identity,
            dataset_identity=readout.dataset_identity,
            pool=inputs.pool,
            keep=inputs.keep,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
            route_kind=request.route.kind.value,
            proposal_ids=[str(frame.info["structure_id"]) for frame in proposal],
            normal_train_ids=train_ids,
            train_ids=train_ids,
            fp_ids=fp_ids,
            teacher_metrics=teacher_metrics,
            outcomes=outcomes,
            route_thresholds=route_thresholds,
        )
    _publish_joint_single_teacher_round_screen_report(
        request,
        teacher_frames=teacher_scored,
        teacher_flags=teacher_flags,
        teacher_thresholds={
            "floor": float(inputs.teacher_p["floor"]),
            "trust": float(inputs.teacher_p["trust"]),
            "ceiling": inputs.teacher_p.get("ceiling"),
        },
        observations=observations,
        outcomes=outcomes,
        published_train=len(train),
        published_fp_candidates=len(fp_candidate_frames),
    )
    report.snapshot(
        _screen_output_counts(
            inputs.system_ids,
            observations=observations,
            train=train,
            fp_candidates=fp_candidate_frames,
        )
    )
    report.summary(
        ScreenBlockSummary(
            observations=catalog_count,
            train=len(train),
            fp_candidates=len(fp_candidate_frames),
        )
    )
    _close_score_view(teacher_score)
    _cleanup_screen_work(work)
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=len(train),
        fp_candidate_count=len(fp_candidate_frames),
        proposal_ids=tuple(str(frame.info["structure_id"]) for frame in proposal),
        visualization_readout=readout,
    )


def _joint_candidate_frames(request, readout, observations):
    cached = getattr(readout, "candidate_frames", None)
    if cached is not None:
        expected = tuple(str(frame.info["structure_id"]) for frame in observations)
        if tuple(str(frame.info["structure_id"]) for frame in cached) != expected:
            raise ValueError("joint readout cached candidate geometry changed canonical order")
        return list(cached)
    return _replay_with_metadata(request.source, observations,
                                 chunk_size=request.observation_chunk_size)


def _joint_background_frames(request, readout):
    cached = getattr(readout, "background_frames", None)
    if cached is not None:
        expected = tuple(str(identity) for identity in readout.background_vectors)
        if tuple(str(frame.info["structure_id"]) for frame in cached) != expected:
            raise ValueError("joint readout cached background geometry changed canonical order")
        return list(cached)
    return read_extxyz_structures(request.student_train_path, allow_empty=False)


def _joint_student_proposal(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    observations: list[Atoms],
    work: Path,
    report: Any,
) -> tuple[list[Atoms], JointVectorReadout, dict[str, dict[str, int | bool]]]:
    inputs = request.inputs
    groups: dict[str, int] = {}
    for frame in observations:
        group_id = str(frame.info["system_id"])
        groups[group_id] = groups.get(group_id, 0) + 1
    needs_joint = any(count > inputs.pool for count in groups.values())
    needs_vectors = (
        needs_joint
        or inputs.screen_policy in {"DIRECT", "ELEMENT_DIRECT", "FPS"}
    )
    if needs_vectors:
        if request.services.read_student_vectors is None:
            raise ValueError(
                "JOINT_DIRECT screen requires student structural-vector readout"
            )
        report.phase("student structural readout")
        readout = request.services.read_student_vectors(
            frames=observations,
            work_dir=work / "student_joint_direct",
            report=report,
        )
    else:
        readout = JointVectorReadout({}, {}, {}, "", "")
    if inputs.prescreen == "ELEMENT_JOINT_DIRECT" and needs_joint:
        if (
            readout.candidate_element_vectors is None
            or readout.background_element_vectors is None
            or request.student_train_path is None
        ):
            raise ValueError(
                "ELEMENT_JOINT_DIRECT requires complete per-element candidate/background evidence"
            )
        background_frames = _joint_background_frames(request, readout)
        element_candidates = _joint_candidate_frames(request, readout, observations)
        proposal, outcomes = core.select_student_element_joint_proposal(
            element_candidates,
            background_frames=background_frames,
            pool=inputs.pool,
            candidate_element_vectors=readout.candidate_element_vectors,
            background_element_vectors=readout.background_element_vectors,
            type_map=request.student_type_map,
        )
    else:
        proposal, outcomes = core.select_student_joint_proposal(
            observations,
            pool=inputs.pool,
            candidate_vectors=readout.candidate_vectors,
            background_vectors=readout.background_vectors,
        )
    return proposal, readout, outcomes


def _execute_student_first_single_single(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
    pair_guard_decisions: tuple[Any, ...] | None,
) -> ScreenExecutionResult:
    """Execute the proposal-scoped single/single directional policy."""

    inputs = request.inputs
    if inputs.teacher_p is None or inputs.student_p is None or inputs.fp_pool is None:
        raise ValueError("single/single screen requires teacher/student p bands and fp_pool")
    if request.services.score_teacher is None or request.services.score_student is None:
        raise ValueError("single/single screen requires both GMM score services")
    report = request.report
    work = request.stage / ".screen_work"

    report.phase("student scoring")
    student_score = request.services.score_student(
        frames=observations,
        router_dir=request.stage / "gmm_router",
        work_dir=work / "student_gmm",
        emit_features=inputs.screen_policy in {"DIRECT", "FPS", "ELEMENT_DIRECT"},
        report=report,
    )
    student_values = student_score.values
    annotated: list[Atoms] = []
    for frame, (scored_identity, probability) in zip(
        observations,
        _student_probability_items(student_score),
        strict=True,
    ):
        item = frame.copy()
        identity = str(item.info["structure_id"])
        if identity != scored_identity:
            raise ValueError("student compact score order changed before routing")
        item.info["student_p"] = probability
        item.info["acquisition_route"] = "student_gmm_proposal"
        annotated.append(item)
    proposal, student_flags = core.propose_committee_teacher_single_student(
        annotated,
        pool=inputs.pool,
        floor=float(inputs.student_p["floor"]),
        ceiling=inputs.student_p.get("ceiling"),
        entropy=f"{inputs.entropy}/student-proposal",
    )
    ceiling = inputs.student_p.get("ceiling")
    for frame in annotated:
        identity = str(frame.info["structure_id"])
        student_flags[identity].update(
            {
                "student_covered": bool(
                    ceiling is not None and float(frame.info["student_p"]) >= float(ceiling)
                ),
                "train_selected": False,
                "safe_fill_eligible": False,
                "safe_fill_attempted": False,
                "safe_fill_accepted": False,
            }
        )

    report.phase("teacher scoring")
    if proposal:
        teacher_score = request.services.score_teacher(
            frames=proposal,
            work_dir=work / "teacher_gmm_score",
            emit_frame_means=inputs.fp_policy in {"DIRECT", "FPS"},
            report=report,
        )
    else:
        from types import SimpleNamespace

        teacher_score = SimpleNamespace(
            frames=(), frame_means=None, descriptor_schema=None
        )
    normal_teacher_scored = list(teacher_score.frames)
    fp_candidates, teacher_trusted, teacher_flags = core.reserve_teacher_reference(
        normal_teacher_scored,
        fp_pool=inputs.fp_pool,
        teacher_floor=float(inputs.teacher_p["floor"]),
        teacher_trust=float(inputs.teacher_p["trust"]),
        teacher_ceiling=inputs.teacher_p.get("ceiling"),
        entropy=inputs.entropy,
    )

    report.phase("routing")
    element_readout: JointVectorReadout | None = None
    selector_outcomes: dict[str, dict[str, object]] = {}
    if inputs.screen_policy == "ELEMENT_DIRECT":
        if (
            request.services.read_student_vectors is None
            or request.student_train_path is None
        ):
            raise ValueError(
                "ELEMENT_DIRECT screen requires student structural-vector readout"
            )
        report.phase("student structural readout")
        described_frames = _replay_with_metadata(
            request.source,
            observations,
            chunk_size=request.observation_chunk_size,
        )
        element_readout = request.services.read_student_vectors(
            frames=described_frames,
            work_dir=work / "student_element_direct",
            report=report,
            materialized=True,
            **({"reused_score": student_score} if getattr(student_score, "element_frame_means", None) is not None else {}),
        )
        if element_readout.candidate_element_vectors is None:
            raise ValueError(
                "ELEMENT_DIRECT requires complete per-element candidate evidence"
            )
        described_by_id = {
            str(frame.info["structure_id"]): frame for frame in described_frames
        }
        element_candidates = [
            described_by_id[str(frame.info["structure_id"])]
            for frame in teacher_trusted
        ]
        normal_selected, selector_outcomes = core.select_element_per_system(
            element_candidates,
            keep=inputs.keep,
            element_vectors=element_readout.candidate_element_vectors,
            type_map=request.student_type_map,
        )
    else:
        normal_selected = core.select_per_system(
            teacher_trusted,
            pool=-1,
            keep=inputs.keep,
            policy=inputs.screen_policy,
            entropy=f"{inputs.entropy}/post-teacher",
            feature_vectors=student_score.frame_means,
        )
    normal_train_ids = {
        str(frame.info["structure_id"]) for frame in normal_selected
    }
    for identity in normal_train_ids:
        student_flags[identity]["train_selected"] = True
    safe_fill = core.fill_student_reference(
        annotated,
        normal_selected=normal_selected,
        flags=student_flags,
        keep=inputs.keep,
        enabled=inputs.fill_underfilled_from_safe_region,
        entropy=inputs.entropy,
    )
    for identity, flags in student_flags.items():
        flags["safe_fill_attempted"] = bool(flags.get("safe_fill_selected", False))
        flags["safe_fill_accepted"] = False
    safe_teacher_scored: list[Atoms] = []
    safe_fill_accepted_ids: set[str] = set()
    if safe_fill:
        report.phase("teacher safe-fill scoring")
        safe_teacher_score = request.services.score_teacher(
            frames=safe_fill,
            work_dir=work / "teacher_gmm_safe_fill",
            emit_frame_means=False,
            report=report,
        )
        safe_teacher_scored = list(safe_teacher_score.frames)
        _ignored_fp, safe_trusted, safe_teacher_flags = core.reserve_teacher_reference(
            safe_teacher_scored,
            fp_pool=1,
            teacher_floor=float(inputs.teacher_p["floor"]),
            teacher_trust=float(inputs.teacher_p["trust"]),
            teacher_ceiling=inputs.teacher_p.get("ceiling"),
            entropy=f"{inputs.entropy}/safe-fill",
        )
        for flags in safe_teacher_flags.values():
            flags["teacher_fp_candidate"] = False
        teacher_flags.update(safe_teacher_flags)
        safe_fill_accepted_ids = {
            str(frame.info["structure_id"]) for frame in safe_trusted
        }
        for identity in safe_fill_accepted_ids:
            student_flags[identity]["safe_fill_accepted"] = True
            student_flags[identity]["train_selected"] = True
    teacher_scored_frames = [*normal_teacher_scored, *safe_teacher_scored]
    train_ids = normal_train_ids | safe_fill_accepted_ids
    selected = [
        frame
        for frame in annotated
        if str(frame.info["structure_id"]) in train_ids
    ]

    selected_frames = _replay_with_metadata(
        request.source, selected, chunk_size=request.observation_chunk_size
    )
    train = (
        request.services.label_teacher(
            frames=selected_frames,
            work_dir=work / "teacher_label",
            report=report,
            trusted_below=None,
        )
        if selected_frames
        else []
    )
    for frame in fp_candidates:
        frame.info["acquisition_route"] = "student_proposal_teacher_gmm_fp_candidate"
    fp_candidate_frames = _replay_with_metadata(
        request.source, fp_candidates, chunk_size=request.observation_chunk_size
    )

    report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", train, allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", fp_candidate_frames, allow_empty=True
    )
    teacher_means = None
    if teacher_score.frame_means is not None:
        teacher_means = {
            str(frame.info["structure_id"]): row
            for frame, row in zip(
                normal_teacher_scored, teacher_score.frame_means, strict=True
            )
        }
    _publish_gmm_representative_sidecars(
        artifacts=request.artifacts,
        stage=request.stage,
        fp_candidates=fp_candidate_frames,
        selected_train=selected_frames,
        teacher_means=teacher_means,
        student_means=student_score.frame_means,
        teacher_descriptor_schema=teacher_score.descriptor_schema,
        student_descriptor_schema=student_score.descriptor_schema,
        fp_policy=inputs.fp_policy,
        screen_policy=inputs.screen_policy,
    )

    teacher_by_id = {
        str(frame.info["structure_id"]): frame for frame in teacher_scored_frames
    }
    audit_observations: list[Atoms] = []
    for frame in annotated:
        identity = str(frame.info["structure_id"])
        item = frame.copy()
        if identity in teacher_by_id:
            item.info["teacher_p"] = teacher_by_id[identity].info["teacher_p"]
        audit_observations.append(item)
    fp_ids = {str(frame.info["structure_id"]) for frame in fp_candidates}
    request.artifacts.publish_gmm_screen_scores(
        request.stage / "gmm_scores.npz",
        observations=audit_observations,
        teacher_flags=teacher_flags,
        student_flags=student_flags,
        student_values=student_values,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        teacher_p=inputs.teacher_p,
        student_p=inputs.student_p,
        safe_fill_enabled=inputs.fill_underfilled_from_safe_region,
        routing_policy=request.route.screen_routing_policy_id.as_dict(),
        pool=inputs.pool,
        keep=inputs.keep,
        fp_pool=inputs.fp_pool,
        screen_policy=inputs.screen_policy,
    )
    if inputs.screen_policy == "ELEMENT_DIRECT":
        if (
            element_readout is None
            or element_readout.candidate_element_vectors is None
            or element_readout.background_element_vectors is None
            or request.student_train_path is None
        ):
            raise ValueError(
                "ELEMENT_DIRECT publication requires complete per-element evidence"
            )
        if request.route.screen_routing_policy_id is None:
            raise ValueError("single/single route is missing its routing identity")
        request.artifacts.publish_element_direct_screen_evidence(
            request.stage / "element_direct_screen.npz",
            observations=described_frames,
            background_frames=_joint_background_frames(request, element_readout),
            candidate_element_vectors=element_readout.candidate_element_vectors,
            background_element_vectors=element_readout.background_element_vectors,
            descriptor_schema=element_readout.descriptor_schema,
            type_map=request.student_type_map,
            model_identity=element_readout.model_identity,
            dataset_identity=element_readout.dataset_identity,
            prescreen=inputs.prescreen,
            pool=inputs.pool,
            keep=inputs.keep,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
            route_kind=request.route.kind.value,
            proposal_ids=[str(frame.info["structure_id"]) for frame in proposal],
            selector_candidate_ids={
                str(frame.info["structure_id"]) for frame in teacher_trusted
            },
            normal_train_ids=normal_train_ids,
            train_ids=train_ids,
            fp_ids=fp_ids,
            teacher_metrics={
                str(frame.info["structure_id"]): float(frame.info["teacher_p"])
                for frame in normal_teacher_scored
            },
            outcomes={},
            selector_outcomes=selector_outcomes,
            route_thresholds={
                "routing_policy": request.route.screen_routing_policy_id.as_dict(),
                "teacher_p": dict(inputs.teacher_p),
                "student_p": dict(inputs.student_p),
                "quota": {
                    "fp_pool": inputs.fp_pool,
                    "pool": inputs.pool,
                    "keep": inputs.keep,
                },
                "fill_underfilled_from_safe_region": inputs.fill_underfilled_from_safe_region,
                "selector": selector_identity(inputs.screen_policy).as_dict(),
            },
        )
    if pair_guard_decisions is not None:
        if inputs.pair_guard is None:
            raise ValueError("pair-guard decisions require configured thresholds")
        request.artifacts.validate_gmm_pair_guard_alignment(
            request.stage / "gmm_scores.npz",
            request.stage / "pair_guard.npz",
            thresholds=PairGuardThresholds(
                minimum=float(inputs.pair_guard["min"]),
                pairs=dict(inputs.pair_guard.get("pairs", {})),
            ),
        )
    _publish_gmm_screen_summary(
        artifacts=request.artifacts,
        stage=request.stage,
        normalized_observations=normalized_observations,
        observations=teacher_scored_frames,
        pair_guard_config=inputs.pair_guard,
        pair_guard_decisions=pair_guard_decisions,
        teacher_flags=teacher_flags,
        student_flags=student_flags,
        student_values=student_values,
        train=train,
        normal_train_ids=normal_train_ids,
        fp_candidates=fp_candidate_frames,
        teacher_p=inputs.teacher_p,
        student_p=inputs.student_p,
        pool=inputs.pool,
        keep=inputs.keep,
        fill_underfilled_from_safe_region=inputs.fill_underfilled_from_safe_region,
        teacher_router=request.teacher_router,
        student_router=request.stage / "gmm_router",
        operation_profiles=request.services.operation_profiles,
        screen_policy=inputs.screen_policy,
        fp_policy=inputs.fp_policy,
        normal_teacher_ids={
            str(frame.info["structure_id"]) for frame in normal_teacher_scored
        },
    )
    _publish_gmm_round_screen_report(
        request,
        teacher_frames=teacher_scored_frames,
        teacher_flags=teacher_flags,
        teacher_thresholds={
            "floor": float(inputs.teacher_p["floor"]),
            "trust": float(inputs.teacher_p["trust"]),
            "ceiling": inputs.teacher_p.get("ceiling"),
        },
        student_frames=annotated,
        student_flags=student_flags,
        student_thresholds={
            "floor": float(inputs.student_p["floor"]),
            "ceiling": inputs.student_p.get("ceiling"),
        },
        published_train=len(train),
        published_fp_candidates=len(fp_candidate_frames),
    )
    report.snapshot(
        _screen_output_counts(
            inputs.system_ids,
            observations=annotated,
            train=train,
            fp_candidates=fp_candidate_frames,
        )
    )
    report.summary(
        ScreenBlockSummary(
            observations=catalog_count,
            train=len(train),
            fp_candidates=len(fp_candidate_frames),
        )
    )
    _close_score_view(teacher_score)
    if "safe_teacher_score" in locals():
        _close_score_view(safe_teacher_score)
    _close_score_view(student_score)
    _cleanup_screen_work(work)
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=len(train),
        fp_candidate_count=len(fp_candidate_frames),
        proposal_ids=tuple(str(frame.info["structure_id"]) for frame in proposal),
        visualization_readout=element_readout,
    )


def _publish_gmm_round_screen_report(
    request: ScreenExecutionRequest,
    *,
    teacher_frames: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    teacher_thresholds: Mapping[str, float | None],
    student_frames: Sequence[Atoms],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_thresholds: Mapping[str, float | None],
    published_train: int,
    published_fp_candidates: int,
) -> None:
    """Best-effort diagnostic publication for a successful GMM route."""
    try:
        from workflow_core.lifecycle.screen_reporting import build_gmm_round_screen_report

        snapshot = build_gmm_round_screen_report(
            iter_index=request.iter_index,
            round_index=request.round_index,
            route=request.route,
            system_ids=request.inputs.system_ids,
            teacher_frames=teacher_frames,
            teacher_flags=teacher_flags,
            teacher_thresholds=teacher_thresholds,
            student_frames=student_frames,
            student_flags=student_flags,
            student_thresholds=student_thresholds,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        request.artifacts.publish_screen_report(
            request.stage / "screen_report.json", snapshot
        )
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _publish_joint_single_teacher_round_screen_report(
    request: ScreenExecutionRequest,
    *,
    teacher_frames: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    teacher_thresholds: Mapping[str, float | None],
    observations: Sequence[Atoms],
    outcomes: Mapping[str, Mapping[str, int | bool]],
    published_train: int,
    published_fp_candidates: int,
) -> None:
    """Best-effort diagnostic publication for single/single joint DIRECT."""
    try:
        from workflow_core.lifecycle.screen_reporting import (
            build_joint_single_teacher_round_screen_report,
        )

        snapshot = build_joint_single_teacher_round_screen_report(
            iter_index=request.iter_index,
            round_index=request.round_index,
            route=request.route,
            system_ids=request.inputs.system_ids,
            teacher_frames=teacher_frames,
            teacher_flags=teacher_flags,
            teacher_thresholds=teacher_thresholds,
            observations=observations,
            outcomes=outcomes,
            pool=request.inputs.pool,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        request.artifacts.publish_screen_report(
            request.stage / "screen_report.json", snapshot
        )
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _publish_joint_committee_teacher_round_screen_report(
    request: ScreenExecutionRequest,
    *,
    teacher_frames: Sequence[Atoms],
    observations: Sequence[Atoms],
    outcomes: Mapping[str, Mapping[str, int | bool]],
    published_train: int,
    published_fp_candidates: int,
) -> None:
    """Best-effort diagnostic publication for committee/single joint DIRECT."""
    try:
        from workflow_core.lifecycle.screen_reporting import (
            build_joint_committee_teacher_round_screen_report,
        )

        snapshot = build_joint_committee_teacher_round_screen_report(
            iter_index=request.iter_index,
            round_index=request.round_index,
            route=request.route,
            system_ids=request.inputs.system_ids,
            teacher_frames=teacher_frames,
            teacher_lo=float(request.inputs.teacher_force_lo),
            teacher_hi=float(request.inputs.teacher_force_hi),
            observations=observations,
            outcomes=outcomes,
            pool=request.inputs.pool,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        request.artifacts.publish_screen_report(
            request.stage / "screen_report.json", snapshot
        )
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _publish_gmm_committee_round_screen_report(
    request: ScreenExecutionRequest,
    *,
    teacher_frames: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    teacher_thresholds: Mapping[str, float | None],
    student_frames: Sequence[Atoms],
    student_lo: float,
    student_hi: float,
    published_train: int,
    published_fp_candidates: int,
) -> None:
    """Best-effort diagnostic publication for the GMM/committee route."""
    try:
        from workflow_core.lifecycle.screen_reporting import (
            build_gmm_committee_round_screen_report,
        )

        snapshot = build_gmm_committee_round_screen_report(
            iter_index=request.iter_index,
            round_index=request.round_index,
            route=request.route,
            system_ids=request.inputs.system_ids,
            teacher_frames=teacher_frames,
            teacher_flags=teacher_flags,
            teacher_thresholds=teacher_thresholds,
            student_frames=student_frames,
            student_lo=student_lo,
            student_hi=student_hi,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        request.artifacts.publish_screen_report(
            request.stage / "screen_report.json", snapshot
        )
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _publish_committee_gmm_round_screen_report(
    request: ScreenExecutionRequest,
    *,
    teacher_frames: Sequence[Atoms],
    teacher_lo: float,
    teacher_hi: float,
    student_frames: Sequence[Atoms],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_thresholds: Mapping[str, float | None],
    published_train: int,
    published_fp_candidates: int,
) -> None:
    """Best-effort diagnostic publication for the committee/GMM route."""
    try:
        from workflow_core.lifecycle.screen_reporting import (
            build_committee_gmm_round_screen_report,
        )

        snapshot = build_committee_gmm_round_screen_report(
            iter_index=request.iter_index,
            round_index=request.round_index,
            route=request.route,
            system_ids=request.inputs.system_ids,
            teacher_frames=teacher_frames,
            teacher_lo=teacher_lo,
            teacher_hi=teacher_hi,
            student_frames=student_frames,
            student_flags=student_flags,
            student_thresholds=student_thresholds,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        request.artifacts.publish_screen_report(
            request.stage / "screen_report.json", snapshot
        )
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _student_gmm_report_flags(
    frames: Sequence[Atoms],
    flags: Mapping[str, Mapping[str, bool]],
    *,
    ceiling: float | None,
) -> dict[str, dict[str, bool]]:
    """Add the diagnostic covered set without changing proposal flags."""
    result = {identity: dict(values) for identity, values in flags.items()}
    for frame in frames:
        identity = str(frame.info["structure_id"])
        result.setdefault(identity, {})["student_covered"] = bool(
            ceiling is not None and float(frame.info["student_p"]) >= float(ceiling)
        )
    return result


def _publish_committee_committee_round_screen_report(
    request: ScreenExecutionRequest,
    *,
    teacher_frames: Sequence[Atoms],
    teacher_lo: float,
    teacher_hi: float,
    student_frames: Sequence[Atoms],
    student_lo: float,
    student_hi: float,
    published_train: int,
    published_fp_candidates: int,
) -> None:
    """Best-effort diagnostic publication for the committee/committee route."""
    try:
        from workflow_core.lifecycle.screen_reporting import (
            build_committee_committee_round_screen_report,
        )

        snapshot = build_committee_committee_round_screen_report(
            iter_index=request.iter_index,
            round_index=request.round_index,
            route=request.route,
            system_ids=request.inputs.system_ids,
            teacher_frames=teacher_frames,
            teacher_lo=teacher_lo,
            teacher_hi=teacher_hi,
            student_frames=student_frames,
            student_lo=student_lo,
            student_hi=student_hi,
            published_train=published_train,
            published_fp_candidates=published_fp_candidates,
        )
        request.artifacts.publish_screen_report(
            request.stage / "screen_report.json", snapshot
        )
    except Exception as exc:
        warning = getattr(request.report, "warning", None)
        if callable(warning):
            warning(exc)


def _execute_single_teacher_committee_student(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
    pair_guard_decisions: tuple[Any, ...] | None,
) -> ScreenExecutionResult:
    return _execute_student_first_single_teacher_committee_student(
        request,
        core=core,
        catalog_count=catalog_count,
        normalized_observations=normalized_observations,
        observations=observations,
        pair_guard_decisions=pair_guard_decisions,
    )
def _execute_student_first_single_teacher_committee_student(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
    pair_guard_decisions: tuple[Any, ...] | None,
) -> ScreenExecutionResult:
    """Execute the deviation-proposal-scoped single/committee policy."""

    inputs = request.inputs
    if (
        inputs.teacher_p is None
        or inputs.fp_pool is None
        or inputs.student_force_lo is None
        or inputs.student_force_hi is None
    ):
        raise ValueError(
            "single-teacher/committee-student screen requires teacher p, fp_pool, and student force-deviation inputs"
        )
    if request.services.score_teacher is None:
        raise ValueError(
            "single-teacher/committee-student screen requires teacher GMM scoring"
        )
    report = request.report
    work = request.stage / ".screen_work"
    from workflow_policies.resolution import resolve_element_direct_policy

    policy_resolution = resolve_element_direct_policy(
        inputs.screen_policy, descriptor_model_shape="committee"
    )

    report.phase("student proposal")
    proposal, student_flags = core.propose_single_teacher_committee_student(
        observations,
        pool=inputs.pool,
        lo=inputs.student_force_lo,
        hi=inputs.student_force_hi,
    )

    report.phase("teacher scoring")
    if proposal:
        teacher_score = request.services.score_teacher(
            frames=proposal,
            work_dir=work / "teacher_gmm_score",
            emit_frame_means=inputs.fp_policy in {"DIRECT", "FPS"},
            report=report,
        )
    else:
        from types import SimpleNamespace

        teacher_score = SimpleNamespace(
            frames=(), frame_means=None, descriptor_schema=None
        )
    normal_teacher_scored = list(teacher_score.frames)
    fp_candidates, teacher_trusted, teacher_flags = core.reserve_teacher_reference(
        normal_teacher_scored,
        fp_pool=inputs.fp_pool,
        teacher_floor=float(inputs.teacher_p["floor"]),
        teacher_trust=float(inputs.teacher_p["trust"]),
        teacher_ceiling=inputs.teacher_p.get("ceiling"),
        entropy=inputs.entropy,
    )

    report.phase("post-teacher selection")
    normal_selected = core.select_per_system(
        teacher_trusted,
        pool=-1,
        keep=inputs.keep,
        policy=policy_resolution.effective_policy,
        entropy=f"{inputs.entropy}/post-teacher",
        feature_vectors=None,
    )
    normal_train_ids = {
        str(frame.info["structure_id"]) for frame in normal_selected
    }
    for identity in normal_train_ids:
        student_flags[identity]["train_selected"] = True

    safe_attempts = core.fill_committee_student(
        observations,
        normal_selected=normal_selected,
        flags=student_flags,
        keep=inputs.keep,
        lo=inputs.student_force_lo,
        enabled=inputs.fill_underfilled_from_safe_region,
        entropy=f"{inputs.entropy}/committee-student",
    )
    safe_teacher_scored: list[Atoms] = []
    safe_accepted_ids: set[str] = set()
    if safe_attempts:
        report.phase("teacher safe-fill scoring")
        safe_score = request.services.score_teacher(
            frames=safe_attempts,
            work_dir=work / "teacher_gmm_safe_fill",
            emit_frame_means=False,
            report=report,
        )
        safe_teacher_scored = list(safe_score.frames)
        _ignored_fp, safe_trusted, safe_teacher_flags = core.reserve_teacher_reference(
            safe_teacher_scored,
            fp_pool=1,
            teacher_floor=float(inputs.teacher_p["floor"]),
            teacher_trust=float(inputs.teacher_p["trust"]),
            teacher_ceiling=inputs.teacher_p.get("ceiling"),
            entropy=f"{inputs.entropy}/safe-fill",
        )
        for flags in safe_teacher_flags.values():
            flags["teacher_fp_candidate"] = False
        teacher_flags.update(safe_teacher_flags)
        safe_accepted_ids = {
            str(frame.info["structure_id"]) for frame in safe_trusted
        }
    for identity, flags in student_flags.items():
        flags["safe_fill_accepted"] = identity in safe_accepted_ids
        if identity in safe_accepted_ids:
            flags["train_selected"] = True

    train_ids = normal_train_ids | safe_accepted_ids
    selected = [
        frame
        for frame in observations
        if str(frame.info["structure_id"]) in train_ids
    ]
    selected_frames = _replay_with_metadata(
        request.source, selected, chunk_size=request.observation_chunk_size
    )
    train = (
        request.services.label_teacher(
            frames=selected_frames,
            work_dir=work / "teacher_label",
            report=report,
            trusted_below=None,
        )
        if selected_frames
        else []
    )

    for frame in fp_candidates:
        frame.info["acquisition_route"] = (
            "student_deviation_proposal_teacher_gmm_fp_candidate"
        )
    fp_candidate_frames = _replay_with_metadata(
        request.source, fp_candidates, chunk_size=request.observation_chunk_size
    )

    report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", train, allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", fp_candidate_frames, allow_empty=True
    )
    if inputs.fp_policy in {"DIRECT", "FPS"} and fp_candidate_frames:
        if teacher_score.frame_means is None or teacher_score.descriptor_schema is None:
            raise ValueError(
                f"{inputs.fp_policy} mixed-route FP selection requires teacher readout-input frame means"
            )
        normal_means = {
            str(frame.info["structure_id"]): row
            for frame, row in zip(
                normal_teacher_scored, teacher_score.frame_means, strict=True
            )
        }
        _write_frame_means(
            request.artifacts,
            request.stage / "representative_features.npz",
            fp_candidate_frames,
            normal_means,
            descriptor_schema=teacher_score.descriptor_schema,
        )

    teacher_scored_frames = [*normal_teacher_scored, *safe_teacher_scored]
    teacher_by_id = {
        str(frame.info["structure_id"]): frame for frame in teacher_scored_frames
    }
    audit_observations: list[Atoms] = []
    for frame in observations:
        identity = str(frame.info["structure_id"])
        item = frame.copy()
        if identity in teacher_by_id:
            item.info["teacher_p"] = teacher_by_id[identity].info["teacher_p"]
        audit_observations.append(item)
    fp_ids = {str(frame.info["structure_id"]) for frame in fp_candidates}
    thresholds = _single_teacher_committee_student_thresholds(
        inputs, route=request.route
    )
    request.artifacts.publish_student_first_single_teacher_committee_student_audit(
        request.stage / "mixed_route_scores.npz",
        observations=audit_observations,
        teacher_flags=teacher_flags,
        student_flags=student_flags,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        thresholds=thresholds,
        safe_fill_enabled=inputs.fill_underfilled_from_safe_region,
    )
    summary = request.artifacts.mixed_route_summary(
        observations=audit_observations,
        catalog_observations=normalized_observations,
        teacher_flags=teacher_flags,
        student_flags=student_flags,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        thresholds=thresholds,
        keep=inputs.keep,
        fill_underfilled_from_safe_region=inputs.fill_underfilled_from_safe_region,
        normal_teacher_ids={
            str(frame.info["structure_id"]) for frame in normal_teacher_scored
        },
        teacher_scored_ids={
            str(frame.info["structure_id"]) for frame in teacher_scored_frames
        },
    )
    if policy_resolution.fell_back:
        summary["selector_resolution"] = _publish_committee_fallback_resolution(
            request,
            candidates=_selector_candidate_population(
                teacher_trusted, pool=-1
            ),
            selected=normal_selected,
            entropy=f"{inputs.entropy}/post-teacher",
        )
    request.artifacts.write_json(request.stage / "screen_summary.json", summary)
    _publish_gmm_committee_round_screen_report(
        request,
        teacher_frames=teacher_scored_frames,
        teacher_flags=teacher_flags,
        teacher_thresholds={
            "floor": float(inputs.teacher_p["floor"]),
            "trust": float(inputs.teacher_p["trust"]),
            "ceiling": inputs.teacher_p.get("ceiling"),
        },
        student_frames=observations,
        student_lo=float(inputs.student_force_lo),
        student_hi=float(inputs.student_force_hi),
        published_train=len(train),
        published_fp_candidates=len(fp_candidate_frames),
    )
    report.snapshot(
        _screen_output_counts(
            inputs.system_ids,
            observations=observations,
            train=train,
            fp_candidates=fp_candidate_frames,
        )
    )
    report.summary(
        ScreenBlockSummary(
            observations=catalog_count,
            train=len(train),
            fp_candidates=len(fp_candidate_frames),
        )
    )
    _cleanup_screen_work(work)
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=len(train),
        fp_candidate_count=len(fp_candidate_frames),
    )


def _single_teacher_committee_student_thresholds(
    inputs: ScreenRoundInputs,
    *,
    route: RouteSpec | None = None,
) -> dict[str, object]:
    if (
        inputs.teacher_p is None
        or inputs.fp_pool is None
        or inputs.student_force_lo is None
        or inputs.student_force_hi is None
    ):
        raise ValueError("mixed-route threshold inputs are incomplete")
    from workflow_policies.identities import selector_identity

    thresholds: dict[str, object] = {
        "teacher_p": dict(inputs.teacher_p),
        "student_force_deviation": {
            "lo": float(inputs.student_force_lo),
            "hi": float(inputs.student_force_hi),
        },
        "quota": {
            "fp_pool": inputs.fp_pool,
            "pool": inputs.pool,
            "keep": inputs.keep,
        },
        "fill_underfilled_from_safe_region": inputs.fill_underfilled_from_safe_region,
        "selector": selector_identity(inputs.screen_policy).as_dict(),
    }
    if route is not None:
        if route.screen_routing_policy_id is None:
            raise ValueError("single-teacher mixed route is missing its routing identity")
        thresholds = {
            "routing_policy": route.screen_routing_policy_id.as_dict(),
            **thresholds,
        }
    return thresholds


def _execute_joint_committee_teacher_single_student(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
) -> ScreenExecutionResult:
    del normalized_observations
    inputs = request.inputs
    if inputs.teacher_force_lo is None or inputs.teacher_force_hi is None:
        raise ValueError(
            "committee-teacher/single-student JOINT_DIRECT screen requires teacher force bounds"
        )
    report = request.report
    work = request.stage / ".screen_work"
    proposal, readout, outcomes = _joint_student_proposal(
        request, core=core, observations=observations, work=work, report=report
    )
    proposal_frames = _replay_with_metadata(
        request.source, proposal, chunk_size=request.observation_chunk_size
    )
    report.phase("teacher evaluation")
    teacher_evaluated = (
        request.services.label_teacher(
            frames=proposal_frames,
            work_dir=work / "teacher_label",
            report=report,
            trusted_below=None,
        )
        if proposal_frames
        else []
    )
    teacher_low, fp_candidates = core.route_teacher_force(
        teacher_evaluated,
        lower=inputs.teacher_force_lo,
        upper=inputs.teacher_force_hi,
    )
    selector_outcomes: dict[str, dict[str, object]] = {}
    if inputs.screen_policy == "ELEMENT_DIRECT":
        if readout.candidate_element_vectors is None:
            raise ValueError(
                "ELEMENT_DIRECT requires complete per-element candidate evidence"
            )
        train, selector_outcomes = core.select_element_per_system(
            teacher_low,
            keep=inputs.keep,
            element_vectors=readout.candidate_element_vectors,
            type_map=request.student_type_map,
        )
    else:
        train = core.select_per_system(
            teacher_low,
            pool=-1,
            keep=inputs.keep,
            policy=inputs.screen_policy,
            entropy=f"{inputs.entropy}/teacher-low-selector",
            feature_vectors=(
                readout.candidate_vectors
                if inputs.screen_policy in {"DIRECT", "FPS"}
                else None
            ),
        )
    train = core.restore_order(observations, train)
    fp_candidates = core.restore_order(observations, fp_candidates)
    for frame in train:
        frame.info["acquisition_route"] = "student_joint_direct_teacher_committee_low"
    for frame in fp_candidates:
        frame.info["acquisition_route"] = "student_joint_direct_teacher_committee_middle"
    report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", train, allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", fp_candidates, allow_empty=True
    )
    if inputs.screen_policy in {"DIRECT", "FPS"} and train:
        _write_frame_means(
            request.artifacts,
            request.stage / "student_representative_features.npz",
            train,
            readout.candidate_vectors,
            descriptor_schema=readout.descriptor_schema,
        )
    train_ids = {str(frame.info["structure_id"]) for frame in train}
    fp_ids = {str(frame.info["structure_id"]) for frame in fp_candidates}
    route_thresholds = {
        "prescreen": selector_identity(inputs.prescreen).as_dict(),
        "teacher_force_deviation": {
            "lo": inputs.teacher_force_lo,
            "hi": inputs.teacher_force_hi,
        },
        "quota": {"pool": inputs.pool, "keep": inputs.keep},
        "fill_underfilled_from_safe_region": False,
        "selector": selector_identity(inputs.screen_policy).as_dict(),
    }
    teacher_metrics = {
        str(frame.info["structure_id"]): float(frame.info["force_uncertainty"])
        for frame in teacher_evaluated
    }
    if inputs.prescreen == "ELEMENT_JOINT_DIRECT":
        if request.student_train_path is None:
            raise ValueError(
                "ELEMENT_JOINT_DIRECT publication requires student training membership"
            )
        readout_performed = readout.candidate_element_vectors is not None
        if readout_performed != (readout.background_element_vectors is not None):
            raise ValueError(
                "ELEMENT_JOINT_DIRECT publication has incomplete partition evidence"
            )
        if not readout_performed and any(
            not bool(outcome["bypass"]) for outcome in outcomes.values()
        ):
            raise ValueError(
                "ELEMENT_JOINT_DIRECT omitted required non-bypass descriptor evidence"
            )
        request.artifacts.publish_element_direct_screen_evidence(
            request.stage / "element_direct_screen.npz",
            observations=_joint_candidate_frames(request, readout, observations),
            background_frames=_joint_background_frames(request, readout),
            candidate_element_vectors=readout.candidate_element_vectors or {},
            background_element_vectors=readout.background_element_vectors or {},
            descriptor_schema=readout.descriptor_schema,
            type_map=request.student_type_map,
            model_identity=readout.model_identity,
            dataset_identity=readout.dataset_identity,
            prescreen=inputs.prescreen,
            pool=inputs.pool,
            keep=inputs.keep,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
            route_kind=request.route.kind.value,
            proposal_ids=[str(frame.info["structure_id"]) for frame in proposal],
            selector_candidate_ids={
                str(frame.info["structure_id"]) for frame in teacher_low
            },
            normal_train_ids=train_ids,
            train_ids=train_ids,
            fp_ids=fp_ids,
            teacher_metrics=teacher_metrics,
            outcomes=outcomes,
            selector_outcomes=selector_outcomes,
            route_thresholds=route_thresholds,
            readout_performed=readout_performed,
        )
    else:
        request.artifacts.publish_joint_direct_screen_evidence(
            request.stage / "joint_direct_screen.npz",
            observations=observations,
            candidate_vectors=readout.candidate_vectors,
            background_vectors=readout.background_vectors,
            descriptor_schema=readout.descriptor_schema,
            model_identity=readout.model_identity,
            dataset_identity=readout.dataset_identity,
            pool=inputs.pool,
            keep=inputs.keep,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
            route_kind=request.route.kind.value,
            proposal_ids=[str(frame.info["structure_id"]) for frame in proposal],
            normal_train_ids=train_ids,
            train_ids=train_ids,
            fp_ids=fp_ids,
            teacher_metrics=teacher_metrics,
            outcomes=outcomes,
            route_thresholds=route_thresholds,
        )
    _publish_joint_committee_teacher_round_screen_report(
        request,
        teacher_frames=teacher_evaluated,
        observations=observations,
        outcomes=outcomes,
        published_train=len(train),
        published_fp_candidates=len(fp_candidates),
    )
    report.snapshot(
        _screen_output_counts(
            inputs.system_ids,
            observations=observations,
            train=train,
            fp_candidates=fp_candidates,
        )
    )
    report.summary(
        ScreenBlockSummary(
            observations=catalog_count,
            train=len(train),
            fp_candidates=len(fp_candidates),
        )
    )
    _cleanup_screen_work(work)
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=len(train),
        fp_candidate_count=len(fp_candidates),
        proposal_ids=tuple(str(frame.info["structure_id"]) for frame in proposal),
        visualization_readout=readout,
    )


def _execute_committee_teacher_single_student(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
) -> ScreenExecutionResult:
    if request.inputs.prescreen in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
        return _execute_joint_committee_teacher_single_student(
            request,
            core=core,
            catalog_count=catalog_count,
            normalized_observations=normalized_observations,
            observations=observations,
        )
    inputs = request.inputs
    if (
        inputs.student_p is None
        or inputs.teacher_force_lo is None
        or inputs.teacher_force_hi is None
    ):
        raise ValueError(
            "committee-teacher/single-student screen requires student p and teacher force-deviation inputs"
        )
    if request.services.score_student is None:
        raise ValueError(
            "committee-teacher/single-student screen requires student GMM scoring"
        )
    report = request.report
    work = request.stage / ".screen_work"
    report.phase("student scoring")
    student_score = request.services.score_student(
        frames=observations,
        router_dir=request.stage / "gmm_router",
        work_dir=work / "student_gmm",
        emit_features=inputs.screen_policy in {"DIRECT", "FPS", "ELEMENT_DIRECT"},
        report=report,
    )
    annotated: list[Atoms] = []
    for frame in observations:
        item = frame.copy()
        identity = str(item.info["structure_id"])
        item.info["student_p"] = float(student_score.values[identity])
        item.info["acquisition_route"] = (
            "student_gmm_teacher_committee_proposal"
        )
        annotated.append(item)
    proposal_pool, student_flags = core.propose_committee_teacher_single_student(
        annotated,
        pool=inputs.pool,
        floor=float(inputs.student_p["floor"]),
        ceiling=inputs.student_p.get("ceiling"),
        entropy=f"{inputs.entropy}/student-p",
    )
    report.phase("student pool")
    report.snapshot(
        _gmm_scorecard(
            role="student",
            system_ids=inputs.system_ids,
            frames=annotated,
            flags=student_flags,
            probability_key="student_p",
            floor=float(inputs.student_p["floor"]),
            ceiling=inputs.student_p.get("ceiling"),
        )
    )
    proposal_frames = _replay_with_metadata(
        request.source,
        proposal_pool,
        chunk_size=request.observation_chunk_size,
    )
    report.phase("teacher evaluation")
    normal_teacher_evaluated = (
        request.services.label_teacher(
            frames=proposal_frames,
            work_dir=work / "teacher_label",
            report=report,
            trusted_below=None,
        )
        if proposal_frames
        else []
    )
    teacher_low, fp_candidates = core.route_teacher_force(
        normal_teacher_evaluated,
        lower=inputs.teacher_force_lo,
        upper=inputs.teacher_force_hi,
    )
    element_readout: JointVectorReadout | None = None
    selector_outcomes: dict[str, dict[str, object]] = {}
    if inputs.screen_policy == "ELEMENT_DIRECT":
        if (
            request.services.read_student_vectors is None
            or request.student_train_path is None
        ):
            raise ValueError(
                "ELEMENT_DIRECT screen requires student structural-vector readout"
            )
        report.phase("student structural readout")
        described_frames = _replay_with_metadata(
            request.source,
            observations,
            chunk_size=request.observation_chunk_size,
        )
        element_readout = request.services.read_student_vectors(
            frames=described_frames,
            work_dir=work / "student_element_direct",
            report=report,
            materialized=True,
            **({"reused_score": student_score} if getattr(student_score, "element_frame_means", None) is not None else {}),
        )
        if element_readout.candidate_element_vectors is None:
            raise ValueError(
                "ELEMENT_DIRECT requires complete per-element candidate evidence"
            )
        normal_train, selector_outcomes = core.select_element_per_system(
            teacher_low,
            keep=inputs.keep,
            element_vectors=element_readout.candidate_element_vectors,
            type_map=request.student_type_map,
        )
    else:
        normal_train = core.select_per_system(
            teacher_low,
            pool=-1,
            keep=inputs.keep,
            policy=inputs.screen_policy,
            entropy=f"{inputs.entropy}/teacher-low-selector",
            feature_vectors=student_score.frame_means,
        )
    normal_train = core.restore_order(annotated, normal_train)
    fp_candidates = core.restore_order(annotated, fp_candidates)
    safe_attempts = core.fill_committee_teacher_single_student(
        annotated,
        normal_selected=normal_train,
        flags=student_flags,
        keep=inputs.keep,
        ceiling=inputs.student_p.get("ceiling"),
        enabled=inputs.fill_underfilled_from_safe_region,
        entropy=f"{inputs.entropy}/committee-teacher-single-student",
    )
    safe_frames = _replay_with_metadata(
        request.source,
        safe_attempts,
        chunk_size=request.observation_chunk_size,
    )
    safe_teacher_evaluated = (
        request.services.label_teacher(
            frames=safe_frames,
            work_dir=work / "teacher_safe_fill_label",
            report=report,
            trusted_below=None,
        )
        if safe_frames
        else []
    )
    safe_accepted, _safe_fp = core.route_teacher_force(
        safe_teacher_evaluated,
        lower=inputs.teacher_force_lo,
        upper=inputs.teacher_force_hi,
    )
    train = core.restore_order(annotated, [*normal_train, *safe_accepted])
    teacher_evaluated = [*normal_teacher_evaluated, *safe_teacher_evaluated]
    for frame in train:
        frame.info["acquisition_route"] = "teacher_committee_low_student_gmm"
    for frame in fp_candidates:
        frame.info["acquisition_route"] = (
            "teacher_committee_middle_fp_candidate"
        )
    report.phase("routing")
    train_ids = {str(frame.info["structure_id"]) for frame in train}
    normal_train_ids = {str(frame.info["structure_id"]) for frame in normal_train}
    fp_ids = {str(frame.info["structure_id"]) for frame in fp_candidates}
    thresholds = _committee_teacher_single_student_thresholds(inputs)
    report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", train, allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", fp_candidates, allow_empty=True
    )
    if inputs.screen_policy in {"DIRECT", "FPS"} and train:
        if (
            student_score.frame_means is None
            or student_score.descriptor_schema is None
        ):
            raise ValueError(
                f"{inputs.screen_policy} mixed-route train selection requires student readout-input frame means"
            )
        _write_frame_means(
            request.artifacts,
            request.stage / "student_representative_features.npz",
            train,
            student_score.frame_means,
            descriptor_schema=student_score.descriptor_schema,
        )
    request.artifacts.publish_committee_teacher_single_student_audit(
        request.stage / "mixed_route_scores.npz",
        observations=annotated,
        student_flags=student_flags,
        teacher_evaluated=teacher_evaluated,
        normal_teacher_evaluated=normal_teacher_evaluated,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        teacher_lo=inputs.teacher_force_lo,
        teacher_hi=inputs.teacher_force_hi,
        thresholds=thresholds,
        safe_fill_enabled=inputs.fill_underfilled_from_safe_region,
    )
    if inputs.screen_policy == "ELEMENT_DIRECT":
        if (
            element_readout is None
            or element_readout.candidate_element_vectors is None
            or element_readout.background_element_vectors is None
            or request.student_train_path is None
        ):
            raise ValueError(
                "ELEMENT_DIRECT publication requires complete per-element evidence"
            )
        request.artifacts.publish_element_direct_screen_evidence(
            request.stage / "element_direct_screen.npz",
            observations=described_frames,
            background_frames=_joint_background_frames(request, element_readout),
            candidate_element_vectors=element_readout.candidate_element_vectors,
            background_element_vectors=element_readout.background_element_vectors,
            descriptor_schema=element_readout.descriptor_schema,
            type_map=request.student_type_map,
            model_identity=element_readout.model_identity,
            dataset_identity=element_readout.dataset_identity,
            prescreen=inputs.prescreen,
            pool=inputs.pool,
            keep=inputs.keep,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
            route_kind=request.route.kind.value,
            proposal_ids=[
                str(frame.info["structure_id"]) for frame in proposal_pool
            ],
            selector_candidate_ids={
                str(frame.info["structure_id"]) for frame in teacher_low
            },
            normal_train_ids=normal_train_ids,
            train_ids=train_ids,
            fp_ids=fp_ids,
            teacher_metrics={
                str(frame.info["structure_id"]): float(
                    frame.info["force_uncertainty"]
                )
                for frame in normal_teacher_evaluated
            },
            outcomes={},
            selector_outcomes=selector_outcomes,
            route_thresholds=thresholds,
        )
    summary = request.artifacts.committee_teacher_single_student_summary(
        observations=annotated,
        catalog_observations=normalized_observations,
        student_flags=student_flags,
        teacher_evaluated=teacher_evaluated,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        keep=inputs.keep,
        teacher_lo=inputs.teacher_force_lo,
        teacher_hi=inputs.teacher_force_hi,
        thresholds=thresholds,
        fill_underfilled_from_safe_region=inputs.fill_underfilled_from_safe_region,
    )
    request.artifacts.write_json(request.stage / "screen_summary.json", summary)
    report.snapshot(
        _screen_output_counts(
            inputs.system_ids,
            observations=annotated,
            train=train,
            fp_candidates=fp_candidates,
        )
    )
    _publish_committee_gmm_round_screen_report(
        request,
        teacher_frames=teacher_evaluated,
        teacher_lo=float(inputs.teacher_force_lo),
        teacher_hi=float(inputs.teacher_force_hi),
        student_frames=annotated,
        student_flags=_student_gmm_report_flags(
            annotated,
            student_flags,
            ceiling=inputs.student_p.get("ceiling"),
        ),
        student_thresholds={
            "floor": float(inputs.student_p["floor"]),
            "ceiling": inputs.student_p.get("ceiling"),
        },
        published_train=len(train),
        published_fp_candidates=len(fp_candidates),
    )
    report.summary(
        ScreenBlockSummary(
            observations=catalog_count,
            train=len(train),
            fp_candidates=len(fp_candidates),
        )
    )
    _cleanup_screen_work(work)
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=len(train),
        fp_candidate_count=len(fp_candidates),
        proposal_ids=tuple(
            str(frame.info["structure_id"]) for frame in proposal_pool
        ),
        visualization_readout=element_readout,
    )


def _committee_teacher_single_student_thresholds(
    inputs: ScreenRoundInputs,
) -> dict[str, object]:
    if (
        inputs.student_p is None
        or inputs.teacher_force_lo is None
        or inputs.teacher_force_hi is None
    ):
        raise ValueError("mixed-route threshold inputs are incomplete")
    from workflow_policies.identities import selector_identity

    return {
        "student_p": dict(inputs.student_p),
        "teacher_force_deviation": {
            "lo": float(inputs.teacher_force_lo),
            "hi": float(inputs.teacher_force_hi),
        },
        "quota": {"pool": inputs.pool, "keep": inputs.keep},
        "fill_underfilled_from_safe_region": inputs.fill_underfilled_from_safe_region,
        "selector": selector_identity(inputs.screen_policy).as_dict(),
    }


def _execute_committee_committee(
    request: ScreenExecutionRequest,
    *,
    core: ScreenRoundUseCase,
    catalog_count: int,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
) -> ScreenExecutionResult:
    inputs = request.inputs
    if (
        inputs.student_force_lo is None
        or inputs.student_force_hi is None
        or inputs.teacher_force_lo is None
        or inputs.teacher_force_hi is None
    ):
        raise ValueError(
            "committee/committee screen requires student and teacher force-deviation bands"
        )
    from workflow_policies.resolution import resolve_element_direct_policy

    policy_resolution = resolve_element_direct_policy(
        inputs.screen_policy, descriptor_model_shape="committee"
    )
    acquisition, selected = core.select_student_force_band(
        observations,
        lower=inputs.student_force_lo,
        upper=inputs.student_force_hi,
        pool=inputs.pool,
        keep=inputs.keep,
        policy=policy_resolution.effective_policy,
        entropy=inputs.entropy,
    )
    request.report.snapshot(
        _committee_routing_counts(
            inputs.system_ids, acquisition=acquisition, selected=selected
        )
    )
    selected_frames = _replay_with_metadata(
        request.source,
        selected,
        chunk_size=request.observation_chunk_size,
    )
    labeled = request.services.label_teacher(
        frames=selected_frames,
        work_dir=request.stage / ".screen_work" / "teacher_label",
        report=request.report,
        trusted_below=inputs.teacher_force_lo,
    )
    train, fp_candidates = core.route_teacher_force(
        labeled,
        lower=inputs.teacher_force_lo,
        upper=inputs.teacher_force_hi,
        missing_default=0.0,
    )
    normal_train = list(train)
    student_flags = {
        str(frame.info["structure_id"]): {
            "student_in_band": (
                inputs.student_force_lo
                <= float(frame.info["force_uncertainty"])
                < inputs.student_force_hi
            ),
            "student_pool": str(frame.info["structure_id"])
            in {str(item.info["structure_id"]) for item in selected},
        }
        for frame in observations
    }
    safe_attempts = core.fill_committee_student(
        observations,
        normal_selected=normal_train,
        flags=student_flags,
        keep=inputs.keep,
        lo=inputs.student_force_lo,
        enabled=inputs.fill_underfilled_from_safe_region,
        entropy=f"{inputs.entropy}/committee-committee",
    )
    safe_frames = _replay_with_metadata(
        request.source,
        safe_attempts,
        chunk_size=request.observation_chunk_size,
    )
    safe_labeled = (
        request.services.label_teacher(
            frames=safe_frames,
            work_dir=request.stage / ".screen_work" / "teacher_safe_fill_label",
            report=request.report,
            trusted_below=inputs.teacher_force_lo,
        )
        if safe_frames
        else []
    )
    safe_accepted, _safe_fp = core.route_teacher_force(
        safe_labeled,
        lower=inputs.teacher_force_lo,
        upper=inputs.teacher_force_hi,
        missing_default=0.0,
    )
    train = core.restore_order(observations, [*normal_train, *safe_accepted])
    fp_candidates = core.restore_order(observations, fp_candidates)
    request.report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", train, allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", fp_candidates, allow_empty=True
    )
    train_ids = {str(frame.info["structure_id"]) for frame in train}
    normal_train_ids = {str(frame.info["structure_id"]) for frame in normal_train}
    fp_ids = {str(frame.info["structure_id"]) for frame in fp_candidates}
    thresholds = _committee_committee_thresholds(inputs)
    teacher_evaluated = [*labeled, *safe_labeled]
    request.artifacts.publish_committee_force_audit(
        request.stage / "committee_force_scores.npz",
        observations=observations,
        student_flags=student_flags,
        teacher_evaluated=teacher_evaluated,
        normal_teacher_evaluated=labeled,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        teacher_lo=inputs.teacher_force_lo,
        teacher_hi=inputs.teacher_force_hi,
        thresholds=thresholds,
        safe_fill_enabled=inputs.fill_underfilled_from_safe_region,
    )
    summary = request.artifacts.committee_force_summary(
        observations=observations,
        catalog_observations=normalized_observations,
        student_flags=student_flags,
        teacher_evaluated=teacher_evaluated,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        keep=inputs.keep,
        teacher_lo=inputs.teacher_force_lo,
        teacher_hi=inputs.teacher_force_hi,
        thresholds=thresholds,
        safe_fill_enabled=inputs.fill_underfilled_from_safe_region,
    )
    if policy_resolution.fell_back:
        summary["selector_resolution"] = _publish_committee_fallback_resolution(
            request,
            candidates=_selector_candidate_population(
                acquisition, pool=inputs.pool
            ),
            selected=selected,
            entropy=inputs.entropy,
        )
    request.artifacts.write_json(request.stage / "screen_summary.json", summary)
    request.report.snapshot(
        _screen_output_counts(
            inputs.system_ids,
            observations=observations,
            train=train,
            fp_candidates=fp_candidates,
        )
    )
    _publish_committee_committee_round_screen_report(
        request,
        teacher_frames=teacher_evaluated,
        teacher_lo=float(inputs.teacher_force_lo),
        teacher_hi=float(inputs.teacher_force_hi),
        student_frames=observations,
        student_lo=float(inputs.student_force_lo),
        student_hi=float(inputs.student_force_hi),
        published_train=len(train),
        published_fp_candidates=len(fp_candidates),
    )
    request.report.summary(
        ScreenBlockSummary(
            observations=catalog_count,
            train=len(train),
            fp_candidates=len(fp_candidates),
        )
    )
    _cleanup_screen_work(request.stage / ".screen_work")
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=len(train),
        fp_candidate_count=len(fp_candidates),
    )


def _committee_committee_thresholds(inputs: ScreenRoundInputs) -> dict[str, object]:
    if (
        inputs.student_force_lo is None
        or inputs.student_force_hi is None
        or inputs.teacher_force_lo is None
        or inputs.teacher_force_hi is None
    ):
        raise ValueError("committee/committee threshold inputs are incomplete")
    from workflow_policies.identities import selector_identity

    return {
        "student_force_deviation": {
            "lo": float(inputs.student_force_lo),
            "hi": float(inputs.student_force_hi),
        },
        "teacher_force_deviation": {
            "lo": float(inputs.teacher_force_lo),
            "hi": float(inputs.teacher_force_hi),
        },
        "quota": {"pool": inputs.pool, "keep": inputs.keep},
        "fill_underfilled_from_safe_region": inputs.fill_underfilled_from_safe_region,
        "selector": selector_identity(inputs.screen_policy).as_dict(),
    }


def _publish_all_reject(
    request: ScreenExecutionRequest,
    *,
    catalog_count: int,
    normalized_observations: list[Atoms],
    pair_guard_decisions: tuple[Any, ...],
) -> ScreenExecutionResult:
    request.report.phase("finalizing")
    train_path = request.artifacts.write_structures(
        request.stage / "train.xyz", [], allow_empty=True
    )
    fp_candidate_path = request.artifacts.write_structures(
        request.stage / "fp_candidate.xyz", [], allow_empty=True
    )
    inputs = request.inputs
    if inputs.prescreen in {'JOINT_DIRECT', 'ELEMENT_JOINT_DIRECT'}:
        if request.route.kind is RouteKind.SINGLE_TEACHER_SINGLE_STUDENT:
            _publish_joint_single_teacher_round_screen_report(request,
                teacher_frames=(), teacher_flags={}, teacher_thresholds=inputs.teacher_p,
                observations=(), outcomes={}, published_train=0, published_fp_candidates=0)
        else:
            _publish_joint_committee_teacher_round_screen_report(request,
                teacher_frames=(), observations=(), outcomes={}, published_train=0, published_fp_candidates=0)
    elif request.route.kind is RouteKind.SINGLE_TEACHER_SINGLE_STUDENT:
        if inputs.teacher_p is None or inputs.student_p is None:
            raise ValueError(
                "single/single screen requires teacher and student p bands"
            )
        _publish_gmm_screen_summary(
            artifacts=request.artifacts,
            stage=request.stage,
            normalized_observations=normalized_observations,
            observations=[],
            pair_guard_config=inputs.pair_guard,
            pair_guard_decisions=pair_guard_decisions,
                teacher_flags={},
                student_flags={},
                student_values={},
                train=[],
                normal_train_ids=set(),
                fp_candidates=[],
                teacher_p=inputs.teacher_p,
                student_p=inputs.student_p,
                pool=inputs.pool,
                keep=inputs.keep,
                fill_underfilled_from_safe_region=inputs.fill_underfilled_from_safe_region,
            teacher_router=None,
            student_router=None,
            operation_profiles=request.services.operation_profiles,
            screen_policy=inputs.screen_policy,
            fp_policy=inputs.fp_policy,
        )
        _publish_gmm_round_screen_report(
            request,
            teacher_frames=(),
            teacher_flags={},
            teacher_thresholds={
                "floor": float(inputs.teacher_p["floor"]),
                "trust": float(inputs.teacher_p["trust"]),
                "ceiling": inputs.teacher_p.get("ceiling"),
            },
            student_frames=(),
            student_flags={},
            student_thresholds={
                "floor": float(inputs.student_p["floor"]),
                "ceiling": inputs.student_p.get("ceiling"),
            },
            published_train=0,
            published_fp_candidates=0,
        )
    elif request.route.kind is RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT:
        if inputs.teacher_p is None or inputs.student_force_lo is None or inputs.student_force_hi is None:
            raise ValueError("mixed-route all-reject thresholds are incomplete")
        _publish_gmm_committee_round_screen_report(
            request,
            teacher_frames=(),
            teacher_flags={},
            teacher_thresholds={
                "floor": float(inputs.teacher_p["floor"]),
                "trust": float(inputs.teacher_p["trust"]),
                "ceiling": inputs.teacher_p.get("ceiling"),
            },
            student_frames=(),
            student_lo=float(inputs.student_force_lo),
            student_hi=float(inputs.student_force_hi),
            published_train=0,
            published_fp_candidates=0,
        )
    elif request.route.kind is RouteKind.COMMITTEE_TEACHER_SINGLE_STUDENT:
        if inputs.student_p is None or inputs.teacher_force_lo is None or inputs.teacher_force_hi is None:
            raise ValueError("committee-teacher all-reject thresholds are incomplete")
        _publish_committee_gmm_round_screen_report(
            request,
            teacher_frames=(),
            teacher_lo=float(inputs.teacher_force_lo),
            teacher_hi=float(inputs.teacher_force_hi),
            student_frames=(),
            student_flags={},
            student_thresholds={
                "floor": float(inputs.student_p["floor"]),
                "ceiling": inputs.student_p.get("ceiling"),
            },
            published_train=0,
            published_fp_candidates=0,
        )
    elif request.route.kind is RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT:
        if inputs.student_force_lo is None or inputs.student_force_hi is None or inputs.teacher_force_lo is None or inputs.teacher_force_hi is None:
            raise ValueError("committee/committee all-reject thresholds are incomplete")
        _publish_committee_committee_round_screen_report(
            request,
            teacher_frames=(),
            teacher_lo=float(inputs.teacher_force_lo),
            teacher_hi=float(inputs.teacher_force_hi),
            student_frames=(),
            student_lo=float(inputs.student_force_lo),
            student_hi=float(inputs.student_force_hi),
            published_train=0,
            published_fp_candidates=0,
        )
    request.report.snapshot(
        GenericWorkSnapshot(
            activity="downstream skipped",
            completed=0,
            total=0,
            detail="no candidates" if catalog_count == 0 else "pair guard passed 0",
        )
    )
    request.report.summary(
        ScreenBlockSummary(observations=catalog_count, train=0, fp_candidates=0)
    )
    _cleanup_screen_work(request.stage / ".screen_work")
    return ScreenExecutionResult(
        train_path=train_path,
        fp_candidate_path=fp_candidate_path,
        observation_count=catalog_count,
        train_count=0,
        fp_candidate_count=0,
    )


def _deduplicate_observation_catalog(
    core: ScreenRoundUseCase,
    records: Sequence[Any],
) -> tuple[Any, ...]:
    ordered = tuple(records)
    if not ordered:
        raise ValueError("lifecycle screen observation catalog is empty; rebuild rN/sample")
    unique = core.deduplicate(_metadata_observations(ordered))
    by_id: dict[str, Any] = {}
    for record in ordered:
        by_id.setdefault(str(record.structure_id), record)
    return tuple(by_id[str(frame.info["structure_id"])] for frame in unique)


def _metadata_observations(records: Sequence[Any]) -> list[Atoms]:
    frames: list[Atoms] = []
    for record in records:
        frame = Atoms()
        frame.info.update(
            system_id=record.system_id,
            structure_id=record.structure_id,
            structure_source="workflow",
            md_step=record.md_step,
        )
        if record.student_uncertainty is not None:
            frame.info.update(student_uncertainty=record.student_uncertainty,
                              force_uncertainty=record.student_uncertainty,
                              force_std_max=record.student_uncertainty)
        for key in ('phase', 'cycle', 'phase_step', 'mc_attempts', 'md_time_fs',
                    'sampling_role_id', 'selection_group_id', 'condition_group_id', 'task_id'):
            value = getattr(record, key, None)
            if value is not None:
                frame.info[key] = value
        frames.append(frame)
    return frames


def _validate_catalog_selection_provenance(
    records: Sequence[Any], inputs: ScreenRoundInputs
) -> None:
    plan = inputs.selection_plan
    if plan is not None and plan.group_ids != inputs.system_ids:
        raise ValueError("screen selection plan group order disagrees with inputs")
    if not inputs.require_explicit_provenance:
        return
    if plan is None:
        raise ValueError("explicit screen requires a resolved selection plan")
    role_to_group = dict(plan.role_to_group)
    for record in records:
        role_id = getattr(record, "sampling_role_id", None)
        group_id = getattr(record, "selection_group_id", None)
        condition_id = getattr(record, "condition_group_id", None)
        task_id = getattr(record, "task_id", None)
        if any(
            not isinstance(value, str) or not value
            for value in (role_id, group_id, condition_id, task_id)
        ):
            raise ValueError(
                "explicit screen observation lacks required role/group provenance"
            )
        if (
            role_id not in role_to_group
            or role_to_group[role_id] != group_id
            or str(record.system_id) != group_id
        ):
            raise ValueError(
                "explicit screen observation disagrees with the selection plan"
            )


def _publish_selection_group_outcomes(
    request: ScreenExecutionRequest, *, observations: Sequence[Atoms], empty_population: bool = False
) -> None:
    inputs = request.inputs
    if not inputs.require_explicit_provenance:
        return
    plan = inputs.selection_plan
    if plan is None:
        raise ValueError("explicit screen outcome requires a selection plan")
    systems: tuple[str, ...] = ()
    pool_mask = normal_mask = train_mask = fp_mask = np.zeros(0, dtype=np.bool_)
    for name in (
        "element_direct_screen.npz",
        "joint_direct_screen.npz",
        "gmm_scores.npz",
        "mixed_route_scores.npz",
        "committee_force_scores.npz",
    ):
        path = request.stage / name
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as audit:
            systems = tuple(str(item) for item in audit["system_ids"].tolist())
            count = len(systems)
            pool_key = (
                "student_p_pool"
                if "student_p_pool" in audit
                else "student_pool"
                if "student_pool" in audit
                else None
            )
            pool_mask = (
                np.asarray(audit[pool_key], dtype=np.bool_)
                if pool_key is not None
                else np.zeros(count, dtype=np.bool_)
            )
            normal_mask = np.asarray(
                audit["normal_train_selected"], dtype=np.bool_
            )
            train_mask = np.asarray(audit["train_selected"], dtype=np.bool_)
            fp_mask = np.asarray(audit["fp_candidate"], dtype=np.bool_)
        break
    train_frames = read_extxyz_structures(
        request.stage / "train.xyz", allow_empty=True
    )
    fp_frames = read_extxyz_structures(
        request.stage / "fp_candidate.xyz", allow_empty=True
    )
    train_counts = _frame_counts(train_frames)
    fp_counts = _frame_counts(fp_frames)
    joint_outcomes = None
    if not empty_population and inputs.prescreen in {"JOINT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
        with np.load(
            request.stage
            / (
                "element_direct_screen.npz"
                if inputs.prescreen == "ELEMENT_JOINT_DIRECT"
                else "joint_direct_screen.npz"
            ),
            allow_pickle=False,
        ) as audit:
            import json

            joint_outcomes = json.loads(str(audit["outcomes_json"].item()))
    rows = []
    for group_id in plan.group_ids:
        owned = np.asarray([item == group_id for item in systems], dtype=np.bool_)
        normal = int(np.count_nonzero(normal_mask & owned))
        final = int(train_counts.get(group_id, 0))
        fp_count = int(fp_counts.get(group_id, 0))
        if joint_outcomes is not None and group_id not in joint_outcomes:
            # A declared group can have no surviving observations (e.g. every
            # native task failed before its first saved frame). Do not hide a
            # missing outcome for a group that actually owns screened frames.
            if np.any(owned) or final or fp_count:
                raise ValueError(f'missing joint outcome for nonempty group {group_id!r}')
            joint_outcomes[group_id] = dict(candidate_count=0, background_count=0,
                bypass=True, selected_candidate_count=0, selected_background_count=0,
                realized_clusters=0, quota_completion_count=0, underfill=0)
        row = {
                "selection_group_id": group_id,
                "candidate_count": (
                    int(joint_outcomes[group_id]["candidate_count"])
                    if joint_outcomes is not None
                    else int(np.count_nonzero(pool_mask & owned))
                ),
                "proposal_ceiling": inputs.pool,
                "keep_ceiling": inputs.keep,
                "normal_selected_count": normal,
                "safe_fill_count": final - normal,
                "final_selected_count": final,
                "underfill": max(0, inputs.keep - final),
                "fp_proposal_ceiling": inputs.fp_pool,
                "fp_candidate_count": fp_count,
            }
        if joint_outcomes is not None:
            row.update(
                {
                    "background_count": int(
                        joint_outcomes[group_id]["background_count"]
                    ),
                    "bypass": bool(joint_outcomes[group_id]["bypass"]),
                    "joint_selected_candidate_count": int(
                        joint_outcomes[group_id]["selected_candidate_count"]
                    ),
                    "joint_selected_background_count": int(
                        joint_outcomes[group_id]["selected_background_count"]
                    ),
                    "joint_realized_clusters": int(
                        joint_outcomes[group_id]["realized_clusters"]
                    ),
                    "joint_quota_completion_count": int(
                        joint_outcomes[group_id]["quota_completion_count"]
                    ),
                    "joint_underfill": int(joint_outcomes[group_id]["underfill"]),
                    "underfill_reason": (
                        "insufficient-candidates"
                        if int(joint_outcomes[group_id]["underfill"])
                        else "none"
                    ),
                }
            )
        rows.append(row)
    payload = {
        "schema": "dice.lifecycle.selection-group-screen-outcomes",
        "schema_revision": 3 if joint_outcomes is not None else 1,
        "selection_groups": list(plan.group_ids),
        "role_to_group": [list(item) for item in plan.role_to_group],
        "screen_policy": inputs.screen_policy,
        "rows": rows,
    }
    _validate_selection_group_outcomes_payload(payload)
    request.artifacts.write_json(
        request.stage / "selection_group_outcomes.json", payload
    )


def _validate_selection_group_outcomes_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "schema_revision",
        "selection_groups",
        "role_to_group",
        "screen_policy",
        "rows",
    }
    if set(payload) != expected or (
        payload["schema"] != "dice.lifecycle.selection-group-screen-outcomes"
        or payload["schema_revision"] not in {1, 2, 3}
    ):
        raise ValueError("selection-group screen outcome shape is invalid")
    group_ids = tuple(payload["selection_groups"])
    rows = tuple(payload["rows"])
    if (
        not group_ids
        or len(set(group_ids)) != len(group_ids)
        or len(rows) != len(group_ids)
        or tuple(row.get("selection_group_id") for row in rows) != group_ids
    ):
        raise ValueError("selection-group screen outcome order is invalid")
    row_fields = {
        "selection_group_id",
        "candidate_count",
        "proposal_ceiling",
        "keep_ceiling",
        "normal_selected_count",
        "safe_fill_count",
        "final_selected_count",
        "underfill",
        "fp_proposal_ceiling",
        "fp_candidate_count",
    }
    if payload["schema_revision"] == 2:
        row_fields.update(
            {
                "background_count",
                "bypass",
                "joint_selected_candidate_count",
                "joint_selected_background_count",
                "joint_realized_clusters",
                "joint_underfill",
                "underfill_reason",
            }
        )
    elif payload["schema_revision"] == 3:
        row_fields.update(
            {
                "background_count",
                "bypass",
                "joint_selected_candidate_count",
                "joint_selected_background_count",
                "joint_realized_clusters",
                "joint_quota_completion_count",
                "joint_underfill",
                "underfill_reason",
            }
        )
    for row in rows:
        if set(row) != row_fields:
            raise ValueError("selection-group screen outcome row shape is invalid")
        for key in (
            "candidate_count",
            "normal_selected_count",
            "safe_fill_count",
            "final_selected_count",
            "underfill",
            "fp_candidate_count",
        ):
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError("selection-group screen outcome count is invalid")
        keep = row["keep_ceiling"]
        if (
            not isinstance(keep, int)
            or isinstance(keep, bool)
            or keep <= 0
            or row["final_selected_count"]
            != row["normal_selected_count"] + row["safe_fill_count"]
            or row["final_selected_count"] > keep
            or row["underfill"] != keep - row["final_selected_count"]
        ):
            raise ValueError("selection-group screen outcome quota is invalid")
        completion_count = row.get("joint_quota_completion_count", 0)
        invalid_completion = payload["schema_revision"] == 3 and (
            not isinstance(completion_count, int)
            or isinstance(completion_count, bool)
            or completion_count < 0
            or completion_count > row["joint_selected_candidate_count"]
        )
        if payload["schema_revision"] in {2, 3} and (
            not isinstance(row["background_count"], int)
            or row["background_count"] < 0
            or not isinstance(row["bypass"], bool)
            or not isinstance(row["joint_selected_candidate_count"], int)
            or row["joint_selected_candidate_count"] < 0
            or not isinstance(row["joint_selected_background_count"], int)
            or row["joint_selected_background_count"] < 0
            or not isinstance(row["joint_realized_clusters"], int)
            or row["joint_realized_clusters"] < 0
            or row["joint_realized_clusters"]
            != row["joint_selected_candidate_count"]
            - completion_count
            + row["joint_selected_background_count"]
            or invalid_completion
            or not isinstance(row["joint_underfill"], int)
            or row["joint_underfill"] < 0
            or row["joint_selected_candidate_count"]
            > min(row["proposal_ceiling"], row["candidate_count"])
            or row["joint_underfill"]
            != min(row["proposal_ceiling"], row["candidate_count"])
            - row["joint_selected_candidate_count"]
            or row["underfill_reason"]
            not in {
                "none",
                "background-representatives-filtered",
                "realized-clusters",
                "insufficient-candidates",
            }
        ):
            raise ValueError("joint selection-group screen outcome is invalid")


def _apply_pair_guard_source(
    core: ScreenRoundUseCase,
    source: Any,
    *,
    artifacts: Any,
    screen_dir: Path,
    config: Mapping[str, Any] | None,
    records: tuple[Any, ...],
    chunk_size: int,
    ordered_map=None,
) -> tuple[list[Atoms], tuple[Any, ...] | None]:
    metadata = _metadata_observations(records)
    if config is None:
        return metadata, None
    thresholds = PairGuardThresholds(
        minimum=float(config["min"]),
        pairs=dict(config.get("pairs", {})),
    )
    decisions: list[Any] = []
    identities = tuple(str(record.structure_id) for record in records)
    size = _validated_observation_chunk_size(chunk_size)
    chunks = iter(source.replay(identities, chunk_size=size))
    first = tuple(next(chunks, ()))
    # Avoid process startup for a small guard even when it spans many tiny
    # frames. Estimate work from the first replay chunk without a second read.
    atom_work = len(identities) * sum(len(frame) for frame in first) / max(1, len(first))
    chunks = chain((first,), chunks) if first else chunks
    if ordered_map is not None and len(identities) >= 128 and atom_work >= 65_536:
        from workflow_policies.screen_tasks import pair_guard_task
        tasks = ((tuple(chunk), thresholds.minimum, dict(thresholds.pairs)) for chunk in chunks)
        for evaluated in ordered_map(pair_guard_task, tasks):
            decisions.extend(evaluated)
    else:
        for chunk in chunks:
            _passed, evaluated = core.pair_guard(chunk, thresholds)
            decisions.extend(evaluated)
    actual = tuple(str(decision.structure_id) for decision in decisions)
    if actual != identities:
        raise ValueError("lifecycle pair guard replay identities do not match catalog order")
    artifacts.publish_pair_guard(
        screen_dir / "pair_guard.npz", decisions, thresholds=thresholds
    )
    passed = {
        str(decision.structure_id) for decision in decisions if decision.passed
    }
    return (
        [
            frame
            for frame in metadata
            if str(frame.info["structure_id"]) in passed
        ],
        tuple(decisions),
    )


def _validated_observation_chunk_size(value: int) -> int:
    size = int(value)
    if size <= 0:
        raise ValueError("lifecycle observation replay chunk size must be positive")
    return size


def _replay_with_metadata(
    source: Any,
    frames: Sequence[Atoms],
    *,
    chunk_size: int,
) -> list[Atoms]:
    return list(_iter_replay_with_metadata(source, frames, chunk_size=chunk_size))


def _iter_replay_with_metadata(
    source: Any, frames: Sequence[Atoms], *, chunk_size: int,
) -> Iterator[Atoms]:
    identities = tuple(str(frame.info["structure_id"]) for frame in frames)
    metadata = {
        str(frame.info["structure_id"]): dict(frame.info) for frame in frames
    }
    actual: list[str] = []
    for chunk in source.replay(
        identities, chunk_size=_validated_observation_chunk_size(chunk_size)
    ):
        for frame in chunk:
            identity = str(frame.info.get("structure_id", ""))
            if identity not in metadata:
                raise ValueError("lifecycle observation replay returned an unexpected identity")
            frame.info.update(metadata[identity])
            actual.append(identity)
            yield frame
    if tuple(actual) != identities:
        raise ValueError("lifecycle observation replay identities do not match requested order")


def _publish_gmm_representative_sidecars(
    *,
    artifacts: Any,
    stage: Path,
    fp_candidates: list[Atoms],
    selected_train: list[Atoms],
    teacher_means: Mapping[str, np.ndarray] | None,
    student_means: Mapping[str, np.ndarray] | None,
    teacher_descriptor_schema: Any,
    student_descriptor_schema: Any,
    fp_policy: str,
    screen_policy: str,
) -> None:
    if fp_policy in {"DIRECT", "FPS"} and fp_candidates:
        if teacher_means is None:
            raise ValueError(
                f"{fp_policy} FP selection requires teacher readout-input frame means"
            )
        if teacher_descriptor_schema is None:
            raise ValueError(
                "teacher representative evidence requires a readout-input schema"
            )
        _write_frame_means(
            artifacts,
            stage / "representative_features.npz",
            fp_candidates,
            teacher_means,
            descriptor_schema=teacher_descriptor_schema,
        )
    if screen_policy in {"DIRECT", "FPS"} and selected_train:
        if student_means is None:
            raise ValueError(
                f"{screen_policy} student selection requires student readout-input frame means"
            )
        if student_descriptor_schema is None:
            raise ValueError(
                "student representative evidence requires a readout-input schema"
            )
        _write_frame_means(
            artifacts,
            stage / "student_representative_features.npz",
            selected_train,
            student_means,
            descriptor_schema=student_descriptor_schema,
        )


def _write_frame_means(
    artifacts: Any,
    path: Path,
    frames: Sequence[Atoms],
    means: Mapping[str, np.ndarray],
    *,
    descriptor_schema: Any,
) -> None:
    identities = [str(frame.info["structure_id"]) for frame in frames]
    try:
        rows = np.stack(
            [np.asarray(means[identity], dtype=np.float32) for identity in identities]
        )
    except KeyError as exc:
        raise ValueError(
            f"representative frame mean is missing for {exc.args[0]!r}"
        ) from exc
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError(
            "representative frame means must be finite frame x feature rows"
        )
    artifacts.publish_frame_mean_representatives(
        path,
        identities=identities,
        means=rows,
        descriptor_schema=descriptor_schema,
    )


def _frame_counts(frames: Sequence[Atoms]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        system_id = str(frame.info.get("system_id", ""))
        counts[system_id] = counts.get(system_id, 0) + 1
    return counts


def _screen_output_counts(
    system_ids: tuple[str, ...],
    *,
    observations: list[Atoms],
    train: list[Atoms],
    fp_candidates: list[Atoms],
):
    observation_counts = _frame_counts(observations)
    train_counts = _frame_counts(train)
    fp_counts = _frame_counts(fp_candidates)
    rows = (
        SystemCountRow("ALL", (len(observations), len(train), len(fp_candidates))),
        *(
            SystemCountRow(
                system_id,
                (
                    observation_counts.get(system_id, 0),
                    train_counts.get(system_id, 0),
                    fp_counts.get(system_id, 0),
                ),
            )
            for system_id in system_ids
        ),
    )
    return CountFlowSnapshot(
        title="outputs", columns=("observations", "train", "FP"), rows=rows
    )


def _committee_routing_counts(
    system_ids: tuple[str, ...],
    *,
    acquisition: list[Atoms],
    selected: list[Atoms],
):
    acquisition_counts = _frame_counts(acquisition)
    selected_counts = _frame_counts(selected)
    return CountFlowSnapshot(
        title="committee routing",
        columns=("candidate", "selected"),
        rows=(
            SystemCountRow("ALL", (len(acquisition), len(selected))),
            *(
                SystemCountRow(
                    system_id,
                    (
                        acquisition_counts.get(system_id, 0),
                        selected_counts.get(system_id, 0),
                    ),
                )
                for system_id in system_ids
            ),
        ),
    )


def _gmm_routing_counts(
    system_ids: tuple[str, ...],
    *,
    observations: list[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    student_flags: Mapping[str, Mapping[str, bool]],
):
    by_system = {
        str(frame.info["structure_id"]): str(frame.info["system_id"])
        for frame in observations
    }

    def row(system_id: str | None) -> Any:
        identities = {
            identity
            for identity, owning_system in by_system.items()
            if system_id is None or owning_system == system_id
        }
        return RoutingSystemRow(
            "ALL" if system_id is None else system_id,
            (
                sum(
                    bool(
                        teacher_flags.get(identity, {}).get("teacher_candidate")
                    )
                    for identity in identities
                ),
                sum(
                    bool(
                        teacher_flags.get(identity, {}).get(
                            "teacher_fp_candidate"
                        )
                    )
                    for identity in identities
                ),
            ),
            (
                sum(
                    bool(
                        student_flags.get(identity, {}).get("student_candidate")
                    )
                    for identity in identities
                ),
                sum(
                    bool(student_flags.get(identity, {}).get("student_pool"))
                    for identity in identities
                ),
                sum(
                    bool(student_flags.get(identity, {}).get("train_selected"))
                    for identity in identities
                ),
            ),
        )

    return RoutingSnapshot(
        rows=(row(None), *(row(system_id) for system_id in system_ids))
    )


def _gmm_scorecard(
    *,
    role: str,
    system_ids: tuple[str, ...],
    frames: list[Atoms],
    flags: Mapping[str, Mapping[str, bool]],
    probability_key: str,
    floor: float,
    ceiling: float | None,
):
    if role == "teacher":
        configured = [("covered", "teacher_covered")]
        if ceiling is not None:
            configured.append(("candidate", "teacher_candidate"))
        if floor > 0:
            configured.append(("discard", "teacher_floor_discard"))
    elif role == "student":
        configured = []
        if ceiling is not None:
            configured.extend(
                [("covered", "student_covered"), ("candidate", "student_candidate")]
            )
        if floor > 0:
            configured.append(("discard", "student_floor_discard"))
    else:
        raise ValueError(f"unsupported GMM scorecard role {role!r}")

    def make_row(system_id: str | None) -> Any:
        owned = [
            frame
            for frame in frames
            if system_id is None or str(frame.info.get("system_id")) == system_id
        ]
        identities = [str(frame.info["structure_id"]) for frame in owned]
        values = tuple(
            sum(
                bool(flags.get(identity, {}).get(flag_key))
                for identity in identities
            )
            for _column, flag_key in configured
        )
        probabilities = [float(frame.info[probability_key]) for frame in owned]
        return ScorecardRow(
            "ALL" if system_id is None else system_id,
            values,
            None if not probabilities else sum(probabilities) / len(probabilities),
        )

    return ScorecardSnapshot(
        role=role,
        columns=tuple(column for column, _flag_key in configured),
        rows=(make_row(None), *(make_row(system_id) for system_id in system_ids)),
    )


def _selector_candidate_population(
    frames: Sequence[Atoms], *, pool: int
) -> tuple[Atoms, ...]:
    result: list[Atoms] = []
    for system_id in sorted({str(frame.info["system_id"]) for frame in frames}):
        candidates = [
            frame
            for frame in frames
            if str(frame.info["system_id"]) == system_id
        ]
        candidates.sort(key=lambda frame: str(frame.info["structure_id"]))
        candidates.sort(
            key=lambda frame: (
                -float(
                    frame.info.get(
                        "acquisition",
                        frame.info.get(
                            "force_uncertainty",
                            frame.info.get("teacher_p", frame.info.get("p", 0.0)),
                        ),
                    )
                ),
                str(frame.info["structure_id"]),
            )
        )
        result.extend(candidates if pool < 0 else candidates[:pool])
    return tuple(result)


def _publish_committee_fallback_resolution(
    request: ScreenExecutionRequest,
    *,
    candidates: Sequence[Atoms],
    selected: Sequence[Atoms],
    entropy: str,
) -> dict[str, str]:
    from workflow_artifacts.lifecycle.policy_resolution import (
        publish_policy_resolution,
    )
    from workflow_policies.resolution import COMMITTEE_DESCRIPTOR_UNAVAILABLE

    for group_id in request.inputs.system_ids:
        request.report.warning(
            RuntimeError(
                "screen.policy ELEMENT_DIRECT fell back to random for "
                f"{group_id}: {COMMITTEE_DESCRIPTOR_UNAVAILABLE}"
            )
        )
    publish_policy_resolution(
        request.stage / "policy_resolution.json",
        coordinate="screen.policy",
        requested_policy="ELEMENT_DIRECT",
        effective_policy="random",
        fallback_reason=COMMITTEE_DESCRIPTOR_UNAVAILABLE,
        group_ids=request.inputs.system_ids,
        candidates=candidates,
        selected=selected,
        quota=request.inputs.keep,
        entropy=entropy,
    )
    return {
        "requested_policy": "ELEMENT_DIRECT",
        "effective_policy": "random",
        "fallback_reason": COMMITTEE_DESCRIPTOR_UNAVAILABLE,
    }


def _ensure_committee_fallback_resolution(
    request: ScreenExecutionRequest,
) -> None:
    if (
        request.inputs.screen_policy != "ELEMENT_DIRECT"
        or request.route.kind
        not in {
            RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT,
            RouteKind.COMMITTEE_TEACHER_COMMITTEE_STUDENT,
        }
        or (request.stage / "policy_resolution.json").is_file()
    ):
        return
    entropy = (
        f"{request.inputs.entropy}/post-teacher"
        if request.route.kind is RouteKind.SINGLE_TEACHER_COMMITTEE_STUDENT
        else request.inputs.entropy
    )
    resolution = _publish_committee_fallback_resolution(
        request,
        candidates=(),
        selected=(),
        entropy=entropy,
    )
    summary_path = request.stage / "screen_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {
            "counts": {
                "observations": 0,
                "train": 0,
                "fp_candidate": 0,
            }
        }
    )
    summary["selector_resolution"] = resolution
    request.artifacts.write_json(summary_path, summary)


def _cleanup_screen_work(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)


def _student_probability_items(student_score: Any):
    compact = getattr(student_score, "compact_view", None)
    iterator = getattr(compact, "iter_probabilities", None)
    if callable(iterator):
        return iterator()
    return iter(student_score.values.items())


def _close_score_view(score: Any) -> None:
    close = getattr(score, "close", None)
    if callable(close):
        close()


def _publish_gmm_screen_summary(
    *,
    artifacts: Any,
    stage: Path,
    normalized_observations: list[Atoms],
    observations: list[Atoms],
    pair_guard_config: Mapping[str, Any] | None,
    pair_guard_decisions: tuple[Any, ...] | None,
    teacher_flags: Mapping[str, Mapping[str, bool]],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_values: Mapping[str, float],
    train: list[Atoms],
    normal_train_ids: set[str],
    fp_candidates: list[Atoms],
    teacher_p: Mapping[str, float],
    student_p: Mapping[str, float],
    pool: int,
    keep: int,
    fill_underfilled_from_safe_region: bool,
    teacher_router: Path | None,
    student_router: Path | None,
    operation_profiles: ScreenOperationProfileEvidence,
    screen_policy: str,
    fp_policy: str,
    normal_teacher_ids: set[str] | None = None,
) -> None:
    from workflow_policies.identities import (
        GMM_REFERENCE_TAIL,
        PAIR_GUARD,
        STUDENT_REFERENCE_TEACHER_REFERENCE,
        selector_identity,
    )

    summary_schema, summary_schema_revision = artifacts.gmm_summary_identity

    systems = sorted(
        {str(frame.info["system_id"]) for frame in normalized_observations}
    )
    scored_ids = {str(frame.info["structure_id"]) for frame in observations}
    guard_pass = {
        str(decision.structure_id): bool(decision.passed)
        for decision in pair_guard_decisions or ()
    }
    per_system: dict[str, dict[str, int]] = {}
    for system_id in systems:
        identities = [
            str(frame.info["structure_id"])
            for frame in normalized_observations
            if str(frame.info["system_id"]) == system_id
        ]
        passed = [identity for identity in identities if guard_pass.get(identity, True)]
        per_system[system_id] = {
            "observations": len(identities),
            "guard_pass": len(passed),
            "guard_reject": len(identities) - len(passed),
            "teacher_scored": sum(identity in scored_ids for identity in identities),
            "teacher_floor_discard": sum(
                teacher_flags.get(identity, {}).get("teacher_floor_discard", False)
                for identity in identities
            ),
            "teacher_fp_candidate": sum(
                teacher_flags.get(identity, {}).get("teacher_fp_candidate", False)
                for identity in identities
            ),
            "student_eligible": sum(
                teacher_flags.get(identity, {}).get("student_eligible", False)
                for identity in identities
            ),
            "teacher_trust_reject": sum(
                teacher_flags.get(identity, {}).get("teacher_trust_reject", False)
                for identity in identities
            ),
            "student_floor_discard": sum(
                student_flags.get(identity, {}).get("student_floor_discard", False)
                for identity in identities
            ),
            "student_pool": sum(
                student_flags.get(identity, {}).get("student_pool", False)
                for identity in identities
            ),
            "normal_train": sum(identity in normal_train_ids for identity in identities),
            "safe_fill_eligible": sum(
                student_flags.get(identity, {}).get("safe_fill_eligible", False)
                for identity in identities
            ),
            "safe_fill_attempted": sum(
                student_flags.get(identity, {}).get("safe_fill_selected", False)
                for identity in identities
            ),
            "safe_fill_accepted": sum(
                identity in {str(frame.info["structure_id"]) for frame in train}
                and student_flags.get(identity, {}).get("safe_fill_selected", False)
                for identity in identities
            ),
            "train": sum(
                str(frame.info["system_id"]) == system_id for frame in train
            ),
            "train_underfill": max(
                0,
                keep - sum(str(frame.info["system_id"]) == system_id for frame in train),
            ),
            "fp_candidate": sum(
                str(frame.info["system_id"]) == system_id
                for frame in fp_candidates
            ),
        }

    normal_teacher_ids = scored_ids if normal_teacher_ids is None else normal_teacher_ids
    summary = {
        "schema": summary_schema,
        "schema_revision": summary_schema_revision,
        "policy_identity": GMM_REFERENCE_TAIL.as_dict(),
        "directional_policy": STUDENT_REFERENCE_TEACHER_REFERENCE.as_dict(),
        "pair_guard_identity": None if pair_guard_config is None else PAIR_GUARD.as_dict(),
        "thresholds": {
            "teacher": {
                "floor": float(teacher_p["floor"]),
                "trust": float(teacher_p["trust"]),
                "ceiling": (
                    None
                    if "ceiling" not in teacher_p
                    else float(teacher_p["ceiling"])
                ),
            },
            "student": {
                "floor": float(student_p["floor"]),
                "ceiling": (
                    None
                    if "ceiling" not in student_p
                    else float(student_p["ceiling"])
                ),
            },
            "pair_guard": (
                None
                if pair_guard_config is None
                else {
                    "min": float(pair_guard_config["min"]),
                    "pairs": dict(pair_guard_config.get("pairs", {})),
                }
            ),
            "quota": {"pool": pool, "keep": keep},
        },
        "fill_underfilled_from_safe_region": fill_underfilled_from_safe_region,
        "selectors": {
            "screen": selector_identity(screen_policy).as_dict(),
            "fp": selector_identity(fp_policy).as_dict(),
        },
        "counts": {
            "observations": len(normalized_observations),
            "guard_pass": sum(
                guard_pass.get(str(frame.info["structure_id"]), True)
                for frame in normalized_observations
            ),
            "guard_reject": sum(
                not guard_pass.get(str(frame.info["structure_id"]), True)
                for frame in normalized_observations
            ),
            "teacher_scored": len(observations),
            "normal_teacher_scored": len(normal_teacher_ids),
            "teacher_floor_discard": sum(
                item["teacher_floor_discard"] for item in teacher_flags.values()
            ),
            "teacher_fp_candidate": len(fp_candidates),
            "student_eligible": sum(
                item["student_eligible"] for item in teacher_flags.values()
            ),
            "teacher_trust_reject": sum(
                item["teacher_trust_reject"] for item in teacher_flags.values()
            ),
            "student_floor_discard": sum(
                item["student_floor_discard"] for item in student_flags.values()
            ),
            "student_pool": sum(
                item["student_pool"] for item in student_flags.values()
            ),
            "normal_train": len(normal_train_ids),
            "safe_fill_eligible": sum(
                item.get("safe_fill_eligible", False) for item in student_flags.values()
            ),
            "safe_fill_attempted": sum(
                item.get("safe_fill_selected", False) for item in student_flags.values()
            ),
            "safe_fill_accepted": sum(
                str(frame.info["structure_id"]) not in normal_train_ids
                for frame in train
            ),
            "train": len(train),
            "train_underfill": sum(
                values["train_underfill"] for values in per_system.values()
            ),
            "fp_candidate": len(fp_candidates),
        },
        "per_system": per_system,
        "p_bins": {
            "teacher": _probability_bins(
                [float(frame.info["teacher_p"]) for frame in observations],
                thresholds=(
                    teacher_p["floor"],
                    teacher_p["trust"],
                    teacher_p.get("ceiling"),
                ),
            ),
            "student": _probability_bins(
                list(student_values.values()),
                thresholds=(student_p["floor"], student_p.get("ceiling")),
            ),
        },
        "provenance": {
            "teacher": artifacts.router_provenance(teacher_router),
            "student": artifacts.router_provenance(student_router),
            "operation_profiles": {
                "teacher_gmm_score": operation_profiles.teacher_gmm_score,
                "teacher_label": operation_profiles.teacher_label,
                "student_gmm_reference": operation_profiles.student_gmm_reference,
                "student_gmm_score": operation_profiles.student_gmm_score,
            },
        },
    }
    for system_id, counts in summary["per_system"].items():
        counts["normal_teacher_scored"] = sum(
            str(frame.info["system_id"]) == system_id
            and str(frame.info["structure_id"]) in normal_teacher_ids
            for frame in observations
        )
    artifacts.write_json(stage / "screen_summary.json", summary)


def _probability_bins(
    values: list[float], *, thresholds: tuple[float | None, ...]
) -> dict[str, int]:
    edges = sorted(
        {0.0, 1.0, *(float(value) for value in thresholds if value is not None)}
    )
    return {
        f"{left:g}-{right:g}": sum(
            left <= float(value) < right
            or (right == 1.0 and float(value) == 1.0)
            for value in values
        )
        for left, right in zip(edges, edges[1:])
    }


__all__ = [
    "ScreenExecutionRequest",
    "ScreenExecutionResult",
    "ScreenOperationProfileEvidence",
    "ScreenRoundInputs",
    "ScreenServices",
    "execute_screen",
]
