from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from workflow_artifacts.lifecycle.gmm_screen import (
    validate_gmm_pair_guard_alignment,
    validate_gmm_screen_scores,
)
from workflow_artifacts.lifecycle.committee_force import (
    load_committee_force_audit,
    validate_committee_force_summary,
)
from workflow_artifacts.lifecycle.mixed_route import (
    load_mixed_route_audit,
    validate_committee_teacher_single_student_summary,
)
from workflow_artifacts.lifecycle.pair_guard import PairGuardArtifact, load_pair_guard
from workflow_artifacts.lifecycle.representative_features import (
    load_frame_mean_representatives,
)
from workflow_core.structures.formats import (
    read_extxyz_structures,
)
from workflow_core.structures.canonical_identity import validate_canonical_identity
from workflow_artifacts.lifecycle.gmm_io import validate_gmm_router
from workflow_policies.mixed_routes import (
    COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID,
    SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID,
)
from workflow_policies.pair_guard import PairGuardThresholds
from workflow_policies.identities import selector_identity
from workflow_artifacts.lifecycle.screen_report import load_screen_report


@dataclass(frozen=True, slots=True)
class ScreenArtifactValidationRequest:
    """Expected canonical inputs for one completed ``rN/screen`` block."""

    contract: Any
    screen_dir: Path
    screen_policy: str
    fp_policy: str
    pair_guard: Mapping[str, Any] | None
    expected_system_ids: tuple[str, ...]
    expected_type_map: tuple[str, ...] = ()
    expected_thresholds: Mapping[str, object] | None = None
    teacher_router: Path | None = None
    selection_plan: Any | None = None
    require_selection_group_audit: bool = False
    prescreen: str = "GMM"
    joint_model_identity: str | None = None
    joint_dataset_identity: str | None = None
    expected_entropy: str | None = None
    ordered_map: Callable | None = None


def validate_screen_artifacts(
    request: ScreenArtifactValidationRequest,
) -> bool:
    """Return whether canonical screen evidence satisfies its route contract."""

    try:
        summary = json.loads((request.screen_dir / 'screen_summary.json').read_text())
    except (OSError, ValueError):
        summary = {}
    if summary.get('schema') == 'dice.lifecycle.empty-screen':
        return _valid_empty_screen(request, summary)
    contract = str(getattr(request.contract, "value", request.contract))
    if request.prescreen == "ELEMENT_JOINT_DIRECT":
        valid = _valid_element_direct_screen(request)
    elif request.prescreen == "JOINT_DIRECT":
        valid = _valid_joint_direct_screen(request)
    elif contract == "gmm-reference-tail":
        valid = _valid_gmm_screen(request)
    elif contract == "single-teacher/committee-student":
        valid = _valid_single_teacher_committee_student(request)
    elif contract == "committee-teacher/single-student":
        valid = _valid_committee_teacher_single_student(request)
    elif contract == "committee-force-deviation":
        valid = _valid_committee_screen(request)
    else:
        return False
    committee_fallback = request.screen_policy == "ELEMENT_DIRECT" and contract in {
        "single-teacher/committee-student", "committee-force-deviation",
    }
    if valid and committee_fallback:
        valid = _valid_committee_fallback_resolution(request, contract=contract)
    elif valid and request.screen_policy == "ELEMENT_DIRECT" and request.prescreen != "ELEMENT_JOINT_DIRECT":
        valid = _valid_element_direct_selector(request)
    if valid and not committee_fallback and (
        request.screen_dir / "policy_resolution.json"
    ).exists():
        valid = False
    return valid and (
        not request.require_selection_group_audit
        or _valid_selection_group_outcomes(request)
    )


def _valid_committee_fallback_resolution(
    request: ScreenArtifactValidationRequest, *, contract: str
) -> bool:
    if request.expected_entropy is None:
        return False
    try:
        from ase import Atoms
        from workflow_artifacts.lifecycle.policy_resolution import (
            validate_policy_resolution,
        )

        summary = json.loads(
            (request.screen_dir / "screen_summary.json").read_text(
                encoding="utf-8"
            )
        )
        expected_resolution = {
            "requested_policy": "ELEMENT_DIRECT",
            "effective_policy": "random",
            "fallback_reason": "committee_descriptor_unavailable",
        }
        if summary.get("selector_resolution") != expected_resolution:
            return False
        guard = _load_configured_pair_guard(
            request.screen_dir, request.pair_guard
        )
        if (guard is not None and _is_all_reject_guard(guard)) or summary.get('schema') == 'dice.lifecycle.empty-screen':
            train_frames, fp_frames = _canonical_outputs(
                request,
                contract=("gmm" if contract == "single-teacher/committee-student" else "force"),
            )
            if train_frames or fp_frames:
                return False
            entropy = (
                f"{request.expected_entropy}/post-teacher"
                if contract == "single-teacher/committee-student"
                else request.expected_entropy
            )
            return validate_policy_resolution(
                request.screen_dir / "policy_resolution.json",
                coordinate="screen.policy",
                group_ids=request.expected_system_ids,
                candidates=(),
                selected=(),
                quota=int(dict(request.expected_thresholds or {})["quota"]["keep"]),
                entropy=entropy,
            )

        if contract == "single-teacher/committee-student":
            audit = load_mixed_route_audit(
                request.screen_dir / "mixed_route_scores.npz"
            )
            identities, systems = _audit_identity_systems(audit)
            candidate_mask = np.asarray(audit["normal_teacher_scored"], dtype=bool) & np.asarray(
                audit["teacher_trusted"], dtype=bool
            )
            selected_mask = np.asarray(audit["normal_train_selected"], dtype=bool)
            values = np.asarray(audit["student_force_deviation"], dtype=float)
            entropy = f"{request.expected_entropy}/post-teacher"
        else:
            audit = load_committee_force_audit(
                request.screen_dir / "committee_force_scores.npz"
            )
            identities, systems = _audit_identity_systems(audit)
            in_band = np.asarray(audit["student_in_band"], dtype=bool)
            values = np.asarray(audit["student_force_deviation"], dtype=float)
            pool = int(dict(request.expected_thresholds or {})["quota"]["pool"])
            candidate_mask = np.zeros(len(identities), dtype=bool)
            for group_id in request.expected_system_ids:
                indices = [
                    index
                    for index, (system, eligible) in enumerate(
                        zip(systems, in_band, strict=True)
                    )
                    if system == group_id and bool(eligible)
                ]
                indices.sort(key=lambda index: identities[index])
                indices.sort(key=lambda index: (-values[index], identities[index]))
                for index in indices if pool < 0 else indices[:pool]:
                    candidate_mask[index] = True
            selected_mask = np.asarray(
                audit["normal_teacher_evaluated"], dtype=bool
            )
            entropy = request.expected_entropy

        def frames(mask):
            result = []
            for identity, system_id, value, chosen in zip(
                identities, systems, values, mask, strict=True
            ):
                if not bool(chosen):
                    continue
                frame = Atoms()
                frame.info.update(
                    structure_id=identity,
                    system_id=system_id,
                    force_uncertainty=float(value),
                )
                result.append(frame)
            result.sort(key=lambda frame: str(frame.info["structure_id"]))
            result.sort(
                key=lambda frame: (
                    str(frame.info["system_id"]),
                    -float(frame.info["force_uncertainty"]),
                    str(frame.info["structure_id"]),
                )
            )
            return tuple(result)

        candidates = frames(candidate_mask)
        selected = frames(selected_mask)
        return validate_policy_resolution(
            request.screen_dir / "policy_resolution.json",
            coordinate="screen.policy",
            group_ids=request.expected_system_ids,
            candidates=candidates,
            selected=selected,
            quota=int(dict(request.expected_thresholds or {})["quota"]["keep"]),
            entropy=entropy,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _valid_element_direct_selector(
    request: ScreenArtifactValidationRequest,
) -> bool:
    try:
        from workflow_artifacts.lifecycle.element_direct import (
            element_rows_from_payload,
            load_element_direct_screen_evidence,
        )

        audit = load_element_direct_screen_evidence(
            request.screen_dir / "element_direct_screen.npz"
        )
        identities = tuple(str(item) for item in audit["structure_ids"].tolist())
        systems = tuple(str(item) for item in audit["system_ids"].tolist())
        type_map = tuple(str(item) for item in audit["type_map"].tolist())
        if (
            str(audit["prescreen"].item()) != request.prescreen
            or str(audit["screen_policy"].item()) != "ELEMENT_DIRECT"
            or type_map != tuple(request.expected_type_map)
            or request.expected_thresholds is None
            or json.loads(str(audit["route_thresholds_json"].item()))
            != dict(request.expected_thresholds)
            or json.loads(str(audit["screen_policy_json"].item()))
            != selector_identity("ELEMENT_DIRECT").as_dict()
            or str(audit["model_identity"].item()) != request.joint_model_identity
            or str(audit["dataset_identity"].item())
            != request.joint_dataset_identity
        ):
            return False
        _validate_expected_systems(systems, request.expected_system_ids)
        candidate_rows, candidate_presence = element_rows_from_payload(
            audit, background=False
        )
        if not tuple(str(item) for item in audit["background_ids"].tolist()):
            return False
        index_by_id = {identity: index for index, identity in enumerate(identities)}
        system_by_id = dict(zip(identities, systems, strict=True))
        eligible = _flagged_order(audit, identities, "selector_candidate")
        eligible_by_group: dict[str, list[str]] = {}
        for identity in eligible:
            eligible_by_group.setdefault(system_by_id[identity], []).append(identity)
        selected: set[str] = set()
        keep = int(audit["keep"].item())
        from workflow_policies.screen_tasks import element_selection_task
        jobs = []
        for group_ids in eligible_by_group.values():
            indices = tuple(index_by_id[item] for item in group_ids)
            jobs.append((tuple(group_ids), tuple(candidate_rows[index] for index in indices),
                         tuple(candidate_presence[index] for index in indices), type_map, keep))
        for chosen, _audit in (request.ordered_map or map)(element_selection_task, jobs):
            selected.update(chosen)
        expected_normal = tuple(item for item in identities if item in selected)
        if expected_normal != _flagged_order(
            audit, identities, "normal_train_selected"
        ):
            return False
        route_kind = str(audit["route_kind"].item())
        if request.prescreen == "GMM" and route_kind == "single-teacher/single-student":
            scores = validate_gmm_screen_scores(request.screen_dir / "gmm_scores.npz")
            if tuple(str(item) for item in scores["structure_ids"].tolist()) != identities:
                return False
            if _flagged_order(scores, identities, "student_p_pool") != tuple(
                str(item) for item in audit["proposal_ids"].tolist()
            ) or _flagged_order(scores, identities, "student_eligible") != eligible:
                return False
            for field in (
                "normal_train_selected",
                "train_selected",
                "fp_candidate",
            ):
                if _flagged_order(scores, identities, field) != _flagged_order(
                    audit, identities, field
                ):
                    return False
        elif (
            request.prescreen == "GMM"
            and route_kind == "committee-teacher/single-student"
        ):
            mixed = load_mixed_route_audit(
                request.screen_dir / "mixed_route_scores.npz"
            )
            mixed_ids = tuple(
                str(item) for item in mixed["structure_ids"].tolist()
            )
            if mixed_ids != identities:
                return False
            if _flagged_order(
                mixed, identities, "normal_teacher_evaluated"
            ) != tuple(str(item) for item in audit["proposal_ids"].tolist()):
                return False
            if _flagged_order(mixed, identities, "teacher_low") != eligible:
                return False
            for field in (
                "normal_train_selected",
                "train_selected",
                "fp_candidate",
            ):
                if _flagged_order(mixed, identities, field) != _flagged_order(
                    audit, identities, field
                ):
                    return False
        else:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _valid_element_direct_screen(
    request: ScreenArtifactValidationRequest,
) -> bool:
    try:
        guard = _load_configured_pair_guard(request.screen_dir, request.pair_guard)
        if guard is not None and _is_all_reject_guard(guard):
            train_frames, fp_frames = _canonical_outputs(
                request,
                contract=(
                    "gmm"
                    if str(getattr(request.contract, "value", request.contract))
                    == "gmm-reference-tail"
                    else "committee"
                ),
            )
            return _valid_all_reject_screen(
                request.screen_dir,
                train_frames,
                fp_frames,
                guard=guard,
                expected_system_ids=request.expected_system_ids,
            )
        from workflow_artifacts.lifecycle.element_direct import (
            element_rows_from_payload,
            load_element_direct_screen_evidence,
        )

        audit = load_element_direct_screen_evidence(
            request.screen_dir / "element_direct_screen.npz"
        )
        route_kind = str(audit["route_kind"].item())
        output_kind = "gmm" if route_kind.startswith("single-teacher/") else "committee"
        train_frames, fp_frames = _canonical_outputs(request, contract=output_kind)
        if any(
            (request.screen_dir / name).exists()
            for name in (
                "joint_direct_screen.npz", "gmm_router", "gmm_scores.npz",
                "mixed_route_scores.npz", "committee_force_scores.npz",
            )
        ):
            return False
        identities = tuple(str(item) for item in audit["structure_ids"].tolist())
        systems = tuple(str(item) for item in audit["system_ids"].tolist())
        background_ids = tuple(
            str(item) for item in audit["background_ids"].tolist()
        )
        type_map = tuple(str(item) for item in audit["type_map"].tolist())
        if len(identities) != len(set(identities)) or type_map != tuple(
            request.expected_type_map
        ):
            return False
        _validate_expected_systems(systems, request.expected_system_ids)
        if guard is not None and identities != _guard_pass_order(guard):
            return False
        if request.expected_thresholds is None or json.loads(
            str(audit["route_thresholds_json"].item())
        ) != dict(request.expected_thresholds):
            return False
        if json.loads(str(audit["prescreen_policy_json"].item())) != selector_identity(
            request.prescreen
        ).as_dict() or json.loads(
            str(audit["screen_policy_json"].item())
        ) != selector_identity(
            request.screen_policy
        ).as_dict():
            return False
        readout_performed = bool(audit["readout_performed"].item())
        if readout_performed:
            if (
                str(audit["model_identity"].item()) != request.joint_model_identity
                or str(audit["dataset_identity"].item())
                != request.joint_dataset_identity
            ):
                return False
        elif str(audit["model_identity"].item()) or str(
            audit["dataset_identity"].item()
        ):
            return False
        candidate_rows, candidate_presence = element_rows_from_payload(
            audit, background=False
        )
        background_rows, background_presence = element_rows_from_payload(
            audit, background=True
        )
        pool = int(audit["pool"].item())
        grouped: dict[str, list[int]] = {}
        for index, system_id in enumerate(systems):
            grouped.setdefault(system_id, []).append(index)
        proposal_members: set[str] = set()
        from workflow_policies.screen_tasks import element_joint_selection_task, element_selection_task
        jobs = []
        for indices in grouped.values():
            group_ids = tuple(identities[index] for index in indices)
            if pool >= len(indices):
                proposal_members.update(group_ids)
                continue
            if not readout_performed or request.prescreen != "ELEMENT_JOINT_DIRECT" or not background_ids:
                return False
            jobs.append((group_ids, tuple(candidate_rows[index] for index in indices),
                         tuple(candidate_presence[index] for index in indices), background_ids,
                         background_rows, background_presence, type_map, pool))
        for chosen, _audit in (request.ordered_map or map)(element_joint_selection_task, jobs):
            proposal_members.update(chosen)
        proposal = tuple(item for item in identities if item in proposal_members)
        if proposal != tuple(str(item) for item in audit["proposal_ids"].tolist()):
            return False
        if proposal != _flagged_order(audit, identities, "student_pool"):
            return False
        system_by_id = dict(zip(identities, systems, strict=True))
        train_order = _canonical_order(train_frames, system_by_id)
        fp_order = _canonical_order(fp_frames, system_by_id)
        if train_order != _flagged_order(audit, identities, "train_selected"):
            return False
        if fp_order != _flagged_order(audit, identities, "fp_candidate"):
            return False
        metric_ids = tuple(str(item) for item in audit["teacher_metric_ids"].tolist())
        if metric_ids != proposal:
            return False
        metric_field = "teacher_p" if output_kind == "gmm" else "force_uncertainty"
        _validate_frame_scalar_alignment(
            fp_frames,
            identities=metric_ids,
            values=np.asarray(audit["teacher_metrics"], dtype=float),
            field=metric_field,
        )
        if request.screen_policy == "ELEMENT_DIRECT":
            keep = int(audit["keep"].item())
            eligible = _flagged_order(audit, identities, "selector_candidate")
            eligible_by_group: dict[str, list[str]] = {}
            for identity in eligible:
                eligible_by_group.setdefault(system_by_id[identity], []).append(identity)
            expected_selected: set[str] = set()
            index_by_id = {identity: index for index, identity in enumerate(identities)}
            jobs = []
            for group_ids in eligible_by_group.values():
                indices = tuple(index_by_id[item] for item in group_ids)
                jobs.append((tuple(group_ids), tuple(candidate_rows[index] for index in indices),
                             tuple(candidate_presence[index] for index in indices), type_map, keep))
            for chosen, _audit in (request.ordered_map or map)(element_selection_task, jobs):
                expected_selected.update(chosen)
            if tuple(item for item in identities if item in expected_selected) != train_order:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _valid_joint_direct_screen(
    request: ScreenArtifactValidationRequest,
) -> bool:
    try:
        guard = _load_configured_pair_guard(
            request.screen_dir, request.pair_guard
        )
        if guard is not None and _is_all_reject_guard(guard):
            train_frames, fp_frames = _canonical_outputs(
                request,
                contract=(
                    "gmm"
                    if str(getattr(request.contract, "value", request.contract))
                    == "gmm-reference-tail"
                    else "committee"
                ),
            )
            return _valid_all_reject_screen(
                request.screen_dir,
                train_frames,
                fp_frames,
                guard=guard,
                expected_system_ids=request.expected_system_ids,
            )
        from workflow_artifacts.lifecycle.joint_direct import (
            load_joint_direct_screen_evidence,
        )

        audit = load_joint_direct_screen_evidence(
            request.screen_dir / "joint_direct_screen.npz"
        )
        route_kind = str(audit["route_kind"].item())
        output_kind = "gmm" if route_kind.startswith("single-teacher/") else "committee"
        train_frames, fp_frames = _canonical_outputs(request, contract=output_kind)
        if any(
            (request.screen_dir / name).exists()
            for name in ("gmm_router", "gmm_scores.npz", "mixed_route_scores.npz", "committee_force_scores.npz")
        ):
            return False
        identities = tuple(str(item) for item in audit["structure_ids"].tolist())
        systems = tuple(str(item) for item in audit["system_ids"].tolist())
        if len(identities) != len(set(identities)):
            return False
        _validate_expected_systems(systems, request.expected_system_ids)
        if guard is not None and identities != _guard_pass_order(guard):
            return False
        if request.expected_thresholds is None or json.loads(
            str(audit["route_thresholds_json"].item())
        ) != dict(request.expected_thresholds):
            return False
        pool = int(audit["pool"].item())
        if pool <= 0:
            return False
        candidate_ids = tuple(
            str(item) for item in audit["readout_candidate_ids"].tolist()
        )
        candidate_vectors = dict(
            zip(candidate_ids, audit["candidate_vectors"], strict=True)
        )
        background_ids = tuple(str(item) for item in audit["background_ids"].tolist())
        background_vectors = dict(
            zip(background_ids, audit["background_vectors"], strict=True)
        )
        actual_model_identity = str(audit["model_identity"].item())
        actual_dataset_identity = str(audit["dataset_identity"].item())
        if background_ids:
            if (
                request.joint_model_identity is None
                or request.joint_dataset_identity is None
                or actual_model_identity != request.joint_model_identity
                or actual_dataset_identity != request.joint_dataset_identity
            ):
                return False
        elif actual_model_identity or actual_dataset_identity:
            return False
        grouped: dict[str, list[str]] = {}
        for identity, system_id in zip(identities, systems, strict=True):
            grouped.setdefault(system_id, []).append(identity)
        proposal_members: set[str] = set()
        from workflow_policies.screen_tasks import joint_selection_task
        jobs = []
        background = None
        for group_candidates in grouped.values():
            if pool >= len(group_candidates):
                proposal_members.update(group_candidates)
                continue
            if not background_ids:
                return False
            if background is None:
                background = np.stack([background_vectors[item] for item in background_ids])
            jobs.append((group_candidates,
                         np.stack([candidate_vectors[item] for item in group_candidates]),
                         background_ids, background, pool))
        for chosen, _audit in (request.ordered_map or map)(joint_selection_task, jobs):
            proposal_members.update(chosen)
        proposal = [
            identity for identity in identities if identity in proposal_members
        ]
        if tuple(proposal) != tuple(str(item) for item in audit["proposal_ids"].tolist()):
            return False
        if tuple(
            identity
            for identity, chosen in zip(identities, audit["student_pool"], strict=True)
            if bool(chosen)
        ) != tuple(proposal):
            return False
        system_by_id = dict(zip(identities, systems, strict=True))
        train_order = _canonical_order(train_frames, system_by_id)
        fp_order = _canonical_order(fp_frames, system_by_id)
        if train_order != _flagged_order(audit, identities, "train_selected"):
            return False
        if fp_order != _flagged_order(audit, identities, "fp_candidate"):
            return False
        metric_ids = tuple(
            str(item) for item in audit["teacher_metric_ids"].tolist()
        )
        if metric_ids != tuple(proposal):
            return False
        metric_field = "teacher_p" if output_kind == "gmm" else "force_uncertainty"
        _validate_frame_scalar_alignment(
            fp_frames,
            identities=metric_ids,
            values=np.asarray(audit["teacher_metrics"], dtype=float),
            field=metric_field,
        )
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _valid_gmm_screen(request: ScreenArtifactValidationRequest) -> bool:
    path = request.screen_dir
    try:
        train_frames, fp_frames = _canonical_outputs(request, contract="gmm")
        guard = _load_configured_pair_guard(path, request.pair_guard)
        if guard is not None and _is_all_reject_guard(guard):
            return _valid_all_reject_screen(
                path,
                train_frames,
                fp_frames,
                guard=guard,
                expected_system_ids=request.expected_system_ids,
            )
        if any(
            (path / name).exists()
            for name in ("mixed_route_scores.npz", "committee_force_scores.npz")
        ):
            return False
        if request.teacher_router is None:
            return False
        student_manifest = validate_gmm_router(path / "gmm_router")
        teacher_manifest = validate_gmm_router(request.teacher_router)
        scores = validate_gmm_screen_scores(path / "gmm_scores.npz")
        if request.expected_thresholds is None:
            return False
        _validate_gmm_effective_policy(scores, request.expected_thresholds)
        if guard is not None:
            validate_gmm_pair_guard_alignment(
                path / "gmm_scores.npz",
                path / "pair_guard.npz",
                thresholds=_pair_guard_thresholds(request.pair_guard),
            )
        identities = tuple(str(item) for item in scores["structure_ids"].tolist())
        audit_systems = tuple(str(item) for item in scores["system_ids"].tolist())
        _validate_expected_systems(audit_systems, request.expected_system_ids)
        system_by_id = dict(zip(identities, audit_systems, strict=True))
        train_order = _canonical_order(train_frames, system_by_id)
        fp_order = _canonical_order(fp_frames, system_by_id)
        _validate_frame_scalar_alignment(
            fp_frames,
            identities=identities,
            values=np.asarray(scores["teacher_p"], dtype=float),
            field="teacher_p",
        )
        _validate_optional_frame_mean_evidence(
            path / "representative_features.npz",
            identities=fp_order,
            policy=request.fp_policy,
            descriptor_schema=dict(teacher_manifest["descriptor_schema"]),
        )
        _validate_optional_frame_mean_evidence(
            path / "student_representative_features.npz",
            identities=train_order,
            policy=request.screen_policy,
            descriptor_schema=dict(student_manifest["descriptor_schema"]),
        )
        expected_train = tuple(
            identity
            for identity, chosen in zip(
                identities, scores["train_selected"], strict=True
            )
            if bool(chosen)
        )
        expected_fp = tuple(
            identity
            for identity, chosen in zip(
                identities, scores["fp_candidate"], strict=True
            )
            if bool(chosen)
        )
        return train_order == expected_train and fp_order == expected_fp
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validate_gmm_effective_policy(
    scores: Mapping[str, np.ndarray], expected: Mapping[str, object]
) -> None:
    if int(np.asarray(scores["schema_revision"]).item()) != 5:
        raise ValueError("GMM effective policy schema is unsupported")
    actual = {
        "routing_policy": {
            "name": str(np.asarray(scores["routing_policy"]).item()),
            "revision": int(np.asarray(scores["routing_policy_revision"]).item()),
        },
        "teacher_p": _gmm_teacher_thresholds(scores),
        "student_p": _gmm_student_thresholds(scores),
        "quota": {
            "fp_pool": int(np.asarray(scores["fp_pool"]).item()),
            "pool": int(np.asarray(scores["pool"]).item()),
            "keep": int(np.asarray(scores["keep"]).item()),
        },
        "fill_underfilled_from_safe_region": bool(
            np.asarray(scores["safe_fill_enabled"]).item()
        ),
        "selector": {
            "name": str(np.asarray(scores["selector"]).item()),
            "revision": int(np.asarray(scores["selector_revision"]).item()),
        },
    }
    if actual != dict(expected):
        raise ValueError("student-first GMM effective policy evidence drifted")


def _gmm_teacher_thresholds(scores: Mapping[str, np.ndarray]) -> dict[str, float]:
    value = {
        "floor": float(np.asarray(scores["teacher_floor"]).item()),
        "trust": float(np.asarray(scores["teacher_trust"]).item()),
    }
    if bool(np.asarray(scores["teacher_ceiling_configured"]).item()):
        value["ceiling"] = float(np.asarray(scores["teacher_ceiling"]).item())
    return value


def _gmm_student_thresholds(scores: Mapping[str, np.ndarray]) -> dict[str, float]:
    value = {"floor": float(np.asarray(scores["student_floor"]).item())}
    if bool(np.asarray(scores["student_ceiling_configured"]).item()):
        value["ceiling"] = float(np.asarray(scores["student_ceiling"]).item())
    return value


def _valid_single_teacher_committee_student(
    request: ScreenArtifactValidationRequest,
) -> bool:
    path = request.screen_dir
    try:
        train_frames, fp_frames = _canonical_outputs(request, contract="gmm")
        guard = _load_configured_pair_guard(path, request.pair_guard)
        if guard is not None and _is_all_reject_guard(guard):
            return _valid_all_reject_screen(
                path,
                train_frames,
                fp_frames,
                guard=guard,
                expected_system_ids=request.expected_system_ids,
            )
        if request.expected_thresholds is None or request.teacher_router is None:
            return False
        if any(
            (path / name).exists()
            for name in (
                "gmm_router",
                "gmm_scores.npz",
                "student_representative_features.npz",
                "committee_force_scores.npz",
            )
        ):
            return False
        audit = load_mixed_route_audit(path / "mixed_route_scores.npz")
        if (
            str(np.asarray(audit["policy"]).item())
            != SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID
        ):
            return False
        if audit["thresholds"] != dict(request.expected_thresholds):
            return False
        identities, audit_systems = _audit_identity_systems(audit)
        _validate_expected_systems(audit_systems, request.expected_system_ids)
        if guard is not None and identities != _guard_pass_order(guard):
            return False
        system_by_id = dict(zip(identities, audit_systems, strict=True))
        train_order = _canonical_order(train_frames, system_by_id)
        fp_order = _canonical_order(fp_frames, system_by_id)
        expected_train = _flagged_order(audit, identities, "train_selected")
        expected_fp = _flagged_order(audit, identities, "fp_candidate")
        if train_order != expected_train or fp_order != expected_fp:
            return False
        _validate_frame_scalar_alignment(
            fp_frames,
            identities=identities,
            values=np.asarray(audit["teacher_p"], dtype=float),
            field="teacher_p",
        )
        _validate_frame_scalar_alignment(
            fp_frames,
            identities=identities,
            values=np.asarray(audit["student_force_deviation"], dtype=float),
            field="force_uncertainty",
        )
        teacher_manifest = validate_gmm_router(request.teacher_router)
        _validate_optional_frame_mean_evidence(
            path / "representative_features.npz",
            identities=fp_order,
            policy=request.fp_policy,
            descriptor_schema=dict(teacher_manifest["descriptor_schema"]),
        )
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _valid_committee_teacher_single_student(
    request: ScreenArtifactValidationRequest,
) -> bool:
    path = request.screen_dir
    try:
        train_frames, fp_frames = _canonical_outputs(request, contract="force")
        guard = _load_configured_pair_guard(path, request.pair_guard)
        if guard is not None and _is_all_reject_guard(guard):
            return _valid_all_reject_screen(
                path,
                train_frames,
                fp_frames,
                guard=guard,
                expected_system_ids=request.expected_system_ids,
            )
        if request.expected_thresholds is None:
            return False
        if any(
            (path / name).exists()
            for name in (
                "gmm_scores.npz",
                "representative_features.npz",
                "committee_force_scores.npz",
            )
        ):
            return False
        audit = load_mixed_route_audit(path / "mixed_route_scores.npz")
        if audit["policy"] != COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID:
            return False
        if audit["thresholds"] != dict(request.expected_thresholds):
            return False
        summary = validate_committee_teacher_single_student_summary(
            path / "screen_summary.json"
        )
        if (
            summary["thresholds"] != dict(request.expected_thresholds)
            or summary["fill_underfilled_from_safe_region"]
            != bool(np.asarray(audit["safe_fill_enabled"]).item())
        ):
            return False
        identities, audit_systems = _audit_identity_systems(audit)
        _validate_expected_systems(audit_systems, request.expected_system_ids)
        if guard is not None and identities != _guard_pass_order(guard):
            return False
        student_manifest = validate_gmm_router(path / "gmm_router")
        system_by_id = dict(zip(identities, audit_systems, strict=True))
        train_order = _canonical_order(train_frames, system_by_id)
        fp_order = _canonical_order(fp_frames, system_by_id)
        expected_train = _flagged_order(audit, identities, "train_selected")
        expected_fp = _flagged_order(audit, identities, "fp_candidate")
        if train_order != expected_train or fp_order != expected_fp:
            return False
        if set(train_order) & set(fp_order):
            return False
        student_pool = np.asarray(audit["student_pool"], dtype=bool)
        normal_teacher_evaluated = np.asarray(
            audit["normal_teacher_evaluated"], dtype=bool
        )
        teacher_evaluated = np.asarray(audit["teacher_evaluated"], dtype=bool)
        teacher_low = np.asarray(audit["teacher_low"], dtype=bool)
        teacher_middle = np.asarray(audit["teacher_middle"], dtype=bool)
        teacher_high = np.asarray(audit["teacher_high"], dtype=bool)
        if not np.array_equal(student_pool, normal_teacher_evaluated):
            return False
        if not np.array_equal(
            teacher_evaluated, teacher_low | teacher_middle | teacher_high
        ):
            return False
        if np.any(
            (
                teacher_low.astype(int)
                + teacher_middle.astype(int)
                + teacher_high.astype(int)
            )
            > 1
        ):
            return False
        if np.any(np.asarray(audit["train_selected"], dtype=bool) & ~teacher_low):
            return False
        if not np.array_equal(
            np.asarray(audit["fp_candidate"], dtype=bool),
            teacher_middle & normal_teacher_evaluated,
        ):
            return False
        _validate_frame_scalar_alignment(
            [*train_frames, *fp_frames],
            identities=identities,
            values=np.asarray(audit["student_p"], dtype=float),
            field="student_p",
        )
        _validate_sparse_force_alignment([*train_frames, *fp_frames], audit)
        _validate_committee_teacher_single_student_summary_alignment(summary, audit)
        _validate_optional_frame_mean_evidence(
            path / "student_representative_features.npz",
            identities=train_order,
            policy=request.screen_policy,
            descriptor_schema=dict(student_manifest["descriptor_schema"]),
        )
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validate_committee_teacher_single_student_summary_alignment(
    summary: Mapping[str, Any], audit: Mapping[str, Any]
) -> None:
    flags = {
        name: np.asarray(audit[name], dtype=bool)
        for name in (
            "student_floor_discard", "student_candidate", "student_pool",
            "teacher_evaluated", "teacher_low", "teacher_middle", "teacher_high",
            "normal_train_selected", "safe_fill_eligible", "safe_fill_attempted",
            "safe_fill_accepted", "train_selected", "fp_candidate",
        )
    }
    expected = {
        "student_floor_discard": int(np.count_nonzero(flags["student_floor_discard"])),
        "student_candidate": int(np.count_nonzero(flags["student_candidate"])),
        "student_pool": int(np.count_nonzero(flags["student_pool"])),
        "teacher_evaluated": int(np.count_nonzero(flags["teacher_evaluated"])),
        "teacher_low": int(np.count_nonzero(flags["teacher_low"])),
        "teacher_middle": int(np.count_nonzero(flags["teacher_middle"])),
        "teacher_high": int(np.count_nonzero(flags["teacher_high"])),
        "normal_train": int(np.count_nonzero(flags["normal_train_selected"])),
        "safe_fill_eligible": int(np.count_nonzero(flags["safe_fill_eligible"])),
        "safe_fill_attempted": int(np.count_nonzero(flags["safe_fill_attempted"])),
        "safe_fill_accepted": int(np.count_nonzero(flags["safe_fill_accepted"])),
        "train_selected": int(np.count_nonzero(flags["train_selected"])),
        "fp_candidate": int(np.count_nonzero(flags["fp_candidate"])),
    }
    for name, value in expected.items():
        if summary["counts"][name] != value:
            raise ValueError("committee-teacher mixed summary does not match audit")


def _valid_committee_screen(request: ScreenArtifactValidationRequest) -> bool:
    path = request.screen_dir
    try:
        train_frames, fp_frames = _canonical_outputs(request, contract="force")
        if any(
            (path / name).exists()
            for name in (
                "gmm_router",
                "gmm_scores.npz",
                "mixed_route_scores.npz",
                "student_representative_features.npz",
            )
        ):
            return False
        guard = _load_configured_pair_guard(path, request.pair_guard)
        if guard is not None and _is_all_reject_guard(guard):
            return _valid_all_reject_screen(
                path,
                train_frames,
                fp_frames,
                guard=guard,
                expected_system_ids=request.expected_system_ids,
            )
        if request.expected_thresholds is None:
            return False
        audit = load_committee_force_audit(path / "committee_force_scores.npz")
        summary = validate_committee_force_summary(path / "screen_summary.json")
        if audit["thresholds"] != dict(request.expected_thresholds):
            return False
        if summary["thresholds"] != dict(request.expected_thresholds):
            return False
        if summary["fill_underfilled_from_safe_region"] != bool(
            audit["safe_fill_enabled"].item()
        ):
            return False
        identities, audit_systems = _audit_identity_systems(audit)
        _validate_expected_systems(audit_systems, request.expected_system_ids)
        train_order = _canonical_order(train_frames, dict(zip(identities, audit_systems, strict=True)))
        fp_order = _canonical_order(fp_frames, dict(zip(identities, audit_systems, strict=True)))
        if set(train_order) & set(fp_order):
            return False
        if train_order != _flagged_order(audit, identities, "train_selected"):
            return False
        if fp_order != _flagged_order(audit, identities, "fp_candidate"):
            return False
        _validate_sparse_force_alignment([*train_frames, *fp_frames], audit)
        if guard is not None:
            pass_order = _guard_pass_order(guard)
            if identities != pass_order:
                return False
            if not _is_ordered_unique_subset(train_order, pass_order):
                return False
            if not _is_ordered_unique_subset(fp_order, pass_order):
                return False
        elif tuple(identities) != tuple(str(item) for item in identities):
            return False
        _validate_committee_force_summary_alignment(
            summary=summary,
            audit=audit,
            systems=audit_systems,
            expected_system_ids=request.expected_system_ids,
        )
        representative = path / "representative_features.npz"
        if request.fp_policy == "random" and representative.exists():
            return False
        _validate_legacy_representative_evidence(
            representative, expected_identities=(*train_order, *fp_order)
        )
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _validate_committee_force_summary_alignment(
    *,
    summary: Mapping[str, Any],
    audit: Mapping[str, Any],
    systems: tuple[str, ...],
    expected_system_ids: tuple[str, ...],
) -> None:
    flags = {
        name: np.asarray(audit[name], dtype=bool)
        for name in (
            "student_in_band", "student_pool", "normal_teacher_evaluated",
            "teacher_evaluated", "teacher_low", "teacher_middle", "teacher_high",
            "normal_train_selected", "safe_fill_eligible", "safe_fill_attempted",
            "safe_fill_accepted", "train_selected", "fp_candidate",
        )
    }
    expected = {
        "student_in_band": int(np.count_nonzero(flags["student_in_band"])),
        "student_pool": int(np.count_nonzero(flags["student_pool"])),
        "normal_teacher_evaluated": int(np.count_nonzero(flags["normal_teacher_evaluated"])),
        "teacher_evaluated": int(np.count_nonzero(flags["teacher_evaluated"])),
        "teacher_low": int(np.count_nonzero(flags["teacher_low"])),
        "teacher_middle": int(np.count_nonzero(flags["teacher_middle"])),
        "teacher_high": int(np.count_nonzero(flags["teacher_high"])),
        "normal_train": int(np.count_nonzero(flags["normal_train_selected"])),
        "safe_fill_eligible": int(np.count_nonzero(flags["safe_fill_eligible"])),
        "safe_fill_attempted": int(np.count_nonzero(flags["safe_fill_attempted"])),
        "safe_fill_accepted": int(np.count_nonzero(flags["safe_fill_accepted"])),
        "train_selected": int(np.count_nonzero(flags["train_selected"])),
        "fp_candidate": int(np.count_nonzero(flags["fp_candidate"])),
    }
    for name, value in expected.items():
        if summary["counts"][name] != value:
            raise ValueError("committee-force summary does not match audit")
    if summary["counts"]["guard_pass"] != len(systems):
        raise ValueError("committee-force summary guard count is inconsistent")
    for system_id in expected_system_ids:
        owned = np.asarray(systems) == system_id
        counts = summary["per_system"].get(system_id)
        if not isinstance(counts, Mapping):
            raise ValueError("committee-force summary system counts are missing")
        for name, flags_name in (
            ("student_in_band", "student_in_band"), ("student_pool", "student_pool"),
            ("normal_teacher_evaluated", "normal_teacher_evaluated"), ("teacher_evaluated", "teacher_evaluated"),
            ("teacher_low", "teacher_low"), ("teacher_middle", "teacher_middle"),
            ("teacher_high", "teacher_high"), ("normal_train", "normal_train_selected"),
            ("safe_fill_eligible", "safe_fill_eligible"), ("safe_fill_attempted", "safe_fill_attempted"),
            ("safe_fill_accepted", "safe_fill_accepted"), ("train_selected", "train_selected"),
            ("fp_candidate", "fp_candidate"),
        ):
            if counts[name] != int(np.count_nonzero(flags[flags_name] & owned)):
                raise ValueError("committee-force per-system summary does not match audit")


def _canonical_outputs(
    request: ScreenArtifactValidationRequest, *, contract: str
) -> tuple[list[Any], list[Any]]:
    path = request.screen_dir
    train = read_extxyz_structures(path / "train.xyz", allow_empty=True)
    candidates = read_extxyz_structures(path / "fp_candidate.xyz", allow_empty=True)
    _validate_canonical_screen_frames(
        train,
        expected_system_ids=request.expected_system_ids,
        expected_type_map=request.expected_type_map,
        require_labels=True,
        score_name=None,
    )
    _validate_canonical_screen_frames(
        candidates,
        expected_system_ids=request.expected_system_ids,
        expected_type_map=request.expected_type_map,
        require_labels=False,
        score_name="teacher_p" if contract == "gmm" else "force_uncertainty",
    )
    return train, candidates


def _valid_selection_group_outcomes(
    request: ScreenArtifactValidationRequest,
) -> bool:
    try:
        plan = request.selection_plan
        if plan is None or plan.group_ids != request.expected_system_ids:
            return False
        payload = json.loads(
            (request.screen_dir / "selection_group_outcomes.json").read_text(
                encoding="utf-8"
            )
        )
        if payload.get("schema_revision") in {2, 3}:
            return _valid_joint_selection_group_outcomes(
                request, payload=payload, plan=plan
            )
        fields = {
            "schema",
            "schema_revision",
            "selection_groups",
            "role_to_group",
            "screen_policy",
            "rows",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload["schema"]
            != "dice.lifecycle.selection-group-screen-outcomes"
            or payload["schema_revision"] != 1
            or tuple(payload["selection_groups"]) != plan.group_ids
            or tuple(tuple(item) for item in payload["role_to_group"])
            != plan.role_to_group
            or payload["screen_policy"] != request.screen_policy
        ):
            return False
        rows = tuple(payload["rows"])
        if tuple(row.get("selection_group_id") for row in rows) != plan.group_ids:
            return False
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
        systems: tuple[str, ...] = ()
        pool_mask = normal_mask = np.zeros(0, dtype=np.bool_)
        for name in (
            "gmm_scores.npz",
            "mixed_route_scores.npz",
            "committee_force_scores.npz",
        ):
            path = request.screen_dir / name
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as audit:
                systems = tuple(str(item) for item in audit["system_ids"].tolist())
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
                    else np.zeros(len(systems), dtype=np.bool_)
                )
                normal_mask = np.asarray(
                    audit["normal_train_selected"], dtype=np.bool_
                )
            break
        train = read_extxyz_structures(
            request.screen_dir / "train.xyz", allow_empty=True
        )
        fp = read_extxyz_structures(
            request.screen_dir / "fp_candidate.xyz", allow_empty=True
        )
        mapping = dict(plan.role_to_group)
        for frame in (*train, *fp):
            role_id = frame.info.get("sampling_role_id")
            group_id = frame.info.get("selection_group_id")
            if (
                role_id not in mapping
                or mapping[role_id] != group_id
                or frame.info.get("system_id") != group_id
            ):
                return False
        train_counts = _counts_by_system(train)
        fp_counts = _counts_by_system(fp)
        for row in rows:
            if not isinstance(row, dict) or set(row) != row_fields:
                return False
            group_id = row["selection_group_id"]
            count_fields = (
                "candidate_count",
                "normal_selected_count",
                "safe_fill_count",
                "final_selected_count",
                "underfill",
                "fp_candidate_count",
            )
            if any(
                not isinstance(row[key], int)
                or isinstance(row[key], bool)
                or row[key] < 0
                for key in count_fields
            ):
                return False
            keep = row["keep_ceiling"]
            pool = row["proposal_ceiling"]
            fp_pool = row["fp_proposal_ceiling"]
            if (
                not isinstance(keep, int)
                or isinstance(keep, bool)
                or keep <= 0
                or pool == 0
                or not isinstance(pool, int)
                or isinstance(pool, bool)
                or fp_pool is not None
                and (
                    not isinstance(fp_pool, int)
                    or isinstance(fp_pool, bool)
                    or fp_pool == 0
                )
            ):
                return False
            owned = np.asarray([item == group_id for item in systems], dtype=np.bool_)
            candidate_count = int(np.count_nonzero(pool_mask & owned))
            normal_count = int(np.count_nonzero(normal_mask & owned))
            final_count = int(train_counts.get(group_id, 0))
            fp_count = int(fp_counts.get(group_id, 0))
            if (
                row["candidate_count"] != candidate_count
                or row["normal_selected_count"] != normal_count
                or row["safe_fill_count"] != final_count - normal_count
                or row["final_selected_count"] != final_count
                or row["fp_candidate_count"] != fp_count
                or row["underfill"] != keep - final_count
                or final_count > keep
                or pool != -1
                and candidate_count > pool
                or fp_pool not in (None, -1)
                and fp_count > fp_pool
            ):
                return False
        load_screen_report(
            request.screen_dir / "screen_report.json",
            expected_system_ids=plan.group_ids,
        )
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _valid_joint_selection_group_outcomes(request, *, payload, plan) -> bool:
    fields = {
        "schema", "schema_revision", "selection_groups", "role_to_group",
        "screen_policy", "rows",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["schema"] != "dice.lifecycle.selection-group-screen-outcomes"
        or tuple(payload["selection_groups"]) != plan.group_ids
        or tuple(tuple(item) for item in payload["role_to_group"]) != plan.role_to_group
        or payload["screen_policy"] != request.screen_policy
    ):
        return False
    if request.prescreen == "ELEMENT_JOINT_DIRECT":
        from workflow_artifacts.lifecycle.element_direct import (
            load_element_direct_screen_evidence,
        )

        audit = load_element_direct_screen_evidence(
            request.screen_dir / "element_direct_screen.npz"
        )
    else:
        from workflow_artifacts.lifecycle.joint_direct import (
            load_joint_direct_screen_evidence,
        )

        audit = load_joint_direct_screen_evidence(
            request.screen_dir / "joint_direct_screen.npz"
        )
    outcomes = json.loads(str(audit["outcomes_json"].item()))
    observed_systems = set(str(item) for item in audit["system_ids"].tolist())
    train = read_extxyz_structures(request.screen_dir / "train.xyz", allow_empty=True)
    fp = read_extxyz_structures(
        request.screen_dir / "fp_candidate.xyz", allow_empty=True
    )
    train_counts = _counts_by_system(train)
    fp_counts = _counts_by_system(fp)
    rows = tuple(payload["rows"])
    if tuple(row.get("selection_group_id") for row in rows) != plan.group_ids:
        return False
    extra = {
        "background_count", "bypass", "joint_selected_candidate_count",
        "joint_selected_background_count", "joint_realized_clusters",
        "joint_underfill", "underfill_reason",
    }
    if payload["schema_revision"] == 3:
        extra.add("joint_quota_completion_count")
    base = {
        "selection_group_id", "candidate_count", "proposal_ceiling",
        "keep_ceiling", "normal_selected_count", "safe_fill_count",
        "final_selected_count", "underfill", "fp_proposal_ceiling",
        "fp_candidate_count",
    }
    for row in rows:
        group_id = row["selection_group_id"]
        final = train_counts.get(group_id, 0)
        expected = outcomes.get(group_id)
        if group_id not in outcomes:
            # The evidence inventory contains only observed groups. An active
            # group with no observations still has an explicit zero summary.
            # Missing evidence for a nonempty group remains corruption.
            if (group_id in observed_systems
                or final or fp_counts.get(group_id, 0)):
                return False
            expected = dict(candidate_count=0, background_count=0, bypass=True,
                selected_candidate_count=0, selected_background_count=0,
                realized_clusters=0, quota_completion_count=0, underfill=0)
            if (row.get('normal_selected_count') != 0
                or row.get('safe_fill_count') != 0
                or row.get('keep_ceiling') != int(audit['keep'].item())
                or row.get('proposal_ceiling') != int(audit['pool'].item())
                or row.get('underfill') != int(audit['keep'].item())
                or row.get('underfill_reason') != 'none'):
                return False
        if (
            set(row) != base | extra
            or row["candidate_count"] != expected["candidate_count"]
            or row["background_count"] != expected["background_count"]
            or row["bypass"] != expected["bypass"]
            or row["joint_selected_candidate_count"]
            != expected["selected_candidate_count"]
            or row["joint_selected_background_count"]
            != expected["selected_background_count"]
            or row["joint_realized_clusters"] != expected["realized_clusters"]
            or row.get("joint_quota_completion_count", 0)
            != expected.get("quota_completion_count", 0)
            or row["joint_underfill"] != expected["underfill"]
            or row["final_selected_count"] != final
            or row["fp_candidate_count"] != fp_counts.get(group_id, 0)
        ):
            return False
    return True


def _counts_by_system(frames: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        system_id = str(frame.info["system_id"])
        counts[system_id] = counts.get(system_id, 0) + 1
    return counts


def _validate_canonical_screen_frames(
    frames: list[Any],
    *,
    expected_system_ids: tuple[str, ...],
    expected_type_map: tuple[str, ...],
    require_labels: bool,
    score_name: str | None,
) -> None:
    allowed_systems = set(expected_system_ids)
    allowed_elements = set(expected_type_map)
    identities: list[str] = []
    for index, frame in enumerate(frames):
        identity = validate_canonical_identity(frame)
        if identity["system_id"] not in allowed_systems:
            raise ValueError("canonical screen frame has an unexpected system")
        if allowed_elements and not set(frame.get_chemical_symbols()).issubset(
            allowed_elements
        ):
            raise ValueError("canonical screen frame has an unsupported element")
        if (
            not np.isfinite(np.asarray(frame.positions, dtype=float)).all()
            or not np.isfinite(np.asarray(frame.cell.array, dtype=float)).all()
        ):
            raise ValueError("canonical screen frame has non-finite geometry")
        identities.append(identity["structure_id"])
        results = getattr(getattr(frame, "calc", None), "results", {}) or {}
        if require_labels:
            if "energy" not in results or "forces" not in results:
                raise ValueError(
                    f"canonical screen train frame {index} lacks labels"
                )
            energy = np.asarray(results["energy"], dtype=float)
            forces = np.asarray(results["forces"], dtype=float)
            if (
                energy.size != 1
                or not np.isfinite(energy).all()
                or forces.shape != (len(frame), 3)
                or not np.isfinite(forces).all()
            ):
                raise ValueError(f"canonical screen train frame {index} lacks labels")
        if score_name is not None:
            value = results.get(score_name, frame.info.get(score_name))
            if value is None or not np.isfinite(np.asarray(value)).all():
                raise ValueError("canonical screen candidate lacks finite route score")
    if len(identities) != len(set(identities)):
        raise ValueError("canonical screen identities are duplicated")


def _canonical_order(
    frames: list[Any], system_by_id: Mapping[str, str]
) -> tuple[str, ...]:
    order = tuple(str(frame.info["structure_id"]) for frame in frames)
    if len(order) != len(set(order)):
        raise ValueError("canonical screen identities are duplicated")
    for frame in frames:
        identity = str(frame.info["structure_id"])
        if system_by_id.get(identity) != str(frame.info.get("system_id", "")):
            raise ValueError("canonical screen system ids do not match route evidence")
    return order


def _canonical_order_for_systems(
    frames: list[Any], expected_system_ids: tuple[str, ...]
) -> tuple[str, ...]:
    order = tuple(str(frame.info["structure_id"]) for frame in frames)
    if len(order) != len(set(order)):
        raise ValueError("canonical screen identities are duplicated")
    allowed = set(expected_system_ids)
    if any(str(frame.info.get("system_id", "")) not in allowed for frame in frames):
        raise ValueError("canonical screen contains an unexpected system id")
    return order


def _audit_identity_systems(
    audit: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    identities = tuple(str(item) for item in np.asarray(audit["structure_ids"]).tolist())
    systems = tuple(str(item) for item in np.asarray(audit["system_ids"]).tolist())
    if len(identities) != len(set(identities)) or len(systems) != len(identities):
        raise ValueError("route audit identity/system alignment is invalid")
    return identities, systems


def _flagged_order(
    audit: Mapping[str, object], identities: tuple[str, ...], name: str
) -> tuple[str, ...]:
    return tuple(
        identity
        for identity, selected in zip(
            identities, np.asarray(audit[name], dtype=bool), strict=True
        )
        if bool(selected)
    )


def _validate_frame_scalar_alignment(
    frames: list[Any],
    *,
    identities: tuple[str, ...],
    values: np.ndarray,
    field: str,
) -> None:
    if values.shape != (len(identities),):
        raise ValueError("route scalar evidence shape changed")
    by_id = dict(zip(identities, values.tolist(), strict=True))
    for frame in frames:
        identity = str(frame.info["structure_id"])
        expected = float(by_id[identity])
        actual = float(frame.info[field])
        if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(
            actual, expected, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "canonical screen scalar disagrees with route audit: "
                f"{identity} {field} {actual} != {expected}"
            )


def _validate_sparse_force_alignment(
    frames: list[Any], audit: Mapping[str, object]
) -> None:
    identities = tuple(
        str(item)
        for item in np.asarray(
            audit["teacher_force_deviation_structure_ids"]
        ).tolist()
    )
    values = np.asarray(audit["teacher_force_deviation"], dtype=float)
    _validate_frame_scalar_alignment(
        frames,
        identities=identities,
        values=values,
        field="force_uncertainty",
    )


def _validate_expected_systems(
    systems: tuple[str, ...], expected_system_ids: tuple[str, ...]
) -> None:
    allowed = set(expected_system_ids)
    if any(system not in allowed for system in systems):
        raise ValueError("route audit contains an unexpected system id")


def _load_configured_pair_guard(
    path: Path, config: Mapping[str, Any] | None
) -> PairGuardArtifact | None:
    artifact_path = path / "pair_guard.npz"
    if config is None:
        if artifact_path.exists():
            raise ValueError("unconfigured lifecycle screen published pair-guard evidence")
        return None
    return load_pair_guard(artifact_path, thresholds=_pair_guard_thresholds(config))


def _pair_guard_thresholds(config: Mapping[str, Any] | None) -> PairGuardThresholds:
    if config is None:
        raise ValueError("pair-guard thresholds are not configured")
    return PairGuardThresholds(
        minimum=float(config["min"]), pairs=dict(config.get("pairs", {}))
    )


def _guard_pass_order(guard: PairGuardArtifact) -> tuple[str, ...]:
    return tuple(
        str(identity)
        for identity, passed in zip(
            guard.structure_ids.tolist(), guard.passed.tolist(), strict=True
        )
        if bool(passed)
    )


def _is_all_reject_guard(guard: PairGuardArtifact) -> bool:
    return len(guard.structure_ids) > 0 and not bool(np.any(guard.passed))


def _valid_all_reject_screen(
    path: Path,
    train_frames: list[Any],
    fp_frames: list[Any],
    *,
    guard: PairGuardArtifact,
    expected_system_ids: tuple[str, ...],
) -> bool:
    if train_frames or fp_frames:
        return False
    try:
        _validate_expected_systems(
            tuple(str(item) for item in guard.system_ids.tolist()),
            expected_system_ids,
        )
    except ValueError:
        return False
    forbidden = (
        "gmm_router",
        "gmm_scores.npz",
        "mixed_route_scores.npz",
        "committee_force_scores.npz",
        "representative_features.npz",
        "student_representative_features.npz",
        ".screen_work",
    )
    return not any((path / name).exists() for name in forbidden)


def _validate_optional_frame_mean_evidence(
    path: Path,
    *,
    identities: tuple[str, ...],
    policy: str,
    descriptor_schema: Any,
) -> None:
    required = policy in {"DIRECT", "FPS"} and bool(identities)
    if required:
        load_frame_mean_representatives(
            path,
            expected_identities=identities,
            expected_descriptor_schema=descriptor_schema,
        )
    elif path.exists():
        raise ValueError(f"unexpected frame-mean representative evidence: {path}")


def _validate_legacy_representative_evidence(
    path: Path, *, expected_identities: tuple[str, ...]
) -> None:
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as payload:
        identities = tuple(str(item) for item in payload["ids"].tolist())
        if identities != expected_identities or len(identities) != len(set(identities)):
            raise ValueError(
                "committee representative identities do not match canonical outputs"
            )
        expected_fields = {
            "ids",
            *(f"frame_{index:06d}" for index in range(len(identities))),
        }
        if set(payload.files) != expected_fields:
            raise ValueError("committee representative evidence fields are malformed")
        for index in range(len(identities)):
            values = np.asarray(payload[f"frame_{index:06d}"], dtype=np.float32)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("committee representative evidence is non-finite")


def _is_ordered_unique_subset(
    values: tuple[str, ...], reference: tuple[str, ...]
) -> bool:
    if len(values) != len(set(values)):
        return False
    positions = {identity: index for index, identity in enumerate(reference)}
    try:
        indices = tuple(positions[identity] for identity in values)
    except KeyError:
        return False
    return all(left < right for left, right in zip(indices, indices[1:]))


__all__ = [
    "ScreenArtifactValidationRequest",
    "validate_screen_artifacts",
]


def _valid_empty_screen(request, summary):
    """Validate a canonical zero-population result, not a completion marker."""
    try:
        expected = {'schema', 'schema_revision', 'route', 'system_ids', 'observations',
                    'train', 'fp_candidates', 'screen_policy', 'fp_policy', 'prescreen'}
        contract = str(getattr(request.contract, 'value', request.contract))
        fallback = request.screen_policy == 'ELEMENT_DIRECT' and contract in {
            'single-teacher/committee-student', 'committee-force-deviation'}
        if fallback:
            expected.add('selector_resolution')
            if not _valid_committee_fallback_resolution(request, contract=contract): return False
        if set(summary) != expected or summary['schema_revision'] != 1:
            return False
        route = str(getattr(request.contract, 'value', request.contract))
        routes = {'gmm-reference-tail': 'single-teacher/single-student',
                  'committee-force-deviation': 'committee-teacher/committee-student'}
        if summary['route'] != routes.get(route, route): return False
        if (summary['system_ids'] != list(request.expected_system_ids)
            or summary['screen_policy'] != request.screen_policy or summary['fp_policy'] != request.fp_policy
            or summary['prescreen'] != request.prescreen
            or any(type(summary[k]) is not int or summary[k] != 0 for k in ('observations', 'train', 'fp_candidates'))):
            return False
        if any(read_extxyz_structures(request.screen_dir/name, allow_empty=True)
               for name in ('train.xyz', 'fp_candidate.xyz')): return False
        if any((request.screen_dir/name).exists() for name in (
            'gmm_scores.npz', 'mixed_route_scores.npz', 'committee_force_scores.npz', 'gmm_router')):
            return False
        guard = _load_configured_pair_guard(request.screen_dir, request.pair_guard)
        if guard is not None and len(guard.structure_ids): return False
        return request.require_selection_group_audit and _valid_selection_group_outcomes(request)
    except (OSError, ValueError, KeyError, TypeError):
        return False
