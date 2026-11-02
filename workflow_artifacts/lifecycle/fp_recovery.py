"""Bounded read-only validation of the terminal FP publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np

from workflow_artifacts.lifecycle.representative_features import (
    load_frame_mean_representatives,
)
from workflow_core.structures.canonical_identity import validate_canonical_identity
from workflow_core.structures.exact_identity import exact_structure_hash
from workflow_core.structures.formats import read_extxyz_structures
from workflow_policies.final_fp import select_final_fp_frames
from workflow_policies.final_fp import select_final_fp_joint_frames_with_outcomes


def validate_final_fp_artifacts(
    *,
    fp_root: str | Path,
    candidate_paths: Sequence[str | Path],
    facts: object,
    descriptor_schema: Mapping[str, object] | None = None,
    teacher_training_frames: Sequence[object] | None = None,
    teacher_router_path: str | Path | None = None,
    teacher_training_path: str | Path | None = None,
) -> bool:
    """Load required committed policy evidence and replay terminal selection."""

    try:
        needs_descriptor_schema = facts.requires_frame_mean_evidence or (
            facts.effective_policy
            in {"JOINT_DIRECT", "ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"}
        )
        if needs_descriptor_schema:
            if teacher_router_path is not None:
                from workflow_policies.gmm_reference_tail import validate_gmm_router

                manifest = validate_gmm_router(teacher_router_path)
                descriptor_schema = dict(manifest["descriptor_schema"])
            elif descriptor_schema is None:
                return False

        if facts.effective_policy == "ELEMENT_JOINT_DIRECT":
            if teacher_training_path is not None:
                teacher_training_frames = tuple(
                    read_extxyz_structures(
                        teacher_training_path, allow_empty=False
                    )
                )
            elif teacher_training_frames is None:
                return False

        root = _directory(fp_root)
        candidates = tuple(Path(path) for path in candidate_paths)
        expected_rounds = (
            tuple(range(facts.round_count))
            if facts.all_rn
            else (facts.round_count - 1,)
        )
        if len(candidates) != len(expected_rounds):
            return False
        for path, round_index in zip(candidates, expected_rounds, strict=True):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.name != "fp_candidate.xyz"
                or path.parent.name != "screen"
                or path.parent.parent.name != f"r{round_index}"
                or path.parent.parent.parent.name != f"iter{facts.iter_index}"
            ):
                return False

        merged: dict[str, object] = {}
        features: dict[str, np.ndarray] = {}
        for path in candidates:
            frames = read_extxyz_structures(path, allow_empty=True)
            identities = _validate_frames(
                frames, facts.system_ids, selection_plan=facts.selection_plan
            )
            sidecar = path.parent / "representative_features.npz"
            if facts.requires_frame_mean_evidence and frames:
                if descriptor_schema is None or sidecar.is_symlink():
                    return False
                features.update(
                    load_frame_mean_representatives(
                        sidecar,
                        expected_identities=identities,
                        expected_descriptor_schema=descriptor_schema,
                    )
                )
            elif sidecar.exists() or sidecar.is_symlink():
                return False
            for frame in frames:
                merged[str(frame.info["structure_id"])] = frame

        selected_path = root / "selected.xyz"
        if selected_path.is_symlink() or not selected_path.is_file():
            return False
        selected = read_extxyz_structures(selected_path, allow_empty=True)
        selected_ids = _validate_frames(
            selected, facts.system_ids, selection_plan=facts.selection_plan
        )
        if len(selected_ids) != len(set(selected_ids)):
            return False
        group_ids = (
            facts.selection_plan.group_ids
            if facts.require_selection_group_audit
            else facts.system_ids
        )
        effective_policy = facts.effective_policy
        if effective_policy in {"ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
            if not merged:
                if (root / "element_direct_fp.npz").exists():
                    return False
                expected = []
                expected_outcomes = {
                    group_id: _empty_element_outcome(facts.keep)
                    for group_id in group_ids
                }
            else:
                from workflow_artifacts.lifecycle.element_direct_fp import (
                    element_rows_from_fp_payload,
                    load_element_direct_fp_evidence,
                )
                from workflow_policies.final_fp import (
                    select_final_fp_element_frames_with_outcomes,
                    select_final_fp_element_joint_frames_with_outcomes,
                )

                evidence = load_element_direct_fp_evidence(
                    root / "element_direct_fp.npz"
                )
                identities = tuple(
                    str(item) for item in evidence["structure_ids"].tolist()
                )
                background_ids = tuple(
                    str(item) for item in evidence["background_ids"].tolist()
                )
                expected_rounds = (
                    tuple(range(facts.round_count))
                    if facts.all_rn
                    else (facts.round_count - 1,)
                )
                if (
                    identities != tuple(merged)
                    or tuple(str(item) for item in evidence["group_ids"].tolist())
                    != tuple(group_ids)
                    or tuple(int(item) for item in evidence["candidate_rounds"].tolist())
                    != expected_rounds
                    or int(evidence["keep"].item()) != facts.keep
                    or bool(evidence["all_rn"].item()) != facts.all_rn
                    or str(evidence["policy"].item()) != facts.policy
                    or descriptor_schema is None
                    or json.loads(str(evidence["descriptor_schema_json"].item()))
                    != dict(descriptor_schema)
                    or str(evidence["model_identity"].item())
                    != facts.joint_model_identity
                    or str(evidence["dataset_identity"].item())
                    != facts.joint_dataset_identity
                    or tuple(str(item) for item in evidence["type_map"].tolist())
                    != facts.element_type_map
                    or not _presence_matches_frames(
                        tuple(merged.values()),
                        np.asarray(evidence["candidate_presence"], dtype=bool),
                        facts.element_type_map,
                    )
                ):
                    return False
                candidate_rows, _candidate_presence = element_rows_from_fp_payload(
                    evidence, background=False
                )
                candidate_vectors = {
                    element: {
                        identity: np.asarray(rows[element])[0]
                        for identity, rows in zip(
                            identities, candidate_rows, strict=True
                        )
                        if element in rows
                    }
                    for element in facts.element_type_map
                    if any(element in rows for rows in candidate_rows)
                }
                if effective_policy == "ELEMENT_DIRECT":
                    if background_ids or teacher_training_frames is not None:
                        return False
                    expected, expected_outcomes = (
                        select_final_fp_element_frames_with_outcomes(
                            list(merged.values()),
                            keep=facts.keep,
                            candidate_element_vectors=candidate_vectors,
                            type_map=facts.element_type_map,
                            group_ids=group_ids,
                        )
                    )
                else:
                    background = tuple(teacher_training_frames or ())
                    if (
                        not background
                        or background_ids
                        != tuple(
                            str(frame.info["structure_id"])
                            for frame in background
                        )
                        or not _presence_matches_frames(
                            background,
                            np.asarray(evidence["background_presence"], dtype=bool),
                            facts.element_type_map,
                        )
                    ):
                        return False
                    background_rows, _background_presence = (
                        element_rows_from_fp_payload(evidence, background=True)
                    )
                    background_vectors = {
                        element: {
                            identity: np.asarray(rows[element])[0]
                            for identity, rows in zip(
                                background_ids, background_rows, strict=True
                            )
                            if element in rows
                        }
                        for element in facts.element_type_map
                        if any(element in rows for rows in background_rows)
                    }
                    expected, expected_outcomes = (
                        select_final_fp_element_joint_frames_with_outcomes(
                            list(merged.values()),
                            keep=facts.keep,
                            candidate_element_vectors=candidate_vectors,
                            background_frames=background,
                            background_element_vectors=background_vectors,
                            type_map=facts.element_type_map,
                            group_ids=group_ids,
                        )
                    )
                normalized_outcomes = json.loads(
                    json.dumps(expected_outcomes, sort_keys=True)
                )
                if (
                    json.loads(str(evidence["outcomes_json"].item()))
                    != normalized_outcomes
                ):
                    return False
                replay_ids = tuple(
                    str(item) for item in evidence["selected_ids"].tolist()
                )
                expected_ids = tuple(
                    str(frame.info["structure_id"]) for frame in expected
                )
                if (
                    replay_ids != expected_ids
                    or set(replay_ids)
                    != {
                        identity
                        for identity, chosen in zip(
                            identities,
                            np.asarray(evidence["selected"], dtype=bool),
                            strict=True,
                        )
                        if bool(chosen)
                    }
                ):
                    return False
        elif effective_policy == "JOINT_DIRECT":
            if not merged:
                if (root / "joint_direct_fp.npz").exists():
                    return False
                expected = []
                expected_outcomes = {
                    group_id: {
                        "candidate_count": 0,
                        "background_count": 0,
                        "bypass": False,
                        "realized_clusters": 0,
                        "selected_background_count": 0,
                        "selected_candidate_count": 0,
                        "quota_completion_count": 0,
                        "underfill": 0,
                        "underfill_reason": "empty-candidates",
                    }
                    for group_id in group_ids
                }
            else:
                from workflow_artifacts.lifecycle.joint_direct_fp import (
                    load_joint_direct_fp_evidence,
                )

                evidence = load_joint_direct_fp_evidence(
                    root / "joint_direct_fp.npz"
                )
                identities = tuple(
                    str(item) for item in evidence["structure_ids"].tolist()
                )
                if (
                    identities != tuple(merged)
                    or tuple(str(item) for item in evidence["group_ids"].tolist())
                    != tuple(group_ids)
                    or int(evidence["keep"].item()) != facts.keep
                    or bool(evidence["all_rn"].item()) != facts.all_rn
                    or descriptor_schema is None
                    or json.loads(str(evidence["descriptor_schema_json"].item()))
                    != dict(descriptor_schema)
                    or str(evidence["model_identity"].item())
                    != facts.joint_model_identity
                    or str(evidence["dataset_identity"].item())
                    != facts.joint_dataset_identity
                ):
                    return False
                candidate_vectors = dict(
                    zip(identities, evidence["candidate_vectors"], strict=True)
                )
                background_ids = tuple(
                    str(item) for item in evidence["background_ids"].tolist()
                )
                background_vectors = dict(
                    zip(
                        background_ids,
                        evidence["background_vectors"],
                        strict=True,
                    )
                )
                expected, expected_outcomes = select_final_fp_joint_frames_with_outcomes(
                    list(merged.values()),
                    keep=facts.keep,
                    candidate_vectors=candidate_vectors,
                    background_vectors=background_vectors,
                    group_ids=group_ids,
                )
                if json.loads(str(evidence["outcomes_json"].item())) != expected_outcomes:
                    return False
                expected_mask = np.asarray(evidence["selected"], dtype=bool)
                selected_ids = tuple(
                    str(item) for item in evidence["selected_ids"].tolist()
                )
                if set(selected_ids) != {
                    identity
                    for identity, chosen in zip(
                        identities, expected_mask, strict=True
                    )
                    if bool(chosen)
                }:
                    return False
                replay_ids = selected_ids
                if replay_ids != tuple(
                    str(frame.info["structure_id"]) for frame in expected
                ):
                    return False
        else:
            expected = select_final_fp_frames(
                list(merged.values()),
                keep=facts.keep,
                policy=effective_policy,
                entropy=f"iter{facts.iter_index}/fp",
                feature_vectors=features if facts.requires_frame_mean_evidence else None,
                group_ids=(
                    facts.selection_plan.group_ids
                    if facts.require_selection_group_audit
                    else None
                ),
            )
        if selected_ids != tuple(
            str(frame.info["structure_id"]) for frame in expected
        ):
            return False
        for actual, source in zip(selected, expected, strict=True):
            if (
                str(actual.info.get("system_id", ""))
                != str(source.info.get("system_id", ""))
                or tuple(actual.get_chemical_symbols())
                != tuple(source.get_chemical_symbols())
                or exact_structure_hash(actual) != exact_structure_hash(source)
            ):
                return False
        if effective_policy in {"ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
            if not _valid_element_group_outcomes(
                root,
                facts=facts,
                expected_outcomes=expected_outcomes,
            ):
                return False
        elif effective_policy == "JOINT_DIRECT":
            if not _valid_joint_group_outcomes(
                root,
                facts=facts,
                merged=tuple(merged.values()),
                selected=tuple(selected),
                expected_outcomes=expected_outcomes,
            ):
                return False
        elif facts.require_selection_group_audit and not _valid_group_outcomes(
            root,
            facts=facts,
            merged=tuple(merged.values()),
            selected=tuple(selected),
        ):
            return False
        if facts.fallback_reason is not None:
            from workflow_artifacts.lifecycle.policy_resolution import (
                validate_policy_resolution,
            )

            if not validate_policy_resolution(
                root / "policy_resolution.json",
                coordinate="fp.policy",
                group_ids=group_ids,
                candidates=tuple(merged.values()),
                selected=tuple(selected),
                quota=facts.keep,
                entropy=f"iter{facts.iter_index}/fp",
            ) or not _valid_fp_fallback_summary(root, facts=facts, merged=merged, selected=selected):
                return False
        elif (root / "policy_resolution.json").exists() or (
            root / "fp_summary.json"
        ).exists():
            return False
    except (OSError, TypeError, ValueError, KeyError):
        return False
    return True


def _validate_frames(
    frames,
    system_ids: tuple[str, ...],
    *,
    selection_plan=None,
) -> tuple[str, ...]:
    allowed = set(system_ids)
    identities: list[str] = []
    for frame in frames:
        identity = validate_canonical_identity(frame)
        if identity["system_id"] not in allowed:
            raise ValueError("terminal FP frame has an unexpected system")
        if selection_plan is not None:
            mapping = dict(selection_plan.role_to_group)
            role_id = frame.info.get("sampling_role_id")
            group_id = frame.info.get("selection_group_id")
            if (
                role_id not in mapping
                or mapping[role_id] != group_id
                or identity["system_id"] != group_id
            ):
                raise ValueError("terminal FP frame has inconsistent group provenance")
        if (
            not np.isfinite(np.asarray(frame.positions, dtype=float)).all()
            or not np.isfinite(np.asarray(frame.cell.array, dtype=float)).all()
        ):
            raise ValueError("terminal FP frame has non-finite geometry")
        identities.append(identity["structure_id"])
    if len(identities) != len(set(identities)):
        raise ValueError("terminal FP frame identities must be ordered unique")
    return tuple(identities)


def _valid_group_outcomes(root: Path, *, facts, merged, selected) -> bool:
    path = root / "selection_group_outcomes.json"
    if path.is_symlink() or not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema",
        "schema_revision",
        "selection_groups",
        "role_to_group",
        "policy",
        "keep",
        "rows",
    }
    plan = facts.selection_plan
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload["schema"] != "dice.lifecycle.selection-group-fp-outcomes"
        or payload["schema_revision"] != 1
        or tuple(payload["selection_groups"]) != plan.group_ids
        or tuple(tuple(item) for item in payload["role_to_group"])
        != plan.role_to_group
        or payload["policy"] != facts.effective_policy
        or payload["keep"] != facts.keep
    ):
        return False
    rows = tuple(payload["rows"])
    if tuple(row.get("selection_group_id") for row in rows) != plan.group_ids:
        return False
    row_fields = {
        "selection_group_id",
        "eligible_count",
        "selected_count",
        "underfill",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != row_fields:
            return False
        group_id = row["selection_group_id"]
        eligible = sum(
            str(frame.info["system_id"]) == group_id for frame in merged
        )
        selected_count = sum(
            str(frame.info["system_id"]) == group_id for frame in selected
        )
        if (
            row["eligible_count"] != eligible
            or row["selected_count"] != selected_count
            or row["underfill"] != facts.keep - selected_count
            or selected_count > facts.keep
        ):
            return False
    return True


def _valid_joint_group_outcomes(
    root: Path, *, facts, merged, selected, expected_outcomes
) -> bool:
    payload = json.loads(
        (root / "selection_group_outcomes.json").read_text(encoding="utf-8")
    )
    group_ids = (
        facts.selection_plan.group_ids
        if facts.require_selection_group_audit
        else facts.system_ids
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "dice.lifecycle.selection-group-fp-outcomes"
        or payload.get("schema_revision") != 3
        or tuple(payload.get("selection_groups", ())) != tuple(group_ids)
        or payload.get("policy") != "JOINT_DIRECT"
        or payload.get("keep") != facts.keep
    ):
        return False
    rows = tuple(payload.get("rows", ()))
    if tuple(row.get("selection_group_id") for row in rows) != tuple(group_ids):
        return False
    for row in rows:
        group_id = row["selection_group_id"]
        eligible = sum(str(frame.info["system_id"]) == group_id for frame in merged)
        selected_count = sum(
            str(frame.info["system_id"]) == group_id for frame in selected
        )
        if (
            row != {"selection_group_id": group_id, **expected_outcomes[group_id]}
            or row["candidate_count"] != eligible
            or row["selected_candidate_count"] != selected_count
        ):
            return False
    return True


def _valid_element_group_outcomes(root: Path, *, facts, expected_outcomes) -> bool:
    path = root / "selection_group_outcomes.json"
    if path.is_symlink() or not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    group_ids = (
        facts.selection_plan.group_ids
        if facts.require_selection_group_audit
        else facts.system_ids
    )
    rows = tuple(payload.get("rows", ())) if isinstance(payload, dict) else ()
    normalized = json.loads(json.dumps(expected_outcomes, sort_keys=True))
    return bool(
        payload.get("schema") == "dice.lifecycle.selection-group-fp-outcomes"
        and payload.get("schema_revision") == 4
        and tuple(payload.get("selection_groups", ())) == tuple(group_ids)
        and payload.get("policy") == facts.policy
        and payload.get("keep") == facts.keep
        and tuple(row.get("selection_group_id") for row in rows)
        == tuple(group_ids)
        and all(
            row == {"selection_group_id": group_id, **normalized[group_id]}
            for group_id, row in zip(group_ids, rows, strict=True)
        )
    )


def _presence_matches_frames(frames, presence: np.ndarray, type_map) -> bool:
    elements = tuple(type_map or ())
    if presence.shape != (len(frames), len(elements)):
        return False
    allowed = set(elements)
    for row_index, frame in enumerate(frames):
        chemical = set(frame.get_chemical_symbols())
        if not chemical or not chemical.issubset(allowed):
            return False
        if tuple(
            element
            for element_index, element in enumerate(elements)
            if bool(presence[row_index, element_index])
        ) != tuple(element for element in elements if element in chemical):
            return False
    return True


def _empty_element_outcome(keep: int) -> dict[str, object]:
    return {
        "candidate_count": 0,
        "background_count": 0,
        "bypass": False,
        "realized_clusters": 0,
        "selected_background_count": 0,
        "selected_candidate_count": 0,
        "quota_completion_count": 0,
        "underfill": int(keep),
        "underfill_reason": "empty-candidates",
        "active_elements": (),
        "inactive_elements": (),
        "element_dimensions": {},
        "presence_counts": {},
    }


def _valid_fp_fallback_summary(root: Path, *, facts, merged, selected) -> bool:
    path = root / "fp_summary.json"
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {
        "schema": "dice.lifecycle.fp-summary",
        "schema_revision": 1,
        "candidates": len(merged),
        "unique": len(merged),
        "selected": len(selected),
        "selector_resolution": {
            "requested_policy": facts.policy,
            "effective_policy": facts.effective_policy,
            "fallback_reason": facts.fallback_reason,
        },
    }


def _directory(path: str | Path) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_dir():
        raise ValueError("terminal FP root must be a regular directory")
    return value.resolve()


__all__ = ["validate_final_fp_artifacts"]
