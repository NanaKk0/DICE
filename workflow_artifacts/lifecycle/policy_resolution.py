"""Canonical audit for the declared committee ELEMENT_DIRECT fallback."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import tempfile

from ase import Atoms

from workflow_policies.identities import selector_identity
from workflow_policies.resolution import COMMITTEE_DESCRIPTOR_UNAVAILABLE


SCHEMA = "dice.lifecycle.policy-resolution"
SCHEMA_REVISION = 1


def publish_policy_resolution(
    path: str | Path,
    *,
    coordinate: str,
    requested_policy: str,
    effective_policy: str,
    fallback_reason: str,
    group_ids: Sequence[str],
    candidates: Sequence[Atoms],
    selected: Sequence[Atoms],
    quota: int,
    entropy: str,
) -> Path:
    payload = _payload(
        coordinate=coordinate,
        requested_policy=requested_policy,
        effective_policy=effective_policy,
        fallback_reason=fallback_reason,
        group_ids=group_ids,
        candidates=candidates,
        selected=selected,
        quota=quota,
        entropy=entropy,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def load_policy_resolution(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("policy resolution must be a regular file")
    payload = json.loads(target.read_text(encoding="utf-8"))
    fields = {
        "schema", "schema_revision", "coordinate", "requested_policy",
        "requested_policy_identity", "effective_policy",
        "effective_policy_identity", "fallback_reason", "quota", "entropy",
        "selection_groups", "rows",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("policy resolution fields are invalid")
    groups = tuple(payload["selection_groups"])
    rows = tuple(payload["rows"])
    if (
        payload["schema"] != SCHEMA
        or payload["schema_revision"] != SCHEMA_REVISION
        or payload["coordinate"] not in {"screen.policy", "fp.policy"}
        or payload["requested_policy"] != "ELEMENT_DIRECT"
        or payload["effective_policy"] != "random"
        or payload["fallback_reason"] != COMMITTEE_DESCRIPTOR_UNAVAILABLE
        or payload["requested_policy_identity"]
        != selector_identity("ELEMENT_DIRECT").as_dict()
        or payload["effective_policy_identity"]
        != selector_identity("random").as_dict()
        or isinstance(payload["quota"], bool)
        or not isinstance(payload["quota"], int)
        or payload["quota"] <= 0
        or not isinstance(payload["entropy"], str)
        or not payload["entropy"]
        or any(not isinstance(item, str) or not item for item in groups)
        or len(groups) != len(set(groups))
        or len(rows) != len(groups)
    ):
        raise ValueError("policy resolution header is invalid")
    row_fields = {
        "selection_group_id", "candidate_ids", "selected_ids",
        "candidate_count", "selected_count", "underfill", "warning_emitted",
    }
    for group_id, row in zip(groups, rows, strict=True):
        if not isinstance(row, dict) or set(row) != row_fields:
            raise ValueError("policy resolution row fields are invalid")
        candidate_ids = tuple(row["candidate_ids"])
        selected_ids = tuple(row["selected_ids"])
        if (
            row["selection_group_id"] != group_id
            or len(candidate_ids) != len(set(candidate_ids))
            or len(selected_ids) != len(set(selected_ids))
            or not set(selected_ids).issubset(candidate_ids)
            or row["candidate_count"] != len(candidate_ids)
            or row["selected_count"] != len(selected_ids)
            or row["underfill"]
            != max(0, int(payload["quota"]) - len(selected_ids))
            or row["warning_emitted"] is not True
        ):
            raise ValueError("policy resolution row alignment is invalid")
    return payload


def validate_policy_resolution(
    path: str | Path,
    *,
    coordinate: str,
    group_ids: Sequence[str],
    candidates: Sequence[Atoms],
    selected: Sequence[Atoms],
    quota: int,
    entropy: str,
) -> bool:
    try:
        actual = load_policy_resolution(path)
        expected = _payload(
            coordinate=coordinate,
            requested_policy="ELEMENT_DIRECT",
            effective_policy="random",
            fallback_reason=COMMITTEE_DESCRIPTOR_UNAVAILABLE,
            group_ids=group_ids,
            candidates=candidates,
            selected=selected,
            quota=quota,
            entropy=entropy,
        )
        return actual == expected
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _payload(
    *,
    coordinate: str,
    requested_policy: str,
    effective_policy: str,
    fallback_reason: str,
    group_ids: Sequence[str],
    candidates: Sequence[Atoms],
    selected: Sequence[Atoms],
    quota: int,
    entropy: str,
) -> dict[str, object]:
    if coordinate not in {"screen.policy", "fp.policy"}:
        raise ValueError("policy resolution coordinate is invalid")
    if (
        requested_policy != "ELEMENT_DIRECT"
        or effective_policy != "random"
        or fallback_reason != COMMITTEE_DESCRIPTOR_UNAVAILABLE
    ):
        raise ValueError("policy resolution is not the declared committee fallback")
    groups = tuple(str(item) for item in group_ids)
    if (
        isinstance(quota, bool)
        or not isinstance(quota, int)
        or quota <= 0
        or not entropy
        or any(not item for item in groups)
        or len(groups) != len(set(groups))
    ):
        raise ValueError("policy resolution context is invalid")
    candidate_frames = tuple(candidates)
    selected_frames = tuple(selected)
    candidate_ids = tuple(str(frame.info["structure_id"]) for frame in candidate_frames)
    selected_ids = tuple(str(frame.info["structure_id"]) for frame in selected_frames)
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or len(selected_ids) != len(set(selected_ids))
        or not set(selected_ids).issubset(candidate_ids)
    ):
        raise ValueError("policy resolution membership is invalid")
    rows = []
    for group_id in groups:
        group_candidates = [
            identity
            for identity, frame in zip(candidate_ids, candidate_frames, strict=True)
            if str(frame.info["system_id"]) == group_id
        ]
        group_selected = [
            identity
            for identity, frame in zip(selected_ids, selected_frames, strict=True)
            if str(frame.info["system_id"]) == group_id
        ]
        rows.append(
            {
                "selection_group_id": group_id,
                "candidate_ids": group_candidates,
                "selected_ids": group_selected,
                "candidate_count": len(group_candidates),
                "selected_count": len(group_selected),
                "underfill": max(0, quota - len(group_selected)),
                "warning_emitted": True,
            }
        )
    return {
        "schema": SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "coordinate": coordinate,
        "requested_policy": requested_policy,
        "requested_policy_identity": selector_identity(requested_policy).as_dict(),
        "effective_policy": effective_policy,
        "effective_policy_identity": selector_identity(effective_policy).as_dict(),
        "fallback_reason": fallback_reason,
        "quota": quota,
        "entropy": str(entropy),
        "selection_groups": list(groups),
        "rows": rows,
    }


__all__ = [
    "SCHEMA",
    "SCHEMA_REVISION",
    "load_policy_resolution",
    "publish_policy_resolution",
    "validate_policy_resolution",
]
