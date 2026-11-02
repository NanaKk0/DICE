from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_policies.identities import selector_identity


SCHEMA_ID = "dice.lifecycle.committee-force-screen-audit"
SCHEMA_REVISION = 1
POLICY_ID = "committee-force-deviation"
POLICY_REVISION = 1
SUMMARY_SCHEMA = "dice.diagnostics.committee-force-screen-summary"
SUMMARY_REVISION = 1


def publish_committee_force_audit(
    path: str | Path,
    *,
    observations: Sequence[Atoms],
    student_flags: Mapping[str, Mapping[str, bool]],
    teacher_evaluated: Sequence[Atoms],
    normal_teacher_evaluated: Sequence[Atoms],
    train_ids: set[str],
    normal_train_ids: set[str],
    fp_ids: set[str],
    teacher_lo: float,
    teacher_hi: float,
    thresholds: Mapping[str, object],
    safe_fill_enabled: bool,
) -> Path:
    frames = list(observations)
    identities = tuple(str(frame.info["structure_id"]) for frame in frames)
    if len(identities) != len(set(identities)):
        raise ValueError("committee-force audit identities must be unique")
    systems = tuple(str(frame.info["system_id"]) for frame in frames)
    student_deviation = np.asarray(
        [float(frame.info["force_uncertainty"]) for frame in frames], dtype=np.float64
    )
    if not np.isfinite(student_deviation).all() or np.any(student_deviation < 0):
        raise ValueError("committee-force student deviations must be finite and non-negative")
    evaluated_ids = tuple(str(frame.info["structure_id"]) for frame in teacher_evaluated)
    evaluated_deviation = np.asarray(
        [float(frame.info["force_uncertainty"]) for frame in teacher_evaluated], dtype=np.float64
    )
    if len(evaluated_ids) != len(set(evaluated_ids)):
        raise ValueError("committee-force teacher evaluation identities must be unique")
    if not np.isfinite(evaluated_deviation).all() or np.any(evaluated_deviation < 0):
        raise ValueError("committee-force teacher deviations must be finite and non-negative")
    deviation_by_id = dict(zip(evaluated_ids, evaluated_deviation.tolist(), strict=True))
    if set(deviation_by_id) - set(identities):
        raise ValueError("committee-force teacher evaluation contains an unknown identity")
    normal_evaluated_ids = {
        str(frame.info["structure_id"]) for frame in normal_teacher_evaluated
    }
    if not normal_evaluated_ids <= set(deviation_by_id):
        raise ValueError("committee-force normal evaluation is absent from teacher evidence")

    def student_flag(identity: str, name: str) -> bool:
        return bool(student_flags.get(identity, {}).get(name, False))

    payload = {
        "schema": np.asarray(SCHEMA_ID),
        "schema_revision": np.asarray(SCHEMA_REVISION, dtype=np.int64),
        "policy": np.asarray(POLICY_ID),
        "policy_revision": np.asarray(POLICY_REVISION, dtype=np.int64),
        "structure_ids": np.asarray(identities, dtype=np.str_),
        "system_ids": np.asarray(systems, dtype=np.str_),
        "student_force_deviation": student_deviation,
        "teacher_force_deviation_structure_ids": np.asarray(evaluated_ids, dtype=np.str_),
        "teacher_force_deviation": evaluated_deviation,
        "teacher_force_deviation_applicable": np.asarray(
            [identity in deviation_by_id for identity in identities], dtype=np.bool_
        ),
        "guard_pass": np.ones(len(frames), dtype=np.bool_),
        "student_in_band": np.asarray(
            [student_flag(identity, "student_in_band") for identity in identities], dtype=np.bool_
        ),
        "student_pool": np.asarray(
            [student_flag(identity, "student_pool") for identity in identities], dtype=np.bool_
        ),
        "normal_teacher_evaluated": np.asarray(
            [identity in normal_evaluated_ids for identity in identities], dtype=np.bool_
        ),
        "teacher_evaluated": np.asarray(
            [identity in deviation_by_id for identity in identities], dtype=np.bool_
        ),
        "teacher_low": np.asarray(
            [identity in deviation_by_id and deviation_by_id[identity] < teacher_lo for identity in identities],
            dtype=np.bool_,
        ),
        "teacher_middle": np.asarray(
            [
                identity in deviation_by_id
                and teacher_lo <= deviation_by_id[identity] < teacher_hi
                for identity in identities
            ],
            dtype=np.bool_,
        ),
        "teacher_high": np.asarray(
            [identity in deviation_by_id and deviation_by_id[identity] >= teacher_hi for identity in identities],
            dtype=np.bool_,
        ),
        "normal_train_selected": np.asarray(
            [identity in normal_train_ids for identity in identities], dtype=np.bool_
        ),
        "safe_fill_enabled": np.asarray(safe_fill_enabled, dtype=np.bool_),
        "safe_fill_eligible": np.asarray(
            [student_flag(identity, "safe_fill_eligible") for identity in identities], dtype=np.bool_
        ),
        "safe_fill_attempted": np.asarray(
            [student_flag(identity, "safe_fill_attempted") for identity in identities], dtype=np.bool_
        ),
        "safe_fill_accepted": np.asarray(
            [identity in train_ids and student_flag(identity, "safe_fill_attempted") for identity in identities],
            dtype=np.bool_,
        ),
        "train_selected": np.asarray(
            [identity in train_ids for identity in identities], dtype=np.bool_
        ),
        "fp_candidate": np.asarray(
            [identity in fp_ids for identity in identities], dtype=np.bool_
        ),
        "thresholds": np.asarray(json.dumps(dict(thresholds), sort_keys=True, separators=(",", ":"))),
    }
    output = Path(path)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return output


def load_committee_force_audit(path: str | Path) -> dict[str, object]:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "schema", "schema_revision", "policy", "policy_revision", "structure_ids", "system_ids",
            "student_force_deviation", "teacher_force_deviation_structure_ids", "teacher_force_deviation",
            "teacher_force_deviation_applicable", "guard_pass", "student_in_band", "student_pool",
            "normal_teacher_evaluated", "teacher_evaluated", "teacher_low", "teacher_middle", "teacher_high",
            "normal_train_selected", "safe_fill_enabled", "safe_fill_eligible", "safe_fill_attempted",
            "safe_fill_accepted", "train_selected", "fp_candidate", "thresholds",
        }
        if set(payload.files) != required:
            raise ValueError("committee-force audit fields are malformed")
        result = {name: np.asarray(payload[name]) for name in payload.files}
    _validate_audit_arrays(result)
    if str(result["schema"].item()) != SCHEMA_ID or int(result["schema_revision"].item()) != SCHEMA_REVISION:
        raise ValueError("committee-force audit schema is unsupported")
    if str(result["policy"].item()) != POLICY_ID or int(result["policy_revision"].item()) != POLICY_REVISION:
        raise ValueError("committee-force audit policy is unsupported")
    try:
        thresholds = json.loads(str(result["thresholds"].item()))
    except json.JSONDecodeError as exc:
        raise ValueError("committee-force audit thresholds are malformed") from exc
    if not isinstance(thresholds, dict):
        raise ValueError("committee-force audit thresholds must be an object")
    result["thresholds"] = thresholds
    _validate_evidence(result, thresholds)
    return result


def committee_force_summary(
    *,
    observations: Sequence[Atoms],
    catalog_observations: Sequence[Atoms],
    student_flags: Mapping[str, Mapping[str, bool]],
    teacher_evaluated: Sequence[Atoms],
    train_ids: set[str],
    normal_train_ids: set[str],
    fp_ids: set[str],
    keep: int,
    teacher_lo: float,
    teacher_hi: float,
    thresholds: Mapping[str, object],
    safe_fill_enabled: bool,
) -> dict[str, object]:
    frames = list(observations)
    catalog = list(catalog_observations)
    pass_ids = {str(frame.info["structure_id"]) for frame in frames}
    deviation_by_id = {
        str(frame.info["structure_id"]): float(frame.info["force_uncertainty"])
        for frame in teacher_evaluated
    }

    def counts(owned: Sequence[Atoms]) -> dict[str, int]:
        identities = [str(frame.info["structure_id"]) for frame in owned]
        passed = [identity for identity in identities if identity in pass_ids]
        return {
            "observations": len(identities),
            "guard_pass": len(passed),
            "guard_reject": len(identities) - len(passed),
            "student_in_band": sum(bool(student_flags.get(identity, {}).get("student_in_band")) for identity in identities),
            "student_pool": sum(bool(student_flags.get(identity, {}).get("student_pool")) for identity in identities),
            "normal_teacher_evaluated": sum(
                bool(student_flags.get(identity, {}).get("student_pool")) for identity in identities
            ),
            "teacher_evaluated": sum(identity in deviation_by_id for identity in identities),
            "teacher_low": sum(identity in deviation_by_id and deviation_by_id[identity] < teacher_lo for identity in identities),
            "teacher_middle": sum(
                identity in deviation_by_id and teacher_lo <= deviation_by_id[identity] < teacher_hi
                for identity in identities
            ),
            "teacher_high": sum(identity in deviation_by_id and deviation_by_id[identity] >= teacher_hi for identity in identities),
            "normal_train": sum(identity in normal_train_ids for identity in identities),
            "safe_fill_eligible": sum(bool(student_flags.get(identity, {}).get("safe_fill_eligible")) for identity in identities),
            "safe_fill_attempted": sum(bool(student_flags.get(identity, {}).get("safe_fill_attempted")) for identity in identities),
            "safe_fill_accepted": sum(
                identity in train_ids and bool(student_flags.get(identity, {}).get("safe_fill_attempted"))
                for identity in identities
            ),
            "train_selected": sum(identity in train_ids for identity in identities),
            "train_underfill": max(0, keep - sum(identity in train_ids for identity in identities)),
            "fp_candidate": sum(identity in fp_ids for identity in identities),
        }

    systems = sorted({str(frame.info["system_id"]) for frame in catalog})
    return {
        "schema": SUMMARY_SCHEMA,
        "schema_revision": SUMMARY_REVISION,
        "policy_identity": {"name": POLICY_ID, "revision": POLICY_REVISION},
        "fill_underfilled_from_safe_region": safe_fill_enabled,
        "thresholds": dict(thresholds),
        "counts": counts(catalog),
        "per_system": {
            system_id: counts([frame for frame in catalog if str(frame.info["system_id"]) == system_id])
            for system_id in systems
        },
    }


def validate_committee_force_summary(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("committee-force screen summary is invalid") from exc
    required = {
        "schema", "schema_revision", "policy_identity", "fill_underfilled_from_safe_region",
        "thresholds", "counts", "per_system",
    }
    optional = {"selector_resolution"}
    if set(payload) not in {frozenset(required), frozenset(required | optional)} or payload["schema"] != SUMMARY_SCHEMA or payload["schema_revision"] != SUMMARY_REVISION:
        raise ValueError("committee-force screen summary schema is unsupported")
    if "selector_resolution" in payload and payload["selector_resolution"] != {
        "requested_policy": "ELEMENT_DIRECT",
        "effective_policy": "random",
        "fallback_reason": "committee_descriptor_unavailable",
    }:
        raise ValueError("committee-force selector resolution is invalid")
    if payload["policy_identity"] != {"name": POLICY_ID, "revision": POLICY_REVISION}:
        raise ValueError("committee-force screen summary policy is invalid")
    if not isinstance(payload["fill_underfilled_from_safe_region"], bool) or not isinstance(payload["thresholds"], dict):
        raise ValueError("committee-force screen summary settings are invalid")
    expected_count_fields = {
        "observations", "guard_pass", "guard_reject", "student_in_band", "student_pool",
        "normal_teacher_evaluated", "teacher_evaluated", "teacher_low", "teacher_middle", "teacher_high",
        "normal_train", "safe_fill_eligible", "safe_fill_attempted", "safe_fill_accepted",
        "train_selected", "train_underfill", "fp_candidate",
    }
    for counts in [payload["counts"], *payload["per_system"].values()]:
        if not isinstance(counts, dict) or set(counts) != expected_count_fields:
            raise ValueError("committee-force screen summary counts are malformed")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("committee-force screen summary counts are invalid")
        if (
            counts["guard_pass"] + counts["guard_reject"] != counts["observations"]
            or counts["normal_train"] + counts["safe_fill_accepted"] != counts["train_selected"]
            or counts["safe_fill_accepted"] > counts["safe_fill_attempted"]
            or counts["safe_fill_attempted"] > counts["safe_fill_eligible"]
        ):
            raise ValueError("committee-force screen summary counts are inconsistent")
    return payload


def _validate_audit_arrays(audit: Mapping[str, np.ndarray]) -> None:
    for name in ("schema", "policy", "thresholds"):
        value = np.asarray(audit[name])
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError(f"committee-force audit {name} must be one Unicode scalar")
    for name in ("schema_revision", "policy_revision"):
        value = np.asarray(audit[name])
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError(f"committee-force audit {name} must be one integer scalar")
    value = np.asarray(audit["safe_fill_enabled"])
    if value.shape != () or value.dtype != np.dtype(np.bool_):
        raise ValueError("committee-force audit safe_fill_enabled must be one boolean scalar")
    for name in ("structure_ids", "system_ids", "teacher_force_deviation_structure_ids"):
        value = np.asarray(audit[name])
        if value.ndim != 1 or value.dtype.kind != "U" or any(not str(item) for item in value.tolist()):
            raise ValueError(f"committee-force audit {name} is malformed")
    for name in ("student_force_deviation", "teacher_force_deviation"):
        value = np.asarray(audit[name])
        if value.ndim != 1 or value.dtype.kind != "f" or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"committee-force audit {name} is malformed")
    structural = {
        "schema", "schema_revision", "policy", "policy_revision", "thresholds", "safe_fill_enabled",
        "structure_ids", "system_ids", "student_force_deviation", "teacher_force_deviation_structure_ids",
        "teacher_force_deviation",
    }
    size = len(np.asarray(audit["structure_ids"]))
    for name, raw in audit.items():
        if name in structural:
            continue
        value = np.asarray(raw)
        if value.ndim != 1 or value.dtype != np.dtype(np.bool_) or len(value) != size:
            raise ValueError(f"committee-force audit {name} is malformed")
    if len(np.asarray(audit["system_ids"])) != size or len(set(np.asarray(audit["structure_ids"]).tolist())) != size:
        raise ValueError("committee-force audit identities are malformed")


def _validate_evidence(audit: Mapping[str, object], thresholds: Mapping[str, object]) -> None:
    try:
        student_band = thresholds["student_force_deviation"]
        teacher_band = thresholds["teacher_force_deviation"]
        quota = thresholds["quota"]
        if not all(isinstance(value, Mapping) for value in (student_band, teacher_band, quota)):
            raise TypeError
        student_lo, student_hi = float(student_band["lo"]), float(student_band["hi"])
        teacher_lo, teacher_hi = float(teacher_band["lo"]), float(teacher_band["hi"])
        pool, keep = int(quota["pool"]), int(quota["keep"])
        selector = thresholds["selector"]
        enabled = thresholds["fill_underfilled_from_safe_region"]
        if (
            not 0 <= student_lo < student_hi or not 0 <= teacher_lo < teacher_hi
            or pool == 0 or pool < -1 or keep <= 0
            or not isinstance(enabled, bool)
            or not isinstance(selector, Mapping)
            or selector != selector_identity(str(selector.get("name", ""))).as_dict()
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("committee-force audit thresholds are malformed") from exc
    student = np.asarray(audit["student_force_deviation"], dtype=float)
    systems = np.asarray(audit["system_ids"], dtype=np.str_)
    identities = tuple(str(item) for item in np.asarray(audit["structure_ids"]).tolist())
    evaluated_ids = tuple(str(item) for item in np.asarray(audit["teacher_force_deviation_structure_ids"]).tolist())
    deviations = np.asarray(audit["teacher_force_deviation"], dtype=float)
    if len(evaluated_ids) != len(deviations) or len(evaluated_ids) != len(set(evaluated_ids)):
        raise ValueError("committee-force teacher evidence is malformed")
    deviation_by_id = dict(zip(evaluated_ids, deviations.tolist(), strict=True))
    flags = {name: np.asarray(audit[name], dtype=bool) for name in (
        "teacher_force_deviation_applicable", "guard_pass", "student_in_band", "student_pool",
        "normal_teacher_evaluated", "teacher_evaluated", "teacher_low", "teacher_middle", "teacher_high",
        "normal_train_selected", "safe_fill_eligible", "safe_fill_attempted", "safe_fill_accepted",
        "train_selected", "fp_candidate",
    )}
    if not flags["guard_pass"].all():
        raise ValueError("committee-force partial audit must contain only guard-passing frames")
    in_band = (student >= student_lo) & (student < student_hi)
    applicable = np.asarray([identity in deviation_by_id for identity in identities], dtype=bool)
    low = np.asarray([identity in deviation_by_id and deviation_by_id[identity] < teacher_lo for identity in identities], dtype=bool)
    middle = np.asarray([identity in deviation_by_id and teacher_lo <= deviation_by_id[identity] < teacher_hi for identity in identities], dtype=bool)
    high = np.asarray([identity in deviation_by_id and deviation_by_id[identity] >= teacher_hi for identity in identities], dtype=bool)
    safe_eligible = bool(np.asarray(audit["safe_fill_enabled"]).item()) & (student < student_lo)
    if bool(np.asarray(audit["safe_fill_enabled"]).item()) != enabled:
        raise ValueError("committee-force audit safe-fill setting is inconsistent")
    if not (
        np.array_equal(flags["student_in_band"], in_band)
        and np.array_equal(flags["teacher_force_deviation_applicable"], applicable)
        and np.array_equal(flags["teacher_evaluated"], applicable)
        and np.array_equal(flags["teacher_low"], low)
        and np.array_equal(flags["teacher_middle"], middle)
        and np.array_equal(flags["teacher_high"], high)
        and np.array_equal(flags["safe_fill_eligible"], safe_eligible)
    ):
        raise ValueError("committee-force audit flags are inconsistent")
    if (
        not np.array_equal(flags["student_pool"], flags["normal_teacher_evaluated"])
        or np.any(flags["student_pool"] & ~in_band)
        or np.any(flags["safe_fill_attempted"] & ~flags["safe_fill_eligible"])
        or np.any(flags["normal_train_selected"] & ~flags["normal_teacher_evaluated"])
        or np.any(flags["normal_train_selected"] & ~low)
        or np.any(flags["safe_fill_accepted"] & ~flags["safe_fill_attempted"])
        or np.any(flags["safe_fill_accepted"] & ~low)
        or not np.array_equal(flags["teacher_evaluated"], flags["normal_teacher_evaluated"] | flags["safe_fill_attempted"])
        or not np.array_equal(flags["train_selected"], flags["normal_train_selected"] | flags["safe_fill_accepted"])
        or not np.array_equal(flags["fp_candidate"], middle & flags["normal_teacher_evaluated"])
    ):
        raise ValueError("committee-force audit routing flags are inconsistent")
    for system_id in set(systems.tolist()):
        owned = systems == system_id
        expected = int(np.count_nonzero(in_band & owned))
        if pool > 0:
            expected = min(expected, pool)
        expected = min(expected, keep)
        if int(np.count_nonzero(flags["student_pool"] & owned)) != expected:
            raise ValueError(f"committee-force normal selection is inconsistent for {system_id!r}")


__all__ = [
    "POLICY_ID", "POLICY_REVISION", "SCHEMA_ID", "SCHEMA_REVISION", "SUMMARY_SCHEMA", "SUMMARY_REVISION",
    "committee_force_summary", "load_committee_force_audit", "publish_committee_force_audit",
    "validate_committee_force_summary",
]
