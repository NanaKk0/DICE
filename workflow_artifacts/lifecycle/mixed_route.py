from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_policies.mixed_routes import (
    COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID,
    COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_REVISION,
    SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID,
    SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_REVISION,
)
from workflow_artifacts.identities import RevisionedIdentity
from workflow_policies.identities import (
    STUDENT_DEVIATION_TEACHER_REFERENCE,
    TEACHER_DEVIATION_STUDENT_REFERENCE,
    selector_identity,
)


_SCHEMA_IDENTITY = RevisionedIdentity("dice.lifecycle.mixed-route-audit", 2)
SCHEMA_ID = _SCHEMA_IDENTITY.name
SCHEMA_REVISION = _SCHEMA_IDENTITY.revision
_HISTORICAL_SCHEMA_ID = "v2-mixed-route-audit@1"
_HISTORICAL_COMMITTEE_TEACHER_POLICY = "teacher-force-deviation_student-reference-tail@1"
_STUDENT_FIRST_SCHEMA_REVISION = 3


def publish_student_first_single_teacher_committee_student_audit(
    path: str | Path,
    *,
    observations: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    student_flags: Mapping[str, Mapping[str, bool]],
    train_ids: set[str],
    normal_train_ids: set[str],
    fp_ids: set[str],
    thresholds: Mapping[str, object],
    safe_fill_enabled: bool,
) -> Path:
    frames = list(observations)
    identities = tuple(str(frame.info["structure_id"]) for frame in frames)
    systems = tuple(str(frame.info["system_id"]) for frame in frames)
    if len(identities) != len(set(identities)):
        raise ValueError("student-first mixed audit identities must be unique")
    deviation = np.asarray(
        [float(frame.info["force_uncertainty"]) for frame in frames], dtype=np.float64
    )
    teacher_scored = np.asarray(
        ["teacher_p" in frame.info for frame in frames], dtype=np.bool_
    )
    teacher_p = np.asarray(
        [float(frame.info["teacher_p"]) if "teacher_p" in frame.info else np.nan for frame in frames],
        dtype=np.float64,
    )

    def teacher(identity: str, name: str) -> bool:
        return bool(teacher_flags.get(identity, {}).get(name, False))

    def student(identity: str, name: str) -> bool:
        return bool(student_flags.get(identity, {}).get(name, False))

    payload = {
        "schema": np.asarray(SCHEMA_ID),
        "schema_revision": np.asarray(_STUDENT_FIRST_SCHEMA_REVISION, dtype=np.int64),
        "policy": np.asarray(SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID),
        "policy_revision": np.asarray(SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_REVISION, dtype=np.int64),
        "structure_ids": np.asarray(identities, dtype=np.str_),
        "system_ids": np.asarray(systems, dtype=np.str_),
        "student_force_deviation": deviation,
        "student_force_deviation_applicable": np.ones(len(frames), dtype=np.bool_),
        "guard_pass": np.ones(len(frames), dtype=np.bool_),
        "student_in_band": np.asarray([student(identity, "student_in_band") for identity in identities], dtype=np.bool_),
        "student_pool": np.asarray([student(identity, "student_pool") for identity in identities], dtype=np.bool_),
        "normal_teacher_scored": np.asarray([student(identity, "student_pool") for identity in identities], dtype=np.bool_),
        "teacher_scored": teacher_scored,
        "teacher_p": teacher_p,
        "teacher_p_applicable": teacher_scored.copy(),
        "teacher_floor_discard": np.asarray([teacher(identity, "teacher_floor_discard") for identity in identities], dtype=np.bool_),
        "teacher_fp_pool": np.asarray([teacher(identity, "teacher_fp_candidate") for identity in identities], dtype=np.bool_),
        "teacher_trusted": np.asarray([teacher(identity, "student_eligible") for identity in identities], dtype=np.bool_),
        "normal_train_selected": np.asarray([identity in normal_train_ids for identity in identities], dtype=np.bool_),
        "safe_fill_enabled": np.asarray(safe_fill_enabled, dtype=np.bool_),
        "safe_fill_eligible": np.asarray([student(identity, "safe_fill_eligible") for identity in identities], dtype=np.bool_),
        "safe_fill_attempted": np.asarray([student(identity, "safe_fill_attempted") for identity in identities], dtype=np.bool_),
        "safe_fill_accepted": np.asarray([student(identity, "safe_fill_accepted") for identity in identities], dtype=np.bool_),
        "train_selected": np.asarray([identity in train_ids for identity in identities], dtype=np.bool_),
        "fp_candidate": np.asarray([identity in fp_ids for identity in identities], dtype=np.bool_),
        "thresholds": np.asarray(json.dumps(dict(thresholds), sort_keys=True, separators=(",", ":"))),
    }
    output = Path(path)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    load_mixed_route_audit(output)
    return output


def publish_committee_teacher_single_student_audit(
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
        raise ValueError("mixed-route audit identities must be unique")
    systems = tuple(str(frame.info["system_id"]) for frame in frames)
    student_p = np.asarray([float(frame.info["student_p"]) for frame in frames], dtype=np.float64)
    if not np.isfinite(student_p).all() or np.any((student_p < 0) | (student_p > 1)):
        raise ValueError("mixed-route student p evidence must be finite in [0, 1]")
    raw_evaluated_ids = tuple(str(frame.info["structure_id"]) for frame in teacher_evaluated)
    raw_deviations = tuple(
        float(frame.info["force_uncertainty"]) for frame in teacher_evaluated
    )
    if len(raw_evaluated_ids) != len(set(raw_evaluated_ids)):
        raise ValueError("mixed-route teacher evaluation identities must be unique")
    raw_deviation_by_id = dict(zip(raw_evaluated_ids, raw_deviations, strict=True))
    evaluated_ids = tuple(identity for identity in identities if identity in raw_deviation_by_id)
    deviations = np.asarray(
        [raw_deviation_by_id[identity] for identity in evaluated_ids], dtype=np.float64
    )
    if not np.isfinite(deviations).all() or np.any(deviations < 0):
        raise ValueError("mixed-route teacher force deviation must be finite and non-negative")
    deviation_by_id = dict(zip(evaluated_ids, deviations.tolist(), strict=True))
    normal_evaluated_ids = {
        str(frame.info["structure_id"]) for frame in normal_teacher_evaluated
    }

    def student_flag(identity: str, name: str) -> bool:
        return bool(student_flags.get(identity, {}).get(name, False))

    applicable = np.asarray([identity in deviation_by_id for identity in identities], dtype=np.bool_)
    payload = {
        "schema": np.asarray(SCHEMA_ID),
        "schema_revision": np.asarray(SCHEMA_REVISION, dtype=np.int64),
        "policy": np.asarray(COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID),
        "policy_revision": np.asarray(
            COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_REVISION, dtype=np.int64
        ),
        "structure_ids": np.asarray(identities, dtype=np.str_),
        "system_ids": np.asarray(systems, dtype=np.str_),
        "student_p": student_p,
        "student_p_applicable": np.ones(len(frames), dtype=np.bool_),
        "teacher_force_deviation_structure_ids": np.asarray(evaluated_ids, dtype=np.str_),
        "teacher_force_deviation": deviations,
        "teacher_force_deviation_applicable": applicable,
        "guard_pass": np.ones(len(frames), dtype=np.bool_),
        "student_floor_discard": np.asarray(
            [student_flag(identity, "student_floor_discard") for identity in identities], dtype=np.bool_
        ),
        "student_candidate": np.asarray(
            [student_flag(identity, "student_candidate") for identity in identities], dtype=np.bool_
        ),
        "student_pool": np.asarray(
            [student_flag(identity, "student_pool") for identity in identities], dtype=np.bool_
        ),
        "normal_teacher_evaluated": np.asarray(
            [identity in normal_evaluated_ids for identity in identities], dtype=np.bool_
        ),
        "teacher_evaluated": applicable.copy(),
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
            [identity in train_ids and student_flag(identity, "safe_fill_attempted") for identity in identities], dtype=np.bool_
        ),
        "train_selected": np.asarray([identity in train_ids for identity in identities], dtype=np.bool_),
        "fp_candidate": np.asarray([identity in fp_ids for identity in identities], dtype=np.bool_),
        "thresholds": np.asarray(json.dumps(dict(thresholds), sort_keys=True, separators=(",", ":"))),
    }
    output = Path(path)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return output


def load_mixed_route_audit(
    path: str | Path,
) -> dict[str, np.ndarray | str | dict[str, object]]:
    try:
        with np.load(Path(path), allow_pickle=False) as probe:
            raw_policy = str(probe["policy"].item())
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("mixed-route audit is malformed") from exc
    if raw_policy == SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID:
        return _load_student_first_single_teacher_committee_student_audit(path)
    return _load_committee_teacher_single_student_audit(path)


def _load_committee_teacher_single_student_audit(
    path: str | Path,
) -> dict[str, np.ndarray | str | dict[str, object]]:
    required = {
        "schema", "schema_revision", "policy", "policy_revision",
        "structure_ids", "system_ids", "student_p",
        "student_p_applicable", "teacher_force_deviation_structure_ids",
        "teacher_force_deviation", "teacher_force_deviation_applicable", "guard_pass",
        "student_floor_discard", "student_candidate", "student_pool",
        "normal_teacher_evaluated", "teacher_evaluated", "teacher_low",
        "teacher_middle", "teacher_high", "normal_train_selected",
        "safe_fill_enabled", "safe_fill_eligible", "safe_fill_attempted",
        "safe_fill_accepted", "train_selected", "fp_candidate", "thresholds",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            raw_policy = str(payload["policy"].item()) if "policy" in payload.files else ""
            historical = raw_policy == _HISTORICAL_COMMITTEE_TEACHER_POLICY
            expected = required - {"schema_revision", "policy_revision"} if historical else required
            if set(payload.files) != expected:
                raise ValueError("committee-teacher mixed audit fields are malformed")
            result = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("committee-teacher mixed audit is malformed") from exc
    if historical:
        if str(result["schema"].item()) != _HISTORICAL_SCHEMA_ID:
            raise ValueError("mixed-route audit schema is unsupported")
        result["schema"] = np.asarray(SCHEMA_ID)
        result["schema_revision"] = np.asarray(SCHEMA_REVISION, dtype=np.int64)
        result["policy"] = np.asarray(COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID)
        result["policy_revision"] = np.asarray(
            COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_REVISION, dtype=np.int64
        )
    _validate_mixed_route_array_schema(
        result, COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID
    )
    if (
        str(result["schema"].item()) != SCHEMA_ID
        or int(result["schema_revision"].item()) != SCHEMA_REVISION
        or str(result["policy"].item()) != COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID
        or int(result["policy_revision"].item())
        != COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_REVISION
    ):
        raise ValueError("committee-teacher mixed audit schema is unsupported")
    identities = np.asarray(result["structure_ids"], dtype=np.str_)
    systems = np.asarray(result["system_ids"], dtype=np.str_)
    count = len(identities)
    if count != len(set(identities.tolist())) or len(systems) != count:
        raise ValueError("mixed-route audit identity rows are malformed")
    compact_fields = {
        "teacher_force_deviation_structure_ids", "teacher_force_deviation"
    }
    for name, values in result.items():
        if name in {
            "schema", "schema_revision", "policy", "policy_revision",
            "thresholds", "safe_fill_enabled",
        } or name in compact_fields:
            continue
        if len(np.asarray(values)) != count:
            raise ValueError("mixed-route audit arrays are not identity aligned")
    probabilities = np.asarray(result["student_p"], dtype=float)
    deviations = np.asarray(result["teacher_force_deviation"], dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.isfinite(deviations).all()
        or np.any(deviations < 0)
    ):
        raise ValueError("mixed-route numerical evidence is invalid")
    if not np.asarray(result["student_p_applicable"], dtype=bool).all():
        raise ValueError("single-student mixed audit is missing applicable student p")
    evaluated_ids = tuple(
        str(item)
        for item in np.asarray(
            result["teacher_force_deviation_structure_ids"]
        ).tolist()
    )
    if len(evaluated_ids) != len(deviations) or len(evaluated_ids) != len(set(evaluated_ids)):
        raise ValueError("committee-teacher deviation rows are malformed")
    applicable_ids = tuple(
        identity
        for identity, applicable in zip(
            identities.tolist(),
            np.asarray(result["teacher_force_deviation_applicable"], dtype=bool),
            strict=True,
        )
        if bool(applicable)
    )
    if evaluated_ids != applicable_ids:
        raise ValueError("committee-teacher deviation applicability is misaligned")
    try:
        thresholds = json.loads(str(result["thresholds"].item()))
    except json.JSONDecodeError as exc:
        raise ValueError("mixed-route audit thresholds are malformed") from exc
    if not isinstance(thresholds, dict):
        raise ValueError("mixed-route audit thresholds must be an object")
    if historical and thresholds.get("selector") in {"DIRECT", "FPS", "random"}:
        thresholds = dict(thresholds)
        thresholds["selector"] = selector_identity(
            str(thresholds["selector"])
        ).as_dict()
    _validate_committee_teacher_single_student_evidence(result, thresholds)
    result["thresholds"] = thresholds
    return result


def _load_student_first_single_teacher_committee_student_audit(
    path: str | Path,
) -> dict[str, np.ndarray | str | dict[str, object]]:
    required = {
        "schema", "schema_revision", "policy", "policy_revision", "structure_ids",
        "system_ids", "student_force_deviation", "student_force_deviation_applicable",
        "guard_pass", "student_in_band", "student_pool", "normal_teacher_scored",
        "teacher_scored", "teacher_p", "teacher_p_applicable", "teacher_floor_discard",
        "teacher_fp_pool", "teacher_trusted", "normal_train_selected",
        "safe_fill_enabled", "safe_fill_eligible", "safe_fill_attempted",
        "safe_fill_accepted", "train_selected", "fp_candidate", "thresholds",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if set(payload.files) != required:
                raise ValueError("student-first mixed audit fields are malformed")
            result = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("student-first mixed audit is malformed") from exc
    if (
        str(result["schema"].item()) != SCHEMA_ID
        or int(result["schema_revision"].item()) != _STUDENT_FIRST_SCHEMA_REVISION
        or str(result["policy"].item()) != SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID
        or int(result["policy_revision"].item()) != SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_REVISION
    ):
        raise ValueError("student-first mixed audit schema is unsupported")
    identities = np.asarray(result["structure_ids"])
    systems = np.asarray(result["system_ids"])
    count = len(identities)
    if identities.ndim != 1 or identities.dtype.kind != "U" or len(set(identities.tolist())) != count or systems.shape != (count,) or systems.dtype.kind != "U":
        raise ValueError("student-first mixed audit identities are invalid")
    scalar_names = {"schema", "schema_revision", "policy", "policy_revision", "safe_fill_enabled", "thresholds"}
    numeric_names = {"student_force_deviation", "teacher_p"}
    for name, value in result.items():
        if name in scalar_names or name in numeric_names or name in {"structure_ids", "system_ids"}:
            continue
        if value.shape != (count,) or value.dtype != np.bool_:
            raise ValueError("student-first mixed audit flags are invalid")
    deviation = np.asarray(result["student_force_deviation"], dtype=np.float64)
    teacher_p = np.asarray(result["teacher_p"], dtype=np.float64)
    teacher_scored = np.asarray(result["teacher_scored"], dtype=bool)
    if deviation.shape != (count,) or not np.isfinite(deviation).all() or np.any(deviation < 0):
        raise ValueError("student-first mixed audit deviation is invalid")
    if teacher_p.shape != (count,) or np.any(~np.isfinite(teacher_p[teacher_scored])) or np.any((teacher_p[teacher_scored] < 0) | (teacher_p[teacher_scored] > 1)) or np.any(~np.isnan(teacher_p[~teacher_scored])):
        raise ValueError("student-first mixed audit teacher applicability is invalid")
    if not np.asarray(result["student_force_deviation_applicable"], dtype=bool).all() or not np.asarray(result["guard_pass"], dtype=bool).all() or not np.array_equal(np.asarray(result["teacher_p_applicable"], dtype=bool), teacher_scored):
        raise ValueError("student-first mixed audit applicability is inconsistent")
    try:
        thresholds = json.loads(str(result["thresholds"].item()))
    except json.JSONDecodeError as exc:
        raise ValueError("student-first mixed audit thresholds are malformed") from exc
    if not isinstance(thresholds, dict):
        raise ValueError("student-first mixed audit thresholds are malformed")
    _validate_student_first_single_teacher_committee_student_evidence(result, thresholds)
    result["thresholds"] = thresholds
    return result


def _validate_student_first_single_teacher_committee_student_evidence(
    audit: Mapping[str, object], thresholds: Mapping[str, object]
) -> None:
    try:
        if thresholds["routing_policy"] != {
            "name": SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_ID,
            "revision": SINGLE_TEACHER_COMMITTEE_STUDENT_STUDENT_FIRST_POLICY_REVISION,
        }:
            raise ValueError
        teacher_band = thresholds["teacher_p"]
        student_band = thresholds["student_force_deviation"]
        quota = thresholds["quota"]
        if not all(isinstance(item, Mapping) for item in (teacher_band, student_band, quota)):
            raise TypeError
        floor = float(teacher_band["floor"])
        trust = float(teacher_band["trust"])
        ceiling = teacher_band.get("ceiling")
        ceiling_value = None if ceiling is None else float(ceiling)
        lo = float(student_band["lo"])
        hi = float(student_band["hi"])
        fp_pool = int(quota["fp_pool"])
        pool = int(quota["pool"])
        keep = int(quota["keep"])
        selector = thresholds["selector"]
        safe_enabled = thresholds["fill_underfilled_from_safe_region"]
        if not isinstance(safe_enabled, bool) or not isinstance(selector, Mapping) or selector != selector_identity(str(selector.get("name", ""))).as_dict():
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("student-first mixed audit thresholds are malformed") from exc
    deviation = np.asarray(audit["student_force_deviation"], dtype=float)
    probability = np.asarray(audit["teacher_p"], dtype=float)
    systems = np.asarray(audit["system_ids"])
    flags = {name: np.asarray(audit[name], dtype=bool) for name in (
        "student_in_band", "student_pool", "normal_teacher_scored", "teacher_scored",
        "teacher_floor_discard", "teacher_fp_pool", "teacher_trusted",
        "normal_train_selected", "safe_fill_eligible", "safe_fill_attempted",
        "safe_fill_accepted", "train_selected", "fp_candidate",
    )}
    expected_band = (deviation >= lo) & (deviation < hi)
    expected_pool = np.zeros(len(deviation), dtype=bool)
    identities = np.asarray(audit["structure_ids"], dtype=np.str_)
    for system_id in set(str(item) for item in systems.tolist()):
        candidates = [
            index
            for index in range(len(deviation))
            if str(systems[index]) == system_id and bool(expected_band[index])
        ]
        candidates.sort(key=lambda index: (-deviation[index], str(identities[index])))
        chosen = candidates if pool == -1 else candidates[:pool]
        expected_pool[chosen] = True
    if not np.array_equal(flags["student_in_band"], expected_band) or not np.array_equal(flags["student_pool"], expected_pool) or not np.array_equal(flags["normal_teacher_scored"], flags["student_pool"]):
        raise ValueError("student-first mixed proposal evidence is inconsistent")
    expected_safe = safe_enabled & (deviation < lo)
    if bool(np.asarray(audit["safe_fill_enabled"]).item()) != safe_enabled or not np.array_equal(flags["safe_fill_eligible"], expected_safe) or np.any(flags["safe_fill_attempted"] & ~expected_safe) or np.any(flags["safe_fill_attempted"] & flags["student_pool"]):
        raise ValueError("student-first mixed safe-fill evidence is inconsistent")
    if not np.array_equal(flags["teacher_scored"], flags["normal_teacher_scored"] | flags["safe_fill_attempted"]):
        raise ValueError("student-first mixed teacher applicability is inconsistent")
    floor_discard = flags["teacher_scored"] & (probability < floor)
    trusted = flags["teacher_scored"] & ~floor_discard & (probability > trust)
    if not np.array_equal(flags["teacher_floor_discard"], floor_discard) or not np.array_equal(flags["teacher_trusted"], trusted):
        raise ValueError("student-first mixed teacher trust evidence is inconsistent")
    teacher_upper = probability < ceiling_value if ceiling_value is not None else probability <= 1
    if np.any(flags["fp_candidate"] & (~flags["normal_teacher_scored"] | floor_discard | ~teacher_upper)) or not np.array_equal(flags["fp_candidate"], flags["teacher_fp_pool"]):
        raise ValueError("student-first mixed FP evidence is inconsistent")
    if np.any(flags["normal_train_selected"] & (~flags["normal_teacher_scored"] | ~trusted)) or not np.array_equal(flags["safe_fill_accepted"], flags["safe_fill_attempted"] & trusted) or not np.array_equal(flags["train_selected"], flags["normal_train_selected"] | flags["safe_fill_accepted"]) or np.any(flags["safe_fill_accepted"] & flags["fp_candidate"]):
        raise ValueError("student-first mixed train evidence is inconsistent")
    for system_id in set(str(item) for item in systems.tolist()):
        owned = systems == system_id
        normal_candidates = int(np.count_nonzero(flags["normal_teacher_scored"] & ~floor_discard & teacher_upper & owned))
        expected_fp_count = normal_candidates if fp_pool == -1 else min(normal_candidates, fp_pool)
        if int(np.count_nonzero(flags["fp_candidate"] & owned)) != expected_fp_count:
            raise ValueError("student-first mixed FP quota is inconsistent")
        trusted_normal = int(np.count_nonzero(flags["normal_teacher_scored"] & trusted & owned))
        if int(np.count_nonzero(flags["normal_train_selected"] & owned)) != min(trusted_normal, keep):
            raise ValueError("student-first mixed selector count is inconsistent")
        shortfall = max(0, keep - int(np.count_nonzero(flags["normal_train_selected"] & owned)))
        expected_fill_count = min(shortfall, int(np.count_nonzero(expected_safe & owned)))
        if int(np.count_nonzero(flags["safe_fill_attempted"] & owned)) != expected_fill_count or int(np.count_nonzero(flags["train_selected"] & owned)) > keep:
            raise ValueError("student-first mixed safe-fill or train quota is inconsistent")


def _validate_mixed_route_array_schema(
    audit: Mapping[str, np.ndarray], policy: str
) -> None:
    for name in ("schema", "policy", "thresholds"):
        value = np.asarray(audit[name])
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError(f"mixed-route audit {name} must be one Unicode scalar")
    for name in ("schema_revision", "policy_revision"):
        value = np.asarray(audit[name])
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError(f"mixed-route audit {name} must be one integer scalar")
    value = np.asarray(audit["safe_fill_enabled"])
    if value.shape != () or value.dtype != np.dtype(np.bool_):
        raise ValueError("mixed-route audit safe_fill_enabled must be one boolean scalar")
    identity_fields = ["structure_ids", "system_ids"]
    if policy == COMMITTEE_TEACHER_SINGLE_STUDENT_POLICY_ID:
        identity_fields.append("teacher_force_deviation_structure_ids")
    for name in identity_fields:
        value = np.asarray(audit[name])
        if value.ndim != 1 or value.dtype.kind != "U":
            raise ValueError(f"mixed-route audit {name} must be a one-dimensional Unicode array")
        if any(not str(item) for item in value.tolist()):
            raise ValueError(f"mixed-route audit {name} contains an empty identity")
    numerical_fields = ("student_p", "teacher_force_deviation")
    for name in numerical_fields:
        value = np.asarray(audit[name])
        if value.ndim != 1 or value.dtype.kind != "f":
            raise ValueError(f"mixed-route audit {name} must be a one-dimensional floating array")
    structural = {
        "schema", "schema_revision", "policy", "policy_revision", "thresholds",
        "safe_fill_enabled",
        *identity_fields, *numerical_fields,
    }
    for name, raw in audit.items():
        if name in structural:
            continue
        value = np.asarray(raw)
        if value.ndim != 1 or value.dtype != np.dtype(np.bool_):
            raise ValueError(f"mixed-route audit {name} must be a one-dimensional boolean array")
    if not np.asarray(audit["guard_pass"], dtype=bool).all():
        raise ValueError("partial-pass mixed audit guard flags must all pass")




def _validate_committee_teacher_single_student_evidence(
    audit: Mapping[str, object], thresholds: Mapping[str, object]
) -> None:
    try:
        student_band = thresholds["student_p"]
        teacher_band = thresholds["teacher_force_deviation"]
        quota = thresholds["quota"]
        if not all(isinstance(item, Mapping) for item in (student_band, teacher_band, quota)):
            raise TypeError
        floor = float(student_band["floor"])
        ceiling = student_band.get("ceiling")
        ceiling_value = None if ceiling is None else float(ceiling)
        lo = float(teacher_band["lo"])
        hi = float(teacher_band["hi"])
        pool = int(quota["pool"])
        keep = int(quota["keep"])
        selector = thresholds["selector"]
        if not isinstance(selector, Mapping) or selector != selector_identity(
            str(selector.get("name", ""))
        ).as_dict():
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("committee-teacher mixed audit thresholds are malformed") from exc
    probability = np.asarray(audit["student_p"], dtype=float)
    floor_discard = probability < floor
    candidate = (probability >= floor) & (
        True if ceiling_value is None else probability < ceiling_value
    )
    flags = {
        name: np.asarray(audit[name], dtype=bool)
        for name in (
            "student_floor_discard",
            "student_candidate",
            "student_pool",
            "normal_teacher_evaluated",
            "teacher_evaluated",
            "teacher_low",
            "teacher_middle",
            "teacher_high",
            "normal_train_selected",
            "safe_fill_eligible",
            "safe_fill_attempted",
            "safe_fill_accepted",
            "train_selected",
            "fp_candidate",
        )
    }
    if not np.array_equal(flags["student_floor_discard"], floor_discard):
        raise ValueError("single-student mixed audit floor flags are inconsistent")
    if not np.array_equal(flags["student_candidate"], candidate):
        raise ValueError("single-student mixed audit candidate flags are inconsistent")
    if np.any(flags["student_pool"] & ~candidate):
        raise ValueError("single-student mixed audit pool flags are inconsistent")
    if not np.array_equal(flags["normal_teacher_evaluated"], flags["student_pool"]):
        raise ValueError("committee-teacher mixed audit normal evaluation flags are inconsistent")
    evaluated_ids = tuple(
        str(item)
        for item in np.asarray(audit["teacher_force_deviation_structure_ids"]).tolist()
    )
    deviation_by_id = dict(
        zip(
            evaluated_ids,
            np.asarray(audit["teacher_force_deviation"], dtype=float).tolist(),
            strict=True,
        )
    )
    identities = tuple(str(item) for item in np.asarray(audit["structure_ids"]).tolist())
    low = np.asarray(
        [identity in deviation_by_id and deviation_by_id[identity] < lo for identity in identities],
        dtype=bool,
    )
    middle = np.asarray(
        [
            identity in deviation_by_id and lo <= deviation_by_id[identity] < hi
            for identity in identities
        ],
        dtype=bool,
    )
    high = np.asarray(
        [identity in deviation_by_id and deviation_by_id[identity] >= hi for identity in identities],
        dtype=bool,
    )
    if not np.array_equal(flags["teacher_low"], low):
        raise ValueError("committee-teacher mixed audit low flags are inconsistent")
    if not np.array_equal(flags["teacher_middle"], middle):
        raise ValueError("committee-teacher mixed audit middle flags are inconsistent")
    if not np.array_equal(flags["teacher_high"], high):
        raise ValueError("committee-teacher mixed audit high flags are inconsistent")
    covered = (
        np.zeros(probability.shape, dtype=bool)
        if ceiling_value is None
        else probability >= ceiling_value
    )
    safe_eligible = bool(np.asarray(audit["safe_fill_enabled"]).item()) & covered
    if not np.array_equal(flags["safe_fill_eligible"], safe_eligible):
        raise ValueError("committee-teacher mixed audit safe-fill eligibility is inconsistent")
    if (
        np.any(flags["safe_fill_attempted"] & ~flags["safe_fill_eligible"])
        or np.any(flags["normal_train_selected"] & ~flags["normal_teacher_evaluated"])
        or np.any(flags["normal_train_selected"] & ~low)
        or np.any(flags["safe_fill_accepted"] & ~flags["safe_fill_attempted"])
        or np.any(flags["safe_fill_accepted"] & ~low)
        or not np.array_equal(
            flags["teacher_evaluated"],
            flags["normal_teacher_evaluated"] | flags["safe_fill_attempted"],
        )
        or not np.array_equal(
            flags["train_selected"],
            flags["normal_train_selected"] | flags["safe_fill_accepted"],
        )
    ):
        raise ValueError("committee-teacher mixed audit train flags are inconsistent")
    if not np.array_equal(
        flags["fp_candidate"], middle & flags["normal_teacher_evaluated"]
    ):
        raise ValueError("committee-teacher mixed audit FP flags are inconsistent")
    systems = np.asarray(audit["system_ids"], dtype=np.str_)
    _validate_per_system_selection_counts(
        systems=systems,
        eligible=candidate,
        selected=flags["student_pool"],
        quota=pool,
        label="single-student pool",
    )
    _validate_per_system_selection_counts(
        systems=systems,
        eligible=low & flags["normal_teacher_evaluated"],
        selected=flags["normal_train_selected"],
        quota=keep,
        label="committee-teacher train",
    )


def _validate_per_system_selection_counts(
    *,
    systems: np.ndarray,
    eligible: np.ndarray,
    selected: np.ndarray,
    quota: int,
    label: str,
) -> None:
    if quota == 0 or quota < -1:
        raise ValueError(f"{label} quota is malformed")
    for system_id in set(str(item) for item in systems.tolist()):
        owned = systems == system_id
        expected = int(np.count_nonzero(eligible & owned))
        if quota > 0:
            expected = min(expected, quota)
        actual = int(np.count_nonzero(selected & owned))
        if actual != expected:
            raise ValueError(f"{label} count is inconsistent for system {system_id!r}")


def mixed_route_summary(
    *,
    observations: Sequence[Atoms],
    catalog_observations: Sequence[Atoms] | None = None,
    teacher_flags: Mapping[str, Mapping[str, bool]],
    student_flags: Mapping[str, Mapping[str, bool]],
    train_ids: set[str],
    normal_train_ids: set[str],
    fp_ids: set[str],
    thresholds: Mapping[str, object],
    keep: int,
    fill_underfilled_from_safe_region: bool,
    normal_teacher_ids: set[str] | None = None,
    teacher_scored_ids: set[str] | None = None,
) -> dict[str, object]:
    frames = list(observations)
    catalog = frames if catalog_observations is None else list(catalog_observations)
    pass_ids = {str(frame.info["structure_id"]) for frame in frames}
    systems = sorted({str(frame.info["system_id"]) for frame in catalog})
    normal_teacher = pass_ids if normal_teacher_ids is None else normal_teacher_ids
    teacher_scored = pass_ids if teacher_scored_ids is None else teacher_scored_ids

    def counts(owned_catalog: Sequence[Atoms]) -> dict[str, int]:
        identities = [str(frame.info["structure_id"]) for frame in owned_catalog]
        passed = [identity for identity in identities if identity in pass_ids]
        requested = keep * len(
            {str(frame.info["system_id"]) for frame in owned_catalog}
        )
        result = {
            "observations": len(identities),
            "guard_pass": len(passed),
            "guard_reject": len(identities) - len(passed),
            "teacher_scored": sum(identity in teacher_scored for identity in identities),
            "teacher_floor_discard": sum(
                bool(teacher_flags.get(identity, {}).get("teacher_floor_discard"))
                for identity in identities
            ),
            "teacher_fp_pool": sum(identity in fp_ids for identity in identities),
            "teacher_trusted": sum(
                bool(teacher_flags.get(identity, {}).get("student_eligible"))
                for identity in identities
            ),
            "student_in_band": sum(
                bool(student_flags.get(identity, {}).get("student_in_band"))
                for identity in identities
            ),
            "student_pool": sum(
                bool(student_flags.get(identity, {}).get("student_pool"))
                for identity in identities
            ),
            "normal_train": sum(identity in normal_train_ids for identity in identities),
            "safe_fill_eligible": sum(
                bool(student_flags.get(identity, {}).get("safe_fill_eligible"))
                for identity in identities
            ),
            "safe_fill_attempted": sum(
                bool(student_flags.get(identity, {}).get("safe_fill_attempted"))
                for identity in identities
            ),
            "safe_fill_accepted": sum(
                identity in train_ids
                and bool(student_flags.get(identity, {}).get("safe_fill_attempted"))
                for identity in identities
            ),
            "train_selected": sum(identity in train_ids for identity in identities),
            "train_underfill": max(
                0, requested - sum(identity in train_ids for identity in identities)
            ),
            "fp_candidate": sum(identity in fp_ids for identity in identities),
        }
        result["normal_teacher_scored"] = sum(
            identity in normal_teacher for identity in identities
        )
        return result

    result: dict[str, object] = {
        "schema": "dice.diagnostics.mixed-route-summary",
        "schema_revision": 2,
        "policy_identity": STUDENT_DEVIATION_TEACHER_REFERENCE.as_dict(),
        "directional_policy": STUDENT_DEVIATION_TEACHER_REFERENCE.as_dict(),
        "fill_underfilled_from_safe_region": fill_underfilled_from_safe_region,
        "thresholds": dict(thresholds),
        "counts": counts(catalog),
        "per_system": {
            system_id: counts(
                [frame for frame in catalog if str(frame.info["system_id"]) == system_id]
            )
            for system_id in systems
        },
    }
    return result


def committee_teacher_single_student_summary(
    *,
    observations: Sequence[Atoms],
    catalog_observations: Sequence[Atoms] | None,
    student_flags: Mapping[str, Mapping[str, bool]],
    teacher_evaluated: Sequence[Atoms],
    train_ids: set[str],
    normal_train_ids: set[str],
    fp_ids: set[str],
    keep: int,
    teacher_lo: float,
    teacher_hi: float,
    thresholds: Mapping[str, object],
    fill_underfilled_from_safe_region: bool,
) -> dict[str, object]:
    frames = list(observations)
    catalog = frames if catalog_observations is None else list(catalog_observations)
    pass_ids = {str(frame.info["structure_id"]) for frame in frames}
    deviation_by_id = {
        str(frame.info["structure_id"]): float(frame.info["force_uncertainty"])
        for frame in teacher_evaluated
    }
    systems = sorted({str(frame.info["system_id"]) for frame in catalog})

    def counts(owned: Sequence[Atoms]) -> dict[str, int]:
        identities = [str(frame.info["structure_id"]) for frame in owned]
        passed = [identity for identity in identities if identity in pass_ids]
        low = [
            identity
            for identity in identities
            if identity in deviation_by_id and deviation_by_id[identity] < teacher_lo
        ]
        selected = sum(identity in train_ids for identity in identities)
        return {
            "observations": len(identities),
            "guard_pass": len(passed),
            "guard_reject": len(identities) - len(passed),
            "student_scored": len(passed),
            "student_floor_discard": sum(
                bool(student_flags.get(identity, {}).get("student_floor_discard"))
                for identity in identities
            ),
            "student_candidate": sum(
                bool(student_flags.get(identity, {}).get("student_candidate"))
                for identity in identities
            ),
            "student_pool": sum(
                bool(student_flags.get(identity, {}).get("student_pool"))
                for identity in identities
            ),
            "teacher_evaluated": sum(identity in deviation_by_id for identity in identities),
            "teacher_low": len(low),
            "teacher_middle": sum(
                identity in deviation_by_id
                and teacher_lo <= deviation_by_id[identity] < teacher_hi
                for identity in identities
            ),
            "teacher_high": sum(
                identity in deviation_by_id and deviation_by_id[identity] >= teacher_hi
                for identity in identities
            ),
            "normal_train": sum(identity in normal_train_ids for identity in identities),
            "safe_fill_eligible": sum(
                bool(student_flags.get(identity, {}).get("safe_fill_eligible"))
                for identity in identities
            ),
            "safe_fill_attempted": sum(
                bool(student_flags.get(identity, {}).get("safe_fill_attempted"))
                for identity in identities
            ),
            "safe_fill_accepted": sum(
                identity in train_ids
                and bool(student_flags.get(identity, {}).get("safe_fill_attempted"))
                for identity in identities
            ),
            "train_selected": selected,
            "train_underfill": max(0, keep - selected),
            "fp_candidate": sum(identity in fp_ids for identity in identities),
        }

    return {
        "schema": "dice.diagnostics.mixed-route-summary",
        "schema_revision": 2,
        "policy_identity": TEACHER_DEVIATION_STUDENT_REFERENCE.as_dict(),
        "fill_underfilled_from_safe_region": fill_underfilled_from_safe_region,
        "thresholds": dict(thresholds),
        "counts": counts(catalog),
        "per_system": {
            system_id: counts(
                [frame for frame in catalog if str(frame.info["system_id"]) == system_id]
            )
            for system_id in systems
        },
    }


def validate_committee_teacher_single_student_summary(
    path: str | Path,
) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("committee-teacher mixed summary is invalid") from exc
    required = {
        "schema", "schema_revision", "policy_identity",
        "fill_underfilled_from_safe_region", "thresholds", "counts", "per_system",
    }
    if set(payload) != required:
        raise ValueError("committee-teacher mixed summary fields are malformed")
    if (
        payload["schema"] != "dice.diagnostics.mixed-route-summary"
        or payload["schema_revision"] != 2
        or payload["policy_identity"] != TEACHER_DEVIATION_STUDENT_REFERENCE.as_dict()
        or not isinstance(payload["fill_underfilled_from_safe_region"], bool)
        or not isinstance(payload["thresholds"], dict)
        or not isinstance(payload["per_system"], dict)
    ):
        raise ValueError("committee-teacher mixed summary schema is invalid")
    fields = {
        "observations", "guard_pass", "guard_reject", "student_scored",
        "student_floor_discard", "student_candidate", "student_pool",
        "teacher_evaluated", "teacher_low", "teacher_middle", "teacher_high",
        "normal_train", "safe_fill_eligible", "safe_fill_attempted",
        "safe_fill_accepted", "train_selected", "train_underfill", "fp_candidate",
    }
    for counts in [payload["counts"], *payload["per_system"].values()]:
        if not isinstance(counts, dict) or set(counts) != fields:
            raise ValueError("committee-teacher mixed summary counts are malformed")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("committee-teacher mixed summary counts are invalid")
        if (
            counts["guard_pass"] + counts["guard_reject"] != counts["observations"]
            or counts["normal_train"] + counts["safe_fill_accepted"] != counts["train_selected"]
            or counts["safe_fill_accepted"] > counts["safe_fill_attempted"]
            or counts["safe_fill_attempted"] > counts["safe_fill_eligible"]
        ):
            raise ValueError("committee-teacher mixed summary counts are inconsistent")
    return payload


__all__ = [
    "SCHEMA_ID",
    "committee_teacher_single_student_summary",
    "load_mixed_route_audit",
    "mixed_route_summary",
    "publish_student_first_single_teacher_committee_student_audit",
    "publish_committee_teacher_single_student_audit",
    "validate_committee_teacher_single_student_summary",
]
