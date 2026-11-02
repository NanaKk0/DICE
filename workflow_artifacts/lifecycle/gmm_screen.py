from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_policies.pair_guard import PairGuardThresholds
from workflow_artifacts.identities import RevisionedIdentity
from workflow_policies.identities import (
    GMM_REFERENCE_TAIL,
    PAIR_GUARD,
    selector_identity,
)


POLICY_ID = GMM_REFERENCE_TAIL.name
POLICY_REVISION = GMM_REFERENCE_TAIL.revision
_SCORE_IDENTITY = RevisionedIdentity("dice.lifecycle.gmm-screen-scores", 5)
SCORE_SCHEMA = _SCORE_IDENTITY.name
SCORE_SCHEMA_REVISION = _SCORE_IDENTITY.revision
_SUMMARY_IDENTITY = RevisionedIdentity("dice.lifecycle.gmm-screen-summary", 6)
SUMMARY_SCHEMA = _SUMMARY_IDENTITY.name
SUMMARY_SCHEMA_REVISION = _SUMMARY_IDENTITY.revision
_STUDENT_FIRST_ROUTING_POLICY = RevisionedIdentity(
    "student-reference-tail_teacher-reference-tail", 1
)


def publish_gmm_screen_scores(
    path: str | Path,
    *,
    observations: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_values: Mapping[str, float],
    train_ids: set[str],
    normal_train_ids: set[str] | None = None,
    fp_ids: set[str],
    teacher_p: Mapping[str, float],
    student_p: Mapping[str, float],
    safe_fill_enabled: bool = False,
    routing_policy: Mapping[str, object] | None = None,
    pool: int | None = None,
    keep: int | None = None,
    fp_pool: int | None = None,
    screen_policy: str | None = None,
) -> Path:
    return _publish_student_first_gmm_screen_scores(
        path,
        observations=observations,
        teacher_flags=teacher_flags,
        student_flags=student_flags,
        student_values=student_values,
        train_ids=train_ids,
        normal_train_ids=normal_train_ids,
        fp_ids=fp_ids,
        teacher_p=teacher_p,
        student_p=student_p,
        safe_fill_enabled=safe_fill_enabled,
        routing_policy=routing_policy,
        pool=pool,
        keep=keep,
        fp_pool=fp_pool,
        screen_policy=screen_policy,
    )
def _publish_student_first_gmm_screen_scores(
    path: str | Path,
    *,
    observations: Sequence[Atoms],
    teacher_flags: Mapping[str, Mapping[str, bool]],
    student_flags: Mapping[str, Mapping[str, bool]],
    student_values: Mapping[str, float],
    train_ids: set[str],
    normal_train_ids: set[str] | None,
    fp_ids: set[str],
    teacher_p: Mapping[str, float],
    student_p: Mapping[str, float],
    safe_fill_enabled: bool,
    routing_policy: Mapping[str, object] | None,
    pool: int | None,
    keep: int | None,
    fp_pool: int | None,
    screen_policy: str | None,
) -> Path:
    if routing_policy != _STUDENT_FIRST_ROUTING_POLICY.as_dict():
        raise ValueError("student-first GMM routing policy is invalid")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value == 0 or value < -1
        for value in (pool, fp_pool)
    ) or not isinstance(keep, int) or isinstance(keep, bool) or keep <= 0:
        raise ValueError("student-first GMM quotas are invalid")
    if screen_policy is None:
        raise ValueError("student-first GMM selector is missing")
    selector = selector_identity(screen_policy)
    if normal_train_ids is None:
        normal_train_ids = set(train_ids)
    identities = [str(frame.info["structure_id"]) for frame in observations]
    systems = [str(frame.info["system_id"]) for frame in observations]
    aligned_student_values = getattr(student_values, "align", None)
    if len(set(identities)) != len(identities):
        raise ValueError("student-first GMM identities are incomplete")
    if callable(aligned_student_values):
        student_probability_values, student_value_coverage = aligned_student_values(
            identities, missing=np.nan
        )
        if not np.all(student_value_coverage):
            raise ValueError("student-first GMM identities are incomplete")
    else:
        if set(student_values) != set(identities):
            raise ValueError("student-first GMM identities are incomplete")
        student_probability_values = np.asarray(
            [float(student_values[identity]) for identity in identities], dtype=np.float64
        )
    teacher_floor, teacher_trust, teacher_ceiling, teacher_ceiling_configured = _teacher_thresholds(teacher_p)
    student_floor, student_ceiling, student_ceiling_configured = _student_thresholds(student_p)
    teacher_scored = np.asarray(
        ["teacher_p" in frame.info for frame in observations], dtype=np.bool_
    )
    teacher_probabilities = np.asarray(
        [float(frame.info["teacher_p"]) if "teacher_p" in frame.info else np.nan for frame in observations],
        dtype=np.float64,
    )

    def teacher(name: str) -> np.ndarray:
        return np.asarray(
            [bool(teacher_flags.get(identity, {}).get(name, False)) for identity in identities],
            dtype=np.bool_,
        )

    def student(name: str) -> np.ndarray:
        return np.asarray(
            [bool(student_flags.get(identity, {}).get(name, False)) for identity in identities],
            dtype=np.bool_,
        )

    normal_teacher_scored = student("student_pool")
    safe_fill_attempted = student("safe_fill_attempted") | student("safe_fill_selected")
    safe_fill_accepted = student("safe_fill_accepted")
    arrays = {
        "schema": np.asarray(SCORE_SCHEMA, dtype=np.str_),
        "schema_revision": np.asarray(SCORE_SCHEMA_REVISION, dtype=np.int64),
        "policy": np.asarray(POLICY_ID, dtype=np.str_),
        "policy_revision": np.asarray(POLICY_REVISION, dtype=np.int64),
        "routing_policy": np.asarray(_STUDENT_FIRST_ROUTING_POLICY.name, dtype=np.str_),
        "routing_policy_revision": np.asarray(_STUDENT_FIRST_ROUTING_POLICY.revision, dtype=np.int64),
        "selector": np.asarray(selector.name, dtype=np.str_),
        "selector_revision": np.asarray(selector.revision, dtype=np.int64),
        "structure_ids": np.asarray(identities, dtype=np.str_),
        "system_ids": np.asarray(systems, dtype=np.str_),
        "pool": np.asarray(pool, dtype=np.int64),
        "keep": np.asarray(keep, dtype=np.int64),
        "fp_pool": np.asarray(fp_pool, dtype=np.int64),
        "teacher_floor": np.asarray(teacher_floor, dtype=np.float64),
        "teacher_trust": np.asarray(teacher_trust, dtype=np.float64),
        "teacher_ceiling": np.asarray(teacher_ceiling, dtype=np.float64),
        "teacher_ceiling_configured": np.asarray(teacher_ceiling_configured, dtype=np.bool_),
        "student_floor": np.asarray(student_floor, dtype=np.float64),
        "student_ceiling": np.asarray(student_ceiling, dtype=np.float64),
        "student_ceiling_configured": np.asarray(student_ceiling_configured, dtype=np.bool_),
        "student_p": student_probability_values,
        "student_scored": np.ones(len(identities), dtype=np.bool_),
        "student_floor_discard": student("student_floor_discard"),
        "student_p_pool": normal_teacher_scored,
        "normal_teacher_scored": normal_teacher_scored,
        "teacher_scored": teacher_scored,
        "teacher_p": teacher_probabilities,
        "teacher_floor_discard": teacher("teacher_floor_discard"),
        "teacher_fp_candidate": teacher("teacher_fp_candidate"),
        "student_eligible": teacher("student_eligible"),
        "teacher_trust_reject": teacher("teacher_trust_reject"),
        "normal_train_selected": np.asarray([identity in normal_train_ids for identity in identities], dtype=np.bool_),
        "safe_fill_enabled": np.asarray(safe_fill_enabled, dtype=np.bool_),
        "safe_fill_eligible": student("safe_fill_eligible"),
        "safe_fill_attempted": safe_fill_attempted,
        "safe_fill_accepted": safe_fill_accepted,
        "train_selected": np.asarray([identity in train_ids for identity in identities], dtype=np.bool_),
        "fp_candidate": np.asarray([identity in fp_ids for identity in identities], dtype=np.bool_),
    }
    target = Path(path)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    validate_gmm_screen_scores(target)
    return target


def validate_gmm_screen_scores(path: str | Path) -> dict[str, np.ndarray]:
    return _validate_student_first_gmm_screen_scores(path)


def _validate_student_first_gmm_screen_scores(path: str | Path) -> dict[str, np.ndarray]:
    required = {
        "schema", "schema_revision", "policy", "policy_revision", "routing_policy",
        "routing_policy_revision", "selector", "selector_revision",
        "structure_ids", "system_ids", "pool", "keep", "fp_pool",
        "teacher_floor", "teacher_trust", "teacher_ceiling", "teacher_ceiling_configured",
        "student_floor", "student_ceiling", "student_ceiling_configured",
        "student_p", "student_scored", "student_floor_discard", "student_p_pool",
        "normal_teacher_scored", "teacher_scored", "teacher_p", "teacher_floor_discard",
        "teacher_fp_candidate", "student_eligible", "teacher_trust_reject",
        "normal_train_selected", "safe_fill_enabled", "safe_fill_eligible",
        "safe_fill_attempted", "safe_fill_accepted", "train_selected", "fp_candidate",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            if set(payload.files) != required:
                raise ValueError("student-first GMM score fields are incompatible")
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("malformed student-first GMM screen score audit") from exc
    if (
        str(arrays["schema"].item()) != SCORE_SCHEMA
        or int(arrays["schema_revision"].item()) != SCORE_SCHEMA_REVISION
        or str(arrays["policy"].item()) != POLICY_ID
        or int(arrays["policy_revision"].item()) != POLICY_REVISION
        or str(arrays["routing_policy"].item()) != _STUDENT_FIRST_ROUTING_POLICY.name
        or int(arrays["routing_policy_revision"].item()) != _STUDENT_FIRST_ROUTING_POLICY.revision
        or selector_identity(str(arrays["selector"].item())).revision != int(arrays["selector_revision"].item())
    ):
        raise ValueError("malformed student-first GMM screen score schema")
    identities = arrays["structure_ids"]
    systems = arrays["system_ids"]
    n = len(identities)
    if identities.ndim != 1 or identities.dtype.kind not in "US" or len(set(identities.tolist())) != n:
        raise ValueError("student-first GMM identities are invalid")
    if systems.shape != (n,) or systems.dtype.kind not in "US" or any(not str(item) for item in systems):
        raise ValueError("student-first GMM systems are invalid")
    scalar_ints = ("pool", "keep", "fp_pool")
    if any(arrays[name].shape != () or arrays[name].dtype.kind not in "iu" for name in scalar_ints):
        raise ValueError("student-first GMM quotas are invalid")
    pool, keep, fp_pool = (int(arrays[name].item()) for name in scalar_ints)
    if pool == 0 or pool < -1 or keep <= 0 or fp_pool == 0 or fp_pool < -1:
        raise ValueError("student-first GMM quotas are invalid")
    scalar_floats = ("teacher_floor", "teacher_trust", "teacher_ceiling", "student_floor", "student_ceiling")
    scalar_bools = ("teacher_ceiling_configured", "student_ceiling_configured", "safe_fill_enabled")
    if any(arrays[name].shape != () or arrays[name].dtype != np.float64 for name in scalar_floats) or any(arrays[name].shape != () or arrays[name].dtype != np.bool_ for name in scalar_bools):
        raise ValueError("student-first GMM thresholds are invalid")
    vector_bools = required - {
        "schema", "schema_revision", "policy", "policy_revision", "routing_policy",
        "routing_policy_revision", "selector", "selector_revision",
        "structure_ids", "system_ids", "pool", "keep", "fp_pool", "teacher_p", "student_p",
        *scalar_floats, *scalar_bools,
    }
    if any(arrays[name].shape != (n,) or arrays[name].dtype != np.bool_ for name in vector_bools):
        raise ValueError("student-first GMM flags are invalid")
    student_probabilities = arrays["student_p"]
    teacher_probabilities = arrays["teacher_p"]
    teacher_scored = arrays["teacher_scored"]
    if student_probabilities.shape != (n,) or student_probabilities.dtype != np.float64 or np.any(~np.isfinite(student_probabilities)) or np.any((student_probabilities < 0) | (student_probabilities > 1)) or not np.all(arrays["student_scored"]):
        raise ValueError("student-first GMM student applicability is invalid")
    if teacher_probabilities.shape != (n,) or teacher_probabilities.dtype != np.float64 or np.any(~np.isfinite(teacher_probabilities[teacher_scored])) or np.any((teacher_probabilities[teacher_scored] < 0) | (teacher_probabilities[teacher_scored] > 1)) or np.any(~np.isnan(teacher_probabilities[~teacher_scored])):
        raise ValueError("student-first GMM teacher applicability is invalid")
    teacher_floor = float(arrays["teacher_floor"].item())
    teacher_trust = float(arrays["teacher_trust"].item())
    teacher_ceiling = float(arrays["teacher_ceiling"].item())
    student_floor = float(arrays["student_floor"].item())
    student_ceiling = float(arrays["student_ceiling"].item())
    teacher_ceiling_configured = bool(arrays["teacher_ceiling_configured"].item())
    student_ceiling_configured = bool(arrays["student_ceiling_configured"].item())
    if not 0 <= teacher_floor < teacher_ceiling <= 1 or not 0 <= teacher_trust <= 1 or not 0 <= student_floor < student_ceiling <= 1:
        raise ValueError("student-first GMM thresholds are invalid")
    if (not teacher_ceiling_configured and teacher_ceiling != 1.0) or (not student_ceiling_configured and student_ceiling != 1.0):
        raise ValueError("student-first GMM omitted ceilings are invalid")
    student_floor_discard = student_probabilities < student_floor
    student_upper = student_probabilities < student_ceiling if student_ceiling_configured else student_probabilities <= 1
    proposal = arrays["student_p_pool"]
    normal_teacher = arrays["normal_teacher_scored"]
    safe_eligible = arrays["safe_fill_eligible"]
    safe_attempted = arrays["safe_fill_attempted"]
    safe_accepted = arrays["safe_fill_accepted"]
    normal_train = arrays["normal_train_selected"]
    if not np.array_equal(arrays["student_floor_discard"], student_floor_discard) or np.any(proposal & (student_floor_discard | ~student_upper)) or not np.array_equal(proposal, normal_teacher):
        raise ValueError("student-first GMM proposal evidence is inconsistent")
    expected_safe = bool(arrays["safe_fill_enabled"].item()) & student_ceiling_configured & (student_probabilities >= student_ceiling)
    if not np.array_equal(safe_eligible, expected_safe) or np.any(safe_attempted & ~safe_eligible) or np.any(safe_attempted & proposal):
        raise ValueError("student-first GMM safe-fill evidence is inconsistent")
    if not np.array_equal(teacher_scored, normal_teacher | safe_attempted):
        raise ValueError("student-first GMM teacher applicability is inconsistent")
    expected_floor = teacher_scored & (teacher_probabilities < teacher_floor)
    trusted = teacher_scored & ~expected_floor & (teacher_probabilities > teacher_trust)
    trust_reject = teacher_scored & ~expected_floor & ~trusted
    if not np.array_equal(arrays["teacher_floor_discard"], expected_floor) or not np.array_equal(arrays["student_eligible"], trusted) or not np.array_equal(arrays["teacher_trust_reject"], trust_reject):
        raise ValueError("student-first GMM teacher trust evidence is inconsistent")
    teacher_upper = teacher_probabilities < teacher_ceiling if teacher_ceiling_configured else teacher_probabilities <= 1
    fp_candidate = arrays["fp_candidate"]
    if np.any(fp_candidate & (~normal_teacher | expected_floor | ~teacher_upper)) or not np.array_equal(fp_candidate, arrays["teacher_fp_candidate"]):
        raise ValueError("student-first GMM FP evidence is inconsistent")
    if np.any(normal_train & (~normal_teacher | ~trusted)) or not np.array_equal(safe_accepted, safe_attempted & trusted) or not np.array_equal(arrays["train_selected"], normal_train | safe_accepted) or np.any(safe_accepted & fp_candidate):
        raise ValueError("student-first GMM train evidence is inconsistent")
    for system_id in set(str(item) for item in systems.tolist()):
        owned = systems == system_id
        if pool > 0 and int(np.count_nonzero(normal_teacher & owned)) > pool:
            raise ValueError("student-first GMM proposal exceeds pool")
        if fp_pool > 0 and int(np.count_nonzero(fp_candidate & owned)) > fp_pool:
            raise ValueError("student-first GMM FP candidates exceed fp_pool")
        initial_shortfall = max(0, keep - int(np.count_nonzero(normal_train & owned)))
        if int(np.count_nonzero(safe_attempted & owned)) > initial_shortfall:
            raise ValueError("student-first GMM safe-fill exceeds initial shortfall")
        if int(np.count_nonzero(arrays["train_selected"] & owned)) > keep:
            raise ValueError("student-first GMM train output exceeds keep")
    return arrays


def validate_gmm_pair_guard_alignment(
    score_path: str | Path,
    guard_path: str | Path,
    *,
    thresholds: PairGuardThresholds,
) -> None:
    from workflow_artifacts.lifecycle.pair_guard import load_pair_guard

    scores = validate_gmm_screen_scores(score_path)
    guard = load_pair_guard(guard_path, thresholds=thresholds)
    scored_ids = tuple(str(item) for item in scores["structure_ids"].tolist())
    pass_ids = tuple(
        str(identity)
        for identity, passed in zip(guard.structure_ids.tolist(), guard.passed.tolist(), strict=True)
        if bool(passed)
    )
    if scored_ids != pass_ids:
        raise ValueError("GMM score identities do not equal ordered pair-guard pass identities")


def _teacher_thresholds(value: Mapping[str, float]) -> tuple[float, float, float, bool]:
    if set(value) not in ({"floor", "trust"}, {"floor", "trust", "ceiling"}):
        raise ValueError("GMM screen teacher p thresholds are invalid")
    floor = float(value["floor"])
    trust = float(value["trust"])
    configured = "ceiling" in value
    ceiling = float(value.get("ceiling", 1.0))
    if not 0 <= floor < ceiling <= 1 or not 0 <= trust <= 1:
        raise ValueError("GMM screen teacher p thresholds are invalid")
    return floor, trust, ceiling, configured


def _student_thresholds(value: Mapping[str, float]) -> tuple[float, float, bool]:
    if set(value) not in ({"floor"}, {"floor", "ceiling"}):
        raise ValueError("GMM screen student p thresholds are invalid")
    floor = float(value["floor"])
    configured = "ceiling" in value
    ceiling = float(value.get("ceiling", 1.0))
    if not 0 <= floor < ceiling <= 1:
        raise ValueError("GMM screen student p thresholds are invalid")
    return floor, ceiling, configured


def validate_gmm_screen_summary(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("malformed GMM screen summary") from exc
    if not isinstance(payload, dict):
        raise ValueError("malformed GMM screen summary")
    if (
        payload.get("schema") != SUMMARY_SCHEMA
        or payload.get("schema_revision") != SUMMARY_SCHEMA_REVISION
        or payload.get("policy_identity") != GMM_REFERENCE_TAIL.as_dict()
        or payload.get("directional_policy")
        != _STUDENT_FIRST_ROUTING_POLICY.as_dict()
    ):
        raise ValueError("GMM screen summary schema or policy is incompatible")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != {"teacher", "student", "pair_guard", "quota"}:
        raise ValueError("GMM screen summary effective thresholds are missing")
    teacher = thresholds["teacher"]
    student = thresholds["student"]
    pair_guard = thresholds["pair_guard"]
    quota = thresholds["quota"]
    if not isinstance(teacher, dict) or set(teacher) != {"floor", "trust", "ceiling"}:
        raise ValueError("GMM screen summary teacher thresholds are invalid")
    if not isinstance(student, dict) or set(student) != {"floor", "ceiling"}:
        raise ValueError("GMM screen summary student thresholds are invalid")
    try:
        teacher_floor = float(teacher["floor"])
        teacher_trust = float(teacher["trust"])
        teacher_ceiling = None if teacher["ceiling"] is None else float(teacher["ceiling"])
        student_floor = float(student["floor"])
        student_ceiling = None if student["ceiling"] is None else float(student["ceiling"])
    except (TypeError, ValueError) as exc:
        raise ValueError("GMM screen summary effective thresholds are invalid") from exc
    if (
        not math.isfinite(teacher_floor)
        or not 0 <= teacher_floor < 1
        or not math.isfinite(teacher_trust)
        or not 0 <= teacher_trust <= 1
        or (teacher_ceiling is not None and (not math.isfinite(teacher_ceiling) or not teacher_floor < teacher_ceiling <= 1))
        or not math.isfinite(student_floor)
        or not 0 <= student_floor < 1
        or (student_ceiling is not None and (not math.isfinite(student_ceiling) or not student_floor < student_ceiling <= 1))
    ):
        raise ValueError("GMM screen summary effective thresholds are invalid")
    if (
        not isinstance(quota, dict)
        or set(quota) != {"pool", "keep"}
        or any(not isinstance(quota[name], int) or isinstance(quota[name], bool) for name in quota)
        or quota["pool"] == 0
        or quota["pool"] < -1
        or quota["keep"] <= 0
    ):
        raise ValueError("GMM screen summary effective quota is invalid")
    if not isinstance(payload.get("fill_underfilled_from_safe_region"), bool):
        raise ValueError("GMM screen summary safe-fill setting is invalid")
    pair_guard_identity = payload.get("pair_guard_identity")
    if pair_guard is None:
        if pair_guard_identity is not None:
            raise ValueError("GMM screen summary pair-guard policy is inconsistent")
    else:
        if pair_guard_identity != PAIR_GUARD.as_dict() or not isinstance(pair_guard, dict) or set(pair_guard) != {"min", "pairs"}:
            raise ValueError("GMM screen summary pair-guard evidence is invalid")
        minimum = pair_guard["min"]
        pairs = pair_guard["pairs"]
        if (
            not isinstance(minimum, (int, float))
            or isinstance(minimum, bool)
            or not math.isfinite(float(minimum))
            or float(minimum) <= 0.0
            or not isinstance(pairs, dict)
            or any(
                not isinstance(key, str)
                or "-" not in key
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < float(minimum)
                for key, value in pairs.items()
            )
        ):
            raise ValueError("GMM screen summary pair-guard thresholds are invalid")
        try:
            normalized_pair_guard = PairGuardThresholds(
                minimum=float(minimum),
                pairs=pairs,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("GMM screen summary pair-guard thresholds are invalid") from exc
        if dict(normalized_pair_guard.pairs) != {
            str(key): float(value) for key, value in pairs.items()
        }:
            raise ValueError("GMM screen summary pair-guard pair keys are not canonical")
    count_names = {
        "observations", "guard_pass", "guard_reject", "teacher_scored",
        "normal_teacher_scored",
        "teacher_floor_discard", "teacher_fp_candidate", "student_eligible",
        "teacher_trust_reject", "student_floor_discard", "student_pool", "normal_train",
        "safe_fill_eligible", "safe_fill_attempted", "safe_fill_accepted", "train",
        "train_underfill", "fp_candidate",
    }
    counts = payload.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != count_names
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values())
    ):
        raise ValueError("GMM screen summary counts are missing")
    if (
        counts["guard_pass"] + counts["guard_reject"] != counts["observations"]
        or counts["normal_teacher_scored"] != counts["student_pool"]
        or counts["teacher_scored"]
        != counts["normal_teacher_scored"] + counts["safe_fill_attempted"]
        or counts["normal_teacher_scored"] > counts["guard_pass"]
    ):
        raise ValueError("GMM screen summary guard and score counts are inconsistent")
    if (
        counts["normal_train"] + counts["safe_fill_accepted"] != counts["train"]
        or counts["safe_fill_accepted"] > counts["safe_fill_attempted"]
        or counts["safe_fill_attempted"] > counts["safe_fill_eligible"]
        or counts["train_underfill"] != max(0, quota["keep"] * len(payload.get("per_system", {})) - counts["train"])
    ):
        raise ValueError("GMM screen summary safe-fill counts are inconsistent")
    per_system = payload.get("per_system")
    if not isinstance(per_system, dict) or any(
        not isinstance(system_id, str)
        or not system_id
        or not isinstance(system_counts, dict)
        or set(system_counts) != count_names
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in system_counts.values())
        or system_counts["guard_pass"] + system_counts["guard_reject"] != system_counts["observations"]
        or system_counts["normal_teacher_scored"] != system_counts["student_pool"]
        or system_counts["teacher_scored"]
        != system_counts["normal_teacher_scored"]
        + system_counts["safe_fill_attempted"]
        or system_counts["normal_teacher_scored"] > system_counts["guard_pass"]
        or system_counts["normal_train"] + system_counts["safe_fill_accepted"] != system_counts["train"]
        or system_counts["safe_fill_accepted"] > system_counts["safe_fill_attempted"]
        or system_counts["safe_fill_attempted"] > system_counts["safe_fill_eligible"]
        or system_counts["train_underfill"] != max(0, quota["keep"] - system_counts["train"])
        for system_id, system_counts in per_system.items()
    ):
        raise ValueError("GMM screen per-system guard and route counts are invalid")
    if any(
        sum(system_counts[name] for system_counts in per_system.values()) != counts[name]
        for name in count_names
    ):
        raise ValueError("GMM screen global and per-system counts are inconsistent")
    selectors = payload.get("selectors")
    if not isinstance(selectors, dict) or set(selectors) != {"screen", "fp"}:
        raise ValueError("GMM screen summary selector evidence is invalid")
    try:
        for value in selectors.values():
            if not isinstance(value, dict) or value != selector_identity(value.get("name", "")).as_dict():
                raise ValueError("unsupported selector identity")
    except (TypeError, ValueError) as exc:
        raise ValueError("GMM screen summary selector evidence is invalid") from exc
    p_bins = payload.get("p_bins")
    expected_bins = {
        "teacher": _p_bin_labels((teacher_floor, teacher_trust, teacher_ceiling)),
        "student": _p_bin_labels((student_floor, student_ceiling)),
    }
    if not isinstance(p_bins, dict) or set(p_bins) != {"teacher", "student"}:
        raise ValueError("GMM screen summary p bins are missing")
    for role, labels in expected_bins.items():
        bins = p_bins[role]
        if (
            not isinstance(bins, dict)
            or set(bins) != labels
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in bins.values())
        ):
            raise ValueError("GMM screen summary p bins do not match effective thresholds")
    if not isinstance(payload.get("provenance"), dict):
        raise ValueError("GMM screen summary provenance is missing")
    return payload


def _p_bin_labels(thresholds: tuple[float | None, ...]) -> set[str]:
    edges = sorted({0.0, 1.0, *(float(value) for value in thresholds if value is not None)})
    return {f"{left:g}-{right:g}" for left, right in zip(edges, edges[1:])}


__all__ = [
    "POLICY_ID", "POLICY_REVISION", "SCORE_SCHEMA", "SCORE_SCHEMA_REVISION",
    "SUMMARY_SCHEMA", "SUMMARY_SCHEMA_REVISION", "publish_gmm_screen_scores",
    "validate_gmm_pair_guard_alignment", "validate_gmm_screen_scores", "validate_gmm_screen_summary",
]
