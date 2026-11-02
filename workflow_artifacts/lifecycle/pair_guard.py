from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from workflow_artifacts.identities import RevisionedIdentity
from workflow_policies.pair_guard import (
    POLICY_ID,
    POLICY_REVISION,
    PairGuardDecision,
    PairGuardThresholds,
)


_PAIR_GUARD_IDENTITY = RevisionedIdentity("dice.lifecycle.pair-guard", 1)
PAIR_GUARD_SCHEMA = _PAIR_GUARD_IDENTITY.name
PAIR_GUARD_SCHEMA_REVISION = _PAIR_GUARD_IDENTITY.revision
_HISTORICAL_PAIR_GUARD_SCHEMA = "dice.v2.pair_guard.v1"
_HISTORICAL_POLICY_ID = "pair-guard@1"
_FIELDS = {
    "schema",
    "schema_revision",
    "policy",
    "policy_revision",
    "structure_ids",
    "system_ids",
    "passed",
    "reject_rules",
    "reject_distances",
}


@dataclass(frozen=True)
class PairGuardArtifact:
    structure_ids: np.ndarray
    system_ids: np.ndarray
    passed: np.ndarray
    reject_rules: np.ndarray
    reject_distances: np.ndarray


def publish_pair_guard(
    path: str | Path,
    decisions: Sequence[PairGuardDecision],
    *,
    thresholds: PairGuardThresholds,
) -> Path:
    arrays = {
        "schema": np.asarray(PAIR_GUARD_SCHEMA, dtype=np.str_),
        "schema_revision": np.asarray(PAIR_GUARD_SCHEMA_REVISION, dtype=np.int64),
        "policy": np.asarray(POLICY_ID, dtype=np.str_),
        "policy_revision": np.asarray(POLICY_REVISION, dtype=np.int64),
        "structure_ids": np.asarray([item.structure_id for item in decisions], dtype=np.str_),
        "system_ids": np.asarray([item.system_id for item in decisions], dtype=np.str_),
        "passed": np.asarray([item.passed for item in decisions], dtype=np.bool_),
        "reject_rules": np.asarray([item.reject_rule for item in decisions], dtype=np.str_),
        "reject_distances": np.asarray(
            [0.0 if item.passed else item.reject_distance for item in decisions],
            dtype=np.float64,
        ),
    }
    _validated_artifact(arrays, thresholds=thresholds)
    target = Path(path)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    load_pair_guard(target, thresholds=thresholds)
    return target


def load_pair_guard(path: str | Path, *, thresholds: PairGuardThresholds) -> PairGuardArtifact:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            fields = set(payload.files)
            if fields not in (
                _FIELDS,
                _FIELDS - {"schema_revision", "policy_revision"},
            ):
                raise ValueError("pair-guard artifact fields are incompatible")
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
        if (
            str(arrays["schema"].item()) == _HISTORICAL_PAIR_GUARD_SCHEMA
            and str(arrays["policy"].item()) == _HISTORICAL_POLICY_ID
            and "schema_revision" not in arrays
            and "policy_revision" not in arrays
        ):
            arrays["schema"] = np.asarray(PAIR_GUARD_SCHEMA, dtype=np.str_)
            arrays["schema_revision"] = np.asarray(PAIR_GUARD_SCHEMA_REVISION, dtype=np.int64)
            arrays["policy"] = np.asarray(POLICY_ID, dtype=np.str_)
            arrays["policy_revision"] = np.asarray(POLICY_REVISION, dtype=np.int64)
        return _validated_artifact(arrays, thresholds=thresholds)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("malformed pair-guard artifact") from exc


def validate_pair_guard(path: str | Path, *, thresholds: PairGuardThresholds) -> PairGuardArtifact:
    return load_pair_guard(path, thresholds=thresholds)


def _validated_artifact(
    arrays: dict[str, np.ndarray],
    *,
    thresholds: PairGuardThresholds,
) -> PairGuardArtifact:
    schema = arrays["schema"]
    schema_revision = arrays["schema_revision"]
    policy = arrays["policy"]
    policy_revision = arrays["policy_revision"]
    if schema.ndim != 0 or schema.dtype.kind not in "US" or str(schema.item()) != PAIR_GUARD_SCHEMA:
        raise ValueError("pair-guard artifact schema is incompatible")
    if schema_revision.ndim != 0 or schema_revision.dtype.kind not in "iu" or int(schema_revision.item()) != PAIR_GUARD_SCHEMA_REVISION:
        raise ValueError("pair-guard artifact schema is incompatible")
    if policy.ndim != 0 or policy.dtype.kind not in "US" or str(policy.item()) != POLICY_ID:
        raise ValueError("pair-guard artifact policy is incompatible")
    if policy_revision.ndim != 0 or policy_revision.dtype.kind not in "iu" or int(policy_revision.item()) != POLICY_REVISION:
        raise ValueError("pair-guard artifact policy is incompatible")

    structure_ids = arrays["structure_ids"]
    system_ids = arrays["system_ids"]
    passed = arrays["passed"]
    reject_rules = arrays["reject_rules"]
    reject_distances = arrays["reject_distances"]
    if structure_ids.ndim != 1 or structure_ids.dtype.kind not in "US":
        raise ValueError("pair-guard structure identities are invalid")
    count = len(structure_ids)
    identities = [str(item) for item in structure_ids.tolist()]
    if any(not item or item != item.strip() for item in identities) or len(set(identities)) != count:
        raise ValueError("pair-guard structure identities are invalid")
    systems = [str(item) for item in system_ids.tolist()] if system_ids.ndim == 1 else []
    if (
        system_ids.shape != (count,)
        or system_ids.dtype.kind not in "US"
        or any(not item or item != item.strip() for item in systems)
    ):
        raise ValueError("pair-guard system identities are invalid")
    if passed.shape != (count,) or passed.dtype != np.bool_:
        raise ValueError("pair-guard pass flags are invalid")
    if reject_rules.shape != (count,) or reject_rules.dtype.kind not in "US":
        raise ValueError("pair-guard reject rules are invalid")
    if reject_distances.shape != (count,) or reject_distances.dtype != np.float64:
        raise ValueError("pair-guard reject distances are invalid")
    if not np.all(np.isfinite(reject_distances)):
        raise ValueError("pair-guard reject distances are invalid")

    allowed_rules = {"min", *thresholds.pairs.keys()}
    for index in range(count):
        rule = str(reject_rules[index])
        distance = float(reject_distances[index])
        if bool(passed[index]):
            if rule or distance != 0.0:
                raise ValueError("pair-guard pass evidence is inconsistent")
            continue
        if rule not in allowed_rules or distance < 0.0:
            raise ValueError("pair-guard reject evidence is invalid")
        threshold = thresholds.minimum if rule == "min" else thresholds.pairs[rule]
        if distance >= threshold:
            raise ValueError("pair-guard reject evidence does not violate its rule")

    return PairGuardArtifact(structure_ids, system_ids, passed, reject_rules, reject_distances)


__all__ = [
    "PAIR_GUARD_SCHEMA",
    "PAIR_GUARD_SCHEMA_REVISION",
    "PairGuardArtifact",
    "load_pair_guard",
    "publish_pair_guard",
    "validate_pair_guard",
]
