from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from ase import Atoms
import numpy as np


SCHEMA = "dice.lifecycle.joint-direct-screen"
SCHEMA_REVISION = 2


def publish_joint_direct_screen_evidence(
    path: str | Path,
    *,
    observations: Sequence[Atoms],
    candidate_vectors: Mapping[str, np.ndarray],
    background_vectors: Mapping[str, np.ndarray],
    descriptor_schema: Mapping[str, object] | None,
    model_identity: str | None,
    dataset_identity: str | None,
    pool: int,
    keep: int,
    screen_policy: str,
    fp_policy: str,
    route_kind: str,
    proposal_ids: Sequence[str],
    normal_train_ids: set[str],
    train_ids: set[str],
    fp_ids: set[str],
    teacher_metrics: Mapping[str, float],
    outcomes: Mapping[str, Mapping[str, int | bool]],
    route_thresholds: Mapping[str, object],
) -> None:
    frames = tuple(observations)
    structure_ids = tuple(str(frame.info["structure_id"]) for frame in frames)
    system_ids = tuple(str(frame.info["system_id"]) for frame in frames)
    if len(set(structure_ids)) != len(structure_ids):
        raise ValueError("joint DIRECT screen evidence requires unique observations")
    readout_ids, readout_rows = _vectors(candidate_vectors)
    background_ids, background_rows = _vectors(background_vectors)
    if readout_rows.size and background_rows.size and (
        readout_rows.shape[1] != background_rows.shape[1]
    ):
        raise ValueError("joint DIRECT screen evidence feature widths differ")
    proposal = tuple(str(identity) for identity in proposal_ids)
    proposal_set = set(proposal)
    if len(proposal_set) != len(proposal) or not proposal_set.issubset(structure_ids):
        raise ValueError("joint DIRECT screen proposal is not an ordered observation subset")
    metric_ids = tuple(str(identity) for identity in teacher_metrics)
    metric_values = np.asarray(
        [float(teacher_metrics[identity]) for identity in metric_ids], dtype=np.float64
    )
    if not np.isfinite(metric_values).all():
        raise ValueError("joint DIRECT teacher metrics must be finite")
    payload = {
        "schema": np.asarray(SCHEMA),
        "schema_revision": np.asarray(SCHEMA_REVISION, dtype=np.int64),
        "structure_ids": np.asarray(structure_ids),
        "system_ids": np.asarray(system_ids),
        "readout_candidate_ids": np.asarray(readout_ids),
        "candidate_vectors": readout_rows,
        "background_ids": np.asarray(background_ids),
        "background_vectors": background_rows,
        "descriptor_schema_json": np.asarray(
            json.dumps(dict(descriptor_schema or {}), sort_keys=True, separators=(",", ":"))
        ),
        "model_identity": np.asarray("" if model_identity is None else model_identity),
        "dataset_identity": np.asarray("" if dataset_identity is None else dataset_identity),
        "pool": np.asarray(pool, dtype=np.int64),
        "keep": np.asarray(keep, dtype=np.int64),
        "screen_policy": np.asarray(screen_policy),
        "fp_policy": np.asarray(fp_policy),
        "route_kind": np.asarray(route_kind),
        "policy_name": np.asarray("joint-direct"),
        "policy_revision": np.asarray(2, dtype=np.int64),
        "route_thresholds_json": np.asarray(
            json.dumps(dict(route_thresholds), sort_keys=True, separators=(",", ":"))
        ),
        "proposal_ids": np.asarray(proposal),
        "student_pool": np.asarray(
            [identity in proposal_set for identity in structure_ids], dtype=np.bool_
        ),
        "normal_train_selected": np.asarray(
            [identity in normal_train_ids for identity in structure_ids], dtype=np.bool_
        ),
        "train_selected": np.asarray(
            [identity in train_ids for identity in structure_ids], dtype=np.bool_
        ),
        "fp_candidate": np.asarray(
            [identity in fp_ids for identity in structure_ids], dtype=np.bool_
        ),
        "teacher_metric_ids": np.asarray(metric_ids),
        "teacher_metrics": metric_values,
        "outcomes_json": np.asarray(
            json.dumps(dict(outcomes), sort_keys=True, separators=(",", ":"))
        ),
    }
    np.savez(Path(path), **payload)


def load_joint_direct_screen_evidence(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {name: np.array(archive[name]) for name in archive.files}
    required = {
        "schema", "schema_revision", "structure_ids", "system_ids",
        "readout_candidate_ids", "candidate_vectors", "background_ids",
        "background_vectors", "descriptor_schema_json", "model_identity",
        "dataset_identity", "pool", "keep", "screen_policy", "fp_policy",
        "route_kind", "policy_name", "policy_revision",
        "route_thresholds_json", "proposal_ids", "student_pool",
        "normal_train_selected", "train_selected", "fp_candidate",
        "teacher_metric_ids", "teacher_metrics", "outcomes_json",
    }
    if set(payload) != required or str(payload["schema"].item()) != SCHEMA or int(
        payload["schema_revision"].item()
    ) != SCHEMA_REVISION:
        raise ValueError("joint DIRECT screen evidence schema is invalid")
    count = len(payload["structure_ids"])
    if any(
        np.asarray(payload[name]).shape != (count,)
        for name in (
            "system_ids", "student_pool", "normal_train_selected",
            "train_selected", "fp_candidate",
        )
    ):
        raise ValueError("joint DIRECT screen evidence observation alignment is invalid")
    if payload["candidate_vectors"].ndim != 2 or len(
        payload["candidate_vectors"]
    ) != len(payload["readout_candidate_ids"]):
        raise ValueError("joint DIRECT candidate vector shape is invalid")
    if payload["background_vectors"].ndim != 2 or len(
        payload["background_vectors"]
    ) != len(payload["background_ids"]):
        raise ValueError("joint DIRECT background vector shape is invalid")
    if not np.isfinite(payload["candidate_vectors"]).all() or not np.isfinite(
        payload["background_vectors"]
    ).all():
        raise ValueError("joint DIRECT screen vectors must be finite")
    json.loads(str(payload["descriptor_schema_json"].item()))
    json.loads(str(payload["outcomes_json"].item()))
    json.loads(str(payload["route_thresholds_json"].item()))
    if (
        str(payload["policy_name"].item()) != "joint-direct"
        or int(payload["policy_revision"].item()) != 2
    ):
        raise ValueError("joint DIRECT screen policy identity is invalid")
    return payload


def _vectors(values: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray]:
    identities = tuple(str(identity) for identity in values)
    if len(set(identities)) != len(identities):
        raise ValueError("joint DIRECT vector identities must be unique")
    if not identities:
        return (), np.empty((0, 0), dtype=np.float32)
    rows = np.stack([np.asarray(values[identity], dtype=np.float32) for identity in identities])
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("joint DIRECT vectors must be finite frame x feature rows")
    return identities, rows


__all__ = [
    "SCHEMA",
    "SCHEMA_REVISION",
    "load_joint_direct_screen_evidence",
    "publish_joint_direct_screen_evidence",
]
