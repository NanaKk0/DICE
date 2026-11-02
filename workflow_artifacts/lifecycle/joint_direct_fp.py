from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from ase import Atoms
import numpy as np


SCHEMA = "dice.lifecycle.joint-direct-fp"
SCHEMA_REVISION = 2


def publish_joint_direct_fp_evidence(
    path: str | Path,
    *,
    candidates: Sequence[Atoms],
    candidate_vectors: Mapping[str, np.ndarray],
    background_vectors: Mapping[str, np.ndarray],
    descriptor_schema: Mapping[str, object],
    model_identity: str,
    dataset_identity: str,
    group_ids: Sequence[str],
    keep: int,
    all_rn: bool,
    selected_ids: Sequence[str],
    outcomes: Mapping[str, Mapping[str, object]],
) -> Path:
    frames = tuple(candidates)
    identities = tuple(str(frame.info["structure_id"]) for frame in frames)
    systems = tuple(str(frame.info["system_id"]) for frame in frames)
    if len(set(identities)) != len(identities):
        raise ValueError("joint DIRECT FP candidates must be ordered unique")
    candidate_ids, candidate_rows = _vectors(candidate_vectors)
    background_ids, background_rows = _vectors(background_vectors)
    if candidate_ids != identities:
        raise ValueError("joint DIRECT FP vectors changed candidate order")
    if not background_ids or candidate_rows.shape[1] != background_rows.shape[1]:
        raise ValueError("joint DIRECT FP background is empty or width-mismatched")
    selected = tuple(str(identity) for identity in selected_ids)
    if len(set(selected)) != len(selected) or not set(selected).issubset(identities):
        raise ValueError("joint DIRECT FP selection is not a candidate subset")
    payload = {
        "schema": np.asarray(SCHEMA),
        "schema_revision": np.asarray(SCHEMA_REVISION, dtype=np.int64),
        "structure_ids": np.asarray(identities),
        "system_ids": np.asarray(systems),
        "candidate_vectors": candidate_rows,
        "background_ids": np.asarray(background_ids),
        "background_vectors": background_rows,
        "descriptor_schema_json": np.asarray(json.dumps(dict(descriptor_schema), sort_keys=True, separators=(",", ":"))),
        "model_identity": np.asarray(str(model_identity)),
        "dataset_identity": np.asarray(str(dataset_identity)),
        "group_ids": np.asarray(tuple(str(item) for item in group_ids)),
        "keep": np.asarray(int(keep), dtype=np.int64),
        "all_rn": np.asarray(bool(all_rn), dtype=np.bool_),
        "policy_name": np.asarray("joint-direct"),
        "policy_revision": np.asarray(2, dtype=np.int64),
        "selected": np.asarray([identity in set(selected) for identity in identities], dtype=np.bool_),
        "selected_ids": np.asarray(selected),
        "outcomes_json": np.asarray(json.dumps(dict(outcomes), sort_keys=True, separators=(",", ":"))),
    }
    target = Path(path)
    np.savez_compressed(target, **payload)
    return target


def load_joint_direct_fp_evidence(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {name: np.array(archive[name]) for name in archive.files}
    required = {
        "schema", "schema_revision", "structure_ids", "system_ids",
        "candidate_vectors", "background_ids", "background_vectors",
        "descriptor_schema_json", "model_identity", "dataset_identity",
        "group_ids", "keep", "all_rn", "policy_name", "policy_revision",
        "selected", "selected_ids", "outcomes_json",
    }
    if set(payload) != required or str(payload["schema"].item()) != SCHEMA or int(payload["schema_revision"].item()) != SCHEMA_REVISION:
        raise ValueError("joint DIRECT FP evidence schema is invalid")
    count = len(payload["structure_ids"])
    if (
        len(set(str(item) for item in payload["structure_ids"].tolist())) != count
        or payload["system_ids"].shape != (count,)
        or payload["selected"].shape != (count,)
        or len(set(str(item) for item in payload["selected_ids"].tolist()))
        != len(payload["selected_ids"])
        or payload["candidate_vectors"].ndim != 2
        or payload["candidate_vectors"].shape[0] != count
        or payload["background_vectors"].ndim != 2
        or payload["background_vectors"].shape[0] != len(payload["background_ids"])
        or not len(payload["background_ids"])
        or payload["candidate_vectors"].shape[1] != payload["background_vectors"].shape[1]
        or not np.isfinite(payload["candidate_vectors"]).all()
        or not np.isfinite(payload["background_vectors"]).all()
        or str(payload["policy_name"].item()) != "joint-direct"
        or int(payload["policy_revision"].item()) != 2
    ):
        raise ValueError("joint DIRECT FP evidence alignment is invalid")
    json.loads(str(payload["descriptor_schema_json"].item()))
    json.loads(str(payload["outcomes_json"].item()))
    return payload


def _vectors(values: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray]:
    identities = tuple(str(identity) for identity in values)
    if not identities:
        return (), np.empty((0, 0), dtype=np.float32)
    rows = np.stack([np.asarray(values[identity], dtype=np.float32) for identity in identities])
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("joint DIRECT FP vectors must be finite frame rows")
    return identities, rows


__all__ = [
    "SCHEMA",
    "SCHEMA_REVISION",
    "load_joint_direct_fp_evidence",
    "publish_joint_direct_fp_evidence",
]
