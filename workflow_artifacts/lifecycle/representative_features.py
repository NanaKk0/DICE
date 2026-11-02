from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from workflow_backends.descriptor_stream import validated_readout_input_schema
from workflow_artifacts.identities import RevisionedIdentity
from workflow_policies.identities import READOUT_INPUT_FRAME_MEAN

_FRAME_MEAN_IDENTITY = RevisionedIdentity(
    "dice.lifecycle.frame-mean-representatives", 2
)
FRAME_MEAN_SCHEMA = _FRAME_MEAN_IDENTITY.name
FRAME_MEAN_SCHEMA_REVISION = _FRAME_MEAN_IDENTITY.revision
_HISTORICAL_FRAME_MEAN_SCHEMA = "dice.v2.frame-mean-representatives.v2"


def _canonical_schema(schema: Mapping[str, object]) -> tuple[dict[str, object], str]:
    value = validated_readout_input_schema(schema)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return value, encoded


def readout_input_frame_mean_identity(schema: Mapping[str, object]) -> dict[str, object]:
    _value, encoded = _canonical_schema(schema)
    return {
        **READOUT_INPUT_FRAME_MEAN.as_dict(),
        "descriptor_schema_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def publish_frame_mean_representatives(
    path: str | Path,
    *,
    identities: Sequence[str],
    means: np.ndarray,
    descriptor_schema: Mapping[str, object],
) -> Path:
    schema, encoded_schema = _canonical_schema(descriptor_schema)
    descriptor_identity = readout_input_frame_mean_identity(schema)
    ordered = tuple(str(identity) for identity in identities)
    rows = np.asarray(means, dtype=np.float32)
    _validate_values(
        ordered,
        rows,
        expected_width=int(schema["width"]),
    )
    target = Path(path)
    with target.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(FRAME_MEAN_SCHEMA, dtype=np.str_),
            schema_revision=np.asarray(FRAME_MEAN_SCHEMA_REVISION, dtype=np.int64),
            descriptor_name=np.asarray(descriptor_identity["name"], dtype=np.str_),
            descriptor_revision=np.asarray(descriptor_identity["revision"], dtype=np.int64),
            descriptor_schema_sha256=np.asarray(
                descriptor_identity["descriptor_schema_sha256"], dtype=np.str_
            ),
            descriptor_schema=np.asarray(encoded_schema, dtype=np.str_),
            ids=np.asarray(ordered, dtype=np.str_),
            means=rows,
        )
    load_frame_mean_representatives(
        target,
        expected_identities=ordered,
        expected_descriptor_schema=schema,
    )
    return target


def load_frame_mean_representatives(
    path: str | Path,
    *,
    expected_identities: Sequence[str],
    expected_descriptor_schema: Mapping[str, object],
) -> dict[str, np.ndarray]:
    expected_schema, encoded_schema = _canonical_schema(expected_descriptor_schema)
    expected_identity = readout_input_frame_mean_identity(expected_schema)
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"frame-mean representative evidence is missing: {target}")
    try:
        with np.load(target, allow_pickle=False) as payload:
            fields = set(payload.files)
            canonical_fields = {
                "schema", "schema_revision", "descriptor_name", "descriptor_revision",
                "descriptor_schema_sha256", "descriptor_schema", "ids", "means",
            }
            historical_fields = {
                "schema", "descriptor_identity", "descriptor_schema", "ids", "means"
            }
            if fields not in (canonical_fields, historical_fields):
                raise ValueError("incompatible fields")
            artifact_schema = np.asarray(payload["schema"])
            descriptor_schema = np.asarray(payload["descriptor_schema"])
            identities = np.asarray(payload["ids"])
            means = np.asarray(payload["means"])
            if fields == canonical_fields:
                schema_revision = np.asarray(payload["schema_revision"])
                descriptor_name = np.asarray(payload["descriptor_name"])
                descriptor_revision = np.asarray(payload["descriptor_revision"])
                descriptor_schema_sha256 = np.asarray(payload["descriptor_schema_sha256"])
                historical_identity = None
            else:
                historical_identity = np.asarray(payload["descriptor_identity"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("malformed frame-mean representative evidence") from exc
    if historical_identity is None:
        if (
            artifact_schema.shape != ()
            or str(artifact_schema.item()) != FRAME_MEAN_SCHEMA
            or schema_revision.shape != ()
            or schema_revision.dtype.kind not in "iu"
            or int(schema_revision.item()) != FRAME_MEAN_SCHEMA_REVISION
            or descriptor_name.shape != ()
            or str(descriptor_name.item()) != expected_identity["name"]
            or descriptor_revision.shape != ()
            or descriptor_revision.dtype.kind not in "iu"
            or int(descriptor_revision.item()) != expected_identity["revision"]
            or descriptor_schema_sha256.shape != ()
            or str(descriptor_schema_sha256.item())
            != expected_identity["descriptor_schema_sha256"]
            or descriptor_schema.shape != ()
            or str(descriptor_schema.item()) != encoded_schema
        ):
            raise ValueError("frame-mean representative descriptor is incompatible")
    else:
        if (
            artifact_schema.shape != ()
            or str(artifact_schema.item()) != _HISTORICAL_FRAME_MEAN_SCHEMA
            or historical_identity.shape != ()
            or descriptor_schema.shape != ()
        ):
            raise ValueError("malformed frame-mean representative evidence schema")
        historical_encoded = str(descriptor_schema.item())
        expected_historical_identity = (
            "readout-input-frame-mean@1:"
            + hashlib.sha256(historical_encoded.encode("utf-8")).hexdigest()
        )
        try:
            historical_descriptor = validated_readout_input_schema(
                json.loads(historical_encoded)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("frame-mean representative descriptor is incompatible") from exc
        if (
            str(historical_identity.item()) != expected_historical_identity
            or historical_descriptor != expected_schema
        ):
            raise ValueError("frame-mean representative descriptor is incompatible")
    if identities.ndim != 1 or identities.dtype.kind not in "US":
        raise ValueError("frame-mean representative identities are malformed")
    ordered = tuple(str(identity) for identity in identities.tolist())
    expected = tuple(str(identity) for identity in expected_identities)
    if ordered != expected:
        raise ValueError("frame-mean representative identities do not match candidate order")
    _validate_values(
        ordered,
        means,
        expected_width=int(expected_schema["width"]),
    )
    return {identity: means[index] for index, identity in enumerate(ordered)}


def _validate_values(
    identities: tuple[str, ...],
    means: np.ndarray,
    *,
    expected_width: int,
) -> None:
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("frame-mean representative identities must be non-empty and unique")
    if means.dtype != np.float32 or means.ndim != 2 or means.shape[0] != len(identities):
        raise ValueError("frame-mean representatives must be ordered float32 frame x feature rows")
    if means.shape[1] != expected_width:
        raise ValueError(f"frame-mean representative width must be {expected_width}")
    if means.shape[1] <= 0 or not np.isfinite(means).all():
        raise ValueError("frame-mean representatives must be finite with positive feature width")


__all__ = [
    "FRAME_MEAN_SCHEMA",
    "FRAME_MEAN_SCHEMA_REVISION",
    "load_frame_mean_representatives",
    "publish_frame_mean_representatives",
    "readout_input_frame_mean_identity",
]
