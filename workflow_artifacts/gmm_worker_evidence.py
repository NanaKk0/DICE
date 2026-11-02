from __future__ import annotations

import copy
import numpy as np
from typing import Any, Mapping, Sequence

from workflow_artifacts.identities import RevisionedIdentity


GMM_FRAME_SCORE_RESULT = RevisionedIdentity(
    "dice.lifecycle.gmm-frame-score-result", 1
)
GMM_SCORE_CHUNKS = RevisionedIdentity("dice.lifecycle.gmm-score-chunks", 1)

_HISTORICAL_FRAME_SCORE_RESULT = "dice.v2.gmm-frame-score-result.v1"
_HISTORICAL_SCORE_CHUNKS = "dice.v2.gmm-score-chunks.v1"


def element_means_payload(values):
    return None if values is None else {
        element: {"frame_ids": list(identities), "means": np.asarray(rows, dtype=np.float32).tolist()}
        for element, (identities, rows) in values.items()
    }


def parse_element_means(payload, *, frame_ids, width):
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("GMM element means must be a mapping")
    positions = {identity: index for index, identity in enumerate(frame_ids)}
    result = {}
    for element, raw in payload.items():
        if not isinstance(element, str) or not element or not isinstance(raw, dict) or set(raw) != {"frame_ids", "means"}:
            raise ValueError("GMM element means contain invalid evidence")
        identities = tuple(raw["frame_ids"])
        try:
            ordinals = tuple(positions[identity] for identity in identities)
        except (KeyError, TypeError) as exc:
            raise ValueError("GMM element means contain an unknown frame") from exc
        rows = np.asarray(raw["means"], dtype=np.float32)
        if not identities or ordinals != tuple(sorted(set(ordinals))) or rows.shape != (len(identities), width) or not np.isfinite(rows).all():
            raise ValueError("GMM element means changed canonical order or width")
        result[element] = (identities, rows)
    return result


def gmm_frame_score_result_payload(**payload: Any) -> dict[str, Any]:
    return {
        **GMM_FRAME_SCORE_RESULT.as_dict(
            name_field="schema", revision_field="schema_revision"
        ),
        **payload,
    }


def gmm_score_chunks_payload(
    *, frame_count: int, chunks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        **GMM_SCORE_CHUNKS.as_dict(
            name_field="schema", revision_field="schema_revision"
        ),
        "frame_count": frame_count,
        "chunks": [dict(chunk) for chunk in chunks],
    }


def normalize_gmm_frame_score_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize(
        payload,
        current=GMM_FRAME_SCORE_RESULT,
        historical=_HISTORICAL_FRAME_SCORE_RESULT,
    )


def normalize_gmm_score_chunks(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize(
        payload,
        current=GMM_SCORE_CHUNKS,
        historical=_HISTORICAL_SCORE_CHUNKS,
    )


def _normalize(
    payload: Mapping[str, Any],
    *,
    current: RevisionedIdentity,
    historical: str,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    if current.matches(value, name_field="schema", revision_field="schema_revision"):
        return value
    if value.get("schema") == historical and "schema_revision" not in value:
        value.update(
            current.as_dict(name_field="schema", revision_field="schema_revision")
        )
        return value
    raise ValueError("unsupported GMM worker evidence schema")


__all__ = [
    "GMM_FRAME_SCORE_RESULT",
    "GMM_SCORE_CHUNKS",
    "gmm_frame_score_result_payload",
    "gmm_score_chunks_payload",
    "normalize_gmm_frame_score_result",
    "normalize_gmm_score_chunks",
]
