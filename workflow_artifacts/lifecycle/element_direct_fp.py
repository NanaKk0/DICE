"""Canonical replay evidence for terminal element-explicit DIRECT policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile

from ase import Atoms
import numpy as np

from workflow_policies.identities import selector_identity


SCHEMA = "dice.lifecycle.element-direct-fp"
SCHEMA_REVISION = 1


def publish_element_direct_fp_evidence(
    path: str | Path,
    *,
    candidates: Sequence[Atoms],
    background_frames: Sequence[Atoms],
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    descriptor_schema: Mapping[str, object],
    type_map: Sequence[str],
    model_identity: str,
    dataset_identity: str,
    policy: str,
    background_mode: str,
    candidate_rounds: Sequence[int],
    group_ids: Sequence[str],
    keep: int,
    all_rn: bool,
    selected_ids: Sequence[str],
    outcomes: Mapping[str, Mapping[str, object]],
    requested_policy: str | None = None,
    effective_policy: str | None = None,
    fallback_reason: str | None = None,
) -> Path:
    """Publish one atomically replaced, native-free terminal replay record."""

    frames = tuple(candidates)
    background = tuple(background_frames)
    elements = tuple(str(item) for item in type_map)
    identities = _frame_ids(frames, "candidate")
    background_ids = _frame_ids(background, "background")
    if not elements or len(elements) != len(set(elements)):
        raise ValueError("terminal element DIRECT type_map must be ordered unique")
    if policy not in {"ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"}:
        raise ValueError("terminal element DIRECT policy is invalid")
    expected_mode = (
        "teacher-training" if policy == "ELEMENT_JOINT_DIRECT" else "none"
    )
    if background_mode != expected_mode:
        raise ValueError("terminal element DIRECT background mode is invalid")
    if policy == "ELEMENT_JOINT_DIRECT" and not background:
        raise ValueError("terminal element joint DIRECT background must be non-empty")
    if policy == "ELEMENT_DIRECT" and (
        background or background_element_vectors
    ):
        raise ValueError("terminal ELEMENT_DIRECT cannot publish background evidence")
    candidate_presence, candidate_rows = _dense_element_rows(
        frames,
        identities=identities,
        element_vectors=candidate_element_vectors,
        type_map=elements,
        width=int(descriptor_schema.get("width", 0)),
    )
    background_presence, background_rows = _dense_element_rows(
        background,
        identities=background_ids,
        element_vectors=background_element_vectors,
        type_map=elements,
        width=int(descriptor_schema.get("width", 0)),
    )
    selected = tuple(str(item) for item in selected_ids)
    if len(selected) != len(set(selected)) or not set(selected).issubset(identities):
        raise ValueError("terminal element DIRECT selection is not a candidate subset")
    rounds = tuple(int(item) for item in candidate_rounds)
    if any(item < 0 for item in rounds) or len(rounds) != len(set(rounds)):
        raise ValueError("terminal element DIRECT candidate rounds are invalid")
    requested = policy if requested_policy is None else str(requested_policy)
    effective = policy if effective_policy is None else str(effective_policy)
    reason = "none" if fallback_reason is None else str(fallback_reason)
    if requested != policy or effective != policy or reason != "none":
        raise ValueError("executed element policy cannot contain fallback audit")
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "schema_revision": np.asarray(SCHEMA_REVISION, dtype=np.int64),
        "structure_ids": np.asarray(identities),
        "system_ids": np.asarray(
            tuple(str(frame.info["system_id"]) for frame in frames)
        ),
        "background_ids": np.asarray(background_ids),
        "type_map": np.asarray(elements),
        "candidate_presence": candidate_presence,
        "background_presence": background_presence,
        "descriptor_schema_json": np.asarray(_json(descriptor_schema)),
        "model_identity": np.asarray(str(model_identity)),
        "dataset_identity": np.asarray(str(dataset_identity)),
        "policy": np.asarray(policy),
        "policy_identity_json": np.asarray(
            _json(selector_identity(policy).as_dict())
        ),
        "requested_policy": np.asarray(requested),
        "effective_policy": np.asarray(effective),
        "fallback_reason": np.asarray(reason),
        "background_mode": np.asarray(background_mode),
        "candidate_rounds": np.asarray(rounds, dtype=np.int64),
        "group_ids": np.asarray(tuple(str(item) for item in group_ids)),
        "keep": np.asarray(int(keep), dtype=np.int64),
        "all_rn": np.asarray(bool(all_rn), dtype=np.bool_),
        "selected": np.asarray(
            [identity in set(selected) for identity in identities], dtype=np.bool_
        ),
        "selected_ids": np.asarray(selected),
        "outcomes_json": np.asarray(_json(outcomes)),
        "preparation_json": np.asarray(
            _json(
                {
                    "element_pooling": "mean",
                    "presence_coordinate": 1.0,
                    "composition_scale": "inverse-sqrt-present-elements",
                    "global_preprocessing": False,
                }
            )
        ),
    }
    for index, element in enumerate(elements):
        payload[f"candidate_element_{index:04d}"] = candidate_rows[element]
        payload[f"background_element_{index:04d}"] = background_rows[element]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def load_element_direct_fp_evidence(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {name: np.array(archive[name]) for name in archive.files}
    if (
        str(payload.get("schema", np.asarray("")).item()) != SCHEMA
        or int(payload.get("schema_revision", np.asarray(-1)).item())
        != SCHEMA_REVISION
    ):
        raise ValueError("terminal element DIRECT evidence schema is invalid")
    base = {
        "schema", "schema_revision", "structure_ids", "system_ids",
        "background_ids", "type_map", "candidate_presence",
        "background_presence", "descriptor_schema_json", "model_identity",
        "dataset_identity", "policy", "policy_identity_json",
        "requested_policy", "effective_policy", "fallback_reason",
        "background_mode", "candidate_rounds", "group_ids", "keep",
        "all_rn", "selected", "selected_ids", "outcomes_json",
        "preparation_json",
    }
    elements = tuple(str(item) for item in payload["type_map"].tolist())
    element_fields = {
        *(f"candidate_element_{index:04d}" for index in range(len(elements))),
        *(f"background_element_{index:04d}" for index in range(len(elements))),
    }
    if set(payload) != base | element_fields:
        raise ValueError("terminal element DIRECT evidence fields are invalid")
    count = len(payload["structure_ids"])
    background_count = len(payload["background_ids"])
    if (
        not elements
        or len(elements) != len(set(elements))
        or len(set(str(item) for item in payload["structure_ids"].tolist()))
        != count
        or len(set(str(item) for item in payload["background_ids"].tolist()))
        != background_count
        or payload["system_ids"].shape != (count,)
        or payload["selected"].shape != (count,)
        or payload["candidate_presence"].shape != (count, len(elements))
        or payload["background_presence"].shape
        != (background_count, len(elements))
    ):
        raise ValueError("terminal element DIRECT evidence alignment is invalid")
    schema = json.loads(str(payload["descriptor_schema_json"].item()))
    width = int(schema["width"])
    if width <= 0:
        raise ValueError("terminal element DIRECT descriptor width is invalid")
    for index in range(len(elements)):
        candidate = payload[f"candidate_element_{index:04d}"]
        background = payload[f"background_element_{index:04d}"]
        if (
            candidate.shape != (count, width)
            or background.shape != (background_count, width)
            or not np.isfinite(candidate).all()
            or not np.isfinite(background).all()
        ):
            raise ValueError("terminal element DIRECT element rows are invalid")
    policy = str(payload["policy"].item())
    mode = str(payload["background_mode"].item())
    if (
        policy not in {"ELEMENT_DIRECT", "ELEMENT_JOINT_DIRECT"}
        or json.loads(str(payload["policy_identity_json"].item()))
        != selector_identity(policy).as_dict()
        or str(payload["requested_policy"].item()) != policy
        or str(payload["effective_policy"].item()) != policy
        or str(payload["fallback_reason"].item()) != "none"
        or mode
        != ("teacher-training" if policy == "ELEMENT_JOINT_DIRECT" else "none")
        or (policy == "ELEMENT_JOINT_DIRECT") != bool(background_count)
    ):
        raise ValueError("terminal element DIRECT policy audit is invalid")
    preparation = json.loads(str(payload["preparation_json"].item()))
    if preparation != {
        "element_pooling": "mean",
        "presence_coordinate": 1.0,
        "composition_scale": "inverse-sqrt-present-elements",
        "global_preprocessing": False,
    }:
        raise ValueError("terminal element DIRECT preparation facts are invalid")
    json.loads(str(payload["outcomes_json"].item()))
    return payload


def element_rows_from_fp_payload(
    payload: Mapping[str, np.ndarray], *, background: bool
) -> tuple[tuple[dict[str, np.ndarray], ...], tuple[tuple[str, ...], ...]]:
    prefix = "background" if background else "candidate"
    identities = payload["background_ids" if background else "structure_ids"]
    elements = tuple(str(item) for item in payload["type_map"].tolist())
    presence = np.asarray(payload[f"{prefix}_presence"], dtype=bool)
    rows: list[dict[str, np.ndarray]] = []
    declarations: list[tuple[str, ...]] = []
    for row_index in range(len(identities)):
        declared = tuple(
            element
            for element_index, element in enumerate(elements)
            if bool(presence[row_index, element_index])
        )
        rows.append(
            {
                element: np.asarray(
                    payload[f"{prefix}_element_{element_index:04d}"][row_index]
                )[None, :]
                for element_index, element in enumerate(elements)
                if bool(presence[row_index, element_index])
            }
        )
        declarations.append(declared)
    return tuple(rows), tuple(declarations)


def _dense_element_rows(
    frames: Sequence[Atoms],
    *,
    identities: Sequence[str],
    element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    type_map: tuple[str, ...],
    width: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observed_widths = {
        int(np.asarray(vector).size)
        for identities_by_element in element_vectors.values()
        for vector in identities_by_element.values()
    }
    if width is None:
        if len(observed_widths) != 1:
            raise ValueError("terminal element DIRECT evidence width is inconsistent")
        resolved_width = next(iter(observed_widths))
    else:
        resolved_width = int(width)
        if observed_widths and observed_widths != {resolved_width}:
            raise ValueError("terminal element DIRECT evidence width is inconsistent")
    if resolved_width <= 0:
        raise ValueError("terminal element DIRECT evidence width must be positive")
    presence = np.zeros((len(frames), len(type_map)), dtype=np.bool_)
    rows = {
        element: np.zeros((len(frames), resolved_width), dtype=np.float32)
        for element in type_map
    }
    allowed = set(type_map)
    if not set(element_vectors).issubset(allowed):
        raise ValueError("terminal element DIRECT evidence contains unknown element")
    for row_index, (frame, identity) in enumerate(
        zip(frames, identities, strict=True)
    ):
        chemical = set(frame.get_chemical_symbols())
        if not chemical or not chemical.issubset(allowed):
            raise ValueError("terminal element DIRECT chemistry is outside type_map")
        for element_index, element in enumerate(type_map):
            has_evidence = str(identity) in element_vectors.get(element, {})
            if (element in chemical) != has_evidence:
                raise ValueError(
                    "terminal element DIRECT evidence must match chemical presence"
                )
            if not has_evidence:
                continue
            vector = np.asarray(
                element_vectors[element][str(identity)], dtype=np.float32
            )
            if vector.shape != (resolved_width,) or not np.isfinite(vector).all():
                raise ValueError("terminal element DIRECT vector is malformed")
            presence[row_index, element_index] = True
            rows[element][row_index] = vector
    return presence, rows


def _frame_ids(frames: Sequence[Atoms], label: str) -> tuple[str, ...]:
    identities = tuple(str(frame.info["structure_id"]) for frame in frames)
    if len(identities) != len(set(identities)):
        raise ValueError(f"terminal element DIRECT {label} order is not unique")
    return identities


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "SCHEMA",
    "SCHEMA_REVISION",
    "element_rows_from_fp_payload",
    "load_element_direct_fp_evidence",
    "publish_element_direct_fp_evidence",
]
