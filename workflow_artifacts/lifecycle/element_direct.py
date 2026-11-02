from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from ase import Atoms
import numpy as np

from workflow_policies.identities import GMM_REFERENCE_TAIL, selector_identity


SCHEMA = "dice.lifecycle.element-direct-screen"
SCHEMA_REVISION = 1


def publish_element_direct_screen_evidence(
    path: str | Path,
    *,
    observations: Sequence[Atoms],
    background_frames: Sequence[Atoms],
    candidate_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    background_element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    descriptor_schema: Mapping[str, object],
    type_map: Sequence[str],
    model_identity: str,
    dataset_identity: str,
    prescreen: str,
    pool: int,
    keep: int,
    screen_policy: str,
    fp_policy: str,
    route_kind: str,
    proposal_ids: Sequence[str],
    selector_candidate_ids: set[str],
    normal_train_ids: set[str],
    train_ids: set[str],
    fp_ids: set[str],
    teacher_metrics: Mapping[str, float],
    outcomes: Mapping[str, Mapping[str, object]],
    selector_outcomes: Mapping[str, Mapping[str, object]],
    route_thresholds: Mapping[str, object],
    readout_performed: bool = True,
) -> None:
    frames = tuple(observations)
    background = tuple(background_frames)
    elements = tuple(str(item) for item in type_map)
    if not elements or len(elements) != len(set(elements)):
        raise ValueError("element DIRECT evidence requires ordered unique type_map")
    structure_ids = _frame_ids(frames, "candidate")
    background_ids = _frame_ids(background, "background")
    system_ids = tuple(str(frame.info["system_id"]) for frame in frames)
    if not isinstance(readout_performed, bool):
        raise ValueError("element DIRECT readout flag must be boolean")
    if readout_performed:
        candidate_presence, candidate_rows = _dense_element_rows(
            frames,
            identities=structure_ids,
            element_vectors=candidate_element_vectors,
            type_map=elements,
        )
        background_presence, background_rows = _dense_element_rows(
            background,
            identities=background_ids,
            element_vectors=background_element_vectors,
            type_map=elements,
        )
    else:
        if candidate_element_vectors or background_element_vectors or descriptor_schema:
            raise ValueError("element DIRECT bypass cannot publish descriptor evidence")
        candidate_presence = _chemical_presence(frames, elements)
        background_presence = _chemical_presence(background, elements)
        candidate_rows = {
            element: np.empty((len(frames), 0), dtype=np.float32)
            for element in elements
        }
        background_rows = {
            element: np.empty((len(background), 0), dtype=np.float32)
            for element in elements
        }
    proposal = tuple(str(item) for item in proposal_ids)
    proposal_set = set(proposal)
    if not _ordered_subset(proposal, structure_ids):
        raise ValueError("element DIRECT proposal is not an ordered candidate subset")
    metric_ids = tuple(str(item) for item in teacher_metrics)
    if metric_ids != proposal:
        raise ValueError("element DIRECT teacher metrics must follow proposal order")
    metric_values = np.asarray(
        [float(teacher_metrics[item]) for item in metric_ids], dtype=np.float64
    )
    if not np.isfinite(metric_values).all():
        raise ValueError("element DIRECT teacher metrics must be finite")
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "schema_revision": np.asarray(SCHEMA_REVISION, dtype=np.int64),
        "structure_ids": np.asarray(structure_ids),
        "system_ids": np.asarray(system_ids),
        "background_ids": np.asarray(background_ids),
        "type_map": np.asarray(elements),
        "candidate_presence": candidate_presence,
        "background_presence": background_presence,
        "descriptor_schema_json": np.asarray(_json(descriptor_schema)),
        "readout_performed": np.asarray(readout_performed, dtype=np.bool_),
        "model_identity": np.asarray(str(model_identity)),
        "dataset_identity": np.asarray(str(dataset_identity)),
        "prescreen": np.asarray(str(prescreen)),
        "prescreen_policy_json": np.asarray(
            _json(
                GMM_REFERENCE_TAIL.as_dict()
                if str(prescreen) == "GMM"
                else selector_identity(str(prescreen)).as_dict()
            )
        ),
        "pool": np.asarray(pool, dtype=np.int64),
        "keep": np.asarray(keep, dtype=np.int64),
        "screen_policy": np.asarray(str(screen_policy)),
        "screen_policy_json": np.asarray(
            _json(selector_identity(str(screen_policy)).as_dict())
        ),
        "fp_policy": np.asarray(str(fp_policy)),
        "route_kind": np.asarray(str(route_kind)),
        "route_thresholds_json": np.asarray(_json(route_thresholds)),
        "proposal_ids": np.asarray(proposal),
        "student_pool": np.asarray(
            [item in proposal_set for item in structure_ids], dtype=np.bool_
        ),
        "normal_train_selected": np.asarray(
            [item in normal_train_ids for item in structure_ids], dtype=np.bool_
        ),
        "selector_candidate": np.asarray(
            [item in selector_candidate_ids for item in structure_ids], dtype=np.bool_
        ),
        "train_selected": np.asarray(
            [item in train_ids for item in structure_ids], dtype=np.bool_
        ),
        "fp_candidate": np.asarray(
            [item in fp_ids for item in structure_ids], dtype=np.bool_
        ),
        "teacher_metric_ids": np.asarray(metric_ids),
        "teacher_metrics": metric_values,
        "outcomes_json": np.asarray(_json(outcomes)),
        "selector_outcomes_json": np.asarray(_json(selector_outcomes)),
    }
    for index, element in enumerate(elements):
        payload[f"candidate_element_{index:04d}"] = candidate_rows[element]
        payload[f"background_element_{index:04d}"] = background_rows[element]
    np.savez(Path(path), **payload)


def load_element_direct_screen_evidence(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {name: np.array(archive[name]) for name in archive.files}
    base = {
        "schema", "schema_revision", "structure_ids", "system_ids",
        "background_ids", "type_map", "candidate_presence",
        "background_presence", "descriptor_schema_json", "readout_performed", "model_identity",
        "dataset_identity", "prescreen", "prescreen_policy_json", "pool",
        "keep", "screen_policy", "screen_policy_json", "fp_policy",
        "route_kind", "route_thresholds_json", "proposal_ids", "student_pool",
        "normal_train_selected", "selector_candidate", "train_selected", "fp_candidate",
        "teacher_metric_ids", "teacher_metrics", "outcomes_json",
        "selector_outcomes_json",
    }
    if str(payload.get("schema", np.asarray("")).item()) != SCHEMA or int(
        payload.get("schema_revision", np.asarray(-1)).item()
    ) != SCHEMA_REVISION:
        raise ValueError("element DIRECT screen evidence schema is invalid")
    elements = tuple(str(item) for item in payload["type_map"].tolist())
    element_fields = {
        *(f"candidate_element_{index:04d}" for index in range(len(elements))),
        *(f"background_element_{index:04d}" for index in range(len(elements))),
    }
    if set(payload) != base | element_fields:
        raise ValueError("element DIRECT screen evidence fields are invalid")
    count = len(payload["structure_ids"])
    background_count = len(payload["background_ids"])
    if payload["candidate_presence"].shape != (count, len(elements)) or payload[
        "background_presence"
    ].shape != (background_count, len(elements)):
        raise ValueError("element DIRECT screen presence alignment is invalid")
    if any(
        payload[name].shape != (count,)
        for name in (
            "system_ids", "student_pool", "normal_train_selected", "selector_candidate",
            "train_selected", "fp_candidate",
        )
    ):
        raise ValueError("element DIRECT screen candidate alignment is invalid")
    descriptor_schema = json.loads(str(payload["descriptor_schema_json"].item()))
    readout_performed = bool(payload["readout_performed"].item())
    width = int(descriptor_schema["width"]) if readout_performed else 0
    if readout_performed != bool(descriptor_schema):
        raise ValueError("element DIRECT readout flag disagrees with descriptor schema")
    for index in range(len(elements)):
        candidate = payload[f"candidate_element_{index:04d}"]
        background = payload[f"background_element_{index:04d}"]
        if candidate.shape != (count, width) or background.shape != (
            background_count,
            width,
        ) or not np.isfinite(candidate).all() or not np.isfinite(background).all():
            raise ValueError("element DIRECT screen element rows are invalid")
    for name in (
        "prescreen_policy_json", "screen_policy_json", "route_thresholds_json",
        "outcomes_json", "selector_outcomes_json",
    ):
        json.loads(str(payload[name].item()))
    return payload


def element_rows_from_payload(
    payload: Mapping[str, np.ndarray], *, background: bool
) -> tuple[tuple[dict[str, np.ndarray], ...], tuple[tuple[str, ...], ...]]:
    prefix = "background" if background else "candidate"
    identities = tuple(str(item) for item in payload[
        "background_ids" if background else "structure_ids"
    ].tolist())
    elements = tuple(str(item) for item in payload["type_map"].tolist())
    presence = np.asarray(payload[f"{prefix}_presence"], dtype=bool)
    rows: list[dict[str, np.ndarray]] = []
    declared: list[tuple[str, ...]] = []
    for row_index, _identity in enumerate(identities):
        frame_rows: dict[str, np.ndarray] = {}
        frame_presence = tuple(
            element
            for element_index, element in enumerate(elements)
            if presence[row_index, element_index]
        )
        for element_index, element in enumerate(elements):
            if presence[row_index, element_index]:
                frame_rows[element] = np.asarray(
                    payload[f"{prefix}_element_{element_index:04d}"][row_index]
                )[None, :]
        rows.append(frame_rows)
        declared.append(frame_presence)
    return tuple(rows), tuple(declared)


def _dense_element_rows(
    frames: Sequence[Atoms],
    *,
    identities: Sequence[str],
    element_vectors: Mapping[str, Mapping[str, np.ndarray]],
    type_map: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    schema_widths = {
        len(np.asarray(vector))
        for values in element_vectors.values()
        for vector in values.values()
    }
    if len(schema_widths) != 1:
        raise ValueError("element DIRECT evidence must share one feature width")
    width = next(iter(schema_widths), 0)
    if not width and frames:
        raise ValueError("element DIRECT evidence is empty")
    presence = np.zeros((len(frames), len(type_map)), dtype=np.bool_)
    rows = {
        element: np.zeros((len(frames), width), dtype=np.float32)
        for element in type_map
    }
    for row_index, (frame, identity) in enumerate(zip(frames, identities, strict=True)):
        chemical = set(frame.get_chemical_symbols())
        if not chemical or not chemical.issubset(type_map):
            raise ValueError("element DIRECT chemical composition is invalid")
        for element_index, element in enumerate(type_map):
            has_element = element in chemical
            has_evidence = identity in element_vectors.get(element, {})
            if has_element != has_evidence:
                raise ValueError("element DIRECT evidence disagrees with chemical presence")
            if has_element:
                vector = np.asarray(element_vectors[element][identity], dtype=np.float32)
                if vector.shape != (width,) or not np.isfinite(vector).all():
                    raise ValueError("element DIRECT evidence vector is invalid")
                presence[row_index, element_index] = True
                rows[element][row_index] = vector
    return presence, rows


def _frame_ids(frames: Sequence[Atoms], label: str) -> tuple[str, ...]:
    identities = tuple(str(frame.info.get("structure_id", "")) for frame in frames)
    if any(not item for item in identities) or len(identities) != len(set(identities)):
        raise ValueError(f"element DIRECT {label} identities must be ordered unique")
    return identities


def _chemical_presence(
    frames: Sequence[Atoms], type_map: tuple[str, ...]
) -> np.ndarray:
    presence = np.zeros((len(frames), len(type_map)), dtype=np.bool_)
    for row_index, frame in enumerate(frames):
        chemical = set(frame.get_chemical_symbols())
        if not chemical or not chemical.issubset(type_map):
            raise ValueError("element DIRECT chemical composition is invalid")
        for element_index, element in enumerate(type_map):
            presence[row_index, element_index] = element in chemical
    return presence


def _ordered_subset(values: Sequence[str], reference: Sequence[str]) -> bool:
    positions = {item: index for index, item in enumerate(reference)}
    try:
        indices = tuple(positions[item] for item in values)
    except KeyError:
        return False
    return len(indices) == len(set(indices)) and all(
        left < right for left, right in zip(indices, indices[1:])
    )


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


__all__ = [
    "SCHEMA",
    "SCHEMA_REVISION",
    "element_rows_from_payload",
    "load_element_direct_screen_evidence",
    "publish_element_direct_screen_evidence",
]
