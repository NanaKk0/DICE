from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms

from workflow_core.lifecycle.descriptor_stream import (
    AtomDescriptorChunk,
    DescriptorStreamHeader,
    validated_descriptor_chunks,
)


@dataclass(frozen=True, slots=True)
class VisualizationDraft:
    """Borrow existing descriptor rows; packing runs in the queue feeder, not screen."""
    candidate_ids: tuple[str, ...]
    background_ids: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    candidate_elements: Mapping[str, Mapping[str, np.ndarray]]
    background_elements: Mapping[str, Mapping[str, np.ndarray]]
    descriptor_schema: Mapping[str, object]
    type_map: tuple[str, ...]
    model_identity: str
    dataset_identity: str
    candidate_vectors: Mapping[str, np.ndarray] | None = None
    background_vectors: Mapping[str, np.ndarray] | None = None

    def packed(self):
        def pack(elements):
            return {element: (tuple(rows), np.stack(tuple(rows.values())).astype(np.float32, copy=False))
                    for element, rows in elements.items() if rows}
        structures = None
        if self.candidate_vectors is not None or self.background_vectors is not None:
            if self.candidate_vectors is None or self.background_vectors is None:
                raise ValueError("structure vectors must cover both populations")
            structures = tuple(
                structure_rows_from_vectors(ids, vectors, width=int(self.descriptor_schema["width"]))
                for ids, vectors in ((self.candidate_ids, self.candidate_vectors),
                                     (self.background_ids, self.background_vectors))
            )
        return dict(structure_populations=structures, candidate_ids=self.candidate_ids, background_ids=self.background_ids,
                    proposal_ids=self.proposal_ids, candidate_elements=pack(self.candidate_elements),
                    background_elements=pack(self.background_elements),
                    descriptor_schema=dict(self.descriptor_schema), type_map=self.type_map,
                    model_identity=self.model_identity, dataset_identity=self.dataset_identity)

    def __reduce__(self):
        # multiprocessing.Queue serializes asynchronously. Never send Atoms,
        # calculators, an observation source, or unpicklable MappingProxyTypes.
        try:
            payload = self.packed()
        except Exception as exc:
            payload = {'packing_error': f'{type(exc).__name__}: {exc}'}
        return dict, (payload,)


def visualization_from_draft(payload, selected_ids):
    if 'packing_error' in payload:
        raise ValueError(payload['packing_error'])
    def population(key, ids_key):
        ids = payload[ids_key]
        expected = set(ids)
        result = {}
        covered = set()
        for element, (row_ids, vectors) in payload[key].items():
            if element not in payload['type_map']:
                raise ValueError('visualization element outside type map')
            membership = set(row_ids)
            if not membership <= expected or tuple(i for i in ids if i in membership) != tuple(row_ids):
                raise ValueError('visualization element identity/order mismatch')
            result[element] = ElementFrameRows(tuple(row_ids), vectors)
            covered.update(row_ids)
        if covered != expected:
            raise ValueError('visualization descriptor population is incomplete')
        return result
    return build_visualization_request(kind='screen', layers=(
        VisualizationLayer('sample', payload['candidate_ids']),
        VisualizationLayer('pool', payload['proposal_ids']),
        VisualizationLayer('train', payload['background_ids']),
        VisualizationLayer('select', tuple(selected_ids))),
        populations=(population('candidate_elements', 'candidate_ids'),
                     population('background_elements', 'background_ids')),
        structure_populations=payload.get('structure_populations'),
        descriptor_schema=payload['descriptor_schema'], type_map=payload['type_map'],
        model_identity=payload['model_identity'], dataset_identity=payload['dataset_identity'])


@dataclass(frozen=True, slots=True)
class ElementFrameRows:
    frame_ids: tuple[str, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        identities = tuple(str(item) for item in self.frame_ids)
        vectors = np.asarray(self.vectors, dtype=np.float32)
        if (
            any(not item for item in identities)
            or len(set(identities)) != len(identities)
            or vectors.ndim != 2
            or vectors.shape[0] != len(identities)
            or vectors.shape[1] <= 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("element descriptor rows are malformed")
        object.__setattr__(self, "frame_ids", identities)
        object.__setattr__(self, "vectors", np.ascontiguousarray(vectors))


@dataclass(frozen=True, slots=True)
class VisualizationLayer:
    name: str
    frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        name = str(self.name)
        identities = tuple(str(item) for item in self.frame_ids)
        if (
            not name
            or any(not item for item in identities)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("visualization layer identities are malformed")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frame_ids", identities)


@dataclass(frozen=True, slots=True)
class VisualizationRequest:
    kind: str
    layers: tuple[VisualizationLayer, ...]
    elements: Mapping[str, ElementFrameRows]
    descriptor_schema: Mapping[str, object]
    type_map: tuple[str, ...]
    model_identity: str
    dataset_identity: str
    source_digest: str
    structure: ElementFrameRows | None = None


def aggregate_element_frame_means(
    header: DescriptorStreamHeader,
    chunks: Iterable[AtomDescriptorChunk],
) -> Mapping[str, ElementFrameRows]:
    collected: dict[str, list[tuple[int, str, np.ndarray]]] = {
        element: [] for element in header.type_map
    }
    for chunk in validated_descriptor_chunks(header, chunks):
        for local, (ordinal, identity) in enumerate(
            zip(chunk.frame_ordinals, chunk.frame_ids, strict=True)
        ):
            start = int(chunk.frame_ptr[local])
            stop = int(chunk.frame_ptr[local + 1])
            frame_types = chunk.type_indices[start:stop]
            frame_rows = chunk.descriptors[start:stop]
            for type_index in dict.fromkeys(int(item) for item in frame_types):
                mask = frame_types == type_index
                mean = np.asarray(frame_rows[mask].mean(axis=0), dtype=np.float32)
                collected[header.type_map[type_index]].append(
                    (int(ordinal), identity, mean)
                )
    result: dict[str, ElementFrameRows] = {}
    for element, rows in collected.items():
        if not rows:
            continue
        rows.sort(key=lambda item: item[0])
        result[element] = ElementFrameRows(
            tuple(item[1] for item in rows),
            np.stack([item[2] for item in rows]).astype(np.float32, copy=False),
        )
    return MappingProxyType(result)


def structure_rows_from_vectors(
    identities: Sequence[str], vectors: Mapping[str, np.ndarray], *, width: int,
) -> ElementFrameRows:
    """Retain existing all-atom means without averaging elements or replaying models."""
    ordered = tuple(identities)
    if tuple(vectors) != ordered:
        raise ValueError("structure descriptor population identity/order mismatch")
    values = (np.stack([vectors[identity] for identity in ordered]) if ordered
              else np.empty((0, width), dtype=np.float32))
    rows = ElementFrameRows(ordered, values)
    if rows.vectors.shape[1] != width:
        raise ValueError("structure descriptor width mismatch")
    return rows


def element_rows_from_frame_vectors(
    frames: Sequence[Atoms],
    vectors: Mapping[str, np.ndarray],
    *,
    descriptor_schema: Mapping[str, object],
    type_map: Sequence[str],
) -> Mapping[str, ElementFrameRows]:
    width = int(descriptor_schema.get("width", 0))
    ordered_type_map = tuple(str(item) for item in type_map)
    rows: dict[str, list[tuple[str, np.ndarray]]] = {
        item: [] for item in ordered_type_map
    }
    identities: list[str] = []
    for frame in frames:
        identity = str(frame.info.get("structure_id", ""))
        if not identity or identity in identities or identity not in vectors:
            raise ValueError("frame-vector visualization identities are incomplete")
        vector = np.asarray(vectors[identity], dtype=np.float32)
        if vector.shape != (width,) or not np.isfinite(vector).all():
            raise ValueError("frame-vector visualization rows are malformed")
        identities.append(identity)
        for element in dict.fromkeys(frame.get_chemical_symbols()):
            if element not in rows:
                raise ValueError("frame element is outside visualization type_map")
            rows[element].append((identity, vector))
    if set(vectors) != set(identities):
        raise ValueError("frame-vector visualization contains unexpected identities")
    return MappingProxyType(
        {
            element: ElementFrameRows(
                tuple(identity for identity, _vector in values),
                np.stack([vector for _identity, vector in values]),
            )
            for element, values in rows.items()
            if values
        }
    )


def element_rows_from_element_vectors(
    frames: Iterable[Atoms],
    vectors: Mapping[str, Mapping[str, np.ndarray]],
    *,
    descriptor_schema: Mapping[str, object],
    type_map: Sequence[str],
) -> Mapping[str, ElementFrameRows]:
    """Validate exact per-frame means over atoms of each present element."""

    width = int(descriptor_schema.get("width", 0))
    ordered_type_map = tuple(str(item) for item in type_map)
    expected: dict[str, list[str]] = {element: [] for element in ordered_type_map}
    seen: set[str] = set()
    for frame in frames:
        identity = str(frame.info.get("structure_id", ""))
        if not identity or identity in seen:
            raise ValueError("element-vector visualization identities are incomplete")
        seen.add(identity)
        for element in dict.fromkeys(frame.get_chemical_symbols()):
            if element not in expected:
                raise ValueError("frame element is outside visualization type_map")
            expected[element].append(identity)
    if set(vectors) != {element for element, ids in expected.items() if ids}:
        raise ValueError("element-vector visualization changed present elements")
    result: dict[str, ElementFrameRows] = {}
    for element, identities in expected.items():
        if not identities:
            continue
        element_vectors = vectors[element]
        if tuple(element_vectors) != tuple(identities):
            raise ValueError("element-vector visualization changed canonical order")
        rows = np.stack(
            [np.asarray(element_vectors[identity], dtype=np.float32) for identity in identities]
        )
        if rows.shape != (len(identities), width) or not np.isfinite(rows).all():
            raise ValueError("element-vector visualization rows are malformed")
        result[element] = ElementFrameRows(tuple(identities), rows)
    return MappingProxyType(result)


def build_visualization_request(
    *,
    kind: str,
    layers: Sequence[VisualizationLayer],
    populations: Sequence[Mapping[str, ElementFrameRows]],
    structure_populations: Sequence[ElementFrameRows] | None = None,
    descriptor_schema: Mapping[str, object],
    type_map: Sequence[str],
    model_identity: str,
    dataset_identity: str,
) -> VisualizationRequest:
    if kind not in {"screen", "fp"}:
        raise ValueError("visualization kind must be screen or fp")
    ordered_layers = tuple(layers)
    expected_layer_names = (
        ("sample", "pool", "train", "select")
        if kind == "screen"
        else ("pool", "train", "select")
    )
    if tuple(layer.name for layer in ordered_layers) != expected_layer_names:
        raise ValueError("visualization layers do not match the diagnostic kind")
    schema = dict(descriptor_schema)
    width = int(schema.get("width", 0))
    merged: dict[str, dict[str, np.ndarray]] = {}
    order: dict[str, list[str]] = {}
    for population in populations:
        for element, element_rows in population.items():
            if element_rows.vectors.shape[1] != width:
                raise ValueError("visualization populations changed descriptor width")
            owned = merged.setdefault(str(element), {})
            owned_order = order.setdefault(str(element), [])
            for identity, vector in zip(
                element_rows.frame_ids, element_rows.vectors, strict=True
            ):
                if identity in owned:
                    if not np.array_equal(owned[identity], vector):
                        raise ValueError("visualization identity has conflicting descriptors")
                    continue
                owned[identity] = vector
                owned_order.append(identity)
    elements = {
        element: ElementFrameRows(
            tuple(order[element]),
            np.stack([values[identity] for identity in order[element]]),
        )
        for element, values in merged.items()
    }
    available = {identity for rows in elements.values() for identity in rows.frame_ids}
    missing = [
        (layer.name, identity)
        for layer in ordered_layers
        for identity in layer.frame_ids
        if identity not in available
    ]
    if missing:
        raise ValueError(
            "visualization layer identity has no descriptor row: "
            f"{missing[0][0]}={missing[0][1]}"
        )
    structure = None
    if structure_populations is not None:
        structure_vectors: dict[str, np.ndarray] = {}
        for rows in structure_populations:
            if rows.vectors.shape[1] != width:
                raise ValueError("structure descriptor width mismatch")
            for identity, vector in zip(rows.frame_ids, rows.vectors, strict=True):
                if identity in structure_vectors:
                    if not np.array_equal(structure_vectors[identity], vector):
                        raise ValueError("structure identity has conflicting descriptors")
                else:
                    structure_vectors[identity] = vector
        expected_ids = tuple(dict.fromkeys(
            identity for layer in ordered_layers for identity in layer.frame_ids
        ))
        if tuple(structure_vectors) != expected_ids or set(structure_vectors) != available:
            raise ValueError("structure descriptor population identity/order mismatch")
        structure = ElementFrameRows(
            expected_ids,
            np.stack(tuple(structure_vectors.values())) if expected_ids
            else np.empty((0, width), dtype=np.float32),
        )
    type_map_tuple = tuple(str(item) for item in type_map)
    metadata = {
        "kind": kind,
        "layers": [
            {"name": layer.name, "frame_ids": list(layer.frame_ids)}
            for layer in ordered_layers
        ],
        "descriptor_schema": schema,
        "type_map": list(type_map_tuple),
        "model_identity": str(model_identity),
        "dataset_identity": str(dataset_identity),
        "elements": {
            element: list(rows.frame_ids) for element, rows in elements.items()
        },
    }
    if structure is not None:
        metadata["structure"] = list(structure.frame_ids)
        metadata["structure_aggregation"] = "all-atom-mean"
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for element in type_map_tuple:
        rows = elements.get(element)
        if rows is not None:
            digest.update(rows.vectors.tobytes(order="C"))
    if structure is not None:
        digest.update(structure.vectors.tobytes(order="C"))
    return VisualizationRequest(
        kind=kind,
        layers=ordered_layers,
        elements=MappingProxyType(elements),
        descriptor_schema=MappingProxyType(schema),
        type_map=type_map_tuple,
        model_identity=str(model_identity),
        dataset_identity=str(dataset_identity),
        source_digest=digest.hexdigest(),
        structure=structure,
    )


__all__ = [
    "ElementFrameRows",
    "VisualizationLayer",
    "VisualizationRequest",
    "structure_rows_from_vectors",
    "aggregate_element_frame_means",
    "build_visualization_request",
    "element_rows_from_element_vectors",
    "element_rows_from_frame_vectors",
]
