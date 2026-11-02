from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

import numpy as np

from workflow_core.identities import RevisionedIdentity
from workflow_policies.identities import VALIDATED_FINAL_LAYER_NAME


READOUT_INPUT_IDENTITY = RevisionedIdentity("readout-input", 1)
_HISTORICAL_READOUT_INPUT_NAME = "readout-input@1"
_HISTORICAL_VALIDATED_FINAL_LAYER_STRATEGY = "validated-final-layer-name@1"


def validated_readout_input_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Validate the backend-neutral, logic-owned readout-input contract."""
    value = copy.deepcopy(dict(schema))
    if value.get("name") == _HISTORICAL_READOUT_INPUT_NAME and "revision" not in value:
        value.update(READOUT_INPUT_IDENTITY.as_dict())
    if (
        not READOUT_INPUT_IDENTITY.matches(value)
        or not isinstance(value.get("backend"), str)
        or not value["backend"]
        or not isinstance(value.get("model_family"), str)
        or not value["model_family"]
        or value.get("definition") != "atomic-energy readout inputs"
        or value.get("dtype") != "float32"
        or value.get("normalized") is not False
        or value.get("constant_terms_included") is not False
        or value.get("deduplication") != "unique-native-input"
    ):
        raise ValueError(
            "reference-tail routing requires a structural readout-input schema"
        )
    width = value.get("width")
    components = value.get("components")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(components, list)
        or not components
    ):
        raise ValueError("readout-input schema requires positive component coverage")
    expected_offset = 0
    ids: set[str] = set()
    native_sources: set[str] = set()
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("readout-input schema contains an invalid component")
        component_id = component.get("id")
        native_source = component.get("native_source")
        offset = component.get("offset")
        component_width = component.get("width")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in ids
            or not isinstance(native_source, str)
            or not native_source
            or native_source in native_sources
            or component.get("invariance") != "scalar"
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset != expected_offset
            or not isinstance(component_width, int)
            or isinstance(component_width, bool)
            or component_width <= 0
        ):
            raise ValueError("readout-input schema component coverage is invalid")
        ids.add(component_id)
        native_sources.add(native_source)
        expected_offset += component_width
    if expected_offset != width:
        raise ValueError("readout-input schema components do not cover its width")
    resolver = value.get("resolver")
    if resolver is not None:
        if not isinstance(resolver, dict):
            raise ValueError("readout-input resolver evidence is invalid")
        if (
            resolver.get("strategy") == _HISTORICAL_VALIDATED_FINAL_LAYER_STRATEGY
            and "strategy_name" not in resolver
            and "strategy_revision" not in resolver
        ):
            resolver.pop("strategy")
            resolver.update(
                VALIDATED_FINAL_LAYER_NAME.as_dict(
                    name_field="strategy_name",
                    revision_field="strategy_revision",
                )
            )
        if (
            resolver.get("strategy_name") != VALIDATED_FINAL_LAYER_NAME.name
            or resolver.get("strategy_revision") != VALIDATED_FINAL_LAYER_NAME.revision
            or not isinstance(resolver.get("energy_tensor"), str)
            or not resolver["energy_tensor"]
        ):
            raise ValueError("readout-input resolver evidence is invalid")
    return value


@dataclass(frozen=True, slots=True)
class DescriptorStreamHeader:
    schema: Mapping[str, object]
    type_map: tuple[str, ...]
    frame_count: int

    def __post_init__(self) -> None:
        schema = dict(self.schema)
        type_map = tuple(str(item) for item in self.type_map)
        if (
            schema.get("dtype") != "float32"
            or not isinstance(schema.get("width"), int)
            or schema["width"] <= 0
        ):
            raise ValueError(
                "descriptor stream schema requires a positive float32 feature width"
            )
        if not type_map or any(not item for item in type_map) or len(set(type_map)) != len(type_map):
            raise ValueError(
                "descriptor stream type_map must contain unique non-empty elements"
            )
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count < 0
        ):
            raise ValueError(
                "descriptor stream frame_count must be a non-negative integer"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "type_map", type_map)


@dataclass(frozen=True, slots=True)
class AtomDescriptorChunk:
    descriptors: np.ndarray
    type_indices: np.ndarray
    frame_ptr: np.ndarray
    frame_ordinals: np.ndarray
    frame_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        descriptors = np.asarray(self.descriptors)
        type_indices = np.asarray(self.type_indices)
        frame_ptr = np.asarray(self.frame_ptr)
        frame_ordinals = np.asarray(self.frame_ordinals)
        frame_ids = tuple(str(item) for item in self.frame_ids)
        if (
            descriptors.dtype != np.float32
            or descriptors.ndim != 2
            or descriptors.shape[1] <= 0
            or not np.all(np.isfinite(descriptors))
        ):
            raise ValueError(
                "descriptor chunks require finite float32 atom x feature rows"
            )
        if type_indices.shape != (len(descriptors),) or not np.issubdtype(
            type_indices.dtype, np.integer
        ):
            raise ValueError("descriptor chunk type indices must match atom rows")
        if frame_ptr.ndim != 1 or not np.issubdtype(frame_ptr.dtype, np.integer):
            raise ValueError(
                "descriptor chunk frame pointers must be an integer vector"
            )
        if frame_ordinals.ndim != 1 or not np.issubdtype(
            frame_ordinals.dtype, np.integer
        ):
            raise ValueError(
                "descriptor chunk frame ordinals must be an integer vector"
            )
        if len(frame_ids) != len(frame_ordinals) or len(frame_ptr) != len(frame_ordinals) + 1:
            raise ValueError(
                "descriptor chunk frame pointers, ordinals and identities are inconsistent"
            )
        frame_ptr = frame_ptr.astype(np.int64, copy=False)
        frame_ordinals = frame_ordinals.astype(np.int64, copy=False)
        if len(frame_ids) == 0:
            if len(descriptors) != 0 or frame_ptr.shape != (1,) or int(frame_ptr[0]) != 0:
                raise ValueError(
                    "an empty descriptor chunk must contain no atom rows"
                )
        elif (
            int(frame_ptr[0]) != 0
            or int(frame_ptr[-1]) != len(descriptors)
            or np.any(np.diff(frame_ptr) <= 0)
        ):
            raise ValueError(
                "descriptor chunk frame pointers must delimit complete non-empty frames"
            )
        if any(not identity for identity in frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError(
                "descriptor chunk frame identities must be unique and non-empty"
            )
        if len(set(frame_ordinals.tolist())) != len(frame_ordinals):
            raise ValueError("descriptor chunk frame ordinals must be unique")
        object.__setattr__(self, "descriptors", np.ascontiguousarray(descriptors))
        object.__setattr__(
            self, "type_indices", type_indices.astype(np.int32, copy=False)
        )
        object.__setattr__(self, "frame_ptr", frame_ptr)
        object.__setattr__(self, "frame_ordinals", frame_ordinals)
        object.__setattr__(self, "frame_ids", frame_ids)


def validated_descriptor_chunks(
    header: DescriptorStreamHeader, chunks: Iterable[AtomDescriptorChunk]
) -> Iterator[AtomDescriptorChunk]:
    seen_ordinals: set[int] = set()
    seen_ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, AtomDescriptorChunk):
            raise TypeError("descriptor stream yielded a non-chunk value")
        if chunk.descriptors.shape[1] != header.schema["width"]:
            raise ValueError("descriptor chunk width does not match stream schema")
        if np.any(chunk.type_indices < 0) or np.any(
            chunk.type_indices >= len(header.type_map)
        ):
            raise ValueError(
                "descriptor chunk type index is outside the stream type_map"
            )
        if np.any(chunk.frame_ordinals < 0) or np.any(
            chunk.frame_ordinals >= header.frame_count
        ):
            raise ValueError(
                "descriptor chunk frame ordinal is outside canonical frame order"
            )
        ordinals = set(int(item) for item in chunk.frame_ordinals)
        identities = set(chunk.frame_ids)
        if seen_ordinals & ordinals:
            raise ValueError(
                "descriptor stream contains a duplicate canonical frame ordinal"
            )
        if seen_ids & identities:
            raise ValueError(
                "descriptor stream contains a duplicate canonical frame identity"
            )
        seen_ordinals.update(ordinals)
        seen_ids.update(identities)
        yield chunk
    if seen_ordinals != set(range(header.frame_count)) or len(seen_ids) != header.frame_count:
        raise ValueError("descriptor stream is missing canonical frame membership")


__all__ = [
    "READOUT_INPUT_IDENTITY",
    "AtomDescriptorChunk",
    "DescriptorStreamHeader",
    "validated_descriptor_chunks",
    "validated_readout_input_schema",
]
