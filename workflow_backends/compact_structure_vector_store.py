from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from workflow_backends.descriptor_stream import validated_readout_input_schema


SCHEMA = "dice.compact-structure-vectors"
VERSION = 1


def write_compact_structure_vector_store(
    root: str | Path,
    *,
    descriptor_schema: Mapping[str, object],
    frame_ids: Sequence[str],
    frame_means: np.ndarray,
    model_digest: str,
    element_means: Mapping[str, tuple[Sequence[str], np.ndarray]] | None = None,
) -> dict[str, object]:
    """Persist canonical structure vectors and return one bounded receipt."""

    target = Path(root).resolve()
    staging = target.with_name(f".{target.name}.tmp")
    schema = validated_readout_input_schema(descriptor_schema)
    identities = tuple(str(item) for item in frame_ids)
    means = np.asarray(frame_means, dtype=np.float32)
    if (
        not identities
        or any(not item for item in identities)
        or len(set(identities)) != len(identities)
        or means.shape != (len(identities), int(schema["width"]))
        or not np.isfinite(means).all()
        or not isinstance(model_digest, str)
        or not model_digest
    ):
        raise ValueError("compact structure-vector evidence is invalid")
    arrays = {
        "frame_ids": np.asarray(identities, dtype=np.str_),
        "frame_means": np.ascontiguousarray(means),
    }
    element_evidence: dict[str, dict[str, object]] = {}
    for element, raw_values in sorted((element_means or {}).items()):
        element_ids = tuple(str(item) for item in raw_values[0])
        element_rows = np.asarray(raw_values[1], dtype=np.float32)
        if (
            not element
            or not element_ids
            or any(identity not in identities for identity in element_ids)
            or len(set(element_ids)) != len(element_ids)
            or element_rows.shape != (len(element_ids), int(schema["width"]))
            or not np.isfinite(element_rows).all()
        ):
            raise ValueError("compact element structure-vector evidence is invalid")
        arrays[f"element_{element}_frame_ids"] = np.asarray(
            element_ids, dtype=np.str_
        )
        arrays[f"element_{element}_vectors"] = np.ascontiguousarray(element_rows)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    evidence: dict[str, object] = {}
    for name, array in arrays.items():
        path = staging / f"{name}.npy"
        with path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        evidence[name] = {
            "file": path.name,
            "dtype": np.dtype(array.dtype).str,
            "shape": list(array.shape),
            "sha256": _digest(path),
        }
    for element in sorted(element_means or {}):
        element_evidence[element] = {
            "frame_ids": evidence.pop(f"element_{element}_frame_ids"),
            "vectors": evidence.pop(f"element_{element}_vectors"),
        }
    receipt = {
        "schema": SCHEMA,
        "version": VERSION,
        "root": str(target),
        "frame_count": len(identities),
        "model_digest": model_digest,
        "descriptor_schema": schema,
        "arrays": evidence,
        "elements": element_evidence,
    }
    (staging / "receipt.json").write_text(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    return receipt


def read_compact_structure_vector_store(
    receipt: Mapping[str, object],
    *,
    expected_root: str | Path,
    expected_model_digest: str,
    include_elements: bool = False,
) -> tuple[dict[str, object], tuple[str, ...], np.ndarray] | tuple[
    dict[str, object],
    tuple[str, ...],
    np.ndarray,
    dict[str, tuple[tuple[str, ...], np.ndarray]],
]:
    """Validate and materialize one compact structure-vector worker result."""

    value = dict(receipt)
    root = Path(str(value.get("root", ""))).resolve()
    expected = Path(expected_root).resolve()
    count = value.get("frame_count")
    if (
        value.get("schema") != SCHEMA
        or value.get("version") != VERSION
        or root != expected
        or value.get("model_digest") != expected_model_digest
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise ValueError("compact structure-vector receipt identity is invalid")
    schema = validated_readout_input_schema(
        dict(value.get("descriptor_schema", {}))
    )
    raw_arrays = value.get("arrays")
    if not isinstance(raw_arrays, dict) or set(raw_arrays) != {
        "frame_ids",
        "frame_means",
    }:
        raise ValueError("compact structure-vector receipt arrays are invalid")
    identity_array = _read_array(root, raw_arrays["frame_ids"])
    means = _read_array(root, raw_arrays["frame_means"])
    if identity_array.shape != (count,) or identity_array.dtype.kind != "U":
        raise ValueError("compact structure-vector identities are invalid")
    identities = tuple(str(item) for item in identity_array.tolist())
    if any(not item for item in identities) or len(set(identities)) != count:
        raise ValueError("compact structure-vector identities are invalid")
    expected_shape = (count, int(schema["width"]))
    if means.dtype != np.float32 or means.shape != expected_shape or not np.isfinite(means).all():
        raise ValueError("compact structure-vector means are invalid")
    result = (
        schema,
        identities,
        np.array(means, dtype=np.float32, order="C", copy=True),
    )
    if not include_elements:
        return result
    raw_elements = value.get("elements", {})
    if not isinstance(raw_elements, dict):
        raise ValueError("compact element structure-vector receipt is invalid")
    elements: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for element, raw_element in raw_elements.items():
        if not isinstance(raw_element, dict) or set(raw_element) != {
            "frame_ids",
            "vectors",
        }:
            raise ValueError("compact element structure-vector receipt is invalid")
        element_ids_array = _read_array(root, raw_element["frame_ids"])
        element_rows = _read_array(root, raw_element["vectors"])
        element_ids = tuple(str(item) for item in element_ids_array.tolist())
        if (
            element_ids_array.dtype.kind != "U"
            or any(identity not in identities for identity in element_ids)
            or len(set(element_ids)) != len(element_ids)
            or element_rows.dtype != np.float32
            or element_rows.shape != (len(element_ids), int(schema["width"]))
            or not np.isfinite(element_rows).all()
        ):
            raise ValueError("compact element structure vectors are invalid")
        elements[str(element)] = (
            element_ids,
            np.array(element_rows, dtype=np.float32, order="C", copy=True),
        )
    return (*result, elements)


def _read_array(root: Path, raw_evidence: object) -> np.ndarray:
    if not isinstance(raw_evidence, dict) or set(raw_evidence) != {
        "file",
        "dtype",
        "shape",
        "sha256",
    }:
        raise ValueError("compact structure-vector array evidence is invalid")
    filename = raw_evidence.get("file")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("compact structure-vector array path is invalid")
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file() or _digest(path) != raw_evidence.get("sha256"):
        raise ValueError("compact structure-vector array digest is invalid")
    array = np.load(path, allow_pickle=False)
    if (
        np.dtype(array.dtype).str != raw_evidence.get("dtype")
        or list(array.shape) != raw_evidence.get("shape")
    ):
        raise ValueError("compact structure-vector array metadata is invalid")
    return array


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "SCHEMA",
    "VERSION",
    "read_compact_structure_vector_store",
    "write_compact_structure_vector_store",
]
