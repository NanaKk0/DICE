from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from workflow_artifacts.identities import RevisionedIdentity
from workflow_core.lifecycle.descriptor_stream import validated_readout_input_schema
from workflow_policies.identities import GMM_REFERENCE_TAIL


MANIFEST_NAME = "manifest.json"
FRAME_CALIBRATION_NAME = "frame_calibration.npz"
GMM_ROUTER_IDENTITY = RevisionedIdentity("dice.lifecycle.gmm-router", 1)
_HISTORICAL_GMM_ROUTER_SCHEMA = "dice.v2.gmm-router.v1"
_HISTORICAL_GMM_POLICY = "gmm-reference-tail@4"


def publish_gmm_router_bundle(
    router_dir: Path,
    *,
    manifest: Mapping[str, Any],
    element_arrays: Mapping[str, Mapping[str, np.ndarray]],
    frame_statistics: np.ndarray,
) -> None:
    """Publish a portable JSON + non-object NPZ router bundle."""
    target = Path(router_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        published_manifest = json.loads(json.dumps(dict(manifest)))
        for element, arrays in element_arrays.items():
            filename = str(manifest["elements"][element]["file"])
            element_path = staging / filename
            np.savez_compressed(element_path, **{key: np.asarray(value) for key, value in arrays.items()})
            published_manifest["elements"][element]["file_sha256"] = _sha256_file(element_path)
        np.savez_compressed(
            staging / FRAME_CALIBRATION_NAME,
            sorted_reference_frame_statistics=np.asarray(frame_statistics, dtype=np.float64),
        )
        published_manifest["frame_calibration"] = {
            "file": FRAME_CALIBRATION_NAME,
            "file_sha256": _sha256_file(staging / FRAME_CALIBRATION_NAME),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(published_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            raise FileExistsError(f"GMM router already exists: {target}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_gmm_router_bundle(
    router_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Load bundle arrays without pickle-capable deserialization."""
    root = Path(router_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("GMM router root must be a regular directory")
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = _normalized_manifest_identity(manifest)
    if (
        not GMM_ROUTER_IDENTITY.matches(
            manifest, name_field="schema", revision_field="schema_revision"
        )
        or manifest.get("policy_name") != GMM_REFERENCE_TAIL.name
        or manifest.get("policy_revision") != GMM_REFERENCE_TAIL.revision
    ):
        raise ValueError(
            "retired or malformed router artifact; rebuild with "
            "gmm-reference-tail using the gmm_reference operation role"
        )
    _validate_exact_bundle_inventory(root, manifest)
    element_arrays: dict[str, dict[str, np.ndarray]] = {}
    for element, metadata in dict(manifest.get("elements", {})).items():
        element_path = _bundle_file(root, metadata.get("file"))
        _validate_file_digest(element_path, metadata.get("file_sha256"))
        with np.load(element_path, allow_pickle=False) as payload:
            element_arrays[str(element)] = {name: np.asarray(payload[name]) for name in payload.files}
    frame_metadata = dict(manifest.get("frame_calibration", {}))
    frame_path = _bundle_file(root, frame_metadata.get("file"))
    _validate_file_digest(frame_path, frame_metadata.get("file_sha256"))
    with np.load(frame_path, allow_pickle=False) as payload:
        frame_statistics = np.asarray(payload["sorted_reference_frame_statistics"])
    _validate_loaded_bundle(manifest, element_arrays, frame_statistics)
    return manifest, element_arrays, frame_statistics


def _validate_loaded_bundle(
    manifest: Mapping[str, Any],
    element_arrays: Mapping[str, Mapping[str, np.ndarray]],
    frame_statistics: np.ndarray,
) -> None:
    required_manifest_keys = {
        "schema",
        "schema_revision",
        "policy_name",
        "policy_revision",
        "policy_constants",
        "descriptor_schema",
        "type_map",
        "reference_frames",
        "model_digest",
        "reference_digest",
        "software",
        "elements",
        "frame_calibration",
    }
    if set(manifest) != required_manifest_keys:
        raise ValueError("malformed GMM router manifest inventory")
    if not GMM_ROUTER_IDENTITY.matches(
        manifest, name_field="schema", revision_field="schema_revision"
    ):
        raise ValueError("malformed GMM router manifest schema")
    if (
        manifest.get("policy_name") != GMM_REFERENCE_TAIL.name
        or manifest.get("policy_revision") != GMM_REFERENCE_TAIL.revision
    ):
        raise ValueError("malformed GMM router policy id")
    if not all(isinstance(manifest.get(name), str) and manifest.get(name) for name in ("model_digest", "reference_digest")):
        raise ValueError("malformed GMM router provenance digests")
    software = manifest.get("software")
    if not isinstance(software, dict) or not all(
        isinstance(software.get(name), str) and software.get(name) for name in ("numpy", "scikit_learn")
    ):
        raise ValueError("malformed GMM router software provenance")
    schema = dict(manifest.get("descriptor_schema", {}))
    width = schema.get("width")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0 or schema.get("dtype") != "float32":
        raise ValueError("malformed GMM router descriptor schema")
    type_map = manifest.get("type_map")
    element_metadata = manifest.get("elements")
    if (
        not isinstance(type_map, list)
        or not type_map
        or len(set(type_map)) != len(type_map)
        or not isinstance(element_metadata, dict)
        or set(type_map) != set(element_metadata)
        or set(type_map) != set(element_arrays)
    ):
        raise ValueError("malformed GMM router element manifest")
    reference_frames = manifest.get("reference_frames")
    if not isinstance(reference_frames, int) or isinstance(reference_frames, bool) or reference_frames <= 0:
        raise ValueError("malformed GMM router reference frame count")
    _require_numeric_vector(
        np.asarray(frame_statistics),
        shape=(reference_frames,),
        label="malformed GMM router frame calibration",
        sorted_values=True,
    )
    if set(dict(manifest.get("frame_calibration", {}))) != {
        "file",
        "file_sha256",
    }:
        raise ValueError("malformed GMM router frame calibration inventory")

    required = {
        "scaler_mean",
        "scaler_scale",
        "pca_mean",
        "pca_components",
        "gmm_weights",
        "gmm_means",
        "gmm_covariances",
        "gmm_precisions_cholesky",
        "sorted_reference_nll",
    }
    for element in type_map:
        metadata = dict(element_metadata[element])
        arrays = element_arrays[element]
        prefix = f"malformed GMM router element {element!r}"
        if set(metadata) != {
            "file",
            "file_sha256",
            "reference_atoms",
            "feature_width",
            "computed_pca_dimensions",
            "pca_dimensions",
            "pca_explained_variance",
            "gmm_converged",
            "gmm_iterations",
        }:
            raise ValueError(prefix)
        latent = metadata.get("pca_dimensions")
        reference_atoms = metadata.get("reference_atoms")
        if (
            set(arrays) != required
            or not isinstance(latent, int)
            or isinstance(latent, bool)
            or not 1 <= latent <= width
            or not isinstance(reference_atoms, int)
            or isinstance(reference_atoms, bool)
            or reference_atoms < 8
            or metadata.get("feature_width") != width
            or metadata.get("gmm_converged") is not True
            or not isinstance(metadata.get("computed_pca_dimensions"), int)
            or not latent <= metadata["computed_pca_dimensions"] <= min(32, width, reference_atoms)
            or not isinstance(metadata.get("gmm_iterations"), int)
            or metadata["gmm_iterations"] <= 0
            or not isinstance(metadata.get("pca_explained_variance"), (int, float))
            or not 0.0 < float(metadata["pca_explained_variance"]) <= 1.0 + 1e-12
        ):
            raise ValueError(prefix)
        expected_shapes = {
            "scaler_mean": (width,),
            "scaler_scale": (width,),
            "pca_mean": (width,),
            "pca_components": (latent, width),
            "gmm_weights": (8,),
            "gmm_means": (8, latent),
            "gmm_covariances": (8, latent, latent),
            "gmm_precisions_cholesky": (8, latent, latent),
            "sorted_reference_nll": (reference_atoms,),
        }
        for name, shape in expected_shapes.items():
            _require_numeric_vector(
                np.asarray(arrays[name]),
                shape=shape,
                label=prefix,
                sorted_values=name == "sorted_reference_nll",
            )
        if np.any(np.asarray(arrays["scaler_scale"]) <= 0.0):
            raise ValueError(prefix)
        weights = np.asarray(arrays["gmm_weights"])
        if np.any(weights <= 0.0) or not np.isclose(float(np.sum(weights)), 1.0):
            raise ValueError(prefix)
        for covariance in np.asarray(arrays["gmm_covariances"]):
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError(prefix) from exc


def _validate_exact_bundle_inventory(
    root: Path, manifest: Mapping[str, Any]
) -> None:
    elements = manifest.get("elements")
    frame = manifest.get("frame_calibration")
    if not isinstance(elements, Mapping) or not isinstance(frame, Mapping):
        raise ValueError("malformed GMM router file inventory")
    expected = {MANIFEST_NAME, _bundle_file(root, frame.get("file")).name}
    for metadata in elements.values():
        if not isinstance(metadata, Mapping):
            raise ValueError("malformed GMM router element file inventory")
        expected.add(_bundle_file(root, metadata.get("file")).name)
    actual = {path.name for path in root.iterdir()}
    if actual != expected or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ValueError("GMM router directory inventory is not exact")


def _normalized_manifest_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = json.loads(json.dumps(dict(payload)))
    if (
        manifest.get("schema") == _HISTORICAL_GMM_ROUTER_SCHEMA
        and "schema_revision" not in manifest
        and manifest.get("policy_id") == _HISTORICAL_GMM_POLICY
        and "policy_name" not in manifest
        and "policy_revision" not in manifest
        and "policy" in manifest
        and "policy_constants" not in manifest
    ):
        manifest.update(
            GMM_ROUTER_IDENTITY.as_dict(
                name_field="schema", revision_field="schema_revision"
            )
        )
        manifest["policy_name"] = GMM_REFERENCE_TAIL.name
        manifest["policy_revision"] = GMM_REFERENCE_TAIL.revision
        manifest["policy_constants"] = manifest.pop("policy")
        manifest.pop("policy_id")
    descriptor = manifest.get("descriptor_schema")
    if isinstance(descriptor, Mapping):
        manifest["descriptor_schema"] = validated_readout_input_schema(descriptor)
    return manifest


def _require_numeric_vector(
    values: np.ndarray,
    *,
    shape: tuple[int, ...],
    label: str,
    sorted_values: bool = False,
) -> None:
    if values.shape != shape or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError(label)
    if sorted_values and np.any(values[1:] < values[:-1]):
        raise ValueError(label)


def _validate_file_digest(path: Path, expected: object) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or _sha256_file(path) != expected:
        raise ValueError(f"GMM router file digest mismatch: {path.name}")


def _bundle_file(root: Path, filename: object) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("malformed GMM router file path")
    return root / filename


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["GMM_ROUTER_IDENTITY", "load_gmm_router_bundle", "publish_gmm_router_bundle"]
