from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from workflow_artifacts.lifecycle.gmm_router import load_gmm_router_bundle, publish_gmm_router_bundle
from workflow_core.lifecycle.descriptor_stream import (
    AtomDescriptorChunk,
    DescriptorStreamHeader,
    validated_descriptor_chunks,
    validated_readout_input_schema,
)
from workflow_artifacts.lifecycle.gmm_router import GMM_ROUTER_IDENTITY
from workflow_policies.identities import GMM_REFERENCE_TAIL


POLICY_ID = GMM_REFERENCE_TAIL.name
POLICY_REVISION = GMM_REFERENCE_TAIL.revision
POLICY_RANDOM_SEED = 20260629
PCA_EXPLAINED_VARIANCE = 0.99
PCA_MAX_DIMENSION = 32
GMM_COMPONENTS = 8
GMM_REG_COVAR = 1e-6
FRAME_ATOM_FRACTION = 0.05


@dataclass(frozen=True)
class GMMReferenceFitRequest:
    router_dir: Path
    descriptors: np.ndarray
    elements: Sequence[str]
    frame_indices: np.ndarray
    frame_ids: Sequence[str]
    type_map: Sequence[str]
    descriptor_schema: Mapping[str, object]
    model_digest: str
    reference_digest: str


@dataclass(frozen=True)
class GMMScoreRequest:
    router_dir: Path
    descriptors: np.ndarray
    elements: Sequence[str]
    frame_indices: np.ndarray
    frame_ids: Sequence[str]
    descriptor_schema: Mapping[str, object]
    model_digest: str
    reference_digest: str
    emit_frame_means: bool = False


@dataclass(frozen=True)
class GMMReferenceStreamFitRequest:
    router_dir: Path
    header: DescriptorStreamHeader
    chunks: Iterable[AtomDescriptorChunk]
    model_digest: str
    reference_digest: str
    spill: bool = False
    work_dir: Path | None = None


@dataclass(frozen=True)
class GMMStreamScoreRequest:
    router_dir: Path
    header: DescriptorStreamHeader
    chunks: Iterable[AtomDescriptorChunk]
    model_digest: str
    reference_digest: str
    emit_frame_means: bool = False


@dataclass(frozen=True)
class GMMReferenceFitResult:
    router_dir: Path
    policy_id: str
    element_pca_dimensions: Mapping[str, int]
    descriptor_schema: Mapping[str, object] | None = None
    policy_revision: int = POLICY_REVISION


@dataclass(frozen=True)
class GMMScoreResult:
    frame_ids: np.ndarray
    frame_p: np.ndarray
    frame_statistics: np.ndarray
    frame_means: np.ndarray | None = None
    descriptor_schema: Mapping[str, object] | None = None
    element_means: Mapping[str, tuple[tuple[str, ...], np.ndarray]] | None = None


def fit_gmm_router_stream(request: GMMReferenceStreamFitRequest) -> GMMReferenceFitResult:
    """Collect every reference atom in canonical order and fit the fixed router."""
    if not request.spill:
        collected = _collect_reference_stream_memory(request.header, request.chunks)
        return _fit_collected_reference_stream(request, *collected)
    if request.work_dir is None:
        raise ValueError("explicit GMM reference spill requires an operation-local work_dir")
    work_root = Path(request.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gmm-reference-spill-", dir=work_root) as temporary:
        collected = _collect_reference_stream_spill(
            request.header,
            request.chunks,
            workspace=Path(temporary),
        )
        return _fit_collected_reference_stream(request, *collected)


def _fit_collected_reference_stream(
    request: GMMReferenceStreamFitRequest,
    descriptors: np.ndarray,
    type_indices: np.ndarray,
    frame_indices: np.ndarray,
    frame_ids: tuple[str, ...],
) -> GMMReferenceFitResult:
    elements = np.asarray(request.header.type_map, dtype=np.str_)[type_indices]
    return fit_gmm_router(
        GMMReferenceFitRequest(
            router_dir=request.router_dir,
            descriptors=descriptors,
            elements=elements,
            frame_indices=frame_indices,
            frame_ids=frame_ids,
            type_map=request.header.type_map,
            descriptor_schema=request.header.schema,
            model_digest=request.model_digest,
            reference_digest=request.reference_digest,
        )
    )


def _collect_reference_stream_memory(
    header: DescriptorStreamHeader,
    chunks: Iterable[AtomDescriptorChunk],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    frame_rows: list[np.ndarray | None] = [None] * header.frame_count
    frame_types: list[np.ndarray | None] = [None] * header.frame_count
    frame_ids: list[str | None] = [None] * header.frame_count
    for chunk in validated_descriptor_chunks(header, chunks):
        for local_index, ordinal_value in enumerate(chunk.frame_ordinals):
            ordinal = int(ordinal_value)
            start, stop = int(chunk.frame_ptr[local_index]), int(chunk.frame_ptr[local_index + 1])
            frame_rows[ordinal] = chunk.descriptors[start:stop]
            frame_types[ordinal] = chunk.type_indices[start:stop]
            frame_ids[ordinal] = chunk.frame_ids[local_index]
    descriptors = np.concatenate(frame_rows, axis=0)
    type_indices = np.concatenate(frame_types, axis=0)
    frame_indices = np.concatenate(
        [np.full(len(rows), ordinal, dtype=np.int64) for ordinal, rows in enumerate(frame_rows)]
    )
    return descriptors, type_indices, frame_indices, tuple(str(item) for item in frame_ids)


def _collect_reference_stream_spill(
    header: DescriptorStreamHeader,
    chunks: Iterable[AtomDescriptorChunk],
    *,
    workspace: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    emission_descriptors = workspace / "emission-descriptors.bin"
    emission_types = workspace / "emission-types.bin"
    spans: list[tuple[int, int] | None] = [None] * header.frame_count
    frame_ids: list[str | None] = [None] * header.frame_count
    atom_count = 0
    with emission_descriptors.open("wb") as descriptor_handle, emission_types.open("wb") as type_handle:
        for chunk in validated_descriptor_chunks(header, chunks):
            chunk.descriptors.tofile(descriptor_handle)
            chunk.type_indices.astype(np.int32, copy=False).tofile(type_handle)
            for local_index, ordinal_value in enumerate(chunk.frame_ordinals):
                ordinal = int(ordinal_value)
                start = atom_count + int(chunk.frame_ptr[local_index])
                stop = atom_count + int(chunk.frame_ptr[local_index + 1])
                spans[ordinal] = (start, stop)
                frame_ids[ordinal] = chunk.frame_ids[local_index]
            atom_count += len(chunk.descriptors)
    if atom_count <= 0:
        raise ValueError("GMM reference descriptor stream must contain at least one atom")
    width = int(header.schema["width"])
    emitted_rows = np.memmap(emission_descriptors, mode="r", dtype=np.float32, shape=(atom_count, width))
    emitted_types = np.memmap(emission_types, mode="r", dtype=np.int32, shape=(atom_count,))
    canonical_rows = np.memmap(
        workspace / "canonical-descriptors.bin", mode="w+", dtype=np.float32, shape=(atom_count, width)
    )
    canonical_types = np.memmap(
        workspace / "canonical-types.bin", mode="w+", dtype=np.int32, shape=(atom_count,)
    )
    canonical_membership = np.memmap(
        workspace / "canonical-frame-membership.bin", mode="w+", dtype=np.int64, shape=(atom_count,)
    )
    target = 0
    for ordinal, raw_span in enumerate(spans):
        if raw_span is None:
            raise ValueError("descriptor stream is missing canonical frame membership")
        start, stop = raw_span
        count = stop - start
        canonical_rows[target : target + count] = emitted_rows[start:stop]
        canonical_types[target : target + count] = emitted_types[start:stop]
        canonical_membership[target : target + count] = ordinal
        target += count
    canonical_rows.flush()
    canonical_types.flush()
    canonical_membership.flush()
    del emitted_rows, emitted_types
    emission_descriptors.unlink()
    emission_types.unlink()
    return canonical_rows, canonical_types, canonical_membership, tuple(str(item) for item in frame_ids)


def score_gmm_router_stream(request: GMMStreamScoreRequest) -> GMMScoreResult:
    """Score and release complete-frame atomic chunks in one pass."""
    manifest, element_arrays, sorted_frame_statistics = load_gmm_router_bundle(Path(request.router_dir))
    descriptor_schema = validated_readout_input_schema(request.header.schema)
    _validate_router_manifest(
        manifest,
        descriptor_schema=descriptor_schema,
        model_digest=str(request.model_digest),
        reference_digest=str(request.reference_digest),
    )
    router_elements = tuple(str(item) for item in manifest.get("type_map", []))
    frame_ids: list[str | None] = [None] * request.header.frame_count
    frame_statistics = np.empty(request.header.frame_count, dtype=np.float64)
    frame_means = (
        np.empty((request.header.frame_count, int(request.header.schema["width"])), dtype=np.float32)
        if request.emit_frame_means
        else None
    )
    element_rows = {element: [] for element in request.header.type_map} if request.emit_frame_means else None
    for chunk in validated_descriptor_chunks(request.header, request.chunks):
        chunk_elements = np.asarray(request.header.type_map, dtype=np.str_)[chunk.type_indices]
        unknown = sorted(set(str(item) for item in chunk_elements) - set(router_elements))
        if unknown:
            raise ValueError(f"candidate elements are absent from GMM router type_map: {unknown}")
        for local_index, ordinal_value in enumerate(chunk.frame_ordinals):
            ordinal = int(ordinal_value)
            start, stop = int(chunk.frame_ptr[local_index]), int(chunk.frame_ptr[local_index + 1])
            frame_descriptors = chunk.descriptors[start:stop]
            frame_elements = chunk_elements[start:stop]
            atom_p = np.empty(stop - start, dtype=np.float64)
            # Keep every linear-algebra call frame-local.  BLAS kernels may
            # associate operations differently when otherwise identical rows
            # share a larger replay chunk, which can move an ill-conditioned
            # NLL across one empirical calibration boundary.  A frame is the
            # policy's release unit, so its math must not depend on how files
            # were grouped into a chunk manifest.
            for element in router_elements:
                atom_indices = np.flatnonzero(frame_elements == element)
                if atom_indices.size == 0:
                    continue
                arrays = element_arrays[element]
                raw = frame_descriptors[atom_indices].astype(np.float64, copy=False)
                standardized = (raw - arrays["scaler_mean"]) / arrays["scaler_scale"]
                latent = (
                    standardized - arrays["pca_mean"]
                ) @ arrays["pca_components"].T
                nll = _full_gmm_nll(
                    latent,
                    weights=arrays["gmm_weights"],
                    means=arrays["gmm_means"],
                    precisions_cholesky=arrays["gmm_precisions_cholesky"],
                )
                atom_p[atom_indices] = empirical_upper_tail_p(
                    arrays["sorted_reference_nll"], nll
                )
            frame_ids[ordinal] = chunk.frame_ids[local_index]
            frame_statistics[ordinal] = _one_frame_statistic(atom_p)
            if frame_means is not None:
                frame_means[ordinal] = np.mean(
                    frame_descriptors, axis=0, dtype=np.float32
                )
                for type_index in dict.fromkeys(int(item) for item in chunk.type_indices[start:stop]):
                    element_rows[request.header.type_map[type_index]].append((
                        ordinal, chunk.frame_ids[local_index],
                        np.mean(frame_descriptors[chunk.type_indices[start:stop] == type_index], axis=0, dtype=np.float32),
                    ))
    frame_p = empirical_upper_tail_p(sorted_frame_statistics, frame_statistics)
    return GMMScoreResult(
        frame_ids=np.asarray(frame_ids, dtype=np.str_),
        frame_p=frame_p,
        frame_statistics=frame_statistics,
        frame_means=frame_means,
        descriptor_schema=descriptor_schema,
        element_means=None if element_rows is None else {
            element: (tuple(item[1] for item in sorted(rows, key=lambda item: item[0])),
                      np.stack([item[2] for item in sorted(rows, key=lambda item: item[0])]))
            for element, rows in element_rows.items() if rows
        },
    )


def fit_gmm_router(request: GMMReferenceFitRequest) -> GMMReferenceFitResult:
    """Fit and publish the fixed per-element GMM reference-tail policy."""
    descriptors, elements, frame_indices, frame_ids, schema = _validate_descriptor_input(
        descriptors=request.descriptors,
        elements=request.elements,
        frame_indices=request.frame_indices,
        frame_ids=request.frame_ids,
        descriptor_schema=request.descriptor_schema,
        allow_empty=False,
    )
    declared_type_map = tuple(str(item) for item in request.type_map)
    if (
        not declared_type_map
        or any(not item for item in declared_type_map)
        or len(set(declared_type_map)) != len(declared_type_map)
    ):
        raise ValueError("GMM router type_map must contain non-empty unique elements")
    actual_elements = set(str(item) for item in np.unique(elements))
    undeclared = tuple(sorted(actual_elements - set(declared_type_map)))
    if undeclared:
        raise ValueError(
            f"GMM reference elements are absent from the declared type_map: {undeclared}"
        )
    type_map = tuple(
        element for element in declared_type_map if element in actual_elements
    )

    try:
        import sklearn
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - deployment preflight owns this path
        raise RuntimeError("gmm-reference-tail fitting requires scikit-learn") from exc

    element_arrays: dict[str, dict[str, np.ndarray]] = {}
    element_manifest: dict[str, dict[str, object]] = {}
    atom_p = np.empty(descriptors.shape[0], dtype=np.float64)

    for element_index, element in enumerate(type_map):
        atom_indices = np.flatnonzero(elements == element)
        if atom_indices.size < GMM_COMPONENTS:
            raise ValueError(
                f"GMM router element {element!r} requires at least {GMM_COMPONENTS} reference atoms; "
                f"got {atom_indices.size}"
            )
        raw = descriptors[atom_indices].astype(np.float64, copy=False)
        scaler = StandardScaler(copy=True).fit(raw)
        standardized = scaler.transform(raw)
        _require_finite(standardized, f"standardized descriptors for element {element}")

        computed_components = min(PCA_MAX_DIMENSION, standardized.shape[0], standardized.shape[1])
        pca = PCA(
            n_components=computed_components,
            svd_solver="randomized",
            random_state=POLICY_RANDOM_SEED,
        ).fit(standardized)
        cumulative = np.cumsum(np.asarray(pca.explained_variance_ratio_, dtype=np.float64))
        reached = np.flatnonzero(cumulative >= PCA_EXPLAINED_VARIANCE)
        retained_components = int(reached[0] + 1) if reached.size else int(computed_components)
        components = np.asarray(pca.components_[:retained_components], dtype=np.float64)
        latent = (standardized - np.asarray(pca.mean_, dtype=np.float64)) @ components.T
        _require_finite(latent, f"PCA transform for element {element}")

        mixture = GaussianMixture(
            n_components=GMM_COMPONENTS,
            covariance_type="full",
            reg_covar=GMM_REG_COVAR,
            random_state=POLICY_RANDOM_SEED,
            max_iter=200,
        ).fit(latent)
        if not bool(mixture.converged_):
            raise RuntimeError(
                f"GMM router element {element!r} did not converge after {mixture.n_iter_} iterations"
            )
        nll = -np.asarray(mixture.score_samples(latent), dtype=np.float64)
        _require_finite(nll, f"reference NLL for element {element}")
        sorted_nll = np.sort(nll)
        atom_p[atom_indices] = empirical_upper_tail_p(sorted_nll, nll)

        filename = f"element_{element_index:03d}.npz"
        element_arrays[element] = {
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
            "pca_mean": np.asarray(pca.mean_, dtype=np.float64),
            "pca_components": components,
            "gmm_weights": np.asarray(mixture.weights_, dtype=np.float64),
            "gmm_means": np.asarray(mixture.means_, dtype=np.float64),
            "gmm_covariances": np.asarray(mixture.covariances_, dtype=np.float64),
            "gmm_precisions_cholesky": np.asarray(mixture.precisions_cholesky_, dtype=np.float64),
            "sorted_reference_nll": sorted_nll,
        }
        element_manifest[element] = {
            "file": filename,
            "reference_atoms": int(atom_indices.size),
            "feature_width": int(descriptors.shape[1]),
            "computed_pca_dimensions": int(computed_components),
            "pca_dimensions": retained_components,
            "pca_explained_variance": float(cumulative[retained_components - 1]),
            "gmm_converged": True,
            "gmm_iterations": int(mixture.n_iter_),
        }

    frame_statistics = fixed_top_fraction_frame_statistic(atom_p, frame_indices, len(frame_ids))
    sorted_frame_statistics = np.sort(frame_statistics)
    manifest = {
        **GMM_ROUTER_IDENTITY.as_dict(
            name_field="schema", revision_field="schema_revision"
        ),
        "policy_name": POLICY_ID,
        "policy_revision": POLICY_REVISION,
        "policy_constants": _policy_manifest(),
        "descriptor_schema": schema,
        "type_map": list(type_map),
        "reference_frames": len(frame_ids),
        "model_digest": str(request.model_digest),
        "reference_digest": str(request.reference_digest),
        "software": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "elements": element_manifest,
    }
    publish_gmm_router_bundle(
        Path(request.router_dir),
        manifest=manifest,
        element_arrays=element_arrays,
        frame_statistics=sorted_frame_statistics,
    )
    return GMMReferenceFitResult(
        router_dir=Path(request.router_dir),
        policy_id=POLICY_ID,
        element_pca_dimensions={element: int(metadata["pca_dimensions"]) for element, metadata in element_manifest.items()},
        descriptor_schema=schema,
    )


def score_gmm_router(request: GMMScoreRequest) -> GMMScoreResult:
    """Score candidate frames using only persisted numeric router state."""
    descriptors, elements, frame_indices, frame_ids, schema = _validate_descriptor_input(
        descriptors=request.descriptors,
        elements=request.elements,
        frame_indices=request.frame_indices,
        frame_ids=request.frame_ids,
        descriptor_schema=request.descriptor_schema,
        allow_empty=True,
    )
    manifest, element_arrays, sorted_frame_statistics = load_gmm_router_bundle(Path(request.router_dir))
    _validate_router_manifest(
        manifest,
        descriptor_schema=schema,
        model_digest=str(request.model_digest),
        reference_digest=str(request.reference_digest),
    )

    atom_p = np.empty(descriptors.shape[0], dtype=np.float64)
    router_elements = tuple(str(item) for item in manifest.get("type_map", []))
    unknown = sorted(set(str(item) for item in elements) - set(router_elements))
    if unknown:
        raise ValueError(f"candidate elements are absent from GMM router type_map: {unknown}")
    for element in router_elements:
        atom_indices = np.flatnonzero(elements == element)
        if atom_indices.size == 0:
            continue
        arrays = element_arrays[element]
        raw = descriptors[atom_indices].astype(np.float64, copy=False)
        standardized = (raw - arrays["scaler_mean"]) / arrays["scaler_scale"]
        latent = (standardized - arrays["pca_mean"]) @ arrays["pca_components"].T
        nll = _full_gmm_nll(
            latent,
            weights=arrays["gmm_weights"],
            means=arrays["gmm_means"],
            precisions_cholesky=arrays["gmm_precisions_cholesky"],
        )
        atom_p[atom_indices] = empirical_upper_tail_p(arrays["sorted_reference_nll"], nll)

    frame_statistics = fixed_top_fraction_frame_statistic(atom_p, frame_indices, len(frame_ids))
    frame_p = empirical_upper_tail_p(sorted_frame_statistics, frame_statistics)
    frame_means = None
    element_means = None
    if request.emit_frame_means:
        frame_means = np.empty((len(frame_ids), descriptors.shape[1]), dtype=np.float64)
        order = np.argsort(frame_indices, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(np.bincount(frame_indices, minlength=len(frame_ids)))))
        collected = {}
        for ordinal, identity in enumerate(frame_ids):
            indices = order[offsets[ordinal]:offsets[ordinal + 1]]
            rows, symbols = descriptors[indices], elements[indices]
            frame_means[ordinal] = rows.astype(np.float64).mean(axis=0)
            for element in np.unique(symbols):
                collected.setdefault(str(element), []).append(
                    (identity, rows[symbols == element].mean(axis=0, dtype=np.float32))
                )
        element_means = {
            element: (tuple(identity for identity, row in values), np.stack([row for identity, row in values]))
            for element, values in sorted(collected.items())
        }
    return GMMScoreResult(
        frame_ids=np.asarray(frame_ids, dtype=np.str_),
        frame_p=frame_p,
        frame_statistics=frame_statistics,
        frame_means=frame_means,
        descriptor_schema=schema,
        element_means=element_means,
    )


def validate_gmm_router(
    router_dir: str | Path,
    *,
    descriptor_schema: Mapping[str, object] | None = None,
    model_digest: str | None = None,
    reference_digest: str | None = None,
) -> dict[str, object]:
    """Validate a durable router and optional expected provenance."""
    manifest, _, _ = load_gmm_router_bundle(Path(router_dir))
    _validate_router_manifest(
        manifest,
        descriptor_schema=None if descriptor_schema is None else dict(descriptor_schema),
        model_digest=model_digest,
        reference_digest=reference_digest,
    )
    return manifest


def _validate_router_manifest(
    manifest: Mapping[str, object],
    *,
    descriptor_schema: Mapping[str, object] | None,
    model_digest: str | None,
    reference_digest: str | None,
) -> None:
    if (
        not GMM_ROUTER_IDENTITY.matches(
            manifest, name_field="schema", revision_field="schema_revision"
        )
        or manifest.get("policy_name") != POLICY_ID
        or manifest.get("policy_revision") != POLICY_REVISION
    ):
        raise ValueError("unsupported or malformed GMM router manifest")
    if manifest.get("policy_constants") != _policy_manifest():
        raise ValueError("GMM router policy constants do not match gmm-reference-tail")
    if descriptor_schema is not None and dict(manifest.get("descriptor_schema", {})) != dict(descriptor_schema):
        raise ValueError("candidate descriptor schema does not match GMM router")
    if model_digest is not None and manifest.get("model_digest") != str(model_digest):
        raise ValueError("candidate model digest does not match GMM router")
    if reference_digest is not None and manifest.get("reference_digest") != str(reference_digest):
        raise ValueError("candidate reference digest does not match GMM router")


def _validate_descriptor_input(
    *,
    descriptors: np.ndarray,
    elements: Sequence[str],
    frame_indices: np.ndarray,
    frame_ids: Sequence[str],
    descriptor_schema: Mapping[str, object],
    allow_empty: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, object]]:
    values = np.asarray(descriptors)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] == 0
        or (values.shape[0] == 0 and not allow_empty)
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(f"GMM descriptors must be a {qualifier} atom x feature float32 array")
    _require_finite(values, "GMM descriptors")
    atom_elements = np.asarray(tuple(str(item) for item in elements), dtype=np.str_)
    membership = np.asarray(frame_indices)
    identities = tuple(str(item) for item in frame_ids)
    if atom_elements.shape != (values.shape[0],) or membership.shape != (values.shape[0],):
        raise ValueError("GMM descriptor element and frame membership must match atom rows")
    if not np.issubdtype(membership.dtype, np.integer):
        raise ValueError("GMM frame membership must use integer indices")
    membership = membership.astype(np.int64, copy=False)
    if (not identities and not allow_empty) or len(set(identities)) != len(identities):
        raise ValueError("GMM frame identities must be unique and non-empty for reference fitting")
    if values.shape[0] == 0 and identities:
        raise ValueError("an empty GMM candidate descriptor sequence cannot declare frame identities")
    if np.any(membership < 0) or np.any(membership >= len(identities)):
        raise ValueError("GMM frame membership index is outside the frame identity sequence")
    counts = np.bincount(membership, minlength=len(identities))
    if np.any(counts == 0):
        raise ValueError("every GMM frame identity must own at least one atom descriptor")
    schema = validated_readout_input_schema(descriptor_schema)
    if schema.get("width") != values.shape[1] or schema.get("dtype") != "float32":
        raise ValueError("GMM descriptor schema width/dtype does not match descriptor rows")
    return values, atom_elements, membership, identities, schema


def _policy_manifest() -> dict[str, object]:
    return {
        "random_seed": POLICY_RANDOM_SEED,
        "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
        "pca_max_dimensions": PCA_MAX_DIMENSION,
        "gmm_components": GMM_COMPONENTS,
        "gmm_covariance_type": "full",
        "gmm_reg_covar": GMM_REG_COVAR,
        "frame_atom_fraction": FRAME_ATOM_FRACTION,
        "empirical_tail": "inclusive-smoothed-upper-tail",
    }


def empirical_upper_tail_p(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the inclusive, finite-sample-smoothed reference upper-tail p."""
    reference = np.asarray(sorted_reference, dtype=np.float64)
    query = np.asarray(values, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or np.any(reference[1:] < reference[:-1]):
        raise ValueError("GMM empirical calibration must be a non-empty sorted vector")
    _require_finite(reference, "GMM empirical calibration")
    _require_finite(query, "GMM empirical query")
    count_ge = reference.size - np.searchsorted(reference, query, side="left")
    return (count_ge.astype(np.float64) + 1.0) / (reference.size + 1.0)


def fixed_top_fraction_frame_statistic(
    atom_p: np.ndarray,
    frame_indices: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    """Average the largest ceil(5% * atoms) atom surprises per frame."""
    probabilities = np.asarray(atom_p, dtype=np.float64)
    if probabilities.ndim != 1 or np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0.0) or np.any(probabilities > 1.0):
        raise ValueError("GMM atom p values must be finite values in (0, 1]")
    surprises = -np.log(probabilities)
    result = np.empty(frame_count, dtype=np.float64)
    for frame_index in range(frame_count):
        frame_surprises = surprises[frame_indices == frame_index]
        keep = max(1, math.ceil(FRAME_ATOM_FRACTION * frame_surprises.size))
        result[frame_index] = float(np.mean(np.partition(frame_surprises, frame_surprises.size - keep)[-keep:]))
    _require_finite(result, "GMM frame statistics")
    return result


def _one_frame_statistic(atom_p: np.ndarray) -> float:
    probabilities = np.asarray(atom_p, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("GMM atom p values must be finite values in (0, 1]")
    surprises = -np.log(probabilities)
    keep = max(1, math.ceil(FRAME_ATOM_FRACTION * surprises.size))
    return float(np.mean(np.partition(surprises, surprises.size - keep)[-keep:]))


def _full_gmm_nll(
    values: np.ndarray,
    *,
    weights: np.ndarray,
    means: np.ndarray,
    precisions_cholesky: np.ndarray,
) -> np.ndarray:
    from scipy.special import logsumexp

    x = np.asarray(values, dtype=np.float64)
    mixture_weights = np.asarray(weights, dtype=np.float64)
    mixture_means = np.asarray(means, dtype=np.float64)
    mixture_precisions = np.asarray(precisions_cholesky, dtype=np.float64)
    component_log_prob = np.empty((x.shape[0], mixture_weights.size), dtype=np.float64)
    dimension = x.shape[1]
    log_determinants = np.sum(
        np.log(mixture_precisions.reshape(mixture_weights.size, -1)[:, :: dimension + 1]),
        axis=1,
    )
    for component in range(mixture_weights.size):
        precision_cholesky = mixture_precisions[component]
        transformed = np.dot(x, precision_cholesky) - np.dot(mixture_means[component], precision_cholesky)
        quadratic = np.sum(np.square(transformed), axis=1)
        component_log_prob[:, component] = (
            -0.5 * (dimension * np.log(2.0 * np.pi).astype(x.dtype) + quadratic)
            + log_determinants[component]
            + np.log(mixture_weights[component])
        )
    nll = -logsumexp(component_log_prob, axis=1)
    _require_finite(nll, "persisted GMM NLL")
    return nll


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(np.asarray(values))):
        raise ValueError(f"{label} contains non-finite values")


__all__ = [
    "GMMReferenceFitRequest",
    "GMMReferenceFitResult",
    "GMMReferenceStreamFitRequest",
    "GMMScoreRequest",
    "GMMScoreResult",
    "GMMStreamScoreRequest",
    "POLICY_ID",
    "empirical_upper_tail_p",
    "fit_gmm_router",
    "fit_gmm_router_stream",
    "fixed_top_fraction_frame_statistic",
    "score_gmm_router",
    "score_gmm_router_stream",
    "validate_gmm_router",
]
