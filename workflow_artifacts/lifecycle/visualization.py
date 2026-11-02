from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Callable, Mapping

import numpy as np

from workflow_core.lifecycle.visualization import (
    ElementFrameRows,
    VisualizationLayer,
    VisualizationRequest,
    build_visualization_request,
)


_CACHE_SCHEMA = "dice.diagnostics.selection-visualization-cache"
_CACHE_REVISION = 2
_MANIFEST_SCHEMA = "dice.diagnostics.selection-visualization"
_MANIFEST_REVISION = 2
_PROGRESS_SCHEMA = "dice.diagnostics.selection-visualization-progress"
_PROGRESS_REVISION = 1
_LANDMARK_MAX_ROWS = 20_000
_TRANSFORM_BATCH_ROWS = 20_000
_TRANSFORM_TAIL_MERGE_MAX_ROWS = 10_000
_LANDMARK_SELECTION = "layer-priority-sha256-v1"
_LAYER_COLORS = {
    "screen": {
        "sample": "#B9D7EA",
        "pool": "#0072B2",
        "train": "#009E73",
        "select": "#D55E00",
    },
    "fp": {
        "pool": "#0072B2",
        "train": "#009E73",
        "select": "#D55E00",
    },
}


class VisualizationDependencyError(RuntimeError):
    pass


class _ProgressRecorder:
    def __init__(self, path: Path, *, source_digest: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._source_digest = str(source_digest)
        self._started = time.perf_counter()

    def record(self, event: str, **fields: object) -> None:
        payload = {
            "schema": _PROGRESS_SCHEMA,
            "schema_revision": _PROGRESS_REVISION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": _elapsed(self._started),
            "source_digest": self._source_digest,
            "event": str(event),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _elapsed(started: float) -> float:
    return float(round(max(0.0, time.perf_counter() - started), 6))


def _effective_numba_threads() -> int:
    import numba

    return int(numba.get_num_threads())


def publish_visualization_request(
    root: str | Path, request: VisualizationRequest
) -> Path:
    target = Path(root)
    stage = _temporary_sibling(target)
    try:
        metadata = _request_metadata(request)
        arrays: dict[str, np.ndarray] = {}
        for index, element in enumerate(request.type_map):
            rows = request.elements.get(element)
            if rows is None:
                continue
            arrays[f"element_{index}_ids"] = np.asarray(rows.frame_ids, dtype=np.str_)
            arrays[f"element_{index}_vectors"] = rows.vectors
        if request.structure is not None:
            arrays["structure_ids"] = np.asarray(request.structure.frame_ids, dtype=np.str_)
            arrays["structure_vectors"] = request.structure.vectors
        with (stage / "descriptors.npz").open("wb") as handle:
            np.savez(handle, **arrays)
        (stage / "request.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded = _load_visualization_request(stage)
        if loaded.source_digest != request.source_digest:
            raise ValueError("visualization cache source digest changed during publication")
        _replace_directory(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def load_visualization_request(root: str | Path) -> VisualizationRequest:
    return _load_visualization_request(Path(root))


def _load_visualization_request(root: Path) -> VisualizationRequest:
    try:
        metadata = json.loads((root / "request.json").read_text(encoding="utf-8"))
        with np.load(root / "descriptors.npz", allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("visualization cache is malformed") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != _CACHE_SCHEMA
        or metadata.get("schema_revision") not in (1, _CACHE_REVISION)
    ):
        raise ValueError("visualization cache schema is unsupported")
    type_map = tuple(str(item) for item in metadata.get("type_map", ()))
    element_names = tuple(str(item) for item in metadata.get("elements", ()))
    elements: dict[str, ElementFrameRows] = {}
    expected_arrays: set[str] = set()
    for element in element_names:
        if element not in type_map:
            raise ValueError("visualization cache element is outside type_map")
        index = type_map.index(element)
        ids_name = f"element_{index}_ids"
        vectors_name = f"element_{index}_vectors"
        expected_arrays.update((ids_name, vectors_name))
        elements[element] = ElementFrameRows(
            tuple(str(item) for item in arrays[ids_name].tolist()),
            arrays[vectors_name],
        )
    structure_populations = None
    if metadata["schema_revision"] == _CACHE_REVISION:
        if metadata.get("structure_aggregation") != "all-atom-mean":
            raise ValueError("visualization cache structure aggregation is unsupported")
        expected_arrays.update(("structure_ids", "structure_vectors"))
        structure_populations = (ElementFrameRows(
            tuple(str(item) for item in arrays["structure_ids"].tolist()),
            arrays["structure_vectors"],
        ),)
    if set(arrays) != expected_arrays:
        raise ValueError("visualization cache arrays are incompatible")
    request = build_visualization_request(
        kind=str(metadata["kind"]),
        layers=tuple(
            VisualizationLayer(
                str(layer["name"]),
                tuple(str(item) for item in layer["frame_ids"]),
            )
            for layer in metadata["layers"]
        ),
        populations=(elements,),
        structure_populations=structure_populations,
        descriptor_schema=dict(metadata["descriptor_schema"]),
        type_map=type_map,
        model_identity=str(metadata["model_identity"]),
        dataset_identity=str(metadata["dataset_identity"]),
    )
    if request.source_digest != metadata.get("source_digest"):
        raise ValueError("visualization cache source digest is stale")
    return request


def render_visualization(
    request: VisualizationRequest, output_dir: str | Path, *, view_map=None
) -> Path:
    target = Path(output_dir)
    stage = _temporary_sibling(target)
    progress = _ProgressRecorder(
        target.with_name(f"{target.name}.progress.jsonl"),
        source_digest=request.source_digest,
    )
    progress.record(
        "render_started",
        elements=len(request.elements),
        rows=sum(len(rows.frame_ids) for rows in request.elements.values()),
    )
    try:
        request_metadata = _request_metadata(request)
        request_metadata.pop("schema")
        request_metadata.pop("schema_revision")
        manifest: dict[str, object] = {
            "schema": _MANIFEST_SCHEMA,
            "schema_revision": _MANIFEST_REVISION,
            **request_metadata,
            "parameters": {
                "row_normalization": "l2",
                "pca_max_components": 50,
                "landmark_max_rows": _LANDMARK_MAX_ROWS,
                "landmark_selection": _LANDMARK_SELECTION,
                "transform_batch_rows": _TRANSFORM_BATCH_ROWS,
                "transform_tail_merge_max_rows": _TRANSFORM_TAIL_MERGE_MAX_ROWS,
                "random_state": 42,
                "metric": "euclidean",
                "n_neighbors_max": 30,
                "min_dist": 0.1,
                "n_jobs": 1,
                "effective_numba_threads": _effective_numba_threads(),
            },
            "style": {"layer_colors": dict(_LAYER_COLORS[request.kind])},
            "progress_log": progress.path.name,
            "elements": {},
            "files": {},
            "warnings": [],
        }
        element_manifest: dict[str, object] = {}
        manifest["structure"] = {"status": "unavailable", "reason": "legacy-element-only-cache"}
        views = [(element, request.elements[element]) for element in request.type_map
                 if element in request.elements]
        if request.structure is not None:
            views.append(("structure", request.structure))
        mapper = view_map if view_map is not None and sum(len(rows.frame_ids) >= 3 for _, rows in views) > 1 else _serial_view_map
        tasks = ((element, rows, request.layers, request.kind, stage) for element, rows in views)
        for element, entry in mapper(
            _render_view_task, tasks,
            on_progress=lambda event, fields: progress.record(event, **fields),
        ):
            if element == "structure":
                manifest["structure"] = entry
            else:
                element_manifest[element] = entry
        manifest["elements"] = element_manifest
        files = {
            path.name: _file_digest(path)
            for path in sorted(stage.iterdir())
            if path.is_file()
        }
        manifest["files"] = files
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_directory(stage, target)
        progress.record("render_completed", output=str(target))
    except BaseException as exc:
        progress.record(
            "render_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def _serial_view_map(function, items, *, on_progress):
    for item in items:
        yield function(item, progress=on_progress)


def _render_view_task(payload, *, progress):
    """Render one independent view; the parent owns publication and progress."""
    element, rows, layers, kind, stage = payload
    stage = Path(stage)
    is_structure = element == "structure"
    prefix = "structure" if is_structure else "element"
    view = {"view": "structure"} if is_structure else {"element": element}
    if len(rows.frame_ids) < 3:
        progress(f"{prefix}_skipped", {**view, "rows": len(rows.frame_ids),
                                    "reason": "fewer-than-three-usable-rows"})
        return element, {"status": "skipped", "reason": "fewer-than-three-usable-rows",
                         "rows": len(rows.frame_ids)}
    started = time.perf_counter()
    progress(f"{prefix}_started", {**view, "rows": len(rows.frame_ids)})
    def reduction_progress(event, fields):
        progress(event, {**view, "rows": len(rows.frame_ids), **fields})
    coordinates, landmark_mask, transform_batches, reduction_timings = _reduce(
        rows.vectors, identities=rows.frame_ids, layers=layers, progress=reduction_progress,
    )
    safe = _safe_element_name(element)
    coordinate_path = stage / f"{safe}.npz"
    memberships = tuple(set(layer.frame_ids) for layer in layers)
    masks = {f"layer_{index}": np.asarray([identity in membership for identity in rows.frame_ids], dtype=np.bool_)
             for index, membership in enumerate(memberships)}
    with coordinate_path.open("wb") as handle:
        np.savez(handle, structure_ids=np.asarray(rows.frame_ids, dtype=np.str_),
                            source_vectors=np.asarray(rows.vectors, dtype=np.float32),
                            coordinates=np.asarray(coordinates, dtype=np.float32),
                            landmark_mask=landmark_mask, **masks)
    png, svg = stage / f"{safe}.png", stage / f"{safe}.svg"
    _plot_element(coordinates, rows.frame_ids, layers, kind=kind, element=element, png=png, svg=svg)
    timings = {**reduction_timings, f"{prefix}_total": _elapsed(started)}
    progress(f"{prefix}_completed", {**view, "rows": len(rows.frame_ids), "timing_seconds": timings})
    return element, {"status": "rendered", "rows": len(rows.frame_ids),
                     "landmark_rows": int(landmark_mask.sum()),
                     "transformed_rows": int((~landmark_mask).sum()),
                     "transform_batches": transform_batches, "timing_seconds": timings,
                     "files": [png.name, svg.name, coordinate_path.name]}


def visualization_output_is_current(request: VisualizationRequest, output_dir: str | Path) -> bool:
    """Check diagnostic capability and files without promoting them to block authority."""
    output = Path(output_dir)
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("source_digest") != request.source_digest:
            return False
        # Revision-1 caches cannot reconstruct all-atom means. Preserve historical
        # output without scheduling an upgrade or model replay merely for a new view.
        if request.structure is None:
            return True
        if (manifest.get("schema") != _MANIFEST_SCHEMA
                or manifest.get("schema_revision") != _MANIFEST_REVISION
                or manifest.get("structure_aggregation") != "all-atom-mean"):
            return False
        elements = manifest.get("elements")
        inventory = manifest.get("files")
        if not isinstance(elements, dict) or not isinstance(inventory, dict):
            return False
        views = [(name, rows, elements.get(name)) for name, rows in request.elements.items()]
        views.append(("structure", request.structure, manifest.get("structure")))
        for name, rows, entry in views:
            if not isinstance(entry, dict) or entry.get("rows") != len(rows.frame_ids):
                return False
            if len(rows.frame_ids) < 3:
                if (entry.get("status") != "skipped"
                        or entry.get("reason") != "fewer-than-three-usable-rows"):
                    return False
                continue
            expected = [f"{_safe_element_name(name)}.{suffix}" for suffix in ("png", "svg", "npz")]
            if entry.get("status") != "rendered" or entry.get("files") != expected:
                return False
            if any(filename not in inventory or not (output / filename).is_file()
                   or (output / filename).stat().st_size == 0 for filename in expected):
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def render_visualization_cache(cache_dir: str | Path, output_dir: str | Path, *, view_map=None) -> Path:
    return render_visualization(load_visualization_request(cache_dir), output_dir, view_map=view_map)


def _reduce(
    vectors: np.ndarray,
    *,
    identities: tuple[str, ...],
    layers: tuple[VisualizationLayer, ...],
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    try:
        from sklearn.decomposition import PCA
        import umap
    except (ImportError, OSError) as exc:
        raise VisualizationDependencyError(
            "selection visualization requires scikit-learn and umap-learn"
        ) from exc
    values = np.asarray(vectors)
    if values.ndim != 2 or len(values) != len(identities):
        raise ValueError("visualization vectors changed canonical row coverage")
    landmark_indices = _landmark_indices(identities, layers)
    landmark_mask = np.zeros(len(values), dtype=np.bool_)
    landmark_mask[landmark_indices] = True
    transformed_rows = int((~landmark_mask).sum())
    _emit_reduction_progress(
        progress,
        "landmarks_selected",
        landmark_rows=len(landmark_indices),
        transformed_rows=transformed_rows,
        transform_batch_rows=_TRANSFORM_BATCH_ROWS,
    )
    normalized_landmarks = _normalized_rows(values[landmark_indices])
    components = min(
        50,
        normalized_landmarks.shape[1],
        normalized_landmarks.shape[0] - 1,
    )
    pca = PCA(n_components=components, random_state=42)
    reduced_landmarks = pca.fit_transform(normalized_landmarks)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(30, len(reduced_landmarks) - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        transform_seed=42,
        n_jobs=1,
    )
    _emit_reduction_progress(
        progress, "umap_fit_started", landmark_rows=len(landmark_indices)
    )
    fit_started = time.perf_counter()
    coordinates = np.empty((len(values), 2), dtype=np.float32)
    coordinates[landmark_indices] = np.asarray(
        reducer.fit_transform(reduced_landmarks), dtype=np.float32
    )
    fit_seconds = _elapsed(fit_started)
    _emit_reduction_progress(
        progress,
        "umap_fit_completed",
        landmark_rows=len(landmark_indices),
        seconds=fit_seconds,
    )
    remaining = np.flatnonzero(~landmark_mask)
    batches = _transform_index_batches(remaining)
    transform_batches = 0
    transform_started = time.perf_counter()
    batch_count = len(batches)
    _emit_reduction_progress(
        progress,
        "transform_started",
        transformed_rows=len(remaining),
        batches=batch_count,
    )
    completed_rows = 0
    for batch in batches:
        batch_started = time.perf_counter()
        _emit_reduction_progress(
            progress,
            "transform_batch_started",
            batch=transform_batches + 1,
            batches=batch_count,
            batch_rows=len(batch),
            completed_rows=completed_rows,
            transformed_rows=len(remaining),
        )
        reduced = pca.transform(_normalized_rows(values[batch]))
        coordinates[batch] = np.asarray(reducer.transform(reduced), dtype=np.float32)
        transform_batches += 1
        completed_rows += len(batch)
        _emit_reduction_progress(
            progress,
            "transform_batch_completed",
            batch=transform_batches,
            batches=batch_count,
            batch_rows=len(batch),
            completed_rows=completed_rows,
            transformed_rows=len(remaining),
            seconds=_elapsed(batch_started),
        )
    transform_seconds = _elapsed(transform_started)
    _emit_reduction_progress(
        progress,
        "transform_completed",
        transformed_rows=len(remaining),
        batches=transform_batches,
        seconds=transform_seconds,
    )
    return (
        coordinates,
        landmark_mask,
        transform_batches,
        {"umap_fit": fit_seconds, "transform": transform_seconds},
    )


def _transform_index_batches(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    if not len(indices):
        return ()
    starts = list(range(0, len(indices), _TRANSFORM_BATCH_ROWS))
    if (
        len(starts) > 1
        and len(indices) - starts[-1] <= _TRANSFORM_TAIL_MERGE_MAX_ROWS
    ):
        starts.pop()
    ends = (*starts[1:], len(indices))
    return tuple(indices[start:end] for start, end in zip(starts, ends, strict=True))


def _emit_reduction_progress(
    callback: Callable[[str, Mapping[str, object]], None] | None,
    event: str,
    **fields: object,
) -> None:
    if callback is not None:
        callback(event, fields)


def _landmark_indices(
    identities: tuple[str, ...], layers: tuple[VisualizationLayer, ...]
) -> np.ndarray:
    if len(identities) <= _LANDMARK_MAX_ROWS:
        return np.arange(len(identities), dtype=np.int64)
    index_by_id = {identity: index for index, identity in enumerate(identities)}
    selected: list[int] = []
    selected_ids: set[str] = set()

    def extend(candidates) -> None:
        available = _LANDMARK_MAX_ROWS - len(selected)
        if available <= 0:
            return
        ordered = sorted(
            {
                str(identity)
                for identity in candidates
                if identity in index_by_id and identity not in selected_ids
            },
            key=_landmark_key,
        )
        for identity in ordered[:available]:
            selected_ids.add(identity)
            selected.append(index_by_id[identity])

    for layer in reversed(layers[1:]):
        extend(layer.frame_ids)
    extend(identities)
    if len(selected) != _LANDMARK_MAX_ROWS:
        raise ValueError("visualization landmark selection changed row coverage")
    return np.asarray(selected, dtype=np.int64)


def _landmark_key(identity: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(
        f"{_LANDMARK_SELECTION}\0{identity}".encode("utf-8")
    ).digest()
    return digest, identity


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    return rows / np.where(norms > 0, norms, 1.0)[:, None]


def _plot_element(
    coordinates: np.ndarray,
    identities: tuple[str, ...],
    layers: tuple[VisualizationLayer, ...],
    *,
    kind: str,
    element: str,
    png: Path,
    svg: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise VisualizationDependencyError(
            "selection visualization requires matplotlib"
        ) from exc
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for index, layer in enumerate(layers):
        membership = set(layer.frame_ids)
        mask = np.asarray([identity in membership for identity in identities])
        if not mask.any():
            continue
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18 + index * 7,
            alpha=0.72 + index * 0.07,
            c=_LAYER_COLORS[kind][layer.name],
            label=f"{layer.name} (n={int(mask.sum())})",
            linewidths=0,
            zorder=index + 1,
        )
    axis.set(title=("Structure (all-atom mean) descriptor UMAP" if element == "structure"
                    else f"{element} descriptor UMAP"), xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(frameon=False)
    figure.savefig(png, dpi=220)
    figure.savefig(svg)
    plt.close(figure)


def _request_metadata(request: VisualizationRequest) -> dict[str, object]:
    return {
        "schema": _CACHE_SCHEMA,
        "schema_revision": _CACHE_REVISION if request.structure is not None else 1,
        **({"structure_aggregation": "all-atom-mean"} if request.structure is not None else {}),
        "kind": request.kind,
        "layers": [
            {"name": layer.name, "frame_ids": list(layer.frame_ids)}
            for layer in request.layers
        ],
        "descriptor_schema": dict(request.descriptor_schema),
        "type_map": list(request.type_map),
        "model_identity": request.model_identity,
        "dataset_identity": request.dataset_identity,
        "elements": [
            element for element in request.type_map if element in request.elements
        ],
        "source_digest": request.source_digest,
    }


def _temporary_sibling(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))


def _replace_directory(stage: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.old-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(stage, target)
    except BaseException:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_element_name(element: str) -> str:
    if not element or any(not (character.isalnum() or character in "_-") for character in element):
        raise ValueError("visualization element name is unsafe")
    return element


__all__ = [
    "VisualizationDependencyError",
    "load_visualization_request",
    "publish_visualization_request",
    "render_visualization",
    "visualization_output_is_current",
    "render_visualization_cache",
]
