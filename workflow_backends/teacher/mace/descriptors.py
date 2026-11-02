from __future__ import annotations

import concurrent.futures
import copy
import inspect
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from workflow_backends.descriptor_stream import (
    AtomDescriptorChunk,
    DescriptorStreamHeader,
    READOUT_INPUT_IDENTITY,
)


def _product_output_irreps(product: object) -> object:
    from e3nn import o3

    for raw in (
        getattr(getattr(product, "linear", None), "irreps_out", None),
        getattr(product, "irreps_out", None),
        getattr(product, "target_irreps", None),
    ):
        if raw is not None:
            return o3.Irreps(str(raw))
    raise ValueError(
        f"MACE readout-input cannot resolve product irreps for {product.__class__.__name__}"
    )


def _readout_connected_input_blocks(readout: object, product_irreps: object) -> set[int]:
    from e3nn import o3

    first_linear = getattr(readout, "linear_1", None)
    if first_linear is None:
        first_linear = getattr(readout, "linear", None)
    final_linear = getattr(readout, "linear_2", None)
    if final_linear is None:
        final_linear = getattr(readout, "linear", None)
    if first_linear is None or final_linear is None:
        raise ValueError(
            f"MACE readout-input does not support {readout.__class__.__name__}: "
            "missing native linear readout topology"
        )
    native_input = getattr(first_linear, "irreps_in", None)
    native_output = getattr(final_linear, "irreps_out", None)
    if native_input is None or o3.Irreps(str(native_input)) != product_irreps:
        raise ValueError("MACE product/readout irreps do not describe the same native input")
    if native_output is None:
        raise ValueError("MACE energy readout is missing output irreps")
    output_irreps = o3.Irreps(str(native_output))
    if not output_irreps or any(int(ir.l) != 0 for _mul, ir in output_irreps):
        raise ValueError("MACE energy readout does not terminate in scalar outputs")
    instructions = getattr(first_linear, "instructions", None)
    if not instructions:
        raise ValueError("MACE energy readout exposes no connected native input blocks")
    connected = {int(instruction.i_in) for instruction in instructions}
    if any(index < 0 or index >= len(product_irreps) for index in connected):
        raise ValueError("MACE energy readout instruction references an invalid input block")
    return connected


def build_mace_readout_input_schema(model: object) -> dict[str, Any]:
    """Describe unique invariant product blocks that feed learned energy readouts."""

    products = tuple(getattr(model, "products", ()))
    energy_readouts = tuple(getattr(model, "readouts", ()))
    les_readouts = tuple(getattr(model, "les_readouts", ()))
    if not products or len(products) != len(energy_readouts):
        raise ValueError("MACE readout-input requires one learned energy readout per product")
    if les_readouts and len(les_readouts) != len(products):
        raise ValueError("MACE-LES readout-input requires one LES readout per product")

    components: list[dict[str, Any]] = []
    raw_offset = 0
    output_offset = 0
    layer_irreps: list[str] = []
    for layer_index, product in enumerate(products):
        irreps = _product_output_irreps(product)
        layer_irreps.append(str(irreps))
        connected = _readout_connected_input_blocks(energy_readouts[layer_index], irreps)
        roles = ["energy"]
        if les_readouts:
            connected.update(_readout_connected_input_blocks(les_readouts[layer_index], irreps))
            roles.append("les_energy")

        local_offset = 0
        for block_index, (multiplicity, irrep) in enumerate(irreps):
            width = int(multiplicity) * int(irrep.dim)
            if block_index in connected and int(irrep.l) == 0:
                components.append(
                    {
                        "id": f"energy_readout_{layer_index}_scalar_{block_index}",
                        "offset": int(output_offset),
                        "width": int(width),
                        "native_source": (
                            f"products.{layer_index}:block.{block_index}:"
                            f"{int(multiplicity)}x{irrep}"
                        ),
                        "native_offset": int(raw_offset + local_offset),
                        "irrep": str(irrep),
                        "invariance": "scalar",
                        "readout_roles": roles,
                    }
                )
                output_offset += width
            local_offset += width
        if local_offset != int(irreps.dim):
            raise ValueError("MACE product irreps do not cover their declared native width")
        raw_offset += local_offset

    if not components:
        raise ValueError("MACE energy readouts expose no connected invariant product blocks")
    return {
        **READOUT_INPUT_IDENTITY.as_dict(),
        "backend": "mace",
        "model_family": model.__class__.__name__,
        "definition": "atomic-energy readout inputs",
        "width": int(output_offset),
        "raw_width": int(raw_offset),
        "dtype": "float32",
        "normalized": False,
        "components": components,
        "product_irreps": layer_irreps,
        "constant_terms_included": False,
        "deduplication": "unique-native-input",
    }


def select_mace_readout_input(
    node_features: np.ndarray,
    schema: dict[str, Any],
    *,
    frame_index: int,
) -> np.ndarray:
    """Select one model-derived readout-input row set from concatenated products."""

    values = np.asarray(node_features)
    raw_width = schema.get("raw_width")
    output_width = schema.get("width")
    components = schema.get("components")
    if (
        values.ndim != 2
        or not isinstance(raw_width, int)
        or values.shape[1] != raw_width
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"MACE readout-input frame {frame_index} requires finite atom x {raw_width} node_feats"
        )
    if not isinstance(output_width, int) or output_width <= 0 or not isinstance(components, list):
        raise ValueError("MACE readout-input schema has invalid width or components")
    selected: list[np.ndarray] = []
    expected_output_offset = 0
    native_ranges: list[tuple[int, int]] = []
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("MACE readout-input schema contains an invalid component")
        offset = component.get("offset")
        width = component.get("width")
        native_offset = component.get("native_offset")
        if (
            not isinstance(offset, int)
            or not isinstance(width, int)
            or not isinstance(native_offset, int)
            or width <= 0
            or offset != expected_output_offset
            or native_offset < 0
            or native_offset + width > raw_width
        ):
            raise ValueError("MACE readout-input schema component coverage is invalid")
        native_range = (native_offset, native_offset + width)
        if any(native_range[0] < stop and start < native_range[1] for start, stop in native_ranges):
            raise ValueError("MACE readout-input schema repeats a native input block")
        native_ranges.append(native_range)
        selected.append(values[:, native_range[0] : native_range[1]])
        expected_output_offset += width
    if expected_output_offset != output_width:
        raise ValueError("MACE readout-input schema components do not cover its width")
    return np.ascontiguousarray(np.concatenate(selected, axis=1), dtype=np.float32)


def _descriptor_only_node_features(model: object, data: dict[str, object]) -> object:
    from mace.modules.utils import get_edge_vectors_and_lengths
    import torch

    node_features = model.node_embedding(data["node_attrs"])
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=data["positions"],
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    edge_attributes = model.spherical_harmonics(vectors)
    radial_output = model.radial_embedding(
        lengths,
        data["node_attrs"],
        data["edge_index"],
        model.atomic_numbers,
    )
    if isinstance(radial_output, tuple):
        if len(radial_output) != 2:
            raise ValueError("MACE radial embedding returned an unsupported tuple shape")
        edge_features, cutoff = radial_output
    else:
        edge_features = radial_output
        cutoff = None
    layers = []
    for layer_index, (interaction, product) in enumerate(
        zip(model.interactions, model.products)
    ):
        interaction_arguments = {
            "node_attrs": data["node_attrs"],
            "node_feats": node_features,
            "edge_attrs": edge_attributes,
            "edge_feats": edge_features,
            "edge_index": data["edge_index"],
        }
        interaction_call = getattr(interaction, "forward", interaction)
        parameters = inspect.signature(interaction_call).parameters
        accepts_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_keywords or "cutoff" in parameters:
            interaction_arguments["cutoff"] = cutoff
        if accepts_keywords or "first_layer" in parameters:
            interaction_arguments["first_layer"] = layer_index == 0
        node_features, skip_connection = interaction(**interaction_arguments)
        node_features = product(
            node_feats=node_features,
            sc=skip_connection,
            node_attrs=data["node_attrs"],
        )
        layers.append(node_features)
    return torch.cat(layers, dim=-1)


class _MaceReadoutInputExporter:
    """Keep raw node features private behind canonical readout-input selection."""

    def __init__(self, model_path: str, *, device: str = "cuda", default_dtype: str = "float32") -> None:
        from mace.calculators import MACECalculator

        self.calculator = MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype=default_dtype,
        )
        if self.calculator.num_models != 1:
            raise ValueError("MACE lifecycle descriptor export requires exactly one model")
        self.model = self.calculator.models[0]
        self.model.eval()
        self._readout_input_schema: dict[str, Any] | None = None

    def readout_input_schema(self) -> dict[str, Any]:
        if self._readout_input_schema is None:
            self._readout_input_schema = build_mace_readout_input_schema(self.model)
        return copy.deepcopy(self._readout_input_schema)

    def _export_raw(self, atoms_list: list[object]) -> list[np.ndarray]:
        if not atoms_list:
            raise ValueError("MACE descriptor export requires at least one frame")
        import torch

        result: list[np.ndarray] = []
        for atoms in atoms_list:
            batch = self.calculator._atoms_to_batch(atoms)
            data = self.calculator._clone_batch(batch).to_dict()
            with torch.no_grad():
                node_features = _descriptor_only_node_features(self.model, data)
            result.append(np.ascontiguousarray(node_features.detach().cpu().numpy(), dtype=np.float32))
        return result

    def export_readout_input(self, atoms_list: list[object]) -> list[np.ndarray]:
        schema = self.readout_input_schema()
        return [
            select_mace_readout_input(values, schema, frame_index=frame_index)
            for frame_index, values in enumerate(self._export_raw(atoms_list))
        ]


_WORKER_EXPORTER: _MaceReadoutInputExporter | None = None
_WORKER_BATCH_SIZE = 1


def _set_mace_thread_env(threads: int) -> None:
    if threads <= 0:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(threads))
    try:
        import torch

        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(max(1, min(2, int(threads))))
    except Exception:
        return


def _node_feature_worker_init(
    model_path: str,
    device: str,
    default_dtype: str,
    batch_size: int,
    torch_threads: int,
) -> None:
    global _WORKER_BATCH_SIZE, _WORKER_EXPORTER
    _set_mace_thread_env(torch_threads)
    _WORKER_BATCH_SIZE = max(1, int(batch_size))
    _WORKER_EXPORTER = _MaceReadoutInputExporter(
        model_path,
        device=device,
        default_dtype=default_dtype,
    )


def _readout_input_worker_describe(_task: object) -> dict[str, Any]:
    if _WORKER_EXPORTER is None:
        raise RuntimeError("MACE node-feature worker is not initialized")
    return _WORKER_EXPORTER.readout_input_schema()


def _readout_input_worker_export(task: tuple[int, list[tuple[int, object]]]) -> dict[str, object]:
    """Select model-derived readout inputs before the worker IPC boundary."""

    worker_id, frames = task
    if _WORKER_EXPORTER is None:
        raise RuntimeError("MACE node-feature worker is not initialized")
    features: list[np.ndarray] = []
    atoms_list = [atoms for _frame_index, atoms in frames]
    for start in range(0, len(atoms_list), _WORKER_BATCH_SIZE):
        features.extend(
            _WORKER_EXPORTER.export_readout_input(
                atoms_list[start : start + _WORKER_BATCH_SIZE]
            )
        )
    return {
        "worker_id": int(worker_id),
        "frame_indices": [int(frame_index) for frame_index, _atoms in frames],
        "features": features,
    }


class MaceNodeFeatureWorkerPool:
    """Export node features with persistent spawned MACE worker processes."""

    def __init__(
        self,
        model_path: str,
        *,
        workers: int = 4,
        batch_size: int = 1,
        torch_threads: int = 1,
        device: str = "cuda",
        default_dtype: str = "float32",
    ) -> None:
        if workers <= 0 or batch_size <= 0 or torch_threads <= 0:
            raise ValueError("MACE descriptor workers, batch size, and torch threads must be positive")
        self.model_path = str(Path(model_path).resolve())
        self.workers = int(workers)
        self.batch_size = int(batch_size)
        self.torch_threads = int(torch_threads)
        self.device = str(device)
        self.default_dtype = str(default_dtype)
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._readout_input_schema: dict[str, Any] | None = None

    def _pool(self) -> concurrent.futures.ProcessPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_node_feature_worker_init,
                initargs=(
                    self.model_path,
                    self.device,
                    self.default_dtype,
                    self.batch_size,
                    self.torch_threads,
                ),
            )
        return self._executor

    def readout_input_schema(self) -> dict[str, Any]:
        if self._readout_input_schema is None:
            resolved = list(self._pool().map(_readout_input_worker_describe, (None,)))
            if len(resolved) != 1:
                raise RuntimeError("MACE readout-input schema discovery returned invalid evidence")
            self._readout_input_schema = dict(resolved[0])
        return copy.deepcopy(self._readout_input_schema)

    def iter_readout_input(
        self,
        atoms_list: list[object],
    ) -> Iterator[tuple[tuple[int, ...], tuple[np.ndarray, ...]]]:
        """Yield dynamic-width readout inputs selected inside persistent workers."""

        if not atoms_list:
            raise ValueError("MACE descriptor export requires at least one frame")
        self.readout_input_schema()
        tasks = [
            (
                task_index % self.workers,
                [
                    (frame_index, atoms_list[frame_index])
                    for frame_index in range(start, min(start + self.batch_size, len(atoms_list)))
                ],
            )
            for task_index, start in enumerate(range(0, len(atoms_list), self.batch_size))
        ]
        for result in self._pool().map(_readout_input_worker_export, tasks):
            yield (
                tuple(int(item) for item in result["frame_indices"]),
                tuple(np.ascontiguousarray(item, dtype=np.float32) for item in result["features"]),
            )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def __enter__(self) -> MaceNodeFeatureWorkerPool:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def mace_readout_input_descriptor_stream(
    frames: Sequence[object],
    *,
    type_map: Sequence[str],
    exporter: object,
) -> tuple[DescriptorStreamHeader, Iterable[AtomDescriptorChunk]]:
    """Expose model-derived MACE readout-input chunks in canonical frame order."""

    canonical_frames = tuple(frames)
    return mace_readout_input_descriptor_chunk_stream(
        (canonical_frames,),
        frame_count=len(canonical_frames),
        type_map=type_map,
        exporter=exporter,
    )


def mace_readout_input_descriptor_chunk_stream(
    frame_chunks: Iterable[Sequence[object]],
    *,
    frame_count: int,
    type_map: Sequence[str],
    exporter: object,
) -> tuple[DescriptorStreamHeader, Iterable[AtomDescriptorChunk]]:
    """Replay bounded frame chunks through one persistent MACE worker pool."""

    if frame_count < 0:
        raise ValueError("MACE readout-input chunk stream requires a non-negative frame count")
    elements = tuple(str(item) for item in type_map)
    schema = dict(exporter.readout_input_schema())
    header = DescriptorStreamHeader(
        schema=schema,
        type_map=elements,
        frame_count=frame_count,
    )
    type_indices = {element: index for index, element in enumerate(elements)}
    width = int(schema["width"])

    def chunks() -> Iterator[AtomDescriptorChunk]:
        offset = 0
        for raw_chunk in frame_chunks:
            canonical_frames = tuple(raw_chunk)
            if not canonical_frames:
                raise ValueError("MACE readout-input chunk stream does not accept empty chunks")
            for local_ordinals, feature_frames in exporter.iter_readout_input(list(canonical_frames)):
                if len(local_ordinals) != len(feature_frames):
                    raise ValueError("MACE readout-input worker returned mismatched frame membership")
                rows: list[np.ndarray] = []
                types: list[int] = []
                pointers = [0]
                identities: list[str] = []
                global_ordinals: list[int] = []
                for local_ordinal, feature_rows in zip(local_ordinals, feature_frames):
                    if local_ordinal < 0 or local_ordinal >= len(canonical_frames):
                        raise ValueError("MACE readout-input worker returned an invalid canonical frame ordinal")
                    global_ordinal = offset + int(local_ordinal)
                    frame = canonical_frames[local_ordinal]
                    symbols = tuple(str(item) for item in frame.get_chemical_symbols())
                    values = np.asarray(feature_rows)
                    if (
                        values.dtype != np.float32
                        or values.shape != (len(symbols), width)
                        or not values.flags.c_contiguous
                        or not np.all(np.isfinite(values))
                    ):
                        raise ValueError(
                            f"MACE readout-input frame {global_ordinal} requires finite contiguous "
                            f"float32 atom x {width} rows"
                        )
                    try:
                        types.extend(type_indices[symbol] for symbol in symbols)
                    except KeyError as exc:
                        raise ValueError(
                            f"MACE readout-input frame element is absent from type_map: {exc.args[0]!r}"
                        ) from exc
                    identity = str(frame.info.get("structure_id", ""))
                    if not identity:
                        raise ValueError(
                            f"MACE readout-input frame {global_ordinal} is missing canonical structure_id"
                        )
                    rows.append(values)
                    pointers.append(pointers[-1] + len(values))
                    identities.append(identity)
                    global_ordinals.append(global_ordinal)
                yield AtomDescriptorChunk(
                    descriptors=np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32),
                    type_indices=np.asarray(types, dtype=np.int32),
                    frame_ptr=np.asarray(pointers, dtype=np.int64),
                    frame_ordinals=np.asarray(global_ordinals, dtype=np.int64),
                    frame_ids=tuple(identities),
                )
            offset += len(canonical_frames)
        if offset != frame_count:
            raise ValueError(
                f"MACE readout-input chunk stream expected {frame_count} frames, received {offset}"
            )

    return header, chunks()


__all__ = [
    "build_mace_readout_input_schema", "select_mace_readout_input",
    "MaceNodeFeatureWorkerPool",
    "mace_readout_input_descriptor_stream", "mace_readout_input_descriptor_chunk_stream",
]
