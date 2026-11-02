from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Iterator, Sequence

from ase import Atoms
import numpy as np

from workflow_backends.descriptor_stream import (
    AtomDescriptorChunk,
    DescriptorStreamHeader,
    READOUT_INPUT_IDENTITY,
)
from workflow_policies.identities import VALIDATED_FINAL_LAYER_NAME


@dataclass(frozen=True, slots=True)
class DeepMDTFReadoutResolution:
    tensors: tuple[Any, ...]
    widths: tuple[int, ...]
    projection_operations: tuple[str, ...]
    strategy_name: str
    strategy_revision: int
    energy_tensor: str


def _static_shape(value: Any) -> tuple[int | None, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    if hasattr(shape, "as_list"):
        shape = shape.as_list()
    try:
        return tuple(None if item is None else int(item) for item in shape)
    except (TypeError, ValueError):
        return ()


def _feeds_tensor(source: Any, target: Any) -> bool:
    pending = [source]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if value is target or getattr(value, "name", None) == getattr(target, "name", None):
            return True
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        consumers = getattr(value, "consumers", None)
        for operation in (() if not callable(consumers) else consumers()):
            pending.extend(tuple(getattr(operation, "outputs", ())))
    return False


def resolve_deepmd_tf_readout_inputs(
    graph: Any,
    *,
    energy_tensor: Any,
    type_count: int,
) -> DeepMDTFReadoutResolution:
    """Validate the frozen-graph final-layer naming strategy."""

    if isinstance(type_count, bool) or not isinstance(type_count, int) or type_count <= 0:
        raise ValueError("DeepMD-TF readout-input resolver requires a positive type count")
    if energy_tensor is None or not isinstance(getattr(energy_tensor, "name", None), str):
        raise ValueError("DeepMD-TF readout-input resolver requires the native atomic-energy tensor")
    tensors: list[Any] = []
    widths: list[int] = []
    projections: list[str] = []
    try:
        operations = tuple(
            graph.get_operation_by_name(f"load/final_layer_type_{index}/MatMul")
            for index in range(type_count)
        )
    except Exception as exc:
        raise ValueError(
            "unsupported DeepMD-TF readout-input graph: validated final-layer-name "
            "could not resolve every type projection"
        ) from exc
    for operation in operations:
        inputs = tuple(getattr(operation, "inputs", ()))
        outputs = tuple(getattr(operation, "outputs", ()))
        if getattr(operation, "type", None) != "MatMul" or len(inputs) != 2 or len(outputs) != 1:
            raise ValueError("DeepMD-TF final projection is not one scalar MatMul")
        output_shape = _static_shape(outputs[0])
        candidates: list[tuple[Any, int]] = []
        for hidden_index, weight_index in ((0, 1), (1, 0)):
            hidden_shape = _static_shape(inputs[hidden_index])
            weight_shape = _static_shape(inputs[weight_index])
            if (
                len(hidden_shape) == 2
                and len(weight_shape) == 2
                and isinstance(hidden_shape[1], int)
                and hidden_shape[1] > 0
                and weight_shape == (hidden_shape[1], 1)
            ):
                candidates.append((inputs[hidden_index], hidden_shape[1]))
        if len(candidates) != 1 or len(output_shape) != 2 or output_shape[1] != 1:
            raise ValueError("DeepMD-TF final projection does not expose one model-width hidden input")
        if not _feeds_tensor(outputs[0], energy_tensor):
            raise ValueError("DeepMD-TF named final projection does not feed the native atomic energy")
        hidden, width = candidates[0]
        tensors.append(hidden)
        widths.append(width)
        projections.append(str(operation.name))
    if len(set(widths)) != 1 or len({str(tensor.name) for tensor in tensors}) != type_count:
        raise ValueError("DeepMD-TF per-type final projections have incompatible hidden inputs")
    return DeepMDTFReadoutResolution(
        tensors=tuple(tensors),
        widths=tuple(widths),
        projection_operations=tuple(projections),
        strategy_name=VALIDATED_FINAL_LAYER_NAME.name,
        strategy_revision=VALIDATED_FINAL_LAYER_NAME.revision,
        energy_tensor=str(energy_tensor.name),
    )


@contextmanager
def open_deepmd_readout_input_descriptor_stream(
    model_path: str | Path,
    frames: Sequence[Atoms],
    *,
    chunk_frames: int = 50,
) -> Iterator[tuple[DescriptorStreamHeader, Iterator[AtomDescriptorChunk]]]:
    """Open one model-owning energy readout-input stream over complete topology batches."""
    canonical_frames = list(frames)
    if not canonical_frames:
        raise ValueError("DeepMD-TF readout-input descriptor stream requires at least one frame")
    with open_deepmd_readout_input_descriptor_chunk_stream(
        model_path,
        (canonical_frames,),
        frame_count=len(canonical_frames),
        chunk_frames=chunk_frames,
    ) as stream:
        yield stream


@contextmanager
def open_deepmd_readout_input_descriptor_chunk_stream(
    model_path: str | Path,
    frame_chunks: Iterator[Sequence[Atoms]] | Sequence[Sequence[Atoms]],
    *,
    frame_count: int,
    chunk_frames: int = 50,
) -> Iterator[tuple[DescriptorStreamHeader, Iterator[AtomDescriptorChunk]]]:
    """Replay bounded geometry chunks through one loaded DeepPot session."""

    if frame_count <= 0:
        raise ValueError("DeepMD-TF readout-input chunk stream requires a positive frame count")
    try:
        from deepmd.infer import DeepPot
    except Exception as exc:  # pragma: no cover - native environment only.
        raise RuntimeError("DeepMD-TF readout-input export requires deepmd.infer.DeepPot in the selected profile") from exc
    deep_pot = DeepPot(str(Path(model_path).absolute()))
    deep_eval = getattr(deep_pot, "deep_eval", deep_pot)
    graph = deep_pot.get_model()
    type_map = tuple(str(item) for item in deep_pot.get_type_map())
    try:
        output_tensors = getattr(deep_eval, "output_tensors", {})
        resolution = resolve_deepmd_tf_readout_inputs(
            graph,
            energy_tensor=output_tensors.get("energy") if isinstance(output_tensors, dict) else None,
            type_count=len(type_map),
        )
        tensors = resolution.tensors
    except Exception as exc:
        _close_deepmd_session(deep_eval)
        raise ValueError(
            "unsupported DeepMD-TF GMM model: final atomic-energy readout inputs "
            "could not be validated"
        ) from exc

    def _chunks() -> Iterator[AtomDescriptorChunk]:
        size = max(1, int(chunk_frames))
        offset = 0
        for raw_chunk in frame_chunks:
            canonical_frames = list(raw_chunk)
            if not canonical_frames:
                raise ValueError("DeepMD-TF readout-input chunk stream does not accept empty chunks")
            groups: dict[tuple[str, ...], list[tuple[int, Atoms]]] = {}
            for local_ordinal, frame in enumerate(canonical_frames):
                symbols = tuple(str(item) for item in frame.get_chemical_symbols())
                unknown = sorted(set(symbols) - set(type_map))
                if unknown:
                    raise ValueError(
                        f"DeepMD-TF readout-input contains symbols absent from model type_map: {unknown}"
                    )
                groups.setdefault(symbols, []).append((local_ordinal, frame))
            for group in groups.values():
                for start in range(0, len(group), size):
                    batch = group[start : start + size]
                    local_ordinals, atoms_list = zip(*batch)
                    ordinals = tuple(offset + ordinal for ordinal in local_ordinals)
                    symbols = tuple(str(item) for item in atoms_list[0].get_chemical_symbols())
                    atom_types = np.asarray([type_map.index(symbol) for symbol in symbols], dtype=np.int32)
                    coords = np.stack([atoms.get_positions() for atoms in atoms_list], axis=0)
                    cells = np.stack([atoms.get_cell().array.reshape(9) for atoms in atoms_list], axis=0)
                    feed, imap, natoms_vec, ghost_map, _sel_at, _sel_imap = deep_eval._prepare_feed_dict(
                        coords, cells, np.tile(atom_types[None, :], (len(atoms_list), 1))
                    )
                    if ghost_map is not None:
                        raise ValueError("DeepMD-TF readout-input GMM export does not support neighbor-list ghost atoms")
                    fetches: list[Any] = []
                    fetched: list[int] = []
                    for type_index, tensor in enumerate(tensors):
                        if int(natoms_vec[type_index + 2]) > 0:
                            fetches.append(tensor)
                            fetched.append(type_index)
                    values = deep_eval.sess.run(fetches, feed) if fetches else []
                    blocks: list[np.ndarray] = []
                    for type_index, rows in zip(fetched, values):
                        count = int(natoms_vec[type_index + 2])
                        raw = np.asarray(rows, dtype=np.float32)
                        if raw.ndim != 2 or raw.shape[0] != len(atoms_list) * count:
                            raise ValueError("DeepMD-TF readout-input export returned an unexpected per-type row count")
                        blocks.append(raw.reshape(len(atoms_list), count, -1))
                    if not blocks:
                        raise ValueError("DeepMD-TF readout-input export found no model-type rows")
                    sorted_rows = np.concatenate(blocks, axis=1)
                    original_rows = np.empty_like(sorted_rows)
                    original_rows[:, np.asarray(imap[: len(symbols)], dtype=np.int64), :] = sorted_rows
                    descriptors = np.ascontiguousarray(original_rows.reshape(-1, original_rows.shape[-1]), dtype=np.float32)
                    yield AtomDescriptorChunk(
                        descriptors=descriptors,
                        type_indices=np.tile(atom_types, len(atoms_list)),
                        frame_ptr=np.arange(0, descriptors.shape[0] + 1, len(symbols), dtype=np.int64),
                        frame_ordinals=np.asarray(ordinals, dtype=np.int64),
                        frame_ids=tuple(str(atoms.info.get("structure_id", "")) for atoms in atoms_list),
                    )
            offset += len(canonical_frames)
        if offset != frame_count:
            raise ValueError(
                f"DeepMD-TF readout-input chunk stream expected {frame_count} frames, received {offset}"
            )

    chunks = _chunks()
    try:
        first = next(chunks)
        width = int(first.descriptors.shape[1])
        if width != resolution.widths[0]:
            raise ValueError("DeepMD-TF observed final hidden width disagrees with its scalar projection")
        schema = {
            **READOUT_INPUT_IDENTITY.as_dict(),
            "backend": "deepmd-tf",
            "model_family": "se_e2_a",
            "definition": "atomic-energy readout inputs",
            "width": width,
            "dtype": "float32",
            "normalized": False,
            "type_map": list(type_map),
            "components": [
                {
                    "id": "atomic_energy_final_hidden",
                    "offset": 0,
                    "width": width,
                    "native_source": "per-type final fitting hidden activation",
                    "invariance": "scalar",
                    "resolved_tensors": [str(tensor.name) for tensor in tensors],
                    "projection_operations": list(resolution.projection_operations),
                }
            ],
            "constant_terms_included": False,
            "deduplication": "unique-native-input",
            "resolver": {
                "strategy_name": resolution.strategy_name,
                "strategy_revision": resolution.strategy_revision,
                "energy_tensor": resolution.energy_tensor,
            },
        }
        header = DescriptorStreamHeader(schema=schema, type_map=type_map, frame_count=frame_count)
        yield header, itertools.chain((first,), chunks)
    finally:
        chunks.close()
        _close_deepmd_session(deep_eval)


def _close_deepmd_session(deep_eval: Any) -> None:
    close = getattr(getattr(deep_eval, "sess", None), "close", None)
    if callable(close):
        close()


__all__ = [
    "open_deepmd_readout_input_descriptor_stream",
    "open_deepmd_readout_input_descriptor_chunk_stream",
    "resolve_deepmd_tf_readout_inputs",
]
