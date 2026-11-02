"""Model-derived DPA4 inputs to the final atomic-energy projection."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from workflow_core.model_families import canonical_deepmd_descriptor_family
from workflow_backends.descriptor_stream import (
    READOUT_INPUT_IDENTITY,
    AtomDescriptorChunk,
    DescriptorStreamHeader,
    validated_readout_input_schema,
)

@dataclass(frozen=True, slots=True)
class DPA4FittingMetadata:
    model_family: str
    type_map: tuple[str, ...]
    configured_neuron: tuple[int, ...]
    resolved_neuron: tuple[int, ...]
    resolved_width: int
    connected_hidden_indices: tuple[int, ...]
    connected_width: int
    upstream_descriptor_width: int | None
    numb_fparam: int
    numb_aparam: int
    dim_case_embd: int
    output_has_bias: bool
    min_abs_output_weight: float


class DeepMDPTReadoutInputExporter:
    """Own one DPA4 readout source and export connected fitting-last rows."""

    def __init__(
        self,
        terminal_path: str | Path,
        *,
        device: str = "cuda",
        terminal_model: Any | None = None,
        model_family: str = "dpa4",
    ) -> None:
        requested_family = canonical_deepmd_descriptor_family(model_family)
        raw_readout_source = Path(terminal_path)
        self.readout_source_path = raw_readout_source.resolve()
        self.terminal_path = self.readout_source_path
        loaded_native_model = terminal_model is None
        if terminal_model is None:
            if raw_readout_source.suffix not in {".pt", ".pt2"}:
                raise ValueError(
                    "DPA4 descriptor readout source requires native suffix .pt or .pt2"
                )
            self._validate_native_model(
                raw_readout_source,
                suffix=raw_readout_source.suffix,
                role="readout source",
            )
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            # The fitting-last API is a model introspection path; compiled
            # inference can hide the Python-side embedding hook.
            os.environ["DP_COMPILE_INFER"] = "0"
            import torch
            from deepmd.infer import DeepPot

            torch.use_deterministic_algorithms(True)

            terminal_model = DeepPot(
                str(self.readout_source_path),
                device=device,
                auto_batch_size=False,
                neighbor_graph_method=(
                    "auto" if requested_family == "dpa4c" else "dense"
                ),
            )
        self.terminal_model = terminal_model
        self.readout_model = terminal_model
        self._requires_pinned_host = loaded_native_model
        self.metadata = self._resolve_metadata(terminal_model)
        self._native_float_dtype = (
            np.float32 if self.metadata.model_family == "dpa4c" else np.float64
        )
        if requested_family not in {"dpa4", "dpa4c"} or (
            loaded_native_model and self.metadata.model_family != requested_family
        ):
            raise ValueError(
                "DeepMD PT readout model family does not match its selected graph route"
            )

    @staticmethod
    def _validate_native_model(path: Path, *, suffix: str, role: str) -> None:
        raw = Path(path)
        if raw.is_symlink() or not raw.is_file() or raw.stat().st_size <= 0:
            raise ValueError(f"DPA4 descriptor {role} must be a non-empty regular file: {raw}")
        if raw.suffix != suffix:
            raise ValueError(f"DPA4 descriptor {role} requires native suffix {suffix}")

    @staticmethod
    def _resolve_metadata(deep_pot: Any) -> DPA4FittingMetadata:
        type_map = tuple(str(item) for item in deep_pot.get_type_map())
        if not type_map or len(set(type_map)) != len(type_map):
            raise ValueError("DPA4 descriptor model has an invalid type_map")
        try:
            model_def = dict(deep_pot.get_model_def_script())
        except Exception as exc:
            raise ValueError("DPA4 descriptor model must expose its native model definition") from exc
        descriptor = dict(model_def.get("descriptor", {}))
        native_model_family = str(descriptor.get("type", ""))
        model_family = canonical_deepmd_descriptor_family(native_model_family)
        if model_family not in {"dpa4", "dpa4c"}:
            raise ValueError(
                "DeepMD-PT readout model must declare a DPA4/SeZM or DPA4C descriptor"
            )
        fitting_def = dict(model_def.get("fitting_net", {}))
        configured = tuple(int(value) for value in fitting_def.get("neuron", ()))
        try:
            deep_eval = deep_pot.deep_eval
            backend_model = getattr(deep_eval, "_dpmodel", None)
            if backend_model is not None:
                atomic_model = backend_model.get_dp_atomic_model()
            else:
                model_wrapper = getattr(deep_eval, "dp", None)
                model_dict = getattr(model_wrapper, "model", None)
                if model_dict is None or "Default" not in model_dict:
                    raise ValueError("DeepMD inference model has no Default model branch")
                model = model_dict["Default"]
                atomic_model = getattr(model, "atomic_model", None)
                if atomic_model is None:
                    raise ValueError("DeepMD inference model has no atomic model")
            upstream_width = int(atomic_model.descriptor.get_dim_out())
            fitting = atomic_model.fitting_net
            resolved_neuron = tuple(int(value) for value in fitting.neuron)
            network_collection = getattr(fitting, "nets", None)
            if network_collection is not None:
                networks = tuple(
                    network
                    for network in network_collection._networks
                    if network is not None
                )
            else:
                network_collection = getattr(fitting, "filter_layers", None)
                networks = tuple(
                    network
                    for network in getattr(network_collection, "networks", ())
                    if network is not None
                )
            if len(networks) != 1:
                raise ValueError("expected one shared fitting network")
            output_layer = getattr(networks[0], "output_layer", None)
            if output_layer is None:
                output_layer = networks[0].layers[-1]
            matrix = getattr(output_layer, "w", None)
            if matrix is None:
                matrix = output_layer.matrix
            matrix_values = (
                matrix.detach().cpu().numpy()
                if hasattr(matrix, "detach")
                else np.asarray(matrix)
            )
            matrix_shape = tuple(int(value) for value in matrix_values.shape)
            vacuum_mask = getattr(fitting, "remove_vaccum_contribution", None)
        except Exception as exc:
            raise ValueError("DPA4 descriptor model lacks resolved fitting/output metadata") from exc
        if len(matrix_shape) != 2 or matrix_shape[0] <= 0 or matrix_shape[1] != 1:
            raise ValueError("DPA4 final atomic-energy projection must have shape hidden_width x 1")
        width = matrix_shape[0]
        if resolved_neuron and resolved_neuron[-1] != width:
            raise ValueError("DPA4 resolved fitting width disagrees with its energy projection")
        weights = np.asarray(matrix_values[:, 0], dtype=np.float64)
        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "DPA4 final projection contains non-finite atomic-energy weights"
            )
        if vacuum_mask is not None and any(bool(value) for value in vacuum_mask):
            raise ValueError(
                "DPA4 readout export does not support active atom_ener / "
                "remove_vaccum_contribution; the effective vacuum-conditioned hidden input "
                "is not exposed by eval_fitting_last_layer"
            )
        connected_hidden_indices = tuple(
            int(index) for index in np.flatnonzero(weights != 0.0)
        )
        if not connected_hidden_indices:
            raise ValueError(
                "DPA4 final atomic-energy projection has no exact nonzero hidden connection"
            )
        metadata = DPA4FittingMetadata(
            model_family=model_family,
            type_map=type_map,
            configured_neuron=configured,
            resolved_neuron=resolved_neuron,
            resolved_width=width,
            connected_hidden_indices=connected_hidden_indices,
            connected_width=len(connected_hidden_indices),
            upstream_descriptor_width=upstream_width,
            numb_fparam=int(fitting.numb_fparam),
            numb_aparam=int(fitting.numb_aparam),
            dim_case_embd=int(fitting.dim_case_embd),
            output_has_bias=(
                getattr(output_layer, "b", None) is not None
                or getattr(output_layer, "bias", None) is not None
            ),
            min_abs_output_weight=float(np.min(np.abs(weights[list(connected_hidden_indices)]))),
        )
        if min(metadata.numb_fparam, metadata.numb_aparam, metadata.dim_case_embd) < 0:
            raise ValueError("DPA4 fitting conditioning dimensions must be non-negative")
        return metadata

    def preflight(self, frames: Sequence[Atoms]) -> None:
        """Validate bounded readout-source fitting-last outputs.

        This deliberately does not evaluate energy or force on a second model:
        the paired checkpoint is the authority for readout extraction, while
        the frozen terminal remains the authority for inference and labeling.
        """

        if not frames:
            raise ValueError("DPA4 readout-source preflight requires at least one frame")
        for frame in tuple(frames)[:2]:
            inputs = self._native_inputs(frame)
            features = self._fitting_last_layer(inputs)
            expected_features = (1, len(frame), self.metadata.resolved_width)
            if features.shape != expected_features or not np.all(np.isfinite(features)):
                raise ValueError(
                    "DPA4 readout-source fitting-last preflight returned invalid values"
                )

    def export_frames(self, frames: Sequence[tuple[int, Atoms]]) -> AtomDescriptorChunk:
        if not frames:
            raise ValueError("DPA4 readout-input export requires a non-empty frame chunk")
        descriptor_rows: list[np.ndarray] = []
        type_rows: list[np.ndarray] = []
        frame_ptr = [0]
        ordinals: list[int] = []
        identities: list[str] = []
        for ordinal, frame in frames:
            inputs = self._native_inputs(frame)
            values = self._fitting_last_layer(inputs)
            expected = (1, len(frame), self.metadata.resolved_width)
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"DPA4 fitting-last output {values.shape} does not match {expected} or is nonfinite"
                )
            descriptor_rows.append(values[0][:, self.metadata.connected_hidden_indices])
            type_rows.append(inputs["atom_types"].astype(np.int32, copy=False))
            frame_ptr.append(frame_ptr[-1] + len(frame))
            ordinals.append(int(ordinal))
            identity = str(frame.info.get("structure_id", ""))
            if not identity:
                raise ValueError("DPA4 readout-input export requires canonical structure_id")
            identities.append(identity)
        return AtomDescriptorChunk(
            descriptors=np.ascontiguousarray(np.concatenate(descriptor_rows, axis=0), dtype=np.float32),
            type_indices=np.concatenate(type_rows),
            frame_ptr=np.asarray(frame_ptr, dtype=np.int64),
            frame_ordinals=np.asarray(ordinals, dtype=np.int64),
            frame_ids=tuple(identities),
        )

    def export_scheduled_batch(
        self,
        frames: Sequence[Any],
        *,
        buffer_cache: Any,
    ) -> AtomDescriptorChunk:
        """Run one public fitting-last forward for one compatible frame batch."""
        if not frames:
            raise ValueError("DeepMD PT scheduled batch cannot be empty")
        inputs = buffer_cache.acquire(self, frames)
        values = self._fitting_last_layer(inputs)
        expected = (len(frames), len(frames[0].frame), self.metadata.resolved_width)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(
                f"DeepMD PT batched fitting-last output {values.shape} does not match {expected}"
            )
        selected = values[:, :, self.metadata.connected_hidden_indices]
        atoms = len(frames[0].frame)
        return AtomDescriptorChunk(
            descriptors=np.ascontiguousarray(
                selected.reshape(-1, selected.shape[-1]), dtype=np.float32
            ),
            type_indices=np.tile(
                np.asarray(inputs["atom_types"], dtype=np.int32), len(frames)
            ),
            frame_ptr=np.arange(0, (len(frames) + 1) * atoms, atoms, dtype=np.int64),
            frame_ordinals=np.asarray([item.ordinal for item in frames], dtype=np.int64),
            frame_ids=tuple(
                str(item.frame.info.get("structure_id", "")) for item in frames
            ),
        )

    def _fitting_last_layer(self, inputs: dict[str, Any]) -> np.ndarray:
        self._set_terminal_case(inputs["case_index"])
        try:
            raw = self.readout_model.eval_fitting_last_layer(
                inputs["coords"],
                inputs["cells"],
                inputs["atom_types"],
                fparam=inputs["fparam"],
                aparam=inputs["aparam"],
                mixed_type=False,
                dtype="fp32",
            )
        except (AttributeError, NotImplementedError) as exc:
            raise ValueError(
                "DeepMD-PT readout source does not expose scalar fitting-last input"
            ) from exc
        return np.asarray(raw, dtype=np.float32)

    def _native_inputs(self, frame: Atoms) -> dict[str, Any]:
        symbols = frame.get_chemical_symbols()
        missing = sorted(set(symbols) - set(self.metadata.type_map))
        if missing:
            raise ValueError(f"DPA4 descriptor type_map lacks elements: {missing}")
        atom_types = np.asarray(
            [self.metadata.type_map.index(symbol) for symbol in symbols], dtype=np.int32
        )
        coords = np.asarray(
            frame.get_positions(), dtype=self._native_float_dtype
        )[None, :, :]
        cells = (
            np.asarray(
                frame.get_cell().array, dtype=self._native_float_dtype
            ).reshape(1, 9)
            if np.any(frame.get_pbc()) else None
        )
        fparam = self._frame_parameter(frame)
        aparam = self._atom_parameter(frame)
        case_index = self._case_index(frame)
        return {
            "coords": coords,
            "cells": cells,
            "atom_types": atom_types,
            "fparam": fparam,
            "aparam": aparam,
            "case_index": case_index,
        }

    def _frame_parameter(self, frame: Atoms) -> np.ndarray | None:
        width = self.metadata.numb_fparam
        if width == 0:
            return None
        if "deepmd_fparam" not in frame.info:
            raise ValueError("DPA4 model requires atoms.info['deepmd_fparam']")
        value = np.asarray(
            frame.info["deepmd_fparam"], dtype=self._native_float_dtype
        )
        if value.shape != (width,) or not np.all(np.isfinite(value)):
            raise ValueError(f"deepmd_fparam must be a finite vector of width {width}")
        return value.reshape(1, width)

    def _atom_parameter(self, frame: Atoms) -> np.ndarray | None:
        width = self.metadata.numb_aparam
        if width == 0:
            return None
        if "deepmd_aparam" not in frame.arrays:
            raise ValueError("DPA4 model requires atoms.arrays['deepmd_aparam']")
        value = np.asarray(
            frame.arrays["deepmd_aparam"], dtype=self._native_float_dtype
        )
        expected = (len(frame), width)
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise ValueError(f"deepmd_aparam must have finite shape {expected}")
        return value.reshape(1, len(frame), width)

    def _case_index(self, frame: Atoms) -> int | None:
        width = self.metadata.dim_case_embd
        if width == 0:
            return None
        raw = frame.info.get("deepmd_case_index")
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ValueError("DPA4 model requires integer atoms.info['deepmd_case_index']")
        value = int(raw)
        if value < 0 or value >= width:
            raise ValueError(f"deepmd_case_index must be in [0, {width})")
        return value

    def _set_terminal_case(self, case_index: int | None) -> None:
        if case_index is None:
            return
        candidates = [
            self.terminal_model,
            getattr(self.terminal_model, "deep_eval", None),
        ]
        deep_eval = getattr(self.terminal_model, "deep_eval", None)
        backend_model = getattr(deep_eval, "_dpmodel", None)
        if backend_model is not None:
            candidates.append(backend_model)
            atomic_model = backend_model.get_dp_atomic_model()
            candidates.extend((atomic_model, atomic_model.fitting_net))
        for candidate in candidates:
            setter = getattr(candidate, "set_case_embd", None)
            if callable(setter):
                setter(case_index)
                return
        raise ValueError(
            "DPA4 terminal declares case embedding but exposes no case setter"
        )

    def schema(self) -> dict[str, object]:
        component: dict[str, object] = {
            "id": "atomic_energy_fitting_last_hidden",
            "offset": 0,
            "width": self.metadata.connected_width,
            "native_source": "DeepPot.eval_fitting_last_layer",
            "invariance": "scalar",
            "configured_fitting_neuron": list(self.metadata.configured_neuron),
            "resolved_fitting_neuron": list(self.metadata.resolved_neuron),
            "resolved_fitting_width": self.metadata.resolved_width,
            "retained_hidden_indices": list(self.metadata.connected_hidden_indices),
            "upstream_descriptor_width": self.metadata.upstream_descriptor_width,
            "energy_output_weight_shape": [self.metadata.resolved_width, 1],
            "energy_output_has_bias": self.metadata.output_has_bias,
            "energy_connected_hidden_width": self.metadata.connected_width,
            "energy_connection_test": "exact-nonzero-final-projection-weight",
            "min_abs_energy_output_weight": self.metadata.min_abs_output_weight,
            "conditioning": {
                "numb_fparam": self.metadata.numb_fparam,
                "numb_aparam": self.metadata.numb_aparam,
                "dim_case_embd": self.metadata.dim_case_embd,
                "fparam_source": "atoms.info.deepmd_fparam",
                "aparam_source": "atoms.arrays.deepmd_aparam",
                "case_source": "atoms.info.deepmd_case_index",
            },
        }
        return validated_readout_input_schema({
            **READOUT_INPUT_IDENTITY.as_dict(),
            "backend": "deepmd-pt",
            "model_family": self.metadata.model_family,
            "definition": "atomic-energy readout inputs",
            "width": self.metadata.connected_width,
            "dtype": "float32",
            "normalized": False,
            "type_map": list(self.metadata.type_map),
            "components": [component],
            "constant_terms_included": False,
            "deduplication": "unique-native-input",
        })


def dpa4_readout_input_descriptor_stream(
    frames: Sequence[Atoms],
    *,
    exporter: DeepMDPTReadoutInputExporter,
    type_map: Sequence[str] | None = None,
    chunk_frames: int = 16,
    batch_size: int = 9472,
) -> tuple[DescriptorStreamHeader, Iterable[AtomDescriptorChunk]]:
    if not frames:
        raise ValueError("DPA4 readout-input stream requires non-empty canonical frames")
    canonical = tuple(frames)
    return dpa4_readout_input_descriptor_chunk_stream(
        (canonical,),
        frame_count=len(canonical),
        exporter=exporter,
        type_map=type_map,
        chunk_frames=chunk_frames,
        batch_size=batch_size,
    )


def dpa4_readout_input_descriptor_chunk_stream(
    frame_chunks: Iterable[Sequence[Atoms]],
    *,
    frame_count: int,
    exporter: DeepMDPTReadoutInputExporter,
    type_map: Sequence[str] | None = None,
    chunk_frames: int = 16,
    batch_size: int = 9472,
) -> tuple[DescriptorStreamHeader, Iterable[AtomDescriptorChunk]]:
    """Stream bounded replay chunks through one loaded terminal and preflight."""
    if isinstance(chunk_frames, bool) or int(chunk_frames) <= 0:
        raise ValueError("DPA4 descriptor chunk_frames must be positive")
    if isinstance(frame_count, bool) or int(frame_count) <= 0:
        raise ValueError("DPA4 descriptor frame_count must be positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("DPA4 runtime.batch_size must be positive total atoms, not frames")
    active_type_map = (
        exporter.metadata.type_map
        if type_map is None
        else tuple(str(item) for item in type_map)
    )
    if (
        not active_type_map
        or any(not item for item in active_type_map)
        or len(set(active_type_map)) != len(active_type_map)
    ):
        raise ValueError("DPA4 active type_map must contain unique elements")
    missing_native = tuple(
        item for item in active_type_map if item not in exporter.metadata.type_map
    )
    if missing_native:
        raise ValueError(
            f"DPA4 active type_map contains elements absent from the native model: {missing_native}"
        )
    active_indices = {element: index for index, element in enumerate(active_type_map)}
    def canonical_frames() -> Iterator[tuple[int, Atoms]]:
        ordinal = 0
        for replay_chunk in frame_chunks:
            bounded = tuple(replay_chunk)
            if not bounded:
                raise ValueError("DPA4 descriptor replay chunks must be non-empty")
            for frame in bounded:
                identity = str(frame.info.get("structure_id", ""))
                if not identity:
                    raise ValueError(
                        "DPA4 descriptor stream requires canonical structure_id values"
                    )
                yield ordinal, frame
                ordinal += 1
        if ordinal != int(frame_count):
            raise ValueError("DPA4 descriptor replay changed canonical frame count")

    from workflow_backends.deepmd_pt_runtime import (
        PinnedHostBufferLRU,
        controlled_export_batches,
        cuda_memory_controller,
        record_runtime_diagnostics,
        schedule_same_topology_batches,
    )

    scheduled = schedule_same_topology_batches(
        canonical_frames(),
        exporter=exporter,
        batch_size=batch_size,
        pending_watermark=max(64, 4 * int(chunk_frames)),
    )
    buffer_cache = PinnedHostBufferLRU(
        allocator=(
            None
            if exporter._requires_pinned_host
            else lambda shape, dtype: np.empty(shape, dtype=dtype)
        )
    )
    if exporter._requires_pinned_host:
        memory_controller, memory_snapshot, memory_evict = cuda_memory_controller()
    else:
        memory_controller = memory_snapshot = memory_evict = None

    def export_batch(batch: tuple[Any, ...]) -> Iterator[AtomDescriptorChunk]:
        frame_elements = {
            str(symbol)
            for item in batch
            for symbol in item.frame.get_chemical_symbols()
        }
        missing_active = tuple(sorted(frame_elements - set(active_type_map)))
        if missing_active:
            raise ValueError(
                "DPA4 descriptor frames contain elements absent from the active "
                f"type_map: {missing_active}"
            )
        for raw in controlled_export_batches(
            exporter,
            batch,
            buffer_cache=buffer_cache,
            controller=memory_controller,
            snapshot=memory_snapshot,
            evict=memory_evict,
        ):
            route_types = np.asarray(
                [
                    active_indices[str(symbol)]
                    for ordinal in raw.frame_ordinals
                    for symbol in next(
                        item.frame for item in batch if item.ordinal == int(ordinal)
                    ).get_chemical_symbols()
                ],
                dtype=np.int32,
            )
            yield AtomDescriptorChunk(
                descriptors=raw.descriptors,
                type_indices=route_types,
                frame_ptr=raw.frame_ptr,
                frame_ordinals=raw.frame_ordinals,
                frame_ids=raw.frame_ids,
            )

    batch_iterator = scheduled
    try:
        first_batch = next(batch_iterator)
    except StopIteration as exc:
        raise ValueError("DPA4 descriptor stream requires non-empty replay chunks") from exc
    first_chunks = export_batch(first_batch)
    first = next(first_chunks)
    schema = exporter.schema()
    if first.descriptors.shape[1] != schema["width"]:
        raise ValueError("DPA4 observed fitting width disagrees with resolved projection metadata")

    def chunks() -> Iterator[AtomDescriptorChunk]:
        try:
            yield first
            yield from first_chunks
            for batch in batch_iterator:
                yield from export_batch(batch)
        finally:
            record_runtime_diagnostics(
                host_cache=buffer_cache.report(),
                memory_controller=(
                    None if memory_controller is None else memory_controller.report()
                ),
            )

    header = DescriptorStreamHeader(
        schema=schema,
        type_map=active_type_map,
        frame_count=int(frame_count),
    )
    return header, chunks()


DPA4ReadoutInputExporter = DeepMDPTReadoutInputExporter
deepmd_pt_readout_input_descriptor_chunk_stream = (
    dpa4_readout_input_descriptor_chunk_stream
)
deepmd_pt_readout_input_descriptor_stream = dpa4_readout_input_descriptor_stream


__all__ = [
    "DPA4FittingMetadata",
    "DPA4ReadoutInputExporter",
    "DeepMDPTReadoutInputExporter",
    "dpa4_readout_input_descriptor_chunk_stream",
    "dpa4_readout_input_descriptor_stream",
    "deepmd_pt_readout_input_descriptor_chunk_stream",
    "deepmd_pt_readout_input_descriptor_stream",
]
