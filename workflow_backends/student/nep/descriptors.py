"""Model-owned NEP inputs to the final atomic-energy projection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from workflow_backends.descriptor_stream import (
    AtomDescriptorChunk,
    DescriptorStreamHeader,
    READOUT_INPUT_IDENTITY,
    validated_readout_input_schema,
)
from workflow_policies.identities import TORCHNEP_READOUT_INPUT


NEP_READOUT_INPUT_API = TORCHNEP_READOUT_INPUT.name
NEP_READOUT_INPUT_API_REVISION = TORCHNEP_READOUT_INPUT.revision
_HISTORICAL_NEP_READOUT_INPUT_API = "torchnep-readout-input@1"


@dataclass(frozen=True, slots=True)
class NEPReadoutMetadata:
    type_map: tuple[str, ...]
    hidden_widths: tuple[int, ...]
    width: int
    runtime: str
    runtime_version: str
    source_commit: str
    native_source: str


class NEPReadoutInputExporter:
    """Open one canonical NEP4 model and consume its exact native hook."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        runtime_factory: Callable[[str], Any] | None = None,
    ) -> None:
        raw = Path(model_path)
        if (
            raw.is_symlink()
            or raw.suffix != ".txt"
            or not raw.is_file()
            or raw.stat().st_size <= 0
        ):
            raise ValueError(
                f"NEP readout-input requires a non-empty regular native .txt model: {raw}"
            )
        self.model_path = raw.resolve()
        model_type_map, model_hidden = _parse_nep4_architecture(self.model_path)
        factory = runtime_factory or _open_selected_runtime
        try:
            runtime = factory(str(self.model_path))
        except Exception as exc:
            raise RuntimeError("failed to open the selected NEP readout-input runtime") from exc
        capability = getattr(runtime, "readout_input_capability", None)
        if not callable(capability):
            raise RuntimeError(
                f"selected NEP runtime lacks required {NEP_READOUT_INPUT_API} native hook"
            )
        try:
            metadata = _validated_runtime_metadata(capability())
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"selected NEP runtime has an invalid {NEP_READOUT_INPUT_API} capability"
            ) from exc
        if metadata.type_map != model_type_map:
            raise ValueError("NEP runtime type map differs from the canonical model header")
        if metadata.hidden_widths != model_hidden:
            raise ValueError("NEP runtime ANN architecture differs from the canonical model header")
        self.runtime = runtime
        self.metadata = metadata

    def export_frames(self, frames: Sequence[tuple[int, Atoms]]) -> AtomDescriptorChunk:
        if not frames:
            raise ValueError("NEP readout-input export requires a non-empty frame chunk")
        descriptor_rows: list[np.ndarray] = []
        type_rows: list[np.ndarray] = []
        frame_ptr = [0]
        ordinals: list[int] = []
        identities: list[str] = []
        for ordinal, frame in frames:
            identity = str(frame.info.get("structure_id", ""))
            if not identity:
                raise ValueError("NEP readout-input export requires canonical structure_id")
            symbols = tuple(frame.get_chemical_symbols())
            missing = sorted(set(symbols) - set(self.metadata.type_map))
            if missing:
                raise ValueError(f"NEP readout-input type map lacks elements: {missing}")
            compute = getattr(self.runtime, "compute", None)
            if not callable(compute):
                raise RuntimeError(
                    f"selected NEP runtime lacks required {NEP_READOUT_INPUT_API} compute hook"
                )
            result = compute(
                species=symbols,
                positions=np.asarray(frame.get_positions(), dtype=np.float64),
                cell=_runtime_cell(frame, cutoff_hint=8.0),
                return_readout_input=True,
            )
            if not isinstance(result, Mapping) or "readout_input" not in result:
                raise ValueError("NEP native readout-input output is missing")
            values = _to_numpy(result["readout_input"], dtype=np.float32)
            expected = (len(frame), self.metadata.width)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"NEP native readout-input output {values.shape} does not match {expected} or is nonfinite"
                )
            descriptor_rows.append(np.ascontiguousarray(values, dtype=np.float32))
            type_rows.append(np.asarray(
                [self.metadata.type_map.index(symbol) for symbol in symbols],
                dtype=np.int32,
            ))
            frame_ptr.append(frame_ptr[-1] + len(frame))
            ordinals.append(int(ordinal))
            identities.append(identity)
        return AtomDescriptorChunk(
            descriptors=np.ascontiguousarray(np.concatenate(descriptor_rows), dtype=np.float32),
            type_indices=np.concatenate(type_rows),
            frame_ptr=np.asarray(frame_ptr, dtype=np.int64),
            frame_ordinals=np.asarray(ordinals, dtype=np.int64),
            frame_ids=tuple(identities),
        )

    def schema(self) -> dict[str, object]:
        return validated_readout_input_schema({
            **READOUT_INPUT_IDENTITY.as_dict(),
            "backend": "nep",
            "model_family": "nep4",
            "definition": "atomic-energy readout inputs",
            "width": self.metadata.width,
            "dtype": "float32",
            "normalized": False,
            "type_map": list(self.metadata.type_map),
            "components": [{
                "id": "atomic_energy_last_hidden",
                "offset": 0,
                "width": self.metadata.width,
                "native_source": self.metadata.native_source,
                "invariance": "scalar",
                "ann_hidden_widths": list(self.metadata.hidden_widths),
                "runtime": self.metadata.runtime,
                "runtime_version": self.metadata.runtime_version,
                "source_commit": self.metadata.source_commit,
                "output_weight_applied": False,
            }],
            "constant_terms_included": False,
            "deduplication": "unique-native-input",
        })


def nep_readout_input_descriptor_stream(
    frames: Sequence[Atoms],
    *,
    exporter: NEPReadoutInputExporter,
    chunk_frames: int = 16,
) -> tuple[DescriptorStreamHeader, Iterable[AtomDescriptorChunk]]:
    if not frames:
        raise ValueError("NEP readout-input stream requires non-empty canonical frames")
    if isinstance(chunk_frames, bool) or int(chunk_frames) <= 0:
        raise ValueError("NEP descriptor chunk_frames must be positive")
    canonical = tuple(frames)
    identities = tuple(str(frame.info.get("structure_id", "")) for frame in canonical)
    if any(not item for item in identities) or len(set(identities)) != len(identities):
        raise ValueError("NEP descriptor stream requires unique canonical structure_id values")

    def chunks() -> Iterator[AtomDescriptorChunk]:
        for start in range(0, len(canonical), int(chunk_frames)):
            stop = min(start + int(chunk_frames), len(canonical))
            yield exporter.export_frames(tuple(enumerate(canonical[start:stop], start=start)))

    return (
        DescriptorStreamHeader(
            schema=exporter.schema(),
            type_map=exporter.metadata.type_map,
            frame_count=len(canonical),
        ),
        chunks(),
    )


def _parse_nep4_architecture(path: Path) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("NEP readout-input model header must be readable text") from exc
    if not lines:
        raise ValueError("NEP readout-input model header is empty")
    header = lines[0].split()
    if len(header) < 3 or header[0] not in {"nep4", "nep4_zbl"}:
        raise ValueError("NEP readout-input currently requires a native NEP4 model")
    try:
        type_count = int(header[1])
    except ValueError as exc:
        raise ValueError("NEP4 header has an invalid type count") from exc
    type_map = tuple(header[2:])
    if type_count <= 0 or len(type_map) != type_count or len(set(type_map)) != type_count:
        raise ValueError("NEP4 header has an invalid type map")
    ann_rows = [line.split() for line in lines if line.split()[:1] == ["ANN"]]
    if len(ann_rows) != 1 or len(ann_rows[0]) not in {2, 3}:
        raise ValueError("NEP4 model must contain one ANN row with one or two hidden widths")
    try:
        widths = tuple(int(value) for value in ann_rows[0][1:])
    except ValueError as exc:
        raise ValueError("NEP4 ANN widths must be integers") from exc
    if not widths or widths[0] <= 0 or any(value < 0 for value in widths):
        raise ValueError("NEP4 ANN widths are invalid")
    hidden = (widths[0],) if len(widths) == 1 or widths[1] == 0 else widths
    return type_map, hidden


def _validated_runtime_metadata(raw: Any) -> NEPReadoutMetadata:
    if not isinstance(raw, Mapping):
        raise ValueError("wrong NEP readout-input capability API")
    value = dict(raw)
    if value.get("api") == _HISTORICAL_NEP_READOUT_INPUT_API and "api_revision" not in value:
        value["api"] = NEP_READOUT_INPUT_API
        value["api_revision"] = NEP_READOUT_INPUT_API_REVISION
    if (
        value.get("api") != NEP_READOUT_INPUT_API
        or value.get("api_revision") != NEP_READOUT_INPUT_API_REVISION
    ):
        raise ValueError("wrong NEP readout-input capability API")
    type_map = tuple(value.get("type_map", ()))
    hidden_widths = tuple(value.get("hidden_widths", ()))
    if (
        not type_map
        or any(not isinstance(item, str) or not item for item in type_map)
        or len(set(type_map)) != len(type_map)
        or not hidden_widths
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in hidden_widths)
    ):
        raise ValueError("invalid NEP readout-input type map or hidden widths")
    width = value.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width != hidden_widths[-1]:
        raise ValueError("NEP readout-input width differs from final hidden width")
    strings = {
        name: value.get(name)
        for name in ("runtime", "runtime_version", "source_commit", "native_source")
    }
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise ValueError("NEP readout-input runtime provenance is incomplete")
    return NEPReadoutMetadata(
        type_map=type_map,
        hidden_widths=hidden_widths,
        width=width,
        runtime=strings["runtime"],
        runtime_version=strings["runtime_version"],
        source_commit=strings["source_commit"],
        native_source=strings["native_source"],
    )


def _runtime_cell(frame: Atoms, *, cutoff_hint: float) -> np.ndarray:
    positions = np.asarray(frame.get_positions(), dtype=np.float64)
    if positions.shape != (len(frame), 3) or not np.isfinite(positions).all():
        raise ValueError("NEP readout-input frame requires finite Cartesian positions")
    pbc = tuple(bool(value) for value in frame.get_pbc())
    if all(pbc):
        cell = np.asarray(frame.get_cell().array, dtype=np.float64)
        if (
            cell.shape != (3, 3)
            or not np.isfinite(cell).all()
            or abs(float(np.linalg.det(cell))) <= 1.0e-12
        ):
            raise ValueError("periodic NEP readout-input frame requires a finite full-rank cell")
        return cell
    if any(pbc):
        raise ValueError("NEP readout-input does not support partially periodic frames")
    span = np.ptp(positions, axis=0) if len(frame) else np.zeros(3)
    return np.diag(span + 4.0 * float(cutoff_hint) + 1.0)


def _to_numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value, dtype=dtype)


def _open_selected_runtime(model_path: str) -> Any:
    try:
        import torch
        from torchnep.nep import NEPCalculator
    except Exception as exc:  # pragma: no cover - selected profile owns runtime
        raise RuntimeError("selected NEP profile requires the pinned torchnep runtime") from exc
    return NEPCalculator(model_path, dtype=torch.float64, device="cpu")
