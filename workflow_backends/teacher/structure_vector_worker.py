from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from workflow_backends.compact_structure_vector_store import (
    write_compact_structure_vector_store,
)
from workflow_backends.descriptor_stream import validated_descriptor_chunks
from workflow_core.structures.formats import read_extxyz_structures


RESULT_PREFIX = "DICE_TEACHER_STRUCTURE_VECTORS="


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(header, chunks):
    identities: list[str | None] = [None] * header.frame_count
    means: list[np.ndarray | None] = [None] * header.frame_count
    element_values: dict[str, list[tuple[int, str, np.ndarray]]] = {
        str(element): [] for element in header.type_map
    }
    for chunk in validated_descriptor_chunks(header, chunks):
        for local, ordinal_value in enumerate(chunk.frame_ordinals):
            ordinal = int(ordinal_value)
            start = int(chunk.frame_ptr[local])
            stop = int(chunk.frame_ptr[local + 1])
            identity = str(chunk.frame_ids[local])
            frame_rows = chunk.descriptors[start:stop]
            frame_types = chunk.type_indices[start:stop]
            identities[ordinal] = identity
            means[ordinal] = np.asarray(frame_rows.mean(axis=0), dtype=np.float32)
            for type_index in dict.fromkeys(int(item) for item in frame_types):
                element_values[header.type_map[type_index]].append(
                    (
                        ordinal,
                        identity,
                        np.asarray(
                            frame_rows[frame_types == type_index].mean(axis=0),
                            dtype=np.float32,
                        ),
                    )
                )
    if any(item is None for item in identities) or any(item is None for item in means):
        raise ValueError("teacher structural readout changed canonical frame coverage")
    ordered_ids = tuple(str(item) for item in identities)
    rows = np.ascontiguousarray(np.stack(means), dtype=np.float32)
    elements = {}
    for element, values in element_values.items():
        if not values:
            continue
        values.sort(key=lambda item: item[0])
        elements[element] = (
            tuple(item[1] for item in values),
            np.ascontiguousarray(np.stack([item[2] for item in values])),
        )
    return dict(header.schema), ordered_ids, rows, elements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mace", "deepmd"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--type-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=9472)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    frames = read_extxyz_structures(args.input, allow_empty=False)
    type_map = json.loads(Path(args.type_map).read_text(encoding="utf-8"))
    if not isinstance(type_map, list) or not type_map:
        raise ValueError("teacher structural readout type map is invalid")
    if args.backend == "deepmd":
        from workflow_backends.teacher.deepmd.descriptors import (
            DPA4ReadoutInputExporter,
            dpa4_readout_input_descriptor_stream,
        )

        exporter = DPA4ReadoutInputExporter(args.model, device="cuda")
        header, chunks = dpa4_readout_input_descriptor_stream(
            frames,
            exporter=exporter,
            type_map=tuple(str(item) for item in type_map),
            chunk_frames=args.chunk_frames,
            batch_size=args.batch_size,
        )
        values = _aggregate(header, chunks)
    else:
        from workflow_backends.teacher.mace.descriptors import (
            MaceNodeFeatureWorkerPool,
            mace_readout_input_descriptor_stream,
        )

        with MaceNodeFeatureWorkerPool(
            args.model,
            workers=args.workers,
            batch_size=args.batch_size,
            torch_threads=args.torch_threads,
            device="cuda",
            default_dtype="float32",
        ) as exporter:
            header, chunks = mace_readout_input_descriptor_stream(
                frames,
                type_map=tuple(str(item) for item in type_map),
                exporter=exporter,
            )
            values = _aggregate(header, chunks)
    schema, identities, rows, elements = values
    receipt = write_compact_structure_vector_store(
        args.output,
        descriptor_schema=schema,
        frame_ids=identities,
        frame_means=rows,
        element_means=elements,
        model_digest=_digest(Path(args.model).resolve()),
    )
    print(RESULT_PREFIX + json.dumps(receipt, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
