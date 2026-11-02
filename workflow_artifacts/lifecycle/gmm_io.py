from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from workflow_core.lifecycle.descriptor_stream import validated_descriptor_chunks
from workflow_policies.gmm_reference_tail import (
    GMMReferenceFitRequest,
    GMMReferenceFitResult,
    GMMReferenceStreamFitRequest,
    GMMScoreRequest,
    GMMScoreResult,
    GMMStreamScoreRequest,
    POLICY_ID,
    POLICY_REVISION,
    fit_gmm_router,
    score_gmm_router,
    score_gmm_router_stream,
    validate_gmm_router,
)


def fit_gmm_router_stream(request: GMMReferenceStreamFitRequest):
    """Adapt a descriptor stream and keep optional spill wholly operation-local."""
    if not request.spill:
        from workflow_policies.gmm_reference_tail import fit_gmm_router_stream as fit

        return fit(request)
    if request.work_dir is None:
        raise ValueError(
            "explicit GMM reference spill requires an operation-local work_dir"
        )
    work_root = Path(request.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".gmm-reference-spill-", dir=work_root
    ) as temporary:
        descriptors, type_indices, frame_indices, frame_ids = _collect_spill(
            request.header,
            request.chunks,
            workspace=Path(temporary),
        )
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


def _collect_spill(header, chunks, *, workspace: Path):
    emission_descriptors = workspace / "emission-descriptors.bin"
    emission_types = workspace / "emission-types.bin"
    spans: list[tuple[int, int] | None] = [None] * header.frame_count
    frame_ids: list[str | None] = [None] * header.frame_count
    atom_count = 0
    with emission_descriptors.open("wb") as descriptor_handle, emission_types.open(
        "wb"
    ) as type_handle:
        for chunk in validated_descriptor_chunks(header, chunks):
            chunk.descriptors.tofile(descriptor_handle)
            chunk.type_indices.astype(np.int32, copy=False).tofile(type_handle)
            for local_index, ordinal_value in enumerate(chunk.frame_ordinals):
                ordinal = int(ordinal_value)
                spans[ordinal] = (
                    atom_count + int(chunk.frame_ptr[local_index]),
                    atom_count + int(chunk.frame_ptr[local_index + 1]),
                )
                frame_ids[ordinal] = chunk.frame_ids[local_index]
            atom_count += len(chunk.descriptors)
    if atom_count <= 0:
        raise ValueError("GMM reference descriptor stream must contain at least one atom")
    width = int(header.schema["width"])
    emitted_rows = np.memmap(
        emission_descriptors,
        mode="r",
        dtype=np.float32,
        shape=(atom_count, width),
    )
    emitted_types = np.memmap(
        emission_types, mode="r", dtype=np.int32, shape=(atom_count,)
    )
    canonical_rows = np.memmap(
        workspace / "canonical-descriptors.bin",
        mode="w+",
        dtype=np.float32,
        shape=(atom_count, width),
    )
    canonical_types = np.memmap(
        workspace / "canonical-types.bin",
        mode="w+",
        dtype=np.int32,
        shape=(atom_count,),
    )
    membership = np.memmap(
        workspace / "canonical-frame-membership.bin",
        mode="w+",
        dtype=np.int64,
        shape=(atom_count,),
    )
    target = 0
    for ordinal, raw_span in enumerate(spans):
        if raw_span is None:
            raise ValueError("descriptor stream is missing canonical frame membership")
        start, stop = raw_span
        count = stop - start
        canonical_rows[target : target + count] = emitted_rows[start:stop]
        canonical_types[target : target + count] = emitted_types[start:stop]
        membership[target : target + count] = ordinal
        target += count
    canonical_rows.flush()
    canonical_types.flush()
    membership.flush()
    del emitted_rows, emitted_types
    emission_descriptors.unlink()
    emission_types.unlink()
    return canonical_rows, canonical_types, membership, tuple(
        str(item) for item in frame_ids
    )


__all__ = [
    "GMMReferenceFitRequest",
    "GMMReferenceFitResult",
    "GMMReferenceStreamFitRequest",
    "GMMScoreRequest",
    "GMMScoreResult",
    "GMMStreamScoreRequest",
    "POLICY_ID",
    "POLICY_REVISION",
    "fit_gmm_router",
    "fit_gmm_router_stream",
    "score_gmm_router",
    "score_gmm_router_stream",
    "validate_gmm_router",
]
