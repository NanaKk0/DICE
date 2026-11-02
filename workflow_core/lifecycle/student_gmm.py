from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from workflow_core.lifecycle.student_readout import StudentGMMScore
from workflow_policies.gmm_reference_tail import GMMReferenceFitResult


def fit_student_gmm_router(
    *,
    student: Any,
    model: str | Path,
    reference_path: str | Path,
    router_dir: str | Path,
    work_dir: str | Path,
    type_map: Sequence[str],
    progress_reporter: Any | None = None,
) -> GMMReferenceFitResult:
    """Compatibility helper for direct native-operation tests and smoke scripts."""

    return student.fit_gmm_reference(
        model=Path(model).resolve(),
        reference_path=Path(reference_path).resolve(),
        router_dir=Path(router_dir),
        work_dir=Path(work_dir),
        type_map=tuple(str(item) for item in type_map),
        progress_reporter=progress_reporter,
    )


def score_student_gmm_router(
    *,
    student: Any,
    model: str | Path,
    reference_path: str | Path,
    observation_path: str | Path,
    router_dir: str | Path,
    frames: Sequence[Atoms],
    work_dir: str | Path,
    emit_frame_means: bool,
    fit: GMMReferenceFitResult,
    progress_reporter: Any | None = None,
    progress_systems: Mapping[str, str] | None = None,
) -> StudentGMMScore:
    canonical_frames = tuple(frames)
    if not canonical_frames:
        return StudentGMMScore(
            values={},
            frame_means={} if emit_frame_means else None,
            fit=fit,
            descriptor_schema=fit.descriptor_schema,
        )
    expected = tuple(
        str(frame.info.get("structure_id", "")) for frame in canonical_frames
    )
    if any(not identity for identity in expected) or len(set(expected)) != len(
        expected
    ):
        raise ValueError(
            "student GMM candidates require unique non-empty canonical identities"
        )
    scored = student.score_gmm_router(
        model=Path(model).resolve(),
        reference_path=Path(reference_path).resolve(),
        observation_path=Path(observation_path).resolve(),
        ordered_identities=expected,
        router_dir=Path(router_dir),
        work_dir=Path(work_dir),
        emit_frame_means=emit_frame_means,
        progress_reporter=progress_reporter,
        progress_systems=(
            None if progress_systems is None else dict(progress_systems)
        ),
    )
    actual = tuple(str(item) for item in scored.frame_ids.tolist())
    if actual != expected:
        raise ValueError(
            "student GMM score identities do not match canonical candidate order"
        )
    if scored.descriptor_schema != fit.descriptor_schema or scored.descriptor_schema is None:
        raise ValueError(
            "student GMM reference and score descriptor schemas do not match"
        )
    values = {
        identity: float(value)
        for identity, value in zip(actual, scored.frame_p, strict=True)
    }
    means = None
    if emit_frame_means:
        if scored.frame_means is None or scored.frame_means.shape[0] != len(actual):
            raise ValueError(
                "student GMM score is missing requested readout-input frame means"
            )
        means = {
            identity: np.ascontiguousarray(row, dtype=np.float32)
            for identity, row in zip(actual, scored.frame_means, strict=True)
        }
    return StudentGMMScore(
        values=values,
        frame_means=means,
        fit=fit,
        descriptor_schema=scored.descriptor_schema,
    )


def fit_and_score_student_gmm_screen(
    *,
    student: Any,
    model: str | Path,
    reference_path: str | Path,
    frame_chunks: Iterable[Sequence[Atoms]],
    router_dir: str | Path,
    frames: Sequence[Atoms],
    work_dir: str | Path,
    type_map: Sequence[str],
    emit_frame_means: bool,
    progress_reporter: Any | None = None,
    progress_systems: Mapping[str, str] | None = None,
) -> StudentGMMScore:
    """Compatibility helper for the explicit native DPA4C throughput smoke."""

    expected = tuple(str(frame.info.get("structure_id", "")) for frame in frames)
    if not expected or any(not item for item in expected) or len(set(expected)) != len(
        expected
    ):
        raise ValueError("DPA4C fused screen requires unique canonical candidates")
    result = student.fit_and_score_gmm_screen(
        model=Path(model).resolve(),
        reference_path=Path(reference_path).resolve(),
        frame_chunks=frame_chunks,
        ordered_identities=expected,
        router_dir=Path(router_dir),
        work_dir=Path(work_dir),
        type_map=tuple(str(item) for item in type_map),
        emit_frame_means=emit_frame_means,
        progress_reporter=progress_reporter,
        progress_systems=(
            None if progress_systems is None else dict(progress_systems)
        ),
    )
    scored = result.score
    if tuple(scored.identities) != expected:
        raise ValueError("DPA4C fused score identities do not match canonical order")
    if scored.descriptor_schema != result.fit.descriptor_schema:
        raise ValueError("DPA4C fused reference and score descriptor schemas differ")
    means = scored.means if emit_frame_means else None
    if emit_frame_means and means is None:
        raise ValueError("DPA4C fused score is missing requested frame means")
    return StudentGMMScore(
        values=scored.values,
        frame_means=means,
        fit=result.fit,
        descriptor_schema=scored.descriptor_schema,
        compact_view=scored,
    )


__all__ = [
    "fit_and_score_student_gmm_screen",
    "fit_student_gmm_router",
    "score_student_gmm_router",
]
