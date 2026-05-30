from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from workflow_backends.student.base import StudentSamplingSummary
from workflow_core.common.io import write_candidate_xyz


def normalize_lammps_sampling_summary(
    *,
    summary_path: str | Path,
    summary_payload: dict[str, Any],
) -> StudentSamplingSummary:
    payload = dict(summary_payload or {})
    resolved_summary_path = Path(summary_path).resolve()
    work_dir = resolved_summary_path.parent
    payload.setdefault("sampling_summary_path", str(resolved_summary_path))
    payload.setdefault("selected_count", int(payload.get("candidate_count", 0) or 0))
    payload.setdefault("effective_sample_count", int(payload.get("total_frames", 0) or 0))
    payload.setdefault("high_unc_count", int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0))
    effective_sample_count = int(payload.get("effective_sample_count", 0) or 0)
    if "high_unc_ratio" not in payload:
        payload["high_unc_ratio"] = float(payload["high_unc_count"] / effective_sample_count) if effective_sample_count else 0.0
    payload.setdefault("failed_uncertainty_count", int(payload.get("failed_count", 0) or 0))
    payload.setdefault("sampling_backend", "lammps")

    selected_source = Path(str(payload.get("selected_xyz") or payload.get("candidate_xyz") or "")).resolve()
    failed_source = Path(str(payload.get("failed_xyz") or "")).resolve() if payload.get("failed_xyz") else None
    uncertainty_source = (
        Path(str(payload.get("uncertainty_records_jsonl") or "")).resolve()
        if payload.get("uncertainty_records_jsonl")
        else None
    )
    candidate_by_system_source = (
        Path(str(payload.get("candidate_by_system_dir") or "")).resolve()
        if payload.get("candidate_by_system_dir")
        else None
    )
    failed_by_system_source = (
        Path(str(payload.get("failed_by_system_dir") or "")).resolve()
        if payload.get("failed_by_system_dir")
        else None
    )

    candidate_xyz = selected_source if selected_source.exists() else work_dir / "candidate.xyz"
    failed_xyz = failed_source if failed_source is not None and failed_source.exists() else work_dir / "failed.xyz"
    uncertainty_jsonl = (
        uncertainty_source
        if uncertainty_source is not None and uncertainty_source.exists()
        else work_dir / "uncertainty_records.jsonl"
    )
    if not candidate_xyz.exists():
        write_candidate_xyz(candidate_xyz, [])
    if not failed_xyz.exists():
        write_candidate_xyz(failed_xyz, [])
    _copy_optional_tree(candidate_by_system_source, work_dir / "candidate_by_system")
    _copy_optional_tree(failed_by_system_source, work_dir / "failed_by_system")
    payload["selected_xyz"] = str(candidate_xyz.resolve())
    payload["candidate_xyz"] = str(candidate_xyz.resolve())
    payload["failed_xyz"] = str(failed_xyz.resolve())
    payload["sampling_summary_path"] = str(resolved_summary_path)
    return StudentSamplingSummary(
        backend="deepmd",
        sampling_backend="lammps",
        summary_path=str(resolved_summary_path),
        work_root=str(work_dir.resolve()),
        selected_xyz=str(candidate_xyz.resolve()),
        candidate_xyz=str(candidate_xyz.resolve()),
        failed_xyz=str(failed_xyz.resolve()),
        uncertainty_records_jsonl=str(uncertainty_jsonl.resolve()) if uncertainty_jsonl.exists() else None,
        effective_sample_count=int(payload.get("effective_sample_count", 0) or 0),
        selected_count=int(payload.get("selected_count", 0) or 0),
        candidate_count=int(payload.get("candidate_count", 0) or 0),
        failed_count=int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0),
        high_unc_count=int(payload.get("high_unc_count", 0) or 0),
        high_unc_ratio=float(payload.get("high_unc_ratio", 0.0) or 0.0),
        payload=payload,
    )


def normalize_gpumd_sampling_summary(
    *,
    summary_path: str | Path,
    summary_payload: dict[str, Any],
) -> StudentSamplingSummary:
    payload = dict(summary_payload or {})
    resolved_summary_path = Path(summary_path).resolve()
    payload.setdefault("selected_count", int(payload.get("candidate_count", 0) or 0))
    payload.setdefault("effective_sample_count", int(payload.get("total_frames", 0) or 0))
    payload.setdefault("high_unc_count", int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0))
    effective_sample_count = int(payload.get("effective_sample_count", 0) or 0)
    if "high_unc_ratio" not in payload:
        payload["high_unc_ratio"] = float(payload["high_unc_count"] / effective_sample_count) if effective_sample_count else 0.0
    payload.setdefault("failed_uncertainty_count", int(payload.get("failed_count", 0) or 0))
    payload.setdefault("sampling_backend", "gpumd")
    return StudentSamplingSummary(
        backend="nep",
        sampling_backend=str(payload.get("sampling_backend", "gpumd")),
        summary_path=str(resolved_summary_path),
        work_root=str(resolved_summary_path.parent.resolve()),
        selected_xyz=str(Path(str(payload.get("selected_xyz") or payload.get("candidate_xyz") or "")).resolve()),
        candidate_xyz=str(Path(str(payload.get("candidate_xyz") or "")).resolve()),
        failed_xyz=str(Path(str(payload.get("failed_xyz") or "")).resolve()),
        uncertainty_records_jsonl=(
            str(Path(str(payload.get("uncertainty_records_jsonl"))).resolve())
            if payload.get("uncertainty_records_jsonl")
            else None
        ),
        effective_sample_count=int(payload.get("effective_sample_count", 0) or 0),
        selected_count=int(payload.get("selected_count", 0) or 0),
        candidate_count=int(payload.get("candidate_count", 0) or 0),
        failed_count=int(payload.get("failed_uncertainty_count", payload.get("failed_count", 0)) or 0),
        high_unc_count=int(payload.get("high_unc_count", 0) or 0),
        high_unc_ratio=float(payload.get("high_unc_ratio", 0.0) or 0.0),
        payload=payload,
    )


def _copy_optional_tree(source: Path | None, target: Path) -> None:
    if source is None or not source.exists() or source == target:
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
