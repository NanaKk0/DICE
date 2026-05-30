from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_backends.lammps.raw_sampling import run_lammps_model_devi_sampling_raw
from workflow_core.student.sampling_postprocess import postprocess_student_sampling_result


def run_lammps_model_devi_sampling(
    *,
    initial_structure_paths: list[str],
    committee_pool: list[dict[str, Any]],
    cfg: dict[str, Any],
    work_root: str | Path,
    summary: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> str:
    context = context or {}
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    backend_result = run_lammps_model_devi_sampling_raw(
        initial_structure_paths=initial_structure_paths,
        committee_pool=committee_pool,
        cfg=cfg,
        work_root=work_root,
        summary=summary,
        context=context,
    )
    summary_path = postprocess_student_sampling_result(
        backend_result=backend_result,
        cfg=cfg,
        work_root=work_root,
        summary=summary,
        context=context,
    )
    return str(summary_path.resolve())
