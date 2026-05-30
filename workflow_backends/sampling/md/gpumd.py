from __future__ import annotations

from pathlib import Path

from workflow_backends.execution_context import normalize_backend_request_context
from workflow_backends.gpumd.active_parser import parse_active_out
from workflow_backends.gpumd.sampler import prepare_gpumd_sampling_tasks, run_gpumd_sampling, run_gpumd_sampling_raw
from workflow_runtime.environment import sampling_backend_role


class GPUMDSamplingBackend:
    backend_name = "gpumd"

    def sample_with_committee(self, paths, request) -> str:
        del paths
        if str(request.student_backend or "").strip().lower() != "nep":
            raise ValueError(
                f"sampling_backend='gpumd' requires student_backend='nep', got {request.student_backend!r}"
            )
        normalized = normalize_backend_request_context(request, role=sampling_backend_role(self.backend_name, "sample"))
        context = normalized.context
        summary_path_raw = context.get("committee_summary_path") or request.committee_summary_path
        if not summary_path_raw:
            raise ValueError("GPUMD sampling requires committee_summary_path in request")
        seed_xyz_path = context.get("seed_xyz_path") or request.seed_xyz_path
        if not seed_xyz_path:
            raise ValueError("GPUMD sampling requires seed_xyz_path in request")
        work_root = Path(context.get("work_root", request.work_root)).resolve()
        gpumd_cfg = dict(context.get("gpumd_cfg") or ((request.backend_cfg or {}).get("gpumd", {}) or {}))
        execution = normalized.execution
        gpumd_executable = execution.required_executable("gpumd")
        if gpumd_executable:
            gpumd_cfg["command"] = execution.shell_command([gpumd_executable])
        return run_gpumd_sampling(
            seed_xyz_path=seed_xyz_path,
            model_paths=list(request.committee_model_paths or []),
            gpumd_cfg=gpumd_cfg,
            uncertainty_cfg=dict(context.get("uncertainty_cfg") or ((request.backend_cfg or {}).get("uncertainty", {}) or {})),
            work_root=work_root,
            outer_round=request.teacher_round,
            student_round=request.student_round,
            progress_log_path=context.get("sampling_progress_log_path"),
        )

    def normalize_sampling_summary(self, paths, request, summary_path, summary_payload):
        del paths, request
        from workflow_backends.sampling.summary import normalize_gpumd_sampling_summary

        return normalize_gpumd_sampling_summary(
            summary_path=summary_path,
            summary_payload=dict(summary_payload or {}),
        )


__all__ = [
    "GPUMDSamplingBackend",
    "parse_active_out",
    "prepare_gpumd_sampling_tasks",
    "run_gpumd_sampling",
    "run_gpumd_sampling_raw",
]
