from __future__ import annotations

from workflow_backends.sampling.base import MDSamplingRequest
from workflow_backends.student.base import StudentSamplingRequest


def build_md_sampling_request(request: StudentSamplingRequest) -> MDSamplingRequest:
    return MDSamplingRequest(
        student_backend=str(request.backend_name),
        teacher_round=int(request.outer_round),
        student_round=int(request.student_round),
        committee_summary_path=str(request.committee_summary_path),
        seed_xyz_path=str(request.seed_xyz_path),
        work_root=str(request.work_root),
        backend_cfg=request.backend_cfg,
        reference_xyz_files=list(request.reference_xyz_files or []),
        uncertainty_cfg=dict(request.uncertainty_cfg or {}),
        summary_extra=dict(request.summary_extra or {}),
        context=dict(request.context or {}),
        committee_model_paths=list(request.committee_model_paths or []),
        backend_identity=dict(request.backend_identity or {}),
        stage_execution=dict(request.stage_execution or {}),
        md_expansion=dict(request.md_expansion or {}),
        native_recipe=dict(request.native_recipe or {}),
        scalar_overrides=dict(request.scalar_overrides or {}),
        output_requirements=dict(request.output_requirements or {}),
    )
