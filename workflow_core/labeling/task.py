from pathlib import Path

from workflow_backends.student import get_student_backend
from workflow_backends.student.config import resolve_student_backend_config
from workflow_core.common.io import read_json
from workflow_core.labeling.request import (
    StudentLabelExportRequest,
    TeacherLabelingRequest,
    build_teacher_labeling_request,
)
from workflow_core.labeling.routing import TeacherRoutingPolicy
from workflow_core.labeling.service import run_teacher_labeling


def build_label_core_config(label_cfg, teacher_dir):
    core_cfg = dict(label_cfg)
    core_cfg["teacher_dir"] = str(teacher_dir)
    return core_cfg


def build_label_export_config(paths, round_id, element_order):
    del paths, round_id, element_order
    raise RuntimeError("target-mode labeling requires explicit TeacherLabelingRequest output paths")


def _load_resolved_student_train_config(train_committee_config):
    if not train_committee_config:
        return None, None, None
    config_path = Path(train_committee_config).resolve()
    raw_cfg = read_json(config_path)
    resolved_cfg = resolve_student_backend_config(dict(raw_cfg), config_path=config_path)
    backend = get_student_backend(resolved_cfg["student_backend"])
    return config_path, resolved_cfg, backend


def _resolve_student_export_request(context_or_request):
    if isinstance(context_or_request, TeacherLabelingRequest):
        return context_or_request.student_export
    if isinstance(context_or_request, StudentLabelExportRequest):
        return context_or_request
    if isinstance(context_or_request, dict):
        element_order = context_or_request.get("element_order_override")
        return StudentLabelExportRequest(
            train_committee_config=context_or_request.get("train_committee_config_override"),
            training_stage=context_or_request.get("training_stage_override"),
            valid_fraction=context_or_request.get("valid_fraction_override"),
            element_order=[str(item) for item in element_order] if element_order is not None else None,
        )
    if context_or_request is None:
        return StudentLabelExportRequest()
    raise TypeError(f"Unsupported student export request: {type(context_or_request)!r}")


def resolve_label_train_ratio(label_cfg, context_or_request):
    export_request = _resolve_student_export_request(context_or_request)
    if export_request.valid_fraction is not None:
        valid_fraction = float(export_request.valid_fraction)
        if not 0.0 < valid_fraction < 1.0:
            raise ValueError(f"valid_fraction must be between 0 and 1, got {export_request.valid_fraction!r}")
        return 1.0 - valid_fraction

    train_committee_config = export_request.train_committee_config
    if train_committee_config:
        config_path, resolved_cfg, backend = _load_resolved_student_train_config(train_committee_config)
        if resolved_cfg is None or backend is None:
            raise RuntimeError(f"Unable to load student train config: {train_committee_config}")
        return backend.resolve_train_ratio(
            resolved_cfg,
            config_path=config_path,
            training_stage=export_request.training_stage,
        )

    return float(label_cfg.get("train_ratio", 0.9))


def resolve_label_element_order(label_cfg, context_or_request):
    export_request = _resolve_student_export_request(context_or_request)
    if export_request.element_order is not None:
        return [str(item) for item in export_request.element_order]

    train_committee_config = export_request.train_committee_config
    if train_committee_config:
        config_path, resolved_cfg, backend = _load_resolved_student_train_config(train_committee_config)
        if resolved_cfg is None or backend is None:
            raise RuntimeError(f"Unable to load student train config: {train_committee_config}")
        return backend.resolve_element_order(
            resolved_cfg,
            config_path=config_path,
            training_stage=export_request.training_stage,
        )

    legacy_element_order = label_cfg.get("element_order")
    if isinstance(legacy_element_order, list) and legacy_element_order:
        return [str(item) for item in legacy_element_order]

    raise KeyError("Unable to resolve label export element order from request or student train config")


def export_cfg_preview_label_dir(context_or_request, paths, round_id):
    if isinstance(context_or_request, TeacherLabelingRequest):
        return Path(context_or_request.output_targets.labeled_output_dir).resolve()
    if isinstance(context_or_request, dict):
        override = context_or_request.get("student_round_dir_override")
        if override is not None:
            return Path(override).resolve()
    del paths, round_id
    raise RuntimeError("target-mode labeling requires explicit labeled output paths")


def export_cfg_preview_al_dir(context_or_request, paths, round_id):
    if isinstance(context_or_request, TeacherLabelingRequest):
        return Path(context_or_request.output_targets.active_learning_output_dir).resolve()
    if isinstance(context_or_request, dict):
        override = context_or_request.get("active_learning_round_dir_override")
        if override is not None:
            return Path(override).resolve()
    del paths, round_id
    raise RuntimeError("target-mode labeling requires explicit active-learning output paths")


def _coerce_teacher_labeling_request(paths, round_id, request=None, context=None):
    if isinstance(request, TeacherLabelingRequest):
        return request
    if request is not None and context is not None:
        raise ValueError("Pass either request or context, not both")
    if isinstance(request, dict):
        context = request
    if isinstance(context, TeacherLabelingRequest):
        return context
    if isinstance(context, dict):
        teacher_round = int(context.get("teacher_round_override", round_id))
        if context.get("teacher_dir_override") is None:
            raise ValueError("target-mode labeling context requires teacher_dir_override")
        teacher_dir = Path(context["teacher_dir_override"]).resolve()
        export_request = _resolve_student_export_request(context)
        merged_xyz = context.get("merged_xyz_override")
        if context.get("structure_file_override") is None:
            raise ValueError("target-mode labeling context requires structure_file_override")
        if context.get("student_round_dir_override") is None:
            raise ValueError("target-mode labeling context requires student_round_dir_override")
        if context.get("active_learning_round_dir_override") is None:
            raise ValueError("target-mode labeling context requires active_learning_round_dir_override")
        return build_teacher_labeling_request(
            structure_file=Path(context["structure_file_override"]).resolve(),
            teacher_round=teacher_round,
            teacher_dir=teacher_dir,
            labeled_output_dir=Path(context["student_round_dir_override"]).resolve(),
            active_learning_output_dir=Path(context["active_learning_round_dir_override"]).resolve(),
            include_user_student_train=bool(context.get("include_user_student_train", True)),
            merged_xyz=Path(merged_xyz).resolve() if merged_xyz is not None else None,
            append_active_learning_xyz=context.get("append_active_learning_xyz"),
            train_committee_config=export_request.train_committee_config,
            training_stage=export_request.training_stage,
            teacher_backend=context.get("teacher_backend", context.get("mace_backend")),
            valid_fraction=export_request.valid_fraction,
            element_order=export_request.element_order,
        )
    if request is not None:
        raise TypeError(f"Unsupported teacher labeling request: {type(request)!r}")
    if context is not None:
        raise TypeError(f"Unsupported teacher labeling context: {type(context)!r}")
    raise ValueError("target-mode labeling requires an explicit TeacherLabelingRequest")


def run_label(cfg, paths, round_id, request=None, context=None):
    print("\nStarting LABEL step")
    request = _coerce_teacher_labeling_request(paths, round_id, request=request, context=context)

    config_path = Path(cfg["config"]).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Label config not found: {config_path}")

    label_cfg = read_json(config_path)
    train_ratio = resolve_label_train_ratio(label_cfg, request.student_export)
    element_order = resolve_label_element_order(label_cfg, request.student_export)
    write_deepmd_dataset = True
    train_committee_config = request.student_export.train_committee_config
    if train_committee_config:
        config_path, resolved_cfg, backend = _load_resolved_student_train_config(train_committee_config)
        if resolved_cfg is None or backend is None:
            raise RuntimeError(f"Unable to load student train config: {train_committee_config}")
        del config_path
        write_deepmd_dataset = bool(backend.export_contract(resolved_cfg).writes_deepmd_dataset)

    result = run_teacher_labeling(
        request=request,
        label_cfg=label_cfg,
        paths=paths,
        round_id=round_id,
        element_order=element_order,
        train_ratio=train_ratio,
        write_deepmd_dataset=write_deepmd_dataset,
        backend_log_dir=(context or {}).get("backend_log_dir_override") if isinstance(context, dict) else None,
        context=context if isinstance(context, dict) else None,
        routing_policy=TeacherRoutingPolicy(
            teacher_trust_lo=float(label_cfg.get("teacher_trust_lo", label_cfg.get("candidate_sigma_threshold", 0.10))),
            teacher_trust_hi=(
                float(label_cfg["teacher_trust_hi"])
                if label_cfg.get("teacher_trust_hi") is not None
                else None
            ),
            source="teacher_labeling",
        ),
    )

    print("\nLABEL step finished successfully")
    return result.summary
