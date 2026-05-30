from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from workflow_core.labeling.request import build_teacher_labeling_request
from workflow_core.student_pipeline.stages.harness import StudentStageRunnerContext, run_student_stage_with_harness
from workflow_core.student_pipeline.stages.publish import (
    build_backend_log_publications,
    build_stage_publish_plan,
    prepare_stage_result_publications,
    stage_payload_without_commit_fields,
)
from workflow_core.student_pipeline.stages.base import resolve_stage_config
from workflow_core.structures.formats import concat_extxyz_files
from workflow_runtime.environment import context_with_execution_context, teacher_backend_role
from workflow_core.training_data_assets import validate_path_only_xyz_list


def run_relabel_stage(
    runtime_state: dict[str, Any],
    *,
    student_round: int,
    stage_run_index: int,
    stage_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def execute(context: StudentStageRunnerContext) -> dict[str, Any]:
        executor = stage_executor or (
            lambda stage_context: _default_relabel_stage_executor(runtime_state, stage_context, context.artifacts)
        )
        return dict(executor(context.execution_context) or {})

    def publish_plan(context: StudentStageRunnerContext, stage_result: dict[str, Any]):
        return build_stage_publish_plan(
            student_round=student_round,
            stage_run_index=stage_run_index,
            artifact_publications=prepare_stage_result_publications(
                "relabel",
                staging_dir=context.staging_dir,
                target_root=context.stage_context["stage_dir"],
                stage_result=stage_result,
                role_source_fields={
                    "student_training_labeled": "teacher_train_output_path",
                    "student_training_dataset": "student_training_dataset_path",
                    "teacher_dft_candidate": "teacher_dft_candidate_output_path",
                },
                extxyz_roles=("student_training_labeled", "teacher_dft_candidate"),
            ),
            backend_log_publications=build_backend_log_publications(
                context.staging_dir,
                Path(context.stage_record["log_paths"]["stdout_path"]).resolve().parent / "backend_logs",
            ),
            stage_payload=stage_payload_without_commit_fields(stage_result.get("stage_payload", {})),
        )

    return run_student_stage_with_harness(
        runtime_state,
        student_round=student_round,
        stage_run_index=stage_run_index,
        expected_stage_type="relabel",
        execute=execute,
        build_publish_plan=publish_plan,
    )

def _default_relabel_stage_executor(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
    artifacts,
) -> dict[str, Any]:
    run_label_impl = _load_run_label()
    paths = runtime_state.get("paths")
    if paths is None:
        raise ValueError("runtime_state is missing paths for default relabel execution")
    stage_cfg = resolve_stage_config(stage_context, "relabel")
    if isinstance(dict(runtime_state.get("cfg", {}) or {}).get("environment_profile"), dict):
        stage_cfg.setdefault("environment_profile", dict(runtime_state["cfg"]["environment_profile"]))
    inherited_teacher_backend = _inherited_teacher_backend(runtime_state)
    inherited_teacher_num = _inherited_teacher_num(runtime_state)
    label_config_path = Path(stage_context["config"]).resolve()

    input_xyz = Path(artifacts.inputs.primary_input_xyz).resolve()
    stage_results_dir = Path(stage_context["stage_results_dir"]).resolve()
    label_dir = stage_results_dir / "label"
    active_learning_dir = stage_results_dir / "active_learning"
    merged_xyz = stage_results_dir / "train_merged.xyz"
    bootstrap_payload = _resolve_initial_student_input_payload(runtime_state, stage_context, input_xyz)
    failed_backflow_summary = None
    failed_backflow_cfg = dict(stage_cfg.get("failed_backflow", {}) or {})
    failed_backflow_enabled = bool(failed_backflow_cfg.get("enabled", False))
    failed_backflow_input = artifacts.inputs.failed_backflow_input_xyz if failed_backflow_enabled else None
    candidate_summary: dict[str, Any]
    if _extxyz_has_payload(input_xyz):
        candidate_result = _run_relabel_labeling(
            run_label_impl,
            paths=paths,
            teacher_round=int(stage_context["teacher_round"]),
            label_config_path=label_config_path,
            input_xyz=input_xyz,
            label_dir=label_dir,
            active_learning_dir=active_learning_dir,
            merged_xyz=merged_xyz,
            stage_cfg=stage_cfg,
            inherited_teacher_backend=inherited_teacher_backend,
            inherited_teacher_num=inherited_teacher_num,
            backend_log_dir=stage_results_dir / "backend_logs" / "candidate",
        )
        candidate_summary = dict(candidate_result["summary"] or {})
    elif failed_backflow_enabled and failed_backflow_input is not None and _extxyz_has_payload(failed_backflow_input):
        _ensure_empty_relabel_outputs(label_dir, active_learning_dir)
        candidate_summary = _empty_label_summary()
    else:
        candidate_result = _run_relabel_labeling(
            run_label_impl,
            paths=paths,
            teacher_round=int(stage_context["teacher_round"]),
            label_config_path=label_config_path,
            input_xyz=input_xyz,
            label_dir=label_dir,
            active_learning_dir=active_learning_dir,
            merged_xyz=merged_xyz,
            stage_cfg=stage_cfg,
            inherited_teacher_backend=inherited_teacher_backend,
            inherited_teacher_num=inherited_teacher_num,
            backend_log_dir=stage_results_dir / "backend_logs" / "candidate",
        )
        candidate_summary = dict(candidate_result["summary"] or {})
    if failed_backflow_enabled and failed_backflow_input is not None and _extxyz_has_payload(failed_backflow_input):
        failed_label_dir = stage_results_dir / "label_failed_backflow"
        failed_active_learning_dir = stage_results_dir / "active_learning_failed_backflow"
        failed_merged_xyz = stage_results_dir / "train_failed_backflow_merged.xyz"
        failed_result = _run_relabel_labeling(
            run_label_impl,
            paths=paths,
            teacher_round=int(stage_context["teacher_round"]),
            label_config_path=label_config_path,
            input_xyz=failed_backflow_input.resolve(),
            label_dir=failed_label_dir,
            active_learning_dir=failed_active_learning_dir,
            merged_xyz=failed_merged_xyz,
            stage_cfg=stage_cfg,
            inherited_teacher_backend=inherited_teacher_backend,
            inherited_teacher_num=inherited_teacher_num,
            backend_log_dir=stage_results_dir / "backend_logs" / "failed_backflow",
        )
        _append_xyz_file_text(label_dir / "train.xyz", failed_result["teacher_train_output_path"])
        _append_xyz_file_text(active_learning_dir / "dft.xyz", failed_result["teacher_dft_candidate_output_path"])
        failed_backflow_summary = dict(failed_result["summary"] or {})
        candidate_summary = _merge_label_summaries(candidate_summary, failed_backflow_summary)

    return {
        "teacher_train_output_path": label_dir / "train.xyz",
        "student_training_dataset_path": label_dir,
        "teacher_dft_candidate_output_path": active_learning_dir / "dft.xyz",
        "stage_payload": {
            **dict(candidate_summary),
            **bootstrap_payload,
            "failed_backflow_enabled": failed_backflow_enabled,
            "failed_backflow_summary": failed_backflow_summary,
        },
    }

def _run_relabel_labeling(
    run_label_impl,
    *,
    paths,
    teacher_round: int,
    label_config_path: Path,
    input_xyz: Path,
    label_dir: Path,
    active_learning_dir: Path,
    merged_xyz: Path,
    stage_cfg: dict[str, Any],
    inherited_teacher_backend: str | None = None,
    inherited_teacher_num: int | None = None,
    backend_log_dir: Path | None = None,
) -> dict[str, Any]:
    teacher_backend = _resolve_relabel_teacher_backend(stage_cfg, inherited_teacher_backend)
    request = build_teacher_labeling_request(
        structure_file=input_xyz,
        teacher_round=int(teacher_round),
        teacher_dir=paths.teacher_committee_finetune_artifacts_dir(int(teacher_round)).resolve(),
        labeled_output_dir=label_dir.resolve(),
        active_learning_output_dir=active_learning_dir.resolve(),
        include_user_student_train=bool(stage_cfg.get("include_user_student_train", False)),
        merged_xyz=merged_xyz.resolve(),
        append_active_learning_xyz=None,
        train_committee_config=stage_cfg.get("train_committee_config"),
        training_stage=stage_cfg.get("training_stage"),
        teacher_backend=teacher_backend,
        teacher_num=inherited_teacher_num,
        valid_fraction=stage_cfg.get("valid_fraction"),
        element_order=stage_cfg.get("element_order"),
    )
    summary_kwargs = {}
    if backend_log_dir is not None:
        summary_kwargs["context"] = {"backend_log_dir_override": str(backend_log_dir.resolve())}
    bound_context = context_with_execution_context(
        stage_cfg,
        role=teacher_backend_role(teacher_backend, "relabel"),
        context=summary_kwargs.get("context"),
    )
    if bound_context:
        summary_kwargs["context"] = bound_context
    try:
        summary = run_label_impl(
            {"config": str(label_config_path.resolve())},
            paths,
            int(teacher_round),
            request=request,
            **summary_kwargs,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'context'" not in str(exc) or "context" not in summary_kwargs:
            raise
        summary = run_label_impl(
            {"config": str(label_config_path.resolve())},
            paths,
            int(teacher_round),
            request=request,
        )
    return {
        "summary": dict(summary or {}),
        "teacher_train_output_path": label_dir / "train.xyz",
        "student_training_dataset_path": label_dir,
        "teacher_dft_candidate_output_path": active_learning_dir / "dft.xyz",
    }


def _inherited_teacher_backend(runtime_state: dict[str, Any]) -> str:
    cfg = dict(runtime_state.get("cfg", {}) or {})
    teacher_finetune = dict(cfg.get("teacher_committee_finetune", {}) or {})
    return str(teacher_finetune.get("teacher_backend", "mace") or "mace").strip().lower()


def _inherited_teacher_num(runtime_state: dict[str, Any]) -> int | None:
    cfg = dict(runtime_state.get("cfg", {}) or {})
    train_cfg = dict(cfg.get("teacher_committee_finetune_train_config_snapshot", {}) or {})
    teacher_num = train_cfg.get("teacher_num")
    if teacher_num is None:
        return None
    normalized = int(teacher_num)
    return normalized if normalized > 0 else None


def _resolve_relabel_teacher_backend(stage_cfg: dict[str, Any], inherited_teacher_backend: str | None) -> str:
    inherited = str(inherited_teacher_backend or "mace").strip().lower()
    declared = stage_cfg.get("teacher_backend")
    if declared is not None:
        normalized_declared = str(declared).strip().lower()
        if normalized_declared != inherited:
            raise ValueError(
                "relabel stage teacher_backend conflicts with inherited teacher_committee_finetune.teacher_backend: "
                f"{normalized_declared!r} != {inherited!r}"
            )
    return inherited


def _append_xyz_file_text(target_path: str | Path, source_path: str | Path) -> Path:
    target = Path(target_path)
    source = Path(source_path)
    if not source.exists() or source.stat().st_size == 0:
        if not target.exists():
            target.write_text("", encoding="utf-8")
        return target.resolve()
    sources = [path for path in (target, source) if path.exists()]
    concat_extxyz_files(sources, target, allow_empty_sources=True)
    return target.resolve()


def _extxyz_has_payload(path: str | Path | None) -> bool:
    if path is None:
        return False
    source = Path(path)
    return source.exists() and source.is_file() and source.stat().st_size > 0


def _ensure_empty_relabel_outputs(label_dir: Path, active_learning_dir: Path) -> None:
    (label_dir / "train.xyz").parent.mkdir(parents=True, exist_ok=True)
    (active_learning_dir / "dft.xyz").parent.mkdir(parents=True, exist_ok=True)
    (label_dir / "train.xyz").write_text("", encoding="utf-8")
    (active_learning_dir / "dft.xyz").write_text("", encoding="utf-8")


def _empty_label_summary() -> dict[str, Any]:
    return {"label_counts": {"input": 0, "train": 0, "dft": 0, "discard": 0}}


def _merge_label_summaries(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(primary)
    for key in ("input", "train", "dft", "discard"):
        merged[key] = int(primary.get(key, 0) or 0) + int(secondary.get(key, 0) or 0)
    for key in ("label_counts", "label_system_stats"):
        if key in merged or key in secondary:
            merged[key] = _merge_nested_values(merged.get(key), secondary.get(key))
    return merged


def _merge_nested_values(primary: Any, secondary: Any) -> Any:
    if isinstance(primary, dict) and isinstance(secondary, dict):
        merged = deepcopy(primary)
        for key, value in secondary.items():
            if key in merged:
                merged[key] = _merge_nested_values(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    if isinstance(primary, (int, float)) and isinstance(secondary, (int, float)):
        return primary + secondary
    if secondary is None:
        return deepcopy(primary)
    return deepcopy(secondary)


def _load_run_label():
    from workflow_core.labeling.task import run_label as _impl

    return _impl


def _resolve_initial_student_input_payload(
    runtime_state: dict[str, Any],
    stage_context: dict[str, Any],
    merged_input_xyz: Path,
) -> dict[str, Any]:
    if int(stage_context["teacher_round"]) != 0 or int(stage_context["stage_run_index"]) != 1:
        return {}
    cfg = dict(runtime_state.get("cfg", {}) or {})
    paths = runtime_state.get("paths")
    if paths is None:
        return {}
    initial_inputs = [
        str(path.resolve())
        for path in validate_path_only_xyz_list(
            cfg.get("initial_student_inputs"),
            field_name="initial_student_inputs",
            bare_filename_root=getattr(paths, "user_student_initial", paths.user_assets / "student_initial"),
        )
    ]
    if not initial_inputs:
        return {}
    return {
        "bootstrap_input_xyz": str(Path(merged_input_xyz).resolve()),
        "bootstrap_input_sources": initial_inputs,
    }
