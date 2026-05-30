from __future__ import annotations

from pathlib import Path

from workflow_backends.teacher import get_teacher_backend
from workflow_backends.teacher.base import TeacherCommitteeFinetuneRequest
from workflow_core.structures.datasets import merge_xyz_files


def _print_training_sources(training_input: dict) -> None:
    base_dataset_xyz_files = [Path(path).resolve() for path in training_input.get("base_dataset_xyz_files", [])]
    first_principles_xyz_files = [
        Path(path).resolve() for path in training_input.get("workflow_produced_first_principles_xyz_files", [])
    ]

    if base_dataset_xyz_files:
        print("\nBase teacher finetune datasets:")
        for file_path in base_dataset_xyz_files:
            print(" ", file_path)

    if first_principles_xyz_files:
        print("\nPrior workflow-produced first-principles training data:")
        for file_path in first_principles_xyz_files:
            print(" ", file_path)


def run_finetune(cfg, paths, r, context=None):
    context = dict(context or {})
    teacher_backend_name = str(
        cfg.get("teacher_backend")
        or context.get("teacher_backend")
        or cfg.get("mace_backend")
        or context.get("mace_backend")
        or "mace"
    )
    backend = get_teacher_backend(teacher_backend_name)
    print(f"\nPreparing {backend.backend_name} teacher committee finetune dataset")

    config_path = Path(
        cfg.get(
            "config",
            "configs/pipelines/committee_cycle/teacher_finetune/mace/committee_finetune.json",
        )
    ).resolve()
    if "train_config_snapshot" in context:
        train_cfg = dict(context["train_config_snapshot"])
    else:
        train_cfg = backend.load_committee_finetune_config(config_path)

    training_input = dict(context.get("training_input", {}) or {})
    all_xyz = [Path(path).resolve() for path in training_input.get("all_xyz_files", [])]
    _print_training_sources(training_input)

    if len(all_xyz) == 0:
        raise RuntimeError(
            f"No dataset for {backend.backend_name} finetune teacher committee finetune\n"
            "Please put xyz into:\n"
            "user_assets/teacher_finetune"
        )

    artifact_root = paths.teacher_committee_finetune_artifacts_dir(r)
    merged = Path(context.get("merged_xyz_override", artifact_root / "input" / "merged.xyz")).resolve()
    merge_xyz_files(all_xyz, merged)

    print("\nMerged dataset:")
    print(merged)

    work_root = Path(context.get("work_root_override", artifact_root)).resolve()
    runtime = Path(context.get("runtime_json_override", artifact_root / "teacher_committee_finetune_runtime.json")).resolve()
    default_backend_log_dir = paths.teacher_committee_finetune_backend_logs_dir(r)
    backend_log_dir = Path(
        context.get(
            "backend_log_dir_override",
            context.get("log_dir_override", default_backend_log_dir),
        )
    ).resolve()

    request_context = dict(context)
    request_context.setdefault("teacher_backend", teacher_backend_name)
    request = TeacherCommitteeFinetuneRequest(
        train_cfg=dict(train_cfg),
        config_path=config_path,
        paths=paths,
        teacher_round=int(r),
        merged_xyz=merged,
        work_root=work_root,
        runtime_json=runtime,
        backend_log_dir=backend_log_dir,
        reuse_existing_committee=bool(context.get("reuse_existing_committee", False)),
        context=request_context,
    )
    return backend.finetune_committee(request)
