from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.training_data_assets import validate_path_only_xyz_list


def resolve_teacher_finetune_inputs(paths, teacher_round: int, resolved_cfg: dict[str, Any]) -> dict[str, Any]:
    teacher_cfg = dict(resolved_cfg.get("teacher_committee_finetune", {}) or {})
    train_cfg = dict(resolved_cfg.get("teacher_committee_finetune_train_config_snapshot", {}) or {})

    training_data = dict(train_cfg.get("training_data", teacher_cfg.get("training_data", {})) or {})
    base_payload = training_data.get("base_datasets", [])

    declared_base_dataset_xyz_files: list[str] = []
    physical_xyz_files: list[str] = []
    rgps_xyz_files: list[str] = []
    teacher_asset_root = getattr(paths, "user_teacher_finetune", paths.user_assets / "teacher_finetune")
    declared_base_dataset_xyz_files = [
        str(path.resolve())
        for path in validate_path_only_xyz_list(
            base_payload,
            field_name="training_data.base_datasets",
            bare_filename_root=teacher_asset_root,
            allow_empty=True,
        )
    ]
    include_physical = False
    include_rgps = False

    from workflow_core.first_principles.external_import import resolve_historical_labeled_dft_sources

    historical_labeled_dft_sources = resolve_historical_labeled_dft_sources(
        paths,
        teacher_round,
    )
    workflow_produced_first_principles_sources = historical_labeled_dft_sources
    workflow_produced_first_principles_xyz_files = [str(item["path"]) for item in workflow_produced_first_principles_sources]
    workflow_produced_first_principles_rounds = [int(item["round"]) for item in workflow_produced_first_principles_sources]
    historical_labeled_dft_xyz_files = list(workflow_produced_first_principles_xyz_files)
    historical_labeled_dft_rounds = list(workflow_produced_first_principles_rounds)

    all_xyz_files = (
        declared_base_dataset_xyz_files
        + physical_xyz_files
        + rgps_xyz_files
        + workflow_produced_first_principles_xyz_files
    )

    return {
        "include_physical": include_physical,
        "include_rgps": include_rgps,
        "base_dataset_xyz_files": declared_base_dataset_xyz_files,
        "physical_xyz_files": physical_xyz_files,
        "rgps_xyz_files": rgps_xyz_files,
        "historical_labeled_dft_xyz_files": historical_labeled_dft_xyz_files,
        "historical_labeled_dft_rounds": historical_labeled_dft_rounds,
        "historical_labeled_dft_sources": historical_labeled_dft_sources,
        "workflow_produced_first_principles_xyz_files": workflow_produced_first_principles_xyz_files,
        "workflow_produced_first_principles_rounds": workflow_produced_first_principles_rounds,
        "workflow_produced_first_principles_sources": workflow_produced_first_principles_sources,
        "all_xyz_files": all_xyz_files,
    }


