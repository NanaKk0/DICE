from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.dft.external_import import resolve_historical_labeled_dft_sources
from workflow_core.structures.datasets import collect_xyz_files


def resolve_mace_finetune_inputs(paths, outer_round: int, resolved_cfg: dict[str, Any]) -> dict[str, Any]:
    mace_cfg = dict(resolved_cfg.get("mace_finetune", {}) or {})
    training_data = dict(mace_cfg.get("training_data", {}) or {})
    base_datasets = dict(training_data.get("base_datasets", {}) or {})

    include_physical = bool(base_datasets.get("include_physical", True))
    include_rgps = bool(base_datasets.get("include_rgps", False))
    physical_xyz_files: list[str] = []
    rgps_xyz_files: list[str] = []
    historical_labeled_dft_xyz_files: list[str] = []
    historical_labeled_dft_rounds: list[int] = []
    historical_labeled_dft_sources: list[dict[str, Any]] = []

    if include_physical:
        physical_xyz_files = [str(path.resolve()) for path in collect_xyz_files(paths.user_mace_finetune)]

    if include_rgps:
        rgps_xyz_files = [str(path.resolve()) for path in collect_xyz_files(paths.user_mace_train)]

    historical_labeled_dft_sources = resolve_historical_labeled_dft_sources(
        paths,
        outer_round,
    )
    historical_labeled_dft_xyz_files = [str(item["path"]) for item in historical_labeled_dft_sources]
    historical_labeled_dft_rounds = [int(item["round"]) for item in historical_labeled_dft_sources]

    all_xyz_files = physical_xyz_files + rgps_xyz_files + historical_labeled_dft_xyz_files

    return {
        "include_physical": include_physical,
        "include_rgps": include_rgps,
        "physical_xyz_files": physical_xyz_files,
        "rgps_xyz_files": rgps_xyz_files,
        "historical_labeled_dft_xyz_files": historical_labeled_dft_xyz_files,
        "historical_labeled_dft_rounds": historical_labeled_dft_rounds,
        "historical_labeled_dft_sources": historical_labeled_dft_sources,
        "all_xyz_files": all_xyz_files,
    }


