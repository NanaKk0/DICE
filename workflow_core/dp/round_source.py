from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from workflow_artifacts.dp_round_status import base_round_manifest, count_extxyz_frames
from workflow_core.common.io import ensure_dir, read_json, write_json
from workflow_core.structures.datasets import collect_xyz_files, merge_xyz_files


def resolve_previous_outer_round_committee_root(paths, outer_round: int) -> Path | None:
    if int(outer_round) <= 0:
        return None
    previous_status_path = paths.dp_subworkflow_status_json(outer_round - 1)
    if not previous_status_path.exists():
        return None
    try:
        previous_status = read_json(previous_status_path)
    except Exception:
        return None
    final_dp_model_path = previous_status.get("final_dp_model_path")
    if not final_dp_model_path:
        return None
    summary_path = Path(str(final_dp_model_path)).resolve()
    if not summary_path.exists():
        return None
    committee_root = summary_path.parent
    return committee_root if committee_root.exists() else None


def resolve_round0_source_mode(outer_round: int) -> str:
    if int(outer_round) <= 0:
        return "bootstrap"
    return "previous_outer_all_dft_candidates"


def resolve_previous_outer_all_dft_candidate_xyzs(paths, outer_round: int) -> list[Path]:
    if int(outer_round) <= 0:
        return []
    previous_status_path = paths.dp_subworkflow_status_json(outer_round - 1)
    max_dp_round = 0
    if previous_status_path.exists():
        try:
            previous_status = read_json(previous_status_path)
            max_dp_round = int(previous_status.get("current_dp_round", 0) or 0)
        except Exception:
            max_dp_round = 0
    files: list[Path] = []
    for dp_round in range(max_dp_round + 1):
        dft_xyz = paths.dp_subworkflow_round_active_learning_dir(outer_round - 1, dp_round) / "dft.xyz"
        if dft_xyz.exists() and dft_xyz.stat().st_size > 0:
            files.append(dft_xyz.resolve())
    return files


def build_round0_bootstrap_source(paths, outer_round: int, cfg: dict[str, Any]) -> Path:
    bootstrap_cfg = dict(cfg.get("bootstrap_data", {}) or {})
    include_base_rgps = bool(bootstrap_cfg.get("include_rgps_in_initial_dataset", False))
    bootstrap_inputs: list[Path] = []
    physical_xyz = [Path(path).resolve() for path in collect_xyz_files(paths.user_mace_finetune)]
    bootstrap_inputs.extend(physical_xyz)

    if include_base_rgps:
        rgps_xyz = collect_xyz_files(paths.user_mace_train)
        bootstrap_inputs.extend(rgps_xyz)

    if not bootstrap_inputs:
        raise RuntimeError(
            "No dataset available for DP round0 bootstrap. "
            "Please ensure physical xyz files are available in user_assets/mace_finetune, "
            "and optionally provide RGPS data in user_assets/mace_train when bootstrap RGPS is enabled."
        )

    merged_xyz = paths.committee_cycle_dp_bootstrap_merged_xyz(outer_round).resolve()
    merge_xyz_files(bootstrap_inputs, merged_xyz)
    return merged_xyz


def build_cross_outer_round0_source(paths, outer_round: int, cfg: dict[str, Any], status: dict[str, Any], resume: bool, metrics_logger=None) -> Path:
    dp_round = 0
    manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round)
    manifest = read_json(manifest_path) if manifest_path.exists() else base_round_manifest(outer_round, dp_round)
    merged_xyz = paths.dp_subworkflow_round_entry_merged_xyz(outer_round).resolve()
    source_step = (manifest.get("steps", {}) or {}).get("cross_outer_round0_source", {}) or {}
    if resume and bool(source_step.get("completed")) and merged_xyz.exists() and merged_xyz.stat().st_size > 0:
        return merged_xyz

    source_mode = resolve_round0_source_mode(outer_round)
    if source_mode != "previous_outer_all_dft_candidates":
        raise RuntimeError(
            f"Unsupported cross-outer round0 source_mode '{source_mode}' for outer_round={outer_round}"
        )
    previous_dft_candidates = resolve_previous_outer_all_dft_candidate_xyzs(paths, outer_round)
    merge_inputs: list[Path] = [path.resolve() for path in previous_dft_candidates]
    if not merge_inputs:
        raise RuntimeError(
            f"No previous outer-round dft candidate xyz files found for outer_round={outer_round - 1}"
        )

    merge_xyz_files(merge_inputs, merged_xyz)
    merged_count = count_extxyz_frames(merged_xyz) or 0
    manifest["steps"]["cross_outer_round0_source"] = {
        "completed": True,
        "previous_outer_round": int(outer_round - 1),
        "merged_xyz": str(merged_xyz.resolve()),
        "source_mode": source_mode,
        "previous_outer_all_dft_candidate_xyz_files": [str(path) for path in merge_inputs],
        "merged_count": int(merged_count),
    }
    write_json(manifest_path, manifest)
    return merged_xyz


def snapshot_final_seed_pool(paths, outer_round: int) -> dict[str, str] | None:
    seed_pool_xyz = paths.seed_pool_xyz().resolve()
    if not seed_pool_xyz.exists() or seed_pool_xyz.stat().st_size == 0:
        return None
    snapshot_xyz = paths.seed_pool_final_snapshot_xyz(outer_round).resolve()
    snapshot_meta = paths.seed_pool_final_snapshot_meta_json(outer_round).resolve()
    ensure_dir(snapshot_xyz.parent)
    shutil.copy2(seed_pool_xyz, snapshot_xyz)
    current_meta = paths.seed_pool_meta_json().resolve()
    if current_meta.exists():
        shutil.copy2(current_meta, snapshot_meta)
    payload = {
        "seed_pool_xyz": str(snapshot_xyz),
        "seed_pool_meta_json": str(snapshot_meta) if snapshot_meta.exists() else None,
    }
    return payload
