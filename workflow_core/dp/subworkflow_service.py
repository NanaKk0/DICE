from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_artifacts import dp_round_status as _round_status
from workflow_core.dft.screening import resolve_dft_screening_policy
from workflow_core.dp import round_finalize as _round_finalize
from workflow_core.dp import round_source as _round_source
from workflow_core.dp import round_steps as _round_steps
from workflow_core.dp.config import resolve_dp_active_learning_config
from workflow_core.dp.round_metrics import (
    _dp_data_policy,
    _metrics_attempt,
    _record_dp_training_metrics,
    _record_labeling_metrics,
    _record_sampling_metrics,
)
from workflow_core.dp.sampling_config import resolve_dp_sampling_config
from workflow_core.labeling.task import run_label
from workflow_core.common.io import read_json


def _count_extxyz_frames(path: str | Path | None) -> int | None:
    return _round_status.count_extxyz_frames(path)


def load_dp_active_learning_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = resolve_dp_sampling_config(resolve_dp_active_learning_config(read_json(config_path)))

    payload.setdefault("max_dp_rounds", 2)
    payload.setdefault("stop_ratio_threshold", 0.1)
    payload.setdefault("resume", True)
    payload.setdefault("bootstrap_data", {"include_rgps_in_initial_dataset": False, "physical_dataset_always_enabled": True})
    payload.setdefault(
        "incremental_data",
        {"use_incremental_data_only": True, "train_on_current_round_increment_only": True},
    )
    payload.setdefault(
        "training_policy",
        {
            "round0_stage": "bootstrap",
            "later_round_stage": "finetune",
        },
    )
    payload.setdefault(
        "seed_pool_policy",
        {
            "enabled": True,
            "initialize_at_subworkflow_start": True,
            "max_pool_size_per_system": 2000,
            "refresh_after_training": True,
        },
    )
    payload.setdefault(
        "sampling_seed_policy",
        {
            "driver_policy": "round_robin",
            "md": {
                "timestep_fs": 0.5,
                "eval_every": 20,
                "log_every": 10,
                "seed": 20260319,
                "friction": 0.05,
                "force_temp": True,
                "zero_rotation": False,
            },
        },
    )
    payload.setdefault(
        "sampling_schedule",
        {
            "mode": "long_first_then_short",
            "long_round_selector": "first_sampling_round_only",
            "long_profile_name": "long_exploration",
            "short_profile_name": "short_refinement",
        },
    )
    payload.setdefault(
        "sampling_profiles",
        {
            "long_exploration": {
                "seed_selection": {"n_seed_per_system": 8},
                "sampling_seed_policy": dict(payload.get("sampling_seed_policy", {}) or {}),
                "candidate_train_selection": {"enabled": True, "max_train_candidates_per_system": 200},
                "seed_pool": {"refresh_mode": "merge_old_pool_with_current_candidate_failed"},
            },
            "short_refinement": {
                "seed_selection": {"n_seed_per_system": 8},
                "sampling_seed_policy": dict(payload.get("sampling_seed_policy", {}) or {}),
                "candidate_train_selection": {"enabled": True, "max_train_candidates_per_system": 200},
                "seed_pool": {"refresh_mode": "rescore_existing_pool_only"},
            },
        },
    )
    payload.setdefault("train_dp_committee_config", "configs/dp/student_committee.json")
    payload.setdefault("label_config", "configs/label/label.json")
    payload.setdefault("bootstrap_committee_summary", None)
    payload.setdefault("skip_round0_label", False)
    payload["dft_screening_policy"] = resolve_dft_screening_policy(payload)
    return payload


def _load_status(status_path: Path, outer_round: int, mace_teacher_dir: Path) -> dict[str, Any]:
    return _round_status.load_status(status_path, outer_round, mace_teacher_dir)


def _write_status(status_path: Path, payload: dict[str, Any]) -> Path:
    return _round_status.write_status(status_path, payload)


def _base_round_manifest(outer_round: int, dp_round: int) -> dict[str, Any]:
    return _round_status.base_round_manifest(outer_round, dp_round)


def _manifest_step(manifest: dict[str, Any], step_key: str) -> dict[str, Any]:
    return _round_status.manifest_step(manifest, step_key)


def _step_completed(manifest: dict[str, Any], step_key: str) -> bool:
    return _round_status.step_completed(manifest, step_key)


def _load_completed_sampling_summary_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    return _round_status.load_completed_sampling_summary_from_manifest(manifest, paths, outer_round, dp_round)


def _load_completed_label_info_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    return _round_status.load_completed_label_info_from_manifest(manifest, paths, outer_round, dp_round)


def _load_completed_dft_screening_info_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    return _round_status.load_completed_dft_screening_info_from_manifest(manifest, paths, outer_round, dp_round)


def _load_completed_seed_pool_output_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> Path | None:
    return _round_status.load_completed_seed_pool_output_from_manifest(manifest, paths, outer_round, dp_round)


def _load_completed_seed_update_summary_from_manifest(manifest: dict[str, Any], paths, outer_round: int, dp_round: int) -> dict[str, Any] | None:
    return _round_status.load_completed_seed_update_summary_from_manifest(manifest, paths, outer_round, dp_round)


def _rebuild_status_from_manifests(paths, outer_round: int, cfg: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    return _round_status.rebuild_status_from_manifests(paths, outer_round, cfg, status)


def _sync_screened_dft_pool_from_manifests(paths, outer_round: int, cfg: dict[str, Any]) -> Path | None:
    return _round_status.sync_screened_dft_pool_from_manifests(paths, outer_round, cfg)


def _resolve_previous_outer_round_committee_root(paths, outer_round: int) -> Path | None:
    return _round_source.resolve_previous_outer_round_committee_root(paths, outer_round)


def _resolve_round0_source_mode(outer_round: int) -> str:
    return _round_source.resolve_round0_source_mode(outer_round)


def _resolve_previous_outer_all_dft_candidate_xyzs(paths, outer_round: int) -> list[Path]:
    return _round_source.resolve_previous_outer_all_dft_candidate_xyzs(paths, outer_round)


def _build_round0_bootstrap_source(paths, outer_round: int, cfg: dict[str, Any]) -> Path:
    return _round_source.build_round0_bootstrap_source(paths, outer_round, cfg)


def _build_cross_outer_round0_source(paths, outer_round: int, cfg: dict[str, Any], status: dict[str, Any], resume: bool, metrics_logger=None) -> Path:
    return _round_source.build_cross_outer_round0_source(
        paths,
        outer_round,
        cfg,
        status,
        resume,
        metrics_logger=metrics_logger,
    )


def _snapshot_final_seed_pool(paths, outer_round: int) -> dict[str, str] | None:
    return _round_source.snapshot_final_seed_pool(paths, outer_round)


_training_dataset_roots_for_round = _round_steps._training_dataset_roots_for_round
_cumulative_train_xyz_upto = _round_steps._cumulative_train_xyz_upto
_ensure_round0_labeled = _round_steps._ensure_round0_labeled
_train_dp_round = _round_steps._train_dp_round
_resolve_sampling_profile_name = _round_steps._resolve_sampling_profile_name
_resolve_sampling_profile = _round_steps._resolve_sampling_profile
_is_short_sampling_profile = _round_steps._is_short_sampling_profile
_run_seed_pool = _round_steps._run_seed_pool
_run_sampling = _round_steps._run_sampling
_run_label_for_sampling = _round_steps._run_label_for_sampling
_resolve_dft_screening_stop_reason = _round_finalize._resolve_dft_screening_stop_reason
_finalize_round0_artifacts = _round_finalize._finalize_round0_artifacts


def _sync_round_modules() -> None:
    _round_steps.run_label = run_label
    _round_steps._write_status = _write_status
    _round_finalize.load_dp_active_learning_config = load_dp_active_learning_config
    _round_finalize._load_status = _load_status
    _round_finalize._write_status = _write_status
    _round_finalize._rebuild_status_from_manifests = _rebuild_status_from_manifests
    _round_finalize._sync_screened_dft_pool_from_manifests = _sync_screened_dft_pool_from_manifests
    _round_finalize._dp_data_policy = _dp_data_policy
    _round_finalize._ensure_round0_labeled = _ensure_round0_labeled
    _round_finalize._train_dp_round = _train_dp_round
    _round_finalize._resolve_sampling_profile = _resolve_sampling_profile
    _round_finalize._is_short_sampling_profile = _is_short_sampling_profile
    _round_finalize._run_seed_pool = _run_seed_pool
    _round_finalize._run_sampling = _run_sampling
    _round_finalize._run_label_for_sampling = _run_label_for_sampling
    _round_finalize._load_completed_seed_pool_output_from_manifest = _load_completed_seed_pool_output_from_manifest
    _round_finalize._load_completed_sampling_summary_from_manifest = _load_completed_sampling_summary_from_manifest
    _round_finalize._load_completed_label_info_from_manifest = _load_completed_label_info_from_manifest
    _round_finalize._load_completed_dft_screening_info_from_manifest = _load_completed_dft_screening_info_from_manifest
    _round_finalize._load_completed_seed_update_summary_from_manifest = _load_completed_seed_update_summary_from_manifest
    _round_finalize._training_dataset_roots_for_round = _training_dataset_roots_for_round
    _round_finalize._snapshot_final_seed_pool = _snapshot_final_seed_pool
    _round_finalize._base_round_manifest = _base_round_manifest
    _round_finalize._metrics_attempt = _metrics_attempt


def run_dp_active_learning_subworkflow(cfg_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    _sync_round_modules()
    return _round_finalize.run_dp_active_learning_subworkflow(cfg_path, paths, outer_round, metrics_logger=metrics_logger)
