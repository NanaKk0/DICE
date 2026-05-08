from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json, write_json
from workflow_core.dp import subworkflow_service as _impl

# Re-export selected helpers for compatibility with existing tools/smokes.
run_label = _impl.run_label
load_dp_active_learning_config = _impl.load_dp_active_learning_config
_count_extxyz_frames = _impl._count_extxyz_frames
_load_status = _impl._load_status
_write_status = _impl._write_status
_base_round_manifest = _impl._base_round_manifest
_manifest_step = _impl._manifest_step
_step_completed = _impl._step_completed
_dp_data_policy = _impl._dp_data_policy
_training_dataset_roots_for_round = _impl._training_dataset_roots_for_round
_cumulative_train_xyz_upto = _impl._cumulative_train_xyz_upto
_ensure_round0_labeled = _impl._ensure_round0_labeled
_train_dp_round = _impl._train_dp_round
_resolve_sampling_profile_name = _impl._resolve_sampling_profile_name
_resolve_sampling_profile = _impl._resolve_sampling_profile
_is_short_sampling_profile = _impl._is_short_sampling_profile
_run_seed_pool = _impl._run_seed_pool
_run_sampling = _impl._run_sampling
_run_label_for_sampling = _impl._run_label_for_sampling


def _sync_impl_bindings() -> None:
    _impl.run_label = run_label
    _impl.load_dp_active_learning_config = load_dp_active_learning_config
    _impl._count_extxyz_frames = _count_extxyz_frames
    _impl._load_status = _load_status
    _impl._write_status = _write_status
    _impl._base_round_manifest = _base_round_manifest
    _impl._manifest_step = _manifest_step
    _impl._step_completed = _step_completed
    _impl._dp_data_policy = _dp_data_policy
    _impl._training_dataset_roots_for_round = _training_dataset_roots_for_round
    _impl._cumulative_train_xyz_upto = _cumulative_train_xyz_upto
    _impl._ensure_round0_labeled = _ensure_round0_labeled
    _impl._train_dp_round = _train_dp_round
    _impl._resolve_sampling_profile_name = _resolve_sampling_profile_name
    _impl._resolve_sampling_profile = _resolve_sampling_profile
    _impl._is_short_sampling_profile = _is_short_sampling_profile
    _impl._run_seed_pool = _run_seed_pool
    _impl._run_sampling = _run_sampling
    _impl._run_label_for_sampling = _run_label_for_sampling


def run_dp_active_learning_subworkflow(cfg_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    _sync_impl_bindings()
    return _impl.run_dp_active_learning_subworkflow(cfg_path, paths, outer_round, metrics_logger=metrics_logger)
