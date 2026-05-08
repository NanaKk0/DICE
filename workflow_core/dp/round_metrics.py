from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_artifacts.dp_round_status import count_extxyz_frames
from workflow_core.common.io import read_json


def _metrics_attempt(manifest: dict[str, Any], step_key: str) -> int:
    step_payload = manifest.get("steps", {}).get(step_key, {})
    return int(step_payload.get("metrics_attempt", 0)) + 1


def _record_sampling_metrics(metrics_logger, outer_round: int, step_id: str, sampling_dir: Path) -> None:
    if metrics_logger is None:
        return
    jobs_path = sampling_dir / "jobs.json"
    if not jobs_path.exists():
        return
    try:
        jobs = read_json(jobs_path)
    except Exception:
        return
    for job in jobs:
        runtime = job.get("runtime", {}) or {}
        metrics_logger.log_sampling_metric(
            {
                "round_idx": outer_round,
                "step_id": step_id,
                "sampler_type": "dp_committee_md",
                "trajectory_id": job.get("job_id"),
                "seed_id": job.get("structure_index"),
                "start_time": runtime.get("started_at"),
                "end_time": runtime.get("finished_at"),
                "wall_time_sec": runtime.get("wall_time_sec"),
                "status": runtime.get("status"),
                "temperature": job.get("temperature"),
                "time_step_fs": job.get("timestep_fs"),
                "n_steps_requested": job.get("steps"),
                "n_steps_completed": runtime.get("n_steps_completed"),
                "n_frames_generated": runtime.get("n_frames_generated"),
                "n_frames_kept": runtime.get("n_saved_frames"),
                "n_frames_rejected": runtime.get("n_frames_rejected"),
                "termination_reason": runtime.get("termination_reason"),
                "error_message": runtime.get("error_message"),
            }
        )


def _record_labeling_metrics(metrics_logger, outer_round: int, step_id: str, label_dir: Path, active_learning_dir: Path, structure_file: Path) -> dict[str, int]:
    train_count = count_extxyz_frames(label_dir / "train.xyz") or 0
    dft_count = count_extxyz_frames(active_learning_dir / "dft.xyz") or 0
    discard_count = count_extxyz_frames(label_dir / "discard.xyz") or 0
    input_count = count_extxyz_frames(structure_file) or 0
    success_count = train_count + dft_count + discard_count
    if metrics_logger is not None:
        metrics_logger.log_labeling_metric(
            {
                "round_idx": outer_round,
                "step_id": step_id,
                "labeler_type": "mace_committee",
                "batch_id": step_id,
                "start_time": "",
                "end_time": "",
                "wall_time_sec": "",
                "status": "SUCCESS",
                "input_structure_count": input_count,
                "success_structure_count": success_count,
                "failed_structure_count": max(0, input_count - success_count),
                "avg_time_per_structure_sec": "",
                "output_path": str(label_dir.resolve()),
            }
        )
    return {
        "input": input_count,
        "train": train_count,
        "dft": dft_count,
        "discard": discard_count,
    }


def _record_dp_training_metrics(metrics_logger, outer_round: int, step_id: str, summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        payload = read_json(summary_path)
    except Exception:
        return {}
    members = payload.get("members", [])
    best_rmse_e = None
    best_rmse_f = None
    for member in members:
        if member.get("status") != "success":
            continue
        rmse_e = member.get("final_val_rmse_e")
        rmse_f = member.get("final_val_rmse_f")
        if rmse_e is not None:
            best_rmse_e = rmse_e if best_rmse_e is None else min(best_rmse_e, rmse_e)
        if rmse_f is not None:
            best_rmse_f = rmse_f if best_rmse_f is None else min(best_rmse_f, rmse_f)
        if metrics_logger is not None:
            metrics_logger.log_training_metric(
                {
                    "round_idx": outer_round,
                    "step_id": step_id,
                    "model_family": "DeepMD",
                    "phase": "final",
                    "epoch": "",
                    "global_step": "",
                    "timestamp": "",
                    "train_loss": "",
                    "valid_loss": "",
                    "rmse_e_mev_per_atom": rmse_e,
                    "rmse_f_mev_per_a": rmse_f,
                    "rmse_stress_mev_per_a3": "",
                    "learning_rate": "",
                    "epoch_time_sec": "",
                }
            )
    return {
        "best_rmse_e": best_rmse_e,
        "best_rmse_f": best_rmse_f,
    }


def _dp_data_policy(paths, outer_round: int, cfg: dict[str, Any]) -> dict[str, Any]:
    bootstrap_data = dict(cfg.get("bootstrap_data", {}) or {})
    incremental_data = dict(cfg.get("incremental_data", {}) or {})
    training_policy = dict(cfg.get("training_policy", {}) or {})
    return {
        "bootstrap_data": bootstrap_data,
        "incremental_data": incremental_data,
        "training_policy": training_policy,
        "seed_pool_policy": dict(cfg.get("seed_pool_policy", {}) or {}),
        "sampling_seed_policy": dict(cfg.get("sampling_seed_policy", {}) or {}),
        "sampling_schedule": dict(cfg.get("sampling_schedule", {}) or {}),
        "sampling_profiles": dict(cfg.get("sampling_profiles", {}) or {}),
        "dft_screening_policy": dict(cfg.get("dft_screening_policy", {}) or {}),
        "deprecated_config_fields": list(cfg.get("deprecated_config_fields", [])),
    }
