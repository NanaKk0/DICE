from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_artifacts.dp_round_status import base_round_manifest, count_extxyz_frames, step_completed
from workflow_core.common.io import ensure_dir, load_atoms_list, read_json, write_candidate_xyz, write_json
from workflow_core.dp import round_source as _round_source
from workflow_core.dp.round_metrics import (
    _dp_data_policy,
    _metrics_attempt,
    _record_dp_training_metrics,
    _record_labeling_metrics,
    _record_sampling_metrics,
)
from workflow_core.dp.sample_committee import run_dp_sample_committee
from workflow_core.dp.train_committee import run_dp_train_committee
from workflow_core.labeling.split_stats import summarize_label_split_by_system
from workflow_core.labeling.task import run_label
from workflow_core.seed_pool.manager import select_md_seeds_from_pool
from workflow_runtime.logging.runtime_logging import ensure_step_log_file

_write_status = write_json


def _training_dataset_roots_for_round(paths, outer_round: int, dp_round: int) -> list[str]:
    label_dir = paths.dp_subworkflow_round_label_dir(outer_round, dp_round)
    return [str(label_dir.resolve())] if label_dir.exists() else []


def _cumulative_train_xyz_upto(paths, outer_round: int, dp_round: int) -> list[str]:
    files: list[str] = []
    for history_round in range(dp_round + 1):
        train_xyz = paths.dp_subworkflow_round_train_xyz(outer_round, history_round)
        if train_xyz.exists():
            files.append(str(train_xyz.resolve()))
    return files


def _ensure_round0_labeled(paths, outer_round: int, cfg: dict[str, Any], status: dict[str, Any], resume: bool, metrics_logger=None) -> None:
    dp_round = 0
    manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round)
    manifest = read_json(manifest_path) if manifest_path.exists() else base_round_manifest(outer_round, dp_round)

    if bool(cfg.get("skip_round0_label", False)):
        return

    if resume and step_completed(manifest, "label"):
        label_dir = paths.dp_subworkflow_round_label_dir(outer_round, dp_round)
        active_learning_dir = paths.dp_subworkflow_round_active_learning_dir(outer_round, dp_round)
        if any(
            candidate.exists()
            for candidate in (
                label_dir / "train.xyz",
                label_dir / "discard.xyz",
                active_learning_dir / "dft.xyz",
            )
        ):
            return

    round0_source_mode = _round_source.resolve_round0_source_mode(outer_round)
    if round0_source_mode != "bootstrap":
        mace_train_merged = _round_source.build_cross_outer_round0_source(
            paths,
            outer_round,
            cfg,
            status,
            resume,
            metrics_logger=metrics_logger,
        )
    else:
        mace_train_merged = _round_source.build_round0_bootstrap_source(paths, outer_round, cfg)

    if manifest_path.exists():
        manifest = read_json(manifest_path)

    label_dir = ensure_dir(paths.dp_subworkflow_round_label_dir(outer_round, dp_round))
    al_dir = ensure_dir(paths.dp_subworkflow_round_active_learning_dir(outer_round, dp_round))
    ensure_dir(paths.dp_subworkflow_results_round_dir(outer_round, dp_round))

    label_step_id = f"r{outer_round}_dp_r0_label"
    label_attempt = _metrics_attempt(manifest, "label")
    if metrics_logger is not None:
        metrics_logger.start_step(
            label_step_id,
            "dp_round0_label",
            outer_round,
            payload={
                "attempt": label_attempt,
                "input_path": str(mace_train_merged.resolve()),
                "output_path": str(label_dir.resolve()),
            },
        )
    try:
        run_label(
            {"config": cfg["label_config"]},
            paths,
            outer_round,
            context={
                "structure_file_override": str(mace_train_merged.resolve()),
                "teacher_round_override": outer_round,
                "include_user_dp_train": False,
                "merged_xyz_override": str((label_dir / "train_merged.xyz").resolve()),
                "dp_round_dir_override": str(label_dir.resolve()),
                "active_learning_round_dir_override": str(al_dir.resolve()),
                "append_active_learning_xyz": str(paths.dp_subworkflow_mace_al_pool_xyz(outer_round).resolve()),
                "train_committee_config_override": cfg["train_dp_committee_config"],
                "training_stage_override": "bootstrap",
            },
        )
        label_counts = _record_labeling_metrics(metrics_logger, outer_round, label_step_id, label_dir, al_dir, mace_train_merged)
        status["total_labeled_structures"] = int(status.get("total_labeled_structures", 0) or 0) + int(label_counts["dft"])
        status["total_discarded_structures"] = int(status.get("total_discarded_structures", 0) or 0) + int(label_counts["discard"])
        status["total_train_structures_added"] = int(status.get("total_train_structures_added", 0) or 0) + int(label_counts["train"])
        if metrics_logger is not None:
            metrics_logger.finish_step(
                label_step_id,
                "SUCCESS",
                updates={
                    "attempt": label_attempt,
                    "input_structure_count": label_counts["input"],
                    "output_structure_count": label_counts["train"] + label_counts["dft"],
                    "output_path": str(label_dir.resolve()),
                },
            )
    except KeyboardInterrupt as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(label_step_id, "INTERRUPTED", updates={"attempt": label_attempt, "error_message": str(exc)})
        raise
    except SystemExit as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(label_step_id, "INTERRUPTED", updates={"attempt": label_attempt, "error_message": str(exc)})
        raise
    except Exception as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(
                label_step_id,
                "FAILED",
                updates={"attempt": label_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
            )
        raise

    manifest["steps"]["label"] = {
        "completed": True,
        "structure_file": str(mace_train_merged.resolve()),
        "label_dir": str(label_dir.resolve()),
        "active_learning_dir": str(al_dir.resolve()),
        "round0_source_mode": round0_source_mode,
        "dp_data_policy": _dp_data_policy(paths, outer_round, cfg),
        "metrics_attempt": label_attempt,
    }
    write_json(manifest_path, manifest)
    status["current_dp_round"] = 0
    _write_status(paths.dp_subworkflow_status_json(outer_round), status)


def _train_dp_round(paths, outer_round: int, dp_round: int, cfg: dict[str, Any], status: dict[str, Any], resume: bool, metrics_logger=None) -> Path:
    manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round)
    manifest = read_json(manifest_path) if manifest_path.exists() else base_round_manifest(outer_round, dp_round)
    committee_dir = ensure_dir(paths.dp_subworkflow_round_committee_dir(outer_round, dp_round))
    summary_path = committee_dir / "summary.json"
    arch_meta_path = paths.dp_subworkflow_round_arch_meta_json(outer_round, dp_round)

    if resume and manifest.get("steps", {}).get("train_dp_committee", {}).get("completed") and summary_path.exists():
        return summary_path

    dataset_roots = _training_dataset_roots_for_round(paths, outer_round, dp_round)
    previous_model_root = None
    continue_training_override = False
    training_policy = dict(cfg.get("training_policy", {}) or {})
    round0_stage = str(training_policy.get("round0_stage", "bootstrap")).strip().lower()
    later_round_stage = str(training_policy.get("later_round_stage", "finetune")).strip().lower()

    if dp_round > 0:
        previous_model_root = paths.dp_subworkflow_round_committee_dir(outer_round, dp_round - 1)
        continue_training_override = later_round_stage == "finetune"
    elif outer_round > 0:
        previous_model_root = _round_source.resolve_previous_outer_round_committee_root(paths, outer_round)
        continue_training_override = True

    step_id = f"r{outer_round}_dp_r{dp_round}_train_dp"
    train_attempt = _metrics_attempt(manifest, "train_dp_committee")
    if dp_round == 0:
        if outer_round > 0:
            training_stage = "finetune"
        else:
            training_stage = round0_stage
    else:
        training_stage = "finetune" if continue_training_override else "bootstrap"

    if continue_training_override and previous_model_root is None:
        if dp_round > 0:
            raise FileNotFoundError(
                f"Expected previous DP committee directory for outer_round={outer_round}, dp_round={dp_round - 1}, "
                "but no previous model root was found."
            )
        raise FileNotFoundError(
            f"Expected previous outer-round final DP model for outer_round={outer_round}, "
            "but the previous outer round status/final model could not be resolved."
        )
    if metrics_logger is not None:
        metrics_logger.start_step(
            step_id,
            "dp_train_committee",
            outer_round,
            payload={
                "attempt": train_attempt,
                "input_dataset_size": len(dataset_roots),
                "input_path": ",".join(dataset_roots),
                "output_path": str(committee_dir.resolve()),
            },
        )
    try:
        result = run_dp_train_committee(
            cfg["train_dp_committee_config"],
            paths,
            outer_round,
            context={
                "dataset_roots": dataset_roots,
                "work_root": str(committee_dir.resolve()),
                "summary_path": str(summary_path.resolve()),
                "arch_meta_path": str(arch_meta_path.resolve()),
                "split_meta_path": str((committee_dir / "split_meta.json").resolve()),
                "log_round_id": outer_round,
                "log_prefix": f"dp_subworkflow_r{outer_round}_dp{dp_round}_",
                "previous_model_root": str(previous_model_root.resolve()) if previous_model_root else None,
                "continue_training_override": continue_training_override,
                "training_stage": training_stage,
            },
        )
        training_metrics = _record_dp_training_metrics(metrics_logger, outer_round, step_id, summary_path)
        status["final_dp_model_path"] = str(summary_path.resolve())
        if metrics_logger is not None:
            metrics_logger.finish_step(
                step_id,
                "SUCCESS",
                updates={
                    "attempt": train_attempt,
                    "input_dataset_size": len(dataset_roots),
                    "output_path": str(committee_dir.resolve()),
                    "model_path": str(summary_path.resolve()),
                    **training_metrics,
                },
            )
    except KeyboardInterrupt as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(step_id, "INTERRUPTED", updates={"attempt": train_attempt, "error_message": str(exc)})
        raise
    except SystemExit as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(step_id, "INTERRUPTED", updates={"attempt": train_attempt, "error_message": str(exc)})
        raise
    except Exception as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(
                step_id,
                "FAILED",
                updates={"attempt": train_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
            )
        raise

    manifest["steps"]["train_dp_committee"] = {
        "completed": True,
        "training_mode": "incremental_finetune" if continue_training_override else "bootstrap",
        "training_stage": training_stage,
        "dataset_roots": dataset_roots,
        "summary_path": str(Path(result).resolve()),
        "previous_model_root": str(previous_model_root.resolve()) if previous_model_root is not None else None,
        "arch_meta_path": str(arch_meta_path.resolve()) if arch_meta_path.exists() else None,
        "metrics_attempt": train_attempt,
    }
    write_json(manifest_path, manifest)
    status["current_dp_round"] = dp_round
    if dp_round not in status["finished_rounds"]:
        status["finished_rounds"].append(dp_round)
    _write_status(paths.dp_subworkflow_status_json(outer_round), status)
    return Path(result).resolve()


def _resolve_sampling_profile_name(cfg: dict[str, Any], dp_round: int) -> str:
    schedule = dict(cfg.get("sampling_schedule", {}) or {})
    if str(schedule.get("mode", "long_first_then_short")).strip().lower() == "long_first_then_short":
        if int(dp_round) == 1:
            return str(schedule.get("long_profile_name", "long_exploration"))
        return str(schedule.get("short_profile_name", "short_refinement"))
    return str(schedule.get("long_profile_name", "long_exploration"))


def _resolve_sampling_profile(cfg: dict[str, Any], dp_round: int) -> tuple[str, dict[str, Any]]:
    profile_name = _resolve_sampling_profile_name(cfg, dp_round)
    profiles = dict(cfg.get("sampling_profiles", {}) or {})
    profile = dict(profiles.get(profile_name, {}) or {})
    return profile_name, profile


def _is_short_sampling_profile(cfg: dict[str, Any], dp_round: int, profile_name: str) -> bool:
    schedule = dict(cfg.get("sampling_schedule", {}) or {})
    mode = str(schedule.get("mode", "long_first_then_short")).strip().lower()
    if mode == "long_first_then_short":
        return int(dp_round) > 1
    return str(profile_name).strip() == str(schedule.get("short_profile_name", "short_refinement")).strip()


def _run_seed_pool(paths, outer_round: int, dp_round: int, cfg: dict[str, Any], profile_cfg: dict[str, Any]) -> Path:
    selected_seed_file = Path(select_md_seeds_from_pool(cfg, paths, outer_round, dp_round, profile_cfg=profile_cfg)).resolve()
    seed_pool_xyz = paths.dp_subworkflow_seed_pool_xyz(outer_round, dp_round)
    atoms_list = load_atoms_list(selected_seed_file)
    write_candidate_xyz(seed_pool_xyz, atoms_list)
    return seed_pool_xyz.resolve()


def _run_sampling(
    paths,
    outer_round: int,
    dp_round: int,
    cfg: dict[str, Any],
    profile_name: str,
    profile_cfg: dict[str, Any],
    metrics_logger=None,
    manifest: dict[str, Any] | None = None,
    *,
    committee_summary_override: str | None = None,
    seed_pool_override: str | None = None,
    reference_xyz_files_override: list[str] | None = None,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampling_dir = ensure_dir(paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round))
    if committee_summary_override is not None:
        committee_summary = Path(committee_summary_override).resolve()
    else:
        previous_manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round - 1)
        committee_summary = paths.dp_subworkflow_round_committee_dir(outer_round, dp_round - 1) / "summary.json"
        if previous_manifest_path.exists():
            previous_manifest = read_json(previous_manifest_path)
            train_step = previous_manifest.get("steps", {}).get("train_dp_committee", {})
            if train_step.get("summary_path"):
                committee_summary = Path(train_step["summary_path"]).resolve()
    if not committee_summary.exists():
        raise FileNotFoundError(f"Previous DP committee summary not found: {committee_summary}")

    if reference_xyz_files_override is not None:
        reference_xyz_files = list(reference_xyz_files_override)
    else:
        reference_xyz_files = _cumulative_train_xyz_upto(paths, outer_round, dp_round - 1)
    step_id = f"r{outer_round}_dp_r{dp_round}_sampling"
    sampling_attempt = _metrics_attempt(manifest or {}, "dp_sample_committee")
    if metrics_logger is not None:
        metrics_logger.start_step(
            step_id,
            "dp_sampling",
            outer_round,
            payload={
                "attempt": sampling_attempt,
                "input_path": str(Path(seed_pool_override or paths.dp_subworkflow_seed_pool_xyz(outer_round, dp_round)).resolve()),
                "output_path": str(sampling_dir.resolve()),
            },
        )
    try:
        base_lammps_cfg = dict((cfg.get("sampling_seed_policy", {}) or {}).get("lammps", {}) or {})
        profile_lammps_cfg = dict((profile_cfg.get("sampling_seed_policy", {}) or {}).get("lammps", {}) or {})
        stop_high_unc_threshold = float(profile_lammps_cfg.get("f_trust_lo", base_lammps_cfg.get("f_trust_lo", 0.10)))
        sampling_progress_log = ensure_step_log_file(
            paths,
            "dp_sampling",
            outer_round,
            filename=f"dp_subworkflow_r{outer_round}_dp{dp_round}_sampling.log",
        )
        summary_path = run_dp_sample_committee(
            None,
            paths,
            outer_round,
            context={
                "committee_summary_override": str(committee_summary.resolve()),
                "work_root_override": str(sampling_dir.resolve()),
                "seed_pool_override": str(Path(seed_pool_override or paths.dp_subworkflow_seed_pool_xyz(outer_round, dp_round)).resolve()),
                "reference_xyz_files_override": reference_xyz_files,
                "stop_high_unc_threshold": stop_high_unc_threshold,
                "sampling_profile_name": profile_name,
                "sampling_profile_config": dict(profile_cfg.get("sampling_seed_policy", {}) or {}),
                "sampling_seed_policy": dict(cfg.get("sampling_seed_policy", {}) or {}),
                "candidate_train_selection": dict(profile_cfg.get("candidate_train_selection", {}) or {}),
                "sampling_progress_log_path": str(sampling_progress_log.resolve()),
                "sampling_progress_poll_interval_sec": 2.0,
                "summary_extra": {
                    "outer_round": int(outer_round),
                    "dp_round": int(dp_round),
                    "sampling_profile_name": profile_name,
                    **dict(summary_extra or {}),
                },
            },
        )
        summary_payload = read_json(summary_path)
        _record_sampling_metrics(metrics_logger, outer_round, step_id, sampling_dir)
        if metrics_logger is not None:
            metrics_logger.finish_step(
                step_id,
                "SUCCESS",
                updates={
                    "attempt": sampling_attempt,
                    "input_structure_count": count_extxyz_frames(paths.dp_subworkflow_seed_pool_xyz(outer_round, dp_round)),
                    "output_structure_count": int(summary_payload.get("selected_count", 0)),
                    "output_path": str(sampling_dir.resolve()),
                },
            )
        return summary_payload
    except KeyboardInterrupt as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(step_id, "INTERRUPTED", updates={"attempt": sampling_attempt, "error_message": str(exc)})
        raise
    except SystemExit as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(step_id, "INTERRUPTED", updates={"attempt": sampling_attempt, "error_message": str(exc)})
        raise
    except Exception as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(
                step_id,
                "FAILED",
                updates={"attempt": sampling_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
            )
        raise


def _run_label_for_sampling(paths, outer_round: int, dp_round: int, cfg: dict[str, Any], metrics_logger=None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    label_dir = ensure_dir(paths.dp_subworkflow_round_label_dir(outer_round, dp_round))
    al_dir = ensure_dir(paths.dp_subworkflow_round_active_learning_dir(outer_round, dp_round))
    selected_xyz = paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round) / "selected_candidates.xyz"
    if not selected_xyz.exists():
        raise FileNotFoundError(f"Selected candidate xyz not found: {selected_xyz}")

    step_id = f"r{outer_round}_dp_r{dp_round}_label"
    label_attempt = _metrics_attempt(manifest or {}, "label")
    if metrics_logger is not None:
        metrics_logger.start_step(
            step_id,
            "dp_label",
            outer_round,
            payload={
                "attempt": label_attempt,
                "input_path": str(selected_xyz.resolve()),
                "output_path": str(label_dir.resolve()),
            },
        )
    try:
        run_label(
            {"config": cfg["label_config"]},
            paths,
            outer_round,
            context={
                "structure_file_override": str(selected_xyz.resolve()),
                "teacher_round_override": outer_round,
                "include_user_dp_train": False,
                "merged_xyz_override": str((label_dir / "train_merged.xyz").resolve()),
                "dp_round_dir_override": str(label_dir.resolve()),
                "active_learning_round_dir_override": str(al_dir.resolve()),
                "append_active_learning_xyz": str(paths.dp_subworkflow_mace_al_pool_xyz(outer_round).resolve()),
                "train_committee_config_override": cfg["train_dp_committee_config"],
                "training_stage_override": "finetune",
            },
        )
        counts = _record_labeling_metrics(metrics_logger, outer_round, step_id, label_dir, al_dir, selected_xyz)
        if metrics_logger is not None:
            metrics_logger.finish_step(
                step_id,
                "SUCCESS",
                updates={
                    "attempt": label_attempt,
                    "input_structure_count": counts["input"],
                    "output_structure_count": counts["train"] + counts["dft"],
                    "output_path": str(label_dir.resolve()),
                },
            )
    except KeyboardInterrupt as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(step_id, "INTERRUPTED", updates={"attempt": label_attempt, "error_message": str(exc)})
        raise
    except SystemExit as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(step_id, "INTERRUPTED", updates={"attempt": label_attempt, "error_message": str(exc)})
        raise
    except Exception as exc:
        if metrics_logger is not None:
            metrics_logger.finish_step(
                step_id,
                "FAILED",
                updates={"attempt": label_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
            )
        raise

    return {
        "selected_xyz": str(selected_xyz.resolve()),
        "label_dir": str(label_dir.resolve()),
        "active_learning_dir": str(al_dir.resolve()),
        "metrics_attempt": label_attempt,
        "label_counts": counts,
        "label_system_stats": summarize_label_split_by_system(
            label_dir / "train.xyz",
            al_dir / "dft.xyz",
            label_dir / "discard.xyz",
        ),
    }
