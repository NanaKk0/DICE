from __future__ import annotations

from pathlib import Path

from workflow_core.common.io import ensure_dir, read_json, write_json
from workflow_core.dft.screening import screen_current_round_dft_candidates
from workflow_core.labeling.split_stats import summarize_label_split_by_system
from workflow_core.reporting.data_distribution import plot_dp_round_increment
from workflow_core.seed_pool.manager import init_seed_pool, record_seed_usage_after_sampling, refresh_seed_pool_after_training


# Mutable bindings: subworkflow_service syncs its current compatibility aliases
# into this module before calling the main entrypoint.
load_dp_active_learning_config = None
_load_status = None
_write_status = None
_rebuild_status_from_manifests = None
_sync_screened_dft_pool_from_manifests = None
_dp_data_policy = None
_ensure_round0_labeled = None
_train_dp_round = None
_resolve_sampling_profile = None
_is_short_sampling_profile = None
_run_seed_pool = None
_run_sampling = None
_run_label_for_sampling = None
_load_completed_seed_pool_output_from_manifest = None
_load_completed_sampling_summary_from_manifest = None
_load_completed_label_info_from_manifest = None
_load_completed_dft_screening_info_from_manifest = None
_load_completed_seed_update_summary_from_manifest = None
_training_dataset_roots_for_round = None
_snapshot_final_seed_pool = None
_base_round_manifest = None
_metrics_attempt = None


def _resolve_dft_screening_stop_reason(cfg: dict[str, object], current_screened_count: int, cumulative_screened_count: int) -> str | None:
    policy = dict(cfg.get("dft_screening_policy", {}) or {})
    if not policy.get("enabled", False):
        return None
    current_threshold = int(policy.get("stop_dp_round_when_current_screened_count_gte", 0) or 0)
    cumulative_threshold = int(policy.get("stop_dp_round_when_cumulative_screened_count_gte", 0) or 0)
    if current_threshold > 0 and int(current_screened_count) >= current_threshold:
        return "screened_dft_count_threshold_reached"
    if cumulative_threshold > 0 and int(cumulative_screened_count) >= cumulative_threshold:
        return "cumulative_screened_dft_count_threshold_reached"
    return None


def _finalize_round0_artifacts(
    paths,
    outer_round: int,
    cfg: dict[str, object],
    status: dict[str, object],
    manifest_path: Path,
    manifest: dict[str, object],
    *,
    stop_reason: str | None,
    round0_current_screened_count: int,
    round0_cumulative_screened_count: int,
    round0_dft_screening_info: dict[str, object] | None,
    mark_workflow_stopped: bool,
    status_path: Path,
) -> Path:
    manifest["round_result"] = {
        "outer_round": int(outer_round),
        "dp_round": 0,
        "stopped_before_label": False,
        "stop_reason": stop_reason,
        "committee_summary_path": manifest.get("steps", {}).get("train_dp_committee", {}).get("summary_path"),
        "training_dataset_roots": _training_dataset_roots_for_round(paths, outer_round, 0),
        "dp_data_policy": _dp_data_policy(paths, outer_round, cfg),
        "screened_dft_count": round0_current_screened_count,
        "cumulative_screened_dft_count": round0_cumulative_screened_count,
        "dft_screening_summary_json": str(paths.dp_subworkflow_round_dft_screening_summary_json(outer_round, 0).resolve())
        if round0_dft_screening_info is not None
        else None,
        "screened_dft_pool_xyz": str(paths.dp_subworkflow_screened_dft_pool_xyz(outer_round).resolve())
        if round0_dft_screening_info is not None
        else None,
    }
    if 0 not in status["finished_rounds"]:
        status["finished_rounds"].append(0)
    manifest["completed"] = True
    write_json(manifest_path, manifest)
    status["current_dp_round"] = 0
    if mark_workflow_stopped:
        status["stopped"] = True
        status["stop_reason"] = stop_reason
        final_seed_pool_snapshot = _snapshot_final_seed_pool(paths, outer_round)
        if final_seed_pool_snapshot is not None:
            status["final_seed_pool_snapshot"] = final_seed_pool_snapshot
        _write_status(status_path, status)
    return manifest_path


def run_dp_active_learning_subworkflow(cfg_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    cfg = load_dp_active_learning_config(cfg_path)
    resume = bool(cfg.get("resume", True))
    mace_teacher_dir = paths.mace_committee_dir(outer_round)
    if not mace_teacher_dir.exists():
        raise FileNotFoundError(f"MACE committee directory not found for outer round r{outer_round}: {mace_teacher_dir}")

    ensure_dir(paths.dp_subworkflow_dir(outer_round))
    ensure_dir(paths.dp_subworkflow_results_dir(outer_round))
    status_path = paths.dp_subworkflow_status_json(outer_round)
    status = _load_status(status_path, outer_round, mace_teacher_dir)
    if resume:
        status = _rebuild_status_from_manifests(paths, outer_round, cfg, status)
        _sync_screened_dft_pool_from_manifests(paths, outer_round, cfg)
    status["dp_data_policy"] = _dp_data_policy(paths, outer_round, cfg)
    _write_status(status_path, status)

    seed_pool_cfg = cfg.get("seed_pool_policy", {})
    if seed_pool_cfg.get("enabled", True) and seed_pool_cfg.get("initialize_at_subworkflow_start", True):
        if not paths.seed_pool_xyz().exists():
            init_seed_pool(cfg, paths)

    if not bool(cfg.get("skip_round0_label", False)):
        _ensure_round0_labeled(paths, outer_round, cfg, status, resume=resume, metrics_logger=metrics_logger)

    bootstrap_committee_summary = cfg.get("bootstrap_committee_summary")
    if bootstrap_committee_summary:
        bootstrap_committee_summary = Path(bootstrap_committee_summary).resolve()
        if not bootstrap_committee_summary.exists():
            raise FileNotFoundError(f"bootstrap_committee_summary not found: {bootstrap_committee_summary}")
        manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, 0)
        manifest = read_json(manifest_path) if manifest_path.exists() else _base_round_manifest(outer_round, 0)
        manifest["steps"]["train_dp_committee"] = {
            "completed": True,
            "training_mode": "bootstrap",
            "dataset_roots": _training_dataset_roots_for_round(paths, outer_round, 0),
            "summary_path": str(bootstrap_committee_summary),
            "bootstrap_reused": True,
        }
        write_json(manifest_path, manifest)
        if 0 not in status["finished_rounds"]:
            status["finished_rounds"].append(0)
        status["final_dp_model_path"] = str(bootstrap_committee_summary.resolve())
        status["current_dp_round"] = 0
        _write_status(status_path, status)
    else:
        _train_dp_round(paths, outer_round, 0, cfg, status, resume=resume, metrics_logger=metrics_logger)

    round0_manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, 0)
    round0_manifest = read_json(round0_manifest_path) if round0_manifest_path.exists() else _base_round_manifest(outer_round, 0)
    round0_train_step = (round0_manifest.get("steps", {}) or {}).get("train_dp_committee", {}) or {}
    round0_committee_summary_path = Path(
        round0_train_step.get(
            "summary_path",
            bootstrap_committee_summary or (paths.dp_subworkflow_round_committee_dir(outer_round, 0) / "summary.json"),
        )
    ).resolve()
    if (
        int(outer_round) > 0
        and cfg.get("seed_pool_policy", {}).get("enabled", True)
        and cfg.get("seed_pool_policy", {}).get("refresh_after_training", True)
        and not bool((round0_manifest.get("steps", {}) or {}).get("reset_seed_usage_after_round0_train", {}).get("completed"))
    ):
        base_lammps_cfg = dict((cfg.get("sampling_seed_policy", {}) or {}).get("lammps", {}) or {})
        refresh_summary_path = refresh_seed_pool_after_training(
            cfg,
            paths,
            outer_round,
            0,
            round0_committee_summary_path,
            refresh_mode="rescore_existing_pool_only",
            keep_if_ge_low=None,
            keep_if_lt_high=float(base_lammps_cfg.get("f_trust_hi", 0.40)),
            reset_times_used=True,
        )
        round0_manifest.setdefault("steps", {})["reset_seed_usage_after_round0_train"] = {
            "completed": True,
            "summary_path": str(Path(refresh_summary_path).resolve()),
            "seed_pool_file": str(paths.seed_pool_xyz().resolve()),
            "refresh_mode": "rescore_existing_pool_only",
            "reset_times_used": True,
        }
        write_json(round0_manifest_path, round0_manifest)
    round0_dft_screening_info = _load_completed_dft_screening_info_from_manifest(round0_manifest, paths, outer_round, 0) if resume else None
    if cfg.get("dft_screening_policy", {}).get("enabled", False):
        if round0_dft_screening_info is None:
            round0_dft_screening_info = screen_current_round_dft_candidates(
                paths,
                outer_round,
                0,
                dict(cfg.get("dft_screening_policy", {}) or {}),
            )
            round0_manifest["steps"]["dft_screening"] = {
                "completed": True,
                "summary_path": str(paths.dp_subworkflow_round_dft_screening_summary_json(outer_round, 0).resolve()),
                "screened_xyz": str(paths.dp_subworkflow_round_dft_screened_xyz(outer_round, 0).resolve()),
                "screened_count": int(round0_dft_screening_info.get("screened_count", 0) or 0),
                "cumulative_screened_count": int(
                    round0_dft_screening_info.get("current_outer_cumulative_screened_count", 0) or 0
                ),
            }
            write_json(round0_manifest_path, round0_manifest)
            status["total_screened_dft_structures"] = int(status.get("total_screened_dft_structures", 0) or 0) + int(
                round0_dft_screening_info.get("screened_count", 0) or 0
            )
    round0_current_screened_count = int((round0_dft_screening_info or {}).get("screened_count", 0) or 0)
    round0_cumulative_screened_count = int(
        (round0_dft_screening_info or {}).get("current_outer_cumulative_screened_count", 0) or 0
    )
    round0_post_train_stop_reason = _resolve_dft_screening_stop_reason(
        cfg,
        round0_current_screened_count,
        round0_cumulative_screened_count,
    )

    if round0_post_train_stop_reason is not None:
        round0_manifest = read_json(round0_manifest_path) if round0_manifest_path.exists() else round0_manifest
        _finalize_round0_artifacts(
            paths,
            outer_round,
            cfg,
            status,
            round0_manifest_path,
            round0_manifest,
            stop_reason=round0_post_train_stop_reason,
            round0_current_screened_count=round0_current_screened_count,
            round0_cumulative_screened_count=round0_cumulative_screened_count,
            round0_dft_screening_info=round0_dft_screening_info,
            mark_workflow_stopped=True,
            status_path=status_path,
        )
        return str(status_path.resolve())

    if int(cfg["max_dp_rounds"]) <= 0:
        round0_manifest = read_json(round0_manifest_path) if round0_manifest_path.exists() else _base_round_manifest(outer_round, 0)
        _finalize_round0_artifacts(
            paths,
            outer_round,
            cfg,
            status,
            round0_manifest_path,
            round0_manifest,
            stop_reason="reached_max_dp_rounds",
            round0_current_screened_count=round0_current_screened_count,
            round0_cumulative_screened_count=round0_cumulative_screened_count,
            round0_dft_screening_info=round0_dft_screening_info,
            mark_workflow_stopped=True,
            status_path=status_path,
        )
        return str(status_path.resolve())

    round0_manifest = read_json(round0_manifest_path) if round0_manifest_path.exists() else round0_manifest
    _finalize_round0_artifacts(
        paths,
        outer_round,
        cfg,
        status,
        round0_manifest_path,
        round0_manifest,
        stop_reason=None,
        round0_current_screened_count=round0_current_screened_count,
        round0_cumulative_screened_count=round0_cumulative_screened_count,
        round0_dft_screening_info=round0_dft_screening_info,
        mark_workflow_stopped=False,
        status_path=status_path,
    )
    _write_status(status_path, status)

    stop_reason = None
    stop_round = None

    for dp_round in range(1, int(cfg["max_dp_rounds"]) + 1):
        manifest_path = paths.dp_subworkflow_round_manifest_json(outer_round, dp_round)
        manifest = read_json(manifest_path) if manifest_path.exists() else _base_round_manifest(outer_round, dp_round)
        ensure_dir(paths.dp_subworkflow_results_round_dir(outer_round, dp_round))
        sampling_profile_name, sampling_profile_cfg = _resolve_sampling_profile(cfg, dp_round)

        if resume and bool(manifest.get("completed")):
            if dp_round not in status["finished_rounds"]:
                status["finished_rounds"].append(dp_round)
            status["current_dp_round"] = dp_round
            _write_status(status_path, status)
            if dp_round >= int(cfg["max_dp_rounds"]):
                stop_reason = "reached_max_dp_rounds"
                stop_round = dp_round
                break
            continue

        if cfg.get("seed_pool_policy", {}).get("enabled", True):
            seed_pool_xyz = _load_completed_seed_pool_output_from_manifest(manifest, paths, outer_round, dp_round) if resume else None
            if seed_pool_xyz is None:
                seed_pool_xyz = _run_seed_pool(paths, outer_round, dp_round, cfg, sampling_profile_cfg)
                manifest["steps"]["build_seed_pool"] = {
                    "completed": True,
                    "selected_seed_xyz": str(seed_pool_xyz),
                    "formal_seed_pool_xyz": str(paths.seed_pool_xyz().resolve()),
                    "sampling_profile_name": sampling_profile_name,
                }
                write_json(manifest_path, manifest)

        sampling_summary = _load_completed_sampling_summary_from_manifest(manifest, paths, outer_round, dp_round) if resume else None
        if sampling_summary is None:
            sampling_summary = _run_sampling(
                paths,
                outer_round,
                dp_round,
                cfg,
                sampling_profile_name,
                sampling_profile_cfg,
                metrics_logger=metrics_logger,
                manifest=manifest,
            )
            manifest["steps"]["dp_sample_committee"] = {
                "completed": True,
                "sampling_summary_path": str((paths.dp_subworkflow_round_sampling_dir(outer_round, dp_round) / "sampling_summary.json").resolve()),
                "high_unc_ratio": float(sampling_summary.get("high_unc_ratio", 0.0)),
                "effective_sample_count": int(sampling_summary.get("effective_sample_count", 0)),
                "selected_count": int(sampling_summary.get("selected_count", 0)),
                "metrics_attempt": _metrics_attempt(manifest, "dp_sample_committee"),
                "sampling_profile_name": sampling_profile_name,
            }
            write_json(manifest_path, manifest)
            status["total_candidate_structures"] = int(status.get("total_candidate_structures", 0) or 0) + int(
                sampling_summary.get("candidate_count", 0) or 0
            ) + int(sampling_summary.get("suspicious_candidate_count", 0) or 0)
            status["total_selected_structures"] = int(status.get("total_selected_structures", 0) or 0) + int(
                sampling_summary.get("selected_count", 0) or 0
            )

        if (
            cfg.get("seed_pool_policy", {}).get("enabled", True)
            and _is_short_sampling_profile(cfg, dp_round, sampling_profile_name)
            and not bool((manifest.get("steps", {}) or {}).get("update_seed_usage", {}).get("completed"))
        ):
            updated_seed_pool_path = record_seed_usage_after_sampling(paths, outer_round, dp_round)
            manifest.setdefault("steps", {})["update_seed_usage"] = {
                "completed": True,
                "seed_pool_file": str(Path(updated_seed_pool_path).resolve()),
                "sampling_profile_name": sampling_profile_name,
                "selection_mode": str(
                    (sampling_profile_cfg.get("seed_selection", {}) or {}).get(
                        "selection_mode",
                        "usage_then_uncertainty_per_system",
                    )
                ),
            }
            write_json(manifest_path, manifest)

        seed_update_summary = None
        high_unc_ratio = float(sampling_summary.get("high_unc_ratio", 0.0))
        if high_unc_ratio < float(cfg["stop_ratio_threshold"]):
            stop_reason = "converged_by_uncertainty_ratio"
            stop_round = dp_round
            manifest["round_result"] = {
                "outer_round": int(outer_round),
                "dp_round": int(dp_round),
                "stopped_before_label": True,
                "stop_reason": stop_reason,
                "high_unc_ratio": high_unc_ratio,
                "effective_sample_count": int(sampling_summary.get("effective_sample_count", 0)),
                "selected_count": int(sampling_summary.get("selected_count", 0)),
                "training_dataset_roots": _training_dataset_roots_for_round(paths, outer_round, max(dp_round - 1, 0)),
                "dp_data_policy": _dp_data_policy(paths, outer_round, cfg),
                "seed_pool_update": seed_update_summary,
                "sampling_system_stats": sampling_summary.get("sampling_system_stats") or sampling_summary.get("system_stats"),
            }
            manifest["completed"] = True
            write_json(manifest_path, manifest)
            break

        if int(sampling_summary.get("selected_count", 0)) <= 0:
            stop_reason = "no_selected_candidates"
            stop_round = dp_round
            manifest["round_result"] = {
                "outer_round": int(outer_round),
                "dp_round": int(dp_round),
                "stopped_before_label": True,
                "stop_reason": stop_reason,
                "high_unc_ratio": high_unc_ratio,
                "effective_sample_count": int(sampling_summary.get("effective_sample_count", 0)),
                "selected_count": 0,
                "training_dataset_roots": _training_dataset_roots_for_round(paths, outer_round, max(dp_round - 1, 0)),
                "dp_data_policy": _dp_data_policy(paths, outer_round, cfg),
                "seed_pool_update": seed_update_summary,
                "sampling_system_stats": sampling_summary.get("sampling_system_stats") or sampling_summary.get("system_stats"),
            }
            manifest["completed"] = True
            write_json(manifest_path, manifest)
            break

        label_info = _load_completed_label_info_from_manifest(manifest, paths, outer_round, dp_round) if resume else None
        if label_info is None:
            label_info = _run_label_for_sampling(paths, outer_round, dp_round, cfg, metrics_logger=metrics_logger, manifest=manifest)
            manifest["steps"]["label"] = {
                "completed": True,
                "selected_xyz": label_info["selected_xyz"],
                "label_dir": label_info["label_dir"],
                "active_learning_dir": label_info["active_learning_dir"],
                "metrics_attempt": label_info["metrics_attempt"],
            }
            write_json(manifest_path, manifest)
            label_counts = label_info["label_counts"]
            status["total_labeled_structures"] = int(status.get("total_labeled_structures", 0) or 0) + int(label_counts["dft"])
            status["total_discarded_structures"] = int(status.get("total_discarded_structures", 0) or 0) + int(label_counts["discard"])
            status["total_train_structures_added"] = int(status.get("total_train_structures_added", 0) or 0) + int(label_counts["train"])
        label_counts = label_info["label_counts"]
        label_system_stats = label_info.get("label_system_stats")
        if label_system_stats is None:
            label_dir = Path(label_info["label_dir"]).resolve()
            al_dir = Path(label_info["active_learning_dir"]).resolve()
            label_system_stats = summarize_label_split_by_system(
                label_dir / "train.xyz",
                al_dir / "dft.xyz",
                label_dir / "discard.xyz",
            )

        dft_screening_info = _load_completed_dft_screening_info_from_manifest(manifest, paths, outer_round, dp_round) if resume else None
        if cfg.get("dft_screening_policy", {}).get("enabled", False):
            if dft_screening_info is None:
                dft_screening_info = screen_current_round_dft_candidates(
                    paths,
                    outer_round,
                    dp_round,
                    dict(cfg.get("dft_screening_policy", {}) or {}),
                )
                manifest["steps"]["dft_screening"] = {
                    "completed": True,
                    "summary_path": str(paths.dp_subworkflow_round_dft_screening_summary_json(outer_round, dp_round).resolve()),
                    "screened_xyz": str(paths.dp_subworkflow_round_dft_screened_xyz(outer_round, dp_round).resolve()),
                    "screened_count": int(dft_screening_info.get("screened_count", 0) or 0),
                    "cumulative_screened_count": int(dft_screening_info.get("current_outer_cumulative_screened_count", 0) or 0),
                }
                write_json(manifest_path, manifest)
                status["total_screened_dft_structures"] = int(status.get("total_screened_dft_structures", 0) or 0) + int(
                    dft_screening_info.get("screened_count", 0) or 0
                )
        current_screened_count = int((dft_screening_info or {}).get("screened_count", 0) or 0)
        cumulative_screened_count = int((dft_screening_info or {}).get("current_outer_cumulative_screened_count", 0) or 0)

        committee_summary = _train_dp_round(paths, outer_round, dp_round, cfg, status, resume=resume, metrics_logger=metrics_logger)
        manifest = read_json(manifest_path)
        if cfg.get("seed_pool_policy", {}).get("enabled", True) and cfg.get("seed_pool_policy", {}).get("refresh_after_training", True):
            seed_update_summary = _load_completed_seed_update_summary_from_manifest(manifest, paths, outer_round, dp_round) if resume else None
            if seed_update_summary is None:
                refresh_mode = str(
                    (sampling_profile_cfg.get("seed_pool", {}) or {}).get(
                        "refresh_mode",
                        "merge_old_pool_with_current_candidate_failed" if dp_round == 1 else "rescore_existing_pool_only",
                    )
                )
                profile_lammps_cfg = dict((sampling_profile_cfg.get("sampling_seed_policy", {}) or {}).get("lammps", {}) or {})
                base_lammps_cfg = dict((cfg.get("sampling_seed_policy", {}) or {}).get("lammps", {}) or {})
                refresh_low = None
                if str(refresh_mode).strip().lower() == "merge_old_pool_with_current_candidate_failed":
                    refresh_low = float(profile_lammps_cfg.get("f_trust_lo", base_lammps_cfg.get("f_trust_lo", 0.10)))
                refresh_high = float(profile_lammps_cfg.get("f_trust_hi", base_lammps_cfg.get("f_trust_hi", 0.40)))
                refresh_summary_path = refresh_seed_pool_after_training(
                    cfg,
                    paths,
                    outer_round,
                    dp_round,
                    committee_summary,
                    refresh_mode=refresh_mode,
                    keep_if_ge_low=refresh_low,
                    keep_if_lt_high=refresh_high,
                )
                seed_update_summary = read_json(refresh_summary_path)
                manifest["steps"]["update_seed_pool"] = {
                    "completed": True,
                    "summary_path": str(Path(refresh_summary_path).resolve()),
                    "seed_pool_file": str(paths.seed_pool_xyz().resolve()),
                    "refresh_mode": refresh_mode,
                    "sampling_profile_name": sampling_profile_name,
                }
                write_json(manifest_path, manifest)
        post_train_stop_reason = _resolve_dft_screening_stop_reason(cfg, current_screened_count, cumulative_screened_count)
        manifest["round_result"] = {
            "outer_round": int(outer_round),
            "dp_round": int(dp_round),
            "sampling_profile_name": sampling_profile_name,
            "stopped_before_label": False,
            "stop_reason": post_train_stop_reason,
            "high_unc_ratio": high_unc_ratio,
            "effective_sample_count": int(sampling_summary.get("effective_sample_count", 0)),
            "selected_count": int(sampling_summary.get("selected_count", 0)),
            "committee_summary_path": str(committee_summary.resolve()),
            "training_dataset_roots": _training_dataset_roots_for_round(paths, outer_round, dp_round),
            "dp_data_policy": _dp_data_policy(paths, outer_round, cfg),
            "seed_pool_update": seed_update_summary,
            "candidate_count": int(sampling_summary.get("candidate_count", 0) or 0) + int(sampling_summary.get("suspicious_candidate_count", 0) or 0),
            "selected_count": int(sampling_summary.get("selected_count", 0) or 0),
            "labeled_count": int(label_counts["dft"]),
            "screened_dft_count": current_screened_count,
            "cumulative_screened_dft_count": cumulative_screened_count,
            "dft_screening_summary_json": str(paths.dp_subworkflow_round_dft_screening_summary_json(outer_round, dp_round).resolve()) if dft_screening_info is not None else None,
            "screened_dft_pool_xyz": str(paths.dp_subworkflow_screened_dft_pool_xyz(outer_round).resolve()) if dft_screening_info is not None else None,
            "train_count": int(label_counts["train"]),
            "discard_count": int(label_counts["discard"]),
            "sampling_system_stats": sampling_summary.get("sampling_system_stats") or sampling_summary.get("system_stats"),
            "label_system_stats": label_system_stats,
        }
        plot_payload = plot_dp_round_increment(paths, outer_round, dp_round)
        if plot_payload is not None:
            manifest["distribution_plot"] = plot_payload

        manifest["completed"] = True
        write_json(manifest_path, manifest)

        if post_train_stop_reason is not None:
            stop_reason = post_train_stop_reason
            stop_round = dp_round
            break

        if dp_round >= int(cfg["max_dp_rounds"]):
            stop_reason = "reached_max_dp_rounds"
            stop_round = dp_round
            break

    status["stopped"] = True
    status["stop_reason"] = stop_reason or "reached_max_dp_rounds"
    status["current_dp_round"] = int(stop_round if stop_round is not None else cfg["max_dp_rounds"])
    final_seed_pool_snapshot = _snapshot_final_seed_pool(paths, outer_round)
    if final_seed_pool_snapshot is not None:
        status["final_seed_pool_snapshot"] = final_seed_pool_snapshot
    _write_status(status_path, status)
    return str(status_path.resolve())
