from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.reporting.data_distribution import plot_outer_round_dft_increment
from workflow_core.dft.collect import collect_dft_results
from workflow_core.dft.defaults import (
    DEFAULT_DFT_INPUT_NAME,
    DEFAULT_DFT_JOBID_PREFIX,
    DEFAULT_DFT_TASK_DIR_PREFIX,
    DEFAULT_DFT_TEMPLATE_PATH,
)
from workflow_core.dft.prepare import prepare_dft_jobs
from workflow_core.dft.run import run_dft_jobs
from workflow_core.dft.runtime import resolve_dft_runtime
from workflow_core.common.io import ensure_dir, read_json, write_json
from workflow_runtime.state.run_state import RunStateManager
from workflow_runtime.state.step_runner import execute_step


def load_dft_workflow_config(config_path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(config_path).resolve())
    payload.setdefault("enabled", True)
    payload.setdefault("labelling_style", "cp2k")
    payload.setdefault("submit_config_template", str(config_path))
    payload.setdefault("skip_if_empty", True)
    payload.setdefault("prepare", True)
    payload.setdefault("run", True)
    payload.setdefault("collect", True)
    payload.setdefault("deduplicate", True)
    payload.setdefault("overwrite_existing", False)
    payload.setdefault("missing_cell_policy", "recover_or_fail")
    payload.setdefault("default_cell", None)
    payload["jobid_prefix"] = DEFAULT_DFT_JOBID_PREFIX
    payload["template_path"] = DEFAULT_DFT_TEMPLATE_PATH
    payload["structure_name"] = DEFAULT_DFT_STRUCTURE_NAME
    payload["modified_name"] = DEFAULT_DFT_INPUT_NAME
    payload["task_dir_prefix"] = DEFAULT_DFT_TASK_DIR_PREFIX
    return payload


def build_dft_runtime_cfg(base_cfg: dict[str, Any], paths, outer_round: int) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["jobid"] = str(base_cfg.get("jobid", f"{DEFAULT_DFT_JOBID_PREFIX}_r{outer_round:03d}"))
    screened_pool_xyz = paths.dp_subworkflow_screened_dft_pool_xyz(outer_round).resolve()
    raw_pool_xyz = paths.dp_subworkflow_mace_al_pool_xyz(outer_round).resolve()
    input_xyz = screened_pool_xyz if screened_pool_xyz.exists() and screened_pool_xyz.stat().st_size > 0 else raw_pool_xyz
    cfg["input_xyz"] = str(input_xyz)
    cfg["input_meta"] = str(paths.dft_submit_source_meta_json(outer_round).resolve())
    cfg["task_root"] = str(paths.dft_submit_data_dir(outer_round).resolve())
    cfg["output_root"] = str(paths.dft_submit_results_dir(outer_round).resolve())
    cfg["task_map_path"] = str(paths.dft_submit_task_map_json(outer_round).resolve())
    cfg["record_path"] = str(paths.dft_submit_record_json(outer_round).resolve())
    cfg["status_path"] = str(paths.dft_submit_status_json(outer_round).resolve())
    cfg["labeled_xyz"] = str(paths.dft_submit_labeled_xyz(outer_round).resolve())
    cfg["failed_xyz"] = str(paths.dft_submit_failed_xyz(outer_round).resolve())
    cfg["summary_path"] = str(paths.dft_submit_collect_summary_json(outer_round).resolve())
    cfg["submit_summary_json"] = str(paths.dft_submit_submit_summary_json(outer_round).resolve())
    cfg["template_path"] = str((paths.workspace_root / DEFAULT_DFT_TEMPLATE_PATH).resolve())
    return resolve_dft_runtime(cfg)


def dft_prepare_success_check(cfg: dict[str, Any]) -> bool:
    task_map_path = Path(cfg["task_map_path"]).resolve()
    record_path = Path(cfg["record_path"]).resolve()
    return task_map_path.exists() and task_map_path.stat().st_size > 0 and record_path.exists()


def dft_run_success_check(cfg: dict[str, Any]) -> bool:
    submit_summary = Path(cfg["submit_summary_json"]).resolve()
    return submit_summary.exists() and submit_summary.stat().st_size > 0


def dft_collect_success_check(cfg: dict[str, Any]) -> bool:
    summary_path = Path(cfg["summary_path"]).resolve()
    labeled_xyz = Path(cfg["labeled_xyz"]).resolve()
    failed_xyz = Path(cfg["failed_xyz"]).resolve()
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return False
    return labeled_xyz.exists() or failed_xyz.exists()


def _count_extxyz_frames(path: str | Path | None) -> int | None:
    if path is None:
        return None
    xyz_path = Path(path).resolve()
    if not xyz_path.exists() or not xyz_path.is_file():
        return None
    try:
        count = 0
        with open(xyz_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "Properties=" in line:
                    count += 1
        return count
    except Exception:
        return None


def _manifest_step_attempt(manifest: dict[str, Any], step_key: str) -> int:
    step_payload = manifest.get(step_key, {}) if isinstance(manifest.get(step_key), dict) else {}
    return int(step_payload.get("metrics_attempt", 0)) + 1


def run_dft_workflow(config_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    base_cfg = load_dft_workflow_config(config_path)
    cfg = build_dft_runtime_cfg(base_cfg, paths, outer_round)
    ensure_dir(Path(cfg["task_root"]))
    ensure_dir(Path(cfg["output_root"]))

    input_xyz = Path(cfg["input_xyz"])
    if base_cfg.get("skip_if_empty", True):
        if not input_xyz.exists() or input_xyz.stat().st_size == 0:
            payload = {
                "status": "skipped_empty_input",
                "outer_round": int(outer_round),
                "input_xyz": str(input_xyz),
                "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
            }
            write_json(cfg["status_path"], payload)
            return str(Path(cfg["status_path"]).resolve())

    result_dir = Path(cfg["output_root"]).resolve()
    state_mgr = RunStateManager(
        result_dir / "workflow_state.json",
        result_dir / "markers",
        backup_partial_outputs=False,
        verify_success_markers=True,
    )
    steps = [
        {
            "id": f"r{outer_round}_dft_prepare",
            "name": "prepare",
            "round": outer_round,
            "index": 0,
            "output_dir": str(Path(cfg["task_root"]).resolve()),
            "backup_partial_outputs": False,
        },
        {
            "id": f"r{outer_round}_dft_run",
            "name": "run",
            "round": outer_round,
            "index": 1,
            "output_dir": str(Path(cfg["output_root"]).resolve()),
            "backup_partial_outputs": False,
        },
        {
            "id": f"r{outer_round}_dft_collect",
            "name": "collect",
            "round": outer_round,
            "index": 2,
            "output_dir": str(Path(cfg["output_root"]).resolve()),
            "backup_partial_outputs": False,
        },
    ]
    if state_mgr.exists():
        state_mgr.load()
    else:
        state_mgr.init_state(run_name=f"dft_r{outer_round}", round_id=outer_round, steps=steps)
    state_mgr.reconcile(
        success_checks={
            f"r{outer_round}_dft_prepare": lambda: dft_prepare_success_check(cfg),
            f"r{outer_round}_dft_run": lambda: dft_run_success_check(cfg),
            f"r{outer_round}_dft_collect": lambda: dft_collect_success_check(cfg),
        }
    )
    resume_index = state_mgr.get_resume_index()
    state_mgr.mark_workflow_running(
        current_step_index=resume_index if resume_index < len(steps) else None,
        current_step_id=steps[resume_index]["id"] if resume_index < len(steps) else None,
    )

    manifest: dict[str, Any] = {
        "outer_round": int(outer_round),
        "jobid": cfg["jobid"],
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    if base_cfg.get("prepare", True):
        if resume_index <= 0:
            prepare_attempt = _manifest_step_attempt(manifest, "prepare")
            if metrics_logger is not None:
                metrics_logger.start_step(
                    steps[0]["id"],
                    steps[0]["name"],
                    outer_round,
                    payload={
                        "attempt": prepare_attempt,
                        "input_path": str(input_xyz.resolve()),
                        "output_path": str(Path(cfg["task_root"]).resolve()),
                    },
                )
            try:
                manifest["prepare"] = execute_step(
                    {
                        **steps[0],
                        "run": lambda: prepare_dft_jobs(cfg),
                        "success_check": lambda: dft_prepare_success_check(cfg),
                    },
                    state_mgr,
                )
                if metrics_logger is not None:
                    manifest["prepare"]["metrics_attempt"] = prepare_attempt
                    metrics_logger.finish_step(
                        steps[0]["id"],
                        "SUCCESS",
                        updates={
                            "attempt": prepare_attempt,
                            "input_structure_count": _count_extxyz_frames(input_xyz),
                            "output_structure_count": int(manifest["prepare"].get("n_tasks", 0) or 0),
                            "output_path": str(Path(cfg["task_root"]).resolve()),
                        },
                    )
            except KeyboardInterrupt as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(steps[0]["id"], "INTERRUPTED", updates={"attempt": prepare_attempt, "error_message": str(exc)})
                raise
            except SystemExit as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(steps[0]["id"], "INTERRUPTED", updates={"attempt": prepare_attempt, "error_message": str(exc)})
                raise
            except Exception as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(
                        steps[0]["id"],
                        "FAILED",
                        updates={"attempt": prepare_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                raise
        else:
            manifest["prepare"] = prepare_dft_jobs(cfg)
    if base_cfg.get("run", True):
        if resume_index <= 1:
            run_attempt = _manifest_step_attempt(manifest, "run")
            if metrics_logger is not None:
                metrics_logger.start_step(
                    steps[1]["id"],
                    steps[1]["name"],
                    outer_round,
                    payload={
                        "attempt": run_attempt,
                        "input_path": str(Path(cfg["task_root"]).resolve()),
                        "output_path": str(Path(cfg["output_root"]).resolve()),
                    },
                )
            try:
                manifest["run"] = execute_step(
                    {
                        **steps[1],
                        "run": lambda: run_dft_jobs(cfg),
                        "success_check": lambda: dft_run_success_check(cfg),
                    },
                    state_mgr,
                )
                submit_summary_path = Path(cfg["submit_summary_json"]).resolve()
                submit_summary = {}
                if submit_summary_path.exists():
                    try:
                        submit_summary = read_json(submit_summary_path)
                    except Exception:
                        submit_summary = {}
                if metrics_logger is not None:
                    manifest["run"]["metrics_attempt"] = run_attempt
                    metrics_logger.finish_step(
                        steps[1]["id"],
                        "SUCCESS",
                        updates={
                            "attempt": run_attempt,
                            "input_structure_count": int(submit_summary.get("n_total", 0) or 0),
                            "output_structure_count": int(submit_summary.get("n_success", 0) or 0),
                            "output_path": str(Path(cfg["output_root"]).resolve()),
                        },
                    )
            except KeyboardInterrupt as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(steps[1]["id"], "INTERRUPTED", updates={"attempt": run_attempt, "error_message": str(exc)})
                raise
            except SystemExit as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(steps[1]["id"], "INTERRUPTED", updates={"attempt": run_attempt, "error_message": str(exc)})
                raise
            except Exception as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(
                        steps[1]["id"],
                        "FAILED",
                        updates={"attempt": run_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                raise
        else:
            manifest["run"] = run_dft_jobs(cfg)
    if base_cfg.get("collect", True):
        if resume_index <= 2:
            collect_attempt = _manifest_step_attempt(manifest, "collect")
            if metrics_logger is not None:
                metrics_logger.start_step(
                    steps[2]["id"],
                    steps[2]["name"],
                    outer_round,
                    payload={
                        "attempt": collect_attempt,
                        "input_path": str(Path(cfg["output_root"]).resolve()),
                        "output_path": str(Path(cfg["labeled_xyz"]).resolve()),
                    },
                )
            try:
                manifest["collect"] = execute_step(
                    {
                        **steps[2],
                        "run": lambda: collect_dft_results(cfg),
                        "success_check": lambda: dft_collect_success_check(cfg),
                    },
                    state_mgr,
                )
                if metrics_logger is not None:
                    manifest["collect"]["metrics_attempt"] = collect_attempt
                    metrics_logger.log_labeling_metric(
                        {
                            "round_idx": outer_round,
                            "step_id": steps[2]["id"],
                            "labeler_type": "cp2k_collect",
                            "batch_id": cfg["jobid"],
                            "status": "SUCCESS",
                            "input_structure_count": int(manifest["collect"].get("n_total", 0) or 0),
                            "success_structure_count": int(manifest["collect"].get("n_success", 0) or 0),
                            "failed_structure_count": int(manifest["collect"].get("n_failed", 0) or 0),
                            "output_path": str(Path(cfg["labeled_xyz"]).resolve()),
                        }
                    )
                    metrics_logger.finish_step(
                        steps[2]["id"],
                        "SUCCESS",
                        updates={
                            "attempt": collect_attempt,
                            "input_structure_count": int(manifest["collect"].get("n_total", 0) or 0),
                            "output_structure_count": int(manifest["collect"].get("n_success", 0) or 0),
                            "output_path": str(Path(cfg["labeled_xyz"]).resolve()),
                        },
                    )
                plot_payload = plot_outer_round_dft_increment(paths, outer_round)
                if plot_payload is not None:
                    manifest["distribution_plot"] = plot_payload
            except KeyboardInterrupt as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(steps[2]["id"], "INTERRUPTED", updates={"attempt": collect_attempt, "error_message": str(exc)})
                raise
            except SystemExit as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(steps[2]["id"], "INTERRUPTED", updates={"attempt": collect_attempt, "error_message": str(exc)})
                raise
            except Exception as exc:
                if metrics_logger is not None:
                    metrics_logger.finish_step(
                        steps[2]["id"],
                        "FAILED",
                        updates={"attempt": collect_attempt, "error_type": exc.__class__.__name__, "error_message": str(exc)},
                    )
                raise
        else:
            manifest["collect"] = collect_dft_results(cfg)
    state_mgr.mark_workflow_success()
    prepare_payload = manifest.get("prepare", {}) if isinstance(manifest.get("prepare"), dict) else {}
    collect_payload = manifest.get("collect", {}) if isinstance(manifest.get("collect"), dict) else {}
    status_payload = {
        "status": "completed",
        "outer_round": int(outer_round),
        "jobid": cfg["jobid"],
        "task_root": str(Path(cfg["task_root"]).resolve()),
        "output_root": str(Path(cfg["output_root"]).resolve()),
        "submit_summary_json": str(Path(cfg["submit_summary_json"]).resolve()),
        "summary_json": str(Path(cfg["summary_path"]).resolve()),
        "labeled_xyz": str(Path(cfg["labeled_xyz"]).resolve()),
        "failed_xyz": str(Path(cfg["failed_xyz"]).resolve()),
        "n_tasks": int(prepare_payload.get("n_tasks", 0) or 0),
        "n_total": int(collect_payload.get("n_total", 0) or 0),
        "n_success": int(collect_payload.get("n_success", 0) or 0),
        "n_failed": int(collect_payload.get("n_failed", 0) or 0),
        "dft_runtime_resolution": cfg.get("dft_runtime_resolution"),
    }
    write_json(cfg["status_path"], status_payload)
    return str(Path(cfg["status_path"]).resolve())


