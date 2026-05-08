from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_core.mace.config import resolve_committee_cycle_config
from workflow_core.mace.committee_finetune import run_finetune
from workflow_core.mace.training_data_policy import resolve_mace_finetune_inputs
from workflow_core.pipeline.dp_active_learning_flow import run_dp_active_learning_subworkflow
from workflow_core.common.io import ensure_dir, read_json, write_json
from workflow_runtime.state.run_state import RunStateManager
from workflow_runtime.state.step_runner import execute_step


def load_committee_cycle_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = read_json(config_path)
    return resolve_committee_cycle_config(payload)


def committee_finetune_success_check(paths, outer_round: int, cfg: dict[str, Any]) -> bool:
    work_root = paths.mace_committee_dir(outer_round)
    if not work_root.exists():
        return False
    finetune_config_path = Path(cfg["mace_finetune"]["config"]).resolve()
    finetune_cfg = read_json(finetune_config_path)
    teacher_num = int(finetune_cfg.get("teacher_num", 0))
    if teacher_num <= 0:
        return False
    for teacher_index in range(1, teacher_num + 1):
        model_path = work_root / f"teacher_{teacher_index:02d}" / f"mace_teacher_{teacher_index:02d}.model"
        if not model_path.exists() or model_path.stat().st_size == 0:
            return False
    return True


def committee_dp_subworkflow_success_check(paths, outer_round: int) -> bool:
    status_path = paths.dp_subworkflow_status_json(outer_round)
    if not status_path.exists():
        return False
    try:
        payload = read_json(status_path)
    except Exception:
        return False
    return bool(payload.get("stopped"))


def run_committee_cycle(cfg_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    cfg = load_committee_cycle_config(cfg_path)

    ensure_dir(paths.committee_cycle_dir(outer_round))
    ensure_dir(paths.committee_cycle_results_dir(outer_round))
    result_dir = paths.committee_cycle_results_dir(outer_round)
    training_input = resolve_mace_finetune_inputs(paths, outer_round, cfg)

    state_mgr = RunStateManager(
        result_dir / "workflow_state.json",
        result_dir / "markers",
        backup_partial_outputs=False,
        verify_success_markers=True,
    )
    steps = [
        {
            "id": f"r{outer_round}_finetune",
            "name": "finetune",
            "round": outer_round,
            "index": 0,
            "output_dir": str(paths.mace_committee_dir(outer_round).resolve()),
            "backup_partial_outputs": False,
        },
        {
            "id": f"r{outer_round}_dp_subworkflow",
            "name": "dp_subworkflow",
            "round": outer_round,
            "index": 1,
            "output_dir": str(paths.dp_subworkflow_results_dir(outer_round).resolve()),
            "backup_partial_outputs": False,
        },
    ]
    if state_mgr.exists():
        state_mgr.load()
    else:
        state_mgr.init_state(run_name=f"committee_cycle_r{outer_round}", round_id=outer_round, steps=steps)
    state_mgr.reconcile(
        success_checks={
            f"r{outer_round}_finetune": lambda: committee_finetune_success_check(paths, outer_round, cfg),
            f"r{outer_round}_dp_subworkflow": lambda: committee_dp_subworkflow_success_check(paths, outer_round),
        }
    )
    resume_index = state_mgr.get_resume_index()
    state_mgr.mark_workflow_running(
        current_step_index=resume_index if resume_index < len(steps) else None,
        current_step_id=steps[resume_index]["id"] if resume_index < len(steps) else None,
    )

    finetune_context = {
        "training_input": training_input,
        "merged_xyz_override": str(paths.committee_cycle_finetune_merged_xyz(outer_round).resolve()),
        "reuse_existing_committee": bool(cfg["mace_finetune"].get("reuse_existing_committee", True)),
    }

    if resume_index <= 0:
        finetune_step = {
            **steps[0],
            "run": lambda: run_finetune({"config": cfg["mace_finetune"]["config"]}, paths, outer_round, context=finetune_context),
            "success_check": lambda: committee_finetune_success_check(paths, outer_round, cfg),
        }
        if metrics_logger is not None:
            attempt = int(state_mgr.get_step_record(finetune_step["id"]).get("attempt", 0)) + 1
            metrics_logger.start_step(
                finetune_step["id"],
                finetune_step["name"],
                outer_round,
                payload={
                    "attempt": attempt,
                    "input_path": str(paths.committee_cycle_finetune_merged_xyz(outer_round).resolve()),
                    "output_path": str(paths.mace_committee_dir(outer_round).resolve()),
                },
            )
        try:
            execute_step(finetune_step, state_mgr)
            if metrics_logger is not None:
                metrics_logger.finish_step(
                    finetune_step["id"],
                    "SUCCESS",
                    updates={
                        "input_structure_count": None,
                        "output_path": str(paths.mace_committee_dir(outer_round).resolve()),
                        "model_path": str(paths.mace_committee_dir(outer_round).resolve()),
                    },
                )
        except KeyboardInterrupt as exc:
            if metrics_logger is not None:
                metrics_logger.finish_step(finetune_step["id"], "INTERRUPTED", updates={"error_message": str(exc)})
            raise
        except SystemExit as exc:
            if metrics_logger is not None:
                metrics_logger.finish_step(finetune_step["id"], "INTERRUPTED", updates={"error_message": str(exc)})
            raise
        except Exception as exc:
            if metrics_logger is not None:
                metrics_logger.finish_step(
                    finetune_step["id"],
                    "FAILED",
                    updates={"error_type": exc.__class__.__name__, "error_message": str(exc)},
                )
            raise

    if resume_index <= 1:
        dp_step = {
            **steps[1],
            "run": lambda: run_dp_active_learning_subworkflow(
                cfg["dp_subworkflow"]["config"],
                paths,
                outer_round,
                metrics_logger=metrics_logger,
            ),
            "success_check": lambda: committee_dp_subworkflow_success_check(paths, outer_round),
        }
        if metrics_logger is not None:
            attempt = int(state_mgr.get_step_record(dp_step["id"]).get("attempt", 0)) + 1
            metrics_logger.start_step(
                dp_step["id"],
                dp_step["name"],
                outer_round,
                payload={
                    "attempt": attempt,
                    "output_path": str(paths.dp_subworkflow_results_dir(outer_round).resolve()),
                },
            )
        try:
            dp_status_path = execute_step(dp_step, state_mgr)
            if metrics_logger is not None:
                metrics_logger.finish_step(
                    dp_step["id"],
                    "SUCCESS",
                    updates={"output_path": str(paths.dp_subworkflow_results_dir(outer_round).resolve())},
                )
        except KeyboardInterrupt as exc:
            if metrics_logger is not None:
                metrics_logger.finish_step(dp_step["id"], "INTERRUPTED", updates={"error_message": str(exc)})
            raise
        except SystemExit as exc:
            if metrics_logger is not None:
                metrics_logger.finish_step(dp_step["id"], "INTERRUPTED", updates={"error_message": str(exc)})
            raise
        except Exception as exc:
            if metrics_logger is not None:
                metrics_logger.finish_step(
                    dp_step["id"],
                    "FAILED",
                    updates={"error_type": exc.__class__.__name__, "error_message": str(exc)},
                )
            raise
    else:
        dp_status_path = str(paths.dp_subworkflow_status_json(outer_round).resolve())

    state_mgr.mark_workflow_success()
    dp_status = read_json(Path(dp_status_path))
    status = {
        "status": "completed",
        "outer_round": int(outer_round),
        "finetune_input_merged_xyz": str(paths.committee_cycle_finetune_merged_xyz(outer_round).resolve()),
        "mace_committee_dir": str(paths.mace_committee_dir(outer_round).resolve()),
        "dp_subworkflow_status_json": str(Path(dp_status_path).resolve()),
        "final_dp_model_path": dp_status.get("final_dp_model_path"),
        "stopped": bool(dp_status.get("stopped")),
        "stop_reason": dp_status.get("stop_reason"),
    }
    status_path = paths.committee_cycle_status_json(outer_round)
    write_json(status_path, status)
    return str(status_path.resolve())


