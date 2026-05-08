from __future__ import annotations

from workflow_core.entrypoints._helpers import resolve_config_path
from workflow_core.pipeline.committee_cycle import run_committee_cycle
from workflow_core.pipeline.dp_active_learning_flow import run_dp_active_learning_subworkflow


def run_committee_cycle_task(cfg, paths, round_id: int) -> None:
    cfg_path = resolve_config_path(cfg, "configs/pipeline/committee_cycle.json")
    run_committee_cycle(cfg_path, paths, round_id, metrics_logger=cfg.get("_metrics_logger"))


def run_dp_active_learning_subworkflow_task(cfg, paths, round_id: int) -> None:
    cfg_path = resolve_config_path(cfg, "configs/dp/active_learning_subworkflow.json")
    run_dp_active_learning_subworkflow(cfg_path, paths, round_id)


PIPELINE_TASKS = {
    "committee_cycle": run_committee_cycle_task,
    "dp_active_learning_subworkflow": run_dp_active_learning_subworkflow_task,
}

