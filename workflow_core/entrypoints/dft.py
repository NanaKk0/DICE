from __future__ import annotations

from workflow_core.dft.workflow import run_dft_workflow
from workflow_core.entrypoints._helpers import resolve_config_path


def run_dft_task(cfg, paths, round_id: int) -> None:
    cfg_path = resolve_config_path(cfg, "configs/dft/submit.json")
    run_dft_workflow(cfg_path, paths, round_id, metrics_logger=cfg.get("_metrics_logger"))


DFT_TASKS = {
    "dft": run_dft_task,
}

