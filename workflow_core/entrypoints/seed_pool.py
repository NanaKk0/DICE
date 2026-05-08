from __future__ import annotations

from pathlib import Path

from workflow_core.entrypoints._helpers import resolve_config_path
from workflow_core.pipeline.dp_active_learning_flow import load_dp_active_learning_config
from workflow_core.seed_pool.manager import collect_seed_candidates_from_md, init_seed_pool, update_seed_pool


def run_init_seed_pool_task(cfg, paths, round_id: int) -> None:
    cfg_path = resolve_config_path(cfg, "configs/dp/active_learning_subworkflow.json")
    seed_cfg = load_dp_active_learning_config(cfg_path)
    init_seed_pool(seed_cfg, paths)


def run_update_seed_pool_task(cfg, paths, round_id: int) -> None:
    cfg_path = resolve_config_path(cfg, "configs/dp/active_learning_subworkflow.json")
    workflow_cfg = load_dp_active_learning_config(cfg_path)
    outer_round = int(cfg["outer_round"])
    dp_round = int(cfg["dp_round"])
    sampling_dir = Path(cfg["sampling_dir"]).resolve()
    candidate_file = collect_seed_candidates_from_md(
        workflow_cfg,
        paths,
        outer_round,
        dp_round,
        sampling_dir,
    )
    update_seed_pool(workflow_cfg, paths, outer_round, dp_round, candidate_file)


SEED_POOL_TASKS = {
    "init_seed_pool": run_init_seed_pool_task,
    "update_seed_pool": run_update_seed_pool_task,
}
