from __future__ import annotations

from workflow_core.dp.sample_committee import run_dp_sample_committee
from workflow_core.dp.train_committee import run_dp_train_committee


def run_dp_sample_committee_task(cfg, paths, round_id: int) -> None:
    cfg_path = cfg.get("config", "configs/dp/sample_committee.json")
    run_dp_sample_committee(cfg_path, paths, round_id)


def run_dp_train_committee_task(cfg, paths, round_id: int) -> None:
    cfg_path = cfg.get("config", "configs/dp/student_committee.json")
    run_dp_train_committee(cfg_path, paths, round_id)


DP_TASKS = {
    "train_dp_committee": run_dp_train_committee_task,
    "dp_sample_committee": run_dp_sample_committee_task,
}
