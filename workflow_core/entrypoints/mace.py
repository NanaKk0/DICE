from __future__ import annotations

from workflow_core.mace.benchmark import run_task as run_benchmark_task
from workflow_core.mace.committee_finetune import run_finetune
from workflow_core.mace.foundation import run_train_mace


def run_benchmark(cfg, paths, round_id: int) -> None:
    run_benchmark_task(cfg, paths, round_id)


def run_train_mace_task(cfg, paths, round_id: int) -> None:
    run_train_mace(cfg, paths, round_id)


def run_finetune_mace_task(cfg, paths, round_id: int) -> None:
    run_finetune(cfg, paths, round_id)


MACE_TASKS = {
    "benchmark": run_benchmark,
    "train_mace": run_train_mace_task,
    "finetune_mace": run_finetune_mace_task,
}

