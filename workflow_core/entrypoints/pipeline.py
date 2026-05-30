from __future__ import annotations

from workflow_core.entrypoints._helpers import resolve_config_path


def run_committee_cycle_task(cfg, paths, round_id: int) -> None:
    from workflow_core.committee_cycle.entrypoint import run_committee_cycle

    cfg_path = resolve_config_path(cfg, "configs/pipelines/committee_cycle/pipeline.json")
    run_committee_cycle(cfg_path, paths, round_id, metrics_logger=cfg.get("_metrics_logger"))


def run_teacher_foundation_train_task(cfg, paths, round_id: int) -> None:
    from workflow_core.teacher.foundation_train import run_foundation_train

    return run_foundation_train(cfg, paths, round_id)


def run_first_principles_task(cfg, paths, round_id: int) -> None:
    from workflow_core.first_principles.workflow import run_first_principles_workflow

    cfg_path = cfg.get("config")
    run_first_principles_workflow(cfg_path, paths, round_id, metrics_logger=cfg.get("_metrics_logger"))


PIPELINE_TASKS = {
    "teacher_foundation_train": run_teacher_foundation_train_task,
    "committee_cycle": run_committee_cycle_task,
    "first_principles": run_first_principles_task,
}
