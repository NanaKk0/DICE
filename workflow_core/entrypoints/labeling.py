from __future__ import annotations

from workflow_core.labeling.task import run_label


def run_label_task(cfg, paths, round_id: int) -> None:
    run_label(cfg, paths, round_id)


LABELING_TASKS = {
    "label": run_label_task,
}

