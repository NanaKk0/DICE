from __future__ import annotations

from pathlib import Path

from workflow_core.committee_cycle.config import load_committee_cycle_config
from workflow_core.committee_cycle.plan import build_committee_cycle_plan
from workflow_runtime.plan import RunPlan
from workflow_runtime.runner import run_workflow_plan


def run_committee_cycle(cfg_path: str | Path, paths, outer_round: int, metrics_logger=None) -> str:
    """Compatibility wrapper; target mode should enter through top-level plan expansion."""

    del metrics_logger
    cfg = load_committee_cycle_config(cfg_path)
    round_id = int(outer_round)
    run_workflow_plan(
        RunPlan(
            round_id=round_id,
            steps=build_committee_cycle_plan(cfg, paths, round_id),
        ),
        wf={"round": round_id, "resume": {"enabled": True}},
        paths=paths,
    )
    return str(paths.committee_cycle_artifacts_dir(round_id).resolve())
