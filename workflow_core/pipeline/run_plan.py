from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.pipeline.preflight_plan import build_outer_round_preflight_plan
from workflow_core.pipeline.top_level_steps import build_top_level_steps
from workflow_runtime.paths.path_manager import PathManager
from workflow_runtime.plan import RunPlan


def build_top_level_run_plan(workflow_path: Path, wf: dict[str, Any], paths: PathManager) -> RunPlan:
    del workflow_path
    round_id = int(wf["round"])
    return RunPlan(
        round_id=round_id,
        steps=build_top_level_steps(wf, paths, round_id),
        preflight=build_outer_round_preflight_plan(wf, paths, round_id),
    )
