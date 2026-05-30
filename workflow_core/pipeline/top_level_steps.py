from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from workflow_core.committee_cycle.config import load_committee_cycle_config
from workflow_core.committee_cycle.plan import build_committee_cycle_plan
from workflow_core.pipeline.top_level_observers import build_top_level_step_observer
from workflow_runtime.artifacts.checks import (
    first_principles_complete,
    student_subworkflow_complete,
    teacher_committee_finetune_complete,
    teacher_foundation_train_complete,
)
from workflow_runtime.plan import StepPlan
from workflow_runtime.paths.path_manager import PathManager


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def teacher_foundation_train_success_check(paths: PathManager, round_id: int) -> bool:
    return teacher_foundation_train_complete(paths.teacher_foundation_train_artifacts_dir(round_id))


def committee_cycle_success_check(paths: PathManager, round_id: int) -> bool:
    return teacher_committee_finetune_complete(
        paths.teacher_committee_finetune_artifacts_dir(round_id),
    ) and student_subworkflow_complete(paths.student_subworkflow_artifacts_dir(round_id))


def first_principles_success_check(paths: PathManager, round_id: int) -> bool:
    return first_principles_complete(paths.first_principles_artifacts_dir(round_id))


def _top_level_output_dir(step_name: str, paths: PathManager, round_id: int) -> Path | None:
    if step_name == "teacher_foundation_train":
        return paths.teacher_foundation_train_artifacts_dir(round_id)
    if step_name == "committee_cycle":
        return paths.committee_cycle_artifacts_dir(round_id)
    if step_name == "first_principles":
        return paths.first_principles_artifacts_dir(round_id)
    return None


def _raise_retired_step_id(step_name: str) -> None:
    replacements = {
        "train_mace": "teacher_foundation_train",
        "dft": "first_principles",
    }
    if step_name in replacements:
        raise ValueError(
            f"Top-level step id {step_name!r} is retired in target-mode; "
            f"use {replacements[step_name]!r} instead."
        )


def build_top_level_steps(
    wf: dict[str, Any],
    paths: PathManager,
    round_id: int,
) -> list[StepPlan]:
    success_checks: dict[str, Callable[[], bool]] = {
        "teacher_foundation_train": lambda: teacher_foundation_train_success_check(paths, round_id),
        "committee_cycle": lambda: committee_cycle_success_check(paths, round_id),
        "first_principles": lambda: first_principles_success_check(paths, round_id),
    }
    backup_flags = {
        "teacher_foundation_train": False,
        "committee_cycle": False,
        "first_principles": False,
    }
    steps: list[StepPlan] = []
    for step_name in wf.get("pipeline", []):
        _raise_retired_step_id(step_name)
        cfg = dict(wf.get(step_name, {}))
        if isinstance(wf.get("environment_profile_payload"), dict):
            cfg["environment_profile"] = dict(wf["environment_profile_payload"])
        if step_name == "committee_cycle":
            cfg_path = Path(str(cfg.get("config", "configs/pipelines/committee_cycle/pipeline.json"))).resolve()
            committee_cfg = load_committee_cycle_config(cfg_path)
            if isinstance(cfg.get("environment_profile"), dict):
                committee_cfg["environment_profile"] = dict(cfg["environment_profile"])
            expanded = build_committee_cycle_plan(committee_cfg, paths, round_id)
            for offset, substep in enumerate(expanded):
                substep.index = len(steps) + offset
            steps.extend(expanded)
            continue
        from workflow_core.entrypoints.registry import TASK_REGISTRY

        if step_name not in TASK_REGISTRY:
            raise ValueError(f"Unknown step {step_name}")
        output_dir = _top_level_output_dir(step_name, paths, round_id)
        steps.append(
            StepPlan(
                id=f"r{round_id}_{step_name}",
                name=step_name,
                round_id=round_id,
                index=len(steps),
                output_dir=output_dir,
                backup_partial_outputs=backup_flags.get(step_name, False),
                run=lambda step_name=step_name, cfg=cfg: TASK_REGISTRY[step_name](cfg, paths, round_id),
                success_check=success_checks.get(step_name),
                on_success=build_top_level_step_observer(step_name, paths, round_id),
                artifact_contract_id=step_name,
                artifact_roots=() if output_dir is None else (output_dir,),
            )
        )
    return steps
