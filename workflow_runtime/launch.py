from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from workflow_core.pipeline.run_plan import build_top_level_run_plan
from workflow_core.pipeline.environment_roles import selected_environment_roles_for_workflow
from workflow_core.common.io import write_json
from workflow_runtime.environment_profile import (
    environment_profile_payload,
    load_environment_profile,
    summarize_environment_profile,
    validate_environment_bindings,
)
from workflow_runtime.paths.path_manager import PathManager
from workflow_runtime.plan import RunPlan
from workflow_runtime.runner import run_workflow_plan


@dataclass(frozen=True, slots=True)
class WorkflowLaunch:
    workflow_path: Path
    workflow: dict[str, Any]
    paths: PathManager
    snapshot_path: Path
    plan: RunPlan | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(Path(path).resolve(), encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return dict(payload or {})


def build_runtime_root(workspace_root: str | Path, workflow_path: str | Path, wf: dict[str, Any]) -> Path:
    workflow = dict(wf or {})
    explicit_run_name = workflow.get("run_name")
    if explicit_run_name:
        safe_name = str(explicit_run_name).strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{Path(workflow_path).resolve().stem}_{timestamp}"
    return (Path(workspace_root).resolve() / "runs" / safe_name).resolve()


def initialize_runtime_layout(paths: PathManager) -> None:
    for directory in (
        paths.runtime_root,
        paths.artifacts,
        paths.runtime,
        paths.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_reference(raw_path: Any, *, anchor: Path, config_root: Path | None) -> Any:
    if raw_path in (None, ""):
        return raw_path
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return str(candidate.resolve())
    base = config_root if config_root is not None else anchor.parent
    return str((base / candidate).resolve())


def _resolved_workflow_payload(workflow_path: Path, workflow: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(workflow or {}))
    raw_config_root = resolved.get("config_root")
    config_root = None
    if raw_config_root not in (None, ""):
        config_root = Path(_resolve_reference(raw_config_root, anchor=workflow_path, config_root=None))
        resolved["config_root"] = str(config_root)

    for step_name in list(resolved.get("pipeline", []) or []):
        step_payload = resolved.get(step_name)
        if not isinstance(step_payload, dict):
            continue
        if "config" in step_payload:
            step_payload["config"] = _resolve_reference(step_payload["config"], anchor=workflow_path, config_root=config_root)

    if resolved.get("environment_profile") not in (None, ""):
        resolved["environment_profile"] = _resolve_reference(
            resolved["environment_profile"],
            anchor=workflow_path,
            config_root=config_root,
        )
        profile = load_environment_profile(resolved["environment_profile"])
        validate_environment_bindings(profile, selected_environment_roles_for_workflow(resolved))
        resolved["environment_profile_payload"] = environment_profile_payload(profile)
    return resolved


def _write_launch_snapshot(
    *,
    workflow_path: Path,
    workflow: dict[str, Any],
    paths: PathManager,
    snapshot_path: Path,
) -> Path:
    environment_profile_payload = None
    if workflow.get("environment_profile") not in (None, ""):
        profile = load_environment_profile(workflow["environment_profile"])
        environment_profile_payload = {
            "path": str(profile.path),
            "summary": summarize_environment_profile(profile),
        }
    payload = {
        "schema": "workflow_launch_snapshot.v1",
        "workflow_path": str(workflow_path),
        "workspace_root": str(paths.workspace_root),
        "runtime_root": str(paths.runtime_root),
        "round": int(workflow["round"]),
        "pipeline": list(workflow.get("pipeline", []) or []),
        "workflow": _snapshot_workflow_payload(workflow),
    }
    if environment_profile_payload is not None:
        payload["environment_profile"] = environment_profile_payload
    write_json(snapshot_path, payload)
    return snapshot_path


def _snapshot_workflow_payload(workflow: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(workflow or {}))
    payload.pop("environment_profile_payload", None)
    return payload


def prepare_workflow_launch(
    workflow_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> WorkflowLaunch:
    resolved_workflow_path = Path(workflow_path).resolve()
    raw_workflow = load_yaml(resolved_workflow_path)
    workflow = _resolved_workflow_payload(resolved_workflow_path, raw_workflow)
    launch_workspace_root = Path(workspace_root).resolve() if workspace_root is not None else resolved_workflow_path.parent.resolve()
    launch_runtime_root = (
        Path(runtime_root).resolve()
        if runtime_root is not None
        else build_runtime_root(launch_workspace_root, resolved_workflow_path, workflow)
    )
    paths = PathManager(workspace_root=launch_workspace_root, runtime_root=launch_runtime_root)
    initialize_runtime_layout(paths)
    snapshot_path = paths.workflow_round_snapshot_dir(int(workflow["round"])) / "workflow_launch_snapshot.json"
    _write_launch_snapshot(
        workflow_path=resolved_workflow_path,
        workflow=workflow,
        paths=paths,
        snapshot_path=snapshot_path,
    )
    return WorkflowLaunch(
        workflow_path=resolved_workflow_path,
        workflow=workflow,
        paths=paths,
        snapshot_path=snapshot_path,
    )


def launch_workflow(
    workflow_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> WorkflowLaunch:
    launch = prepare_workflow_launch(workflow_path, workspace_root=workspace_root, runtime_root=runtime_root)
    plan = build_top_level_run_plan(launch.workflow_path, launch.workflow, launch.paths)
    run_workflow_plan(plan, wf=launch.workflow, paths=launch.paths)
    return WorkflowLaunch(
        workflow_path=launch.workflow_path,
        workflow=launch.workflow,
        paths=launch.paths,
        snapshot_path=launch.snapshot_path,
        plan=plan,
    )
