#!/usr/bin/env python3
"""Read-only structural inspection for one DICE lifecycle operation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


def _resolve_dice_root(explicit: str | None) -> tuple[Path, str]:
    """Use a selected checkout or the DICE visible to this Python interpreter."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        required = (
            root / "workflow_app" / "entrypoint.py",
            root / "workflow_core" / "lifecycle" / "config.py",
            root / "workflow_runtime" / "lifecycle" / "environment.py",
            root / "workflow_runtime" / "execution" / "profile.py",
        )
        missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
        if missing:
            raise ValueError("--repo-root is not a DICE checkout; missing: " + ", ".join(missing))
        sys.path.insert(0, str(root))
        return root, "explicit-source"
    spec = importlib.util.find_spec("workflow_core")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError(
            "DICE is not importable in this Python environment. Use the Python "
            "environment where DICE is installed, or pass --repo-root PATH to a DICE checkout."
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent, "python-environment"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _selected_path(explicit: str | None, variable: str) -> tuple[Path | None, str | None]:
    if explicit is not None:
        return Path(explicit).resolve(), "cli"
    inherited = os.environ.get(variable)
    if inherited:
        return Path(inherited).resolve(), variable
    return None, None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize one DICE run without preflight, reconciliation, "
            "state mutation, backend execution, or scheduler contact."
        )
    )
    parser.add_argument("run_json", metavar="run.json")
    parser.add_argument("--repo-root", metavar="PATH", help="Use this DICE checkout instead of the installed package")
    parser.add_argument("--environment", dest="environment_path")
    parser.add_argument("--execution", dest="execution_path")
    parser.add_argument("--resume", metavar="BLOCK")
    parser.add_argument("--pretty", action="store_true")
    return parser


def _sampling_summary(launch: Any) -> dict[str, Any]:
    plan = launch.config.sampling_plan
    explicit = bool(getattr(plan, "explicit", False))
    if not explicit:
        return {
            "form": "legacy",
            "round_count": launch.config.round_count,
        }
    effective_rounds = []
    for round_index in range(launch.config.round_count):
        selection = plan.selection_plan_for_round(round_index)
        tasks = plan.tasks_for_round(round_index)
        effective_rounds.append(
            {
                "round": round_index,
                "sampling_roles": [role for role, _group in selection.role_to_group],
                "condition_groups": list(
                    dict.fromkeys(task.condition_group_id for task in tasks)
                ),
                "selection_groups": list(selection.group_ids),
                "role_to_group": [list(item) for item in selection.role_to_group],
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "condition_group_id": task.condition_group_id,
                        "sampling_role_id": task.sampling_role_id,
                        "selection_group_id": task.selection_group_id,
                        "kind": task.kind,
                        "ensemble": task.ensemble,
                        "recipe_binding": task.recipe_binding,
                        "steps": task.steps,
                        "temperature_k": task.temperature_k,
                        "pressure_bar": task.pressure_bar,
                        "program": _jsonable(task.program),
                        "neb": _jsonable(task.neb),
                        "relax": _jsonable(task.relax),
                        "cycles": task.cycles,
                        "gcmc": _jsonable(task.gcmc),
                        "widom": _jsonable(task.widom),
                        "md_mc": _jsonable(task.md_mc),
                        "protocol_request": _jsonable(task.protocol_request),
                        "lammps": None if task.lammps is None else _jsonable(task.lammps.to_payload()),
                        "recipe_inputs": _jsonable(launch.config.raw['sample']['condition_groups'][task.condition_group_id].get('inputs', {})),
                        "planned_mc_attempts": (task.md_mc.planned_attempts(task.steps) if task.md_mc is not None else None),
                    }
                    for task in tasks
                ],
            }
        )
    all_rounds = bool(launch.config.raw["fp"].get("all_rn", False))
    candidate_rounds = (
        tuple(range(launch.config.round_count))
        if all_rounds
        else (launch.config.round_count - 1,)
    )
    terminal = plan.selection_plan_for_rounds(candidate_rounds)
    return {
        "form": "explicit",
        "round_count": launch.config.round_count,
        "effective_rounds": effective_rounds,
        "terminal_fp": {
            "all_rn": all_rounds,
            "candidate_rounds": list(candidate_rounds),
            "selection_groups": list(terminal.group_ids),
            "role_to_group": [list(item) for item in terminal.role_to_group],
        },
    }


def _policy_resolution_summary(launch: Any) -> dict[str, Any]:
    """Expose known-shape ELEMENT_DIRECT decisions without reading artifacts."""

    from workflow_policies.resolution import resolve_element_direct_policy

    resolved: dict[str, Any] = {}
    for coordinate, owner in (("screen.policy", "student"), ("fp.policy", "teacher")):
        section, field = coordinate.split(".")
        requested = str(launch.config.raw[section][field])
        if requested != "ELEMENT_DIRECT":
            continue
        model_shape = str(launch.config.raw[owner]["model"])
        decision = resolve_element_direct_policy(
            requested,
            descriptor_model_shape=model_shape,
        )
        resolved[coordinate] = {
            "requested_policy": decision.requested_policy,
            "effective_policy": decision.effective_policy,
            "fallback_reason": decision.fallback_reason,
            "descriptor_owner": owner,
            "descriptor_model_shape": model_shape,
            "descriptor_readout": not decision.fell_back,
        }
    return resolved


def inspect(arguments: argparse.Namespace) -> dict[str, Any]:
    repository_root, code_source = _resolve_dice_root(getattr(arguments, "repo_root", None))
    from workflow_core.lifecycle.launch import prepare_lifecycle_launch
    from workflow_runtime.execution.profile import load_execution_profile
    from workflow_runtime.lifecycle.environment import load_environment_profile

    run_json = Path(arguments.run_json).resolve()
    environment_path, environment_source = _selected_path(
        arguments.environment_path, "DICE_ENVIRONMENT"
    )
    execution_path, execution_source = _selected_path(
        arguments.execution_path, "DICE_EXECUTION"
    )

    environment = (
        None
        if environment_path is None
        else load_environment_profile(environment_path)
    )
    execution = (
        None
        if execution_path is None
        else load_execution_profile(execution_path)
    )
    launch = prepare_lifecycle_launch(
        run_json,
        environment_path=environment_path,
        execution_path=execution_path,
        resume=arguments.resume,
    )
    blocks = [block.block_id for block in launch.plan.blocks]
    selected_block = arguments.resume or blocks[0]
    selected_index = blocks.index(selected_block)

    return {
        "schema": "dice.operator-inspection",
        "schema_revision": 4,
        "ok": True,
        "scope": "structural-only",
        "repository_root": str(repository_root),
        "code_source": code_source,
        "run_json": str(run_json),
        "run_root": str(launch.run_root),
        "iter": launch.config.iter_index,
        "route": None if launch.route is None else launch.route.kind.value,
        "blocks": blocks,
        "selected_block": selected_block,
        "selected_suffix": blocks[selected_index:],
        "block_operation_roles": launch.block_operation_role_map,
        "sampling": _sampling_summary(launch),
        "policy_resolutions": _policy_resolution_summary(launch),
        "normalized_run_config": launch.config.raw,
        "environment": (
            None
            if environment is None
            else {
                "path": str(environment.path),
                "selection_source": environment_source,
                "profiles": environment.profiles,
                "bindings": environment.bindings,
            }
        ),
        "execution": (
            None
            if execution is None
            else {
                "path": str(execution.path),
                "selection_source": execution_source,
                "targets": execution.targets,
                "target_pools": execution.target_pools,
                "resource_classes": execution.resource_classes,
                "bindings": execution.bindings,
                "digest": execution.digest,
                "historical": execution.historical,
            }
        ),
        "not_checked": [
            "external input existence and native recipe feasibility",
            "environment activation and executable preflight",
            "execution binding feasibility for the selected suffix",
            "lifecycle result dependencies, canonical artifacts, and exact recovery",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    indent = 2 if arguments.pretty else None
    try:
        result = inspect(arguments)
    except Exception as exc:
        result = {
            "schema": "dice.operator-inspection",
            "schema_revision": 4,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, sort_keys=True, indent=indent))
        return 2
    print(json.dumps(_jsonable(result), sort_keys=True, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
