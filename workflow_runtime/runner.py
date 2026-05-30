from __future__ import annotations

from typing import Any

from workflow_runtime.logging.runtime_logging import print_exception_banner
from workflow_runtime.node_contracts import top_level_runtime_node_contract
from workflow_runtime.paths.path_manager import PathManager
from workflow_runtime.plan import RunPlan
from workflow_runtime.preflight import run_preflight_plan
from workflow_runtime.recovery.cleanup import cleanup_recovery_nodes
from workflow_runtime.recovery.recovery_plan import RecoveryNode, plan_recovery
from workflow_runtime.state.markers import MarkerStateManager
from workflow_runtime.state.signal_handler import install_signal_handlers
from workflow_runtime.state.step_runner import execute_step


def _load_resume_config(wf: dict[str, Any]) -> dict[str, Any]:
    payload = dict(wf.get("resume", {}) or {})
    retired_aliases = ("resume_target", "start_node", "from")
    for key in retired_aliases:
        if payload.get(key) not in (None, ""):
            raise ValueError(f"resume.{key} is retired; use resume.target")
    payload.setdefault("enabled", True)
    return payload


def _resume_target(resume_cfg: dict[str, Any]) -> str | None:
    value = resume_cfg.get("target")
    if value not in (None, ""):
        return str(value)
    return None


def run_workflow_plan(plan: RunPlan, *, wf: dict[str, Any], paths: PathManager) -> None:
    round_id = int(plan.round_id)
    resume_cfg = _load_resume_config(wf)
    marker_dir = paths.workflow_round_marker_dir(round_id)
    resumed_existing_round = marker_dir.exists()
    if plan.preflight is not None and not resumed_existing_round:
        run_preflight_plan(plan.preflight)

    steps = [step.to_runtime_step() for step in plan.steps]
    step_states: dict[str, dict] = {}

    state_mgr = MarkerStateManager(
        marker_dir,
        run_name=paths.runtime_root.name,
        round_id=round_id,
        steps=steps,
        backup_partial_outputs=True,
        verify_success_markers=True,
    )

    if bool(resume_cfg.get("enabled", True)):
        recovery_nodes, recovery_node_step_indexes = _recovery_nodes_for_plan(plan.steps, steps, state_mgr)
        recovery_plan = plan_recovery(recovery_nodes, _resume_target(resume_cfg))
        if recovery_plan.start_node is None:
            resume_index = len(steps)
        else:
            cleanup_recovery_nodes(
                recovery_plan.cleanup_nodes,
                teacher_round=recovery_plan.start_node.teacher_round,
                run_root=paths.runtime_root,
            )
            resume_index = recovery_node_step_indexes[recovery_plan.start_node.node_id]
    else:
        resume_index = 0
    current_step_id: dict[str, str | None] = {"value": None}
    install_signal_handlers(state_mgr, lambda: current_step_id["value"])

    print(f"[INFO] Workspace root: {paths.workspace_root}")
    print(f"[INFO] Runtime root:   {paths.runtime_root}")

    state_mgr.mark_workflow_running(
        current_step_index=resume_index if resume_index < len(steps) else None,
        current_step_id=steps[resume_index]["id"] if resume_index < len(steps) else None,
    )

    try:
        for step in steps[resume_index:]:
            step_name = step["name"]
            step_id = step["id"]

            print("\n==============================")
            print(f"RUN STEP: {step_name} (r{round_id})")
            print("==============================")

            try:
                current_step_id["value"] = step_id
                state_mgr.mark_workflow_running(current_step_index=step["index"], current_step_id=step_id)
                step_states[step_name] = {"status": "running", "step_id": step_id}
                execute_step(step, state_mgr)

                on_success = step.get("on_success")
                if callable(on_success):
                    on_success()

                step_states[step_name] = {"status": "completed", "step_id": step_id}
            except KeyboardInterrupt as exc:
                current_step_id["value"] = None
                state_mgr.mark_workflow_interrupted(str(exc) or "Interrupted by KeyboardInterrupt")
                step_states[step_name] = {
                    "status": "interrupted",
                    "step_id": step_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
                raise
            except SystemExit as exc:
                current_step_id["value"] = None
                state_mgr.mark_workflow_interrupted(f"Interrupted by SystemExit({exc.code})")
                step_states[step_name] = {
                    "status": "interrupted",
                    "step_id": step_id,
                    "error_type": "SystemExit",
                    "error_message": str(exc),
                }
                raise
            except Exception as exc:
                current_step_id["value"] = None
                state_mgr.mark_workflow_failed(str(exc))
                step_states[step_name] = {
                    "status": "failed",
                    "step_id": step_id,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
                print_exception_banner(step_name, paths.runtime_root, exc)
                raise

        current_step_id["value"] = None
        state_mgr.mark_workflow_success()
    except Exception:
        raise


def _recovery_nodes_from_steps(steps: list[dict[str, Any]], state_mgr: MarkerStateManager) -> list[RecoveryNode]:
    nodes: list[RecoveryNode] = []
    for step in steps:
        nodes.append(top_level_runtime_node_contract(step, marker_dir=state_mgr.marker_dir).to_recovery_node())
    return nodes


def _recovery_nodes_for_plan(
    plan_steps: list,
    runtime_steps: list[dict[str, Any]],
    state_mgr: MarkerStateManager,
) -> tuple[list[RecoveryNode], dict[str, int]]:
    nodes: list[RecoveryNode] = []
    node_step_indexes: dict[str, int] = {}
    for step_index, (plan_step, runtime_step) in enumerate(zip(plan_steps, runtime_steps)):
        attached_nodes = tuple(getattr(plan_step, "recovery_nodes", ()) or ())
        if attached_nodes:
            for node in attached_nodes:
                nodes.append(node)
                node_step_indexes[str(node.node_id)] = step_index
            continue
        node = top_level_runtime_node_contract(runtime_step, marker_dir=state_mgr.marker_dir).to_recovery_node()
        nodes.append(node)
        node_step_indexes[node.node_id] = step_index
    return nodes, node_step_indexes
