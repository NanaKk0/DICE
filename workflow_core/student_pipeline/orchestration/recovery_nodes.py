from __future__ import annotations

from typing import Any

from workflow_core.student_pipeline.orchestration.runtime_node_contracts import student_stage_runtime_node_contract
from workflow_runtime.recovery.recovery_plan import RecoveryNode


def build_student_stage_recovery_nodes(runtime_state: dict[str, Any]) -> tuple[RecoveryNode, ...]:
    """Build run-level recovery nodes for concrete student pipeline stages."""

    nodes: list[RecoveryNode] = []
    for invocation in list(runtime_state.get("invocations", []) or []):
        for stage in list(invocation.get("stages", []) or []):
            nodes.append(student_stage_runtime_node_contract(runtime_state, stage).to_recovery_node())
    return tuple(nodes)
