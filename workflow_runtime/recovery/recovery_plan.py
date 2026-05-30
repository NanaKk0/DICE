from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from workflow_runtime.state.markers import read_marker_payload


class RecoveryValidationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryNode:
    node_id: str
    teacher_round: int
    marker_path: Path
    artifact_check: Callable[[], bool]
    identity: Mapping[str, Any] = field(default_factory=dict)
    global_invariant_fingerprint: str = ""
    stage_sequence_fingerprint: str = ""
    stage_local_config_fingerprint: str = ""
    artifact_roots: tuple[Path, ...] = ()
    semantic_log_roots: tuple[Path, ...] = ()
    backend_log_roots: tuple[Path, ...] = ()
    marker_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", str(self.node_id))
        object.__setattr__(self, "teacher_round", int(self.teacher_round))
        object.__setattr__(self, "marker_path", Path(self.marker_path).resolve())
        object.__setattr__(self, "identity", dict(self.identity or {}))
        object.__setattr__(self, "artifact_roots", tuple(Path(path).resolve() for path in self.artifact_roots))
        object.__setattr__(self, "semantic_log_roots", tuple(Path(path).resolve() for path in self.semantic_log_roots))
        object.__setattr__(self, "backend_log_roots", tuple(Path(path).resolve() for path in self.backend_log_roots))
        marker_paths = tuple(Path(path).resolve() for path in self.marker_paths) or (Path(self.marker_path).resolve(),)
        object.__setattr__(self, "marker_paths", marker_paths)


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    start_index: int
    start_node: RecoveryNode | None
    cleanup_nodes: tuple[RecoveryNode, ...]


def plan_recovery(nodes: list[RecoveryNode] | tuple[RecoveryNode, ...], resume_target: str | None = None) -> RecoveryPlan:
    planned_nodes = tuple(nodes)
    if not planned_nodes:
        raise RecoveryValidationError("recovery plan has no nodes")
    if resume_target in (None, ""):
        start_index = _first_incomplete_index(planned_nodes)
    else:
        start_index = _target_index(planned_nodes, str(resume_target))
        _validate_completed_prefix(planned_nodes[:start_index])
    if start_index >= len(planned_nodes):
        return RecoveryPlan(start_index=start_index, start_node=None, cleanup_nodes=())
    start_node = planned_nodes[start_index]
    cleanup_nodes = tuple(node for node in planned_nodes[start_index:] if int(node.teacher_round) == int(start_node.teacher_round))
    return RecoveryPlan(start_index=start_index, start_node=start_node, cleanup_nodes=cleanup_nodes)


def validate_completed_recovery_node(node: RecoveryNode) -> None:
    payload = _validated_marker_payload(node)
    _validate_node_identity(node, payload)
    _validate_global_invariant(node, payload)
    _validate_stage_sequence(node, payload)
    if not _artifact_check_passes(node):
        raise RecoveryValidationError(f"completed stage canonical artifact check failed for {node.node_id}")


def _target_index(nodes: tuple[RecoveryNode, ...], target: str) -> int:
    for index, node in enumerate(nodes):
        if node.node_id == target:
            return index
    raise RecoveryValidationError(f"unknown resume target: {target}")


def _first_incomplete_index(nodes: tuple[RecoveryNode, ...]) -> int:
    for index, node in enumerate(nodes):
        if not _node_is_complete(node):
            return index
    return len(nodes)


def _validate_completed_prefix(nodes: tuple[RecoveryNode, ...]) -> None:
    for node in nodes:
        validate_completed_recovery_node(node)


def _node_is_complete(node: RecoveryNode) -> bool:
    try:
        validate_completed_recovery_node(node)
        return True
    except RecoveryValidationError:
        return False


def _validated_marker_payload(node: RecoveryNode) -> dict[str, Any]:
    if not node.marker_path.exists():
        raise RecoveryValidationError(f"missing success marker for {node.node_id}")
    if node.marker_path.suffix != ".success":
        raise RecoveryValidationError(
            f"invalid marker suffix for {node.node_id}: marker_path expected .success got {node.marker_path.suffix!r}"
        )
    payload = read_marker_payload(node.marker_path)
    status = str(payload.get("status", "")).upper()
    marker = str(payload.get("marker", "")).lower()
    if marker != "success":
        raise RecoveryValidationError(
            f"invalid success marker payload for {node.node_id}: marker expected 'success' got {marker!r}"
        )
    if status != "SUCCESS":
        raise RecoveryValidationError(
            f"invalid success marker payload for {node.node_id}: status expected 'SUCCESS' got {status!r}"
        )
    if str(payload.get("node_id", node.node_id)) != node.node_id:
        raise RecoveryValidationError(f"stage identity mismatch for {node.node_id}: node_id")
    return payload


def _validate_node_identity(node: RecoveryNode, payload: dict[str, Any]) -> None:
    payload_identity = dict(payload.get("identity", {}) or {})
    for key, expected in dict(node.identity or {}).items():
        actual = payload_identity.get(key, payload.get(key))
        if str(actual) != str(expected):
            raise RecoveryValidationError(
                f"stage identity mismatch for {node.node_id}: {key} current={expected!r} marker={actual!r}"
            )


def _validate_global_invariant(node: RecoveryNode, payload: dict[str, Any]) -> None:
    expected = str(node.global_invariant_fingerprint or "")
    if not expected:
        return
    actual = str(payload.get("global_invariant_fingerprint", ""))
    if not actual:
        return
    if actual != expected:
        raise RecoveryValidationError(
            f"global invariant mismatch for {node.node_id}: current={expected!r} marker={actual!r}"
        )


def _validate_stage_sequence(node: RecoveryNode, payload: dict[str, Any]) -> None:
    expected = str(node.stage_sequence_fingerprint or "")
    if not expected:
        return
    actual = str(payload.get("stage_sequence_fingerprint", ""))
    if not actual:
        return
    if actual != expected:
        raise RecoveryValidationError(
            f"stage sequence mismatch for {node.node_id}: current={expected!r} marker={actual!r}"
        )


def _artifact_check_passes(node: RecoveryNode) -> bool:
    try:
        return bool(node.artifact_check())
    except Exception:
        return False
