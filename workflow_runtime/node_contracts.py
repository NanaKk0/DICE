from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from workflow_runtime.artifacts.registry import artifact_check_for

MARKER_SUFFIXES = ("running", "success", "failed", "interrupted")


def _always_complete() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class RuntimeNodeContract:
    node_id: str
    teacher_round: int
    marker_paths_by_suffix: Mapping[str, str | Path]
    artifact_check: Callable[[], bool] = _always_complete
    identity: Mapping[str, Any] = field(default_factory=dict)
    artifact_contract_id: str = ""
    global_invariant_fingerprint: str = ""
    stage_sequence_fingerprint: str = ""
    stage_local_config_fingerprint: str = ""
    artifact_roots: tuple[str | Path, ...] = ()
    semantic_log_roots: tuple[str | Path, ...] = ()
    backend_log_roots: tuple[str | Path, ...] = ()

    def __post_init__(self) -> None:
        marker_paths = {
            str(suffix): Path(path).resolve()
            for suffix, path in dict(self.marker_paths_by_suffix or {}).items()
        }
        if "success" not in marker_paths:
            raise ValueError(f"runtime node contract for {self.node_id!r} is missing a success marker path")
        object.__setattr__(self, "node_id", str(self.node_id))
        object.__setattr__(self, "teacher_round", int(self.teacher_round))
        object.__setattr__(self, "marker_paths_by_suffix", marker_paths)
        object.__setattr__(self, "identity", dict(self.identity or {}))
        object.__setattr__(
            self,
            "artifact_roots",
            tuple(Path(path).resolve() for path in self.artifact_roots),
        )
        object.__setattr__(
            self,
            "semantic_log_roots",
            tuple(Path(path).resolve() for path in self.semantic_log_roots),
        )
        object.__setattr__(
            self,
            "backend_log_roots",
            tuple(Path(path).resolve() for path in self.backend_log_roots),
        )
        object.__setattr__(self, "artifact_contract_id", str(self.artifact_contract_id or self.node_id))
        object.__setattr__(self, "global_invariant_fingerprint", str(self.global_invariant_fingerprint or ""))
        object.__setattr__(self, "stage_sequence_fingerprint", str(self.stage_sequence_fingerprint or ""))
        object.__setattr__(self, "stage_local_config_fingerprint", str(self.stage_local_config_fingerprint or ""))

    @property
    def marker_path(self) -> Path:
        return self.marker_path_for("success")

    @property
    def marker_paths(self) -> tuple[Path, ...]:
        paths = dict(self.marker_paths_by_suffix)
        return tuple(paths[suffix] for suffix in MARKER_SUFFIXES if suffix in paths)

    def marker_path_for(self, suffix: str) -> Path:
        try:
            return dict(self.marker_paths_by_suffix)[str(suffix)]
        except KeyError as exc:
            raise ValueError(f"runtime node contract for {self.node_id!r} has no {suffix!r} marker path") from exc

    def marker_payload(self, *, suffix: str, status: str, timestamp: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "teacher_round": self.teacher_round,
            "status": str(status),
            "marker": str(suffix),
            "identity": dict(self.identity),
            "artifact_contract_id": self.artifact_contract_id,
            "global_invariant_fingerprint": self.global_invariant_fingerprint,
            "stage_sequence_fingerprint": self.stage_sequence_fingerprint,
            "timestamp": str(timestamp),
        }
        if self.stage_local_config_fingerprint:
            payload["stage_local_config_fingerprint"] = self.stage_local_config_fingerprint
        for key, value in self.identity.items():
            if key not in payload or key == "teacher_round":
                payload[key] = value
        return payload

    def to_recovery_node(self):
        from workflow_runtime.recovery.recovery_plan import RecoveryNode

        return RecoveryNode(
            node_id=self.node_id,
            teacher_round=self.teacher_round,
            marker_path=self.marker_path,
            artifact_check=self.artifact_check,
            identity=dict(self.identity),
            global_invariant_fingerprint=self.global_invariant_fingerprint,
            stage_sequence_fingerprint=self.stage_sequence_fingerprint,
            stage_local_config_fingerprint=self.stage_local_config_fingerprint,
            artifact_roots=self.artifact_roots,
            semantic_log_roots=self.semantic_log_roots,
            backend_log_roots=self.backend_log_roots,
            marker_paths=self.marker_paths,
        )


def runtime_node_marker_path(marker_dir: str | Path, node_id: str, suffix: str) -> Path:
    return (Path(marker_dir).resolve() / f"{str(node_id)}.{str(suffix)}").resolve()


def runtime_node_marker_paths(marker_dir: str | Path, node_id: str) -> dict[str, Path]:
    return {suffix: runtime_node_marker_path(marker_dir, node_id, suffix) for suffix in MARKER_SUFFIXES}


def top_level_runtime_node_contract(step: Mapping[str, Any], *, marker_dir: str | Path) -> RuntimeNodeContract:
    node_id = str(step["id"])
    success_check = step.get("success_check")
    artifact_roots = tuple(step.get("artifact_roots", ()) or ())
    if not artifact_roots and step.get("output_dir") is not None:
        artifact_roots = (step["output_dir"],)
    artifact_contract_id = str(step.get("artifact_contract_id", step.get("name", node_id)) or node_id)
    if callable(success_check):
        artifact_check = success_check
    elif artifact_roots:
        registry_checks = tuple(artifact_check_for(artifact_contract_id, root) for root in artifact_roots)
        artifact_check = lambda registry_checks=registry_checks: all(check() for check in registry_checks)
    else:
        artifact_check = _always_complete
    return RuntimeNodeContract(
        node_id=node_id,
        teacher_round=int(step.get("teacher_round", step.get("round", 0))),
        marker_paths_by_suffix=runtime_node_marker_paths(marker_dir, node_id),
        artifact_check=artifact_check,
        identity=dict(step.get("identity", {}) or {}),
        artifact_contract_id=artifact_contract_id,
        global_invariant_fingerprint=str(step.get("global_invariant_fingerprint", "")),
        stage_sequence_fingerprint=str(step.get("stage_sequence_fingerprint", "")),
        stage_local_config_fingerprint=str(step.get("stage_local_config_fingerprint", "")),
        artifact_roots=artifact_roots,
        semantic_log_roots=tuple(step.get("semantic_log_roots", ()) or ()),
        backend_log_roots=tuple(step.get("backend_log_roots", ()) or ()),
    )
