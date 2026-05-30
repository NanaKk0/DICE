from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class StepPlan:
    id: str
    name: str
    round_id: int
    index: int
    run: Callable[[], Any]
    success_check: Callable[[], bool] | None = None
    output_dir: str | Path | None = None
    backup_partial_outputs: bool = False
    prepare: Callable[[], Any] | None = None
    on_success: Callable[[], Any] | None = None
    parent_step: str | None = None
    visibility: str = "user"
    identity: dict[str, Any] = field(default_factory=dict)
    artifact_contract_id: str = ""
    global_invariant_fingerprint: str = ""
    stage_sequence_fingerprint: str = ""
    stage_local_config_fingerprint: str = ""
    artifact_roots: tuple[str | Path, ...] = field(default_factory=tuple)
    semantic_log_roots: tuple[str | Path, ...] = field(default_factory=tuple)
    backend_log_roots: tuple[str | Path, ...] = field(default_factory=tuple)
    recovery_nodes: tuple[Any, ...] = field(default_factory=tuple)

    def to_runtime_step(self) -> dict[str, Any]:
        artifact_roots = tuple(self.artifact_roots)
        if not artifact_roots and self.output_dir is not None:
            artifact_roots = (self.output_dir,)
        return {
            "id": self.id,
            "name": self.name,
            "round": int(self.round_id),
            "index": int(self.index),
            "run": self.run,
            "success_check": self.success_check,
            "output_dir": None if self.output_dir is None else str(Path(self.output_dir).resolve()),
            "backup_partial_outputs": bool(self.backup_partial_outputs),
            "prepare": self.prepare,
            "on_success": self.on_success,
            "parent_step": self.parent_step,
            "visibility": self.visibility,
            "identity": dict(self.identity or {}),
            "artifact_contract_id": self.artifact_contract_id or self.name,
            "global_invariant_fingerprint": self.global_invariant_fingerprint,
            "stage_sequence_fingerprint": self.stage_sequence_fingerprint,
            "stage_local_config_fingerprint": self.stage_local_config_fingerprint,
            "artifact_roots": [str(Path(path).resolve()) for path in artifact_roots],
            "semantic_log_roots": [str(Path(path).resolve()) for path in self.semantic_log_roots],
            "backend_log_roots": [str(Path(path).resolve()) for path in self.backend_log_roots],
        }


@dataclass(slots=True)
class PreflightCheckResult:
    ok: bool
    detail: str = ""
    path: str | Path | None = None
    assets: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PreflightCheck:
    key: str
    run: Callable[[], PreflightCheckResult]
    required: bool = True


@dataclass(slots=True)
class PreflightPlan:
    round_id: int
    run_name: str
    checks: list[PreflightCheck] = field(default_factory=list)
    runtime_root: str | Path | None = None
    pipeline: list[str] = field(default_factory=list)
    previous_round: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    report_path: str | Path | None = None


@dataclass(slots=True)
class RunPlan:
    round_id: int
    steps: list[StepPlan]
    preflight: PreflightPlan | None = None
