from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ase import Atoms


class FirstPrinciplesBackend(Protocol):
    """Backend contract for first-principles prepare/run/collect orchestration."""

    name: str
    structure_name: str
    task_dir_prefix: str

    def resolve_runtime(self, cfg: dict[str, Any]) -> dict[str, Any]:
        ...

    def prepare_task(self, *, task_dir: Path, structure_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        ...

    def run_tasks(self, cfg: dict[str, Any]) -> dict[str, Any]:
        ...

    def task_is_successful(self, task_dir: Path, cfg: dict[str, Any]) -> bool:
        ...

    def parse_labeled_task(self, task_dir: Path, cfg: dict[str, Any]) -> Atoms:
        ...


class ExternalHandoffFirstPrinciplesBackend:
    """Backend for external handoff-only flows where the workflow prepares structures but does not execute labels."""

    name = "external_handoff"
    structure_name = "structure.xyz"
    task_dir_prefix = "task."

    def resolve_runtime(self, cfg: dict[str, Any]) -> dict[str, Any]:
        runtime = dict(cfg)
        runtime["first_principles_backend"] = self.name
        runtime["first_principles_runtime_resolution"] = {"backend": self.name}
        runtime.setdefault("dft_runtime_resolution", runtime["first_principles_runtime_resolution"])
        return runtime

    def prepare_task(self, *, task_dir: Path, structure_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
        del task_dir, structure_path, cfg
        return {}

    def run_tasks(self, cfg: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("external_handoff first-principles backend cannot execute tasks")

    def task_is_successful(self, task_dir: Path, cfg: dict[str, Any]) -> bool:
        del task_dir, cfg
        return False

    def parse_labeled_task(self, task_dir: Path, cfg: dict[str, Any]) -> Atoms:
        del task_dir, cfg
        raise RuntimeError("external_handoff first-principles backend cannot parse labeled tasks")
