from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from workflow_runtime.execution.planning import PlannedExecution


class ExecutionStore:
    def __init__(self, root: str | Path, *, iter_index: int):
        if isinstance(iter_index, bool) or not isinstance(iter_index, int) or iter_index < 0:
            raise ValueError("execution store iter index must be non-negative")
        self.root = Path(root).resolve()
        self.iter_index = iter_index

    def attempt_root(self, planned: PlannedExecution) -> Path:
        block = planned.unit.block_id.replace("/", "__")
        return (
            self.root
            / ".dice_state"
            / "executions"
            / f"iter{self.iter_index}"
            / block
            / planned.unit.unit_id
            / "attempts"
            / planned.attempt_id
        )

    def publish_exact(
        self,
        planned: PlannedExecution,
        name: str,
        payload: Mapping[str, object],
    ) -> Path:
        path = self.attempt_root(planned) / name
        existing = self.read(planned, name)
        normalized = dict(payload)
        if existing is not None:
            if existing != normalized:
                raise ValueError(f"durable execution evidence mismatch for {name}")
            return path
        _atomic_write_json(path, normalized)
        return path

    def read(self, planned: PlannedExecution, name: str) -> dict[str, object] | None:
        path = self.attempt_root(planned) / name
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"durable execution evidence is malformed: {name}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"durable execution evidence must be an object: {name}")
        return value

    def replace(
        self,
        planned: PlannedExecution,
        name: str,
        payload: Mapping[str, object],
    ) -> Path:
        path = self.attempt_root(planned) / name
        _atomic_write_json(path, dict(payload))
        return path


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
