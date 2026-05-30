from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from workflow_core.common.io import append_jsonl, load_jsonl


@dataclass(slots=True)
class UncertaintyRecord:
    backend: str
    sampling_backend: str
    source: str
    task_id: str
    frame_id: str | None
    step: int | None
    temperature: float | None
    force_std_max: float
    route: str
    structure_path: str | None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("metadata") is None:
            payload["metadata"] = {}
        return payload


def route_uncertainty(force_std_max: float, *, trust_lo: float, trust_hi: float) -> str:
    metric = float(force_std_max)
    if metric < float(trust_lo):
        return "accurate"
    if metric > float(trust_hi):
        return "failed_uncertainty"
    if metric == float(trust_hi):
        return "failed_uncertainty"
    return "candidate"


def append_uncertainty_record(path: str | Path, record: UncertaintyRecord) -> None:
    append_jsonl(path, record.to_dict())


def append_uncertainty_records(path: str | Path, records: list[UncertaintyRecord]) -> Path:
    path = Path(path)
    for record in records:
        append_uncertainty_record(path, record)
    return path


def load_uncertainty_records(path: str | Path) -> list[UncertaintyRecord]:
    return [UncertaintyRecord(**row) for row in load_jsonl(path)]
