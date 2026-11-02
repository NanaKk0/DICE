from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Mapping

from .types import (
    DescriptorSnapshot,
    LabelingSnapshot,
    LabelSystemProgress,
    ScoreObservationSnapshot,
    ScoreSystemProgress,
)


SCHEMA = "dice.lifecycle.worker-observation"
SCHEMA_REVISION = 1
_HISTORICAL_SCHEMA = "dice.v2.worker-observation.v1"


@dataclass(frozen=True, slots=True)
class WorkerObservation:
    sequence: int
    kind: str
    frame_ids: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    role: str | None = None
    completed: int | None = None
    total: int | None = None
    width: int | None = None
    elements_completed: int | None = None
    elements_total: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerObservationBatch:
    observations: tuple[WorkerObservation, ...]
    warning: str | None = None


class WorkerObservationWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._sequence = 0

    def label_chunk(self, frame_ids: tuple[str, ...], uncertainties: tuple[float, ...]) -> None:
        self._write("label_chunk", frame_ids=list(frame_ids), values=list(uncertainties))

    def gmm_score_chunk(self, role: str, frame_ids: tuple[str, ...], p_values: tuple[float, ...]) -> None:
        self._write("gmm_score_chunk", role=role, frame_ids=list(frame_ids), values=list(p_values))

    def descriptor_progress(
        self,
        *,
        completed: int,
        total: int,
        width: int | None = None,
        elements_completed: int | None = None,
        elements_total: int | None = None,
    ) -> None:
        self._write(
            "descriptor_progress",
            completed=completed,
            total=total,
            width=width,
            elements_completed=elements_completed,
            elements_total=elements_total,
        )

    def _write(self, kind: str, **payload: object) -> None:
        self._sequence += 1
        row = {
            "schema": SCHEMA,
            "schema_revision": SCHEMA_REVISION,
            "sequence": self._sequence,
            "kind": kind,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()


class WorkerObservationCursor:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._next_sequence = 1
        self._halted = False

    def poll(self) -> WorkerObservationBatch | None:
        if self._halted or not self.path.is_file():
            return None
        rows: list[WorkerObservation] = []
        warning = None
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            while True:
                start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    handle.seek(start)
                    break
                self._offset = handle.tell()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    observation = _validate_observation(payload, expected_sequence=self._next_sequence)
                except Exception as exc:
                    self._halted = True
                    warning = f"worker observation channel stopped at byte {start}: {exc}"
                    break
                rows.append(observation)
                self._next_sequence += 1
        if not rows and warning is None:
            return None
        return WorkerObservationBatch(tuple(rows), warning)


class LabelObservationProjector:
    def __init__(self, systems: Mapping[str, str], *, total: int, trusted_below: float | None = None) -> None:
        self.systems = dict(systems)
        self.total = total
        self.trusted_below = trusted_below
        self._seen: set[str] = set()
        self._counts: dict[str, int] = {}
        self._trusted: dict[str, int] = {}

    def accept(self, observation: WorkerObservation) -> None:
        if observation.kind != "label_chunk":
            return
        for identity, uncertainty in zip(observation.frame_ids, observation.values, strict=True):
            if identity in self._seen or identity not in self.systems:
                raise ValueError("label observation changed or repeated canonical identity")
            self._seen.add(identity)
            system = self.systems[identity]
            self._counts[system] = self._counts.get(system, 0) + 1
            if self.trusted_below is not None and uncertainty < self.trusted_below:
                self._trusted[system] = self._trusted.get(system, 0) + 1

    def snapshot(self) -> LabelingSnapshot:
        totals: dict[str, int] = {}
        for system in self.systems.values():
            totals[system] = totals.get(system, 0) + 1
        rows = [LabelSystemProgress("ALL", len(self._seen), sum(self._trusted.values()) if self.trusted_below is not None else None, self.total)]
        rows.extend(
            LabelSystemProgress(
                system,
                self._counts.get(system, 0),
                self._trusted.get(system, 0) if self.trusted_below is not None else None,
                totals.get(system, 0),
            )
            for system in sorted(set(self.systems.values()))
        )
        return LabelingSnapshot(len(self._seen), self.total, tuple(rows), self.trusted_below)


class ScoreObservationProjector:
    def __init__(self, systems: Mapping[str, str], *, total: int, role: str) -> None:
        self.systems = dict(systems)
        self.total = total
        self.role = role
        self._seen: set[str] = set()
        self._counts: dict[str, int] = {}
        self._sums: dict[str, float] = {}

    def accept(self, observation: WorkerObservation) -> None:
        if observation.kind != "gmm_score_chunk" or observation.role != self.role:
            return
        for identity, value in zip(observation.frame_ids, observation.values, strict=True):
            if identity in self._seen or identity not in self.systems:
                raise ValueError("score observation changed or repeated canonical identity")
            self._seen.add(identity)
            system = self.systems[identity]
            self._counts[system] = self._counts.get(system, 0) + 1
            self._sums[system] = self._sums.get(system, 0.0) + value

    def snapshot(self) -> ScoreObservationSnapshot:
        count = len(self._seen)
        total_sum = sum(self._sums.values())
        rows = [ScoreSystemProgress("ALL", count, None if count == 0 else total_sum / count)]
        rows.extend(
            ScoreSystemProgress(system, self._counts.get(system, 0), None if self._counts.get(system, 0) == 0 else self._sums[system] / self._counts[system])
            for system in sorted(set(self.systems.values()))
        )
        return ScoreObservationSnapshot(self.role, count, self.total, tuple(rows))


class DescriptorObservationProjector:
    def __init__(self, *, label: str) -> None:
        self.label = label
        self._latest: WorkerObservation | None = None

    def accept(self, observation: WorkerObservation) -> None:
        if observation.kind == "descriptor_progress":
            self._latest = observation

    def snapshot(self) -> DescriptorSnapshot:
        if self._latest is None:
            return DescriptorSnapshot(self.label, 0, 0)
        row = self._latest
        return DescriptorSnapshot(self.label, int(row.completed), int(row.total), row.width, row.elements_completed, row.elements_total)


class WorkerObservationScope:
    def __init__(self, operation_root: str | Path) -> None:
        self.directory = Path(operation_root) / ".dice_worker_observations"
        self.path = self.directory / "observations.jsonl"
        self.writer = WorkerObservationWriter(self.path)

    def __enter__(self) -> "WorkerObservationScope":
        self.directory.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


class WorkerObservationProgressProbe:
    """Turn every newly consumed worker row into one absolute projected snapshot."""

    def __init__(self, cursor: WorkerObservationCursor, projector: object) -> None:
        self.cursor = cursor
        self.projector = projector
        self._token = 0

    def poll(self):
        from .operation import ProbeUpdate

        batch = self.cursor.poll()
        if batch is None:
            return None
        for observation in batch.observations:
            self.projector.accept(observation)
        if not batch.observations:
            if batch.warning:
                raise ValueError(batch.warning)
            return None
        self._token += len(batch.observations)
        return ProbeUpdate(str(self._token), self.projector.snapshot(), batch.warning)


def optional_observation_writer(path: str | Path | None) -> WorkerObservationWriter | None:
    return None if path is None else WorkerObservationWriter(path)


def observed_descriptor_chunks(chunks, *, writer: WorkerObservationWriter | None, total: int, width: int):
    completed = 0
    for chunk in chunks:
        completed += len(chunk.frame_ids)
        if writer is not None:
            writer.descriptor_progress(completed=completed, total=total, width=width)
        yield chunk


def emit_gmm_score_chunks(writer: WorkerObservationWriter | None, *, role: str, frame_ids, p_values, chunk_size: int = 256) -> None:
    if writer is None:
        return
    ids = tuple(str(item) for item in frame_ids)
    values = tuple(float(item) for item in p_values)
    for start in range(0, len(ids), chunk_size):
        writer.gmm_score_chunk(role, ids[start : start + chunk_size], values[start : start + chunk_size])


def _validate_observation(payload: object, *, expected_sequence: int) -> WorkerObservation:
    if not isinstance(payload, dict):
        raise ValueError("unsupported observation schema")
    value = dict(payload)
    if value.get("schema") == _HISTORICAL_SCHEMA and "schema_revision" not in value:
        value["schema"] = SCHEMA
        value["schema_revision"] = SCHEMA_REVISION
    if value.get("schema") != SCHEMA or value.get("schema_revision") != SCHEMA_REVISION:
        raise ValueError("unsupported observation schema")
    payload = value
    sequence = payload.get("sequence")
    kind = payload.get("kind")
    if sequence != expected_sequence or kind not in {"label_chunk", "gmm_score_chunk", "descriptor_progress"}:
        raise ValueError("observation sequence or kind is invalid")
    if kind in {"label_chunk", "gmm_score_chunk"}:
        ids = payload.get("frame_ids")
        values = payload.get("values")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or not isinstance(values, list) or len(values) != len(ids):
            raise ValueError("chunk identities and values are invalid")
        numbers = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in numbers):
            raise ValueError("chunk values must be finite")
        if kind == "gmm_score_chunk" and (payload.get("role") not in {"teacher", "student"} or any(value <= 0 or value > 1 for value in numbers)):
            raise ValueError("GMM role or p values are invalid")
        return WorkerObservation(sequence, kind, tuple(str(item) for item in ids), numbers, payload.get("role"))
    completed, total, width = payload.get("completed"), payload.get("total"), payload.get("width")
    if not isinstance(completed, int) or not isinstance(total, int) or completed < 0 or total < completed or (width is not None and (not isinstance(width, int) or width <= 0)):
        raise ValueError("descriptor progress is invalid")
    return WorkerObservation(sequence, kind, completed=completed, total=total, width=width, elements_completed=payload.get("elements_completed"), elements_total=payload.get("elements_total"))
