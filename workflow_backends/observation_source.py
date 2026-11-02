from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

from ase import Atoms


@dataclass(frozen=True, slots=True)
class ObservationLocator:
    """Stable backend-owned coordinates for one native observation."""

    backend: str
    task_index: int
    frame_index: int
    md_step: int | None
    optimizer_step: int | None = None
    image_index: int | None = None


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """Compact backend-neutral screen catalog row without geometry."""

    structure_id: str
    system_id: str
    task_index: int
    frame_index: int
    md_step: int | None
    student_uncertainty: float | None
    locator: ObservationLocator
    sampling_role_id: str | None = None
    selection_group_id: str | None = None
    condition_group_id: str | None = None
    task_id: str | None = None
    optimizer_step: int | None = None
    image_index: int | None = None
    phase: str | None = None
    cycle: int | None = None
    phase_step: int | None = None
    mc_attempts: int | None = None
    md_time_fs: float | None = None


class ReplayableObservationSource(Protocol):
    def catalog(self) -> tuple[ObservationRecord, ...]: ...

    def replay(
        self,
        ordered_identities: Sequence[str],
        *,
        chunk_size: int,
    ) -> Iterator[tuple[Atoms, ...]]: ...


class InMemoryObservationSource:
    """Small test implementation of the production catalog/replay contract."""

    def __init__(
        self,
        records: Sequence[ObservationRecord],
        frames: Sequence[Atoms],
    ) -> None:
        if len(records) != len(frames):
            raise ValueError("observation records and frames must have equal length")
        self._records = tuple(records)
        self._frames: dict[str, Atoms] = {}
        for record, frame in zip(records, frames, strict=True):
            identity = str(frame.info.get("structure_id", ""))
            if identity != record.structure_id or not identity:
                raise ValueError("observation frame identity does not match its catalog record")
            if identity in self._frames:
                raise ValueError("in-memory observation identities must be unique")
            self._frames[identity] = frame.copy()

    def catalog(self) -> tuple[ObservationRecord, ...]:
        return self._records

    def replay(
        self,
        ordered_identities: Sequence[str],
        *,
        chunk_size: int,
    ) -> Iterator[tuple[Atoms, ...]]:
        size = int(chunk_size)
        if size <= 0:
            raise ValueError("observation replay chunk_size must be positive")
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("observation replay identities must be unique and non-empty")
        missing = [identity for identity in identities if identity not in self._frames]
        if missing:
            raise ValueError(f"observation replay identity is absent from the catalog: {missing[0]}")
        for start in range(0, len(identities), size):
            yield tuple(self._frames[identity].copy() for identity in identities[start : start + size])


def observation_identity(
    *,
    run_json_path: str | Path,
    iter_index: int,
    round_index: int,
    system_id: str,
    temperature_k: float,
    pressure_bar: float | None,
    md_step: int,
    sampling_role_id: str | None = None,
    condition_group_id: str | None = None,
    task_id: str | None = None,
) -> str:
    payload = (
        f"{Path(run_json_path).resolve()}\0{int(iter_index)}\0{int(round_index)}\0"
        f"{system_id}\0{float(temperature_k)}\0{pressure_bar}\0{int(md_step)}"
    )
    if sampling_role_id is not None or condition_group_id is not None or task_id is not None:
        payload += (
            f"\0role={sampling_role_id}\0condition={condition_group_id}\0task={task_id}"
        )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "InMemoryObservationSource",
    "ObservationLocator",
    "ObservationRecord",
    "ReplayableObservationSource",
    "observation_identity",
]
