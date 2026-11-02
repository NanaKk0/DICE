from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import json
import os
from pathlib import Path
import shutil
import time

from ase import Atoms

from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)


class RollingExtxyzChannel:
    """Acknowledged bounded extxyz slots for one operation-local score stream."""

    def __init__(
        self,
        root: str | Path,
        *,
        ordered_identities: Sequence[str],
        slot_count: int = 2,
        timeout_seconds: float = 300.0,
        poll_seconds: float = 0.01,
    ) -> None:
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(
            identities
        ):
            raise ValueError(
                "rolling extxyz requires unique non-empty canonical identities"
            )
        if not isinstance(slot_count, int) or isinstance(slot_count, bool) or slot_count < 1:
            raise ValueError("rolling extxyz slot_count must be a positive integer")
        if timeout_seconds <= 0.0 or poll_seconds <= 0.0:
            raise ValueError("rolling extxyz wait intervals must be positive")
        self.root = Path(root).resolve()
        self.ordered_identities = identities
        self.slot_count = slot_count
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.max_inflight_slots = 0
        self.root.mkdir(parents=True, exist_ok=False)
        self._slots = self.root / "slots"
        self._acks = self.root / "acks"
        self._slots.mkdir()
        self._acks.mkdir()
        self._manifest = self.root / "manifest.jsonl"
        self._manifest.touch()

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        ordered_identities: Sequence[str],
        slot_count: int,
        timeout_seconds: float = 300.0,
        poll_seconds: float = 0.01,
    ) -> "RollingExtxyzChannel":
        """Open the consumer side of a channel created by the parent process."""
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(
            identities
        ):
            raise ValueError(
                "rolling extxyz requires unique non-empty canonical identities"
            )
        if not isinstance(slot_count, int) or isinstance(slot_count, bool) or slot_count < 1:
            raise ValueError("rolling extxyz slot_count must be a positive integer")
        instance = object.__new__(cls)
        instance.root = Path(root).resolve()
        instance.ordered_identities = identities
        instance.slot_count = slot_count
        instance.timeout_seconds = float(timeout_seconds)
        instance.poll_seconds = float(poll_seconds)
        instance.max_inflight_slots = 0
        instance._slots = instance.root / "slots"
        instance._acks = instance.root / "acks"
        instance._manifest = instance.root / "manifest.jsonl"
        if not instance._manifest.is_file():
            raise ValueError("rolling extxyz channel manifest is missing")
        return instance

    def produce(
        self,
        chunks: Iterable[Sequence[Atoms]],
        *,
        consumer_alive: Callable[[], bool],
    ) -> None:
        outstanding: dict[int, int] = {}
        emitted: list[str] = []
        sequence = 0
        try:
            with self._manifest.open("a", encoding="utf-8", buffering=1) as manifest:
                for raw_chunk in chunks:
                    chunk = tuple(raw_chunk)
                    if not chunk:
                        raise ValueError("rolling extxyz producer emitted an empty chunk")
                    identities = tuple(
                        str(frame.info.get("structure_id", "")) for frame in chunk
                    )
                    expected = self.ordered_identities[
                        len(emitted) : len(emitted) + len(identities)
                    ]
                    if identities != expected:
                        raise ValueError(
                            "rolling extxyz producer changed canonical identity order"
                        )
                    slot = sequence % self.slot_count
                    previous = outstanding.pop(slot, None)
                    if previous is not None:
                        self._wait_for_ack(
                            slot,
                            previous,
                            other_alive=consumer_alive,
                            other_name="consumer",
                        )
                    target = self._slot_path(slot)
                    temporary = target.with_name(f".{target.name}.{sequence}.tmp")
                    write_extxyz_structures(temporary, chunk, allow_empty=False)
                    os.replace(temporary, target)
                    record = {
                        "kind": "chunk",
                        "sequence": sequence,
                        "slot": slot,
                        "frame_ids": list(identities),
                    }
                    manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
                    manifest.flush()
                    emitted.extend(identities)
                    outstanding[slot] = sequence
                    self.max_inflight_slots = max(
                        self.max_inflight_slots, len(outstanding)
                    )
                    sequence += 1
                if tuple(emitted) != self.ordered_identities:
                    raise ValueError(
                        "rolling extxyz producer reached early EOF before canonical coverage"
                    )
                manifest.write(
                    json.dumps(
                        {"kind": "eof", "chunks": sequence, "frames": len(emitted)},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                manifest.flush()
            for slot, expected_sequence in tuple(outstanding.items()):
                self._wait_for_ack(
                    slot,
                    expected_sequence,
                    other_alive=consumer_alive,
                    other_name="consumer",
                )
        finally:
            for temporary in self._slots.glob(".*.tmp"):
                temporary.unlink(missing_ok=True)

    def consume(
        self, *, producer_alive: Callable[[], bool]
    ) -> Iterator[tuple[Atoms, ...]]:
        expected_sequence = 0
        observed: list[str] = []
        deadline = time.monotonic() + self.timeout_seconds
        with self._manifest.open("r", encoding="utf-8") as manifest:
            while True:
                line = manifest.readline()
                if not line:
                    if not producer_alive():
                        line = manifest.readline()
                        if not line:
                            raise RuntimeError(
                                "rolling extxyz producer exited before terminal EOF"
                            )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("rolling extxyz consumer timed out waiting for input")
                    time.sleep(self.poll_seconds)
                    continue
                deadline = time.monotonic() + self.timeout_seconds
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("rolling extxyz manifest contains malformed JSON") from exc
                if not isinstance(record, dict) or record.get("kind") not in {
                    "chunk",
                    "eof",
                }:
                    raise ValueError("rolling extxyz manifest contains an invalid record")
                if record["kind"] == "eof":
                    if (
                        record.get("chunks") != expected_sequence
                        or record.get("frames") != len(observed)
                        or tuple(observed) != self.ordered_identities
                    ):
                        raise ValueError("rolling extxyz terminal coverage is invalid")
                    return
                sequence = record.get("sequence")
                slot = record.get("slot")
                frame_ids = record.get("frame_ids")
                if (
                    sequence != expected_sequence
                    or not isinstance(slot, int)
                    or not 0 <= slot < self.slot_count
                    or not isinstance(frame_ids, list)
                    or not frame_ids
                    or not all(isinstance(item, str) and item for item in frame_ids)
                ):
                    raise ValueError("rolling extxyz chunk record is invalid")
                frames = tuple(
                    read_extxyz_structures(self._slot_path(slot), allow_empty=False)
                )
                actual = tuple(
                    str(frame.info.get("structure_id", "")) for frame in frames
                )
                if actual != tuple(frame_ids):
                    raise ValueError("rolling extxyz slot identity evidence changed")
                expected = self.ordered_identities[
                    len(observed) : len(observed) + len(actual)
                ]
                if actual != expected:
                    raise ValueError("rolling extxyz consumer changed canonical order")
                yield frames
                observed.extend(actual)
                self._write_ack(slot, expected_sequence)
                expected_sequence += 1

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _slot_path(self, slot: int) -> Path:
        return self._slots / f"slot-{slot:03d}.xyz"

    def _ack_path(self, slot: int) -> Path:
        return self._acks / f"slot-{slot:03d}.json"

    def _write_ack(self, slot: int, sequence: int) -> None:
        target = self._ack_path(slot)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps({"sequence": sequence}, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _wait_for_ack(
        self,
        slot: int,
        sequence: int,
        *,
        other_alive: Callable[[], bool],
        other_name: str,
    ) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        path = self._ack_path(slot)
        while True:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and payload.get("sequence") == sequence:
                return
            if not other_alive():
                raise RuntimeError(
                    f"rolling extxyz {other_name} exited before acknowledging sequence {sequence}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"rolling extxyz timed out waiting for sequence {sequence} acknowledgement"
                )
            time.sleep(self.poll_seconds)


__all__ = ["RollingExtxyzChannel"]
