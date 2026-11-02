from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import queue
import os
import threading
from typing import Any

import numpy as np
from ase import Atoms

_DIAGNOSTIC_LOCK = threading.Lock()
_LAST_DIAGNOSTICS: dict[str, Any] = {}


def frames_per_call(batch_size: int, atoms_per_frame: int) -> int:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(atoms_per_frame, int)
        or isinstance(atoms_per_frame, bool)
        or atoms_per_frame <= 0
    ):
        raise ValueError("DeepMD batch_size and atoms_per_frame must be positive integers")
    return max(1, batch_size // atoms_per_frame)


def topology_key(exporter: Any, frame: Atoms) -> tuple[Any, ...]:
    native = exporter._native_inputs(frame)
    return (
        len(frame),
        tuple(int(item) for item in np.asarray(native["atom_types"]).reshape(-1)),
        tuple(bool(item) for item in np.asarray(frame.pbc, dtype=bool)),
        native["case_index"],
        None if native["fparam"] is None else tuple(native["fparam"].shape[1:]),
        None if native["aparam"] is None else tuple(native["aparam"].shape[1:]),
    )


@dataclass(frozen=True, slots=True)
class ScheduledFrame:
    ordinal: int
    frame: Atoms
    key: tuple[Any, ...]


def schedule_same_topology_batches(
    frames: Iterable[tuple[int, Atoms]],
    *,
    exporter: Any,
    batch_size: int,
    pending_watermark: int = 64,
) -> Iterator[tuple[ScheduledFrame, ...]]:
    """Boundedly group compatible frames and flush the oldest partial queue."""
    if pending_watermark <= 0:
        raise ValueError("DeepMD partial-topology watermark must be positive")
    pending: dict[tuple[Any, ...], deque[ScheduledFrame]] = defaultdict(deque)
    total_pending = 0

    def pop(key: tuple[Any, ...], *, full: bool) -> tuple[ScheduledFrame, ...]:
        nonlocal total_pending
        values = pending[key]
        capacity = frames_per_call(batch_size, len(values[0].frame))
        count = capacity if full else min(capacity, len(values))
        result = tuple(values.popleft() for _ in range(count))
        total_pending -= count
        if not values:
            del pending[key]
        return result

    expected_ordinal = 0
    for ordinal, frame in frames:
        if ordinal != expected_ordinal:
            raise ValueError("DeepMD topology scheduler input ordinal is not canonical")
        expected_ordinal += 1
        key = topology_key(exporter, frame)
        pending[key].append(ScheduledFrame(ordinal, frame, key))
        total_pending += 1
        capacity = frames_per_call(batch_size, len(frame))
        if len(pending[key]) >= capacity:
            yield pop(key, full=True)
        while total_pending >= pending_watermark and pending:
            oldest = min(pending, key=lambda item: pending[item][0].ordinal)
            yield pop(oldest, full=False)
    while pending:
        oldest = min(pending, key=lambda item: pending[item][0].ordinal)
        yield pop(oldest, full=False)


class PinnedHostBufferLRU:
    """Four-shape LRU with both entry and retained-byte bounds."""

    def __init__(
        self,
        *,
        max_shapes: int = 4,
        byte_cap: int = 512 * 1024 * 1024,
        allocator: Callable[[tuple[int, ...], np.dtype], np.ndarray] | None = None,
    ) -> None:
        if max_shapes != 4:
            raise ValueError("DeepMD reusable host cache is fixed at four shapes")
        if byte_cap <= 0:
            raise ValueError("DeepMD reusable host cache byte cap must be positive")
        self.max_shapes = max_shapes
        self.byte_cap = int(byte_cap)
        self._allocator = allocator or _pinned_numpy
        self._entries: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def retained_bytes(self) -> int:
        return sum(_entry_bytes(entry) for entry in self._entries.values())

    def acquire(
        self,
        exporter: Any,
        batch: Sequence[ScheduledFrame],
    ) -> dict[str, Any]:
        if not batch or any(item.key != batch[0].key for item in batch):
            raise ValueError("one pinned DeepMD buffer requires one compatible topology")
        cache_key = (batch[0].key, len(batch))
        entry = self._entries.pop(cache_key, None)
        if entry is None:
            self.misses += 1
            entry = self._allocate(exporter, batch)
            entry_bytes = _entry_bytes(entry)
            if entry_bytes > self.byte_cap:
                raise MemoryError("one DeepMD pinned host buffer exceeds its internal byte cap")
            while self._entries and (
                len(self._entries) >= self.max_shapes
                or self.retained_bytes + entry_bytes > self.byte_cap
            ):
                self._entries.popitem(last=False)
                self.evictions += 1
        else:
            self.hits += 1
        self._entries[cache_key] = entry
        self._fill(exporter, batch, entry)
        return entry

    def _allocate(self, exporter: Any, batch: Sequence[ScheduledFrame]) -> dict[str, Any]:
        first = exporter._native_inputs(batch[0].frame)
        count = len(batch)

        def repeated(value: np.ndarray | None) -> np.ndarray | None:
            if value is None:
                return None
            return self._allocator((count, *value.shape[1:]), np.dtype(value.dtype))

        return {
            "coords": self._allocator(
                (count, len(batch[0].frame), 3),
                np.dtype(first["coords"].dtype),
            ),
            "cells": repeated(first["cells"]),
            "atom_types": np.asarray(first["atom_types"], dtype=np.int32),
            "fparam": repeated(first["fparam"]),
            "aparam": repeated(first["aparam"]),
            "case_index": first["case_index"],
        }

    @staticmethod
    def _fill(
        exporter: Any,
        batch: Sequence[ScheduledFrame],
        entry: dict[str, Any],
    ) -> None:
        for index, item in enumerate(batch):
            native = exporter._native_inputs(item.frame)
            if native["case_index"] != entry["case_index"]:
                raise ValueError("one DeepMD buffer cannot mix case embeddings")
            entry["coords"][index] = native["coords"][0]
            for name in ("cells", "fparam", "aparam"):
                if (entry[name] is None) != (native[name] is None):
                    raise ValueError("one DeepMD buffer cannot mix conditioning shapes")
                if entry[name] is not None:
                    entry[name][index] = native[name][0]

    def report(self) -> dict[str, int]:
        return {
            "max_shapes": self.max_shapes,
            "byte_cap": self.byte_cap,
            "retained_shapes": len(self._entries),
            "retained_bytes": self.retained_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


def _pinned_numpy(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    try:
        import torch

        torch_dtype = {
            np.dtype("float64"): torch.float64,
            np.dtype("float32"): torch.float32,
            np.dtype("int32"): torch.int32,
        }[np.dtype(dtype)]
        return torch.empty(shape, dtype=torch_dtype, pin_memory=True).numpy()
    except (ImportError, KeyError, RuntimeError) as exc:
        raise RuntimeError("DeepMD PT scoring requires reusable pinned host memory") from exc


def _entry_bytes(entry: dict[str, Any]) -> int:
    return sum(value.nbytes for value in entry.values() if isinstance(value, np.ndarray))


@dataclass(frozen=True, slots=True)
class DeviceSnapshot:
    allocated: int
    reserved: int
    free: int
    total: int


class HystereticMemoryController:
    """Admission/eviction/backoff policy with a strict effective device cap."""

    def __init__(self, *, effective_cap_bytes: int, reserve_fraction: float = 0.10) -> None:
        if effective_cap_bytes <= 0 or not 0.0 < reserve_fraction < 1.0:
            raise ValueError("DeepMD device cap and reserve fraction are invalid")
        self.effective_cap_bytes = int(effective_cap_bytes)
        self.usable_bytes = int(effective_cap_bytes * (1.0 - reserve_fraction))
        self.clean_shape_peaks: dict[tuple[Any, ...], int] = {}
        self.actions: deque[dict[str, Any]] = deque(maxlen=128)
        self.action_counts: dict[str, int] = defaultdict(int)

    def _record(self, action: dict[str, Any]) -> None:
        self.actions.append(action)
        self.action_counts[str(action["action"])] += 1

    def observe_clean_peak(self, shape: tuple[Any, ...], *, baseline: int, peak: int) -> None:
        if min(baseline, peak) < 0 or peak < baseline:
            raise ValueError("DeepMD clean-shape memory observation is invalid")
        self.clean_shape_peaks[shape] = max(0, peak - baseline)

    def admit(self, shape: tuple[Any, ...], snapshot: DeviceSnapshot) -> str:
        incremental = self.clean_shape_peaks.get(shape, 0)
        predicted = max(snapshot.allocated + incremental, snapshot.reserved + incremental)
        inactive = snapshot.reserved - snapshot.allocated
        action = "admit"
        reason = "within_cap"
        if predicted > self.usable_bytes or snapshot.free < incremental:
            action, reason = "evict", "predicted_cap_pressure"
        elif inactive > self.effective_cap_bytes // 16 and snapshot.reserved > self.usable_bytes * 3 // 4:
            action, reason = "evict", "inactive_cache_hysteresis"
        self._record({"action": action, "reason": reason, "predicted": predicted})
        return action

    def backoff(self, batch_frames: int) -> int:
        if batch_frames <= 1:
            self._record({"action": "recycle", "reason": "single_frame_pressure"})
            return 1
        reduced = max(1, batch_frames // 2)
        self._record({"action": "backoff", "from": batch_frames, "to": reduced})
        return reduced

    def report(self) -> dict[str, Any]:
        return {
            "effective_cap_bytes": self.effective_cap_bytes,
            "usable_bytes": self.usable_bytes,
            "shape_profiles": len(self.clean_shape_peaks),
            "actions": list(self.actions),
            "action_counts": dict(self.action_counts),
        }


def record_runtime_diagnostics(
    *, host_cache: dict[str, Any], memory_controller: dict[str, Any] | None
) -> None:
    with _DIAGNOSTIC_LOCK:
        bounded_pipeline = {
            key: value
            for key, value in _LAST_DIAGNOSTICS.items()
            if key not in {"host_cache", "memory_controller"}
        }
        _LAST_DIAGNOSTICS.clear()
        _LAST_DIAGNOSTICS.update(
            {"host_cache": host_cache, "memory_controller": memory_controller}
        )
        _LAST_DIAGNOSTICS.update(bounded_pipeline)


def _record_handoff_ring_diagnostics(
    report: dict[str, Any], *, diagnostic_name: str
) -> None:
    with _DIAGNOSTIC_LOCK:
        _LAST_DIAGNOSTICS[diagnostic_name] = dict(report)


def pop_runtime_diagnostics() -> dict[str, Any]:
    with _DIAGNOSTIC_LOCK:
        result = dict(_LAST_DIAGNOSTICS)
        _LAST_DIAGNOSTICS.clear()
        return result


def cuda_allocator_fraction(effective_cap_bytes: int, total_bytes: int) -> float:
    """Translate the effective cap into the CUDA allocator's process limit."""
    if effective_cap_bytes <= 0:
        raise ValueError("DICE internal device-memory limit must be positive")
    if total_bytes <= 0 or effective_cap_bytes > total_bytes:
        raise ValueError("DICE device-memory limit exceeds visible CUDA memory")
    return float(effective_cap_bytes) / float(total_bytes)


def cuda_memory_controller() -> tuple[HystereticMemoryController, Callable[[], DeviceSnapshot], Callable[[], None]]:
    """Resolve the process-local CUDA cap; never offers a CPU execution path."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        free, total = (int(value) for value in torch.cuda.mem_get_info())
    except Exception as exc:
        raise RuntimeError("DeepMD PT scientific scoring requires a real CUDA device") from exc
    raw_cap = os.environ.get("DICE_DEVICE_MEMORY_LIMIT_BYTES")
    cap = min(total, int(raw_cap)) if raw_cap is not None else total
    if cap <= 0:
        raise ValueError("DICE internal device-memory limit must be positive")
    if raw_cap is not None:
        torch.cuda.set_per_process_memory_fraction(cuda_allocator_fraction(cap, total))

    def snapshot() -> DeviceSnapshot:
        current_free, current_total = (int(value) for value in torch.cuda.mem_get_info())
        return DeviceSnapshot(
            allocated=int(torch.cuda.memory_allocated()),
            reserved=int(torch.cuda.memory_reserved()),
            free=current_free,
            total=current_total,
        )

    def evict() -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return HystereticMemoryController(effective_cap_bytes=cap), snapshot, evict


def controlled_export_batches(
    exporter: Any,
    batch: Sequence[ScheduledFrame],
    *,
    buffer_cache: PinnedHostBufferLRU,
    controller: HystereticMemoryController | None,
    snapshot: Callable[[], DeviceSnapshot] | None = None,
    evict: Callable[[], None] | None = None,
) -> Iterator[Any]:
    """Split/retry allocation pressure and fail closed before any CPU fallback."""
    if not batch:
        raise ValueError("DeepMD controlled export requires a non-empty batch")
    if controller is not None:
        if snapshot is None or evict is None:
            raise ValueError("DeepMD memory controller requires snapshot and eviction hooks")
        before = snapshot()
        if controller.admit(batch[0].key, before) == "evict":
            evict()
            before = snapshot()
    else:
        before = None
    try:
        result = exporter.export_scheduled_batch(batch, buffer_cache=buffer_cache)
    except BaseException as exc:
        if not _allocation_pressure(exc):
            raise
        if controller is not None:
            controller.backoff(len(batch))
        if len(batch) > 1:
            midpoint = max(1, len(batch) // 2)
            yield from controlled_export_batches(
                exporter,
                batch[:midpoint],
                buffer_cache=buffer_cache,
                controller=controller,
                snapshot=snapshot,
                evict=evict,
            )
            yield from controlled_export_batches(
                exporter,
                batch[midpoint:],
                buffer_cache=buffer_cache,
                controller=controller,
                snapshot=snapshot,
                evict=evict,
            )
            return
        if evict is not None:
            evict()
        try:
            result = exporter.export_scheduled_batch(batch, buffer_cache=buffer_cache)
        except BaseException as retry_exc:
            if _allocation_pressure(retry_exc):
                raise RuntimeError(
                    "DeepMD single-frame CUDA allocation pressure requires worker recycle; CPU fallback is forbidden"
                ) from retry_exc
            raise
    if controller is not None and snapshot is not None and before is not None:
        after = snapshot()
        controller.observe_clean_peak(
            batch[0].key,
            baseline=before.allocated,
            peak=max(before.allocated, after.allocated),
        )
    yield result


def _allocation_pressure(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or (
        "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()
    )


def bounded_prefetch(
    chunks: Iterable[Any], *, slots: int = 2, diagnostic_name: str = "handoff_ring"
) -> Iterator[Any]:
    """One producer thread and one consumer with bounded descriptor retention."""
    if slots <= 0:
        raise ValueError("DeepMD descriptor ring slots must be positive")
    if not isinstance(diagnostic_name, str) or not diagnostic_name:
        raise ValueError("DeepMD descriptor ring diagnostic name must be non-empty")
    ring: queue.Queue[Any] = queue.Queue(maxsize=slots)
    sentinel = object()
    failure: list[BaseException] = []
    metrics = {
        "slots": int(slots),
        "max_depth": 0,
        "produced": 0,
        "consumed": 0,
        "failed": False,
    }

    def produce() -> None:
        try:
            for chunk in chunks:
                ring.put(chunk)
                metrics["produced"] += 1
                metrics["max_depth"] = max(metrics["max_depth"], ring.qsize())
        except BaseException as exc:
            failure.append(exc)
            metrics["failed"] = True
        finally:
            ring.put(sentinel)

    producer = threading.Thread(target=produce, name="dice-deepmd-gpu-producer", daemon=True)
    producer.start()

    def consume() -> Iterator[Any]:
        try:
            while True:
                item = ring.get()
                if item is sentinel:
                    if failure:
                        raise failure[0]
                    return
                metrics["consumed"] += 1
                yield item
        finally:
            producer.join(timeout=5.0)
            if producer.is_alive():
                raise RuntimeError("DeepMD descriptor producer did not terminate")
            _record_handoff_ring_diagnostics(
                metrics, diagnostic_name=diagnostic_name
            )

    return consume()


__all__ = [
    "DeviceSnapshot",
    "HystereticMemoryController",
    "PinnedHostBufferLRU",
    "ScheduledFrame",
    "bounded_prefetch",
    "controlled_export_batches",
    "cuda_memory_controller",
    "frames_per_call",
    "pop_runtime_diagnostics",
    "record_runtime_diagnostics",
    "schedule_same_topology_batches",
    "topology_key",
]
