"""Bounded complete-frame GPU sharding for backend-owned inference operations."""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np

from workflow_backends.common.cancellation import handled_sigterm
from workflow_backends.common.sampling_pool import sampling_visible_devices
from workflow_backends.execution import ProcessGroupRegistry, operation_execution_scope
from workflow_runtime.execution.allocation import validate_operation_allocation
from workflow_runtime.execution.task_scheduler import AllocationTask, execute_allocation_tasks
from workflow_policies.gmm_reference_tail import GMMScoreResult
from workflow_runtime.lifecycle.progress import GenericWorkSnapshot, OperationProgressReporter


def shard_progress_reporter(parent):
    """Workers own reporter lifetimes; only the parent publishes pool progress."""
    if parent is None:
        return None
    return OperationProgressReporter(
        operation_id=parent.operation_id,
        on_snapshot=lambda _snapshot: parent.heartbeat(), on_heartbeat=parent.heartbeat,
    )


def inference_pool_enabled(context) -> bool:
    return context is not None and context.allocation.gpus_per_node > 1


def execute_inference_shards(*, context, identities, frame_chunks, execute, progress_reporter=None):
    """Read once, route complete frames through two-slot queues, preserve identity order.

    Each callback owns one worker operation for its entire shard. It runs with
    exactly one assigned visible GPU and the shared cancellation registry.
    """
    identities = tuple(identities)
    if not identities or any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("inference shards require unique non-empty canonical identities")
    allocation = context.allocation
    validate_operation_allocation(context.role, allocation)
    if allocation.mode != "pool" or allocation.per_gpu != 1:
        raise ValueError("inference sharding requires pool mode with per_gpu 1")
    devices = sampling_visible_devices(allocation.gpus_per_node)
    count = min(len(identities), allocation.gpus_per_node)
    shard_ids = tuple(identities[index::count] for index in range(count))
    queues = tuple(Queue(maxsize=2) for _ in range(count))
    stop = Event()
    registry = ProcessGroupRegistry()
    producer_errors = []
    completed = []
    progress_lock = Lock()
    completed_frames = 0
    finished = object()

    def put(queue, item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return
            except Full:
                pass

    def produce():
        consumed = 0
        try:
            for raw_chunk in frame_chunks:
                if stop.is_set():
                    return
                chunk = tuple(raw_chunk)
                actual = tuple(str(frame.info.get("structure_id", "")) for frame in chunk)
                if not chunk or actual != identities[consumed:consumed + len(chunk)]:
                    raise ValueError("inference input changed canonical frame order")
                groups = [[] for _ in range(count)]
                for offset, frame in enumerate(chunk):
                    groups[(consumed + offset) % count].append(frame)
                consumed += len(chunk)
                for index, frames in enumerate(groups):
                    if frames:
                        put(queues[index], tuple(frames))
            if consumed != len(identities):
                raise ValueError("inference input omitted canonical frames")
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            for queue in queues:
                put(queue, finished)

    def stream(index):
        while not stop.is_set():
            try:
                item = queues[index].get(timeout=0.1)
            except Empty:
                continue
            if item is finished:
                if producer_errors:
                    raise ValueError("inference input producer failed") from producer_errors[0]
                return
            yield item
        raise RuntimeError("inference shard cancelled")

    def cancel():
        stop.set()
        registry.cancel_all()

    def assigned(index, assignment):
        nonlocal completed_frames
        with operation_execution_scope(environment_overrides=assignment.environment_override, process_registry=registry):
            frames = stream(index)
            result = execute(index, shard_ids[index], frames)
            completed.append(result)
            if isinstance(result, GMMScoreResult):
                returned_ids = tuple(str(identity) for identity in result.frame_ids)
            elif hasattr(result, "identities"):
                returned_ids = tuple(result.identities)
            elif isinstance(result, (list, tuple)):
                returned_ids = tuple(str(frame.info.get("structure_id", "")) for frame in result)
            else:
                returned_ids = shard_ids[index]
            if returned_ids != shard_ids[index]:
                raise ValueError("inference worker changed its assigned canonical identities")
            # A callback may not silently leave its bounded input unconsumed.
            for _ in frames:
                raise ValueError("inference worker did not consume its complete shard")
            if progress_reporter is not None:
                with progress_lock:
                    completed_frames += len(shard_ids[index])
                    try:
                        progress_reporter.report(GenericWorkSnapshot(
                            activity="inference GPU pool", completed=completed_frames,
                            total=len(identities), unit="frames",
                        ))
                    except Exception:
                        # Diagnostic delivery must not alter scientific completion.
                        pass
            return result

    producer = Thread(target=produce, name="dice-inference-input")
    with handled_sigterm("inference"):
        producer.start()
        try:
            return execute_allocation_tasks(
                tuple(AllocationTask(f"shard_{index:03d}", index) for index in range(count)),
                allocation=allocation, visible_devices=devices, execute=assigned, cancel_active=cancel,
            )
        except BaseException:
            for result in completed:
                close = getattr(result, "close", None)
                if callable(close):
                    close(cleanup=True)
            raise
        finally:
            stop.set()
            registry.cancel_all()
            producer.join()


def merge_gmm_shards(results, identities):
    """Validate disjoint coverage and gather independent scores in canonical order."""
    identities = tuple(identities)
    try:
        names = tuple(
            str(identity) for result in results
            for identity in (result.frame_ids if isinstance(result, GMMScoreResult) else result.identities)
        )
        if len(names) != len(identities) or len(set(names)) != len(names) or set(names) != set(identities):
            raise ValueError("GMM shards changed exact candidate coverage")
        schema = results[0].descriptor_schema
        if schema is None or any(result.descriptor_schema != schema for result in results):
            raise ValueError("GMM shards changed descriptor schema")
        positions = {identity: index for index, identity in enumerate(names)}
        order = np.asarray([positions[identity] for identity in identities])
        probabilities = np.concatenate([result.frame_p for result in results])[order].copy()
        statistics = np.concatenate([result.frame_statistics for result in results])[order].copy()
        if probabilities.shape != (len(identities),) or statistics.shape != probabilities.shape or not np.isfinite(probabilities).all() or not np.isfinite(statistics).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("GMM shards returned invalid probabilities or statistics")
        means = None
        if any(result.frame_means is not None for result in results):
            if any(result.frame_means is None for result in results):
                raise ValueError("GMM shards omitted frame means")
            means = np.concatenate([result.frame_means for result in results])[order].copy()
            if means.shape != (len(identities), int(schema["width"])) or not np.isfinite(means).all():
                raise ValueError("GMM shards returned invalid frame means")
        elements = None
        element_values = tuple(getattr(result, "element_means", None) for result in results)
        if any(values is not None for values in element_values):
            if any(values is None for values in element_values):
                raise ValueError("GMM shards omitted element means")
            collected = {}
            for values in element_values:
                for element, (ids, rows) in values.items():
                    current = collected.setdefault(element, {})
                    for identity, row in zip(ids, rows, strict=True):
                        if identity in current or identity not in positions:
                            raise ValueError("GMM element shards changed candidate coverage")
                        current[identity] = np.asarray(row, dtype=np.float32).copy()
                        if current[identity].shape != (int(schema["width"]),) or not np.isfinite(current[identity]).all():
                            raise ValueError("GMM element shards returned invalid means")
            elements = {
                element: (tuple(identity for identity in identities if identity in rows),
                          np.stack([rows[identity] for identity in identities if identity in rows]))
                for element, rows in collected.items()
            }
        return GMMScoreResult(np.asarray(identities), probabilities, statistics, means, schema, elements)
    finally:
        for result in results:
            close = getattr(result, "close", None)
            if callable(close):
                close(cleanup=True)


def merge_labeled_shards(results, identities):
    frames = {}
    for shard in results:
        for frame in shard:
            identity = str(frame.info.get("structure_id", ""))
            if not identity or identity in frames:
                raise ValueError("label shards returned duplicate or missing identities")
            frames[identity] = frame
    if set(frames) != set(identities):
        raise ValueError("label shards changed exact candidate coverage")
    return [frames[identity] for identity in identities]


def single_gpu_context(context):
    return replace(context, allocation=replace(context.allocation, gpus_per_node=1))


def shard_stream_score(executor, method, *, context, identities, frame_chunks, identity_key, **kwargs):
    """Invoke a native stream method once per assigned GPU."""
    worker = replace(executor, gmm_score_context=single_gpu_context(context))
    root = Path(kwargs.pop("work_dir"))
    reporter = kwargs.pop("progress_reporter", None)
    results = execute_inference_shards(
        context=context, identities=identities, frame_chunks=frame_chunks, progress_reporter=reporter,
        execute=lambda index, ids, chunks: getattr(worker, method)(
            **kwargs, **{identity_key: ids}, frame_chunks=chunks,
            work_dir=root / f"shard_{index:03d}", progress_reporter=shard_progress_reporter(reporter),
        ),
    )
    return merge_gmm_shards(results, identities)


def shard_labels(executor, *, context, models, frames, work_dir, **kwargs):
    worker = replace(executor, label_context=single_gpu_context(context))
    identities = tuple(str(frame.info.get("structure_id", "")) for frame in frames)
    reporter = kwargs.pop("progress_reporter", None)
    results = execute_inference_shards(
        context=context, identities=identities, progress_reporter=reporter,
        frame_chunks=(frames[start:start + 64] for start in range(0, len(frames), 64)),
        execute=lambda index, ids, chunks: worker.label(
            models=models, frames=[frame for chunk in chunks for frame in chunk],
            work_dir=Path(work_dir) / f"shard_{index:03d}",
            progress_reporter=shard_progress_reporter(reporter), **kwargs,
        ),
    )
    return merge_labeled_shards(results, identities)


def shard_file_score(executor, model, *, context, candidate_paths, frame_ids_by_chunk, **kwargs):
    """Keep legacy file transports bounded, with one model-owning worker per GPU."""
    from workflow_core.structures.formats import read_extxyz_structures, write_extxyz_structures
    identities = tuple(identity for ids in frame_ids_by_chunk for identity in ids)
    worker = replace(executor, gmm_score_context=single_gpu_context(context))
    root = Path(kwargs.pop("work_dir"))
    reporter = kwargs.pop("progress_reporter", None)

    def execute(index, ids, chunks):
        work = root / f"shard_{index:03d}"
        paths, chunk_ids = [], []
        try:
            for ordinal, chunk in enumerate(chunks):
                paths.append(write_extxyz_structures(work / "inputs" / f"chunk_{ordinal:06d}.xyz", chunk, allow_empty=False))
                chunk_ids.append(tuple(str(frame.info["structure_id"]) for frame in chunk))
            return worker.score_gmm_router_chunks(
                model, candidate_paths=tuple(paths), frame_ids_by_chunk=tuple(chunk_ids),
                work_dir=work, progress_reporter=shard_progress_reporter(reporter), **kwargs,
            )
        finally:
            shutil.rmtree(work / "inputs", ignore_errors=True)

    results = execute_inference_shards(
        context=context, identities=identities, progress_reporter=reporter,
        frame_chunks=(read_extxyz_structures(path, allow_empty=False) for path in candidate_paths),
        execute=execute,
    )
    return merge_gmm_shards(results, identities)
