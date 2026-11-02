from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from workflow_backends.gcmc.stage_scope import single_sample_writer

from workflow_backends.sampling import (
    SamplingBackendBinding,
    SamplingEvidenceRequest,
    SamplingExecutionRequest,
    SamplingExecutionResult,
    observe_path_backed_sampling_evidence,
)
from workflow_core.lifecycle.sampling import (
    SamplingEvidenceFacts,
    SamplingEvidenceFailure,
    SamplingEvidenceRebuildRequired,
    SamplingObservationResult,
    SamplingValidationResult,
)


def _active_sampling_branches(
    request: SamplingEvidenceRequest,
) -> tuple[bool, bool, bool]:
    tasks = request.plan.tasks_for_round(request.round_index)
    return (
        any(task.kind not in {"neb", "relax"} for task in tasks),
        any(task.kind == "neb" for task in tasks),
        any(task.kind == "relax" for task in tasks),
    )


class _HybridObservationSource:
    def __init__(self, *sources: object) -> None:
        self._sources = tuple(sources)
        self.allows_empty = all(getattr(s, "allows_empty", False) for s in sources)
        self.failed_task_count = sum(getattr(source, 'failed_task_count', 0) for source in sources)
        records = tuple(
            sorted(
                (
                    row
                    for source in self._sources
                    for row in source.catalog()
                ),
                key=lambda row: (row.task_index, row.frame_index),
            )
        )
        identities = tuple(row.structure_id for row in records)
        if len(identities) != len(set(identities)):
            raise ValueError("mixed sampling observation identities must be unique")
        self._records = records
        self._owner = {
            row.structure_id: source
            for source in self._sources
            for row in source.catalog()
        }

    def catalog(self) -> tuple[object, ...]:
        return self._records

    def replay(
        self, ordered_identities: Sequence[str], *, chunk_size: int
    ) -> Iterator[tuple[object, ...]]:
        size = int(chunk_size)
        identities = tuple(str(value) for value in ordered_identities)
        if size <= 0:
            raise ValueError("mixed sampling replay chunk_size must be positive")
        if any(not value for value in identities) or len(set(identities)) != len(identities):
            raise ValueError("mixed sampling replay identities must be unique and non-empty")
        missing = tuple(value for value in identities if value not in self._owner)
        if missing:
            raise ValueError(f"mixed sampling replay identity is absent: {missing[0]}")
        output: list[object] = []
        for identity in identities:
            chunks = tuple(self._owner[identity].replay((identity,), chunk_size=1))
            if len(chunks) != 1 or len(chunks[0]) != 1:
                raise ValueError("mixed sampling replay source returned ambiguous evidence")
            output.append(chunks[0][0])
            if len(output) == size:
                yield tuple(output)
                output = []
        if output:
            yield tuple(output)


@dataclass(frozen=True, slots=True)
class HybridSamplingBackendFactory:
    """Compose native MD and ASE path capabilities at the sample boundary."""

    md: object | None = None
    neb: object | None = None
    relax: object | None = None

    def bind(self, binding: SamplingBackendBinding) -> "HybridSamplingNativeOps":
        return HybridSamplingNativeOps(
            md=None if self.md is None else self.md.bind(binding),
            neb=None if self.neb is None else self.neb.bind(binding),
            relax=None if self.relax is None else self.relax.bind(binding),
        )


@dataclass(slots=True)
class HybridSamplingNativeOps:
    md: object | None = None
    neb: object | None = None
    relax: object | None = None

    def _active(self, request: SamplingEvidenceRequest) -> tuple[object, ...]:
        flags = _active_sampling_branches(request)
        branches = (self.md, self.neb, self.relax)
        active = tuple(
            branch for flag, branch in zip(flags, branches, strict=True) if flag
        )
        if any(
            flag and branch is None
            for flag, branch in zip(flags, branches, strict=True)
        ):
            raise ValueError("hybrid sampling is missing an active backend branch")
        return active

    @single_sample_writer
    def execute(self, request: SamplingExecutionRequest) -> SamplingExecutionResult:
        active = self._active(request.evidence)
        tasks = request.evidence.plan.tasks_for_round(request.evidence.round_index)
        if any(task.kind in {'gcmc_md', 'widom'} for task in tasks):
            if self.validate_current(request.evidence).complete:
                return SamplingExecutionResult(request.evidence.sample_dir, len(tasks),
                    getattr(self.open_current(request.evidence).source, 'failed_task_count', 0))
            from workflow_backends.gcmc.adapter import archive_incomplete_sample_tasks
            archive_incomplete_sample_tasks(request.evidence.sample_dir)
        if len(active) == 1:
            return active[0].execute(request)
        if not active:
            raise ValueError("hybrid sampling requires at least one active task")
        results = tuple(branch.execute(request) for branch in active)
        return SamplingExecutionResult(
            request.evidence.sample_dir,
            sum(result.task_count for result in results),
            sum(getattr(result, 'failed_task_count', 0) for result in results),
        )

    def validate_current(self, request: SamplingEvidenceRequest) -> SamplingValidationResult:
        active = self._active(request)
        if len(active) == 1:
            return active[0].validate_current(request)
        if not active:
            return SamplingValidationResult(False)
        return SamplingValidationResult(
            all(branch.validate_current(request).complete for branch in active)
        )

    def open_current(self, request: SamplingEvidenceRequest) -> SamplingObservationResult:
        active = self._active(request)
        if len(active) == 1:
            return active[0].open_current(request)
        if not active:
            raise ValueError("hybrid sampling requires at least one active task")
        return SamplingObservationResult(
            _HybridObservationSource(
                *(branch.open_current(request).source for branch in active),
            )
        )

    def observe_evidence(self, request: SamplingEvidenceRequest) -> SamplingEvidenceFacts:
        return observe_path_backed_sampling_evidence(request)

    def reconstruct_compatible_legacy(
        self, request: SamplingEvidenceRequest
    ) -> SamplingObservationResult:
        raise SamplingEvidenceRebuildRequired(
            "mixed sampling has no historical interpretation; rebuild rN/sample",
            failure=SamplingEvidenceFailure.HISTORICAL_INCOMPATIBLE,
            round_index=request.round_index,
        )


@dataclass(slots=True)
class GcmcMixedNativeOps(HybridSamplingNativeOps):
    """Use the existing composition seam for FLAMES and ordinary native MD."""
    gcmc: object | None = None

    def _active(self, request: SamplingEvidenceRequest) -> tuple[object, ...]:
        kinds = {task.kind for task in request.plan.tasks_for_round(request.round_index)}
        active = []
        if 'gcmc_md' in kinds:
            active.append(self.gcmc)
        if kinds - {'gcmc_md', 'neb', 'relax', 'widom'}:
            active.append(self.md)
        if any(branch is None for branch in active):
            raise ValueError('mixed GCMC sampling is missing an active backend branch')
        return tuple(active)


__all__ = ["HybridSamplingBackendFactory", "HybridSamplingNativeOps", "GcmcMixedNativeOps"]


@dataclass(slots=True)
class WidomMixedNativeOps(HybridSamplingNativeOps):
    """Compose insertion with the existing native sampling branch."""
    widom: object | None = None

    def _active(self, request):
        kinds = {task.kind for task in request.plan.tasks_for_round(request.round_index)}
        active = []
        if kinds - {'widom', 'neb', 'relax'}:
            active.append(self.md)
        if 'widom' in kinds:
            active.append(self.widom)
        if any(branch is None for branch in active):
            raise ValueError('missing active Widom/native sampling branch')
        return tuple(active)
