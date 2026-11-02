from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from workflow_core.lifecycle.student_capabilities import StudentOutcomeStatus


class StudentCompactScoreView(Protocol):
    def close(self, *, cleanup: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class StudentGMMScore:
    """Logic-owned ordered student score with an optional leased native view."""

    values: Mapping[str, float]
    frame_means: Mapping[str, np.ndarray] | None
    fit: Any
    descriptor_schema: Mapping[str, object] | None = None
    compact_view: StudentCompactScoreView | None = None
    status: StudentOutcomeStatus = StudentOutcomeStatus.SUCCEEDED
    element_frame_means: Mapping[str, Mapping[str, np.ndarray]] | None = None

    def __post_init__(self) -> None:
        status = StudentOutcomeStatus(self.status)
        identities = tuple(str(identity) for identity in self.values)
        if any(not identity for identity in identities) or len(set(identities)) != len(
            identities
        ):
            raise ValueError("student GMM score identities must be ordered and unique")
        values = {identity: float(self.values[identity]) for identity in identities}
        if not np.isfinite(tuple(values.values())).all():
            raise ValueError("student GMM score values must be finite")
        if self.frame_means is not None:
            if tuple(self.frame_means) != identities:
                raise ValueError("student GMM frame means changed canonical order")
            if any(
                not np.isfinite(np.asarray(self.frame_means[identity])).all()
                for identity in identities
            ):
                raise ValueError("student GMM frame means must be finite")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", values)

    def close(self) -> None:
        if self.compact_view is not None:
            self.compact_view.close(cleanup=True)


__all__ = ["StudentCompactScoreView", "StudentGMMScore"]
