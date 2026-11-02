from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from workflow_core.lifecycle.state_model import PublicBlockId


ExpectationT = TypeVar("ExpectationT")


@dataclass(frozen=True, slots=True)
class ExpectedArtifactSemantics(Generic[ExpectationT]):
    """Consumer-owned requirements for one canonical public result."""

    block: PublicBlockId
    artifact_kind: str
    facts: ExpectationT

    def __post_init__(self) -> None:
        if not isinstance(self.block, PublicBlockId):
            raise TypeError("artifact expectation requires a PublicBlockId")
        artifact_kind = str(self.artifact_kind).strip()
        if not artifact_kind:
            raise ValueError("artifact kind identity must be non-empty")
        object.__setattr__(self, "artifact_kind", artifact_kind)
        if self.facts is None:
            raise ValueError(
                "artifact expectation requires capability-specific facts"
            )


__all__ = ["ExpectedArtifactSemantics"]
