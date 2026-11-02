from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from workflow_core.lifecycle.artifact_contracts import ExpectedArtifactSemantics


ExpectationT = TypeVar("ExpectationT")
CapabilityArtifactValidator = Callable[[Path, ExpectationT], bool]


@dataclass(frozen=True, slots=True)
class BoundArtifactValidator(Generic[ExpectationT]):
    """One capability validator bound to one path-free semantic expectation."""

    expectation: ExpectedArtifactSemantics[ExpectationT]
    validator: CapabilityArtifactValidator[ExpectationT]

    def __post_init__(self) -> None:
        if not callable(self.validator):
            raise TypeError("bound artifact validator requires a callable")

    def __call__(self, root: Path) -> bool:
        try:
            accepted = self.validator(Path(root), self.expectation.facts)
        except (OSError, TypeError, ValueError):
            return False
        if not isinstance(accepted, bool):
            raise TypeError("capability artifact validator must return a boolean")
        return accepted


__all__ = ["BoundArtifactValidator", "CapabilityArtifactValidator"]
