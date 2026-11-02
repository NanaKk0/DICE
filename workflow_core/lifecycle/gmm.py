from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
from ase import Atoms


class TeacherCompactScoreView(Protocol):
    """Backend-owned bounded score view retained by the canonical result."""

    def close(self, *, cleanup: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class TeacherGMMScore:
    frames: tuple[Atoms, ...]
    frame_means: np.ndarray | None
    descriptor_schema: Mapping[str, object] | None = None
    compact_view: TeacherCompactScoreView | None = None

    def close(self) -> None:
        if self.compact_view is not None:
            self.compact_view.close(cleanup=True)


__all__ = ["TeacherCompactScoreView", "TeacherGMMScore"]
