from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


_STABLE_NAME = re.compile(
    r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9][A-Za-z0-9_-]*)*$"
)
_EMBEDDED_REVISION = re.compile(r"@\d+|(?:^|[._-])v\d+(?:$|[._-])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RevisionedIdentity:
    """A stable semantic name paired with an explicit format/policy revision."""

    name: str
    revision: int

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not _STABLE_NAME.fullmatch(name) or _EMBEDDED_REVISION.search(name):
            raise ValueError(f"invalid stable semantic name: {self.name!r}")
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or self.revision <= 0
        ):
            raise ValueError("revisioned identity requires a positive integer revision")
        object.__setattr__(self, "name", name)

    def as_dict(
        self,
        *,
        name_field: str = "name",
        revision_field: str = "revision",
    ) -> dict[str, object]:
        return {name_field: self.name, revision_field: self.revision}

    def matches(
        self,
        payload: Mapping[str, Any],
        *,
        name_field: str = "name",
        revision_field: str = "revision",
    ) -> bool:
        revision = payload.get(revision_field)
        return (
            payload.get(name_field) == self.name
            and revision == self.revision
            and not isinstance(revision, bool)
        )


__all__ = ["RevisionedIdentity"]
