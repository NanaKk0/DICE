from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetHandle:
    """Path-free identity and exact order of one logical dataset."""

    dataset_id: str
    ordered_structure_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_id",
            _identity(self.dataset_id, label="dataset"),
        )
        object.__setattr__(
            self,
            "ordered_structure_ids",
            _ordered_identities(
                self.ordered_structure_ids,
                label="dataset structures",
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """Path-free identity of one ordered model member presentation."""

    member_id: str
    model_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "member_id",
            _identity(self.member_id, label="model member"),
        )
        object.__setattr__(
            self,
            "model_id",
            _identity(self.model_id, label="model"),
        )


@dataclass(frozen=True, slots=True)
class StructureBatchHandle:
    """Path-free identity and exact order of a bounded structure batch."""

    batch_id: str
    ordered_structure_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "batch_id",
            _identity(self.batch_id, label="structure batch"),
        )
        object.__setattr__(
            self,
            "ordered_structure_ids",
            _ordered_identities(
                self.ordered_structure_ids,
                label="structure batch members",
                allow_empty=True,
            ),
        )


def _identity(value: object, *, label: str) -> str:
    identity = str(value).strip()
    if not identity:
        raise ValueError(f"{label} identity must be non-empty")
    return identity


def _ordered_identities(
    values: tuple[str, ...],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    ordered = tuple(_identity(value, label=label) for value in values)
    if (not ordered and not allow_empty) or len(set(ordered)) != len(ordered):
        raise ValueError(f"{label} must be ordered and unique")
    return ordered


__all__ = ["DatasetHandle", "ModelHandle", "StructureBatchHandle"]
