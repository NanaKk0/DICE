from __future__ import annotations

from pathlib import Path


def resolve_legacy_mace_teacher_model(member_id: str, legacy_handle: Path) -> Path:
    """Resolve one old lifecycle MACE handle from existing backend-owned evidence only."""

    handle = Path(legacy_handle)
    _validate_member(member_id, handle)
    if handle.is_symlink():
        target = handle.resolve()
        if target.suffix == ".model" and target.is_file() and target.stat().st_size > 0:
            return target
        raise ValueError(f"legacy MACE teacher symlink has no valid .model target: {handle}")
    if not handle.is_file() or handle.stat().st_size <= 0:
        raise ValueError(f"legacy MACE teacher handle is missing or empty: {handle}")
    matches = tuple(
        path.resolve()
        for path in handle.parent.iterdir()
        if path != handle
        and not path.is_symlink()
        and path.suffix == ".model"
        and path.is_file()
        and path.stat().st_size > 0
        and path.samefile(handle)
    )
    if len(matches) != 1:
        raise ValueError(
            f"legacy MACE teacher handle requires exactly one same-inode .model sibling: {handle}"
        )
    return matches[0]


def _validate_member(member_id: str, handle: Path) -> None:
    if not str(member_id).strip() or handle.name != "teacher_model" or handle.parent.name != member_id:
        raise ValueError(f"legacy MACE teacher handle does not match member {member_id!r}: {handle}")
