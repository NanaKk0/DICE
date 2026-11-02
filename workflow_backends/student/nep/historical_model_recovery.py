from __future__ import annotations

from pathlib import Path


def resolve_legacy_nep_student_model(member_id: str, legacy_handle: Path) -> Path:
    """Resolve one old lifecycle NEP student handle from its native model/restart pair."""

    handle = Path(legacy_handle)
    if not str(member_id).strip() or handle.name != "student_model" or handle.parent.name != member_id:
        raise ValueError(f"legacy NEP student handle does not match member {member_id!r}: {handle}")
    restart = handle.parent / "nep.restart"
    if restart.is_symlink() or not restart.is_file() or restart.stat().st_size <= 0:
        raise ValueError(f"legacy NEP student recovery requires a non-empty native nep.restart: {restart}")
    if handle.is_symlink():
        target = handle.resolve()
        if _valid_nep_model(target):
            return target
        raise ValueError(f"legacy NEP student symlink has no valid .txt target: {handle}")
    if not handle.is_file() or handle.stat().st_size <= 0:
        raise ValueError(f"legacy NEP student handle is missing or empty: {handle}")
    native = handle.parent / "nep.txt"
    if not _valid_nep_model(native) or not native.samefile(handle):
        raise ValueError(f"legacy NEP student handle does not match native nep.txt: {handle}")
    return native.resolve()


def _valid_nep_model(path: Path) -> bool:
    return (
        path.suffix == ".txt"
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size > 0
    )
