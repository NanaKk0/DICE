from __future__ import annotations

import json
from pathlib import Path


_NATIVE_SUFFIXES = (".pb", ".pt2")


def resolve_legacy_deepmd_student_model(member_id: str, legacy_handle: Path) -> Path:
    """Resolve one old lifecycle DeepMD student handle from explicit native evidence."""

    handle = Path(legacy_handle)
    if not str(member_id).strip() or handle.name != "student_model" or handle.parent.name != member_id:
        raise ValueError(f"legacy DeepMD student handle does not match member {member_id!r}: {handle}")
    metadata_suffix = _metadata_suffix(handle.parent / "deepmd_student_model_metadata.json")
    if handle.is_symlink():
        target = handle.resolve()
        if _valid_target(target, metadata_suffix):
            return target
        raise ValueError(f"legacy DeepMD student symlink has conflicting native evidence: {handle}")
    if not handle.is_file() or handle.stat().st_size <= 0:
        raise ValueError(f"legacy DeepMD student handle is missing or empty: {handle}")
    candidates = tuple(
        path.resolve()
        for path in (handle.parent / "student_native.pb", handle.parent / "student_native.pt2")
        if _valid_target(path, metadata_suffix) and path.samefile(handle)
    )
    if len(candidates) != 1:
        raise ValueError(
            "legacy DeepMD student handle requires one matching "
            f"student_native.pb/student_native.pt2 sibling: {handle}"
        )
    return candidates[0]


def _metadata_suffix(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"legacy DeepMD student metadata is invalid: {path}") from exc
    suffix = payload.get("native_model_suffix")
    if suffix not in _NATIVE_SUFFIXES:
        raise ValueError(f"legacy DeepMD student metadata has invalid native_model_suffix: {path}")
    return str(suffix)


def _valid_target(path: Path, metadata_suffix: str | None) -> bool:
    return (
        path.suffix in _NATIVE_SUFFIXES
        and (metadata_suffix is None or path.suffix == metadata_suffix)
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size > 0
    )
