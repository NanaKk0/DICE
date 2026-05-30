from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workflow_backends.teacher.deepmd.family import get_deepmd_family_adapter


def import_deepmd_teacher_committee(
    committee_dir: str | Path,
    *,
    model_family: str,
    type_map: list[str],
    teacher_num: int | None = None,
) -> dict[str, Any]:
    root = Path(committee_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"DeePMD teacher committee directory not found: {root}")
    adapter = get_deepmd_family_adapter(model_family)
    normalized_type_map = [str(item) for item in type_map]
    if not normalized_type_map:
        raise ValueError("type_map must be a non-empty list")

    teacher_dirs = sorted(path for path in root.glob("teacher_*") if path.is_dir())
    if teacher_num is None:
        teacher_num = len(teacher_dirs)
    teacher_num = int(teacher_num)
    if teacher_num <= 0:
        raise ValueError("teacher_num must be positive")

    members = []
    for teacher_index in range(1, teacher_num + 1):
        member_dir = root / f"teacher_{teacher_index:02d}"
        model_path = member_dir / adapter.primary_model_name(teacher_index)
        if not model_path.exists() or not model_path.is_file() or model_path.stat().st_size <= 0:
            raise RuntimeError(f"Missing non-empty DeePMD teacher primary model: {model_path}")
        metadata = {
            "backend": "deepmd",
            "model_family": adapter.name,
            "teacher_index": teacher_index,
            "type_map": normalized_type_map,
            "primary_model_path": str(model_path.resolve()),
        }
        _write_json(member_dir / "deepmd_teacher_metadata.json", metadata)
        members.append(metadata)

    root_metadata = {
        "backend": "deepmd",
        "model_family": adapter.name,
        "teacher_num": teacher_num,
        "type_map": normalized_type_map,
        "members": members,
        "imported_manual_committee": True,
    }
    _write_json(root / "deepmd_teacher_metadata.json", root_metadata)
    return root_metadata


def _write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved
