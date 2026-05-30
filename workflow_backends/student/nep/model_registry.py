from __future__ import annotations

from pathlib import Path
from typing import Any

from workflow_core.common.io import read_json, write_json


def write_nep_committee_manifest(summary_path: str | Path, payload: dict[str, Any]) -> Path:
    summary_path = Path(summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, payload)
    return summary_path


def _try_resolve_model_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw_path = Path(str(path))
    try:
        resolved = raw_path.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    return str(resolved) if resolved.is_file() else None


def _generation_sort_key(path: Path) -> tuple[int, int, str]:
    generation = -1
    name = path.name
    if name.startswith("nep_gen") and name.endswith(".txt"):
        try:
            generation = int(name[len("nep_gen"):-len(".txt")])
        except ValueError:
            generation = -1
    elif name.startswith("generation") and name.endswith(".txt"):
        try:
            generation = int(name[len("generation"):-len(".txt")])
        except ValueError:
            generation = -1
    resolved = Path(_try_resolve_model_file(path) or path)
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    return generation, mtime_ns, name


def _fallback_model_from_member(member: dict[str, Any]) -> str | None:
    member_dir = Path(str(member.get("member_dir", ""))) if member.get("member_dir") else None
    status_path = Path(str(member.get("stage_status_path", ""))) if member.get("stage_status_path") else None
    if status_path is None and member_dir is not None:
        status_path = member_dir / "stage_status.json"

    if status_path is not None and status_path.exists():
        try:
            status = read_json(status_path)
        except Exception:
            status = {}
        stages = status.get("stages", {}) if isinstance(status, dict) else {}
        if isinstance(stages, dict):
            stage_entries = [
                dict(entry)
                for entry in stages.values()
                if isinstance(entry, dict) and entry.get("status") == "success"
            ]
            stage_entries.sort(key=lambda item: int(item.get("stage_index", -1) or -1), reverse=True)
            for entry in stage_entries:
                for key in ("snapshot_model_path", "model_path"):
                    resolved = _try_resolve_model_file(entry.get(key))
                    if resolved is not None:
                        return resolved

    if member_dir is None:
        return None
    candidates: list[Path] = []
    for pattern in ("nep_stage_*.txt", "nep_gen*.txt", "generation*.txt"):
        candidates.extend(member_dir.glob(pattern))
    candidates = [path for path in candidates if _try_resolve_model_file(path) is not None]
    if not candidates:
        return None
    return _try_resolve_model_file(max(candidates, key=_generation_sort_key))


def collect_nep_model_paths(summary_payload: dict[str, Any]) -> list[str]:
    models = list(summary_payload.get("models", []) or [])
    members = list(summary_payload.get("members", []) or [])
    resolved_models: list[str] = []
    for index in range(max(len(models), len(members))):
        model_path = models[index] if index < len(models) else None
        resolved = _try_resolve_model_file(model_path)
        if resolved is None and index < len(members) and isinstance(members[index], dict):
            resolved = _fallback_model_from_member(members[index])
        if resolved is None and model_path is not None:
            raise RuntimeError(
                "Unable to resolve NEP committee model. "
                f"model_path={model_path}. The path may be missing or may contain a symlink loop."
            )
        if resolved is not None:
            resolved_models.append(resolved)
    return resolved_models
