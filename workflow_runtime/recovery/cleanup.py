from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from workflow_runtime.recovery.recovery_plan import RecoveryNode


class RecoveryCleanupError(RuntimeError):
    pass


def cleanup_recovery_nodes(
    nodes: Iterable[RecoveryNode],
    *,
    teacher_round: int,
    run_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Remove path-owned artifacts, logs, backend logs, and markers for recovery nodes."""

    owned_paths: list[Path] = []
    for node in nodes:
        if int(node.teacher_round) != int(teacher_round):
            continue
        owned_paths.extend(Path(path).resolve() for path in node.artifact_roots)
        owned_paths.extend(Path(path).resolve() for path in node.semantic_log_roots)
        owned_paths.extend(Path(path).resolve() for path in node.backend_log_roots)
        owned_paths.extend(Path(path).resolve() for path in node.marker_paths)

    resolved_root = Path(run_root).resolve() if run_root is not None else _infer_run_root(owned_paths)
    removed: list[Path] = []
    for path in _dedupe_children_before_parents(owned_paths):
        _assert_within(path, resolved_root)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return tuple(removed)


def _infer_run_root(paths: list[Path]) -> Path:
    for path in paths:
        parts = list(path.resolve().parts)
        for canonical in ("artifacts", "runtime", "logs"):
            if canonical not in parts:
                continue
            index = parts.index(canonical)
            if index > 0:
                return Path(*parts[:index]).resolve()
    if paths:
        return paths[0].resolve().parent
    return Path.cwd().resolve()


def _assert_within(path: Path, root: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RecoveryCleanupError(f"cleanup path is outside run root: {resolved}") from exc


def _dedupe_children_before_parents(paths: list[Path]) -> list[Path]:
    unique = sorted({path.resolve() for path in paths}, key=lambda item: len(item.parts))
    result: list[Path] = []
    for path in unique:
        if any(_is_relative_to(path, existing) for existing in result):
            continue
        result.append(path)
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
