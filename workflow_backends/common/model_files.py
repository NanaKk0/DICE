from __future__ import annotations

import os
from pathlib import Path
import shutil


def _resolve_existing_model_file(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model file not found: {path}") from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Model file cannot be resolved, likely due to a symlink loop: {path}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Model file not found: {resolved}")
    return resolved


def link_model_file(
    source: str | Path,
    target: str | Path,
    *,
    isolation_copy_fallback: bool = False,
) -> Path:
    """Present a model as a regular file, using a hardlink before byte-copy."""
    del isolation_copy_fallback  # retained for compatibility with older callers
    source_path = _resolve_existing_model_file(Path(source))
    target_raw = Path(target)
    try:
        target_path = target_raw.parent.resolve(strict=False) / target_raw.name
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Model target path cannot be resolved: {target_raw}") from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path == source_path:
        return target_path
    if target_path.exists() or target_path.is_symlink():
        try:
            if target_path.exists() and os.path.samefile(source_path, target_path):
                return target_path
        except OSError:
            pass
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)
    if target_path.is_symlink() or not target_path.is_file():
        raise RuntimeError(f"Model presentation is not a regular file: {target_path}")
    return target_path
