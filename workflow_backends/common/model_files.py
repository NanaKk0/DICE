from __future__ import annotations

import os
from pathlib import Path


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


def link_model_file(source: str | Path, target: str | Path) -> Path:
    """Expose a model file at target without duplicating model bytes."""
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
        target_path.symlink_to(source_path)
    except OSError as symlink_exc:
        try:
            os.link(source_path, target_path)
        except OSError as hardlink_exc:
            raise RuntimeError(
                "Unable to link model file without copying. "
                f"source={source_path}, target={target_path}. "
                "Enable symlinks for this workspace or keep task/model files on the same filesystem."
            ) from hardlink_exc
    return target_path
