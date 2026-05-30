from __future__ import annotations

from pathlib import Path
from typing import Any


_GLOB_CHARS = set("*?[]")


def infer_workspace_root_from_config(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd().resolve()
    resolved = Path(config_path).resolve()
    if "configs" in resolved.parts:
        parts = list(resolved.parts)
        configs_index = parts.index("configs")
        if configs_index > 0:
            return Path(*parts[:configs_index]).resolve()
    return resolved.parent.resolve()


def validate_path_only_xyz_list(
    payload: Any,
    *,
    field_name: str,
    bare_filename_root: str | Path,
    allow_empty: bool = False,
) -> list[Path]:
    return validate_source_asset_declarations(
        payload,
        field_name=field_name,
        bare_filename_root=bare_filename_root,
        allow_empty=allow_empty,
    )


def validate_source_asset_declarations(
    payload: Any,
    *,
    field_name: str,
    bare_filename_root: str | Path,
    allow_empty: bool = False,
) -> list[Path]:
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a non-empty list")
    if not payload:
        if allow_empty:
            return []
        raise ValueError(f"{field_name} must be a non-empty list")

    resolved_paths: list[Path] = []
    seen: set[str] = set()
    asset_root = Path(bare_filename_root).resolve()
    for index, value in enumerate(payload, start=1):
        if not isinstance(value, str):
            raise ValueError(f"{field_name} entry #{index} must be path strings")
        raw_value = value.strip()
        if not raw_value:
            raise ValueError(f"{field_name} entry #{index} must be a non-empty path string")
        if any(char in raw_value for char in _GLOB_CHARS):
            raise ValueError(f"{field_name} entry #{index} must not contain glob patterns: {raw_value}")

        path = Path(raw_value)
        if path.suffix.lower() != ".xyz":
            raise ValueError(f"{field_name} entry #{index} must point to a .xyz file: {raw_value}")
        if path.is_absolute():
            resolved = path.resolve()
        else:
            if path.name != raw_value:
                raise ValueError(
                    f"{field_name} entry #{index} must be an absolute path or bare filenames only: {raw_value}"
                )
            resolved = (asset_root / raw_value).resolve()

        if not resolved.exists():
            raise ValueError(f"{field_name} entry #{index} not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"{field_name} entry #{index} is not a file: {resolved}")
        resolved_key = str(resolved)
        if resolved_key in seen:
            raise ValueError(f"{field_name} contains duplicate resolved path: {resolved}")
        _validate_canonical_extxyz_source_asset(resolved, field_name=field_name, allow_empty=allow_empty)
        seen.add(resolved_key)
        resolved_paths.append(resolved)
    return resolved_paths


def _validate_canonical_extxyz_source_asset(path: Path, *, field_name: str, allow_empty: bool) -> None:
    try:
        from workflow_core.structures.formats import read_extxyz_structures

        read_extxyz_structures(path, allow_empty=allow_empty)
    except ValueError as exc:
        raise ValueError(f"{field_name} entry is not a valid canonical extxyz source asset: {exc}") from exc
    except ModuleNotFoundError:
        if path.stat().st_size == 0 and not allow_empty:
            raise ValueError(f"{field_name} entry is not a valid canonical extxyz source asset: empty extxyz is not allowed: {path}")


def reject_nested_source_toggle(payload: dict[str, Any], *, field_name: str, forbidden_keys: set[str]) -> None:
    found: list[str] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            next_path = (*path, str(key))
            if key in forbidden_keys:
                found.append(".".join(next_path))
            visit(child, next_path)

    visit(payload, ())
    if found:
        raise ValueError(f"{field_name} rejects source-specific dataset toggles: {', '.join(sorted(found))}")
