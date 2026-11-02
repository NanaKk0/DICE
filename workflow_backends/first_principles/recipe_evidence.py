from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


_REFERENCE_FIELDS = {
    "CP2K": ("template", "assets"),
    "ABACUS": ("template", "pseudopotentials", "orbitals", "assets"),
}


def fp_recipe_input_paths(
    *, run_root: str | Path, calculator: str, recipe: str
) -> tuple[str, ...]:
    """Resolve explicit recipe-owned files for static recipe validation.

    This backend boundary identifies path-bearing manifest fields only. Strict
    scientific validation remains owned by each calculator recipe adapter.
    """
    root = Path(run_root).resolve()
    recipe_relative = _safe_relative_path(recipe, field="fp.label.recipe")
    recipe_path = (root / recipe_relative).resolve()
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read first-principles recipe {recipe_relative}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"first-principles recipe {recipe_relative} is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("first-principles recipe must be a JSON object")
    if calculator not in _REFERENCE_FIELDS:
        raise ValueError(f"unsupported first-principles calculator {calculator!r}")

    references: list[str] = []
    for field in _REFERENCE_FIELDS[calculator]:
        value = payload.get(field)
        if field == "template":
            if isinstance(value, str):
                references.append(value)
            continue
        if value is None:
            continue
        if not isinstance(value, dict):
            # The strict adapter reports the calculator-facing shape error.
            continue
        references.extend(item for item in value.values() if isinstance(item, str))

    result = [recipe_relative.as_posix()]
    for reference in _ordered_unique(references):
        child = _safe_relative_path(reference, field="first-principles recipe asset")
        resolved = (recipe_path.parent / child).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("first-principles recipe asset escapes the run root") from exc
        if not resolved.is_file():
            raise ValueError(
                f"first-principles recipe asset does not exist: {relative.as_posix()}"
            )
        result.append(relative.as_posix())
    return tuple(_ordered_unique(result))


def _safe_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a relative path")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or any(part == ".." for part in candidate.parts)
        or any(char in value for char in "*?[")
    ):
        raise ValueError(f"{field} must be a relative non-glob path")
    return candidate


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = ["fp_recipe_input_paths"]
