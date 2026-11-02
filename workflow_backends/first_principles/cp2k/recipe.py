from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


_RECIPE_KEYS = {"template", "kpoints", "assets", "require_stress"}
_RESERVED_INPUTS = {
    "coordinates.xyz",
    "input.inp",
    "selected.xyz",
    "task_context.json",
    "fp.stdout.log",
    "fp.stderr.log",
    "validated_result.json",
    "validated_result.xyz",
}


@dataclass(frozen=True, slots=True)
class CP2KRecipeAsset:
    destination: Path
    source: Path


@dataclass(frozen=True, slots=True)
class CP2KRecipe:
    path: Path
    template_path: Path
    template: str
    spacing_inv_angstrom: float | None
    assets: tuple[CP2KRecipeAsset, ...]
    require_stress: bool


def load_cp2k_recipe(path: str | Path) -> CP2KRecipe:
    recipe_path = Path(path).resolve()
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read CP2K recipe {recipe_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("CP2K recipe is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("CP2K recipe must be a JSON object")
    unknown = set(payload) - _RECIPE_KEYS
    if unknown or "template" not in payload:
        raise ValueError(
            f"CP2K recipe requires template and has unknown keys: {sorted(unknown)}"
        )
    root = recipe_path.parent.resolve()
    template_path = _source_path(root, payload["template"], field="template")
    template = template_path.read_text(encoding="utf-8")
    _validate_template(template)
    spacing = _parse_kpoints(payload.get("kpoints"))
    require_stress = payload.get("require_stress", False)
    if not isinstance(require_stress, bool):
        raise ValueError("CP2K recipe require_stress must be boolean")
    if require_stress and "STRESS_TENSOR" not in template.upper():
        raise ValueError(
            "CP2K recipe require_stress needs calculator stress output in the template"
        )
    assets_value = payload.get("assets", {})
    if not isinstance(assets_value, dict):
        raise ValueError("CP2K recipe assets must be a destination-to-source object")
    assets: list[CP2KRecipeAsset] = []
    for destination_value, source_value in sorted(assets_value.items()):
        destination = _relative_path(destination_value, field="asset destination")
        if destination.as_posix() in _RESERVED_INPUTS:
            raise ValueError(
                f"CP2K recipe asset destination is reserved: {destination.as_posix()}"
            )
        assets.append(
            CP2KRecipeAsset(
                destination=destination,
                source=_source_path(root, source_value, field="asset source"),
            )
        )
    return CP2KRecipe(
        path=recipe_path,
        template_path=template_path,
        template=template,
        spacing_inv_angstrom=spacing,
        assets=tuple(assets),
        require_stress=require_stress,
    )


def _parse_kpoints(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "spacing_inv_angstrom",
    }:
        raise ValueError("CP2K recipe kpoints must use the exact typed spacing shape")
    spacing = value.get("spacing_inv_angstrom")
    if (
        value.get("mode") != "spacing"
        or isinstance(spacing, bool)
        or not isinstance(spacing, (int, float))
        or not math.isfinite(float(spacing))
        or float(spacing) <= 0.0
    ):
        raise ValueError("CP2K recipe kpoints spacing must be positive and finite")
    return float(spacing)


def _validate_template(template: str) -> None:
    active = "\n".join(line.split("!", 1)[0] for line in template.splitlines())
    run_types = re.findall(r"\bRUN_TYPE\s+([A-Z_]+)", active.upper())
    if run_types != ["ENERGY_FORCE"]:
        raise ValueError(
            "CP2K recipe template must contain exactly one RUN_TYPE ENERGY_FORCE"
        )
    required_sets = {
        "COORDFILE",
        "CELL_A",
        "CELL_B",
        "CELL_C",
        "ANGLE_A",
        "ANGLE_B",
        "ANGLE_C",
    }
    present = set(
        re.findall(r"^\s*@SET\s+([A-Z_]+)\b", active.upper(), flags=re.MULTILINE)
    )
    missing = sorted(required_sets - present)
    if missing:
        raise ValueError(
            f"CP2K recipe template is missing required @SET variables: {missing}"
        )


def _source_path(root: Path, value: Any, *, field: str) -> Path:
    relative = _relative_path(value, field=field)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"CP2K recipe {field} escapes its directory") from exc
    if not resolved.is_file():
        raise ValueError(f"CP2K recipe {field} does not exist: {relative.as_posix()}")
    return resolved


def _relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CP2K recipe {field} must be a relative path")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(char in value for char in "*?[")
    ):
        raise ValueError(f"CP2K recipe {field} must be a safe relative path")
    return candidate


__all__ = ["CP2KRecipe", "CP2KRecipeAsset", "load_cp2k_recipe"]
