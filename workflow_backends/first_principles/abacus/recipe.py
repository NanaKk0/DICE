from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ase.data import atomic_numbers


_RECIPE_KEYS = {
    "template",
    "pseudopotentials",
    "orbitals",
    "kpoints",
    "assets",
    "require_stress",
}
_RESERVED_INPUTS = {
    "INPUT",
    "KPT",
    "STRU",
    "selected.xyz",
    "task_context.json",
    "fp.stdout.log",
    "fp.stderr.log",
    "validated_result.json",
    "validated_result.xyz",
}
_RESERVED_DIRECTORIES = {"pseudopotentials", "orbitals", "OUT.ABACUS"}


@dataclass(frozen=True, slots=True)
class ABACUSElementAsset:
    element: str
    source: Path
    destination: Path


@dataclass(frozen=True, slots=True)
class ABACUSRecipeAsset:
    destination: Path
    source: Path


@dataclass(frozen=True, slots=True)
class ABACUSRecipe:
    path: Path
    template_path: Path
    template: str
    basis_type: str
    pseudopotentials: Mapping[str, Path]
    orbitals: Mapping[str, Path] | None
    spacing_inv_angstrom: float | None
    assets: tuple[ABACUSRecipeAsset, ...]
    require_stress: bool

    def element_assets(
        self, elements: tuple[str, ...]
    ) -> tuple[tuple[ABACUSElementAsset, ...], tuple[ABACUSElementAsset, ...]]:
        selected = set(elements)
        pseudo_elements = set(self.pseudopotentials)
        if pseudo_elements != selected:
            raise ValueError(
                "ABACUS recipe pseudopotentials must exactly cover selected "
                f"elements; selected={sorted(selected)}, mapped={sorted(pseudo_elements)}"
            )
        if self.basis_type == "pw":
            if self.orbitals is not None:
                raise ValueError("ABACUS recipe orbitals must be omitted for a PW template")
            orbital_sources: Mapping[str, Path] = {}
        else:
            if self.orbitals is None:
                raise ValueError("ABACUS recipe orbitals are required for an LCAO template")
            orbital_elements = set(self.orbitals)
            if orbital_elements != selected:
                raise ValueError(
                    "ABACUS recipe orbitals must exactly cover selected elements; "
                    f"selected={sorted(selected)}, mapped={sorted(orbital_elements)}"
                )
            orbital_sources = self.orbitals
        pseudos = tuple(
            ABACUSElementAsset(
                element=element,
                source=self.pseudopotentials[element],
                destination=Path("pseudopotentials")
                / f"{element}{self.pseudopotentials[element].suffix}",
            )
            for element in elements
        )
        orbitals = tuple(
            ABACUSElementAsset(
                element=element,
                source=orbital_sources[element],
                destination=Path("orbitals")
                / f"{element}{orbital_sources[element].suffix}",
            )
            for element in elements
            if element in orbital_sources
        )
        return pseudos, orbitals


def load_abacus_recipe(path: str | Path) -> ABACUSRecipe:
    recipe_path = Path(path).resolve()
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read ABACUS recipe {recipe_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("ABACUS recipe is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("ABACUS recipe must be a JSON object")
    unknown = set(payload) - _RECIPE_KEYS
    required = {"template", "pseudopotentials"}
    missing = required - set(payload)
    if unknown or missing:
        raise ValueError(
            "ABACUS recipe requires template and pseudopotentials; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    root = recipe_path.parent.resolve()
    template_path = _source_path(root, payload["template"], field="template")
    template = template_path.read_text(encoding="utf-8")
    basis_type = _validate_template(template)
    pseudopotentials = _element_mapping(
        root, payload["pseudopotentials"], field="pseudopotentials"
    )
    orbitals_value = payload.get("orbitals")
    orbitals = (
        None
        if orbitals_value is None
        else _element_mapping(root, orbitals_value, field="orbitals")
    )
    spacing = _parse_kpoints(payload.get("kpoints"))
    if spacing is not None:
        _validate_typed_kpoints_template(template)
    require_stress = payload.get("require_stress", False)
    if not isinstance(require_stress, bool):
        raise ValueError("ABACUS recipe require_stress must be boolean")
    if require_stress and _input_value(template, "cal_stress") != "1":
        raise ValueError("ABACUS recipe require_stress needs template cal_stress 1")
    assets_value = payload.get("assets", {})
    if not isinstance(assets_value, dict):
        raise ValueError("ABACUS recipe assets must be a destination-to-source object")
    assets: list[ABACUSRecipeAsset] = []
    for destination_value, source_value in sorted(assets_value.items()):
        destination = _relative_path(destination_value, field="asset destination")
        if (
            destination.as_posix() in _RESERVED_INPUTS
            or destination.parts[0] in _RESERVED_DIRECTORIES
        ):
            raise ValueError(
                "ABACUS recipe asset destination is reserved: "
                f"{destination.as_posix()}"
            )
        assets.append(
            ABACUSRecipeAsset(
                destination=destination,
                source=_source_path(root, source_value, field="asset source"),
            )
        )
    return ABACUSRecipe(
        path=recipe_path,
        template_path=template_path,
        template=template,
        basis_type=basis_type,
        pseudopotentials=pseudopotentials,
        orbitals=orbitals,
        spacing_inv_angstrom=spacing,
        assets=tuple(assets),
        require_stress=require_stress,
    )


def _validate_template(template: str) -> str:
    calculation = _input_value(template, "calculation")
    if calculation != "scf":
        raise ValueError("ABACUS recipe template must select static scf calculation")
    if _input_value(template, "cal_force") != "1":
        raise ValueError("ABACUS recipe template must contain cal_force 1")
    basis_type = _input_value(template, "basis_type")
    if basis_type not in {"pw", "lcao"}:
        raise ValueError("ABACUS recipe template basis_type must be PW or LCAO")
    _require_input_default(template, "suffix", accepted={"abacus"})
    _require_input_default(template, "stru_file", accepted={"stru"})
    _require_input_default(template, "pseudo_dir", accepted={".", "./"})
    if basis_type == "lcao":
        _require_input_default(template, "orbital_dir", accepted={".", "./"})
    return basis_type


def _validate_typed_kpoints_template(template: str) -> None:
    if _input_value(template, "kspacing") is not None:
        raise ValueError(
            "ABACUS recipe typed kpoints cannot be combined with INPUT kspacing"
        )
    _require_input_default(template, "kpoint_file", accepted={"kpt"})
    _require_input_default(template, "gamma_only", accepted={"0", "false"})


def _require_input_default(
    template: str, key: str, *, accepted: set[str]
) -> None:
    value = _input_value(template, key)
    if value is not None and value not in accepted:
        raise ValueError(
            f"ABACUS recipe template {key} must retain the task-local default"
        )


def _input_value(template: str, key: str) -> str | None:
    values: list[str] = []
    for raw in template.splitlines():
        active = raw.split("#", 1)[0].strip()
        fields = active.split()
        if fields and fields[0].lower() == key.lower() and len(fields) >= 2:
            values.append(fields[1].lower())
    if len(values) > 1:
        raise ValueError(f"ABACUS recipe template has duplicate {key} values")
    return values[0] if values else None


def _element_mapping(root: Path, value: Any, *, field: str) -> Mapping[str, Path]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"ABACUS recipe {field} must be a non-empty element mapping")
    result: dict[str, Path] = {}
    for element, source in sorted(value.items()):
        if not isinstance(element, str) or element not in atomic_numbers:
            raise ValueError(f"ABACUS recipe {field} has invalid element {element!r}")
        result[element] = _source_path(root, source, field=f"{field}.{element}")
    return result


def _parse_kpoints(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "spacing_inv_angstrom",
    }:
        raise ValueError("ABACUS recipe kpoints must use the exact typed spacing shape")
    spacing = value.get("spacing_inv_angstrom")
    if (
        value.get("mode") != "spacing"
        or isinstance(spacing, bool)
        or not isinstance(spacing, (int, float))
        or not math.isfinite(float(spacing))
        or float(spacing) <= 0.0
    ):
        raise ValueError("ABACUS recipe kpoints spacing must be positive and finite")
    return float(spacing)


def _source_path(root: Path, value: Any, *, field: str) -> Path:
    relative = _relative_path(value, field=field)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"ABACUS recipe {field} escapes its directory") from exc
    if not resolved.is_file():
        raise ValueError(
            f"ABACUS recipe {field} does not exist: {relative.as_posix()}"
        )
    return resolved


def _relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ABACUS recipe {field} must be a relative path")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(char in value for char in "*?[")
    ):
        raise ValueError(f"ABACUS recipe {field} must be a safe relative path")
    return candidate


__all__ = [
    "ABACUSElementAsset",
    "ABACUSRecipe",
    "ABACUSRecipeAsset",
    "load_abacus_recipe",
]
