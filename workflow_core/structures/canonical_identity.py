from __future__ import annotations

from copy import deepcopy
import re
from dataclasses import dataclass, field

from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator

from workflow_core.structures.exact_identity import exact_structure_hash

VALID_STRUCTURE_SOURCES = frozenset({"external", "workflow"})
_STRUCTURE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SINGLEPOINT_RESULT_KEYS = frozenset(all_properties)


@dataclass
class IdentityAudit:
    generated_structure_ids: list[dict[str, str]] = field(default_factory=list)
    formula_fallbacks: list[dict[str, str]] = field(default_factory=list)
    declared_system_overwrites: list[dict[str, str]] = field(default_factory=list)
    ignored_external_counts: dict[str, int] = field(default_factory=dict)
    external_dedup_dropped_counts: dict[str, int] = field(default_factory=dict)

    def record_generated_structure_id(self, structure_id: str) -> None:
        self.generated_structure_ids.append({"structure_id": _validate_structure_id(structure_id)})

    def record_formula_fallback(self, *, system_id: str, formula: str) -> None:
        self.formula_fallbacks.append({"system_id": _validate_system_id(system_id), "formula": str(formula)})

    def record_declared_system_overwrite(self, *, field: str, previous: object, replacement: object) -> None:
        self.declared_system_overwrites.append(
            {"field": str(field), "previous": str(previous), "replacement": str(replacement)}
        )

    def record_ignored_external(self, *, route: str, count: int) -> None:
        self._increment(self.ignored_external_counts, route, count)

    def record_external_dedup_drop(self, *, route: str, count: int) -> None:
        self._increment(self.external_dedup_dropped_counts, route, count)

    def to_payload(self) -> dict[str, object]:
        return {
            "generated_structure_ids": list(self.generated_structure_ids),
            "formula_fallbacks": list(self.formula_fallbacks),
            "declared_system_overwrites": list(self.declared_system_overwrites),
            "ignored_external_counts": dict(self.ignored_external_counts),
            "external_dedup_dropped_counts": dict(self.external_dedup_dropped_counts),
        }

    @staticmethod
    def _increment(target: dict[str, int], route: str, count: int) -> None:
        value = int(count)
        if value < 0:
            raise ValueError("identity audit counts must be non-negative")
        if value == 0:
            return
        key = str(route)
        target[key] = target.get(key, 0) + value


def structure_id_for_atoms(atoms: Atoms) -> str:
    return f"sha256:{exact_structure_hash(atoms)}"


def formula_system_id(atoms: Atoms) -> str:
    return f"formula:{atoms.get_chemical_formula(mode='hill')}"


def validate_structure_source(value: object) -> str:
    source = str(value).strip()
    if source not in VALID_STRUCTURE_SOURCES:
        allowed = ", ".join(sorted(VALID_STRUCTURE_SOURCES))
        raise ValueError(f"structure_source must be one of: {allowed}")
    return source


def stamp_canonical_identity(atoms: Atoms, *, system_id: str, structure_source: str) -> Atoms:
    item = copy_atoms_preserving_singlepoint_results(atoms)
    item.info["system_id"] = _validate_system_id(system_id)
    item.info["structure_id"] = structure_id_for_atoms(item)
    item.info["structure_source"] = validate_structure_source(structure_source)
    return item


def stamp_external_ingress_identity(
    atoms: Atoms,
    *,
    declared_system_id: str | None = None,
    audit: IdentityAudit | None = None,
) -> Atoms:
    item = copy_atoms_preserving_singlepoint_results(atoms)
    info = item.info or {}
    retired_aliases = ("seed_system_id", "seed_structure_id", "source_system_id", "source_structure_id")
    if declared_system_id is not None:
        system_id = _validate_system_id(declared_system_id)
        if audit is not None:
            for field_name in ("system_id", "seed_system_id"):
                previous = info.get(field_name)
                if previous is not None and str(previous).strip() and str(previous).strip() != system_id:
                    audit.record_declared_system_overwrite(
                        field=field_name,
                        previous=str(previous).strip(),
                        replacement=system_id,
                    )
    else:
        existing_system_id = str(info.get("system_id", "") or "").strip()
        if existing_system_id:
            system_id = existing_system_id
        else:
            system_id = formula_system_id(item)
            if audit is not None:
                audit.record_formula_fallback(system_id=system_id, formula=item.get_chemical_formula(mode="hill"))
    item.info["system_id"] = system_id
    item.info["structure_id"] = structure_id_for_atoms(item)
    item.info["structure_source"] = "external"
    for alias in retired_aliases:
        item.info.pop(alias, None)
    if audit is not None:
        audit.record_generated_structure_id(str(item.info["structure_id"]))
    return item


def copy_atoms_preserving_singlepoint_results(atoms: Atoms) -> Atoms:
    item = atoms.copy()
    results = getattr(getattr(atoms, "calc", None), "results", None)
    if not isinstance(results, dict) or not results:
        return item
    preserved = {key: deepcopy(value) for key, value in results.items() if key in _SINGLEPOINT_RESULT_KEYS}
    if preserved:
        item.calc = SinglePointCalculator(item, **preserved)
    return item


def validate_canonical_identity(atoms: Atoms) -> dict[str, str]:
    info = atoms.info or {}
    system_id = _require_info_string(info, "system_id")
    structure_id = _validate_structure_id(_require_info_string(info, "structure_id"))
    structure_source = _require_info_string(info, "structure_source")

    return {
        "system_id": _validate_system_id(system_id),
        "structure_id": structure_id,
        "structure_source": validate_structure_source(structure_source),
    }


def _validate_system_id(value: object) -> str:
    system_id = str(value).strip()
    if not system_id:
        raise ValueError("system_id is missing or empty")
    return system_id


def _validate_structure_id(value: object) -> str:
    structure_id = str(value).strip()
    if not _STRUCTURE_ID_PATTERN.fullmatch(structure_id):
        raise ValueError("structure_id must use public sha256:<64hex> format")
    return structure_id


def _require_info_string(info: dict, key: str) -> str:
    value = info.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} is missing from canonical structure identity")
    return str(value).strip()
