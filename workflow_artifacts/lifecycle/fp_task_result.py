from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from workflow_backends.first_principles.protocols import FPLabelResult, FPPreparedTask
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)


def validate_fp_label_result(source: Atoms, result: FPLabelResult) -> None:
    task_id = result.task_id
    labeled = result.labeled_atoms
    structure_id = str(source.info.get("structure_id", ""))
    system_id = str(source.info.get("system_id", ""))
    if (
        len(source) != len(labeled)
        or source.get_chemical_symbols() != labeled.get_chemical_symbols()
        or not np.allclose(source.positions, labeled.positions, atol=1e-10, rtol=0.0)
        or not np.allclose(source.cell.array, labeled.cell.array, atol=1e-10, rtol=0.0)
        or not np.array_equal(source.pbc, labeled.pbc)
        or result.structure_id != structure_id
        or result.system_id != system_id
        or str(labeled.info.get("structure_id", "")) != structure_id
        or str(labeled.info.get("system_id", "")) != system_id
    ):
        raise ValueError(
            f"first-principles static structure relationship failed for {task_id}"
        )
    try:
        energy = float(labeled.get_potential_energy())
        forces = np.asarray(labeled.get_forces(), dtype=float)
    except Exception as exc:
        raise ValueError(
            f"first-principles result is missing energy or forces for {task_id}"
        ) from exc
    if forces.shape != (len(labeled), 3) or not np.isfinite(energy) or not np.all(
        np.isfinite(forces)
    ):
        raise ValueError(
            f"first-principles result has non-finite or malformed labels for {task_id}"
        )
    required = tuple(result.required_labels)
    if (
        len(required) != len(set(required))
        or required[:2] != ("energy", "forces")
        or any(item not in {"energy", "forces", "stress", "virial"} for item in required)
    ):
        raise ValueError(f"invalid first-principles label contract for {task_id}")
    if "stress" in required:
        try:
            stress = np.asarray(labeled.get_stress(voigt=True), dtype=float)
        except Exception as exc:
            raise ValueError(
                f"first-principles result is missing required stress for {task_id}"
            ) from exc
        if stress.shape != (6,) or not np.all(np.isfinite(stress)):
            raise ValueError(
                f"first-principles result has non-finite or malformed stress for {task_id}"
            )
    if "virial" in required:
        results = getattr(getattr(labeled, "calc", None), "results", {}) or {}
        virial_value = results.get("virial", labeled.info.get("virial"))
        if virial_value is None:
            raise ValueError(
                f"first-principles result is missing required virial for {task_id}"
            )
        virial = np.asarray(virial_value, dtype=float)
        if virial.shape not in {(6,), (3, 3), (9,)} or not np.all(np.isfinite(virial)):
            raise ValueError(
                f"first-principles result has non-finite or malformed virial for {task_id}"
            )


def write_validated_fp_result(task: FPPreparedTask, result: FPLabelResult) -> None:
    metadata_path = Path(task.task_path) / "validated_result.json"
    result_path = Path(task.task_path) / "validated_result.xyz"
    staged_result = result_path.with_suffix(".xyz.tmp")
    write_extxyz_structures(staged_result, [result.labeled_atoms], allow_empty=False)
    staged_result.replace(result_path)
    payload = {
        "schema": "dice.fp-validated-task-result",
        "schema_revision": 2,
        "task_id": result.task_id,
        "selected_index": result.selected_index,
        "structure_id": result.structure_id,
        "system_id": result.system_id,
        "input_digest": result.input_digest,
        "required_labels": list(result.required_labels),
        "parser_evidence": _json_safe(dict(result.parser_evidence)),
    }
    staged_metadata = metadata_path.with_suffix(".json.tmp")
    staged_metadata.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    staged_metadata.replace(metadata_path)


def read_validated_fp_result(
    task: FPPreparedTask, *, source: Atoms
) -> FPLabelResult | None:
    metadata_path = Path(task.task_path) / "validated_result.json"
    result_path = Path(task.task_path) / "validated_result.xyz"
    if not metadata_path.is_file() or not result_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema",
            "schema_revision",
            "task_id",
            "selected_index",
            "structure_id",
            "system_id",
            "input_digest",
            "required_labels",
            "parser_evidence",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or payload.get("schema") != "dice.fp-validated-task-result"
            or payload.get("schema_revision") != 2
            or payload.get("task_id") != task.task_id
            or payload.get("selected_index") != task.selected_index
            or payload.get("structure_id") != task.structure_id
            or payload.get("system_id") != task.system_id
            or payload.get("input_digest") != task.input_digest
            or payload.get("required_labels") != list(task.required_labels)
            or not isinstance(payload.get("parser_evidence"), dict)
        ):
            return None
        labeled = read_extxyz_structures(result_path, allow_empty=False)
        if len(labeled) != 1:
            return None
        result = FPLabelResult(
            task_id=task.task_id,
            selected_index=task.selected_index,
            structure_id=task.structure_id,
            system_id=task.system_id,
            labeled_atoms=labeled[0],
            input_digest=task.input_digest,
            parser_evidence=dict(payload["parser_evidence"]),
            required_labels=tuple(payload["required_labels"]),
        )
        validate_fp_label_result(source, result)
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("first-principles parser evidence is not JSON serializable")


__all__ = [
    "read_validated_fp_result",
    "validate_fp_label_result",
    "write_validated_fp_result",
]
