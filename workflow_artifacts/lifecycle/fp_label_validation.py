from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from workflow_backends.first_principles.protocols import fp_task_id
from workflow_artifacts.lifecycle.fp_task_result import validate_fp_label_result
from workflow_core.structures.formats import read_extxyz_structures


def validate_fp_label_artifacts(
    *,
    label_root: str | Path,
    selected_path: str | Path,
    calculator: str,
    facts: object | None = None,
) -> bool:
    """Validate FP labels against selected frames and optional expected semantics."""
    root = Path(label_root)
    try:
        if (
            root.is_symlink()
            or not root.is_dir()
            or Path(selected_path).is_symlink()
            or (root / "labeled.xyz").is_symlink()
        ):
            return False
        selected = read_extxyz_structures(selected_path, allow_empty=True)
        if facts is not None:
            structure_ids = tuple(
                str(frame.info.get("structure_id", "")) for frame in selected
            )
            system_ids = tuple(
                str(frame.info.get("system_id", "")) for frame in selected
            )
            task_ids = tuple(fp_task_id(index) for index in range(len(selected)))
            if (
                facts.calculator != str(calculator).upper()
                or facts.selection.ordered_structure_ids != structure_ids
                or facts.system_ids != system_ids
                or facts.task_ids != task_ids
            ):
                return False
        labeled = read_extxyz_structures(root / "labeled.xyz", allow_empty=True)
        if len(labeled) != len(selected):
            return False
        # Reuse the same public relationship/label rules with artifact-shaped
        # result records while keeping task parser details diagnostic.
        from workflow_backends.first_principles.protocols import FPLabelResult

        results = []
        expected_task_names = []
        for index, (source, item) in enumerate(zip(selected, labeled, strict=True)):
            task_id = fp_task_id(index)
            expected_task_names.append(task_id)
            task_path = root / "tasks" / task_id
            task_selected = read_extxyz_structures(
                task_path / "selected.xyz", allow_empty=False
            )
            context = json.loads(
                (task_path / "task_context.json").read_text(encoding="utf-8")
            )
            required_labels = (
                context.get("required_labels") if isinstance(context, dict) else None
            )
            operation = context.get("operation") if isinstance(context, dict) else None
            native_inputs = (
                context.get("native_inputs") if isinstance(context, dict) else None
            )
            if (
                not isinstance(context, dict)
                or context.get("task_id") != task_id
                or context.get("selected_index") != index
                or context.get("structure_id") != source.info.get("structure_id")
                or context.get("system_id") != source.info.get("system_id")
                or context.get("calculator") != calculator
                or not isinstance(context.get("input_digest"), str)
                or not _validate_task_input_digest(
                    task_path,
                    source=source,
                    task_id=task_id,
                    selected_index=index,
                    structure_id=str(source.info.get("structure_id", "")),
                    system_id=str(source.info.get("system_id", "")),
                    input_digest=str(context["input_digest"]),
                    required_labels=required_labels,
                )
                or not isinstance(required_labels, list)
                or not all(isinstance(label, str) for label in required_labels)
                or not _valid_operation_evidence(operation, calculator=calculator)
                or not _valid_native_input_inventory(
                    native_inputs, task_path=task_path
                )
                or len(task_selected) != 1
                or task_selected[0].get_chemical_symbols()
                != source.get_chemical_symbols()
                or not np.allclose(
                    task_selected[0].positions,
                    source.positions,
                    atol=1.0e-10,
                    rtol=0.0,
                )
                or not np.allclose(
                    task_selected[0].cell.array,
                    source.cell.array,
                    atol=1.0e-10,
                    rtol=0.0,
                )
                or not np.array_equal(task_selected[0].pbc, source.pbc)
                or task_selected[0].info.get("structure_id")
                != source.info.get("structure_id")
                or task_selected[0].info.get("system_id")
                != source.info.get("system_id")
            ):
                return False
            results.append(
                FPLabelResult(
                    task_id=task_id,
                    selected_index=index,
                    structure_id=str(source.info.get("structure_id", "")),
                    system_id=str(source.info.get("system_id", "")),
                    labeled_atoms=item,
                    input_digest=str(context["input_digest"]),
                    parser_evidence={},
                    required_labels=tuple(required_labels),
                )
            )
        task_root = root / "tasks"
        if task_root.is_symlink():
            return False
        entries = tuple(task_root.iterdir()) if task_root.is_dir() else ()
        if any(path.is_symlink() or not path.is_dir() for path in entries):
            return False
        actual_task_names = sorted(path.name for path in entries)
        if actual_task_names != expected_task_names:
            return False
        for source, result in zip(selected, results, strict=True):
            validate_fp_label_result(source, result)
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _valid_operation_evidence(operation: object, *, calculator: str) -> bool:
    logical_executable = calculator.lower()
    return bool(
        isinstance(operation, dict)
        and set(operation) == {"role", "profile_id", "logical_executable"}
        and operation.get("role")
        == f"first_principles.{logical_executable}.label"
        and isinstance(operation.get("profile_id"), str)
        and bool(operation["profile_id"])
        and operation.get("logical_executable") == logical_executable
    )


def _validate_task_input_digest(
    task_path: Path,
    *,
    source: object,
    task_id: str,
    selected_index: int,
    structure_id: str,
    system_id: str,
    input_digest: str,
    required_labels: object,
) -> bool:
    """Accept native bundle digests while retaining legacy fixture coverage."""
    if len(input_digest) != 64:
        return False
    try:
        int(input_digest, 16)
    except ValueError:
        return False

    metadata_path = task_path / "validated_result.json"
    if not metadata_path.exists():
        return input_digest == _sha256(task_path / "selected.xyz")
    result_path = task_path / "validated_result.xyz"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_keys = {
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
            or set(payload) != expected_keys
            or payload.get("schema") != "dice.fp-validated-task-result"
            or payload.get("schema_revision") != 2
            or payload.get("task_id") != task_id
            or payload.get("selected_index") != selected_index
            or payload.get("structure_id") != structure_id
            or payload.get("system_id") != system_id
            or payload.get("input_digest") != input_digest
            or payload.get("required_labels") != required_labels
            or not isinstance(payload.get("parser_evidence"), dict)
            or not result_path.is_file()
        ):
            return False
        validated = read_extxyz_structures(result_path, allow_empty=False)
        if len(validated) != 1:
            return False
        from workflow_backends.first_principles.protocols import FPLabelResult

        validate_fp_label_result(
            source,
            FPLabelResult(
                task_id=task_id,
                selected_index=selected_index,
                structure_id=structure_id,
                system_id=system_id,
                labeled_atoms=validated[0],
                input_digest=input_digest,
                parser_evidence=dict(payload["parser_evidence"]),
                required_labels=tuple(required_labels),
            ),
        )
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _valid_native_input_inventory(value: object, *, task_path: Path) -> bool:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
        or "selected.xyz" not in value
    ):
        return False
    for item in value:
        candidate = Path(item)
        if (
            candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or any(char in item for char in "*?[")
            or (task_path / candidate).is_symlink()
            or not (task_path / candidate).is_file()
        ):
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["validate_fp_label_artifacts"]
