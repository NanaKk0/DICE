from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from workflow_core.lifecycle.fp_capabilities import (
    FirstPrinciplesLabelCapability,
    FirstPrinciplesLabelFact,
    FirstPrinciplesLabelIntent,
    FirstPrinciplesLabelOutcome,
    FirstPrinciplesSelectedStructure,
    FirstPrinciplesSelection,
    fp_task_id,
)
from workflow_core.lifecycle.fp_use_cases import FirstPrinciplesLabelUseCase
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)


def label_selected_structures(
    request: object,
    *,
    service: object,
    output_path: str | Path,
) -> object:
    """Compatibility shell for the current path-backed publication wiring.

    New logic consumes ``FirstPrinciplesLabelUseCase``. This bounded shell is
    retained until selected-structure and publication handles are integrated.
    """

    calculator = str(getattr(request, "calculator"))
    operation_role = str(getattr(request, "operation_role"))
    if getattr(service, "calculator", None) != calculator:
        raise ValueError(
            "first-principles service calculator does not match fp.label configuration"
        )
    if getattr(service, "operation_role", None) != operation_role:
        raise ValueError(
            "first-principles service operation role does not match lifecycle selection"
        )
    selected = read_extxyz_structures(
        getattr(request, "selected_path"), allow_empty=True
    )
    batch = getattr(service, "label")(request)
    ordered_results, labeled = _validate_and_order_results(
        selected, tuple(getattr(batch, "results"))
    )
    intent = FirstPrinciplesLabelIntent(
        selection=FirstPrinciplesSelection(
            tuple(
                FirstPrinciplesSelectedStructure(
                    selected_index=index,
                    structure_id=str(source.info.get("structure_id", "")),
                    system_id=str(source.info.get("system_id", "")),
                )
                for index, source in enumerate(selected)
            )
        ),
        calculator=calculator,
        operation_role=operation_role,
        operation_id=str(getattr(request, "operation_id")),
    )
    outcome = FirstPrinciplesLabelOutcome(
        tuple(_semantic_fact(result) for result in ordered_results)
    )
    FirstPrinciplesLabelUseCase(_CompletedCapability(outcome)).execute(intent)
    write_extxyz_structures(output_path, labeled, allow_empty=True)
    return type(batch)(
        tuple(getattr(batch, "results")),
        dict(getattr(batch, "summary_evidence")),
    )


class _CompletedCapability:
    def __init__(self, outcome: FirstPrinciplesLabelOutcome) -> None:
        self.outcome = outcome

    def label(
        self, request: FirstPrinciplesLabelIntent
    ) -> FirstPrinciplesLabelOutcome:
        return self.outcome


def _semantic_fact(result: object) -> FirstPrinciplesLabelFact:
    parser_evidence = getattr(result, "parser_evidence")
    convergence = parser_evidence.get("scf_converged")
    return FirstPrinciplesLabelFact(
        task_id=str(getattr(result, "task_id")),
        selected_index=int(getattr(result, "selected_index")),
        structure_id=str(getattr(result, "structure_id")),
        system_id=str(getattr(result, "system_id")),
        input_digest=str(getattr(result, "input_digest")),
        required_labels=tuple(getattr(result, "required_labels")),
        scf_converged=convergence if isinstance(convergence, bool) else None,
    )


def _validate_and_order_results(
    selected: Sequence[object], results: Sequence[object]
) -> tuple[tuple[object, ...], list[object]]:
    by_index: dict[int, object] = {}
    for result in results:
        selected_index = int(getattr(result, "selected_index"))
        if selected_index in by_index:
            raise ValueError(
                f"duplicate first-principles result for selected index {selected_index}"
            )
        by_index[selected_index] = result
    expected_indices = set(range(len(selected)))
    if set(by_index) != expected_indices:
        raise ValueError("first-principles results do not exactly cover selected structures")

    ordered_results: list[object] = []
    labeled: list[object] = []
    for index, source in enumerate(selected):
        result = by_index[index]
        expected_task = fp_task_id(index)
        structure_id = str(source.info.get("structure_id", ""))
        system_id = str(source.info.get("system_id", ""))
        if (
            getattr(result, "task_id") != expected_task
            or getattr(result, "structure_id") != structure_id
            or getattr(result, "system_id") != system_id
        ):
            raise ValueError(
                f"first-principles result identity mismatch for {expected_task}"
            )
        _validate_native_label_result(source, result)
        item = getattr(result, "labeled_atoms")
        item.info["structure_id"] = structure_id
        item.info["system_id"] = system_id
        ordered_results.append(result)
        labeled.append(item)
    return tuple(ordered_results), labeled


def _validate_native_label_result(source: object, result: object) -> None:
    """Preserve the current compatibility-shell native validation contract."""

    task_id = str(getattr(result, "task_id"))
    labeled = getattr(result, "labeled_atoms")
    structure_id = str(source.info.get("structure_id", ""))
    system_id = str(source.info.get("system_id", ""))
    if (
        len(source) != len(labeled)
        or source.get_chemical_symbols() != labeled.get_chemical_symbols()
        or not np.allclose(source.positions, labeled.positions, atol=1e-10, rtol=0.0)
        or not np.allclose(source.cell.array, labeled.cell.array, atol=1e-10, rtol=0.0)
        or not np.array_equal(source.pbc, labeled.pbc)
        or getattr(result, "structure_id") != structure_id
        or getattr(result, "system_id") != system_id
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
    required = tuple(getattr(result, "required_labels"))
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
        native_results = getattr(getattr(labeled, "calc", None), "results", {}) or {}
        virial_value = native_results.get("virial", labeled.info.get("virial"))
        if virial_value is None:
            raise ValueError(
                f"first-principles result is missing required virial for {task_id}"
            )
        virial = np.asarray(virial_value, dtype=float)
        if virial.shape not in {(6,), (3, 3), (9,)} or not np.all(
            np.isfinite(virial)
        ):
            raise ValueError(
                f"first-principles result has non-finite or malformed virial for {task_id}"
            )


__all__ = ["FirstPrinciplesLabelUseCase", "label_selected_structures"]
