from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from workflow_backends.first_principles.abacus.parser import parse_abacus_output
from workflow_backends.first_principles.abacus.recipe import load_abacus_recipe
from workflow_backends.first_principles.abacus.render import (
    abacus_species_and_atom_order,
    render_abacus_kpt,
    render_abacus_stru,
)
from workflow_backends.first_principles.protocols import (
    FPLabelResult,
    FPParseRequest,
    FPPreparedTask,
    FPPrepareRequest,
    fp_task_id,
)
from workflow_core.structures.formats import (
    read_extxyz_structures,
    write_extxyz_structures,
)


class ABACUSCalculatorAdapter:
    calculator = "ABACUS"
    operation_role = "first_principles.abacus.label"
    logical_executable = "abacus"

    def prepare(self, request: FPPrepareRequest) -> tuple[FPPreparedTask, ...]:
        if request.operation_context.role != self.operation_role:
            raise ValueError("ABACUS adapter operation role does not match context")
        request.operation_context.executable(self.logical_executable)
        recipe = load_abacus_recipe(request.recipe_path)
        selected = read_extxyz_structures(request.selected_path, allow_empty=True)
        task_root = Path(request.task_root)
        task_root.mkdir(parents=True, exist_ok=True)
        prepared: list[FPPreparedTask] = []
        required_labels = (
            ("energy", "forces", "stress", "virial")
            if recipe.require_stress
            else ("energy", "forces")
        )
        for index, atoms in enumerate(selected):
            task_id = fp_task_id(index)
            task_path = task_root / task_id
            task_path.mkdir(parents=True, exist_ok=True)
            species, _ = abacus_species_and_atom_order(atoms)
            pseudos, orbitals = recipe.element_assets(species)
            stru, native_atom_order = render_abacus_stru(
                atoms, pseudopotentials=pseudos, orbitals=orbitals
            )
            selected_bytes = _extxyz_bytes(task_path, atoms)
            native_inputs: list[tuple[Path, bytes]] = [
                (Path("selected.xyz"), selected_bytes),
                (Path("INPUT"), recipe.template.encode("utf-8")),
                (Path("STRU"), stru.encode("utf-8")),
            ]
            mesh = None
            if recipe.spacing_inv_angstrom is not None:
                kpt, mesh = render_abacus_kpt(atoms, recipe.spacing_inv_angstrom)
                native_inputs.append((Path("KPT"), kpt.encode("utf-8")))
            for asset in (*pseudos, *orbitals):
                native_inputs.append((asset.destination, asset.source.read_bytes()))
            for asset in recipe.assets:
                native_inputs.append((asset.destination, asset.source.read_bytes()))
            for relative, content in native_inputs:
                _write_immutable(task_path / relative, content)
            digest = _input_digest(
                recipe_bytes=recipe.path.read_bytes(), native_inputs=native_inputs
            )
            context = {
                "schema": "dice.abacus-static-task-context",
                "schema_revision": 1,
                "task_id": task_id,
                "selected_index": index,
                "structure_id": str(atoms.info.get("structure_id", "")),
                "system_id": str(atoms.info.get("system_id", "")),
                "calculator": self.calculator,
                "mode": "scf",
                "basis_type": recipe.basis_type,
                "input_digest": digest,
                "required_labels": list(required_labels),
                "species_order": list(species),
                "native_atom_order": list(native_atom_order),
                "kpoint_mesh": list(mesh) if mesh is not None else None,
                "native_inputs": [relative.as_posix() for relative, _ in native_inputs],
                "operation": {
                    "role": self.operation_role,
                    "profile_id": request.operation_context.profile_id,
                    "logical_executable": self.logical_executable,
                },
            }
            _write_immutable(
                task_path / "task_context.json",
                json.dumps(context, sort_keys=True).encode("utf-8"),
            )
            prepared.append(
                FPPreparedTask(
                    task_id=task_id,
                    selected_index=index,
                    structure_id=str(atoms.info.get("structure_id", "")),
                    system_id=str(atoms.info.get("system_id", "")),
                    task_path=task_path,
                    input_digest=digest,
                    logical_executable=self.logical_executable,
                    arguments=(),
                    required_labels=required_labels,
                )
            )
        return tuple(prepared)

    def parse(self, request: FPParseRequest) -> FPLabelResult:
        task = request.prepared_task
        terminal = request.completed_task
        if (
            terminal.task_id != task.task_id
            or Path(terminal.workspace).resolve() != Path(task.task_path).resolve()
            or terminal.input_digest != task.input_digest
            or terminal.returncode != 0
        ):
            raise ValueError(f"ABACUS terminal evidence mismatch for {task.task_id}")
        source_frames = read_extxyz_structures(
            Path(task.task_path) / "selected.xyz", allow_empty=False
        )
        if len(source_frames) != 1:
            raise ValueError(
                f"ABACUS task selected inventory is invalid for {task.task_id}"
            )
        source = source_frames[0]
        _species, native_atom_order = abacus_species_and_atom_order(source)
        selected_symbols = tuple(source.get_chemical_symbols())
        native_symbols = tuple(selected_symbols[index] for index in native_atom_order)
        parsed = parse_abacus_output(
            Path(task.task_path) / "OUT.ABACUS" / "running_scf.log",
            native_symbols=native_symbols,
            volume_angstrom3=float(source.cell.volume),
            require_stress="stress" in task.required_labels,
        )
        selected_forces = np.empty_like(parsed.native_forces_ev_per_angstrom)
        for native_index, selected_index in enumerate(native_atom_order):
            selected_forces[selected_index] = parsed.native_forces_ev_per_angstrom[
                native_index
            ]
        labeled = source.copy()
        calculator_values: dict[str, object] = {
            "energy": parsed.energy_ev,
            "forces": selected_forces,
        }
        if parsed.stress_ev_per_angstrom3 is not None:
            calculator_values["stress"] = parsed.stress_ev_per_angstrom3
        labeled.calc = SinglePointCalculator(labeled, **calculator_values)
        if parsed.virial_ev is not None:
            labeled.info["virial"] = np.asarray(parsed.virial_ev, dtype=float)
        return FPLabelResult(
            task_id=task.task_id,
            selected_index=task.selected_index,
            structure_id=task.structure_id,
            system_id=task.system_id,
            labeled_atoms=labeled,
            input_digest=task.input_digest,
            parser_evidence={
                "backend": "ABACUS",
                "mode": "scf",
                "normal_termination": True,
                "scf_converged": parsed.scf_converged,
                "native_atom_order": list(native_atom_order),
                "energy_unit": "eV",
                "force_unit": "eV/angstrom",
                "stress_unit": (
                    "eV/angstrom^3"
                    if parsed.stress_ev_per_angstrom3 is not None
                    else None
                ),
                "stress_convention": (
                    "ASE-positive-tension-voigt-xx-yy-zz-yz-xz-xy"
                    if parsed.stress_ev_per_angstrom3 is not None
                    else None
                ),
                "virial_convention": (
                    "-stress*abs(volume)" if parsed.virial_ev is not None else None
                ),
            },
            required_labels=task.required_labels,
        )


def _extxyz_bytes(task_path: Path, atoms: object) -> bytes:
    candidate = task_path / ".selected.candidate.xyz"
    write_extxyz_structures(candidate, [atoms], allow_empty=False)
    try:
        return candidate.read_bytes()
    finally:
        candidate.unlink(missing_ok=True)


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"immutable ABACUS scientific input changed: {path.name}")
        return
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_bytes(content)
    staged.replace(path)


def _input_digest(
    *, recipe_bytes: bytes, native_inputs: Iterable[tuple[Path, bytes]]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"dice.abacus-static-input\0")
    digest.update(recipe_bytes)
    for relative, content in native_inputs:
        digest.update(b"\0")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


__all__ = ["ABACUSCalculatorAdapter"]
