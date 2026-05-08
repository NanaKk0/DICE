from pathlib import Path

import numpy as np

from workflow_core.structures.formats import write_extxyz_structures
from workflow_core.structures.validation import (
    validate_deepmd_export_inputs,
    validate_written_deepmd_dataset,
)


def _stress_to_matrix(stress_value):
    arr = np.asarray(stress_value, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (9,):
        return arr.reshape(3, 3)
    if arr.shape == (6,):
        xx, yy, zz, yz, xz, xy = arr.tolist()
        return np.array(
            [
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported stress shape for DeepMD export: {arr.shape!r}")


def _get_stress_matrix(atoms):
    stress_value = atoms.info.get("stress")
    if stress_value is not None:
        return _stress_to_matrix(stress_value)
    if atoms.calc is not None:
        try:
            return _stress_to_matrix(atoms.get_stress())
        except Exception:
            pass
    raise RuntimeError(
        "Atoms object is missing reference stress; provide atoms.info['stress'] "
        "or attach a calculator that supports get_stress() before DeepMD export."
    )


def sort_atoms(atoms):
    symbols = atoms.get_chemical_symbols()
    pos = atoms.positions

    order = sorted(
        range(len(atoms)),
        key=lambda i: (symbols[i], pos[i][2], pos[i][1], pos[i][0])
    )

    new_atoms = atoms[order]

    if atoms.calc is not None:
        new_atoms.calc = atoms.calc

    return new_atoms


def xyz_to_deepmd(atoms_list, out_dir, element_order):
    atoms_list = [sort_atoms(atoms) for atoms in atoms_list]
    validation = validate_deepmd_export_inputs(atoms_list, element_order)

    out_dir = Path(out_dir)
    set_dir = out_dir / "set.000"

    set_dir.mkdir(parents=True, exist_ok=True)

    nframes = int(validation["nframes"])
    natoms = int(validation["natoms"])
    types = list(validation["types"])

    with open(out_dir / "type_map.raw", "w") as f:
        f.write(" ".join(element_order) + "\n")

    with open(out_dir / "type.raw", "w") as f:
        f.write(" ".join(map(str, types)) + "\n")

    coords = np.zeros((nframes, natoms, 3))
    forces = np.zeros((nframes, natoms, 3))
    energies = np.zeros(nframes)
    boxes = np.zeros((nframes, 3, 3))
    virials = np.zeros((nframes, 3, 3))

    for i, atoms in enumerate(atoms_list):
        coords[i] = atoms.positions
        force_array = atoms.arrays.get("forces")
        if force_array is None:
            try:
                force_array = atoms.get_forces()
            except Exception as exc:
                raise RuntimeError(
                    "Atoms object is missing reference forces; provide atoms.arrays['forces'] "
                    "or attach a calculator before DeepMD export."
                ) from exc
        forces[i] = np.asarray(force_array, dtype=float)

        energy_value = atoms.info.get("energy")
        if energy_value is None:
            try:
                energy_value = atoms.get_potential_energy()
            except Exception as exc:
                raise RuntimeError(
                    "Atoms object is missing reference energy; provide atoms.info['energy'] "
                    "or attach a calculator before DeepMD export."
                ) from exc
        energies[i] = float(energy_value)

        cell = atoms.get_cell()

        if cell.volume < 1e-6:
            boxes[i] = np.eye(3) * 30.0
        else:
            boxes[i] = cell.array

        stress_matrix = _get_stress_matrix(atoms)
        virials[i] = -stress_matrix * float(abs(cell.volume))

    coords = coords.reshape(nframes, natoms * 3)
    forces = forces.reshape(nframes, natoms * 3)
    virials = virials.reshape(nframes, 9)

    np.save(set_dir / "coord.npy", coords)
    np.save(set_dir / "force.npy", forces)
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "box.npy", boxes)
    np.save(set_dir / "virial.npy", virials)
    validate_written_deepmd_dataset(
        out_dir,
        nframes=nframes,
        natoms=natoms,
        element_order=element_order,
        types=types,
        reference_boxes=boxes,
        require_virial=True,
    )

    print("DeepMD dataset written:", out_dir)


def export_label_results(outputs, export_cfg):
    labeled_output_dir = Path(export_cfg["labeled_output_dir"])
    active_learning_output_dir = Path(export_cfg["active_learning_output_dir"])
    element_order = export_cfg["element_order"]

    labeled_output_dir.mkdir(parents=True, exist_ok=True)
    active_learning_output_dir.mkdir(parents=True, exist_ok=True)

    write_extxyz_structures(labeled_output_dir / "committee_labeled.xyz", outputs.labeled_structures)
    write_extxyz_structures(active_learning_output_dir / "dft.xyz", outputs.dft_structures)
    write_extxyz_structures(labeled_output_dir / "discard.xyz", outputs.discard_structures)
    write_extxyz_structures(labeled_output_dir / "train.xyz", outputs.train_structures)

    for composition_name, datasets in outputs.grouped_train_val.items():
        train_set, val_set = datasets

        if len(train_set) > 0:
            train_dir = labeled_output_dir / "train" / composition_name
            xyz_to_deepmd(train_set, train_dir, element_order)
        else:
            print(f"Skip empty DeepMD train export for composition {composition_name}")

        if len(val_set) > 0:
            val_dir = labeled_output_dir / "val" / composition_name
            xyz_to_deepmd(val_set, val_dir, element_order)
        else:
            print(f"Skip empty DeepMD val export for composition {composition_name}")

