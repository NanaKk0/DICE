from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms


def validate_deepmd_export_inputs(atoms_list: list[Atoms], element_order: list[str]) -> dict[str, object]:
    if not atoms_list:
        raise ValueError("Cannot export empty atoms_list to DeepMD format")

    natoms = len(atoms_list[0])
    ref_symbols = list(atoms_list[0].get_chemical_symbols())
    symbols = set(ref_symbols)

    for symbol in symbols:
        if symbol not in element_order:
            raise RuntimeError(f"Element {symbol} not in element_order {element_order}")

    for atoms in atoms_list:
        if len(atoms) != natoms:
            raise RuntimeError(f"Inconsistent natoms in system: {len(atoms)} vs {natoms}")
        if list(atoms.get_chemical_symbols()) != ref_symbols:
            raise RuntimeError("Atom ordering inconsistent inside system")

    type_dict = {element: i for i, element in enumerate(element_order)}
    types = [type_dict[symbol] for symbol in ref_symbols]
    return {
        "natoms": natoms,
        "nframes": len(atoms_list),
        "ref_symbols": ref_symbols,
        "types": types,
    }


def validate_written_deepmd_dataset(
    out_dir: str | Path,
    *,
    nframes: int,
    natoms: int,
    element_order: list[str],
    types: list[int],
    reference_boxes: np.ndarray,
    require_virial: bool = True,
) -> None:
    out_dir = Path(out_dir).resolve()
    set_dir = out_dir / "set.000"

    required = [
        out_dir / "type_map.raw",
        out_dir / "type.raw",
        set_dir / "coord.npy",
        set_dir / "force.npy",
        set_dir / "energy.npy",
        set_dir / "box.npy",
    ]
    if require_virial:
        required.append(set_dir / "virial.npy")

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"DeepMD export missing required files: {missing}")

    type_map_text = (out_dir / "type_map.raw").read_text(encoding="utf-8").strip()
    if type_map_text != " ".join(element_order):
        raise RuntimeError(
            f"type_map.raw mismatch: current={type_map_text!r}, expected={' '.join(element_order)!r}"
        )

    type_text = (out_dir / "type.raw").read_text(encoding="utf-8").strip()
    if type_text != " ".join(map(str, types)):
        raise RuntimeError(
            f"type.raw mismatch: current={type_text!r}, expected={' '.join(map(str, types))!r}"
        )

    coord = np.load(set_dir / "coord.npy")
    force = np.load(set_dir / "force.npy")
    energy = np.load(set_dir / "energy.npy")
    box = np.load(set_dir / "box.npy")

    expected_frame_atom3 = (nframes, natoms * 3)
    expected_energy = (nframes,)
    expected_box = (nframes, 3, 3)
    if coord.shape != expected_frame_atom3:
        raise RuntimeError(f"coord.npy shape mismatch: current={coord.shape}, expected={expected_frame_atom3}")
    if force.shape != expected_frame_atom3:
        raise RuntimeError(f"force.npy shape mismatch: current={force.shape}, expected={expected_frame_atom3}")
    if energy.shape != expected_energy:
        raise RuntimeError(f"energy.npy shape mismatch: current={energy.shape}, expected={expected_energy}")
    if box.shape != expected_box:
        raise RuntimeError(f"box.npy shape mismatch: current={box.shape}, expected={expected_box}")
    if require_virial:
        virial = np.load(set_dir / "virial.npy")
        expected_virial = (nframes, 9)
        if virial.shape != expected_virial:
            raise RuntimeError(f"virial.npy shape mismatch: current={virial.shape}, expected={expected_virial}")

    if not np.allclose(box, np.asarray(reference_boxes, dtype=float), atol=1e-8, rtol=1e-8):
        raise RuntimeError("box.npy content mismatch against canonical source cells")
