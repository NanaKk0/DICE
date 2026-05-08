from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ase.calculators.singlepoint import SinglePointCalculator
from ase.neighborlist import neighbor_list

from workflow_core.labeling.results import LabelingOutputs
from workflow_core.mace.backend import build_mace_calculator, normalize_mace_backend
from workflow_core.mace.model_check import ensure_committee_backend_compatible


def _ensure_stress_matrix(stress) -> np.ndarray:
    arr = np.asarray(stress, dtype=float)
    if arr.shape == (3, 3):
        return arr
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
    raise ValueError(f"Unsupported stress shape: {arr.shape!r}")


def split_train_validation(atoms_list, cfg):
    train_ratio = float(cfg.get("train_ratio", 0.9))
    seed = cfg.get("split_seed", 42)
    min_val = cfg.get("min_val", 3)

    rng = np.random.default_rng(seed)

    indices = np.arange(len(atoms_list))
    rng.shuffle(indices)

    n_val = max(min_val, int(len(indices) * (1 - train_ratio)))
    n_train = len(indices) - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_set = [atoms_list[i] for i in train_idx]
    val_set = [atoms_list[i] for i in val_idx]

    print(f"{len(train_set)} train / {len(val_set)} val")

    return train_set, val_set


def load_committee(cfg):
    teacher_dir = Path(cfg["teacher_dir"])
    teacher_num = cfg["teacher_num"]
    backend = normalize_mace_backend(cfg.get("mace_backend", "standard"))
    ensure_committee_backend_compatible(teacher_dir, backend, teacher_num=teacher_num)

    calculators = []

    for i in range(1, teacher_num + 1):
        model = teacher_dir / f"teacher_{i:02d}" / f"mace_teacher_{i:02d}.model"

        calc = build_mace_calculator(
            model,
            backend=backend,
            device=cfg.get("device", "cpu"),
            default_dtype=cfg.get("default_dtype", "float32")
        )

        calculators.append(calc)

    print("Loaded teacher models:", len(calculators))

    return calculators


def compute_force_uncertainty(forces):
    forces = np.array(forces)
    f_mean = forces.mean(axis=0)
    diff = forces - f_mean
    sq = np.sum(diff**2, axis=2)
    mean_sq = np.mean(sq, axis=0)
    atom_std = np.sqrt(mean_sq)

    return np.max(atom_std)


def structure_is_valid(atoms, cfg):
    max_force = cfg.get("max_force", 50.0)

    if atoms.calc is not None:
        forces = atoms.get_forces()
        if np.max(np.linalg.norm(forces, axis=1)) > max_force:
            return False

    min_distances = cfg.get(
        "min_distances",
        {
            "Pt-Pt": 2.0,
            "Pt-O": 1.3,
            "O-H": 0.6
        }
    )

    if len(atoms) < 2:
        return True

    i, j, d = neighbor_list("ijd", atoms, 3.0)

    for a, b, dist in zip(i, j, d):
        s1 = atoms[a].symbol
        s2 = atoms[b].symbol
        pair = "-".join(sorted([s1, s2]))

        if pair in min_distances and dist < min_distances[pair]:
            return False

    return True


def committee_label(atoms_list, calculators, cfg):
    labeled = []
    uncertainties = []

    for atoms in tqdm(atoms_list, desc="Committee labeling"):
        energies = []
        forces = []
        stresses = []

        for calc in calculators:
            atoms.calc = calc

            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())

            try:
                stress = atoms.get_stress()
            except Exception:
                stress = np.zeros(6)

            stresses.append(stress)

        energies = np.array(energies)
        forces = np.array(forces)
        stresses = np.array([_ensure_stress_matrix(stress) for stress in stresses], dtype=float)

        energy_mean = energies.mean()
        force_mean = forces.mean(axis=0)
        stress_mean = stresses.mean(axis=0)
        stress_mean = np.clip(stress_mean, -100, 100)

        uncertainty = compute_force_uncertainty(forces)

        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy_mean,
            forces=force_mean,
            stress=stress_mean,
        )

        atoms.info["force_uncertainty"] = float(uncertainty)

        labeled.append(atoms)
        uncertainties.append(uncertainty)

    return labeled, np.array(uncertainties)


def print_uncertainty_histogram(uncertainties):
    print("\nForce uncertainty statistics")
    print("min:", uncertainties.min())
    print("max:", uncertainties.max())
    print("mean:", uncertainties.mean())
    print("median:", np.median(uncertainties))

    bins = np.linspace(uncertainties.min(), uncertainties.max(), 10)
    hist, edges = np.histogram(uncertainties, bins=bins)

    print("\nHistogram")

    for i in range(len(hist)):
        print(f"{edges[i]:.4f} - {edges[i+1]:.4f} : {hist[i]}")


def split_dataset(atoms_list, cfg):
    candidate_threshold = float(cfg.get("candidate_sigma_threshold", 0.10))

    train = []
    dft = []
    discard = []

    for atoms in atoms_list:
        uncertainty = float(atoms.info["force_uncertainty"])
        atoms.info["label_uncertainty_threshold"] = candidate_threshold

        if uncertainty < candidate_threshold:
            atoms.info["label_split"] = "accurate"
            train.append(atoms)
        else:
            atoms.info["label_split"] = "candidate"
            dft.append(atoms)

    print("\nDataset split")
    print("Teacher label candidate threshold:", candidate_threshold)
    print("DP train:", len(train))
    print("DFT:", len(dft))
    print("Discard:", len(discard), "(disabled for teacher labeling)")

    return train, dft, discard


def composition_key(atoms):
    count = Counter(atoms.get_chemical_symbols())

    return tuple(sorted(count.items()))


def composition_name(key):
    name = ""

    for elem, n in key:
        name += f"{elem}{n}"

    return name


def group_by_composition(atoms_list):
    groups = defaultdict(list)

    for atoms in atoms_list:
        key = composition_key(atoms)
        groups[key].append(atoms)

    print("\nComposition groups")

    for key in groups:
        name = composition_name(key)
        print(name, "frames:", len(groups[key]))

    return groups


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


def build_grouped_train_val(train_atoms, cfg):
    grouped_train_val = {}
    groups = group_by_composition(train_atoms)

    for key, atoms_group in groups.items():
        name = composition_name(key)
        grouped_train_val[name] = split_train_validation(atoms_group, cfg)

    return grouped_train_val


def run_labeling(atoms_list, cfg):
    atoms_list = [sort_atoms(atoms) for atoms in atoms_list]

    print("Frames:", len(atoms_list))

    calculators = load_committee(cfg)
    labeled, uncertainties = committee_label(atoms_list, calculators, cfg)

    print_uncertainty_histogram(uncertainties)

    valid_structures = [
        atoms for atoms in labeled if structure_is_valid(atoms, cfg)
    ]

    print("\nValid structures:", len(valid_structures))

    train_structures, dft_structures, discard_structures = split_dataset(
        valid_structures,
        cfg
    )

    grouped_train_val = build_grouped_train_val(train_structures, cfg)

    return LabelingOutputs(
        labeled_structures=labeled,
        uncertainties=uncertainties,
        valid_structures=valid_structures,
        train_structures=train_structures,
        dft_structures=dft_structures,
        discard_structures=discard_structures,
        grouped_train_val=grouped_train_val
    )

