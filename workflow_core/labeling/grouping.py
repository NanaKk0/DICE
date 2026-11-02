from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


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
        key=lambda i: (symbols[i], pos[i][2], pos[i][1], pos[i][0]),
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
