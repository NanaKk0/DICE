"""Explicit acquisition membership invariants, independent of MC/MD engines."""
import numpy as np


def validate_membership(atoms, guest_numbers):
    """Require leading host and explicit whole guests; never infer bonds."""
    try:
        ids, mids = (atoms.arrays[key] for key in ('dice_atom_id', 'dice_molecule_id'))
    except KeyError as exc:
        raise ValueError('both identity and membership arrays are required') from exc
    if any(a.shape != (len(atoms),) or a.dtype.kind not in 'iu' for a in (ids, mids)):
        raise ValueError('identity and membership must be one-dimensional integer arrays')
    if np.any(ids <= 0) or len(set(ids)) != len(atoms) or np.any(mids < 0):
        raise ValueError('identity must be positive unique; membership must be nonnegative')
    n_host = int(np.count_nonzero(mids == 0))
    size = len(guest_numbers)
    if not size or not n_host or np.any(mids[:n_host] != 0) or (len(atoms)-n_host) % size:
        raise ValueError('membership requires leading host and complete contiguous guests')
    guest_ids = []
    for start in range(n_host, len(atoms), size):
        if mids[start] <= 0 or not np.all(mids[start:start+size] == mids[start]):
            raise ValueError('guest membership must be one positive ID per molecule')
        if not np.array_equal(atoms.numbers[start:start+size], guest_numbers):
            raise ValueError('guest membership species/order differs from template')
        guest_ids.append(int(mids[start]))
    if len(set(guest_ids)) != len(guest_ids):
        raise ValueError('guest membership IDs must be unique between molecules')
    return n_host
