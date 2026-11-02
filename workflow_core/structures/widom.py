"""Pure independent insertion geometry, without model or execution knowledge."""
import hashlib
import numpy as np
from ase import Atoms
from ase.data import vdw_radii
from ase.geometry import find_mic
from scipy.spatial.transform import Rotation


def validate_inputs(host, guests, parameters, type_map=None):
    if (not len(host) or not np.all(host.pbc) or not np.isfinite(host.positions).all()
        or not np.isfinite(host.cell.array).all() or np.linalg.matrix_rank(host.cell.array) != 3):
        raise ValueError('widom requires a finite full-rank three-dimensional periodic host')
    if host.constraints:
        raise ValueError('widom does not accept constrained host inputs')
    if not guests or not len(guests[0]):
        raise ValueError('widom guest library must be nonempty')
    for guest in guests:
        if (not np.array_equal(guest.numbers, guests[0].numbers)
            or not np.isfinite(guest.positions).all() or guest.constraints):
            raise ValueError('widom guest library requires finite unconstrained consistent atom order')
    symbols = set(host.get_chemical_symbols()+guests[0].get_chemical_symbols())
    if type_map is not None and not symbols <= set(type_map):
        raise ValueError('widom species outside student.type_map')
    if parameters.overlap_scale is None:
        return None
    supplied = dict(parameters.radii_a)
    if supplied and not symbols <= supplied.keys():
        raise ValueError('widom radii_a must cover all background and guest elements')
    def radii(atoms):
        result = np.array([supplied[s] if supplied else vdw_radii[z]
                           for s, z in zip(atoms.get_chemical_symbols(), atoms.numbers)])
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError('widom missing vdW radii; provide complete radii_a')
        return result
    return radii(host), radii(guests[0])


def insertion_candidates(host, guests, parameters, *, entropy):
    """Yield (one-based attempt, guest index, geometry or rejection None)."""
    radii = validate_inputs(host, guests, parameters)
    material = f'widom-uniform-v1\0{parameters.random_seed}\0{entropy}'.encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
    accepted = 0
    for attempt in range(1, parameters.max_attempts+1):
        index = int(rng.integers(len(guests)))
        translation = rng.random(3) @ host.cell.array
        positions = Rotation.random(random_state=rng).apply(guests[index].positions)+translation
        collision = False
        if radii is not None:
            for position, radius in zip(positions, radii[1]):
                _, distances = find_mic(host.positions-position, host.cell, host.pbc)
                if np.any(distances < parameters.overlap_scale*(radii[0]+radius)):
                    collision = True
                    break
        if collision:
            yield attempt, index, None
            continue
        # Drop inherited calculators, energies, uncertainty and identity arrays.
        frame = Atoms(numbers=np.r_[host.numbers, guests[index].numbers],
                      positions=np.concatenate((host.positions, positions)), cell=host.cell, pbc=host.pbc)
        accepted += 1
        yield attempt, index, frame
        if accepted == parameters.frames:
            return
