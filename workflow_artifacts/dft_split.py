# split_dft_xyz.py

from ase.io import read, write
from pathlib import Path
import sys

xyz = sys.argv[1]
out_dir = Path(sys.argv[2])

atoms_list = read(xyz, ":")

out_dir.mkdir(parents=True, exist_ok=True)

for i, atoms in enumerate(atoms_list):

    d = out_dir / str(i)

    d.mkdir(exist_ok=True)

    write(d / "structure.xyz", atoms)

print("Total structures:", len(atoms_list))