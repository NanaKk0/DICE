# Water example data

The DICE authors provide the pure-water structures and DFT reference labels
in this example under [CC BY 4.0](../../../LICENSE-DATA.md).
Credit the DICE authors and identify any changes when reusing the data.

Each frame contains 64 water molecules (192 atoms). Coordinates and cells are
in Å, total energies in eV, forces in eV/Å, and ASE stresses in eV/Å³.
The example trains on energy and forces.

| File | Frames | Use |
| --- | --- | --- |
| `teacher_initial.extxyz` | 40 | Initial DFT reference data for the teacher |
| `student_initial.extxyz` | 40 | The same initial structures; DICE labels them with the current teacher |
| `demo_iter1_labeled.extxyz` | 20 | Additional DFT reference data supplied through `fp_data` in iter1 |
| `../seeds/seed.extxyz` | 2 | Starting structures for sampling |

The iter1 file carries previously calculated DFT labels to demonstrate adding
reference data. For your own calculations, label the selected FP structures
using settings consistent with the teacher reference data, then supply the
returned data through `fp_data`.

The accompanying foundation model is obtained separately; its source and
licence are in [models/README.md](../models/README.md).
