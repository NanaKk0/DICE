# Data format reference

DICE reads ASE-compatible extended XYZ (`.xyz` or `.extxyz`). The format carries atomic geometry, optional metadata and the labels required by the consuming operation.

## Minimal labelled frame

The example [labelled_h2.extxyz](../../tests/fixtures/labelled_h2.extxyz) illustrates the format:

```text
2
Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.0 system_id=h2_demo pbc="F F F"
H 0.00 0.00 0.00  0.0 0.0 0.0
H 0.74 0.00 0.00  0.0 0.0 0.0
```

Each declared input contains at least one complete frame. Multiple frames can be concatenated in one file.

## Geometry, labels and units

| Quantity | Shape / requirement | Convention |
| --- | --- | --- |
| Species, positions | N symbols, N×3 coordinates | Elements covered by the consuming model; coordinates in Å |
| Lattice, PBC | 3×3 cell, three flags | Lattice in Å; actual boundary conditions; valid periodic cell for periodic sampling |
| `energy` | One finite scalar, required for teacher/FP labels | Total energy in eV per frame, with a consistent reference across the dataset |
| `forces` | Finite N×3 array, required for teacher/FP labels | eV/Å, physical force = minus energy gradient; rows follow geometry atom order |
| `stress` | Required by stress-trained recipes | eV/Å³, ASE tensile-positive convention; 3×3 tensor or Voigt order xx, yy, zz, yz, xz, xy |
| `virial` | Required by virial-trained recipes | eV, using the consuming backend's tensor convention |

ASE stress/virial conversion commonly uses `virial = -volume × stress`; the calculator export and native training recipe determine the required shape, sign and order. ASE exposes energy and forces as calculator results.

Teacher reference data and returned FP data contain reference-calculation labels. Initial student data can contain geometry alone; DICE labels it with the current teacher. Seeds contain the geometry required by their [sampling type](sampling.md).

The ingress validator checks required energy/force shapes and finite values. Stress or virial labels are required when the parsed native recipe contains the corresponding key/word; these tensors are checked for finite values. Their physical unit and convention follow the table above and the consuming recipe.

## Identity and population metadata

| Field or rule | Meaning |
| --- | --- |
| `system_id` | Optional stable population name; teacher/FP ingestion falls back to the Hill formula |
| `structure_id`, `structure_source` | Canonical identities generated on ingestion from the computed geometry and configured data source |
| Duplicate geometry | Required labels must be compatible when teacher bootstrap and FP returns are merged |
| Type map | Each frame's elements must be covered by its consuming role's active type map |

`system_id` also identifies teacher validation strata. The exact current training membership is the descriptor reference population for each model. Dataset locations are listed in the run results documentation.

## Returned FP frames

Each returned frame contains the geometry actually computed and its matching labels. If a calculation changes positions or cell, those computed coordinates and cell accompany its energy, forces and stress/virial. Force arrays retain the same atom ordering as that geometry.

`fp_data` lists successful nonempty labelled files retained for the current teacher update. Iteration-index and data-entry constraints are defined in the [configuration reference](run-json.md#required-data-and-identity). Label export, validation commands and starting the next iter are covered by the [scientific walkthrough](../getting-started/quickstart.md).
