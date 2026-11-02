# Labelled data and reference ingestion

DICE exchanges ASE-readable extended XYZ (`.xyz` or `.extxyz`). Supply teacher
and FP labels in eV, Å and eV/Å, converting units before declaring bootstrap
or FP data.

## Minimal labelled frame

The portable example is [labelled_h2.extxyz](../examples/data/labelled_h2.extxyz):

```text
2
Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.0 system_id=h2_demo pbc="F F F"
H 0.00 0.00 0.00  0.0 0.0 0.0
H 0.74 0.00 0.00  0.0 0.0 0.0
```

This synthetic example illustrates the data schema.
`energy` is total energy in **eV per frame**;
positions and lattice vectors are **Å**, forces **eV/Å**. Force rows follow the
same atom order as species and coordinates. PBC is explicit per axis; give a
valid simulation cell for periodic sampling. Concatenate complete frames for a
multi-frame dataset; each external input must contain at least one frame.

| Quantity | Required shape / role | Convention |
| --- | --- | --- |
| Species, positions | N symbols, N×3 coordinates | Symbols covered by the consumer's type map; coordinates Å. |
| Lattice, PBC | 3×3 cell, three flags | Lattice vectors Å; retain actual boundary conditions. |
| `energy` | One finite scalar; teacher/FP required | Total energy eV, using one consistent electronic-structure energy reference across the dataset. |
| `forces` | Finite N×3; teacher/FP required | eV/Å; physical force = minus energy gradient. ASE reads these into single-point calculator results. |
| `stress` | Required by stress-trained recipes | ASE convention eV/Å³, tensile positive; an extxyz 3×3 tensor or ASE Voigt order xx, yy, zz, yz, xz, xy. Convert external pressure sign/units deliberately. |
| `virial` | Required by virial-trained recipes | Energy units eV. Follow the calculator/backend tensor convention. ASE stress and virial conversions usually require `virial = -volume × stress`; verify the exporting calculator's convention/order. |
| `system_id` | Optional metadata string | Stable population identity; absent teacher/FP IDs fall back to Hill formula. Atom types follow species. |
| `structure_id`, `structure_source` | Generated on ingestion | Canonical values are computed from the actual geometry. |

Energy and force shape/finiteness are validated by `validate_label_properties`.
Stress/virial are required if the selected recipe's parsed JSON/YAML contains
the exact word/key `stress`/`virial`. The ingress validator checks finiteness
of these tensors. Verify tensor shape, sign, units and physical conventions
against the selected calculator and native training recipe before ingestion.

## Three distinct populations

Teacher training consumes labelled `teacher_train_bootstrap`, plus explicit
later-iter `fp_data`. Exact-geometry duplicates must have compatible required
labels. Teacher validation is system-stratified; positive fractions retain
training support per eligible system, while zero uses train for self-evaluation.

Student bootstrap at iter 0 is a geometry population: energy/forces may be
absent, because DICE labels these structures using the current teacher. Later
iterations instead relabel the previous iteration's final-round student train
dataset plus final screen train output in the **same run root**. Explicit
`student.include_teacher_train_data: true` additionally merges current teacher
train membership before relabelling. Teacher holdout data remain separate.

GMM/JOINT reference populations use the workflow-owned **current train membership**.
Teacher GMM reference comes from
`iterN/teach/train/dataset/train.xyz`; student reference comes from its current
`rN/train/dataset/train.xyz`. Each model has its own fitted reference.
Rebuild matching reference evidence through the workflow when changing the
model or its training reference.

## Return external FP labels

`iterN/fp/selected.xyz` contains selected geometries and acquisition metadata.
An empty selection is valid. External calculators may preserve
geometry or return a relaxed geometry; returned **computed geometry** and its
matching labels are authoritative. Keep a separate mapping to the selected
frames for your own provenance and record failures, units and calculator
settings. Each returned force array must match its computed geometry and atom order.

Write successful nonempty returns to an explicitly named relative file, check
the selected teacher recipe's label contract, then declare it in the next
iteration's `fp_data`. For iter 2 and beyond, explicitly retain earlier returned
FP files in the declared teacher inputs if they remain in your training set;
FP returns must contain labels computed by the selected first-principles calculator. See the
[worked iter-0 → iter-1 handoff](../examples/iteration_handoff/README.md).
