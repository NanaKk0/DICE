# Workflow and artifacts

`dice run.json` runs one explicitly configured iteration. Its lifecycle is:

```text
teach/train -> teach/screen -> teach/scan (single teacher only)
-> r0/train -> r0/sample -> r0/screen -> ... -> fp
                                                -> fp/label (optional)
```

`teach/train` prepares the teacher dataset membership and trains ordered model
members. `teach/screen` prepares student training data using teacher labels.
A single teacher also publishes its reusable GMM reference in `teach/scan`.
Each round trains the student, samples structures and screens them for student
training and FP candidacy. These are distinct destinations and can overlap.

Screening is student-first. A single model uses empirical reference-tail support
routing where applicable; a committee uses force-deviation evidence. The
selector is applied after the route's eligibility gates. Pool/keep quotas set
upper bounds on the eligible structures selected by each group.

Explicit seed roles may share a stable `selection_group`; their samples are
combined before the group quota. Per-round seed/condition activation changes
which tasks participate while preserving the group boundary. Sampling tasks
in each round use their declared seed inputs independently.

## Inputs and outputs

The run directory contains `run.json`, all explicitly referenced relative
inputs/recipes, `iterN/` and hidden `.dice_state/`. It can be outside the
installed source tree. Canonical extxyz is the workflow structure format;
native backend files provide execution evidence.

The default endpoint `iterN/fp/selected.xyz` hands selected geometries to an
external labelling process. An empty selection is valid. Return labelled extxyz by declaring its
path in a later iteration's `fp_data`; label/data conventions must match the
chosen backend recipe and chemistry.

See [labelled-data schema and units](data.md), the
[own-system adaptation guide](own-system.md), and the executable
[next-iteration handoff](../examples/iteration_handoff/README.md). Keep prior
iteration history in the same run root; later iterations relabel the prior
final student predecessor. Teacher-reference inclusion is selected explicitly
through `student.include_teacher_train_data`.

Explicit `fp.label` adds one same-host calculator operation and publishes
`iterN/fp/label/labeled.xyz`. The ABACUS adapter requires an installation-specific
native execution check.

Models retain validated native suffixes such as `.model`, `.pb`, `.pt2` and
`.txt`. Published model presentations are regular files; use their declared
format and validated suffix when loading them.

`iterN/status.log`, `iterN/dice.log`, backend logs and receipts are diagnostics.
Only consumer-runnable canonical artifacts establish block completion.
Default-on selection visualization is non-authoritative; tiny populations or
missing optional diagnostic evidence may legitimately skip it.
