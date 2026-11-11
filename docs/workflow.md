# Understand the workflow

DICE alternates teacher-supervised student training and student-driven
exploration. First-principles (FP) labels are external reference information;
teacher labels are model predictions. Keep their origin distinct.

An **iteration** adapts the teacher once and executes one or more student
**rounds**. `dice run.json` advances the configured iteration. A later iteration
is a later invocation with returned FP data and the preceding student history.

```text
External labelled bootstrap + explicitly retained FP returns
    → teach/train: adapt teacher
    → teach/screen: label/screen initial student geometries
    → teach/scan: prepare reference (single teacher)
    → r0/train → r0/sample → r0/screen
    → r1/train → r1/sample → r1/screen → ...
    → fp: select geometries for external calculations
    → optional fp/label: run a supported local FP calculator
```

## What enters and leaves each step

| Step | Input and action | Result used later |
| --- | --- | --- |
| Teacher training | Declared externally labelled baseline and FP returns | Adapted teacher and its train/validation membership |
| Initial student preparation | Iter0 geometry bootstrap, or later-iter student carry-over | Teacher-labelled student starting population |
| Student training | Current accumulated teacher-labelled student data | Student for this round's exploration |
| Sampling | Declared seed geometries, conditions and native recipe | Observations to evaluate and screen |
| Screening | Student proposals followed by teacher evaluation | Trusted additions for student training and FP candidates |
| FP selection | Candidate pools from the configured rounds | `iterN/fp/selected.xyz` for external labelling |

A single model uses its training-reference support score where supported; a
committee uses force-deviation evidence. The routes have different threshold
units. Student training additions and FP candidates are separate destinations;
teacher views may overlap. Selection quotas are upper bounds, not guarantees.
See [configuration](configuration.md) for exact boundaries.

## Rounds, seeds and training policies

`sample.round_count` repeats the declared seed/condition registry. Alternatively,
`sample.rounds` activates declared subsets per round. A seed's
`selection_group` determines the shared screening and FP quota: multiple
conditions do not automatically multiply that group's quota.

Each sampling task starts from its declared seed input; successive rounds do
not implicitly continue the last trajectory. Each executed train block applies
its declared `from_scratch` or `foundation_finetune` policy. Dataset carry-over
does not itself request a checkpoint restart or continued optimizer state.

A round can add zero training frames. The next round may then train on the
same dataset. Inspect the realized populations before interpreting a completed
iteration as model improvement.

## Run directory and outputs

Work in the directory containing `run.json`. Its relative paths identify your
data, models and recipes. Outputs live in `iterN/`, while `.dice_state/`
contains workflow-managed operational records.

| Path | Purpose |
| --- | --- |
| `iterN/teach/train/dataset/train.xyz` | Current teacher train membership and single-teacher GMM reference population |
| `iterN/r0/train/dataset/train.xyz` | Round-0 student train membership and its reference population |
| `iterN/r0/sample/` | Native sampling results and observations |
| `iterN/r0/screen/` | Published training additions and FP-candidate evidence |
| `iterN/fp/selected.xyz` | Terminal unlabelled geometries; an empty selection is valid |
| `iterN/fp/label/labeled.xyz` | Optional supported local FP operation's labelled return |
| `iterN/status.log`, `iterN/dice.log` | Progress and scientific screening diagnostics |

DICE validates the formal outputs needed by their consumers before reusing a
block. Logs and optional visualization aid diagnosis but do not independently
establish completion. [Recovery](recovery.md) explains ordinary continuation
and deliberate suffix rebuilding.

## Start the next iteration

Keep the same run directory and prior student history. Compute FP labels using
a consistent protocol, then explicitly declare successful nonempty returns in
`fp_data`, increment `iter`, and set `student_train_bootstrap` to `[]`.
Retain the external baseline and earlier FP files you still want in teacher
training. The current teacher relabels the preceding final student population;
`student.include_teacher_train_data` optionally adds current teacher train
geometries. See [data conventions](data.md), the [scientific walkthrough](own-system.md)
and the [executable handoff](../examples/iteration_handoff/README.md).
