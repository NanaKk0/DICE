# Inspect results and resume

This guide follows the [Quickstart](../getting-started/quickstart.md): progress, models, sampling, screening, ordinary continuation and explicit rebuilds.

Paths are relative to `examples/dpa4_dpa4c/`. For later iters, substitute their number and use the separate configuration saved for that iter.

## 1. Inspect progress

From another terminal in the run directory:

```sh
tail -n 30 iter0/status.log
```

Use `tail -f iter0/status.log` to follow updates; press `Ctrl+C` to stop viewing.

| Marker | Meaning |
| --- | --- |
| `[STARTED]` | An iter invocation started |
| `[RUNNING]` | Current stage and operation, such as training, labeling or publication |
| `[PROGRESS]` | Training steps, processed frames or selected populations |
| `[SUCCESS]` | A stage published its result successfully |
| `[FAILED]` | A stage attempt failed, with an error and diagnostic path |

The log retains invocation history. Locate the latest `[STARTED]` for the current attempt; earlier completed stages can be reused.

Training records show data populations. Screening records show evaluated frames, training additions and DFT candidates. Terminal selection records show aggregation, deduplication and selection. Interpret each record with its stage coordinate.

Logs support progress and diagnosis. DICE checks published results against their downstream input requirements when continuing or consuming them.

## 2. Read screening reports

```sh
cat iter0/dice.log
```

Reports are grouped by stages such as `r0/screen` and `r1/screen`:

| Field | Meaning |
| --- | --- |
| `N` | Frames evaluated by that role; denominator for table percentages |
| `thresholds` | GMM score thresholds |
| `covered`, `candidate`, `discard` | Populations classified by those thresholds |
| `published: train` | Published student-training additions |
| `published: fp_candidate` | Retained DFT candidates |

The example prescreens up to 200 frames with student JOINT_DIRECT. Teacher GMM trust is `p > 0.1`, and the DFT band is `0 <= p < 0.1`. DIRECT permits up to 50 training additions per round and 20 final DFT selections across both rounds. Compare your run's populations with these configuration limits. See [Configuration screening](../concepts/screening.md) and [Screening fields](../reference/run-json.md#screening-fields-and-terminal-selection).

## 3. Locate key outputs

### Models and training data

| Path | Contents |
| --- | --- |
| `iter0/teach/train/dataset/train.xyz` | Teacher training data |
| `iter0/teach/train/dataset/split_summary.json` | Teacher-data sources and train/validation split |
| `iter0/teach/train/teacher/teacher_model.pt2` | Frozen teacher in this example |
| `iter0/r1/train/dataset/train.xyz` | Second-round student data |
| `iter0/r1/train/student/student_model.pt2` | Frozen second-round student |
| `iter0/r1/sample/sampling_models/student/model_compressed.pt2` | Compressed model used for second-round sampling |

`input.json` stores the generated training input and `lcurve.out` stores learning-curve data. Inspect `deepmd_train.stdout.log` and `deepmd_train.stderr.log` for detailed training output. Freeze/compression logs live with their operations.

### Sampling inputs and trajectories

Seed/condition combinations form separate tasks. For example:

```text
iter0/r1/sample/tasks/task_000000/
```

| File | Contents |
| --- | --- |
| `input.lammps` | Final assembled script |
| `protocol.lammps` | Task sampling protocol |
| `traj.lammpstrj` | LAMMPS trajectory |
| `log.lammps` | Thermodynamic output and run records |
| `lammps_sample.stdout.log`, `lammps_sample.stderr.log` | Process output |

Tasks have separate directories. Eligible positive-step MD observations feed screening.

### Training additions and DFT candidates

| Path | Contents |
| --- | --- |
| `iter0/r0/screen/train.xyz` | First-round teacher-labelled additions for the next student round |
| `iter0/r1/screen/train.xyz` | Terminal-round additions carried into the next iter |
| `iter0/r0/screen/fp_candidate.xyz` | First-round DFT candidates |
| `iter0/r1/screen/fp_candidate.xyz` | Second-round DFT candidates |
| `iter0/fp/selected.xyz` | Final DFT selection |

Read output populations with ASE:

```sh
python - <<'PYCODE'
from ase.io import read

paths = [
    "iter0/r0/screen/train.xyz",
    "iter0/r1/screen/train.xyz",
    "iter0/r0/screen/fp_candidate.xyz",
    "iter0/r1/screen/fp_candidate.xyz",
    "iter0/fp/selected.xyz",
]
for path in paths:
    print(f"{path}: {len(read(path, ':'))} frames")
PYCODE
```

With visualization enabled, `iter0/fp/visualize/` shows structure-level and element-level candidate, background and selection distributions. Data files and reports define the selected membership.

## 4. Continue after interruption or failure

Keep the run configuration, inputs and existing outputs in the original directory. After resolving the reported environment or configuration issue, invoke normally:

```sh
dice run.json --environment environment.json --execution execution.json
```

DICE reuses usable published results and executes the remaining work. A usable terminal result is reused directly.

For `[FAILED]`, inspect the diagnostic path and relevant backend log first. Failed-attempt diagnostics are retained in the stage's `.attempts/`. Keep the directories and use DICE's public continuation/rebuild commands.

See [Troubleshooting](troubleshooting.md) for input, task-log and selection checks.

## 5. Rebuild a selected stage

Use `--resume` to apply changed settings to completed work. It rebuilds the selected stage and **all later stages in the current iter**.

| Goal | Start | Rebuilt work in the two-round example |
| --- | --- | --- |
| Reselect DFT configurations | `fp` | Terminal selection |
| Rescreen second-round data | `r1/screen` | Screening and FP |
| Repeat second-round sampling | `r1/sample` | Sampling, screening and FP |
| Retrain the second-round student | `r1/train` | Training, sampling, screening and FP |
| Repeat both student rounds | `r0/train` | Both student rounds and FP |
| Retrain the teacher | `teach/train` | Teacher preparation and all later current-iter stages |

### Check rebuild feasibility

For example:

```sh
dice run.json --environment environment.json --execution execution.json --probe r1/sample
```

`status: "runnable"` means the exact boundary's required published inputs and suffix checks pass. For `blocked`, inspect `blocking_result`, `consuming_block`, `recommended_resume` and `reason`, then choose the producer that needs repair.

### Preflight and execute

After choosing the boundary:

```sh
dice run.json --environment environment.json --execution execution.json --resume r1/sample --preflight
```

Then:

```sh
dice run.json --environment environment.json --execution execution.json --resume r1/sample
```

This preserves teacher preparation and second-round student training, and rebuilds `r1/sample`, `r1/screen` and `fp`. Rebuilds operate at DICE-stage granularity; MD restarts from the configured seed structures.

See [Configuration overview](configuration.md), [Configure sampling](sampling.md) and [Compute resources](resources.md) for settings. After FP selection, continue with [DFT-label return](../getting-started/quickstart.md#5-prepare-new-labels-for-iter1).
