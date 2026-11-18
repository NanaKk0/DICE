# Run DPA4/DPA4C and continue iterating

This tutorial runs the water example through `iter0` and `iter1`, using an OMat24 DPA4-Neo teacher, a compressed DPA4C student and LAMMPS NVT sampling. Each iter contains two rounds. Teacher-reliability screening supplies teacher-labelled student-training additions and candidates for DFT.

The example includes reference data, starting structures, native recipes and both run configurations. Adjust the environment and execution files for your installation, prepare the foundation model, then run `run.json` and `run_iter1.json` in sequence.

## Before you start

Complete [Installation](installation.md) and prepare:

| Component | Required capabilities |
| --- | --- |
| DeePMD-kit | DPA4 fine-tuning, labeling and descriptors; DPA4C training, freezing, descriptors and GPU compression |
| LAMMPS with DeepMD | Kokkos and `deepmd/kk` for the compressed DPA4C student |
| Resources | One NVIDIA GPU with suitable CPU and memory resources |

Follow the upstream [DeePMD-kit installation](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html) and [LAMMPS integration](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-lammps.html) instructions. See [Configure backend environments](../guides/backends.md) for DICE bindings.

## 1. Enter the example and prepare the model

From the DICE checkout:

```sh
cd examples/dpa4_dpa4c
python prepare_model.py
```

The script downloads the official OMat24 DPA4-Neo v20260805 checkpoint at a fixed model release revision. If you already have it, place it at `models/DPA4-Neo-OMat24-v20260805.pt`. See [Model source and licence](../../examples/dpa4_dpa4c/models/README.md).

The run directory contains:

```text
dpa4_dpa4c/
├── README.md
├── run.json
├── run_iter1.json
├── prepare_model.py
├── environment.json
├── execution.json
├── activate-native.sh
├── models/
│   └── DPA4-Neo-OMat24-v20260805.pt
├── data/
│   ├── teacher_initial.extxyz
│   ├── student_initial.extxyz
│   ├── demo_iter1_labeled.extxyz
│   └── NOTICE.md
├── seeds/
│   └── seed.extxyz
└── recipes/
    ├── teacher.json
    ├── student.json
    └── nvt.lammps
```

See the [example README](../../examples/dpa4_dpa4c/README.md) for the supplied files. Outputs are written to `iter0/` and `iter1/` in this directory.

## 2. Configure your environment and resources

Check:

| File | Adapt to your installation |
| --- | --- |
| `activate-native.sh` | Conda path, backend environment and GPU selection |
| `environment.json` | Activation scripts and `python`, `dp`, `lmp` names or paths |
| `execution.json` | Execution target, GPU count, CPU and memory budgets |

The activation template uses a Conda environment named `dice`. Model and sampling operations share the script by default; assign separate scripts in `environment.json` for separate installations.

The execution profile uses one local GPU and serial task slots. Check CPU, memory and device settings before starting. Activation preserves an existing `CUDA_VISIBLE_DEVICES`, defaulting to device 0 when unset. See [Configure compute resources](../guides/resources.md).

## 3. Preflight and run iter0

With the DICE control environment active, run:

```sh
dice run.json --environment environment.json --execution execution.json --preflight
```

Resolve any reported environment or execution settings, then start:

```sh
dice run.json --environment environment.json --execution execution.json
```

The scientific configuration is:

| Item | Setting |
| --- | --- |
| Initial data | The same 40 water geometries for teacher and student, 192 atoms per frame; the teacher uses DFT labels and the student receives teacher labels |
| Teacher | OMat24 DPA4-Neo fine-tuned for 2500 steps |
| Student | DPA4C trained from scratch for 5000 steps per round, frozen and compressed for sampling |
| Include teacher data | `student.include_teacher_train_data: true`; geometries are merged, deduplicated and labelled by the current teacher |
| Sampling | Two seed structures at 300 K and 600 K, 2500 NVT steps per task, 0.00025 ps timestep, trajectory every 10 steps |
| Rounds | `r0` and `r1`, each with training, sampling and screening |
| Student prescreen | `JOINT_DIRECT`, pool capacity 200 per round |
| Teacher GMM | `trust: 0.1`, `ceiling: 0.1`, `floor: 0.0` |
| Student additions | DIRECT selects up to 50 teacher-trusted frames per round |
| Final DFT selection | DIRECT selects up to 20 frames from candidates across both rounds |

See [run.json](../../examples/dpa4_dpa4c/run.json). Teacher trust uses `p > 0.1`; the DFT-candidate band is `0 <= p < 0.1`. Quotas are upper bounds; inspect the screening reports for the run's actual selections.

Teacher-data inclusion is enabled in both iters. Initial teacher and student geometries overlap and are deduplicated before student labeling.

## 4. Inspect iter0 outputs

| Path | Contents |
| --- | --- |
| `iter0/status.log` | Workflow progress |
| `iter0/dice.log` | Screening reports and selected populations |
| `iter0/r0/`, `iter0/r1/` | Per-round training, sampling and screening outputs |
| `iter0/r0/screen/train.xyz`, `iter0/r1/screen/train.xyz` | Teacher-labelled additions for subsequent student training |
| `iter0/fp/selected.xyz` | Configurations selected for DFT |

See [Inspect results and resume](../guides/results-and-recovery.md) for model, trajectory and data locations, and [Troubleshooting](../guides/troubleshooting.md) for failures.

## 5. Prepare new labels for iter1

The supplied `data/demo_iter1_labeled.extxyz` contains 20 water configurations with existing DFT labels. It demonstrates adding reference data, updating the teacher and continuing the workflow.

`iter0/fp/selected.xyz` contains configurations selected during your run. Continue the tutorial with the supplied labelled file.

DICE receives reference labels through `fp_data`. In a production iteration, calculate DFT labels for `fp/selected.xyz` using settings and energy conventions consistent with the existing reference data. Other suitable external reference data can also be added.

Validate the supplied file from the run directory:

```sh
python - <<'PYCODE'
from ase.io import read
from workflow_core.lifecycle.data import (
    resolve_label_contract,
    validate_label_properties,
)

frames = read("data/demo_iter1_labeled.extxyz", ":")
if not frames:
    raise ValueError("New reference data must contain at least one frame")
teacher_contract, _ = resolve_label_contract(
    "recipes/teacher.json", "recipes/student.json"
)
validate_label_properties(frames, teacher_contract)
print(f"Labels valid: {len(frames)} frames")
PYCODE
```

Returned data retain the calculated elements, atom order, positions and periodic cells, together with `energy` (eV), `forces` (eV/Å) and any other training labels. See [Data format](../reference/data.md).

## 6. Run iter1

The supplied `run_iter1.json` is a copy of `run.json` with three changes, preserving the original configuration:

| Field | `run_iter1.json` |
| --- | --- |
| `iter` | `1` |
| `student_train_bootstrap` | `[]` |
| `fp_data` | `["data/demo_iter1_labeled.extxyz"]` |

Keep `teacher_train_bootstrap` pointing to `data/teacher_initial.extxyz` and `student.include_teacher_train_data` set to `true`. Model and sampling settings are inherited. See [run_iter1.json](../../examples/dpa4_dpa4c/run_iter1.json).

Preflight and run from the same example directory:

```sh
dice run_iter1.json --environment environment.json --execution execution.json --preflight
dice run_iter1.json --environment environment.json --execution execution.json
```

DICE updates the teacher from the initial reference data and new DFT labels. It merges and deduplicates the previous iter's terminal student data and training additions with the current teacher data, labels the geometries with the new teacher, then completes both student rounds in `iter1`.

`student.include_teacher_train_data: true` includes the new reference geometries in student relabeling. Inspect training logs and data files for the resulting populations.

## 7. Inspect iter1 and continue

| Path | Contents |
| --- | --- |
| `iter1/status.log` | Iteration progress |
| `iter1/teach/train/dataset/split_summary.json` | Initial, new and merged reference-data populations |
| `iter1/teach/screen/train.xyz` | Student starting data relabelled by the new teacher |
| `iter1/r0/`, `iter1/r1/` | Training, sampling and screening outputs |
| `iter1/fp/selected.xyz` | The next DFT selection |

Inspect each round's `screen_report.json` and teacher-labelled `screen/train.xyz`. Training additions feed the next student round; terminal-round additions are carried into the next iter. Each iter publishes its final DFT selection at `fp/selected.xyz`.

For later iterations, copy the preceding configuration in the same run directory, update `iter` and `fp_data`, and keep `student_train_bootstrap: []`. For example:

```sh
cp run_iter1.json run_iter2.json
```

In `run_iter2.json`, list cumulatively in `fp_data` the added reference files you want to retain. Keep initial reference files in `teacher_train_bootstrap`. Student geometries carry forward from the previous iter's terminal round.

For your own system, continue with [Prepare input files](../guides/inputs.md) and [Configuration overview](../guides/configuration.md). See [Workflow](../concepts/workflow.md) for data transfer between rounds and iters.
