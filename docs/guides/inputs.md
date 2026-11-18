# Prepare input files

After the [Quickstart](../getting-started/quickstart.md), use this guide to prepare data and native recipes for your system. The layout below is a recommendation; configure the actual filenames and subdirectories in `run.json`.

## 1. Choose a run directory

For example, create `my-dice-run/` outside the checkout:

```sh
mkdir -p ../my-dice-run/data ../my-dice-run/seeds ../my-dice-run/recipes
```

A suggested layout is:

```text
my-dice-run/
├── run.json                       # DICE scientific configuration
├── environment.json               # Software environment profile
├── activate-native.sh             # Backend activation script
├── data/
│   ├── teacher_initial.extxyz      # Initial teacher data with reference labels
│   └── student_initial.extxyz      # Initial student training configurations
├── seeds/
│   └── seed.extxyz                 # Sampling starting structures
├── recipes/
│   ├── teacher.json                # Native DPA4 training recipe
│   ├── student.json                # Native DPA4C training recipe
│   └── nvt.lammps                  # Example LAMMPS sampling protocol
└── models/                        # Optional fine-tuning inputs
    ├── teacher-foundation.pt       # Optional DPA4 checkpoint
    └── student-foundation.pt       # Optional DPA4C checkpoint
```

Input paths in `run.json` are relative to its run directory. A teacher dataset in `data/` is `data/teacher_initial.extxyz`; a file beside `run.json` is `teacher_initial.extxyz`. Use your existing layout by supplying the corresponding paths.

DICE creates output directories such as `iter0/` when it runs.

## 2. Prepare datasets and starting structures

DICE uses ASE-readable extended XYZ (`.xyz` or `.extxyz`). Each input below contains at least one frame:

| Suggested path | Contents | Configuration field |
| --- | --- | --- |
| `data/teacher_initial.extxyz` | Initial DFT reference data: elements, positions, cell, PBC, total frame energy and per-atom forces; include stress or virial when used for training | `teacher_train_bootstrap` |
| `data/student_initial.extxyz` | Initial student geometries; DICE generates their labels with the current teacher | `student_train_bootstrap` |
| `seeds/seed.extxyz` | Starting structures for sampling; periodic NVT uses a valid cell, correct PBC and suitable atomic positions | `sample.seeds` entries' `source` |

Use Å for coordinates/cells, eV for total energy and eV/Å for forces. Match force rows to atom order. Prepare stress/virial units, signs and ordering according to [Data format](../reference/data.md). The models must cover the input elements, and reference labels should use consistent calculation settings and energy conventions.

A sampling seed uses frame zero by default. See [Add starting structures](sampling.md#5-add-starting-structures) for multiple seed roles and multiple frames.

## 3. Prepare native training and sampling recipes

A native recipe is an input read by the backend itself. DICE references it from `run.json` and arranges its use in the workflow.

| File | Role |
| --- | --- |
| `recipes/teacher.json` | Native DPA4 teacher-training recipe |
| `recipes/student.json` | Native DPA4C student-training recipe |
| `recipes/nvt.lammps` | Example native LAMMPS sampling protocol |

DICE organizes training/validation data and binds their paths. Remove pre-existing `training.training_data.systems` and `training.validation_data.systems` from copied DeePMD recipes while retaining the training parameters you want.

Write ordinary LAMMPS syntax in `recipes/nvt.lammps`. DICE can inherit native settings or apply explicit batch conditions; see [Configure sampling](sampling.md).

## 4. Prepare models for the training policy

Teacher and student initialization are configured separately:

| Policy | Prepare |
| --- | --- |
| Train from scratch | A native recipe for the selected model |
| Fine-tune a pretrained model | A native checkpoint and matching training recipe |

For fine-tuning, put the checkpoint in `models/` or another location in the run directory and configure its relative path. Original download filenames can be retained. Choose a model covering your elements and compatible with the backend.

See [DeePMD initialization and native recipes](deepmd.md#initialization-and-native-recipes) for element maps, initialization fields and model formats. Other models use their backend guides.

Continue with [Configuration overview](configuration.md) to connect data, models and sampling in `run.json`.
