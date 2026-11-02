# Run DICE on your own system

This walkthrough connects user-supplied labelled data, foundation checkpoints,
native recipes and one complete DICE configuration for a
DPA4 teacher / DPA4C student / LAMMPS workflow.

First read [models and native environments](native-models.md). The workflow
configuration below uses this snapshot's `dp --pt` adapters. The model guide
explains the matching checkpoints and specialized backend requirements. For a CPU-only
installation/lifecycle check use the [mock](../examples/cpu_mock/README.md).

## 1. Supply the inputs

Create a run directory outside the installed source. The following names are
the concrete relative paths used throughout this walkthrough:

```text
my-system/
  run.json
  environment.json
  activate-native.sh
  data/bootstrap.extxyz
  data/student_seeds.extxyz
  seeds/md.extxyz
  models/teacher-foundation.pt
  models/student-foundation.pt
  recipes/teacher.json
  recipes/student.json
  recipes/nvt.lammps
```

| User-supplied file | Role |
| --- | --- |
| `data/bootstrap.extxyz` | Nonempty DFT-labelled teacher bootstrap; consistent energy/force units, atom/cell order and DFT protocol |
| `data/student_seeds.extxyz` | Nonempty initial student geometries; DICE obtains their labels from the teacher |
| `seeds/md.extxyz` | Periodic starting geometry for exploration; frame zero is used in this example |
| `models/teacher-foundation.pt`, `models/student-foundation.pt` | Genuine compatible foundation checkpoints obtained separately |
| `recipes/teacher.json`, `recipes/student.json` | Matching native training recipes, prepared for those checkpoints |

The active Al/O chemistry below is an illustrative choice. Change both active
type maps together for your own system and verify coverage of every structure.
Supply the actual model checkpoint files for native execution.
The exact labelled-extxyz contract is in [data.md](data.md).
Reserve independent scientific test geometries separately from the training
and `valid_frac` validation populations.

## 2. Write the complete run.json

The following complete configuration illustrates two 200-step NVT exploration
rounds at 300 K, with frame quotas per selection group. Adapt the numerical
settings and native recipes to your material and study.

```json
{
  "iter": 0,
  "teacher_train_bootstrap": ["data/bootstrap.extxyz"],
  "student_train_bootstrap": ["data/student_seeds.extxyz"],
  "fp_data": [],
  "teacher": {
    "backend": "DeepMD", "model": "single", "cli": "pt",
    "type_map": ["Al", "O"],
    "policy": "foundation_finetune",
    "foundation_model": "models/teacher-foundation.pt",
    "recipe": "recipes/teacher.json", "valid_frac": 0.1
  },
  "student": {
    "backend": "DeepMD", "model": "single", "cli": "pt",
    "type_map": ["Al", "O"],
    "policy": "foundation_finetune",
    "foundation_model": "models/student-foundation.pt",
    "recipe": "recipes/student.json", "valid_frac": 0.1,
    "compress": true
  },
  "sampler": {"backend": "LAMMPS"},
  "sample": {
    "seeds": {"bulk": {"source": "seeds/md.extxyz", "selection_group": "bulk"}},
    "condition_groups": {
      "warm": {"kind": "nvt", "recipe": "recipes/nvt.lammps", "temperatures_k": [300], "steps": 200}
    },
    "round_count": 2
  },
  "screen": {
    "pool": 100, "keep": 10, "policy": "DIRECT", "fp_pool": 20,
    "teacher_p": {"floor": 0.0, "ceiling": 0.1, "trust": 0.1},
    "student_p": {"floor": 0.0, "ceiling": 0.1}
  },
  "fp": {"keep": 10, "policy": "DIRECT", "all_rn": true}
}
```

Here teacher labels require `p > 0.1`; the explicitly chosen FP reservation band
is `0 <= p < 0.1`. Both selections exclude a score exactly equal to 0.1
under these settings. To include equality in the FP band, choose and report
an appropriate upper bound. FP reservation and trusted-label selection are
independent views. See [the exact boundary definitions](configuration.md).

The official defaults retained here include the GMM student prescreen,
default-on diagnostics, no pair guard, and safe-region filling. Teacher-data
inclusion in student bootstrap defaults to false. Quotas set the maximum
number of selected structures. Training learning rates, losses, budgets,
batch sizes and thermostat/timestep settings belong to native recipes.

## 3. Connect the native recipes

Prepare the teacher and student JSON files from the configurations matching
your chosen checkpoints. Preserve their native architecture and complete
foundation type map. Choose learning rate, loss weights, budget and batch size
for your study. Remove pre-existing `training_data.systems` and
`validation_data.systems`: DICE binds its generated datasets. DICE supplies
the declared fine-tuning checkpoint and workflow-owned output names.

For the teacher, confirm the DPA4/SeZM training, inference export and paired
representation-readout path. For the student, confirm the DPA4C training,
freeze and compression path. Select the recipe and CLI together using the
compatibility information in [native-models.md](native-models.md).

The following NVT input illustrates the managed LAMMPS contract. Save it as
`recipes/nvt.lammps`. Its 1 fs timestep, 20-step output cadence and 0.1 ps damping must
be reviewed for the intended material before running.

```lammps
units metal
atom_style atomic
boundary p p p
variable dice_trajectory_every_steps equal 20
include dice_task_setup.lammps
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
thermo 20
thermo_style custom step temp pe ke etotal press
velocity all create ${dice_temperature} ${dice_seed} mom yes rot yes dist gaussian
fix thermostat all nvt temp ${dice_temperature} ${dice_temperature} 0.1
run ${dice_run_steps}
```

The managed include owns `read_data`, the pair style, model paths and trajectory
output, supplying the structure, masses, models and observation settings.
With one DPA4C student, DICE chooses Kokkos sampling and requires a compatible
Kokkos build.

## 4. Bind the native environment

Save the following complete profile as `environment.json`. Put your own
activation commands in `activate-native.sh`, making compatible `dp`, `python`
and `lmp` available on PATH with their native shared libraries. Set this
script up for your installation.

```json
{
  "schema": "dice.environment-profile",
  "software_profiles": {
    "models": {
      "environment": {"source_list": ["activate-native.sh"]},
      "executables": {"dp": "dp", "python": "python"}
    },
    "sampling": {
      "environment": {"source_list": ["activate-native.sh"]},
      "executables": {"lmp": "lmp", "python": "python"}
    }
  },
  "bindings": {
    "teacher.deepmd.pt.train": "models",
    "teacher.deepmd.pt.label": "models",
    "teacher.deepmd.pt.gmm_reference": "models",
    "teacher.deepmd.pt.gmm_score": "models",
    "student.deepmd.pt.train": "models",
    "student.deepmd.pt.compress": "models",
    "student.deepmd.pt.gmm_screen": "models",
    "sampler.lammps.pt": "sampling"
  }
}
```

In your activation script set `DICE_LAMMPS_PARALLEL_TASKS=1` for serial
sampling and allocate the required GPU resources for your run. If your installation
requires a wrapper, expose it as `lmp` through the activation script. Other
environment and optional execution settings are described in
[environments.md](environments.md).

## 5. Check, then run

From `my-system/`, check the configuration without activating engines:

```sh
python -c 'from workflow_core.lifecycle.config import load_run_config; c=load_run_config("run.json"); print(c.route.kind.value, c.round_count)'
```

After providing the real files, checking model/build compatibility and choosing
resources, use the installed public interface:

```sh
dice run.json --environment environment.json --preflight
dice run.json --environment environment.json
```

Preflight may source your activation script. The run command starts training
and sampling. The bundled CPU tests check the tutorial's configuration,
environment bindings and managed-recipe syntax.

## 6. Read the results and return DFT labels

| Output | Meaning |
| --- | --- |
| `iter0/teach/` | Adapted teacher, references and initial labelling/screening products |
| `iter0/r0/`, `iter0/r1/` | Student training, sampled observations and screening for each round |
| `iter0/fp/selected.xyz` | Unlabelled geometries selected for external first-principles calculations |
| `iter0/status.log`, `iter0/dice.log` | Lifecycle progress and scientific screening diagnostics |

Canonical artifact names within blocks are explained in
[workflow.md](workflow.md). Check the published canonical outputs for completion.

An empty `selected.xyz` is a valid outcome. Assess the completed model and
sampling coverage to decide whether to stop or prepare a different exploration
plan. The next-iteration procedure below requires nonempty, successfully
computed DFT labels in `fp_data`.

Run DFT externally with your chosen consistent protocol. Keep atom ordering,
cell and geometry paired with the returned labels, and write successfully
labelled extxyz to `data/returned.extxyz`. Keep the completed iteration tree.
For the next invocation, change only the relevant scientific inputs:

- Set `iter` to 1.
- Set `student_train_bootstrap` to `[]`.
- Set `fp_data` to `["data/returned.extxyz"]`, explicitly including earlier
  returned-label files when they must remain teacher inputs.
- Keep `teacher_train_bootstrap` pointing to the original labelled baseline.
- Retain or deliberately update your native recipes and sampling plan.

Recheck, preflight and run the same public command to start the next iteration.
The [CPU handoff test](../examples/iteration_handoff/README.md) demonstrates the
same return-and-carry-over semantics without DFT, training or MD.
