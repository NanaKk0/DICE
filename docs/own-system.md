# Your first scientific run

This walkthrough uses a single DPA4 teacher, a single compressed DPA4C student
and two LAMMPS rounds. It supplies a complete DICE configuration, while you
supply genuine data, matching foundation checkpoints, native training recipes
and a qualified engine installation. Start with [models](native-models.md)
and the [DeepMD guide](deepmd.md).

The example uses Al/O, 300 K and short exploration budgets to explain the
interface. These are **illustrative choices**, not recommended parameters for
a material. Choose validation fractions, training budgets, learning rates,
thresholds and MD settings for your study before executing it.

## Prepare the run directory

Keep the run directory separate from your installed checkout. All scientific
paths below are relative to its `run.json`:

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

| Input | What you must provide |
| --- | --- |
| `bootstrap.extxyz` | Nonempty FP-labelled teacher data with consistent energy, force and applicable stress conventions |
| `student_seeds.extxyz` | Initial student geometries; DICE obtains their labels from the teacher |
| `md.extxyz` | Physically sensible periodic MD starting geometry; this example uses frame zero |
| Foundation checkpoints | Official DPA4/DPA4C checkpoints covering Al/O, with compatible native recipes |
| Training recipes | Native architecture/fine-tuning settings and the intended optimization budgets |
| LAMMPS recipe | Valid dynamics settings and the managed setup include illustrated below |

Reserve independent scientific test data. Teacher `valid_frac` controls its
training validation split, not an independent assessment of scientific accuracy.
See [data preparation](data.md) for labels and population membership.

## Write run.json

This is the full configuration; do not add engine executables or scheduler
settings to it. It chooses a two-round explicit sampling plan and compresses
the DPA4C student for sampling.

```json
{
  "iter": 0,
  "teacher_train_bootstrap": [
    "data/bootstrap.extxyz"
  ],
  "student_train_bootstrap": [
    "data/student_seeds.extxyz"
  ],
  "fp_data": [],
  "teacher": {
    "backend": "DeepMD",
    "model": "single",
    "cli": "pt",
    "type_map": [
      "Al",
      "O"
    ],
    "policy": "foundation_finetune",
    "foundation_model": "models/teacher-foundation.pt",
    "recipe": "recipes/teacher.json",
    "valid_frac": 0.1
  },
  "student": {
    "backend": "DeepMD",
    "model": "single",
    "cli": "pt",
    "type_map": [
      "Al",
      "O"
    ],
    "policy": "foundation_finetune",
    "foundation_model": "models/student-foundation.pt",
    "recipe": "recipes/student.json",
    "valid_frac": 0.1,
    "compress": true
  },
  "sampler": {
    "backend": "LAMMPS"
  },
  "sample": {
    "seeds": {
      "bulk": {
        "source": "seeds/md.extxyz",
        "selection_group": "bulk"
      }
    },
    "condition_groups": {
      "warm": {
        "kind": "nvt",
        "recipe": "recipes/nvt.lammps",
        "temperatures_k": [
          300
        ],
        "steps": 200
      }
    },
    "round_count": 2
  },
  "screen": {
    "pool": 100,
    "keep": 10,
    "policy": "DIRECT",
    "fp_pool": 20,
    "teacher_p": {
      "floor": 0.0,
      "ceiling": 0.1,
      "trust": 0.1
    },
    "student_p": {
      "floor": 0.0,
      "ceiling": 0.1
    }
  },
  "fp": {
    "keep": 10,
    "policy": "DIRECT",
    "all_rn": true
  }
}
```

The quotas count frames per selection group. In this example teacher trust
requires `p > 0.1`, while its FP reservation uses `0 <= p < 0.1`.
The two views are independent; other valid bands can overlap. These p values
are training-reference tail ranks, not probabilities that predictions are
correct. Consult [screening rules](configuration.md) before choosing them.

This configuration leaves default diagnostics enabled, pair guard disabled,
GMM prescreening selected, and teacher-data inclusion in student bootstrap
false. Default safe filling cannot produce more eligible data than the route
provides. The full field reference is [configuration.md](configuration.md).

## Prepare the native recipes

Start with recipes matching the **exact** foundation files. Preserve the
foundation's complete native element map, even when the active workflow map
contains only Al/O. Set architecture, loss weights, learning-rate schedule,
training steps and batch size in those recipes. DICE does not choose those
scientific values for you.

Remove pre-existing `training_data.systems` and `validation_data.systems`:
DICE binds generated data. Do not insert the scientific run's active element
subset into a foundation checkpoint's native map. DICE selects `--pt` for the
DPA4 teacher and `--pt-expt` for the DPA4C student; both public roles declare
`cli: "pt"`. [DeepMD usage](deepmd.md) explains presets, ZBL and exports.

Save the following as `recipes/nvt.lammps`. Its 1 fs timestep, output every
20 steps and 0.1 ps damping are example choices requiring review for your system.
The managed include supplies geometry, pair style, models and trajectory output.

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

## Connect the native environment

Save the following complete profile as `environment.json`. It is also shipped
as [the PT profile template](../examples/profiles/dpa4c_dpa4.json).
Both profiles deliberately source your run-local `activate-native.sh`.

```json
{
  "schema": "dice.environment-profile",
  "software_profiles": {
    "models": {
      "environment": {
        "source_list": [
          "activate-native.sh"
        ]
      },
      "executables": {
        "dp": "dp",
        "python": "python"
      }
    },
    "sampling": {
      "environment": {
        "source_list": [
          "activate-native.sh"
        ]
      },
      "executables": {
        "lmp": "lmp",
        "python": "python"
      }
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

Prepare `activate-native.sh` from the [activation template](../examples/profiles/dpa4c-dpa4.env.sh),
editing it to activate your actual matching Python/C++/LAMMPS stack. It must
expose `python`, `dp` and a compatible `lmp` (or your validated wrapper).
Use `DP_INTERFACE_PREC=high` for the newer qualified stack and
`DICE_LAMMPS_PARALLEL_TASKS=1` for serial sampling. This example does not start
MPS. A single DPA4C student selects Kokkos, so `lmp` must provide `deepmd/kk`.
See [environment profiles](environments.md).

## Validate and run

First inspect the scientific structure without activating engines:

```sh
python -c 'from workflow_core.lifecycle.config import load_run_config; c=load_run_config("run.json"); print(c.route.kind.value, c.round_count)'
```

This checks the configuration shape, not native model accuracy or GPU availability.
After supplying the actual inputs and qualifying your native installation,
activate the DICE control environment, change to `my-system/`, then run:

```sh
dice run.json --environment environment.json --preflight
dice run.json --environment environment.json
```

Preflight may source activation scripts. The second command launches training
and sampling. The bundled CPU checks validate this tutorial's loader/profile
and managed-recipe contracts; they do not execute these native calculations.

## Inspect the result

| Output or observation | Interpretation |
| --- | --- |
| `iter0/teach/train/` | Adapted teacher and dataset membership |
| `iter0/r0/`, `iter0/r1/` | Round-specific student training, sampling and screening |
| No new screen training frames | Next round can train on an unchanged dataset; examine trust, candidate bands and coverage |
| Empty `iter0/fp/selected.xyz` | Valid endpoint; inspect populations/thresholds before changing the exploration plan |
| Failed compression or sampling | Inspect backend logs and model/build compatibility; a failed operation is not a completed sample |
| `iter0/status.log`, `iter0/dice.log` | Lifecycle progress and scientific screening counts |

DICE reuses outputs that pass their consumer checks. A successful iteration
shows workflow completion, not convergence or physical accuracy of the model.
Assess validation/test errors and intended simulation behaviour separately.

## Return FP labels and run iter1

Compute the selected geometries externally with a consistent FP protocol.
Write successful nonempty labelled returns to `data/returned.extxyz` and retain
the completed iteration tree. Update the same `run.json`:

- Set `iter` to 1 and `student_train_bootstrap` to `[]`.
- Set `fp_data` to `["data/returned.extxyz"]`, retaining earlier return files
  explicitly when they must remain teacher inputs in subsequent iterations.
- Keep `teacher_train_bootstrap` pointing to the external baseline.
- Review any intentional recipe, model or exploration changes before running.

Recheck and preflight before the ordinary `dice` invocation. The student starts
from the previous final-round train population plus its screen additions,
relabelled by the current teacher. `include_teacher_train_data` can add current
teacher train geometries explicitly. Data carry-over does not imply optimizer
or weight restart. The [CPU handoff](../examples/iteration_handoff/README.md)
demonstrates ingestion and membership without native computation.
