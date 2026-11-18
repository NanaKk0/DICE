# Configure sampling

DICE combines starting structures, conditions and native protocols into tasks, then passes observations to screening. This guide builds on the [Quickstart](../getting-started/quickstart.md) to configure LAMMPS MD, batch conditions and rounds.

JSON snippets below identify a `sample` object, seed or condition group. See [Backend environments](backends.md) for software and [Compute resources](resources.md) for placement/concurrency.

## 1. Connect seeds and conditions

With `"sampler": {"backend": "LAMMPS"}`, the following small `sample` plan illustrates the interface:

```json
{
  "seeds": {
    "bulk": {
      "source": "seeds/seed.extxyz",
      "selection_group": "bulk"
    }
  },
  "condition_groups": {
    "nvt": {
      "kind": "nvt",
      "recipe": "recipes/nvt.lammps",
      "temperatures_k": [300],
      "steps": 200,
      "timestep_ps": 0.00025,
      "trajectory_every_steps": 10
    }
  },
  "round_count": 2
}
```

| Setting | Meaning |
| --- | --- |
| `seeds.bulk` | A seed role named `bulk`, reading `source` |
| `selection_group: "bulk"` | Observations share the `bulk` selection quotas |
| `condition_groups.nvt` | A condition group referencing the native recipe |
| `round_count: 2` | Repeat this plan in `r0` and `r1` |

Each condition group applies to every declared seed by default. This plan creates one 300 K task per round, driven by that round's student and starting from frame zero of `seeds/seed.extxyz`.

Choose descriptive seed, condition and selection-group names. Paths are relative to `run.json`.

## 2. Use an ordinary LAMMPS template

A complete `recipes/nvt.lammps` protocol is:

```lammps
units metal
atom_style atomic
boundary p p p
timestep 0.00025
velocity all create 300 91731 mom yes rot no dist gaussian
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
fix bath all nvt temp 300 300 0.1
thermo 10
thermo_style custom step temp pe ke etotal press
run 200
```

Write native initialization, velocity creation, controls, groups, constraints and analysis in the recipe. DICE adds the task's structure, student model, masses and standard trajectory output to create `input.lammps`.

| Location | Settings |
| --- | --- |
| Native template | Algorithm, integration/control options, velocity initialization, groups, constraints, fields and extra analysis/output |
| DICE condition group | Supported ensemble, temperature/pressure arrays, steps, timestep, standard output cadence and override targets |
| DICE assembly | Current seed, student model and standard observations |

The template can specify temperatures, pressures and steps directly. Explicit DICE fields apply the overrides described below.

## 3. Choose inheritance and overrides

### Inherit the native protocol

Replace the value of `sample.condition_groups.nvt` with:

```json
{
  "recipe": "recipes/nvt.lammps"
}
```

This retains the native algorithm, 300 K target, 200 steps and 0.00025 ps timestep. DICE supplies standard trajectory output; this recipe-only form uses its default 20-step cadence. Set `trajectory_every_steps` in the condition group to choose another cadence.

### Override selected fields

For example:

```json
{
  "recipe": "recipes/nvt.lammps",
  "temperatures_k": [300, 600],
  "steps": 400,
  "trajectory_every_steps": 10
}
```

For each seed, DICE creates 300 K and 600 K tasks and writes the selected thermostat target and run length. Timestep, damping and other native settings are inherited.

The template's `velocity all create 300 ...` still initializes at 300 K: `temperatures_k` overrides the thermostat target. Velocity initialization and its RNG seed remain native protocol settings.

| Configuration | Behavior |
| --- | --- |
| Recipe only | Inherit the native protocol and conditions |
| Explicit supported DICE field | Override its selected native control |
| Native setting with the DICE field omitted | Inherit the native value; assembly supplies standard output and defaults where needed |
| Missing native control with complete DICE parameters | Generate the control; an NPT controller needs temperature, pressure, damping and pressure-coupling settings |
| Several possible targets | Select the intended control through `targets` |

An explicit `kind` requests an ensemble. For example, `kind: "nvt"` can convert a selected NPT controller to NVT, preserving thermostat options and removing pressure control. For NPT, use the pressure-control template below.

### Select override targets

For multiple thermostat fixes or `run` commands, add `targets`. To modify fix `production` and the second run:

```json
{
  "targets": {
    "thermostat": "production",
    "run": 2
  }
}
```

Merge this object into the condition group alongside the fields being overridden. `thermostat` and `barostat` use native fix IDs; `run` is a one-based ordinal. DICE resolves a unique target automatically; preflight reports ambiguous candidates.

## 4. Batch temperatures and pressures

`temperatures_k: [300, 600]` creates two independent NVT tasks per seed.

NPT uses the temperature/pressure product. Create `recipes/npt.lammps` from the template above, replacing `fix bath` with:

```lammps
fix bath all npt temp 300 300 0.1 iso 1 1 1.0
```

Add the following `npt` entry to `sample.condition_groups`:

```json
{
  "kind": "npt",
  "recipe": "recipes/npt.lammps",
  "temperatures_k": [300, 600],
  "pressures_bar": [1, 1000],
  "steps": 200,
  "timestep_ps": 0.00025,
  "trajectory_every_steps": 10
}
```

Each seed produces four NPT tasks: 300 K/1 bar, 300 K/1000 bar, 600 K/1 bar and 600 K/1000 bar. Tasks retain the template's velocity initialization, damping and `iso` coupling. Enabling both the two-temperature NVT and NPT groups creates six tasks per seed per round.

DICE uses K, bar and ps. The LAMMPS adapter converts overrides for `units metal` or `units real`. Seed PBC and cells should suit the pressure-control directions.

## 5. Add starting structures

Add another entry to `sample.seeds`:

```json
{
  "bulk_b": {
    "source": "seeds/second.extxyz",
    "selection_group": "bulk"
  }
}
```

Prepare the second source file. Both seeds run the active conditions and share the `bulk` selection group.

For a multi-frame source, set `max_frames` on the seed. For example, replace `sample.seeds.bulk` with:

```json
{
  "source": "seeds/seed.extxyz",
  "selection_group": "bulk",
  "max_frames": 2
}
```

DICE deterministically selects up to two distinct source frames for the current iter/round, then expands conditions for each. A one-frame source supplies one starting structure.

A condition group's `"seeds": ["bulk"]` restricts its tasks to that seed. Omission uses all declared seeds.

## 6. Assign tasks to rounds

`round_count` repeats the declared seeds and conditions. Replace it with `sample.rounds` for per-round activation. With both groups declared:

```json
{
  "rounds": [
    {"condition_groups": ["nvt"]},
    {"condition_groups": ["npt"]}
  ]
}
```

`r0` runs NVT and `r1` runs NPT. A row can also select `seeds`; an omitted axis uses all declared members. Execution follows registry declaration order. Each round uses its current student and starts from the active seed structures.

For continuous heating or staged conditions within one task, use a native multistage protocol or a DICE program type. See [Continuous temperature programs](../reference/sampling.md#continuous-temperature-programs).

## 7. Connect sampling to screening

Task outputs are stored under:

```text
iter0/r0/sample/tasks/task_000000/
```

Each LAMMPS MD task retains `protocol.lammps`, final `input.lammps`, standard `traj.lammpstrj` and logs. NVT, NPT and ordinary native MD use the same observation interface; screening reads eligible positive-step frames before evaluation and selection.

`selection_group` connects tasks to quotas. If `bulk` and `bulk_b` share group `bulk`, their active-condition observations participate in that group's selection. `screen.keep: 4` permits up to four training additions per round/group; `fp.keep: 4` permits up to four final DFT selections per group.

Use different selection groups for independent quotas. Condition groups arrange sampling; selection groups apply quotas to their seeds' observations.

`fp.all_rn: true` combines candidate rounds; `false` or omission uses the terminal round. See [Configuration screening](../concepts/screening.md) and [Inspect results and resume](results-and-recovery.md).

## 8. Preflight and inspect effective controls

From the run directory:

```sh
dice run.json --environment environment.json --execution execution.json --preflight
```

Inspect `sampling_controls`: tasks, effective values, field sources and override targets. The engine resolves dynamic native values at execution time.

After running, inspect `input.lammps` and `log.lammps` for final commands and execution. Apply changes to completed sampling through an explicit [stage rebuild](results-and-recovery.md#5-rebuild-a-selected-stage) from the relevant `rN/sample`.

NEB, relaxation and MD/MC share round and selection-group planning. Their parameters and observation formats are in [Sampling types](../reference/sampling.md).
