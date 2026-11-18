# Sampling type reference

This page defines condition-group kinds, their mode-specific fields, units, input requirements and sample outputs. Seed registries, round activation and common data paths are defined in the [run configuration reference](run-json.md#sampling-field-reference).

## Native protocols and DICE overrides

A recipe-only condition group executes a normal LAMMPS or GPUMD protocol. DICE binds the current seed, student model and standard observation output. Explicit fields override the selected native control; omitted fields inherit its value. Multiple possible targets require a named `targets` entry.

DICE temperatures use K, pressures bar and time fields ps. LAMMPS converts overrides for `units metal` and `units real`; GPUMD converts bar to GPa and ps to fs while keeping native ensemble methods and couplings. Velocity initialization remains in the native protocol. Ordinary LAMMPS includes resolve relative to the including source file.

## Kinds and required fields

| Kind | Required fields beyond kind | Optional fields / defaults and restrictions |
| --- | --- | --- |
| Omitted kind / `native` | `recipe` | LAMMPS or GPUMD; inherit native protocol. Optional temperature/pressure batches, steps, timestep/cadence, targets and inputs. |
| `nvt` | `recipe` | Explicit NVT; omitted temperature/steps inherit native settings. |
| `npt` | `recipe` | Explicit NPT; omitted temperature/pressure/steps inherit native settings. |
| `nvt_program` | `recipe`, `segments` | LAMMPS only; one continuous task per active seed. |
| `npt_program` | `recipe`, `segments`, `pressures_bar` | LAMMPS only; one continuous task per seed/pressure. |
| `nvt_cell_program` | `recipe`, `segments` | LAMMPS only; optional cell endpoint on each segment, never on another kind. |
| `neb` | `images` (>0 movable interior images), `fmax_ev_a` (>0 eV/Å), `max_steps` (>0) | `init: linear` (`idpp` alternative), `method: neb` (`dyneb` alternative), `include_history: false`, `climb: false`, `relax: false`, `fixed_atoms: []`. No recipe. |
| `relax` | `fmax_ev_a` (>0 eV/Å), `max_steps` (>0) | `max_step_a` (>0 Å; omitted uses installed ASE FIRE default), `include_history: true`, `history_stride: 1`, `accept_unconverged: false`, `fixed_atoms: []`. Stride valid only with history. No recipe; atomic relaxation at fixed cell. |
| `md_mc` | `recipe`, `ensemble` (`nvt`/`npt`), `temperatures_k`, `steps`, `mc` | LAMMPS only; NPT also requires `pressures_bar`. `mc` is exactly `move: identity_swap`, two distinct student `elements`, positive `every_md_steps`, positive `attempts_per_event`. Steps must cover >=1 event. No default move in run.json. |
| `widom` | `guest` (relative geometry), `frames` (>0), `max_attempts` (>=frames) | LAMMPS route only; `random_seed: 0`, `overlap: {scale: 0.6, radii_a: {}}`; radii Å, positive scale. `overlap: null` disables the filter; a supplied radii override must be nonempty and complete for used elements. No recipe or MD budget. |
| `gcmc_md` | `temperatures_k`, `cycles`, `mc` | LAMMPS only; `random_seed` absent, `md` absent (pure MC), `limits: {}`. Detailed nested fields below. Used for configuration acquisition. |

### Native control fields

Ordinary LAMMPS/GPUMD protocol groups accept these optional fields. Each
explicit field overrides its selected native counterpart; omitted fields inherit.

| Condition-group field | Meaning |
| --- | --- |
| `temperatures_k` | Nonempty unique array of finite positive K; independent tasks at each target thermostat temperature. |
| `pressures_bar` | Finite scalar or nonempty unique array in bar; independent tasks at each target mechanical pressure. |
| `steps` | Positive integer or per-round array; budget of the selected run. |
| `timestep_ps` | Positive finite ps; convert to native engine units. |
| `trajectory_every_steps` | Positive integer; standard observation cadence. |
| `targets.thermostat`, `targets.barostat` | LAMMPS native fix IDs; omitted target selects a unique candidate. |
| `targets.run` | Positive one-based native run ordinal for a steps override. |
| `inputs` | Native task-relative filename → source path relative to `run.json`. |

LAMMPS `nvt/npt` groups also accept `lammps` with optional `control` and optional
`timestep_ps` / `trajectory_every_steps`. These nested time/cadence fields are
aliases of the group fields; write one source for each setting.

| Field within `lammps.control` | Meaning |
| --- | --- |
| `group` | Native atom-group ID for the selected controller. |
| `fix_id` | Native controller ID, corresponding to `targets.thermostat`. |
| `temperature_damping_ps` | Positive finite ps; selected thermostat damping. |
| `pressure_damping_ps` | Positive finite ps; selected barostat damping. |
| `pressure_coupling` | `iso`, `aniso` or `tri`. |

A missing controller can be generated when DICE provides its required target
and damping settings. Existing initialization and unrelated controllers remain
in the native protocol. Historical `control.mode: managed/native` is accepted
for compatibility; new inputs use the ordinary protocol and field rules above.
Examples are listed below under each sampling kind.

## GCMC nested reference

| Field under condition group | Type / default | Meaning |
| --- | --- | --- |
| `mc.guest`, `mc.pressures_bar`, `mc.moves`, `mc.eos` | Required | Guest relative path; reservoir pressure is a nonempty positive bar array; moves positive integer; EOS object. |
| `mc.every` | Positive integer; `100` | MC observation interval in moves. |
| `mc.translate_a`, `mc.rotate_deg` | `0.75` Å, `90` degrees | Positive translation/rotation bounds; rotation <=180. |
| `mc.weights` | `{insertion: .25, deletion: .25, translation: .25, rotation: .25, reinsertion: 0}` | If supplied, all five keys required; nonnegative, sum 1, equal positive insertion/deletion and zero reinsertion. |
| `mc.eos.kind` | Required: `ideal`, `fixed_phi`, `pr` | `phi` required only for fixed_phi, positive dimensionless. PR requires positive `tc_k` (K), `pc_bar` (bar), finite `omega` (dimensionless); these are forbidden for other EOS kinds. |
| `mc.overlap` | `{scale: 0.6, radii_a: {}}` | Positive scale, optional positive per-element radii in Å. Unlike Widom, null is not accepted. |
| `mc.de_max_ev` | Absent / null | Optional positive finite eV bound. |
| `limits.max_guests` | Absent / null | Optional positive integer guest cap. |
| `random_seed` | Absent / null | Optional nonnegative integer. Do not infer a global workflow seed. |
| `md` | Absent | Optional interleaved MD: exactly `ensemble` (`nvt`, `nve`, `npt`), relative `recipe`, positive `steps`, positive `every`; NPT additionally requires finite `pressure_bar` (bar), forbidden otherwise. |

The generated inventory lists accepted keys. The loader enforces compatible
combinations. Advanced kinds require backend-specific input/geometry checks
and a matching native environment.

## Condition fragments

Each JSON block below is one condition-group value under `sample.condition_groups.<name>`. Numerical values illustrate syntax.
The recipes and seed/guest files must exist before native execution.

### Fixed NVT

```json
{"kind": "nvt", "recipe": "recipes/nvt.lammps", "temperatures_k": [300], "steps": 200, "timestep_ps": 0.001, "trajectory_every_steps": 20}
```

### Fixed NPT

```json
{"kind": "npt", "recipe": "recipes/npt.lammps", "temperatures_k": [300, 600], "pressures_bar": [1], "steps": 200, "timestep_ps": 0.001, "trajectory_every_steps": 20}
```

Temperatures are K and pressures bar. The native recipe supplies the matching
thermostat/barostat and their damping. GPUMD groups use GPUMD native recipes.
NPT requires a periodic cell compatible with the selected pressure coupling.

### Continuous temperature programs

```json
{"kind": "nvt_program", "recipe": "recipes/program.lammps", "segments": [{"temperature_k": 300, "steps": 100}, {"start_temperature_k": 300, "stop_temperature_k": 600, "steps": 100}]}
```

```json
{"kind": "npt_program", "recipe": "recipes/program-npt.lammps", "pressures_bar": [1], "segments": [{"start_temperature_k": 300, "stop_temperature_k": 600, "steps": 200}]}
```

A program segment is exactly `{temperature_k, steps}` (hold) or
`{start_temperature_k, stop_temperature_k, steps}` (linear ramp); temperatures
are positive K, steps positive integers.

LAMMPS only. Supply a normal matching NVT/NPT template. DICE carries velocities and cell continuously through the ordered segments in one task.

### NVT cell program

```json
{"kind": "nvt_cell_program", "recipe": "recipes/cell-program.lammps", "segments": [{"temperature_k": 300, "steps": 200, "cell": {"mode": "isotropic_density", "stop_density_g_cm3": 5.0}}]}
```

LAMMPS only. Supply a normal NVT template. DICE generates its continuous cell history. For `nvt_cell_program` only, optional
`cell` is either `{mode: isotropic_density, stop_density_g_cm3: positive}` or
`{mode: symmetric_strain, stop: {xx, yy, zz, yz, xz, xy}}` with all six finite
dimensionless strain components and positive-definite `I+E`. State persists
through the ordered segments; no segment-resume interface exists.

### NEB

```json
{"kind": "neb", "images": 5, "fmax_ev_a": 0.05, "max_steps": 200}
```

ASE-driven DeepMD or NEP calculator route. NEB sources are ordered two-frame endpoints with identical atom count, symbol
order, PBC and exact fixed cell. `fixed_atoms` are unique zero-based atom indices
and endpoint coordinates at fixed indices must coincide. A role cannot serve
NEB and non-NEB groups together. Optional
`climb_preconverge_fmax_ev_a` is >fmax, requires `method: neb`, `climb: true`,
and shares the one path-step budget between preconvergence and climbing. DyNEB
means dynamic relaxation, not doubly nudged NEB. Optional endpoint `relax`
gives each endpoint a separate full step budget before the path. Default
final-only exposes the converged complete band including endpoints; history
exposes initial/completed interior steps. The first student member drives optimization and all members evaluate exposed structures.

### Fixed-cell relaxation

```json
{"kind": "relax", "fmax_ev_a": 0.05, "max_steps": 200, "include_history": true, "history_stride": 10}
```

ASE FIRE with a DeepMD or NEP calculator, optimizing atomic positions at fixed cell. Relax history includes initial and
stride-selected steps plus the final frame exactly once; unconverged endpoints
are accepted only with explicit `accept_unconverged: true`. The first student
member drives optimization and all members evaluate exposed structures.

### Fixed-composition MD/MC

```json
{"kind": "md_mc", "ensemble": "nvt", "recipe": "recipes/swap.lammps", "temperatures_k": [600], "steps": 200, "mc": {"move": "identity_swap", "elements": ["Al", "O"], "every_md_steps": 20, "attempts_per_event": 1}}
```

LAMMPS only. This syntax example swaps two existing element identities and
preserves composition; choose physically meaningful exchange species. NPT also
requires pressures. The first student drives MD and whole-system Metropolis
swaps. Retained chain states enter observations. The recipe supplies a normal NVT/NPT
controller, initialization and timestep; DICE compiles the swap/audit sequence.

### Independent insertion (Widom acquisition)

```json
{"kind": "widom", "guest": "seeds/guest.extxyz", "frames": 10, "max_attempts": 100, "random_seed": 0}
```

DeepMD/LAMMPS-selected Python calculator route; no MD recipe. The host seed and
guest library must have compatible species and geometry. `frames` is the
geometrically qualified target per host task; `max_attempts` bounds retries.
Overlap filtering and underfill are separate from screen selection.

### GCMC/MD acquisition

```json
{"kind": "gcmc_md", "temperatures_k": [300], "cycles": 1, "mc": {"guest": "seeds/guest.extxyz", "pressures_bar": [1], "moves": 100, "eos": {"kind": "ideal"}}}
```

LAMMPS route, with explicit guest and reservoir settings. Omitting `md` gives
pure MC; supported interleaved MD has its own ensemble, recipe, steps and
cadence. Check [nested GCMC fields](#gcmc-nested-reference) for EOS, move weights,
overlap and guest limits. MD fragments preserve native groups, analysis,
auxiliary fixes and thermostat options while DICE carries the particle state
and velocities between segments.

## Sample outputs and screen

| Sampling route | Formal geometry | Coordinate used by screen |
| --- | --- | --- |
| LAMMPS MD / native protocol | Validated custom dump | Native MD step and task/frame ordinal |
| GPUMD | Paired `active.xyz` / `active.out` | MD step, native time and run phase |
| MD/MC | Validated retained chain dump | MD step and MC counters |
| ASE NEB / relax | `observations.extxyz` | Optimizer step and image index where applicable |
| GCMC / Widom | `observations.extxyz` with member evidence | MC/MD phase or insertion proposal |

Each mode publishes geometry together with the ordering, atom mapping and member evidence needed by screen. Additional native outputs remain available in the task directory. Result paths and whole-stage continuation are described in the [run walkthrough](../getting-started/quickstart.md) and [recovery](../guides/results-and-recovery.md).
