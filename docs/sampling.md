# Configure sampling

Sampling turns explicit seed geometries and conditions into observations for
ordinary screening. Begin with fixed NVT/NPT. Advanced modes require matching
native recipes or calculators and appropriate geometry; accepted configuration
alone does not demonstrate stability, convergence or physical accuracy.

## Seeds, conditions, rounds and quotas

Use `sample.seeds` to name geometry sources and `sample.condition_groups` to
name protocols. A condition applies to all declared roles unless it names a
`seeds` subset. Fixed temperature/pressure arrays expand independent tasks.
A seed uses source frame zero unless `max_frames` requests deterministic
sampling of distinct source frames without replacement.

`round_count` repeats the registry. Alternatively, ordered `rounds` activates
known seed/condition subsets; it does not reorder the global registry. Each
round needs a nonempty effective task set. Roles can share a `selection_group`
so extra conditions contribute to a common quota rather than multiplying it.
Tasks start from declared seeds, not implicitly from the previous MD endpoint.

## Condition fragments

Each JSON block below is **one condition-group value**, not a complete
`run.json`. Insert it under `sample.condition_groups.<name>` in the
[complete walkthrough](own-system.md). Numerical values illustrate syntax.
The recipes and seed/guest files must exist before native execution.

### Fixed NVT

```json
{"kind": "nvt", "recipe": "recipes/nvt.lammps", "temperatures_k": [300], "steps": 200}
```

### Fixed NPT

```json
{"kind": "npt", "recipe": "recipes/npt.lammps", "temperatures_k": [300, 600], "pressures_bar": [1], "steps": 200}
```

Both support matching LAMMPS/GPUMD routes. Temperatures are K and pressures
bar. The native recipe owns timestep, output cadence and thermostat/barostat.
NPT requires a sensible periodic cell and validated pressure coupling.

### Continuous temperature programs

```json
{"kind": "nvt_program", "recipe": "recipes/program.lammps", "segments": [{"temperature_k": 300, "steps": 100}, {"start_temperature_k": 300, "stop_temperature_k": 600, "steps": 100}]}
```

```json
{"kind": "npt_program", "recipe": "recipes/program-npt.lammps", "pressures_bar": [1], "segments": [{"start_temperature_k": 300, "stop_temperature_k": 600, "steps": 200}]}
```

LAMMPS only. Segments execute continuously within one task; the native
program-aware recipe must consume the managed schedule. Repeating independent
fixed-temperature tasks is not equivalent to carrying cell and velocities
through a program. There is no segment-level resume.

### NVT cell program

```json
{"kind": "nvt_cell_program", "recipe": "recipes/cell-program.lammps", "segments": [{"temperature_k": 300, "steps": 200, "cell": {"mode": "isotropic_density", "stop_density_g_cm3": 5.0}}]}
```

LAMMPS only, with the native recipe's explicit cell-program capability contract.
An endpoint can instead use a positive-definite symmetric strain; see
[exact cell fields](configuration.md). Density is g/cm³; strain is dimensionless.
DICE owns the supported segment run/deformation commands, so an ordinary NVT
input cannot simply be reused as a cell-program recipe.

### NEB

```json
{"kind": "neb", "images": 5, "fmax_ev_a": 0.05, "max_steps": 200}
```

DeepMD/LAMMPS or NEP/GPUMD calculator route, driven by ASE; no MD recipe.
Assign exactly one seed role whose source contains two ordered endpoints with
identical species/order, PBC and fixed cell. `images` counts movable interior
images. Default output is the converged band; optional history changes exposed
frames. Fixed atoms must coincide at both endpoints. DyNEB means dynamic NEB.

### Fixed-cell relaxation

```json
{"kind": "relax", "fmax_ev_a": 0.05, "max_steps": 200, "include_history": true, "history_stride": 10}
```

Uses the supported DeepMD/NEP calculator and ASE FIRE, without an MD recipe.
It optimizes atomic positions at fixed cell. The first student member drives
optimization; exposed structures use the ordered member evaluations. Review
convergence: unconverged acceptance requires an explicit opt-in.

### Fixed-composition MD/MC

```json
{"kind": "md_mc", "ensemble": "nvt", "recipe": "recipes/swap.lammps", "temperatures_k": [600], "steps": 200, "mc": {"move": "identity_swap", "elements": ["Al", "O"], "every_md_steps": 20, "attempts_per_event": 1}}
```

LAMMPS only. This syntax example swaps two existing element identities and
preserves composition; choose physically meaningful exchange species. NPT also
requires pressures. The first student drives MD and whole-system Metropolis
swaps. Retained chain states, not arbitrary rejected trial geometries, enter
observations. Qualify the native recipe, energy differences and intended ensemble.

### Independent insertion (Widom acquisition)

```json
{"kind": "widom", "guest": "seeds/guest.extxyz", "frames": 10, "max_attempts": 100, "random_seed": 0}
```

DeepMD/LAMMPS-selected Python calculator route; no MD recipe. The host seed and
guest library must have compatible species and geometry. `frames` is the
geometrically qualified target per host task; `max_attempts` bounds retries.
Overlap filtering and underfill are separate from screen selection. This
acquisition route is not itself a thermodynamic insertion estimator.

### GCMC/MD acquisition

```json
{"kind": "gcmc_md", "temperatures_k": [300], "cycles": 1, "mc": {"guest": "seeds/guest.extxyz", "pressures_bar": [1], "moves": 100, "eos": {"kind": "ideal"}}}
```

LAMMPS route, with explicit guest and reservoir settings. Omitting `md` gives
pure MC; supported interleaved MD has its own ensemble, recipe, steps and
cadence. Check [nested GCMC fields](configuration.md) for EOS, move weights,
overlap and guest limits. Collecting configurations does not establish an
adsorption isotherm or convergence.

## Observe and recover

Read task diagnostics and the formal sample outputs, then inspect screen
counts. Some native routes retain validated contiguous failed-task prefixes;
others fail atomically. Recovery targets the entire `rN/sample` block and its
suffix, not a segment, optimizer step or MC move. Follow [recovery](recovery.md)
when changing a completed plan. See [capabilities](support.md) for native scope.
