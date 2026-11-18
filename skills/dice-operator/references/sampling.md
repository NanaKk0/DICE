# Sampling

Use the matching sampling guide/reference and official configuration loader.
Keep the user's native protocol and explicit batch controls separate.

## Native protocols and overrides

- A condition group can provide only a `recipe`; omitted `kind` resolves to the
  native route. Standard LAMMPS or GPUMD syntax expresses the sampling protocol.
- Explicit DICE fields override the corresponding selected native settings.
  Omitted fields inherit. Structural inspection may report unknown native
  temperatures/steps; public preflight resolves the selected protocols.
- Shared controls include `temperatures_k`, `pressures_bar`, `steps`,
  `timestep_ps` and `trajectory_every_steps` where the selected mode supports
  them. Use `targets` to select the relevant fix/run when an override is
  ambiguous. Preserve native groups, coupling/damping and extra outputs.
- A thermostat temperature override leaves native velocity initialization and
  RNG settings intact. Keep those choices in the recipe unless the public
  interface explicitly owns them.
- DICE assembles seed loading, the current student model and standard
  observations. Write clean native inputs; DICE-specific variables, includes
  and capability markers are unnecessary. Declare auxiliary files through
  `inputs` using the supported task-relative names and run-relative sources.

## Task expansion

Seed roles own `selection_group`. Conditions, temperature/pressure arrays and
optional seed `max_frames` expand the task population. Inspect the normalized
plan to establish the effective tasks rather than multiplying guessed counts.
Omitted `max_frames` uses the source's first frame; an explicit positive count
selects distinct available frames within the supported deterministic plan.

Use `sample.rounds` for per-round seed/condition activation or `round_count` for
the repeated plan; the two forms are mutually exclusive. Omitted activation
axes use the declared registry. Selection and FP quotas stay with selection
groups, including groups containing several expanded tasks.

## Mode-specific lookup

| Mode | Interface to consult |
| --- | --- |
| Native MD and minimization | Native protocol plus supported explicit overrides; LAMMPS/GPUMD guide |
| `nvt`, `npt` | MD conditions and native thermostat/barostat protocol |
| `nvt_program`, `npt_program` | Ordered temperature segments; LAMMPS-only NPT program |
| `nvt_cell_program` | LAMMPS temperature/cell segments and cell endpoint constraints |
| `neb` | ASE endpoint bundle, image count, force tolerance, step/history controls |
| `relax` | ASE fixed-cell relaxation, convergence/history and fixed-atom controls |
| `md_mc` | LAMMPS fixed-composition identity swaps, MD/MC budgets and NVT/NPT controls |
| `widom` | Host/guest inputs, candidate target, bounded attempts and overlap controls |
| `gcmc_md` | LAMMPS/FLAMES guest insertion, reservoir conditions, cycles and optional MD segments |

Check the documented backend/role support for the selected mode. Keep MC
attempts, MD steps/time and optimizer steps distinct. A native protocol can
express algorithms beyond DICE's named modes when its existing execution and
observation adapter supports them; custom result types need an explicit adapter.

## Sample to screen

Use DICE's published sample catalog and readers for downstream observations.
LAMMPS/GPUMD trajectories and ASE optimization/path histories have different
coordinates; the producer adapter preserves their identity and eligible frames.
Read indexed observations as needed rather than eagerly scanning every frame
for progress or repeating validation already owned by DICE.

Some modes can retain validated failed-task prefixes while peers continue;
others require atomic success. Keep failure markers and model evidence intact,
and use the owning result validator to decide eligibility. Cancellation is
distinct from task failure. Recovery uses the public `rN/sample` block and its
suffix, rather than inventing task/segment/optimizer-step resume coordinates.
