# Configuration

Use the loaders for the DICE version being operated. Locate exact fields through
its documentation index; use the public configuration, environment, execution
and data references for detailed constraints.

## Four responsibilities

| Input | Owns |
| --- | --- |
| `run.json` | Datasets, teacher/student models, relative recipe paths, sampling conditions, screening and FP selection |
| Environment profile | Software activation, executables, operation-role bindings and supported runtime controls |
| Execution profile | Targets, pools, resources, concurrency and launchers |
| Native recipe | Model architecture and backend scientific settings; native sampling protocols and algorithms |

The run configuration requires `iter`, `teacher_train_bootstrap`,
`student_train_bootstrap`, `fp_data`, `teacher`, `student`, `sampler`, `sample`,
`screen` and `fp`. Optional top-level `visualize: false` disables diagnostic
selection plots. Obtain defaults and legal combinations from the official
loader rather than adding a schema version, arbitrary commands or resource
fields to the scientific configuration.

Paths declared in `run.json` are relative to that file. Use explicit relative
files; absolute paths, `..`, globs and directory discovery are rejected.
Profiles have their own path rules and can bind machine-specific installations.
Keep machine settings in the selected profiles.

## Create or adapt

Read only the user's named references and their relevant inputs. Inherit explicit
user values first, then the named reference and official defaults. Describe any
remaining scientific or execution proposal before running. Preserve unmodified
parts when changing an existing configuration. Useful review items are:

- Data sources, element maps, model shape, training policy, optional foundation
  models and training budgets.
- Sampling recipes, active seeds/conditions per round and selection groups.
- Prescreening, reliability bands, selection policies and group quotas.
- Environment/operation bindings, execution target and resource budget.

Pretrained models are optional. For DeePMD, the documented recommended route is
a DPA4 teacher and DPA4C student. MACE, NEP/GPUMD and built-in CP2K/ABACUS
integrations are experimental; consult their matching guides when requested.

## DeePMD recipes

Keep architecture and native `model.preset` in the DeePMD recipe. Public DICE
`cli: "pt"` covers DPA4 and DPA4C; DICE resolves the native family and selects
the matching backend CLI. Preserve foundation element numbering and all required
conditioning fields. A smaller active element set is distinct from the
foundation's native type map.

DPA4C may use the student compression path and matching LAMMPS/Kokkos environment.
DPA4 compression is unsupported. Verify the selected environment's required
operations with public preflight; use matching native model formats and runtime.
Native ZBL configuration remains in the DeePMD recipe and the complete model
is used by DICE. Keep reference labels intact and avoid adding the short-range
term again in sampling inputs. Charge/spin model conditioning and per-atom
magnetic-spin dynamics are separate backend capabilities.

## Continue with DFT labels

Create a separate configuration for the next iter in the same run directory.
Advance `iter`, supply new labelled data through `fp_data`, and use the ordinary
next-iter student bootstrap behavior (the tutorial sets
`student_train_bootstrap: []`). Preserve input identity and labels according to
the data reference. DICE accepts externally supplied labelled structures;
the quickstart includes the complete handoff example.

`student.include_teacher_train_data` controls teacher-training-set inclusion
in student training. Preserve the user's choice and explain it when changing
the data route. Use the label-validation command from the matching data guide
before launching the next iter.
