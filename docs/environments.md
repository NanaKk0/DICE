# Environment and execution profiles

Use explicit profiles for real runs:

```sh
dice run.json --environment environment.json --preflight
dice run.json --environment environment.json
dice run.json --environment environment.json --execution execution.json
```

`DICE_ENVIRONMENT` and `DICE_EXECUTION` are fallbacks when their CLI option is
absent. Normal native execution needs an environment; execution placement is
optional. The CPU mock uses injected synthetic services for its lifecycle demonstration.

## Environment shape

`examples/profiles/environment.json` illustrates a MACE/DeepMD-TF/LAMMPS
single-model route. It binds operation roles to three independently activated
software profiles and resolves executables from those activated environments.
Edit the adjacent activation scripts to select your own installations. Each script is sourced
only for its operations. LAMMPS serial execution explicitly sets
`DICE_LAMMPS_PARALLEL_TASKS=1`.

Use only the selected canonical roles. Single-teacher operations include
`teacher.mace.train`, `.label`, `.gmm_reference`, `.gmm_score`; the illustrated
student needs `student.deepmd.tf.train`, `.gmm_reference`, `.gmm_score` and
`sampler.lammps.tf`. DeepMD-PT, NEP, compression, NEB, relaxation and optional
FP labeling select their own exact roles. The canonical list lives in
`workflow_runtime/lifecycle/environment.py`; use the roles selected by your workflow.

## Execution placement

For targets, launchers, resource restrictions and remote recovery, read
[execution](execution.md).

`examples/profiles/execution.json` illustrates one local CPU resource class
for the teacher training role. Set scientific resources according to your
model and workload. Real calculations must bind
appropriate resources for every selected operation. The schema supports
local, SSH and Slurm adapters with explicit staging/resource constraints.

Keep targets, resource classes, target pools and launcher shape here. Define
model committee size in the scientific configuration and launcher world size
in the execution profile. Multi-GPU frame pools, distributed training and MPS
use operation-specific combinations; check performance on the selected
hardware. GPU sharing requires an authorized exclusive allocation and a
workflow-owned MPS session. See the [capability matrix](support.md).

Keep passwords, private keys, access tokens and scheduler credentials outside
the repository. Third-party engine and foundation-model licenses
remain independently applicable.

For DPA4/DPA4C, obtain the models and runtime from their respective
projects and qualify your selected model/build combination.
The [walkthrough](own-system.md) supplies a complete PT role profile;
[native-models.md](native-models.md) gives the recorded stack, model-source
links and the mapping from DICE's `cli: "pt"` declaration to the native
`--pt` and `--pt-expt` selectors. For the qualified newer build, set
`DP_INTERFACE_PREC=high` in the native activation script. This selects interface
precision and does not promote the model's internal weights to double precision.
See the [compatibility notes](deepmd-compatibility.md) for the tested scope.

## Use the PT walkthrough profile

The MACE/TF files above are one example, not the activation for every backend.
For the DPA4/DPA4C walkthrough, copy
[examples/profiles/dpa4c_dpa4.json](../examples/profiles/dpa4c_dpa4.json)
into your run directory as `environment.json`. Copy
[the activation template](../examples/profiles/dpa4c-dpa4.env.sh) there as
`activate-native.sh` and replace its installation path.

The profile binds teacher train/label/reference/score, student train/compress/
fused screen and PT LAMMPS sampling. Omit execution placement for the default
local route. Each operation resolves its configured executables after sourcing
its profile's scripts; the control environment need not import the native
engines. A working `dice --help` therefore does not prove a native profile works.

Keep machine paths in activation/execution profiles, scientific choices in
`run.json`, and training/MD parameters in native recipes. Do not add generic
engine commands or CUDA settings as scientific configuration keys.
