# Use DeepMD teachers and students

Use this guide with the [complete scientific walkthrough](own-system.md).
It covers the DPA4 teacher / DPA4C student route and explains the TF alternative.
Exact requirements depend on your checkpoint and engine revision; start with
[model sources and recorded builds](native-models.md).

## Prepare the native installation

Install DeePMD and a compatible LAMMPS build following the
[upstream installation instructions](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html).
For newer DPA4/DPA4C, use the recorded revision or independently qualify the
revision you choose. Python, exported models and LAMMPS must resolve compatible
DeepMD, PyTorch and CUDA libraries. A Python-only import does not establish
LAMMPS compatibility.

In that activated native installation inspect `dp --version`,
`python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'`
and `lmp -help`. Check native fine-tuning, readout, export and a small native
sampling task separately before a long campaign. These commands inspect an
installation; they do not certify numerical accuracy.

The [PT profile template](../examples/profiles/dpa4c_dpa4.json) binds the public
DICE roles. Copy it to your run directory as `environment.json`, prepare
`activate-native.sh` from the [activation template](../examples/profiles/dpa4c-dpa4.env.sh)
and edit the activation path. The [walkthrough](own-system.md) shows both files
in their scientific run context.

## Foundation fine-tuning and native recipes

Use `policy: "foundation_finetune"`, a native `foundation_model` checkpoint
and the JSON recipe supplied for that checkpoint. The public active `type_map`
contains your chemistry, while the learned model retains the foundation's full
native type map and exact order. DICE validates that relationship before native
training and binds its generated datasets and member seeds.

Each executed train block applies its declared policy. For multiple rounds,
using the same foundation asset means new fine-tuning jobs from that foundation;
it does not silently resume the preceding optimizer. Set training budgets,
losses and learning rates in the native recipe.

New recipes may use `model.preset`, for example `dpa4-nano-v20260911` or
`dpa4c-nano-v20260911`. DeePMD expands and validates presets. DICE uses the
family to select native operations and supplies descriptor/fitting member seeds
without reconstructing the preset's architecture.

## Model files and deployment

| Role | Native path and result |
| --- | --- |
| DPA4 teacher | `dp --pt` fine-tuning and ordinary freeze; `.pt2` labels structures, paired `.pt` checkpoint supplies readout |
| DPA4C student | `dp --pt-expt` fine-tuning/freeze; canonical uncompressed `.pt2` supplies student readout |
| DPA4C compressed sampling | `student.compress: true` derives a sampling representation; canonical training model stays available |
| TF student | Public `cli: "tf"`; compatible `se_e2_a` single-student route, `.pb` deployment and TF LAMMPS ABI |

Keep `cli: "pt"` in DICE configurations for both DPA4 and DPA4C; native
selector dispatch belongs to the adapters. DPA4 compression is unsupported.
DICE manages train, freeze and optional student compression commands; there
is no need to manually replace a trained artifact in the lifecycle tree.

A single DPA4C student uses `pair_style deepmd/kk` and
`lmp -k on g 1 -sf kk -in input.lammps`. A committee uses the ordered models
with ordinary `pair_style deepmd` and native deviation output. Kokkos failure
does not fall back to ordinary execution. Qualify the chosen route explicitly.

## Enable ZBL

In a compatible model recipe, this **model fragment** requests analytical ZBL:

```json
{
  "model": {
    "preset": "dpa4c-nano-v20260911",
    "bridging_method": "zbl"
  }
}
```

Use the corresponding DPA4 recipe for a teacher. Fine-tune after enabling ZBL;
when continuing from a ZBL checkpoint, preserve its configuration. DICE also
recognizes the canonical sum of one learned child and one `inner_potential`.
Native DeePMD owns window settings, filtering and parameter validity. Check
version-specific interactions with `training_data.min_pair_dist`.

Labels and deployment use the **complete** model. Do not subtract ZBL from
training labels or add it again in LAMMPS. DICE reads atomic features from the
complete model; learned-child inspection supplies metadata only. General
linear ensembles and legacy tabulated `use_srtab` readout are not enabled by
this analytical-composition support.

## Readout, precision and conditioning

Readout uses `eval_embedding`'s second return, `atomic_feature`, retaining
hidden components connected to the final energy projection. Older APIs fall
back to `eval_fitting_last_layer` when embedding is absent or unimplemented.
Atom order and native-to-active element mapping are preserved. Automatic
neighbour graphs avoid the former explicit dense request on newer DPA4.

Set `DP_INTERFACE_PREC=high` for the recorded newer build. This changes the
interface precision, not the stored weights' internal precision. Follow your
installation's validated activation and library setup.

OMol25 charge/spin-multiplicity conditioning is different from per-atom magnetic
spin dynamics. The recorded Kokkos route uses model-default conditioning and
does not support runtime overrides. Read the [compatibility and qualification
notes](deepmd-compatibility.md), including DPA4's unresolved short-distance
freeze/batch differences, before interpreting ZBL as collision protection.
