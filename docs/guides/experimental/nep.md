# Use a NEP student with GPUMD

The NEP/GPUMD integration is experimental.

NEP students pair with GPUMD sampling and a supported teacher. Install
[GPUMD/NEP](https://github.com/brucefan1983/GPUMD) and a compatible TorchNEP
Python environment for readout/calculator operations. Confirm that the native
version, model format and active element order match your recipe.

This is a **student-object fragment**, using scratch training as an example:

```json
{"backend": "NEP", "model": "single", "type_map": ["Al", "O"], "policy": "from_scratch", "recipe": "recipes/nep.in", "valid_frac": 0.1}
```

Change the full workflow's sampler to `{"backend": "GPUMD"}` and use GPUMD
NVT/NPT recipes in its condition groups. Do not retain the walkthrough's
LAMMPS recipe. Omit DeepMD-only `cli`, `compress` and `train_options`.
Single-model GMM readout requires compatible NEP4. Model-shape constraints
are listed in the [configuration reference](../../reference/run-json.md).

## Prepare inputs and initialization

Supply your native `nep.in` settings, geometry seeds and the teacher-labelled
student population managed by DICE. DICE stages the native member workspace;
NEP training must produce nonempty `nep.txt` and `nep.restart`.

For `foundation_finetune`, provide both `student.foundation_model` and
`student.foundation_restart`, a matching model/restart pair. The adapter stages
them and supplies native fine-tuning input. Every executed train block initializes from that declared model/restart pair with the current dataset.

## Bind operations

Bind `student.nep.train` to an environment exposing `nep` and `python`;
`student.nep.gmm_reference` / `student.nep.gmm_score` need compatible TorchNEP
readout; `sampler.gpumd` exposes `gpumd`. Advanced NEB/relax routes use their
specific `student.nep.neb` / `.relax` Python calculator roles.
Environment profiles use the [same shape](../../reference/environment.md) as other engines,
with these explicit role names and your activation scripts.

Run preflight with the configured profile, then start the DICE iteration.
Sampling fields and outputs are in [sampling](../../reference/sampling.md); continuation and
rebuilding are in [recovery](../results-and-recovery.md).
