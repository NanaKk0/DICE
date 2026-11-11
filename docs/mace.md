# Use a MACE or MACE-LES teacher

A MACE teacher can label data for a compatible DeepMD or NEP student. Start
with the [workflow](workflow.md) and [complete configuration](own-system.md),
then replace the teacher and corresponding environment bindings.

## Prepare inputs and software

Install a compatible [MACE](https://github.com/ACEsuit/mace) / MACE-LES build
in a separate native environment. Provide its training executable
`mace_run_train`, Python inference/readout dependencies, a native YAML recipe
and a compatible foundation `.model` if fine-tuning. Check the model's species,
heads, energy reference and fitting/readout compatibility; support for a model
family does not certify arbitrary checkpoints.

This is a **teacher-object fragment** for the Al/O walkthrough:

```json
{"backend": "MACE", "model": "single", "type_map": ["Al", "O"], "policy": "foundation_finetune", "foundation_model": "models/teacher.model", "recipe": "recipes/teacher.yaml", "valid_frac": 0.1}
```

Omit DeepMD's `cli` and `train_options`. For `from_scratch`, omit the foundation
asset. A committee instead requires `members >= 2` and teacher force-deviation
`lo`/`hi`; consult [configuration](configuration.md).

## Prepare the recipe and bindings

DICE supplies member names, model directory, train/validation files, validation
fraction and member seeds. Do not declare `foundation_model` inside the native
YAML: it conflicts with the workflow-owned initialization policy. Architecture,
loss, optimization budgets and appropriate model options remain native settings.

Bind `teacher.mace.train`, `teacher.mace.label`, and for single-teacher readout
`teacher.mace.gmm_reference` / `teacher.mace.gmm_score` to the activated MACE
profile. Its logical executables are `mace_run_train` and `python`.
The [MACE/TF profile](../examples/profiles/environment.json) and its adjacent
activation scripts show a different route from the DPA4/DPA4C PT template;
copy the matching set and edit activation paths rather than mixing them.

## Check before a campaign

Confirm native training and ASE energy/force/stress labelling on a small
representative input. Single-model selection additionally needs compatible
scalar-energy readout and a fitted reference from the actual teacher train
membership. Committee uncertainty uses ordered independent members instead.

Run public preflight with your real profile, then the ordinary iteration.
Inspect the realized dataset split and native logs. The public CPU example
uses a MACE-shaped double; its success does not validate a real MACE build or
LES physics. See [capabilities](support.md), [data](data.md) and [recovery](recovery.md).
