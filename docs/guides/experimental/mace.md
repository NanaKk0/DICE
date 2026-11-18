# Use a MACE or MACE-LES teacher

The MACE/MACE-LES integration is experimental.

A MACE teacher can label data for a compatible DeepMD or NEP student. Start
with the [workflow](../../concepts/workflow.md) and [complete configuration](../../getting-started/quickstart.md),
then replace the teacher and corresponding environment bindings.

## Prepare inputs and software

Install a compatible [MACE](https://github.com/ACEsuit/mace) / MACE-LES build
in a separate native environment. Provide its training executable
`mace_run_train`, Python inference/readout dependencies, a native YAML recipe
and a compatible foundation `.model` if fine-tuning. Choose a model with the target species, head and energy reference, and with a compatible scalar-energy readout for single-teacher screening.

This is a **teacher-object fragment** for the Al/O walkthrough:

```json
{"backend": "MACE", "model": "single", "type_map": ["Al", "O"], "policy": "foundation_finetune", "foundation_model": "models/teacher.model", "recipe": "recipes/teacher.yaml", "valid_frac": 0.1}
```

Omit DeepMD's `cli` and `train_options`. For `from_scratch`, omit the foundation
asset. A committee instead requires `members >= 2` and teacher force-deviation
`lo`/`hi`; consult [configuration](../../reference/run-json.md).

## Prepare the recipe and bindings

DICE supplies member names, model directory, train/validation files, validation
fraction and member seeds. Declare the foundation through DICE's
`teacher.foundation_model`. Keep architecture, loss, optimization budgets and
model options in the native YAML recipe.

Bind `teacher.mace.train`, `teacher.mace.label`, and for single-teacher readout
`teacher.mace.gmm_reference` / `teacher.mace.gmm_score` to the activated MACE
profile. Its logical executables are `mace_run_train` and `python`.
The [MACE/TF profile](../../../examples/profiles/environment.json) and its adjacent
activation scripts show a different route from the DPA4/DPA4C PT template;
copy the matching set and edit activation paths rather than mixing them.

## Run with DICE

Run public preflight with your profile, then start the iteration. Inspect the
dataset split and native logs. Label requirements are in the
[data reference](../../reference/data.md); continuation and rebuilding are in
[recovery](../results-and-recovery.md).
