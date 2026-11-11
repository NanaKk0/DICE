# Worked next-iteration FP handoff (CPU-only)

From the distribution root after installation:

```sh
python examples/iteration_handoff/check.py /tmp/dice-handoff-demo
```

Choose a new or empty directory. This runs iter 0 and iter 1 through the real
application with the same synthetic CPU backend doubles as
[cpu_mock](../cpu_mock/README.md).
All recipes, model handles, labels and trajectories are fabricated test data.
Use a separate demonstration directory.

The example retains `run_iter0.input.json` and `run_iter1.input.json` as readable
snapshots, but both iterations execute the same `run.json` in one run root.
The prior iter's complete canonical student history remains available.
After iter 0 it reads the actual terminal `iter0/fp/selected.xyz`, shifts each
selected structure by 0.007 Å in x to mimic a changed returned geometry, and
attaches explicitly **synthetic** -1 eV total energy and zero forces in
`data/fp_returned_iter0.extxyz` to test ingestion of a changed computed geometry.

The only scientific changes for iter 1 are:

```json
{
  "iter": 1,
  "student_train_bootstrap": [],
  "fp_data": ["data/fp_returned_iter0.extxyz"]
}
```

This patch lists the changed fields. The script keeps all other
fields from its complete iter-0 fixture, explicitly disables diagnostics for
this bounded check, reloads the real validator and checks both train recipes
and the sampling recipe. It then checks:

- Successful canonical terminal publication for both iterations.
- The returned, changed geometry is present in iter-1 teacher train data with
  its synthetic external labels.
- The prior final-round train dataset and screen train form the iter-1 student
  carry-over population; with `include_teacher_train_data` omitted/false, the
  student bootstrap contains the prior student population. Enable teacher-data
  inclusion explicitly when it is part of the intended training design.
- Required energy/force labels are valid.

The bundled `tests/test_documented_contracts.py` also checks that empty
later-iter FP declarations are rejected by the real loader and that the
example refuses to overwrite an existing nonempty run directory.

For a real handoff, **replace the synthetic return step** with validated output
from your chosen FP calculator. Preserve geometry/force correspondence and
energy/stress conventions; retain successful-frame provenance separately.
An empty terminal selection is a valid outcome. This example uses a nonempty
selection to exercise later-iteration ingestion. For a scientific next iteration,
declare nonempty, successfully computed FP labels in `fp_data`.
See [your own system](../../docs/own-system.md).

At iter 2+, list all returned FP files that must remain in teacher training
explicitly (or include their labelled union in an explicitly declared input).
To include current teacher train geometries in student relabelling, enable
`student.include_teacher_train_data` explicitly.
