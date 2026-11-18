# Configuration overview

After the [Quickstart](../getting-started/quickstart.md), adapt the supplied [run.json](../../examples/dpa4_dpa4c/run.json) for your task. This page maps responsibilities and links the detailed guides.

## Where settings belong

| File | Settings | Details |
| --- | --- | --- |
| `run.json` | Data, teacher/student models, sampling, screening and DFT selection | Sections below |
| Native recipe | Architecture, training parameters, sampling algorithms and backend settings | [Prepare input files](inputs.md), [Configure sampling](sampling.md) |
| `environment.json` and activation scripts | Software, executables and operation bindings | [Configure backend environments](backends.md) |
| `execution.json` | Placement, resources and concurrency | [Configure compute resources](resources.md) |

DICE binds prepared training data and assembles sampling structures, student models and standard observations. Explicit supported sampling fields override the corresponding native settings; omitted fields inherit them.

## Sections of run.json

| Section | Purpose |
| --- | --- |
| `iter`, `teacher_train_bootstrap`, `student_train_bootstrap`, `fp_data` | Iteration number and data inputs |
| `teacher`, `student` | Backend, active elements, native recipes and initialization policy |
| `sampler`, `sample` | Sampling engine, seeds, conditions, rounds and selection groups |
| `screen`, `fp` | Student-training additions, DFT candidates and final selection |

Training steps, learning rates and losses belong in native training recipes. `teacher.policy` and `student.policy` select initialization behavior.

## Continue by task

| Task | Read |
| --- | --- |
| Data, starting structures and layout | [Prepare input files](inputs.md) |
| Model fields and fine-tuning | [Model preparation](inputs.md#4-prepare-models-for-the-training-policy), [DeePMD guide](deepmd.md), [Model fields](../reference/run-json.md#teacher-and-student-fields) |
| Native sampling, batch conditions, seeds and rounds | [Configure sampling](sampling.md) |
| Screening scores, data routes and representative selection | [Configuration screening](../concepts/screening.md) |
| Thresholds, pools and quotas | [Screening reference](../reference/run-json.md#screening-fields-and-terminal-selection) |
| DFT labels and the next iter | [Quickstart: prepare iter1 labels](../getting-started/quickstart.md#5-prepare-new-labels-for-iter1) |
| Apply settings to completed stages | [Rebuild a selected stage](results-and-recovery.md#5-rebuild-a-selected-stage) |

See the [run.json reference](../reference/run-json.md) for types, defaults, units and applicability.
