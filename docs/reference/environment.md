# Environment profile reference

An environment profile binds DICE operations to activated software and logical executables. Its schema is `dice.environment-profile`. Setup steps are in the [scientific walkthrough](../getting-started/quickstart.md); execution targets and resource placement are defined in the [execution reference](execution.md).

## Top-level fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | Required string | `dice.environment-profile` |
| `software_profiles` | Required nonempty object | Named software profiles |
| `bindings` | Required object | Canonical operation role → declared software-profile name |

A software-profile name can be shared by several operations. Each operation selected by the run must have a binding to a profile with the required executables and runtime fields.

## Software-profile fields

Paths below are within `software_profiles.<name>`.

| Field | Type / default | Meaning |
| --- | --- | --- |
| `environment.source_list` | String array; `[]` | Ordered shell activation scripts. Relative paths resolve from the environment profile's directory. Absolute paths are accepted. |
| `executables` | String-to-string object; `{}` | Logical program name → executable name or path. Program names resolve after the activation scripts are sourced. |
| `runtime` | Object; `{}` | Operation-specific execution tuning from the accepted keys below |

A minimal profile binding one training operation is:

```json
{
  "schema": "dice.environment-profile",
  "software_profiles": {
    "models": {
      "environment": {"source_list": ["activate-native.sh"]},
      "executables": {"dp": "dp", "python": "python"}
    }
  },
  "bindings": {"teacher.deepmd.pt.train": "models"}
}
```

The [water example](../../examples/dpa4_dpa4c/environment.json) supplies the complete profile for its teacher, student and sampler.

## Operation roles

| Operation family | Canonical roles |
| --- | --- |
| MACE teacher | `teacher.mace.train`, `teacher.mace.label`, `teacher.mace.gmm_reference`, `teacher.mace.gmm_score` |
| DeepMD-TF teacher | `teacher.deepmd.tf.train`, `teacher.deepmd.tf.label` |
| DeepMD-PT teacher | `teacher.deepmd.pt.train`, `teacher.deepmd.pt.label`, `teacher.deepmd.pt.gmm_reference`, `teacher.deepmd.pt.gmm_score` |
| DeepMD-TF student | `student.deepmd.tf.train`, `student.deepmd.tf.compress`, `student.deepmd.tf.gmm_reference`, `student.deepmd.tf.gmm_score` |
| DeepMD-PT student | `student.deepmd.pt.train`, `student.deepmd.pt.compress`, `student.deepmd.pt.gmm_screen` |
| NEP student | `student.nep.train`, `student.nep.gmm_reference`, `student.nep.gmm_score` |
| LAMMPS / GPUMD sampling | `sampler.lammps.tf`, `sampler.lammps.pt`, `sampler.gpumd` |
| NEB calculators | `student.deepmd.tf.neb`, `student.deepmd.pt.neb`, `student.nep.neb` |
| Relaxation calculators | `student.deepmd.tf.relax`, `student.deepmd.pt.relax`, `student.nep.relax` |
| FP labelling | `first_principles.cp2k.label`, `first_principles.abacus.label` |

Preflight reports the roles selected by the configured model shapes, sampling kinds and label settings in `operation_roles`. DPA4 and DPA4C use the existing PT role names. Backend guides describe the executable requirements for those roles.

## Runtime fields

| Key within `runtime` | Meaning and applicability |
| --- | --- |
| `batch_size` | Positive integer total-atom budget per same-topology DeepMD call. Required for PT teacher GMM reference/score and PT student `gmm_screen`. The example uses `9472`. |
| `descriptor_chunk_frames` | Positive frame count used to transport descriptor work; the default depends on the backend operation. |
| `mace_workers` | MACE inference workers; default `1` |
| `mace_batch_size` | MACE inference batch size; default `1` |
| `mace_threads` | MACE worker threads; default `1` |
| `fp_workers` | Positive integer concurrent local FP workers; required for built-in FP labelling |
| `cpu_threads` | Positive integer threads per local FP worker; required for built-in FP labelling |
| `tf_gpu_allocator` | TensorFlow GPU allocator setting passed to the native environment |
| `gmm_reference_spill` | Boolean, default `false`; enables temporary reference-data spill on applicable GMM paths |

For FP labelling, `fp_workers × cpu_threads` must fit the process's available CPU affinity. Resource classes, GPU placement and operation concurrency are defined in the execution profile. Backend environment variables and their activation belong in the source scripts.

## Profile selection and compatibility

The CLI selects this file through `--environment`; `DICE_ENVIRONMENT` is the fallback. Command syntax is listed in `dice --help`.

The loader also accepts historical revision-4/5 envelopes. Revision 4 can normalize paired PT student GMM reference/score bindings to `student.deepmd.pt.gmm_screen` when both use the same software profile. New profiles use the stable schema shown above.
