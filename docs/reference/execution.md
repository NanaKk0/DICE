# Execution profile reference

An optional `dice.execution-profile` maps operations to execution targets and resources. The CLI selects it through `--execution`, with `DICE_EXECUTION` as fallback. Omitting both uses the default local route.

## Top-level fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | Required string | `dice.execution-profile` |
| `targets` | Object | Named execution targets |
| `target_pools` | Object | Named ordered target pools |
| `resource_classes` | Object | Named resource requests |
| `bindings` | Required nonempty array | Operation/resource/pool assignments |

Object names start with an alphanumeric character and use alphanumerics, `_`, `-` or `.`. A complete local GPU profile is supplied with the [water example](../../examples/dpa4_dpa4c/execution.json).

## Target fields

| Field within `targets.<name>` | Type / applicability | Meaning |
| --- | --- | --- |
| `adapter` | Required: `local`, `ssh`, `slurm` | Execution adapter |
| `max_in_flight` | Required positive integer | Maximum in-flight execution units for this target |
| `host_alias` | SSH only, required | Host alias from the user's SSH configuration |
| `remote_root` | SSH only, required | Safe relative POSIX staging root, without parent traversal |
| `working_root` | Slurm only, required | Absolute shared working root, excluding `/` and parent traversal |
| `stage_mode` | Required for remote adapters | `copy` for SSH; `shared` for Slurm |
| `partition`, `account`, `qos` | Slurm only, optional strings | Scheduler placement fields |
| `resource_policy` | Slurm only, optional object | Scheduler CPU/memory policy below |

Local targets use `adapter` and `max_in_flight`. SSH and Slurm use their respective connection/staging fields. Keep connection credentials in SSH or scheduler configuration.

## Target pools and bindings

| Field | Type | Meaning |
| --- | --- | --- |
| `target_pools.<name>.targets` | Nonempty unique string array | Existing target IDs in pool order |
| `bindings[].operation_role` | String | Canonical role from the [environment reference](environment.md#operation-roles) |
| `bindings[].resource_class` | String | Existing resource-class ID |
| `bindings[].target_pool` | String | Existing target-pool ID |

Each operation/resource-class pair has one binding. Software activation and executable selection remain in the environment profile.

## Resource-class fields

All fields below are within `resource_classes.<name>`.

| Field | Type / default | Meaning |
| --- | --- | --- |
| `mode` | `pool` (default), `distributed` | Independent task pool or distributed training |
| `per_gpu` | Positive integer; `1` in pool mode | Concurrent independent task slots per GPU; omitted in distributed mode |
| `nodes` | Required positive integer | Node count |
| `processes_per_node` | Required positive integer | Controller processes per node |
| `gpus_per_node` | Required nonnegative integer | GPU count; `0` selects CPU resources |
| `cpus_per_process` | Required positive integer | CPU budget per process |
| `memory_mb` | Required positive integer | Memory budget in MB |
| `wall_time_seconds` | Required positive integer | Time budget in seconds |
| `launcher.kind` | Required: `direct`, `torchrun` | Launcher |
| `launcher.rdzv_backend` | Optional: `c10d` | Rendezvous backend for `torchrun` |

Pool mode requires one node, one process per node and `direct`. CPU pools use `per_gpu: 1`. Current stable distributed mode requires one node, one controller process, at least two GPUs and `torchrun`; its accepted role is `teacher.mace.train`.

## Operation-specific resource shapes

| Operation | Accepted shape |
| --- | --- |
| DeepMD teacher/student training | Pool mode; sharing with `per_gpu > 1` requires exactly one explicit local target and uses managed MPS |
| Frame-pool inference roles | Pool mode with `per_gpu: 1`; selected roles accept multiple GPUs |
| LAMMPS sampling | Pool mode; concurrency follows GPU count and `per_gpu` |
| DeepMD relaxation sharing | One GPU, explicit local targets, and at least one CPU per worker |
| MACE distributed training | Stable distributed shape above |

The multi-GPU frame-pool roles are `teacher.mace.label`, `teacher.mace.gmm_score`, `teacher.deepmd.tf.label`, `teacher.deepmd.pt.label`, `teacher.deepmd.pt.gmm_score`, `student.deepmd.tf.gmm_score`, `student.deepmd.pt.gmm_screen` and `student.nep.gmm_score`.

GPU slots must match the visible device allocation. Backend-specific resource constraints are checked during profile loading and suffix preflight. Resource budgets describe placement and concurrency; backend thread settings belong to the activated environment.

## Slurm resource policy

| Field within `resource_policy` | Default / constraint |
| --- | --- |
| `cpu`, `memory` | `explicit` or `scheduler-gpu-default`; default `explicit` |
| `expected_cpus_per_gpu` | Required positive integer with CPU `scheduler-gpu-default` |
| `expected_memory_mb_per_gpu` | Required positive integer with memory `scheduler-gpu-default` |

Scheduler GPU defaults require positive GPU allocation. Declared CPUs per node must match `expected_cpus_per_gpu × gpus_per_node`; declared memory must match `expected_memory_mb_per_gpu × gpus_per_node` for the corresponding policy.

## Historical profiles and recovery

The loader accepts the historical `schema_revision: 1` envelope with its original launcher restrictions. New profiles use the stable schema without that field.

Execution placement keeps the public iter/round stages. Ordinary continuation and whole-stage rebuilding use the [recovery commands](../guides/results-and-recovery.md). Operational records under `.dice_state/executions/` support reattachment and collection; result validation determines stage completion.
