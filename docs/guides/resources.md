# Configure compute resources

Use `execution.json` to choose placement and resources for training and sampling. This guide starts with the local single-GPU [Quickstart](../getting-started/quickstart.md).

## 1. Locate the configuration

Enter the example:

```sh
cd examples/dpa4_dpa4c
```

Edit [execution.json](../../examples/dpa4_dpa4c/execution.json). It owns targets, resource classes and operation bindings. Configure activation scripts and executables as described in [Configure backend environments](backends.md).

## 2. Understand the execution plan

| Section | Example setting |
| --- | --- |
| `targets` | A local target named `local`, with `adapter: "local"` |
| `target_pools` | A pool named `local` containing that target |
| `resource_classes` | A class named `gpu`, using one GPU |
| `bindings` | Teacher training, student training and LAMMPS sampling use that class and pool |

`local` and `gpu` are configurable names; update references when renaming them.

| Operation role | Operation | Resource class | Pool |
| --- | --- | --- | --- |
| `teacher.deepmd.pt.train` | Teacher training | `gpu` | `local` |
| `student.deepmd.pt.train` | Student training | `gpu` | `local` |
| `sampler.lammps.pt` | LAMMPS sampling | `gpu` | `local` |

These operations share the resource settings. Other operations execute locally through their environment bindings.

## 3. Choose devices and budgets

### Select GPUs

Set visible devices in the terminal starting DICE, for example:

```sh
export CUDA_VISIBLE_DEVICES=0
```

The activation script preserves this value. Retain scheduler-provided visibility inside a scheduled allocation.

`gpus_per_node` specifies the allocated count; device visibility specifies which GPUs are available. Keep `gpus_per_node: 1` for one GPU.

### Adjust the resource class

The example's complete `resource_classes.gpu` object is:

```json
{
  "mode": "pool",
  "per_gpu": 1,
  "nodes": 1,
  "processes_per_node": 1,
  "gpus_per_node": 1,
  "cpus_per_process": 4,
  "memory_mb": 16384,
  "wall_time_seconds": 3600,
  "launcher": {"kind": "direct"}
}
```

Check:

| Field | Example | Meaning |
| --- | --- | --- |
| `gpus_per_node` | `1` | One allocated GPU |
| `cpus_per_process` | `4` | CPU budget per process |
| `memory_mb` | `16384` | Host-memory budget in MB |
| `wall_time_seconds` | `3600` | Time budget in seconds |

For local execution these fields describe resource budgets; backend environment settings control thread counts. Slurm uses resource settings in job allocation. GPU-memory demand depends on the model, batch size and concurrency.

For example, change CPU and host-memory budgets to `8` and `32768` as needed. Set local DeePMD threads in the activation script:

```sh
export OMP_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4
export DP_INTER_OP_PARALLELISM_THREADS=1
```

These thread settings match the example's four-CPU budget. Adjust them to your allocation and backend.

Keep `nodes: 1`, `processes_per_node: 1`, `mode: "pool"` and a `direct` launcher for this example. `per_gpu: 1` admits one independent task at a time on each GPU.

## 4. Assign resources by operation

Define separate `training` and `sampling` resource classes when their budgets differ, then change the bindings:

| Operation | `resource_class` | `target_pool` |
| --- | --- | --- |
| `teacher.deepmd.pt.train` | `training` | `local` |
| `student.deepmd.pt.train` | `training` | `local` |
| `sampler.lammps.pt` | `sampling` | `local` |

Provide a complete resource object for each class. For example, assign eight CPUs and 32768 MB to training, four CPUs and 16384 MB to sampling, retaining one GPU and `per_gpu: 1` for both.

Different activation scripts or threading settings belong in the model and sampling profiles in `environment.json`. The shared operation-role names connect the two profiles.

## 5. Check concurrency

DICE follows the teacher and student-stage order. Within an operation, a pool can run independent tasks such as committee members or sampling conditions.

With one GPU and `per_gpu: 1`, sampling tasks use the available slot in sequence. Seeds and conditions determine task count.

| Goal | Configuration |
| --- | --- |
| Independent tasks on several GPUs | Allocate multiple GPUs with pool mode, `per_gpu: 1` and matching visibility |
| Several tasks on one GPU | Increase `per_gpu` for an operation supporting sharing; DICE manages MPS for supported DeepMD committee-training and LAMMPS paths |
| More sampling tasks | Add starting structures, conditions or temperature/pressure combinations in `run.json` |

Resource classes define pool capacity; effective concurrency also depends on pending work and backend capability. Keep `targets.local.max_in_flight: 1` in the example. Sampling GPU slots are determined by that operation's `gpus_per_node` and `per_gpu`.

## 6. Preflight and run

Save the configuration and run:

```sh
dice run.json --environment environment.json --execution execution.json --preflight
```

Check the selected `execution`, required `operation_roles`, and software paths in `operation_environment`. Preflight validates resource shape, bindings and supported placement.

Then start:

```sh
dice run.json --environment environment.json --execution execution.json
```

Inspect progress in `iter0/status.log` and screening in `iter0/dice.log`. Use the same profile arguments for subsequent iters. See [Inspect results and resume](results-and-recovery.md).

For SSH or Slurm, declare the corresponding target and update pools/bindings. Environment profiles prepare software; execution profiles describe connection and scheduling. See [Execution reference](../reference/execution.md).
