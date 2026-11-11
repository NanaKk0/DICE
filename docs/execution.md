# Place calculations on resources

An environment profile answers **which software runs an operation**. An
execution profile answers **where it runs and which resources it receives**.
Keep both separate from scientific `run.json` and native training/MD recipes.
Omitting execution placement preserves the ordinary local route.

```sh
dice run.json --environment environment.json --execution execution.json --preflight
dice run.json --environment environment.json --execution execution.json
```

`DICE_EXECUTION` is the fallback when the CLI option is absent.

## Profile objects

| Object | Responsibility |
| --- | --- |
| `targets` | Local, SSH or Slurm adapter and its site/working-root settings |
| `target_pools` | Explicit sets of declared targets |
| `resource_classes` | Nodes, processes, GPUs, CPUs, memory, wall time and launcher |
| `bindings` | Operation role to resource class and target pool |

The [local CPU example](../examples/profiles/execution.json) illustrates one
teacher training binding. It is not a complete GPU workflow allocation.
Use the installed profile loader and public preflight to validate the complete
roles selected by your actual route. Site-specific paths, partition/account
and connection setup belong here, not in scientific fields.

## Respect operation-specific capabilities

Resource shapes are constrained by the published implementation. A profile
can express sharing/pools without every backend supporting every combination.
In this release, the profile resource-shape validator permits distributed mode
for MACE teacher training; do not infer distributed DPA4C support from newer
unpublished development notes. Native qualification is a separate requirement.

Selected DeepMD committee-training roles can use local MPS sharing. Selected
inference roles allow multi-GPU frame pools with one worker per GPU. LAMMPS
sampling concurrency and DeepMD relaxation sharing have their own rules.
Do not copy a training resource class into every inference/sampling operation.
The exact owner is `workflow_runtime/execution/profile.py`; public preflight
checks additional role/placement constraints.

MPS sharing needs an allocated GPU and a workflow-owned session; increased
concurrency is not a VRAM quota or a guarantee of speedup. Begin with a
qualified serial route before opting into shared resources. Multi-GPU assignment
fixtures do not establish native scaling on physical multiple GPUs.

## Recover remote operations through DICE

The public controller still advances the iteration's normal block sequence.
Operational receipts under `.dice_state/executions/` support reattachment and
collection; scheduler success is not block completion. DICE validates formal
outputs before reuse. Run ordinary continuation or an inspected exact suffix
rebuild as described in [recovery](recovery.md), rather than invoking internal
workers or editing receipts. Keep credentials outside shared profiles.
