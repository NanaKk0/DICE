# CLI reference

`dice` reads one run configuration and advances its iter. Run commands beside the selected configuration; inputs, recipes and outputs are organized relative to that directory. See [Installation](../getting-started/installation.md) and [Quickstart](../getting-started/quickstart.md).

## Invocation

```text
dice run.json [--environment PATH] [--execution PATH]
dice run.json [--environment PATH] [--execution PATH] --preflight
dice run.json [--environment PATH] [--execution PATH] --probe BLOCK
dice run.json [--environment PATH] [--execution PATH] --resume BLOCK [--preflight]
dice run.json [--environment PATH] [--execution PATH] --resume fp --truncate-after-round N
```

`PATH` is a profile path, `BLOCK` a stage coordinate and `N` a zero-based round index. Brackets indicate optional arguments.

## Arguments

| Argument | Purpose |
| --- | --- |
| `run.json` | Required scientific configuration, for example `run.json` or `run_iter1.json` |
| `--environment PATH` | Activation scripts, executables and operation bindings |
| `--execution PATH` | Placement and compute resources |
| `--preflight` | Validate the selected suffix's configuration, inputs, environment, resources and sampling protocols |
| `--probe BLOCK` | Read-only exact-rebuild feasibility query, including suffix checks when dependencies are usable |
| `--resume BLOCK` | Rebuild this stage and the remaining current-iter suffix |
| `--truncate-after-round N` | With executing `--resume fp`, archive rounds beyond N and rebuild terminal selection |
| `-h`, `--help` | Display help |

Each profile is selected by its CLI option, then its environment-variable fallback: `DICE_ENVIRONMENT` or `DICE_EXECUTION`. Supply the environment required by the backend operations. Omitting execution uses default local execution.

See [Backend environments](../guides/backends.md) and [Compute resources](../guides/resources.md).

## Public stages

Actual stages depend on model shape and sampling rounds:

| Coordinate | Work |
| --- | --- |
| `teach/train` | Teacher training |
| `teach/screen` | Initial student labeling/data preparation |
| `teach/scan` | Single-teacher reference preparation |
| `rN/train` | Round-N student training |
| `rN/sample` | Round-N sampling |
| `rN/screen` | Evaluation, training additions and DFT candidates |
| `fp` | Final DFT selection |
| `fp/label` | Reference labeling when `fp.label` is configured |

The two-round example uses `r0` and `r1`. See [Workflow](../concepts/workflow.md).

## Run and continue

Preflight, then run:

```sh
dice run.json --environment environment.json --execution execution.json --preflight
dice run.json --environment environment.json --execution execution.json
```

Successful preflight emits `dice.lifecycle.preflight` JSON with stages, operation environments and, when applicable, `sampling_controls`.

Ordinary invocation reuses usable published results and continues remaining work. A usable terminal result is reused directly. See [DFT-label return](../getting-started/quickstart.md#5-prepare-new-labels-for-iter1) for the next iter.

## Query an exact rebuild boundary

For example:

```sh
dice run.json --environment environment.json --execution execution.json --probe r1/sample
```

Output schema is `dice.lifecycle.resume-feasibility`:

| Field | Meaning |
| --- | --- |
| `status: "runnable"` | Required inputs and suffix checks pass |
| `status: "blocked"` | A required upstream result is missing or unusable |
| `target`, `reason` | Requested coordinate and explanation |
| `blocking_result`, `consuming_block` | Blocking result and its consumer, when blocked |
| `recommended_resume` | Suggested producer to repair, when blocked |

Probe queries rebuild feasibility. Use [logs and outputs](../guides/results-and-recovery.md) to inspect progress and completion.

## Rebuild a selected stage

Choose the intended boundary to apply new settings to completed work. Preflight:

```sh
dice run.json --environment environment.json --execution execution.json --resume r1/sample --preflight
```

Then execute:

```sh
dice run.json --environment environment.json --execution execution.json --resume r1/sample
```

This preserves required usable prefix results and rebuilds second-round sampling, screening and terminal selection. Sampling restarts from configured seeds. See [Rebuild a selected stage](../guides/results-and-recovery.md#5-rebuild-a-selected-stage).

## Append or remove rounds

### Append rounds

Append trailing entries to `sample.rounds`, preserving the former normalized sequence as an exact prefix. Ordinary invocation retains old rounds, starts at the first new student train and refreshes terminal selection. See [Round activation](../guides/sampling.md#6-assign-tasks-to-rounds).

### Remove trailing rounds

For an intentional shrink, shorten `sample.rounds` so round N is terminal. To keep `r0` and `r1`, preflight:

```sh
dice run.json --environment environment.json --execution execution.json --resume fp --preflight
```

Then execute:

```sh
dice run.json --environment environment.json --execution execution.json --resume fp --truncate-after-round 1
```

DICE archives retired higher-round results/evidence under `.dice_state/round_truncations/` and rebuilds FP and its suffix. N must match the shortened configuration's terminal round.

## Combinations and exit status

- `--probe` is mutually exclusive with `--preflight` and `--resume`.
- `--resume BLOCK --preflight` checks that rebuild suffix.
- Truncation is used with executing `--resume fp`; use ordinary suffix preflight first.

| Exit status | Meaning |
| --- | --- |
| `0` | Normal return; inspect probe JSON `status` as well |
| `2` | CLI syntax or argument-combination error |
| Other nonzero | Validation, backend or execution failure; follow terminal/task diagnostics |

See [Troubleshooting](../guides/troubleshooting.md).
