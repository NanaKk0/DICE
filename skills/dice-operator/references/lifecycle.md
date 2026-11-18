# Lifecycle operations

Run commands from `RUN_ROOT` with the selected DICE control environment. Relative
profile arguments are resolved from the command directory. `--environment` and
`--execution` select profiles; `DICE_ENVIRONMENT` and `DICE_EXECUTION` are their
fallbacks. Keep the same selected profiles between preflight and execution.

## Commands

```sh
dice run.json --environment environment.json --execution execution.json --preflight
dice run.json --environment environment.json --execution execution.json
```

The execution profile is optional for default local execution. Supply the
environment bindings required by the selected backend operations. When the
console command is unavailable but the Python package is importable, the same
public entrypoint can be invoked with:

```sh
python -c "from workflow_app.entrypoint import console_main; console_main()" run.json --help
```

| Request | Public operation |
| --- | --- |
| Check configuration structure | Bundled inspector; no native work |
| Check environment, recipes and placement | `dice run.json --environment environment.json --preflight` |
| Run or continue | Ordinary invocation, after preflight |
| Inspect progress | `iterN/status.log`, `iterN/dice.log`, relevant task/backend logs |
| Check exact rebuild feasibility | `dice run.json --probe r1/sample` with selected profiles when applicable |
| Rebuild an exact suffix | `dice run.json --environment environment.json --resume r1/sample --preflight`, then the same command without `--preflight` |

Read the installed `dice --help` for current options. Use public flags; hidden
worker entrypoints are owned by DICE's runtime.

## Blocks and completion

The route is teacher preparation (`teach/train`, `teach/screen`, and single-
teacher `teach/scan`), student rounds (`rN/train`, `rN/sample`, `rN/screen`),
then `fp`. Configured `fp.label` adds `fp/label`. Derive actual block names and
roles from the normalized launch rather than guessing a fixed inventory.

Completion is established by a published result usable by its owning DICE
consumer. Logs, process exits and scheduler/task receipts are diagnostic.
Ordinary invocation reuses usable results; a usable terminal result makes it a
no-op. Changing configuration alone does not invalidate an existing result.
To apply changed settings to completed work, select the intended rebuild block.

Use existing result validation interfaces and indexed reads. Large trajectories
do not require a separate agent-written validation pass, directory inventory or
hash manifest. Retain integrity checks already owned by DICE.

## Probe and rebuild

`--probe BLOCK` is a read-only exact-resume feasibility query. When required
results are usable, it statically checks the selected suffix, external inputs
and supplied profiles. It performs no native work, target contact or cleanup.
Read `status`, `target` and `reason`; blocked results also report
`blocking_result`, `consuming_block` and `recommended_resume`.

`--resume BLOCK` forces that block and the current-iter suffix after dependency
checks and suffix preflight. Preserve the user's exact boundary. If an earlier
producer is recommended, show the additional rebuild scope and seek approval
only when that scope has not already been authorized. DICE owns invalidation;
leave lifecycle directories and `.dice_state` intact for its recovery path.

Stop dependency inspection at an already usable result. For example, a later
training rebuild that consumes an earlier valid screen result does not need to
revalidate that screen's historical raw sample. Recovery remains block-level.

## Change the number of rounds

To extend a completed explicit-round iter, append to `sample.rounds` while
preserving the former normalized sequence as an exact prefix. Preflight and
use ordinary invocation. DICE retains old rounds, refreshes terminal FP and
starts at the first new `rN/train`. Reorder/insertion or ambiguous live work
requires resolving the conflict before changing results.

For an intentional authorized shrink, shorten `sample.rounds`, preflight
`--resume fp`, then execute with `--resume fp --truncate-after-round N`, where
`N` is the retained terminal round. DICE archives the retired evidence in
`.dice_state/round_truncations/` and rebuilds FP. Use this public operation
rather than deleting round directories.

## Start the next iter

Follow [Configuration](configuration.md#continue-with-dft-labels) and the
matching quickstart's label-return steps. Preserve the previous run JSON,
validate the new labels, preflight the new configuration and run it ordinarily.
