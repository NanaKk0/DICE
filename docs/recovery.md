# Continue or rebuild a run

Use the intended run directory and control environment. Ordinary continuation
and deliberate rebuilding have different effects; choose the operation that
matches your intent before executing it.

| Your intent | Operation | Expected scope |
| --- | --- | --- |
| Continue after interruption | Ordinary `dice run.json` | Reuse valid results and perform needed remaining work |
| Reopen a completed iteration without changes | Ordinary invocation | Reuse the consumer-valid terminal result |
| Inspect whether an exact rebuild is feasible | `--probe BLOCK` | Read-only dependency/suffix validation |
| Recompute a selected block | `--resume BLOCK` | Replace that block and every later current-iter result |
| Add new rounds | Append-only round extension, ordinary invocation | Preserve valid exact prefix and rebuild the stale terminal handoff |
| Intentionally remove higher rounds | Explicit truncation procedure | Archive the retired higher-round evidence and rebuild FP |

## Ordinary continuation

```sh
dice run.json --environment environment.json
```

DICE checks formal outputs through their consumers. A valid completed terminal
result can make this a no-op even when some unused prefix evidence or current
settings differ. Editing a recipe does not by itself force a completed train
result to be recomputed. Inspect the dependency boundary and explicitly request
rebuilding when that is your intent.

## Inspect and rebuild an exact suffix

```sh
dice run.json --environment environment.json --probe r1/sample
dice run.json --environment environment.json --resume r1/sample --preflight
```

The probe is read-only. A blocked response can identify an earlier producer
that must be repaired. It does not silently widen the requested target.
After reviewing feasibility and the replacement scope, execute:

```sh
dice run.json --environment environment.json --resume r1/sample
```

This rebuilds r1/sample, r1/screen and the remaining configured current-iteration
suffix, including FP. It is not a request to continue a particular native MD
step or optimizer checkpoint. Public boundaries are teach/train, teach/screen,
teach/scan where applicable, rN/train, rN/sample, rN/screen, fp and optional fp/label.

## Extend or shorten the round plan

An explicit `sample.rounds` array can grow only by appending to its exact former
prefix. After inspection and preflight, ordinary invocation continues at the
first new rN/train and replaces the stale FP handoff. Reordering, insertion,
renumbering or accidental shrink is rejected.

To intentionally shorten a completed plan, first change the scientific plan
to its desired terminal round N. Inspect and preflight the exact FP boundary,
then use `--resume fp --truncate-after-round N`. Here N is the final zero-based
round index in the shortened configuration. DICE archives retired higher-round
evidence beneath `.dice_state/round_truncations/`; ordinary continuation does
not perform this destructive-scope choice for you.

## Diagnose before changing state

Inspect `iterN/status.log` for lifecycle attempts, `iterN/dice.log` for scientific
screening counts and the failed block's backend logs for native errors.
Successful scheduler/process receipts or a SUCCESS log line do not alone prove
that all required artifacts can be consumed.

Leave `.dice_state` and published model/dataset relationships managed by DICE.
Do not repair a run by deleting arbitrary state files or renaming model
suffixes. Sampling-prefix retention is route-specific; use the declared public
recovery boundary. [Troubleshooting](troubleshooting.md) maps common symptoms
to checks; [execution](execution.md) explains remote operational receipts.
