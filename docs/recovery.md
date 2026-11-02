# Recovery and diagnostics

Run from the directory containing the intended `run.json`. Ordinary
invocation follows actual result dependencies and reuses consumer-runnable
canonical artifacts. With a completed terminal result, ordinary invocation
reuses that result. Canonical outputs establish completion; the configuration
specifies the execution plan.

```sh
dice run.json --environment environment.json
```

Use a read-only probe to check the feasibility of an exact rebuild boundary:

```sh
dice run.json --environment environment.json --probe r1/sample
```

A runnable probe statically validates the suffix. A blocked response can report
the blocking result and a recommended earlier producer; review that change
of scope before choosing a different target.

To deliberately rebuild exactly the selected block and its remaining
current-iteration suffix, preflight then execute:

```sh
dice run.json --environment environment.json --resume r1/sample --preflight
dice run.json --environment environment.json --resume r1/sample
```

This replaces the selected block and the remaining current-iteration suffix.
Choose the intended boundary explicitly and use these commands to manage
recovery while preserving the lifecycle state.

## Changing the round sequence

An already-completed iteration can extend an explicit `sample.rounds` array
only by appending to its exact former prefix. Ordinary invocation preserves
old round evidence and rebuilds the stale terminal handoff from the first new
round. Reorder, middle insertion and accidental shrink fail closed.

Intentional shrink is separate: shorten the configuration, preflight
`--resume fp`, then deliberately execute `--resume fp --truncate-after-round N`,
where N is the configured terminal round. DICE archives higher-round evidence
below `.dice_state/round_truncations/` for recovery and traceability.

## What to inspect

Use `iterN/status.log`, `iterN/dice.log`, backend logs, attempts and operation
receipts to diagnose failures. Some sampling routes can retain validated
prefixes of failed tasks; others fail atomically. Canonical consumer validation
determines completion. Logs and optional plots provide diagnostic context.
