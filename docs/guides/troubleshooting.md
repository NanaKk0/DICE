# Troubleshooting

Use this page to locate configuration, data and execution problems. Commands and paths follow the [Quickstart](../getting-started/quickstart.md); run them beside `run.json` and substitute the appropriate iter/round.

## Locate the failure

| Symptom | Start with |
| --- | --- |
| CLI or preflight error | Terminal error and named configuration field/input |
| Stage fails during execution | Latest `[FAILED]` record in `iter0/status.log` and its diagnostic path |
| Few training additions or DFT candidates | `iter0/dice.log`, round reports and group populations |
| Apply changes to existing outputs | [Rebuild a selected stage](results-and-recovery.md#5-rebuild-a-selected-stage) |

## 1. Missing dice command or wrong installation

Check the active control environment:

```sh
command -v dice
python -c 'import sys, workflow_app; print(sys.executable); print(workflow_app.__file__)'
```

Activate the environment containing DICE and confirm command/module paths. Follow [Installation](../getting-started/installation.md) when reinstalling, then run `dice --help`. Backend activation is described in [Configure backend environments](backends.md).

## 2. Preflight field, path or binding errors

Inspect the file and field named by the error:

| Error concerns | Action |
| --- | --- |
| `run.json` field | Check names, types and combinations against [Configuration overview](configuration.md) and the [reference](../reference/run-json.md) |
| Data, recipe or model path | Confirm readability from the run directory and configure a relative path such as `data/teacher_initial.extxyz` |
| Missing operation binding | Bind the reported role in `environment.json` for the selected models/sampler |
| Activation or executable | Check `source_list`, `executables` and [backend invocation](backends.md#verify-backend-invocation) |
| Target, resource class or binding | Check names and supported shapes in [Compute resources](resources.md) |

Repeat preflight after fixing the error:

```sh
dice run.json --environment environment.json --execution execution.json --preflight
```

## 3. Data parsing or label validation fails

Inspect the reported file, frame and property.

1. Read it with ASE and check elements, cell and PBC.
2. Supply all labels required by teacher/reference training: total energies, finite forces with shape `(natoms, 3)`, and stress/virial when requested.
3. Check units and force/atom ordering.
4. Check model element coverage and [DeePMD native type maps](deepmd.md#initialization-and-native-recipes).

See [Prepare input files](inputs.md) and the [label-validation snippet](../getting-started/quickstart.md#5-prepare-new-labels-for-iter1).

## 4. Training, freeze, compression or sampling fails

Inspect recent stage records:

```sh
tail -n 40 iter0/status.log
```

Follow the `[FAILED]` diagnostic path to the task attempt.

| Operation | Logs and inputs |
| --- | --- |
| Training | `deepmd_train.stdout.log`, `deepmd_train.stderr.log`, generated recipe and `lcurve.out` |
| Freeze | Model-operation logs, for example `deepmd_freeze.stderr.log` |
| Compression | Logs under `rN/sample/sampling_models/student/` |
| LAMMPS | Task `lammps_sample.stdout.log`, `lammps_sample.stderr.log`, `log.lammps` and `input.lammps` |

Failed attempts also retain diagnostics under the stage's `.attempts/`; use the path reported by DICE. Fix the cause, repeat preflight and invoke normally. To rebuild completed work, choose an explicit boundary using [Inspect results and resume](results-and-recovery.md).

## 5. Unexpected sampling controls or ambiguous overrides

Inspect preflight `sampling_controls` and task `protocol.lammps`, `input.lammps` and `log.lammps`.

1. Check active seeds, groups and batch conditions for the round.
2. Explicit DICE fields override native counterparts; omission inherits. An explicit `kind` selects an ensemble; recipe-only groups run the native protocol.
3. For multiple targets, select the reported candidate in `targets`. Thermostat/barostat targets use fix IDs; `run` uses a one-based ordinal.
4. Compare generated commands and logs. Native `velocity` commands control initialization; thermostat temperature overrides act on the selected controller.

See [Configure sampling](sampling.md). Apply changes to completed sampling from the corresponding `rN/sample` boundary.

## 6. Few or empty training additions or DFT candidates

Inspect `iter0/dice.log`, `screen_report.json`, `selection_group_outcomes.json` and the published round data.

| Step | Check |
| --- | --- |
| Sampling | Tasks, trajectories and eligible observations; the tutorial's MD route uses positive-step frames |
| Student candidates | Eligible score range and `screen.pool` |
| Teacher assessment | Teacher-label trust and DFT-candidate bands |
| Group selection | Group ownership, `screen.keep`, `screen.fp_pool`, candidate populations and actual selections |
| Final DFT selection | `fp.all_rn`, deduplication and `fp.keep` |

Pool/keep fields are capacity limits. Actual populations depend on eligible configurations. Teacher-training additions and DFT candidates apply their own conditions; a configuration can participate in both routes.

Adjust sampling, bands or quotas to your scientific goal, then rebuild the corresponding stage. See [Screening fields](../reference/run-json.md#screening-fields-and-terminal-selection) and [Read screening reports](results-and-recovery.md#2-read-screening-reports).

## 7. Ordinary invocation reuses results after a config change

A usable published terminal result completes the iter. Choose a rebuild boundary to apply changed settings to completed work: `rN/train` for training, `rN/sample` for sampling, `rN/screen` for screening or `fp` for terminal selection.

Follow [Rebuild a selected stage](results-and-recovery.md#5-rebuild-a-selected-stage) to check feasibility, preflight and execute the intended suffix.

## Reporting an issue

Include in [GitHub Issues](https://github.com/NanaKk0/DICE/issues):

- Expected and observed behavior, exact command and triggering operation.
- DICE version/commit, backend versions and model family.
- Relevant configuration and a small reproducer, retaining operation bindings and resource shape.
- Failed stage, corresponding status record and relevant error-log excerpts.

Remove credentials and private addresses before sharing; use public sample data or identify the public model source.
