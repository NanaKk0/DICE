# Use CP2K or ABACUS labelling

The built-in CP2K and ABACUS integrations are experimental. This guide configures the optional same-host `fp/label` operation after DICE selects the structures for reference calculations. The [scientific walkthrough](../../getting-started/quickstart.md) covers computing and returning DFT labels with a separate calculator workflow.

## Prepare the calculator and recipe

| Calculator | Software and input files |
| --- | --- |
| CP2K | `cp2k` executable; JSON recipe naming a static `ENERGY_FORCE` template, optional staged `assets`, k-point spacing and `require_stress` |
| ABACUS | `abacus` executable; JSON recipe naming a static SCF template, pseudopotentials for selected elements, optional LCAO orbitals, k-point/assets and stress settings |

Recipe asset paths are relative to the recipe directory. CP2K templates use the managed coordinate/cell variables `COORDFILE`, `CELL_A/B/C` and `ANGLE_A/B/C`. ABACUS LCAO recipes also supply the orbital files for their elements.

## Add the label operation

This fragment is the `fp` object in `run.json`:

```json
{
  "keep": 10,
  "policy": "DIRECT",
  "all_rn": true,
  "label": {"calculator": "CP2K", "recipe": "recipes/cp2k.json"}
}
```

Set `calculator` to `ABACUS` and name its matching recipe for that route. The accepted `fp.label` fields are defined in the [configuration reference](../../reference/run-json.md#screening-fields-and-terminal-selection).

## Bind the environment and CPU workers

| Calculator | Environment operation role | Logical executable |
| --- | --- | --- |
| CP2K | `first_principles.cp2k.label` | `cp2k` |
| ABACUS | `first_principles.abacus.label` | `abacus` |

Bind the selected role to its activated software profile. Set positive `runtime.fp_workers` and `runtime.cpu_threads`; their product must fit the controller's available CPU affinity. Profile fields and executable resolution are defined in the [environment reference](../../reference/environment.md).

## Run and collect the labels

Preflight the configured run, then start it with the ordinary DICE command. A successful label operation publishes `iterN/fp/label/labeled.xyz`. Declare that file in the next iter's `fp_data` using the same label-ingestion procedure as other reference data. An empty FP selection produces no new labelled return.

Required labels, units and atom ordering are defined in the [data reference](../../reference/data.md). Calculator task logs provide failure diagnostics; ordinary continuation and whole-block rebuilding use the [recovery commands](../results-and-recovery.md).
