# Return first-principles labels

DICE's default terminal artifact is `iterN/fp/selected.xyz`: selected geometries
and acquisition metadata, ready for external FP calculations. External returns
can come from your chosen calculator if geometry, labels, units and scientific
protocol are consistent. The built-in optional label block supports only its
registered CP2K and ABACUS routes.

## External calculations

1. Read the selection and retain a mapping to submitted geometries.
2. Compute with your chosen consistent electronic-structure protocol. Record
   failures and exclude unsuccessful returns rather than filling labels.
3. Export each computed geometry with matching energy/forces and required
   stress/virial. If relaxation changed geometry, use the computed coordinates
   and their corresponding forces.
4. Validate the [data contract](data.md) and place successful nonempty labelled
   returns in a declared run-local extxyz file.
5. Keep prior iteration history, increment `iter`, set student bootstrap to
   `[]`, and explicitly declare retained return files in `fp_data`.

The [handoff example](../examples/iteration_handoff/README.md) exercises these
membership semantics with synthetic data. It does not run an FP calculator.

## Optional same-host label operation

This **FP-object fragment** appends a CP2K label block:

```json
{"keep": 10, "policy": "DIRECT", "all_rn": true, "label": {"calculator": "CP2K", "recipe": "recipes/cp2k.json"}}
```

Use `ABACUS` with its matching recipe for that adapter. `VASP` is not a legal
built-in calculator name; a VASP external return remains a normal labelled-data
handoff. This block is not a generic command runner or a remote FP scheduler.
Bind `first_principles.cp2k.label` with executable `cp2k`, or
`first_principles.abacus.label` with executable `abacus`, to the native profile.
Configure the same-host CPU resources and bounded workers for that operation.

A successful local block publishes `iterN/fp/label/labeled.xyz`; a later
iteration still declares it explicitly in `fp_data`. An empty terminal selection
is a valid workflow outcome and supplies no new nonempty FP return.

## Native recipe responsibilities

| Calculator | Required preparation |
| --- | --- |
| CP2K | JSON recipe with `template`, optional staged `assets`, typed k-point spacing and `require_stress`; a valid static ENERGY_FORCE template |
| ABACUS | JSON recipe with static SCF template, pseudopotentials covering selected elements, optional LCAO orbitals, typed k-points/assets and stress request |

Recipe asset paths are relative to the recipe directory. CP2K templates must
provide the managed coordinate/cell variable contract (`COORDFILE`, `CELL_A/B/C`
and `ANGLE_A/B/C`). ABACUS PW and LCAO have different orbital requirements.
Consult the shipped recipe loaders under `workflow_backends/first_principles/`
when adapting native inputs; a physically valid template and licensed species
assets must be supplied for your system.

The ABACUS adapter's CPU fixture coverage does not establish native execution
on your installation. Check the complete calculator/parser/units route before
large label pools. Existing labels and a compatible scientific protocol matter
more than a successful process exit. [Recovery](recovery.md) describes block
rebuilding; do not alter task records manually to adopt failed calculations.
