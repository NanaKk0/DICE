# Scientific configuration and field reference

`run.json` defines the scientific workflow through a fixed set of accepted fields.
Its required top-level fields are `iter`, `teacher_train_bootstrap`,
`student_train_bootstrap`, `fp_data`, `teacher`, `student`, `sampler`, `sample`,
`screen`, and `fp`. Optional top-level boolean `visualize` defaults to true;
explicit false disables non-authoritative diagnostics.

Every declared input/recipe/model path is relative to that `run.json`.
List files explicitly using run-root-relative paths. Configure executables and
activation in [environment profiles](environments.md), and hardware placement
in [execution profiles](execution.md).

## Four separate layers

| Layer | Settings |
| --- | --- |
| `run.json` | Data, chemistry/type maps, model roles, train policy, sampling, thresholds, quotas, selection, recipe paths |
| Environment profile | Source scripts, logical executables, role bindings, approved runtime controls |
| Execution profile | Targets, pools, resource classes, role bindings, explicit launchers |
| Native recipe | Backend architecture/input and backend-owned scientific settings |

The CPU mock example generates a complete accepted `run.json` and all referenced
fixture inputs. Read its [configuration ledger](../examples/cpu_mock/README.md)
before adapting it. Its recipes and data are synthetic test fixtures.
Prepare native recipes and model inputs for scientific runs using the
[own-system walkthrough](own-system.md).

Use this page for field lookup; use [screening](screening.md) and [sampling](sampling.md)
for strategy explanations. Start an adaptation with [your own system](own-system.md), then check the
[labelled-data contract](data.md). The [generated accepted-key inventory](configuration-fields.md)
is extracted from this distribution's loader.
Regenerate or check it with `python tools/generate_config_reference.py --check`.

## Required data and identity

“Required” fields must be supplied explicitly. Quotas count frames and apply
separately to each active selection group. Supply finite JSON numbers with
the documented types.

| Field | Type / default | Contract |
| --- | --- | --- |
| `iter` | Integer >= 0; required | Iteration index; outputs are `iterN/` in the run root. |
| `teacher_train_bootstrap` | Nonempty array of paths; required in every iter | Labelled external teacher data. List the full retained external baseline explicitly. |
| `student_train_bootstrap` | Array of paths; required | Nonempty for iter 0; must be `[]` later. The current teacher labels the initial student geometries. |
| `fp_data` | Array of paths; required | Use `[]` at iter 0. For iter > 0, supply nonempty files containing returned **labelled** first-principles data. |
| `visualize` | Boolean; `true` | Default-on, non-authoritative diagnostics; `false` disables them. True normalizes to omission. |

All declared paths resolve from the directory containing `run.json`.
Use nonempty relative file paths within that run root. The loader rejects
parent traversal (`..`) and glob syntax (`*`, `?`, `[`); each input must resolve
to a regular file. Keep private model and data files in your own run directory.
External-file feasibility checks require resolved
regular files. The loader validates path syntax; suffix preflight
checks the files actually needed by that suffix.

Teacher bootstrap and FP data are merged in list order and exact-geometry
duplicates are checked for compatible required labels. Conflicting labels fail
closed. Ingest creates canonical identities from the computed geometry and
run configuration path. `system_id` is
retained when present and otherwise falls back to the Hill formula. Assign
meaningful system IDs when same-formula populations need distinct teacher
validation strata. See [data ingestion](data.md) for the exact separation of
external labels, student seeds and reference populations.

## Teacher and student fields

Both objects are required and closed. Here `role` means either `teacher` or
`student`. Defaults listed here apply to DICE fields; native recipes define
their own backend settings.

| Field | Type / default | Applicability and meaning |
| --- | --- | --- |
| `role.backend` | String; required | Teacher: `MACE` or `DeepMD`. Student: `DeepMD` or `NEP`. |
| `role.model` | `single` or `committee`; required | Number and composition of model members. |
| `role.type_map` | Nonempty unique symbol array; required | Ordered active chemistry; teacher must cover every student element. Every frame must be covered by its consuming role. Preserve the foundation model's native element map separately from this active chemistry. |
| `role.recipe` | Relative path; required | Native training input. DICE supplies training/validation `systems`; omit those fields from DeepMD recipes. |
| `role.policy` | `from_scratch` or `foundation_finetune`; required | Initialization policy. `from_scratch` forbids `foundation_model`; finetuning requires it. Neither value is a restart/resume instruction or a command to carry the previous iter's model weights. |
| `role.foundation_model` | Relative path; no default | Required only for `foundation_finetune`; backend-compatible native foundation model/checkpoint. |
| `role.valid_frac` | Number in [0, 1); required | Requested validation fraction. Teacher 0 uses training frames for self-evaluation. Positive teacher splitting is deterministic and system-stratified, retains >=1 train frame per eligible system, and needs a system with >=2 frames. Student splitting is deterministic by canonical identity/run path; inspect realized counts for small datasets. |
| `role.cli` | `tf` or `pt`; required for DeepMD | Selects the DeepMD interface. Configure the executable in environment bindings; omit this field for MACE/NEP. |
| `role.members` | Integer >=2; required for committee | Forbidden for single models. |
| `teacher.lo`, `teacher.hi` | Numbers satisfying 0 <= lo < hi; committee only | Force-deviation thresholds in eV/Å. |
| `role.train_options.skip_neighbor_stat` | Boolean; `false` | DeepMD only. The sole accepted train option. False/empty options normalize away. |
| `student.compress` | Boolean; `false` | DeepMD only; enables its compression step. Backend/model feasibility still applies. Forbidden for NEP. |
| `student.foundation_restart` | Relative path; no default | Required only for NEP `foundation_finetune`; forbidden on other routes. Separate from `foundation_model`. |
| `student.include_teacher_train_data` | Boolean; `false` | Adds the current teacher **training** membership to the student relabel union. Uses the current teacher, applies the routing/label checks and deduplicates exact geometry. |

For single-model GMM routes, a teacher is MACE or DeepMD-PT **DPA4**; a student
is DeepMD-TF **se_e2_a**, DeepMD-PT **DPA4C**, or **NEP4**. Committee DeepMD
roles can use TF/PT and committee teachers can use MACE. Train-recipe
feasibility validates the required descriptor family. DeepMD students require LAMMPS; NEP requires
GPUMD. Teacher and student can independently be single or committee.

Teacher trains once per iteration on the declared external labelled union;
student trains once per round on the growing teacher-labelled student dataset.
Every executed train block applies its declared initialization policy and
foundation asset. Dataset carry-over supplies the next training population;
the declared policy controls weight initialization. Recovery reuses a
consumer-runnable published train result or rebuilds the explicitly selected
block and suffix.

## Screening fields and terminal selection

For the scientific meaning of GMM p values, committee deviations, trust,
representative selection and quotas, read [screening](screening.md). The table
below is the exact field/boundary reference. Single-model p is dimensionless;
committee thresholds use eV/Å.

| Field | Type / default | Contract |
| --- | --- | --- |
| `screen.pool` | Positive integer or per-round integer array; required | Student proposal capacity per group; array length must equal effective round count. Conditional `-1` means all in-band candidates: allowed for committee students or GMM single students with explicit ceiling, never joint prescreen. |
| `screen.keep` | Positive integer or per-round integer array; required | Maximum student representatives per group; bounded by a positive pool. |
| `screen.fp_pool` | Positive scalar integer; required for single teacher | Teacher FP-candidate reserve per group. `-1` allowed only with explicit `teacher_p.ceiling`; forbidden for committee teachers. This is distinct from terminal `fp.keep`. |
| `screen.policy` | `DIRECT`, `ELEMENT_DIRECT`, `FPS`, `random`; required | Representative selector after student prescreen. Case-sensitive. |
| `screen.prescreen` | `GMM`; default | Single-student-only; alternatives `JOINT_DIRECT` and `ELEMENT_JOINT_DIRECT` require positive pools, forbid `student_p`, and disable safe fill. Explicit GMM normalizes away. |
| `screen.teacher_p.floor` | Number [0,1); `0.0` | Single teacher: p < floor is discarded from both teacher views. |
| `screen.teacher_p.ceiling` | Number (floor,1]; absent | Teacher FP-candidate interval is [floor, ceiling); no ceiling means [floor,1], including p=1. Ceiling does not cap student trust eligibility. |
| `screen.teacher_p.trust` | Number [0,1]; `0.1` | Single teacher: student-label eligibility requires p >= floor **and p > trust**. Equality is rejected. FP reservation is independent: a frame can be in both views. No required ordering between trust and the candidate-band endpoints. |
| `screen.student_p.floor` | Number [0,1); `0.0` | GMM single student: discard p < floor before proposing train representatives. |
| `screen.student_p.ceiling` | Number (floor,1]; absent | GMM candidate interval [floor, ceiling); p >= ceiling is the covered/safe region. With no ceiling candidates include p=1 and there is no such safe region. |
| `screen.fill_underfilled_from_safe_region` | Boolean; `true` unless joint prescreen (`false`) | Uniformly fills a normal GMM shortfall from its covered region, or a committee-student shortfall from deviation < lo, after other eligibility gates. It cannot resurrect teacher-rejected frames. For GMM it cannot resurrect below-floor frames or create a safe region without a ceiling. |
| `screen.lo`, `screen.hi` | 0 <= lo < hi; required for committee student | eV/Å force deviation; student candidate interval [lo,hi). Forbidden for a single student. |
| `screen.pair_guard` | Object; absent (disabled) | `min` required, positive Å; `pairs` defaults `{}` and overrides unordered `A-B` pairs with thresholds >= min, using student elements. Periodic minimum-image geometry is checked; distance **<** threshold rejects, equality passes. |
| `fp.keep` | Positive scalar integer; required | Terminal quota per stable selection group, applied after gathering the selected rounds. |
| `fp.policy` | Selector string; required | Screen selectors plus `JOINT_DIRECT` / `ELEMENT_JOINT_DIRECT` for a single teacher only. |
| `fp.all_rn` | Boolean; `false` | False: final round only. True: union round candidate pools, with later duplicate-frame precedence, then one terminal selection per group. |
| `fp.label` | Object; absent | Optional local label block; exactly `calculator` (`ABACUS` or `CP2K`) and relative `recipe`. Omission stops at `fp/selected.xyz` for external FP handoff. No generic shell command or VASP calculator name is accepted here. |

Committee teacher training-seed trust at iter 0 is deviation < `teacher.lo`.
Round teacher routing uses teacher deviation to separate trusted student labels
from FP candidates; [lo,hi) is the teacher FP-candidate band and >=hi is rejected.
These thresholds use force deviations in eV/Å. Teacher and student gating is
student-first; the retired `screen.gate_order`, `teacher.trust` and
`teacher.cutoff` fields have no aliases.

Joint selectors include the descriptor owner's exact current training set as
non-selectable background; background capacity is not filled back with candidates.
Screen background belongs to the student, terminal FP background to the teacher.
`ELEMENT_DIRECT` instead uses candidates only and equal-weight means for present
elements; absent elements remain absent. On a known committee owner only
candidate-only ELEMENT_DIRECT resolves to random, with explicit
`policy_resolution.json`; joint element selection is invalid for a committee.
Missing/corrupt single-model descriptor evidence never permits that fallback.

## Sampling field reference

For task expansion, geometry requirements and condition examples, see
[sampling](sampling.md). The tables below specify the accepted fields.

For new configurations prefer explicit sampling. `sampler` then contains only
required `backend` (`LAMMPS` or `GPUMD`); recipe paths belong to MD condition
groups. The closed legacy alternative uses `sample.systems.<id>.seed`, optional
system-wide PLUMED, `sample.steps` (one positive integer per round), and
`ensemble`, `temperatures_k`, `pressures_bar` inherited from sample defaults or
overridden per system. It requires `sampler.recipe`, or LAMMPS
`sampler.recipes` with exactly the effective `nvt`/`npt` keys. Choose one sampling layout.

| Field | Type / default | Contract |
| --- | --- | --- |
| `sample.seeds.<role>.source` | Relative extxyz path; required | Geometry source; normally frame 0. |
| `sample.seeds.<role>.selection_group` | Safe ID; defaults to role ID | Immutable screen/FP quota boundary; several seed roles may share it. IDs start alphanumeric and contain only alphanumerics, `_`, `-`. |
| `sample.seeds.<role>.max_frames` | Positive integer; absent | Deterministic distinct-frame subsample per iter/round/role, capped at source length, with no replacement or duplication. Forbidden for NEB endpoint roles. |
| `sample.round_count` | Positive integer; required unless `rounds` | Repeats the declared seed/condition registry. |
| `sample.rounds` | Nonempty ordered object array; alternative to count | Each row optionally has `seeds` and `condition_groups`, nonempty unique known-ID arrays; omitted axis means all. Registry order, not row-list order, determines tasks; every round must have a nonempty seed/group intersection. |
| `sample.condition_groups.<id>.kind` | String; required | Closed kinds below, not native commands. |
| `sample.condition_groups.<id>.seeds` | ID array; all declared roles | Restricts assigned roles. NEB binds exactly one. |

Every MD temperature is a nonempty unique array of finite positive **K**.
NPT pressures are finite **bar**, a scalar or unique nonempty array (negative
values are not syntax-rejected, but physical applicability is your decision).
Ordinary `steps` and GCMC `cycles` are positive integer scalars or arrays
matching the total effective round count, including inactive rounds. MD
timestep, dump stride, thermostat/barostat settings and potential architecture
belong to validated native recipes, not extra run.json keys.

| Kind | Required fields beyond kind | Optional fields / defaults and restrictions |
| --- | --- | --- |
| `nvt` | `recipe`, `temperatures_k`, `steps` | LAMMPS or GPUMD; no pressure. |
| `npt` | `recipe`, `temperatures_k`, `pressures_bar`, `steps` | LAMMPS or GPUMD. |
| `nvt_program` | `recipe`, `segments` | LAMMPS only; one continuous task per active seed. |
| `npt_program` | `recipe`, `segments`, `pressures_bar` | LAMMPS only; one continuous task per seed/pressure. |
| `nvt_cell_program` | `recipe`, `segments` | LAMMPS only; optional cell endpoint on each segment, never on another kind. |
| `neb` | `images` (>0 movable interior images), `fmax_ev_a` (>0 eV/Å), `max_steps` (>0) | `init: linear` (`idpp` alternative), `method: neb` (`dyneb` alternative), `include_history: false`, `climb: false`, `relax: false`, `fixed_atoms: []`. No recipe. |
| `relax` | `fmax_ev_a` (>0 eV/Å), `max_steps` (>0) | `max_step_a` (>0 Å; omitted uses installed ASE FIRE default), `include_history: true`, `history_stride: 1`, `accept_unconverged: false`, `fixed_atoms: []`. Stride valid only with history. No recipe; atomic relaxation at fixed cell. |
| `md_mc` | `recipe`, `ensemble` (`nvt`/`npt`), `temperatures_k`, `steps`, `mc` | LAMMPS only; NPT also requires `pressures_bar`. `mc` is exactly `move: identity_swap`, two distinct student `elements`, positive `every_md_steps`, positive `attempts_per_event`. Steps must cover >=1 event. No default move in run.json. |
| `widom` | `guest` (relative geometry), `frames` (>0), `max_attempts` (>=frames) | LAMMPS route only; `random_seed: 0`, `overlap: {scale: 0.6, radii_a: {}}`; radii Å, positive scale. `overlap: null` disables the filter; a supplied radii override must be nonempty and complete for used elements. No recipe or MD budget. |
| `gcmc_md` | `temperatures_k`, `cycles`, `mc` | LAMMPS only; `random_seed` absent, `md` absent (pure MC), `limits: {}`. Detailed nested fields below. Used for configuration acquisition. |

A program segment is exactly `{temperature_k, steps}` (hold) or
`{start_temperature_k, stop_temperature_k, steps}` (linear ramp); temperatures
are positive K, steps positive integers. For `nvt_cell_program` only, optional
`cell` is either `{mode: isotropic_density, stop_density_g_cm3: positive}` or
`{mode: symmetric_strain, stop: {xx, yy, zz, yz, xz, xy}}` with all six finite
dimensionless strain components and positive-definite `I+E`. State persists
through the ordered segments; no segment-resume interface exists.

NEB sources are ordered two-frame endpoints with identical atom count, symbol
order, PBC and exact fixed cell. `fixed_atoms` are unique zero-based atom indices
and endpoint coordinates at fixed indices must coincide. A role cannot serve
NEB and non-NEB groups together. Optional
`climb_preconverge_fmax_ev_a` is >fmax, requires `method: neb`, `climb: true`,
and shares the one path-step budget between preconvergence and climbing. DyNEB
means dynamic relaxation, not doubly nudged NEB. Optional endpoint `relax`
gives each endpoint a separate full step budget before the path. Default
final-only exposes the converged complete band including endpoints; history
exposes initial/completed interior steps. Relax history includes initial and
stride-selected steps plus the final frame exactly once; unconverged endpoints
are accepted only with explicit `accept_unconverged: true`. Both kinds use the
first student member for optimization and all members for exposed evaluations.

## GCMC nested reference

| Field under condition group | Type / default | Meaning |
| --- | --- | --- |
| `mc.guest`, `mc.pressures_bar`, `mc.moves`, `mc.eos` | Required | Guest relative path; reservoir pressure is a nonempty positive bar array; moves positive integer; EOS object. |
| `mc.every` | Positive integer; `100` | MC observation interval in moves. |
| `mc.translate_a`, `mc.rotate_deg` | `0.75` Å, `90` degrees | Positive translation/rotation bounds; rotation <=180. |
| `mc.weights` | `{insertion: .25, deletion: .25, translation: .25, rotation: .25, reinsertion: 0}` | If supplied, all five keys required; nonnegative, sum 1, equal positive insertion/deletion and zero reinsertion. |
| `mc.eos.kind` | Required: `ideal`, `fixed_phi`, `pr` | `phi` required only for fixed_phi, positive dimensionless. PR requires positive `tc_k` (K), `pc_bar` (bar), finite `omega` (dimensionless); these are forbidden for other EOS kinds. |
| `mc.overlap` | `{scale: 0.6, radii_a: {}}` | Positive scale, optional positive per-element radii in Å. Unlike Widom, null is not accepted. |
| `mc.de_max_ev` | Absent / null | Optional positive finite eV bound. |
| `limits.max_guests` | Absent / null | Optional positive integer guest cap. |
| `random_seed` | Absent / null | Optional nonnegative integer. Do not infer a global workflow seed. |
| `md` | Absent | Optional interleaved MD: exactly `ensemble` (`nvt`, `nve`, `npt`), relative `recipe`, positive `steps`, positive `every`; NPT additionally requires finite `pressure_bar` (bar), forbidden otherwise. |

The generated inventory lists accepted keys. The loader enforces compatible
combinations. Advanced kinds require backend-specific input/geometry checks
and a matching native environment.

## Model roles and sampling

Teacher backends are MACE or DeepMD; student backends are DeepMD or NEP.
DeepMD uses explicit `cli: tf` or `pt`. The DeepMD student pairs with LAMMPS,
and NEP with GPUMD. Native model families such as DPA4/DPA4C are recipe/backend
capabilities selected through the native recipe. Keep foundation native type
maps distinct from the active workflow chemistry.

Legacy `sample.systems` remains accepted. Explicit `sample.seeds` and
`sample.condition_groups` describe independent roles and conditions; ordered
`sample.rounds` can activate subsets. Choose either `rounds` or `round_count`.
Supported condition kinds have closed backend-specific schemas; the source
loader `workflow_core/lifecycle/config.py` is the definitive field validator.
Advanced MD/NEB/relax/insertion configurations need matching validated native
recipes/calculators and installation-specific native checks.

`screen.policy` selects later representatives using DIRECT, ELEMENT_DIRECT,
FPS or random. For a single student, optional `screen.prescreen` selects GMM,
JOINT_DIRECT or ELEMENT_JOINT_DIRECT subject to the loader's cross-field rules.
`fp.policy` also supports joint selectors for a single teacher. Candidate-only
ELEMENT_DIRECT on known committee ownership resolves deterministically to
random with explicit policy-resolution evidence. Single-model descriptor
evidence is validated before descriptor-based selection.

Default terminal FP selects from the final round; `fp.all_rn: true` merges
candidate rounds before one selection per stable group. `fp.keep` limits the
eligible candidate structures in that handoff.

## Validate before computing

```sh
dice run.json --environment environment.json --preflight
```

Preflight checks the selected suffix, external inputs, role bindings and
placement through static validation. Before an actual run, confirm the data,
model, sampling, threshold, resource and calculator choices, and check native
stability and accuracy under the intended scientific conditions.
