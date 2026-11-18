# Scientific configuration and field reference

`run.json` defines the scientific workflow through a fixed set of accepted fields.
Its required top-level fields are `iter`, `teacher_train_bootstrap`,
`student_train_bootstrap`, `fp_data`, `teacher`, `student`, `sampler`, `sample`,
`screen`, and `fp`. Optional top-level boolean `visualize` defaults to true;
explicit false disables the optional diagnostic plots.

Every declared input/recipe/model path is relative to that `run.json`.
List files explicitly using run-root-relative paths. Configure executables and
activation in [environment profiles](environment.md), and hardware placement
in [execution profiles](execution.md).

This page is the field reference. The [walkthrough](../getting-started/quickstart.md) supplies a complete run, and the [data reference](data.md) defines label shapes and units. The [generated key appendix](appendices/configuration-fields.md) lists accepted key names.

## Required data and identity

“Required” fields must be supplied explicitly. Quotas count frames and apply
separately to each active selection group. Supply finite JSON numbers with
the documented types.

| Field | Type / default | Contract |
| --- | --- | --- |
| `iter` | Integer >= 0; required | Iteration index; outputs are `iterN/` in the run root. |
| `teacher_train_bootstrap` | Nonempty array of paths; required in every iter | FP-labelled teacher reference data. List the full retained baseline explicitly. |
| `student_train_bootstrap` | Array of paths; required | Nonempty for iter 0; must be `[]` later. The current teacher labels the initial student geometries. |
| `fp_data` | Array of paths; required | Use `[]` at iter 0. For iter > 0, supply nonempty files containing returned **labelled** first-principles data. |
| `visualize` | Boolean; `true` | Enables diagnostic plots; `false` disables them. |

All declared paths resolve from the directory containing `run.json`.
Use nonempty relative file paths within that run root. Accepted paths have no parent traversal (`..`) or glob syntax (`*`, `?`, `[`), and resolve to regular files. The loader checks path syntax; suffix preflight checks the files consumed by that suffix.

Teacher bootstrap and FP data are merged in list order and exact-geometry
duplicates are checked for compatible required labels. Conflicting labels fail
closed. Ingest creates canonical identities from the computed geometry and
run configuration path. `system_id` is
retained when present and otherwise falls back to the Hill formula. Assign
meaningful system IDs when same-formula populations need distinct teacher
validation strata. See [data ingestion](data.md) for the exact separation of
FP reference labels, student seeds and reference populations.

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
| `role.policy` | `from_scratch` or `foundation_finetune`; required | Initialization policy. `from_scratch` forbids `foundation_model`; finetuning requires it. Every executed train block initializes according to this policy and its declared foundation asset. |
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

## Screening fields and terminal selection

For the scientific meaning of GMM p values, committee deviations, trust,
representative selection and quotas, read [screening](../concepts/screening.md). The table
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
| `fp.label` | Object; absent | Optional local label block with exactly `calculator` (`ABACUS` or `CP2K`) and relative `recipe`. With this object, `fp/label` follows FP selection; omission ends at `fp/selected.xyz`. |

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

The table below defines the seed, condition and round registries. Mode-specific fields are defined in the [sampling type reference](sampling.md).

`sampler.backend` selects `LAMMPS` for DeepMD students or `GPUMD` for NEP students. Explicit sampling uses the fields below; MD recipe paths are condition-group fields.

| Field | Type / default | Contract |
| --- | --- | --- |
| `sample.seeds.<role>.source` | Relative extxyz path; required | Geometry source; normally frame 0. |
| `sample.seeds.<role>.selection_group` | Safe ID; defaults to role ID | Immutable screen/FP quota boundary; several seed roles may share it. IDs start alphanumeric and contain only alphanumerics, `_`, `-`. |
| `sample.seeds.<role>.max_frames` | Positive integer; absent | Deterministic distinct-frame subsample per iter/round/role, capped at source length, with no replacement or duplication. Forbidden for NEB endpoint roles. |
| `sample.round_count` | Positive integer; required unless `rounds` | Repeats the declared seed/condition registry. |
| `sample.rounds` | Nonempty ordered object array; alternative to count | Each row optionally has `seeds` and `condition_groups`, nonempty unique known-ID arrays; omitted axis means all. Registry order, not row-list order, determines tasks; every round must have a nonempty seed/group intersection. |
| `sample.condition_groups.<id>.kind` | String; omitted means `native` | Sampling kind from the [type reference](sampling.md#kinds-and-required-fields). |
| `sample.condition_groups.<id>.seeds` | ID array; all declared roles | Restricts assigned roles. NEB binds exactly one. |

Ordinary `steps` and GCMC `cycles` accept a positive integer or an array matching the total round count, including inactive rounds. Condition-group `inputs` map native task-relative filenames to source paths relative to `run.json`; their exact fields are in the [sampling type reference](sampling.md#native-control-fields).

The alternative legacy layout uses `sample.systems.<id>.seed`, `sample.steps` and NVT/NPT settings inherited from sample defaults or overridden per system. Its native recipes are in `sampler.recipe`, or LAMMPS `sampler.recipes` with the effective `nvt`/`npt` keys. Explicit and legacy fields form separate accepted layouts.

## GCMC nested reference

GCMC fields, units, defaults and their combinations are maintained in the [sampling type reference](sampling.md#gcmc-nested-reference).

## Generated key appendix

See the [generated field inventory](appendices/configuration-fields.md) for loader key sets. The tables above define semantics and applicability.
