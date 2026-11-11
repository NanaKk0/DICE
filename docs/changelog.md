# Changes and version scope

Record the source commit alongside package version `0.1.0`: the current
source-distribution updates retain that package version. Engine revisions and
model-release identifiers are separate version axes.

## 2026-11-12: documentation renovation

- A continuous installation/CPU/real-system path, task-oriented directory and
  clearer workflow/data explanations.
- Portable DPA4/DPA4C PT profile and DeepMD usage separated from qualification.
- Screening and sampling guides, MACE/NEP/FP entry points, execution and recovery
  guidance, and symptom-oriented troubleshooting.
- Markdown rendering, heading/file link checks and grouped local navigation.

These changes document the published software. They do not change scientific
fields, defaults, native training policies, screening or recovery semantics.
GitHub Pages remains an optional separately configured deployment.

## 2026-11-11: newer DeepMD integration

[PR #1](https://github.com/NanaKk0/DICE/pull/1) added native preset/analytical ZBL
family routing, learned-component seeds/type-map checks, full-model atomic
embedding readout and automatic neighbour graphs. It retained compatibility
fallbacks for older readout APIs.

DPA4 short-distance export/batch differences remain unresolved; see
[compatibility](deepmd-compatibility.md). Recorded native GPU acceptance and
public CPU regression cover different capabilities.

## 2026-11-02/03: public distribution baseline

The curated software distribution supplied the iter/round runtime, portable
CPU lifecycle and FP-return demonstrations, documentation and licences.
The software repository's subsequent scope keeps paper-specific resources
outside its public software documentation. Historical exports identify their
own preparation dates; use the intended source revision for reproduction.
