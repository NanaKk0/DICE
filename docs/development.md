# Development, provenance and citation

This directory is a curated export of the authoritative development tree.
Runtime source snapshot: 2026-11-01;
publication-package reorganization: 2026-11-02. Runtime package
files are copied unchanged; distribution metadata, portable documentation,
test selection and the CPU demonstration are release presentation additions.
Apply runtime fixes to the authoritative development checkout, then regenerate
and revalidate a new distribution snapshot.

## Architecture

- `workflow_app`: public CLI, composition and application boundary.
- `workflow_core`: backend-neutral lifecycle and scientific orchestration.
- `workflow_runtime`: publication, recovery, observation and execution adapters.
- `workflow_artifacts`: typed and serialized artifacts.
- `workflow_backends`: native engine operations and evidence parsing.
- `workflow_policies`: screening, routing and representative selection.

The installed console script is `dice = workflow_app.entrypoint:console_main`.
These six packages form the runtime source closure. The distribution also
contains the portable documentation, tests and CPU examples listed above.

## Tests and documentation

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python tools/build_docs.py
```

The curated suite checks structure IO/identity, lifecycle state transitions,
selectors, pair guard, model/artifact handles, feature reuse and the portable
mock lifecycle. Native engine tests use separate installations, authorized
resources and study-specific acceptance criteria.

The documentation builder uses only Python's standard library and produces
static HTML under `_site/`. The included GitHub Pages workflow is manual and
uses native Pages artifacts; enable Pages with GitHub Actions only when ready
to publish. The builder and tests produce local outputs.

## Citation and third-party credit

When citing this software, identify DICE, this snapshot date and the actual
release identifier/archive record supplied by the authors at publication.
For the paper, use the final bibliographic record supplied by the authors.

Also cite the independently used engines and methods (for example MACE,
DeePMD-kit, NEP/GPUMD, LAMMPS, ASE, CP2K or ABACUS) according to their own
documentation and the actual calculation. Third-party code, model, data and
engine licences apply independently.

The authors will finalize software licensing and public release identifiers
before publication.
