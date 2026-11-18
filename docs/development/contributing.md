# Contributing

DICE's installed console entry is
`dice = workflow_app.entrypoint:console_main`. The public runtime is organized
as six packages:

| Package | Responsibility |
| --- | --- |
| `workflow_app` | Public CLI and application composition |
| `workflow_core` | Backend-neutral lifecycle/scientific orchestration |
| `workflow_runtime` | Execution, observation, publication and recovery |
| `workflow_artifacts` | Typed and serialized outputs |
| `workflow_backends` | Native engine operations and parsing |
| `workflow_policies` | Screening and representative selection |

## Test the control package

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python tools/generate_config_reference.py --check
```

The CPU suite covers canonical IO/identity, scientific configuration, public
examples, model assets, selected native-service contracts and recovery. It
uses native-service doubles for control-flow checks. Changes to native adapters also need focused acceptance with the matching engines.

## Build and check documentation

```sh
python -m pip install -e '.[docs]'
python tools/generate_config_reference.py --check
python tools/build_docs.py
python tools/check_docs.py --site _site
python -m http.server --directory _site 8000
```

Open `http://localhost:8000/`. The documentation extra supplies Python-Markdown;
the site uses its fenced code, tables, heading anchors and list support. Local
file/heading links and generated HTML assets are checked. Network URLs are not
fetched by the checker. `_site/` is generated output, not source content.

The [documentation directory](../index.md) supplies grouped navigation. Keep new
pages linked there. Preserve existing page URLs or leave a useful migration
entry. Configuration-key inventory is generated from the loader in the target release.

The CI matrix runs on Python 3.10/3.12 and validates the CPU suite, inventory,
site and local links. The optional Pages workflow is manually triggered after
hosting is configured; changing Markdown does not itself enable hosting.

## Contribute a change

Create a branch for the change and keep related source, tests, examples and user documentation together. Add a regression test for behavior changes. Describe the problem, resulting behavior and validation in the pull request; record user-visible changes in the [changelog](../../CHANGELOG.md).

Documentation follows the reading tasks:

| Page type | Content |
| --- | --- |
| Introduction | What DICE does and the main workflow |
| Tutorial | One complete example, required files, commands and output locations |
| How-to guide | Steps to change one aspect of a working run |
| Concept | Model/data relationships and the meaning of an algorithm |
| Reference | Accepted fields, types, units, defaults and constraints |

Link related material to its owning page. Keep backend installation details in upstream documentation and DICE-specific integration in backend guides. The generated configuration-key appendix is maintained by `tools/generate_config_reference.py`.

For each addition, identify the reader's task and the decision or operation the
text helps them complete. Use project-relative paths or neutral placeholders
for installation paths, hosts and resources. Examples describe inputs,
configuration, commands and output roles. Keep machine-specific builds,
experiment measurements and validation histories in maintainer records.

## Third-party credit and licences

Cite engines and methods actually used: for example MACE, DeePMD-kit, NEP/GPUMD,
LAMMPS, ASE, CP2K and ABACUS. Their model/code/data licences apply independently.
Code is MIT and author-owned documentation/data/figures use CC BY 4.0;
see [LICENSING.md](../../LICENSING.md).

## Developer fixtures and agent skill

The [CPU lifecycle fixture](../../tests/fixtures/cpu_mock/README.md) checks lifecycle coordination and rebuilding. The [handoff fixture](../../tests/fixtures/iteration_handoff/README.md) checks label return and next-iter ingestion. Their synthetic services keep these regression checks independent of native engines.

The [DICE Operator skill](../../skills/dice-operator/README.md) is distributed separately from its tests in `tests/skills/dice_operator/`. Configuration keys are listed in the [generated appendix](../reference/appendices/configuration-fields.md).
