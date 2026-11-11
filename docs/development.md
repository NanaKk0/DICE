# Development and documentation maintenance

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
uses native-service doubles rather than installing PyTorch/DeePMD. Run native
engine acceptance separately on matching installations. Do not treat fixture
success as force-field accuracy or GPU scaling evidence.

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

The [documentation directory](index.md) supplies grouped navigation. Keep new
pages linked there. Preserve existing page URLs or leave a useful migration
entry. Configuration-key inventory is generated from the **published loader**,
not manually copied from another development branch.

The CI matrix runs on Python 3.10/3.12 and validates the CPU suite, inventory,
site and local links. The optional Pages workflow is manually triggered after
hosting is configured; changing Markdown does not itself enable hosting.

## Maintain source and publication separately

This is a curated public software export. In the maintainers' authoritative
development checkout, `docs/public/` mirrors the public documentation paths.
Maintain public text there and export explicitly selected files with
`scripts/export_public_docs.py`; release worktrees are publication targets.
That exporter and internal plans are development-side tools, not requirements
for an ordinary user cloning this repository.

Runtime fixes first belong in the authoritative source, with relevant tests.
Public documentation must describe the actual release implementation, including
its limits; unpublished development capabilities are not release promises.
Publish coherent changes through branches/PRs after checking the exact target
version and preserving other contributors' updates. Record user-visible changes
in [changelog](changelog.md).

## Third-party credit and licences

Cite engines and methods actually used: for example MACE, DeePMD-kit, NEP/GPUMD,
LAMMPS, ASE, CP2K and ABACUS. Their model/code/data licences apply independently.
Code is MIT and author-owned documentation/data/figures use CC BY 4.0;
see [LICENSING.md](../LICENSING.md).
