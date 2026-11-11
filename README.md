# DICE

DICE (Distillation-driven Co-Evolution) coordinates teacher-guided,
student-driven active learning for machine-learning force fields. A teacher
labels and assesses structures, a student explores configurations, and DICE
selects structures for the next first-principles labeling cycle.

[Get started](examples/cpu_mock/README.md) ·
[User guide](docs/workflow.md)

Official repository: [DICE software](https://github.com/NanaKk0/DICE)

This software snapshot includes the **2026-11-11 DeepMD compatibility update**
to the 2026-11-02 distribution. It contains the DICE runtime, user documentation,
configuration guides and CPU examples. See the [DPA4/DPA4C compatibility notes](docs/deepmd-compatibility.md)
for native presets, ZBL, readout and the known short-range numerical limits.

## Install

Use Python 3.10 or newer in a separate control-plane environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
dice --help
python -m pytest -q
```

Installation supplies DICE and its Python control-plane dependencies. Install
the engines needed by your workflow—MACE, DeePMD-kit, NEP/GPUMD, LAMMPS,
CP2K or ABACUS—in separate environments and connect them through an
environment profile. The baseline operating platform is Linux;
native backend, CUDA and scheduler compatibility is installation-specific.

## Try the CPU-only mock example

```sh
python examples/cpu_mock/run.py /tmp/dice-mock-demo
python examples/cpu_mock/run.py /tmp/dice-mock-demo --reuse
```

The first command creates tiny synthetic inputs in a new directory and runs
the real lifecycle with injected fake backend services. The second verifies
ordinary continuation of its completed result. All generated models, labels
and trajectories are synthetic CPU fixtures for learning the workflow.
See [the example guide](examples/cpu_mock/README.md).

## Run a scientific workflow

The sole public workflow interface is `dice run.json`:

```sh
cd /path/to/your/run-directory
dice run.json --environment environment.json --preflight
dice run.json --environment environment.json
```

One invocation executes one iteration:

```text
teach/train -> teach/screen -> teach/scan (single teacher only)
-> r0/train -> r0/sample -> r0/screen -> ... -> fp
```

The default terminal artifact is `iterN/fp/selected.xyz` for external labeling.
Return labeled extxyz through `fp_data` in a later iteration. Explicit
`fp.label` instead appends one optional same-host `fp/label` block.

For your own material, follow [Run DICE on your own system](docs/own-system.md).
It includes a complete scientific configuration, PT environment bindings,
native-recipe responsibilities and the external-DFT return step. Consult
[model sources and native versions](docs/native-models.md) first.
Obtain pretrained DPA4/DPA4C models from their official projects when selecting
foundation fine-tuning, and prepare native recipes for your selected models
and material.

## Documentation

- [Installation and backend boundaries](docs/install.md)
- [Workflow and artifacts](docs/workflow.md)
- [Scientific configuration](docs/configuration.md)
- [Environment and execution profiles](docs/environments.md)
- [Recovery and diagnostics](docs/recovery.md)
- [Support and qualification matrix](docs/support.md)
- [Development and third-party credit](docs/development.md)

Build the dependency-free documentation website with
`python tools/build_docs.py`; open `_site/index.html`, or serve `_site/` with
`python -m http.server --directory _site 8000`. The optional GitHub Pages
workflow is manually triggered after the authors configure hosting.

## Licences

Source code is licensed under MIT. Author-owned data, figures and documentation
are licensed under CC BY 4.0.
See [LICENSING.md](LICENSING.md) for scope and attribution.
