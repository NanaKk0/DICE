# DICE

DICE (Distillation-driven Co-Evolution) coordinates active learning for
machine-learning interatomic potentials. A teacher labels structures, a
student explores configurations, and DICE selects candidates for new
first-principles (FP) calculations. Returned FP labels improve a later iteration.

| Your task | Start here |
| --- | --- |
| Install DICE and try the workflow without GPUs | [Installation](docs/install.md) → [CPU quickstart](docs/quickstart.md) |
| Run your own material with native engines | [Understand the workflow](docs/workflow.md) → [Your first scientific run](docs/own-system.md) |
| Check a field, backend or recovery command | [Documentation directory](docs/index.md) |

## Install and try

DICE's control environment requires Python 3.10 or newer on Linux. From a
shell with your chosen Python available:

```sh
git clone https://github.com/NanaKk0/DICE.git
cd DICE
python -m venv .venv
. .venv/bin/activate
python -m pip install .
dice --help
python examples/cpu_mock/run.py /tmp/dice-first-demo
```

Choose a new or empty demo directory. This example runs two rounds with
synthetic models, labels and trajectories. It checks the real workflow's
coordination and recovery; it does not train a scientific potential.
The [quickstart](docs/quickstart.md) explains its outputs and continuation.

## Run a scientific iteration

Install the engines required by your selected teacher, student and sampler
separately. Prepare labelled bootstrap data, starting geometries, compatible
models and native recipes. In the scientific run directory:

```sh
dice run.json --environment environment.json --preflight
dice run.json --environment environment.json
```

One invocation executes one iteration, containing the configured student rounds:

```text
teach/train → teach/screen → teach/scan (single teacher)
            → r0/train → r0/sample → r0/screen
            → r1/train → r1/sample → r1/screen → fp
```

The default endpoint is `iterN/fp/selected.xyz`, for external FP labelling.
A later invocation declares the returned labels in `fp_data`. An explicit
`fp.label` can instead append a supported local calculator operation.
See the [complete DPA4/DPA4C walkthrough](docs/own-system.md).

## Models and compatibility

Teachers use MACE/MACE-LES or DeepMD; students use DeepMD or NEP. DeepMD
students sample through LAMMPS, and NEP students through GPUMD. Exact model,
build and execution requirements are in the [capability matrix](docs/support.md)
and [model guide](docs/native-models.md).

The 2026-11-11 update adds newer DeepMD preset and ZBL/readout compatibility.
DPA4 retains strict short-distance numerical limitations; see the
[compatibility notes](docs/deepmd-compatibility.md) before collision studies.
Foundation weights and native engine binaries are obtained separately.

## Development and licences

Install `python -m pip install -e '.[test]'` to run `python -m pytest -q`.
See [development](docs/development.md) for documentation builds and testing.

Code is MIT; author-owned documentation, figures and data use CC BY 4.0.
Third-party components retain their own licences. See [LICENSING.md](LICENSING.md).
