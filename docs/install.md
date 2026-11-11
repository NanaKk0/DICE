# Install DICE

Use Linux and Python 3.10 or newer. The DICE control package coordinates
calculations; training, inference and sampling engines are installed separately.
You can complete the CPU quickstart without CUDA, PyTorch or DeePMD.

## Obtain and install the control package

```sh
git clone https://github.com/NanaKk0/DICE.git
cd DICE
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
dice --help
```

The help output should mention `run.json`, `--environment`, `--preflight`,
`--probe` and `--resume`. If the command is absent, confirm that the same
Python environment was active during installation and invocation.

Keep these locations distinct:

| Location | Purpose |
| --- | --- |
| Checkout, such as `DICE/` | Software, documentation and example scripts |
| Control environment, such as `.venv/` | The installed `dice` command and Python dependencies |
| Scientific run directory | Your `run.json`, relative inputs, `iterN/` outputs and `.dice_state/` |

A run directory may be outside the checkout. Activate the control environment
before changing into it. [Try the CPU example next](quickstart.md).

## Development and reproducibility

For editable installation and the CPU regression suite:

```sh
python -m pip install -e '.[test]'
python -m pytest -q
```

The package declares NumPy, ASE, SciPy, PyYAML, scikit-learn, umap-learn and
Matplotlib. `requirements-tested.txt` records one bounded Python 3.10.20 CPU
acceptance environment. On that Python version you may install with
`python -m pip install -c requirements-tested.txt -e '.[test]'`.
These constraints do not describe a native-engine installation or a universal
lock for every Python version. Record the resolved versions used in your study.

The package version is `0.1.0`; also record the source commit because successive
software updates currently retain that package version.

## Prepare engines for scientific work

| Route | Install separately |
| --- | --- |
| MACE/MACE-LES teacher | Compatible Python environment and `mace_run_train` |
| DeepMD teacher/student | Matching TF/PT DeePMD runtime and `dp`; compatible LAMMPS for sampling |
| NEP student | `nep`, `gpumd`, and compatible TorchNEP for readout/calculator operations |
| Optional local FP labels | CP2K or ABACUS, its input, species assets and CPU resources |

Follow each engine's own installation instructions. Obtain genuine model
weights and their matching recipes separately. Bind the engines through an
[environment profile](environments.md), then verify the operations needed by
your route. Use [models and versions](native-models.md) and the
[capability matrix](support.md) before the [scientific walkthrough](own-system.md).
