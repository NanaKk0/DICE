# Installation

## Requirements

The commands use a Linux terminal. Install DICE on a Linux workstation or server; Windows users can work in a WSL2 Linux environment.

Prepare:

| Component | Requirement |
| --- | --- |
| Python | 3.10 or newer for DICE |
| Environment manager | A separate environment created with `venv` or Conda is recommended |
| pip | Installs DICE and its Python dependencies |
| Git | Obtains the source from GitHub |

### Control and backend environments

DICE coordinates calculations; the selected backends perform model training, inference and sampling.

| Environment | Contents and purpose |
| --- | --- |
| DICE control environment | DICE and its Python dependencies; runs `dice`, organizes data and coordinates the workflow |
| Backend environment | The software and dependencies required by each teacher, student and sampling operation |

Teacher, student and sampling operations can use separate backend environments through an environment profile. Connect an existing working installation directly. The Python requirement above applies to the control environment; prepare each backend according to its own requirements.

### Compute resources

The DICE control program can run on a CPU. Training, inference and sampling requirements depend on the backend and task size. For GPU calculations, prepare drivers and CUDA according to the backend's official installation instructions.

## Install DICE

### 1. Obtain the source

Run these commands where you want to keep the checkout:

```sh
git clone https://github.com/NanaKk0/DICE.git
cd DICE
```

Run the remaining installation commands from this source directory.

### 2. Create and activate an environment

With Python's `venv`, first check that `python3` is version 3.10 or newer:

```sh
python3 --version
python3 -m venv .venv
. .venv/bin/activate
```

Alternatively, create a Conda environment:

```sh
conda create -n dice python=3.12 pip
conda activate dice
```

### 3. Install the package

In the activated environment:

```sh
python -m pip install --upgrade pip
python -m pip install .
```

pip installs DICE and its declared Python dependencies. Prepare the model and sampling backends separately, following the links below.

### 4. Check the installation

Inspect the installed package and command help:

```sh
python -m pip show dice-workflow
dice --help
```

`dice-workflow` is the package name; `dice` is the command. The help includes `run.json`, `--environment` and `--preflight`.

Activate this environment when using DICE in a new terminal.

## Connect backends and continue

Follow the [Quickstart](quickstart.md) to prepare the required backend capabilities and run the supplied example. See [Configure backend environments](../guides/backends.md) for backend combinations, activation scripts and invocation checks.
