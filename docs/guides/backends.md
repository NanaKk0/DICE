# Configure backend environments

Connect installed software to DICE and check environment bindings and program invocation. First complete [Installation](../getting-started/installation.md) and install the selected engines using their official instructions.

DICE loads the environment assigned to each operation. The teacher, student and sampler can use different installations. This guide uses one DPA4 teacher, one DPA4C student and LAMMPS sampling.

## Choose model and sampling backends

The recommended combination is a **DPA4 teacher + DPA4C student**, sampled with LAMMPS.

| Route or interface | Status | Settings |
| --- | --- | --- |
| DPA4 teacher + DPA4C student with LAMMPS | Recommended | [Use DeePMD-kit](deepmd.md) |
| MACE/MACE-LES teacher | Experimental | [MACE](experimental/mace.md) |
| NEP student with GPUMD | Experimental | [NEP/GPUMD](experimental/nep.md) |
| Built-in CP2K/ABACUS labeling | Experimental | [CP2K/ABACUS labeling](experimental/cp2k-abacus.md) |

## Capabilities for the DPA4/DPA4C route

| Software | Role | Installation |
| --- | --- | --- |
| DeePMD-kit Python API and CLI | Training, labeling, descriptor export, freezing and DPA4C compression | [DeePMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html) |
| DeePMD-kit C++ interface and LAMMPS integration | Evaluating the student in LAMMPS | [C++ interface](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-from-source.html), [LAMMPS integration](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-lammps.html) |
| LAMMPS | Sampling with the student model | [LAMMPS](https://docs.lammps.org/Install.html) |

Choose a DeePMD-kit version supporting both DPA4 and DPA4C, and prepare its required PyTorch dependencies. DPA4 uses the PyTorch route; DPA4C training, freezing and compression require the corresponding `--pt-expt` route. See the official [DPA4](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html) and [DPA4C](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html) model documentation.

The example uses `deepmd/kk` for a single DPA4C student, so LAMMPS needs Kokkos and the matching DeepMD interface. Use compatible Python, C++ and LAMMPS installations, with the drivers and CUDA required by the chosen GPU build.

## Prepare the connection files

The example supplies an activation script and an environment profile. From the DICE checkout:

```sh
cd examples/dpa4_dpa4c
```

For a separate run directory, prepare the equivalent files there.

| File | Configure |
| --- | --- |
| [activate-native.sh](../../examples/dpa4_dpa4c/activate-native.sh) | Backend activation and required environment variables |
| [environment.json](../../examples/dpa4_dpa4c/environment.json) | Operation bindings and the `python`, `dp` and `lmp` executables |

## Edit the activation script

Set `activate-native.sh` to load your backend installation. For example, with the required engines in a Conda environment named `dice`:

```bash
#!/usr/bin/env bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate dice

export DICE_LAMMPS_PARALLEL_TASKS=1
```

Replace `/path/to/miniconda3` and the environment name with your installation. You can also source an activation script supplied by that installation.

`DICE_LAMMPS_PARALLEL_TASKS=1` makes this example call LAMMPS one task at a time. Put any required executable-search and shared-library settings in the activation script as well.

## Edit the environment profile

The template binds model operations to `models` and LAMMPS sampling to `sampling`. Both initially use `activate-native.sh`.

Check:

- `environment.source_list`: activation-script paths, relative to `environment.json` when relative.
- `executables`: program names or absolute paths after activation. For a program named `lmp_mpi`, bind the logical `lmp` executable to `lmp_mpi`.
- `runtime.batch_size`: the descriptor atom budget per call. The template uses `9472` total atoms.

For separate model and sampling installations, assign an activation script to each profile. DICE loads the environment needed by the current operation.

Select the profile with `--environment environment.json`. Scientific inputs belong in `run.json`; compute placement and resources are described in [Configure compute resources](resources.md).

## Verify backend invocation

Use the filenames and program names matching your installation.

### 1. Check activation

From the example directory, load the backend in a subshell and inspect the programs:

```bash
(
    source ./activate-native.sh || exit 1
    command -v python
    command -v dp
    command -v lmp
    python -c 'import sys; print(sys.executable)'
)
```

The paths should resolve to your backend installation. Replace `lmp` with `lmp_mpi` when appropriate, consistently with the environment profile. The subshell leaves the parent terminal's DICE environment active.

For separate backend environments, check each activation script with its required programs.

### 2. Check backend capabilities

Inspect DeePMD and both required PyTorch command routes:

```bash
(
    source ./activate-native.sh || exit 1
    dp --version
    dp --pt --help
    dp --pt-expt --help
)
```

DPA4 uses `--pt` and DPA4C uses `--pt-expt`; DICE selects the route from the native model configuration.

Inspect PyTorch and GPU visibility:

```bash
(
    source ./activate-native.sh || exit 1
    python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
)
```

For an NVIDIA GPU route, `CUDA available` should be `True`. CPU configurations can report `False`.

Inspect LAMMPS capabilities:

```bash
(
    source ./activate-native.sh || exit 1
    lmp -help
)
```

The single-DPA4C example requires Kokkos and `deepmd/kk`. Built-in or automatically loaded plugins expose their packages and pair styles in the help. For manually loaded plugins, follow the [official integration instructions](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-lammps.html) and check the required pair style after loading.

Use the model activation script for DeePMD/PyTorch checks and the sampling script for LAMMPS when those installations differ.

### 3. Run DICE preflight

The example includes the data, training recipes and sampling protocol referenced by `run.json`. Check its execution profile as described in the [Quickstart](../getting-started/quickstart.md), then run from the example directory with the control environment active:

```sh
dice run.json --environment environment.json --execution execution.json --preflight
```

Preflight checks configuration, external inputs, operation bindings, activation scripts and executables. Successful output uses schema `dice.lifecycle.preflight`:

| Field | Check |
| --- | --- |
| `blocks` | Planned workflow stages |
| `operation_roles` | Training, labeling, screening and sampling operations |
| `operation_environment` | Environment, activation script and program for each operation |

Confirm that the reported paths match your installation. Resolve any reported input or environment error and repeat preflight. Continue with the [Quickstart](../getting-started/quickstart.md) to run the workflow.
