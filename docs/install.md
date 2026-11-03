# Installation

Install the control plane in its own Python 3.10+ environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
dice --help
```

For development, use `python -m pip install -e '.[test]'`, then
`python -m pytest -q`. The test suite exercises the DICE control layer with
CPU fixtures. The documented operating baseline is Linux. Check each native
engine on the hardware and platform selected for your calculations.

The Python package declares NumPy, ASE, SciPy, PyYAML, scikit-learn,
umap-learn and Matplotlib for the control layer. Save resolved versions for each study.
The distribution preserves upstream version `0.1.0`; the snapshot date
identifies this source distribution.

`requirements-tested.txt` records the versions used for this export's bounded
CPU acceptance on Python 3.10.20. To use those constraints, install with
`python -m pip install -c requirements-tested.txt -e '.[test]'`.
Configure native backend environments separately as described below.

## External engines

For a DPA4/DPA4C workflow, start with the model sources,
recorded native stack and CLI compatibility boundaries in
[Models and native environments](native-models.md).

Install each engine, its runtime libraries and required model or species data
using the corresponding project's instructions and licence.

- MACE/MACE-LES teacher operations need a compatible MACE Python environment
  and the configured `mace_run_train` executable.
- DeepMD teacher/student operations need the matching TensorFlow or PyTorch
  DeePMD-kit installation and `dp`; their sampling needs compatible LAMMPS.
- NEP training/sampling needs NEP/GPUMD executables; descriptor/calculator
  operations need the compatible TorchNEP Python environment.
- Optional same-host FP labeling needs the selected CP2K or ABACUS executable,
  its native recipe, species data and sufficient CPU resources.

Select backend environments by operation roles in an external environment
profile. Use the [capability matrix](support.md) to identify the required native
operations, then check them with your model and installation.

See [environment and execution profiles](environments.md) before running real
calculations.
