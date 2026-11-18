# DPA4 → DPA4C water example

Run iter0 and iter1 with an OMat24 DPA4-Neo teacher, a compressed DPA4C student
and two LAMMPS NVT rounds per iter. The supplied files include labelled water
data, sampling seeds, native recipes and both scientific configurations.
See [data/NOTICE.md](data/NOTICE.md) for data use and attribution.

Prepare DeePMD with DPA4 `--pt`, DPA4C `--pt-expt`, descriptor readout and
GPU compression, plus LAMMPS with Kokkos and `deepmd/kk`. See the
[backend environment guide](../../docs/reference/environment.md).

## Prepare

With DICE installed and its control environment active, enter this directory
from the checkout root:

```sh
cd examples/dpa4_dpa4c
python prepare_model.py
```

The downloader obtains the pinned official OMat24 DPA4-Neo v20260805 foundation.
An existing copy can be placed at
`models/DPA4-Neo-OMat24-v20260805.pt`; see [model details](models/README.md).

Edit the machine-specific files:

| File | Set for your installation |
| --- | --- |
| `activate-native.sh` | Conda installation, backend environment and visible GPU |
| `environment.json` | Activation scripts and `python`, `dp`, `lmp` executable names or paths |
| `execution.json` | Local target and GPU/CPU resource allocation |

The supplied execution profile uses one GPU and runs one task at a time.
The activation script preserves an inherited `CUDA_VISIBLE_DEVICES`, otherwise
selecting GPU 0.

## Run iter0

```sh
dice run.json --environment environment.json --execution execution.json --preflight
dice run.json --environment environment.json --execution execution.json
```

| Setting | Value |
| --- | --- |
| Initial references | The same 40 water frames for teacher and student, 192 atoms each |
| Teacher | DPA4-Neo foundation fine-tuning, 2500 steps |
| Student | DPA4C training from scratch, 5000 steps per round; frozen and compressed for sampling |
| Teacher training geometries | `student.include_teacher_train_data: true`; merged into the student population, deduplicated and labelled by the current teacher |
| Sampling | Two seeds × 300/600 K NVT, 2500 steps per task, 0.00025 ps timestep, save every 10 steps |
| Student prescreen | JOINT_DIRECT, pool 200 per round |
| Teacher GMM | Trust `p > 0.1`; FP candidate band `0 <= p < 0.1` |
| Training selection | DIRECT, up to 50 teacher-trusted structures per round |
| FP selection | DIRECT, up to 20 structures from both rounds |

Selection quotas are upper bounds; screening reports give the resulting
populations.

Read `iter0/status.log` for progress and `iter0/dice.log` for screening counts.
`iter0/r0/` and `iter0/r1/` contain training, sampling and screening outputs;
`iter0/fp/selected.xyz` contains structures selected for DFT labelling.

## Run iter1

The supplied `run_iter1.json` retains `run.json` and changes three fields:
`iter` is 1, `student_train_bootstrap` is empty, and `fp_data` names
`data/demo_iter1_labeled.extxyz`.

The 20 additional frames carry previously calculated DFT labels and demonstrate
the next-iter handoff. Production workflows ordinarily
calculate the structures selected in `iter0/fp/selected.xyz`; DICE also accepts
other compatible, externally supplied reference data through `fp_data`.

```sh
dice run_iter1.json --environment environment.json --execution execution.json --preflight
dice run_iter1.json --environment environment.json --execution execution.json
```

DICE trains the teacher on 40 initial plus 20 additional reference frames. With
`student.include_teacher_train_data` enabled in both configurations, it merges
the current teacher training geometries with the previous iter's student
population and relabels the union. Training data and screening reports give
the resulting population sizes. Read `iter1/dice.log` and
`iter1/fp/selected.xyz` for the new results. Preserve one configuration file per
iter when continuing with your own DFT data.

For your own system, see the [configuration reference](../../docs/reference/run-json.md),
[sampling guide](../../docs/reference/sampling.md) and [data guide](../../docs/reference/data.md).
