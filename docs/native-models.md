# Models and native environments

The control environment runs DICE. Native training, teacher labelling and
student sampling use separately installed software profiles. Obtain foundation
weights and matching recipes from their projects; neither weights nor native
engine binaries are bundled with DICE.

## Model sources and identities

| Family | Sources |
| --- | --- |
| DPA4 | [Official guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html), [OMat24 community releases](https://huggingface.co/deepmodelingcommunity/DPA4-OMat24) |
| DPA4C | [Official guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html), [upstream source](https://github.com/deepmodeling/deepmd-kit) |
| Historical DPA4-Neo OMat24 | `DPA4-Neo-OMat24-v20260627.pt` and its matching JSON from the [author release](https://huggingface.co/OutisLi/DPA4-OMat24-v20260627/tree/main) |

Record the exact model identity and native recipe. Preserve the foundation's
complete native element map; DICE's active `type_map` specifies the target
chemistry separately. An explicit learned-child map in an analytical
composition must also preserve the foundation map and its order.

## Recorded native builds

| Component | Newer qualification, 2026-11-11 | Historical baseline |
| --- | --- | --- |
| Python | 3.11 | 3.11 |
| DeePMD-kit | Upstream revision `3b2555c46ecdbd2fcc8ddc029d09195a4e3c0f3a`; local version marker `3.2.0+g3b2555c46` | Specialized development build `3.2.0b1.dev20260728` |
| PyTorch | 2.11.0+cu130 | 2.11.0+cu130 |
| CUDA | 13.0 | 13.0 |
| LAMMPS | 22Jul2025 Update 4 | 22Jul2025 Update 4 |
| GPU | RTX 4090 | Ada89 build with Kokkos 4.6.2 CUDA/Serial |

The newer version marker identifies the tested local build, rather than an
official release with that name. Use the matching engine revision and model
release, set `DP_INTERFACE_PREC=high` in the activation script, and qualify the
Python/C++/LAMMPS stack together. This interface setting leaves native model
weights at their original precision. See the [installation guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html).

Historical model exports may require their original runtime or re-exporting;
arbitrary old `.pt2` packages are not guaranteed to load in the newer build.
The newer path uses upstream embedding APIs and does not require the
specialized historical exporter.

## CLI compatibility

Keep `cli: "pt"` in DICE's teacher/student declarations and environment roles.
Do not replace it with `cli: "pt-expt"` in `run.json`.

| Operation | Native selector |
| --- | --- |
| DPA4 teacher train / freeze | `dp --pt` |
| DPA4C student train / freeze / compress | `dp --pt-expt`, selected from the native recipe's family |

DPA4 does not support compression. A single DPA4C student selects Kokkos
LAMMPS sampling (`deepmd/kk`); a DPA4C committee uses ordinary DeepMD
LAMMPS and its native model-deviation output. Each route needs a compatible
LAMMPS installation. Kokkos failure does not trigger an ordinary fallback.

## Model files along the workflow

| File role | Consumer | Meaning |
| --- | --- | --- |
| Foundation checkpoint plus matching recipe | Train adapters | Initialization for `foundation_finetune` |
| Teacher `.pt2` inference export | Teacher labelling | Full-model energy, force and stress |
| Paired DPA4 `.pt` readout checkpoint | Teacher readout | Exact checkpoint selected for freezing; representations use its native atomic features |
| Student `.pt2` inference export | Student readout and sample preparation | Canonical trained student; optional DPA4C compression prepares the sampling model |
| Sampling `.pt2` | Compatible LAMMPS/Kokkos | Deployment artifact tied to the selected native runtime and hardware |

Read the [preset, ZBL, readout and qualification notes](deepmd-compatibility.md)
before using the newer models. Continue with the
[workflow walkthrough](own-system.md) and [environment bindings](environments.md).
