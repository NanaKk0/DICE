# Capabilities and requirements

Select a workflow from the matrix below and prepare its corresponding native
environment. Compatibility depends on engine revision, model family, recipe,
chemistry, hardware and execution mode. The bundled automated tests cover
the control layer using CPU fixtures.

| Surface | Operations | Requirements and checks |
| --- | --- | --- |
| Public iter/round lifecycle | Canonical publication, role composition, screening, exact suffix recovery | Curated CPU tests and synthetic full-lifecycle example |
| MACE/MACE-LES teacher | Train, label and compatible single-model readout | Compatible MACE/MACE-LES environment and training executable |
| DeepMD TF/PT teacher/student | Role-specific training, inference, datasets and selected family readout | Matching model family, DeePMD backend and sampling installation |
| DeepMD DPA4 | Train, foundation finetune preserving native type map, paired SeZM graph export/ASE label/readout | Compatible model-specific SeZM exporter; see native-models.md |
| DeepMD DPA4C | Train, finetune, freeze, compression, ordinary/Kokkos LAMMPS and descriptor/GMM paths | Recorded working stack in native-models.md; check the selected build |
| NEP/GPUMD | Student training, sampling, TorchNEP-compatible descriptor/calculator operations | Compatible NEP/GPUMD executables and TorchNEP environment |
| ASE NEB/relax | DeepMD TF/PT and NEP calculator paths with closed condition schemas | Native installation/calculator qualification required; block-level recovery only |
| GCMC/Widom | DeepMD/LAMMPS-selected acquisition routes | Acquisition sampling; evaluate thermodynamic convergence for the application |
| Fixed-composition MD/MC | DeepMD/LAMMPS identity-swap NVT/NPT acquisition | Short native DPA4C checks available; qualify additional model/execution combinations separately |
| Optional CP2K label | Same-host CPU static ENERGY_FORCE | CP2K executable, recipe, species assets and CPU resources |
| Optional ABACUS label | Same-host static scf adapter | Adapter has fixture tests; native execution requires an installation-specific check |
| Local/SSH/Slurm execution | Explicit profiles, resources and durable operational receipts | Configure targets and resources, then check deployment on the selected site |
| MPS/multi-GPU/distributed modes | Selected operation-specific opt-ins | Use supported operation/profile combinations on authorized hardware |

For committee uncertainty, use independently trained members. Before production,
check numerical behaviour, memory requirements and dynamics under the intended
simulation conditions.

## Reporting a problem

Provide the snapshot date, Python/dependency versions, engine version/build,
model family, operating system, selected operation roles, redacted profiles,
minimal `run.json`, failed block and relevant diagnostic log excerpt. State
whether it reproduces with the CPU mock or only a native engine. Remove
credentials, private data and unreleasable model assets before sharing.

The public repository's issue tracker will provide the reporting location
when the software is published.
