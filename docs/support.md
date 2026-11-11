# Capabilities and qualification

Choose a supported route, then verify its exact native model/build and scientific
regime. The public Python 3.10/3.12 suite validates control behaviour and examples.
Native engine checks are separate, model-specific evidence. Neither type of
execution test replaces independent scientific accuracy or convergence tests.

| Route | Control coverage / public interface | Native scope and limits | Guide |
| --- | --- | --- | --- |
| Iter/round lifecycle | Dataset/artifact validation, screening, recovery, CPU iter0 and iter1 examples | Native operations require their selected installations | [Workflow](workflow.md) |
| DPA4 teacher | Preset/family routing, foundation typemap, paired readout, native command contracts | Recorded Nano OMat24/OMol25 fine-tuning/freeze/readout; short-distance DPA4 consistency remains limited | [DeepMD](deepmd.md), [qualification](deepmd-compatibility.md) |
| DPA4C student | Exportable dispatch, seeds, ZBL composition, readout and model assets | Recorded fine-tuning/freeze/compression and ordinary/Kokkos sampling on the specified build | [Models](native-models.md) |
| MACE/MACE-LES teacher | Native train/label/readout adapters and synthetic lifecycle route | Compatible model/head/readout and native installation qualification required | [MACE](mace.md) |
| TF DeepMD | TF role bindings, `.pb` artifacts and native command contracts | Matching TF/DeepMD/LAMMPS stack; single-student GMM uses `se_e2_a` | [DeepMD](deepmd.md) |
| NEP/GPUMD | NEP model/restart, GPUMD sampling and readout/calculator interfaces | Compatible NEP4/TorchNEP/GPUMD needed for the selected operations | [NEP](nep.md) |
| NVT/NPT and programs | Closed condition schemas, task/group expansion and recipe contracts | Program/cell variants have LAMMPS-specific contracts | [Sampling](sampling.md) |
| NEB/relax | ASE calculator routes, fixed geometry constraints and whole-block recovery | Qualify optimizer, convergence and selected model locally | [Sampling](sampling.md) |
| MD/MC, insertion, GCMC | Acquisition routes and typed configuration | Native route-specific requirements; no blanket thermodynamic convergence claim | [Sampling](sampling.md) |
| Optional CP2K/ABACUS | Recipe, parser and label-publication interfaces | Matching static calculator inputs/assets; ABACUS native execution remains installation-specific | [FP labels](first-principles.md) |
| Local/SSH/Slurm | Explicit profiles and operational receipts | Site-specific staging/resources and role constraints | [Execution](execution.md) |
| MPS/multi-GPU/distributed | Operation-specific resource-shape constraints | Assignment fixtures do not establish physical multi-GPU scaling | [Execution](execution.md) |

## Recorded newer DeepMD qualification

The newer build is pinned in [models](native-models.md). OMat24 DPA4/DPA4C
completed the recorded two-round YSZ workflow, including compressed Kokkos
sampling and continuation. Independent OMol25 checks cover system charge/spin
conditioning, not magnetic-spin dynamics. Some short-distance DPA4 comparisons
failed strict parity. See the [detailed scope](deepmd-compatibility.md), including
unchanged student dataset membership and absent new DFT labels in that run.

This matrix distinguishes implemented interfaces from specific native tests;
it does not advertise every combination of backend, checkpoint, hardware and
scientific regime as validated. Check the inputs needed by your selected route
and assess model accuracy with independent reference data.

## Report a problem

Use [troubleshooting](troubleshooting.md), then provide the source commit,
Python/dependency/native versions, model family, failed block, minimal config
and relevant redacted logs in the [issue tracker](https://github.com/NanaKk0/DICE/issues).
State whether the CPU example reproduces it. See [recovery](recovery.md) for
continuation and deliberate rebuild commands.
