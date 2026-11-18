# DICE documentation

Start with **Introduction → Installation → Quickstart**. The supplied water example covers the first iter, DFT-label return and the next iter.

## Get started

| Page | Task |
| --- | --- |
| [Introduction to DICE](getting-started/introduction.md) | Understand the workflow and preparation checklist |
| [Installation](getting-started/installation.md) | Install DICE and check its CLI |
| [Quickstart](getting-started/quickstart.md) | Run the water example and continue iterating |

## How-to guides

Choose a guide for the aspect of your run you want to change.

| Page | Task |
| --- | --- |
| [Configure backend environments](guides/backends.md) | Activation scripts, executables and operation bindings |
| [Prepare input files](guides/inputs.md) | Data, seeds, native recipes and optional models |
| [Configuration overview](guides/configuration.md) | Find each setting's owner and detailed guide |
| [Configure compute resources](guides/resources.md) | Placement, budgets and concurrency |
| [Configure sampling](guides/sampling.md) | Native protocols, overrides, batch conditions, seeds and rounds |
| [Inspect results and resume](guides/results-and-recovery.md) | Locate outputs, continue or rebuild |
| [Troubleshooting](guides/troubleshooting.md) | Diagnose configuration, data and task failures |

### Model and labeling backends

| Page | Topic |
| --- | --- |
| [Use DeePMD-kit](guides/deepmd.md) | Fine-tuning, element maps, exports and student compression |
| [MACE](guides/experimental/mace.md) | Teacher integration and environment bindings |
| [NEP/GPUMD](guides/experimental/nep.md) | Student training, readout and sampling |
| [CP2K/ABACUS labeling](guides/experimental/cp2k-abacus.md) | Built-in labeling and result interfaces |

## Concepts

| Page | Topic |
| --- | --- |
| [Workflow](concepts/workflow.md) | Iterations, rounds, teacher updates and data flow |
| [Configuration screening](concepts/screening.md) | GMM support, committee evidence, data routes and representative selection |

## Reference

| Page | Contents |
| --- | --- |
| [CLI](reference/cli.md) | Options, stages, preflight and recovery |
| [run.json](reference/run-json.md) | Scientific fields, defaults, units and constraints |
| [Environment profile](reference/environment.md) | Activation, logical executables and operation bindings |
| [Execution profile](reference/execution.md) | Targets, pools, resource classes and bindings |
| [Data format](reference/data.md) | Extended XYZ labels, units, elements and array shapes |
| [Sampling types](reference/sampling.md) | Mode-specific configuration, inputs and observations |

## Development and contributing

| Page | Contents |
| --- | --- |
| [Contributing](development/contributing.md) | Development, package responsibilities, tests and documentation |

See the project [README](../README.md). Report reproducible issues through [GitHub Issues](https://github.com/NanaKk0/DICE/issues), including relevant configuration, stage and log excerpts.
