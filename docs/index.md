# Documentation

Start with the task you want to complete. Commands assume that you have
activated your DICE control environment; native engines use their own profiles.

## First use

| Read in order | What you will accomplish |
| --- | --- |
| [Install](install.md) | Obtain the software, install the control package and verify the CLI |
| [CPU quickstart](quickstart.md) | Run two synthetic rounds, inspect outputs and practise continuation |
| [Workflow](workflow.md) | Understand the data flow and the difference between rounds and iterations |
| [Your own system](own-system.md) | Prepare a real DPA4/DPA4C workflow and return external FP labels |

## Prepare and configure

| Guide | Use it for |
| --- | --- |
| [Data](data.md) | Label units, extxyz, validation populations and returned FP data |
| [Screening](screening.md) | Support scores, trust, quotas and representative selection |
| [Sampling](sampling.md) | Seeds, conditions, native protocols and advanced acquisition |
| [Configuration](configuration.md) | Scientific fields, defaults, applicability and boundary rules |
| [Accepted-key appendix](configuration-fields.md) | Automatically generated field inventory |
| [Environments](environments.md) | Engine activation and executable bindings |
| [Execution](execution.md) | Local/SSH/Slurm placement and operation-specific resources |
| [Models](native-models.md) | Foundation sources, native versions and model file roles |
| [DeepMD usage](deepmd.md) | Fine-tuning, exports, compression, ZBL and Kokkos |
| [DeepMD compatibility](deepmd-compatibility.md) | Presets, ZBL, readout and measured numerical limitations |

## Other engines and labels

| Guide | Use it for |
| --- | --- |
| [MACE/MACE-LES](mace.md) | Teacher recipes, initialization and operation bindings |
| [NEP/GPUMD](nep.md) | Student model/restart pairs, readout and GPUMD |
| [First principles](first-principles.md) | External returns and optional local CP2K/ABACUS labels |

## Operate and maintain

| Guide | Use it for |
| --- | --- |
| [Troubleshooting](troubleshooting.md) | Diagnose common input, model, environment and screening failures |
| [Recovery](recovery.md) | Continue after interruption or rebuild an explicitly selected suffix |
| [Capabilities](support.md) | Check backend requirements and qualification boundaries |
| [Development](development.md) | Run tests, build documentation and understand package responsibilities |

## Executable examples

- [CPU lifecycle](../examples/cpu_mock/README.md): synthetic backend services,
  complete iter0 and two rounds, ordinary continuation and explicit recovery.
- [FP return](../examples/iteration_handoff/README.md): synthetic external
  labels returned to iter1 in the same run directory.

Both examples are control-flow demonstrations. For scientific work supply
real labels, models, recipes and matching native installations.
