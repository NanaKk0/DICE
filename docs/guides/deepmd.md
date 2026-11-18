# Use DeePMD-kit

The recommended combination is a DPA4 teacher and a DPA4C student with LAMMPS sampling. This guide covers model initialization, native recipe integration, model files and deployment through DICE.

## Prepare the native installation

Prepare DeePMD-kit and its compatible LAMMPS integration using the [upstream installation instructions](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html). The Python and C++ installations must support the model family and exported model format used by the run.

Use the [PT profile template](../../examples/profiles/dpa4c_dpa4.json) to bind the DICE operations. Set its activation scripts and executable paths for your installation. The [walkthrough](../getting-started/quickstart.md) shows these files in a complete run; the [environment reference](../reference/environment.md) defines their fields.

## Initialization and native recipes

Choose the initialization policy separately in `teacher` and `student`:

| Policy | DICE inputs | Initialization of each executed train block |
| --- | --- | --- |
| `from_scratch` | Native training `recipe` | Initialize the model from scratch and train on the current dataset |
| `foundation_finetune` | Native `recipe` and `foundation_model` checkpoint | Initialize from the declared checkpoint and fine-tune on the current dataset |

Each train block applies its declared policy. Reusing a foundation checkpoint across rounds starts each fine-tuning job from that checkpoint with the newly accumulated dataset. Ordinary workflow continuation reuses completed training results; deliberate rebuilding follows the [recovery commands](results-and-recovery.md).

For fine-tuning, keep the foundation model's full native element map and order in the native recipe. DICE's `type_map` names the ordered active elements of the target system. The active map can be an ordered unique subset of the foundation map. Scratch training uses the active map as the native map.

Model releases supply the foundation checkpoint and its matching native recipe. Model definitions and release links are in the official [DPA4](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html) and [DPA4C](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html) guides.

DICE supplies dataset paths and member seeds. Keep architecture and optimization settings in the native recipe. New recipes can use `model.preset`, such as `dpa4-nano-v20260911` or `dpa4c-nano-v20260911`; DeePMD expands the preset, and DICE selects the matching native operations.

## Model files and deployment

Keep `cli: "pt"` in DICE for both DPA4 and DPA4C. DICE selects the native command family:

| Role or setting | Native path | Model used by DICE |
| --- | --- | --- |
| DPA4 teacher | `dp --pt` training and freeze | `.pt2` for labels; the paired final `.pt` checkpoint for descriptor readout |
| DPA4C student | `dp --pt-expt` training and freeze | Uncompressed `.pt2` for student readout |
| DPA4C with `student.compress: true` | Sampling-stage compression | Derived compressed model for sampling; uncompressed model retained for readout |
| TF student with `cli: "tf"` | TensorFlow training and freeze | `.pb`, with a matching TF LAMMPS integration |

Compression is available for compatible student model families, including DPA4C. DPA4 uses its uncompressed model path. DICE runs the required freeze and compression operations and publishes their model files.

A single DPA4C student samples with Kokkos `pair_style deepmd/kk`. A DeepMD student committee uses ordered member models with ordinary `pair_style deepmd` and native force-deviation output. Prepare LAMMPS for the selected route. Locations of models and native logs are described in the [run results walkthrough](../getting-started/quickstart.md).

## Enable ZBL

In a compatible native recipe, this model fragment enables analytical ZBL:

```json
{
  "model": {
    "preset": "dpa4c-nano-v20260911",
    "bridging_method": "zbl"
  }
}
```

Use the corresponding DPA4 preset for a teacher. Keep the ZBL settings when continuing from a checkpoint that contains them. The recognized composed form contains exactly one learned DPA4/DPA4C child and one `inner_potential` child.

Supply complete reference energies and forces in the training data. DICE uses the complete model for labels and deployment, and exports its learned atomic features for GMM and descriptor-based selection. Configure LAMMPS with the model once; its ZBL contribution is already included. Native ZBL parameter definitions belong to DeePMD's model configuration.

## Precision and model conditioning

DICE uses the activated backend environment and the native model's precision settings. `DP_INTERFACE_PREC` controls DeePMD interface precision; the model's internal precision is specified by the native model configuration. Set these in the backend environment and recipe as needed by your installation.

For charge/spin-multiplicity-conditioned checkpoints such as OMol25, the LAMMPS route uses the model's stored default conditioning. Runtime overrides of those conditioning values are outside the current DICE sampling interface.

## Model sources

| Family | Source |
| --- | --- |
| DPA4 | [Official guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html), [OMat24 releases](https://huggingface.co/deepmodelingcommunity/DPA4-OMat24) |
| DPA4C | [Official guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html), [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) |

Use a checkpoint and recipe from the same release and follow its model-card licence.
