# Models and native environments

The control environment runs DICE. Native training, teacher labelling and
student sampling run through separately installed and activated software profiles.

## Model sources and identities

| Role | Paper model identity | Upstream source |
| --- | --- | --- |
| Teacher foundation | DPA4-Neo, OMat24, `DPA4-Neo-OMat24-v20260627.pt` | [Author release](https://huggingface.co/OutisLi/DPA4-OMat24-v20260627/tree/main), including its matching JSON; [current community releases](https://huggingface.co/deepmodelingcommunity/DPA4-OMat24) |
| Student foundation | DPA4C-Nano, locally recorded as `DPA4C-Nano-zeng222-model.ckpt-1000000.pt` | [DPA4C official guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html) and [upstream source](https://github.com/deepmodeling/deepmd-kit). For the paper's exact historical checkpoint, request this version from the model authors. |

The filenames above identify the paper's foundation inputs. For a new study,
obtain a compatible checkpoint and matching native recipe from the model
project, and record both versions. Preserve the foundation's complete native
element map; DICE's active `type_map` specifies the target chemistry separately.

## Recorded maintained environment

The following stack was used in the maintained native environment.

| Component | Recorded version / build | Purpose |
| --- | --- | --- |
| Python | 3.11 | Native engine environment; DICE control tests separately used Python 3.10.20 |
| DeePMD-kit | 3.2.0b1.dev20260728, specialized DPA4/DPA4C development build | DPA4 training/readout and DPA4C training/export, including the model-specific SeZM exporter |
| PyTorch | 2.11.0+cu130 | Training and exported-model runtime |
| CUDA | 13.0 | Native GPU build/runtime |
| GCC | 12.4 | Native compilation |
| LAMMPS | 22Jul2025 Update 4, [base revision](https://github.com/lammps/lammps/commit/611ca3b7f8525ba802373b04f9e3d632d515f7e3) | DeepMD pair-style integration and sampling |
| Kokkos | 4.6.2, CUDA/Serial, Ada89 | Single-DPA4C GPU sampling |

This stack executed a DPA4-Neo/DPA4C-Nano Al–O iteration through external-DFT
selection. Native execution with this software snapshot requires its
model-specific SeZM export interface. Arrange access to a compatible build
before configuring the native roles; the software authors can advise on the
specialized historical exporter.

## CLI compatibility

This DICE snapshot accepts DeepMD `cli: "pt"` or `"tf"`. Its recorded DPA4 and
DPA4C adapters invoke `dp --pt`. In contrast, the current official
[DPA4C documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4c.html)
uses the PyTorch Exportable backend, `dp --pt-expt`. Configure this DICE
snapshot with `cli: "pt"` and a backend implementing its recorded adapters.
The upstream `pt-expt` interface requires a corresponding adapter integration.

For upstream installation instructions use the
[DeePMD installation guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html),
[DPA4 guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html)
and DPA4C guide above. Native use of this DICE snapshot requires a backend
implementing the recorded operation contracts. For historical replay, obtain
both the matching student foundation and the compatible specialized backend.

## Model files along the workflow

| File role | Consumer | Meaning |
| --- | --- | --- |
| Foundation checkpoint plus matching native recipe | Teacher/student train adapters | Initialization for `foundation_finetune` |
| Adapted teacher inference export | Teacher labelling | Energy/force evaluation; DPA4 also keeps its paired readout checkpoint for representations |
| Student inference export | Student readout and sample preparation | Canonical trained student; optional compression prepares the sampling representation |
| Sampling `.pt2` | Compatible LAMMPS/Kokkos | Compiled deployment artifact with target hardware/build requirements |

The paper's retained `.pt2` exports target Linux x86_64, compute capability
8.9 and AVX512 (some AVX512_VNNI). See the paper repository's `MODEL_USE.md`
and `MODEL_CARDS.json` before model evaluation. Use the backend's supported
export procedure when preparing a different model format or target architecture.

## Checks before scientific use

- Inspect the actual model's native type map and pair it with its matching recipe.
- Record `dp --version`, Python/PyTorch/CUDA versions and the LAMMPS build.
- Check training, export, teacher labelling/readout and student sampling as
  separate capabilities on your installation before a production campaign.
- The bundled CPU tests check configuration, role bindings and workflow
  behaviour; run the corresponding native checks on your installation.

Continue with [the complete workflow walkthrough](own-system.md) and
[environment bindings](environments.md).
