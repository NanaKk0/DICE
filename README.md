# DICE

`DICE (Distillation-driven Co-Evolution)` is a closed-loop active-learning
workflow for machine-learning force fields. The shipped workflow uses a
teacher-guided, student-driven loop: by default a MACE / MACE-LES teacher
committee labels and filters candidate structures, while a DeepMD student
committee performs low-cost sampling and uncertainty screening. DeepMD DPA-4 is
also supported as an upstream teacher backend through the
`teacher_backend = "deepmd"` configuration path.

This repository is the `v0.1.0` release surface. It contains the runnable
workflow code, example pipeline configuration, user asset directories, and the
four top-level documents needed to operate and understand the current mainline.

## Current Mainline

The formal top-level workflow is:

- `workflow.py`
  - `committee_cycle`
    - MACE / MACE-LES teacher committee finetune by default
    - DeepMD active-learning student subworkflow

The teacher backend is configurable. The default example uses:

- [`configs/pipelines/committee_cycle/teacher_committee_finetune/mace/committee_finetune.json`](./configs/pipelines/committee_cycle/teacher_committee_finetune/mace/committee_finetune.json)

The release also ships a DeepMD DPA-4 teacher option:

- [`configs/pipelines/committee_cycle/teacher_committee_finetune/deepmd/dpa4/committee_finetune.json`](./configs/pipelines/committee_cycle/teacher_committee_finetune/deepmd/dpa4/committee_finetune.json)
- [`configs/pipelines/teacher_foundation_train/deepmd/dpa4/foundation_train.json`](./configs/pipelines/teacher_foundation_train/deepmd/dpa4/foundation_train.json)

The example top-level workflow file is:

- [`Pt-water-interface_roundN.yaml`](./Pt-water-interface_roundN.yaml)

The example pipeline configuration selected by that YAML is:

- [`configs/pipelines/committee_cycle/pipeline.json`](./configs/pipelines/committee_cycle/pipeline.json)

At this release stage, first-principles calculations are usually completed
outside the workflow and then backfilled through `user_assets/dft/rX/labeled.xyz`.
The codebase also includes CP2K backend plumbing for internal first-principles
execution, but that path should be treated as optional unless your environment
has been configured and validated for it.

## Quick Start

On the target Linux server, launch the formal workflow with:

```bash
python workflow.py Pt-water-interface_roundN.yaml
```

To keep a server run alive after disconnecting, wrap the command with your local
process manager, for example:

```bash
nohup python workflow.py Pt-water-interface_roundN.yaml > logs/pt-water-interface_r1_1.log 2>&1 &
```

## User Inputs

User-maintained input data lives under `user_assets/`.

Common inputs are:

- `user_assets/foundation_models/*.model`
  - MACE foundation model used for teacher finetune
- `user_assets/teacher_finetune/*.xyz`
  - teacher finetune base datasets, such as `base.xyz`
- `user_assets/student_initial/*.xyz`
  - initial student relabel / train inputs, such as `bootstrap.xyz`
- `user_assets/seed_pool/*.xyz`
  - initial seed pool inputs, such as `initial_physical_seed.xyz`
- `user_assets/dft/rX/labeled.xyz`
  - externally completed labeled DFT data for outer round `X`

The current release tracks placeholder directories with `.gitkeep` files. Large
models, trajectories, and DFT results should be supplied by the user and are not
expected to be committed.

## Runtime Outputs

Workflow outputs are written to:

- `runs/<run_name>/artifacts/...`
- `runs/<run_name>/runtime/...`
- `runs/<run_name>/logs/...`

`runs/` is generated at runtime and is not part of the release package.

## Documentation

- User workflow manual: [`USER_MANUAL.md`](./USER_MANUAL.md)
- Core algorithm guide: [`CORE_ALGORITHM_GUIDE.md`](./CORE_ALGORITHM_GUIDE.md)
- Environment requirements: [`ENVIRONMENT_REQUIREMENTS.md`](./ENVIRONMENT_REQUIREMENTS.md)
- Example per-round YAML: [`Pt-water-interface_roundN.yaml`](./Pt-water-interface_roundN.yaml)

## Repository Layout

- `workflow.py`: top-level workflow entrypoint
- `configs/`: environment examples and pipeline configuration
- `workflow_core/`: workflow orchestration and domain logic
- `workflow_backends/`: MACE, DeepMD, LAMMPS, GPUMD, and CP2K backend adapters
- `workflow_policies/`: selection, routing, and screening policies
- `workflow_artifacts/`: structured artifact and report helpers
- `workflow_runtime/`: runtime paths, logs, markers, resume, and state handling
- `user_assets/`: user-provided models, seed data, and DFT backfill inputs

## Release Notes

The default example remains the MACE / MACE-LES teacher plus DeepMD student
mainline. DPA-4 teacher training and relabeling are part of the supported
teacher backend surface, and are selected by switching the committee-cycle
teacher backend and config path.
