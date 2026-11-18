#!/usr/bin/env bash
# Set the Conda installation path and the name of your backend environment.
source /path/to/miniconda3/etc/profile.d/conda.sh || return 1
conda activate dice || return 1

# Keep an existing device allocation; otherwise use local GPU 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DICE_LAMMPS_PARALLEL_TASKS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
