#!/usr/bin/env bash
source /home/nanako/miniconda3/etc/profile.d/conda.sh
conda activate deepmd-lammps-dpa4
export PYTHONPATH=/mnt/c/Users/1/Desktop/mlff_workflow
export DP_INTERFACE_PREC=low
export DP_INFER_BATCH_SIZE=65536
export OMP_NUM_THREADS=1
export NUM_WORKERS=0
export XDG_CACHE_HOME=/tmp/xdg-cache-dpa4
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-dpa4
export LAMMPS_PLUGIN_PATH=/mnt/c/Users/1/Desktop/mlff_workflow/software/deepmd-lammps-dpa4/deepmd-root/lib/deepmd_lmp
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:/mnt/c/Users/1/Desktop/mlff_workflow/software/deepmd-lammps-dpa4/deepmd-root/lib:/mnt/c/Users/1/Desktop/mlff_workflow/software/deepmd-lammps-dpa4/deepmd-root/lib/deepmd_lmp:${LD_LIBRARY_PATH:-}
