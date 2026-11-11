#!/usr/bin/env bash
# Copy to your run directory as activate-native.sh and edit this installation path.
# The native activation must expose compatible python, dp, lmp and shared libraries.
if [[ ! -r /absolute/path/to/native/activate.sh ]]; then
    echo "Edit activate-native.sh to source your qualified native installation." >&2
    return 1
fi
source /absolute/path/to/native/activate.sh
export DP_INTERFACE_PREC=high
export DICE_LAMMPS_PARALLEL_TASKS=1
