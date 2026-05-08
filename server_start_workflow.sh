#!/usr/bin/env bash
set -euo pipefail

# Server launch script for the formal workflow.
# Default behavior:
# - submit itself into background with nohup unless explicitly disabled
# - run workflow.py inside the deepmd environment
# - expose one CUDA device unless overridden
# - set process-wide thread env for the Python/DeepMD process
# - let DFT tasks auto-switch into the cp2k environment via CP2K_EXEC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/software/miniconda3}"
MAIN_ENV="${MAIN_ENV:-deepmd}"
CP2K_ENV="${CP2K_ENV:-cp2k}"
WORKFLOW_YAML="${1:-Pt-water-interface_roundN.yaml}"
LOG_BASENAME_INPUT="${2:-}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

if [[ "${MLFF_WORKFLOW_FOREGROUND:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  if [[ -n "${LOG_BASENAME_INPUT}" ]]; then
    LOG_NAME="$(basename "${LOG_BASENAME_INPUT}")"
  else
    WORKFLOW_LABEL="$(basename "${WORKFLOW_YAML}")"
    WORKFLOW_LABEL="${WORKFLOW_LABEL%.yaml}"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_NAME="${WORKFLOW_LABEL}_${TIMESTAMP}.log"
  fi
  if [[ "${LOG_NAME}" != *.log ]]; then
    LOG_NAME="${LOG_NAME}.log"
  fi
  LOG_FILE="${LOG_DIR}/${LOG_NAME}"

  echo "[INFO] Submitting workflow in background"
  echo "[INFO] Workflow: ${WORKFLOW_YAML}"
  echo "[INFO] Log file: ${LOG_FILE}"

  MLFF_WORKFLOW_FOREGROUND=1 nohup bash "${BASH_SOURCE[0]}" "${WORKFLOW_YAML}" "${LOG_NAME}" > "${LOG_FILE}" 2>&1 < /dev/null &
  JOB_PID=$!
  disown "${JOB_PID}" 2>/dev/null || true

  echo "[INFO] Background PID: ${JOB_PID}"
  exit 0
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${MAIN_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"
export TF_INTRA_OP_PARALLELISM_THREADS="${TF_INTRA_OP_PARALLELISM_THREADS:-32}"
export TF_INTER_OP_PARALLELISM_THREADS="${TF_INTER_OP_PARALLELISM_THREADS:-2}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export ABSL_LOG_LEVEL="${ABSL_LOG_LEVEL:-2}"

export CP2K_EXEC="${CP2K_EXEC:-${CONDA_ROOT}/envs/${CP2K_ENV}/bin/cp2k.ssmp}"

cd "${SCRIPT_DIR}"

echo "[INFO] Workspace: ${SCRIPT_DIR}"
echo "[INFO] Workflow:  ${WORKFLOW_YAML}"
echo "[INFO] Main env:  ${MAIN_ENV}"
echo "[INFO] CP2K exec: ${CP2K_EXEC}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[INFO] TF_INTRA_OP_PARALLELISM_THREADS=${TF_INTRA_OP_PARALLELISM_THREADS}"
echo "[INFO] TF_INTER_OP_PARALLELISM_THREADS=${TF_INTER_OP_PARALLELISM_THREADS}"
echo "[INFO] TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL}"
echo "[INFO] ABSL_LOG_LEVEL=${ABSL_LOG_LEVEL}"

python workflow.py "${WORKFLOW_YAML}"
