#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIX_DATA_ROOT="${MIX_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix}"
SOURCE_VARIANT="${SOURCE_VARIANT:-with_enhance}"
REFERENCE_DATA_SCOPE="${REFERENCE_DATA_SCOPE:-bulk_interface}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_STAMP

find_latest_file() {
  local root="$1"
  local pattern="$2"
  if [[ ! -d "${root}" ]]; then
    return 0
  fi
  find "${root}" -type f -name "${pattern}" -printf '%T@ %p\n' \
    | sort -nr \
    | sed -n '1s/^[^ ]* //p'
}

A_OUTPUT_ROOT="${A_OUTPUT_ROOT:-${MIX_DATA_ROOT}/local_runs/mix_further_ft/from_${SOURCE_VARIANT}/A_energy_only}"
if [[ -z "${ENERGY_ONLY_CKPT:-}" ]]; then
  ENERGY_ONLY_CKPT="$(find_latest_file "${A_OUTPUT_ROOT}" '*best.ckpt')"
fi
if [[ -z "${ENERGY_ONLY_CKPT:-}" || ! -f "${ENERGY_ONLY_CKPT}" ]]; then
  echo "ENERGY_ONLY_CKPT is required, or run train_A_energy_only.sh first." >&2
  echo "Searched under: ${A_OUTPUT_ROOT}" >&2
  exit 1
fi

export DATA_SCOPE="${DATA_SCOPE:-B_force_polish_from_A}"
export REFERENCE_ROOT="${REFERENCE_ROOT:-${MIX_DATA_ROOT}/references/mix_further_ft/from_${SOURCE_VARIANT}/${REFERENCE_DATA_SCOPE}}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ENERGY_ONLY_CKPT}}"
export FORCE_TRAINING_STRATEGY="${FORCE_TRAINING_STRATEGY:-all}"
export VALIDATION_FORCE_MODE="${VALIDATION_FORCE_MODE:-all}"
export REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
export LR="${LR:-1e-5}"
export E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-100}"
export F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export WANDB_NAME="${WANDB_NAME:-${RUN_STAMP}-B_force_polish_from_A}"

exec bash "${SCRIPT_DIR}/train.sh" "$@"
