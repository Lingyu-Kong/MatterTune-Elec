#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIX_DATA_ROOT="${MIX_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix}"
SOURCE_VARIANT="${SOURCE_VARIANT:-with_enhance}"
REFERENCE_DATA_SCOPE="${REFERENCE_DATA_SCOPE:-bulk_interface}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_STAMP

export DATA_SCOPE="${DATA_SCOPE:-A_energy_only}"
export REFERENCE_ROOT="${REFERENCE_ROOT:-${MIX_DATA_ROOT}/references/mix_further_ft/from_${SOURCE_VARIANT}/${REFERENCE_DATA_SCOPE}}"
export FORCE_TRAINING_STRATEGY="${FORCE_TRAINING_STRATEGY:-none}"
export F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-0}"
export VALIDATION_FORCE_MODE="${VALIDATION_FORCE_MODE:-energy_only}"
export REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
export LR="${LR:-3e-5}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export WANDB_NAME="${WANDB_NAME:-${RUN_STAMP}-A_energy_only}"

exec bash "${SCRIPT_DIR}/train.sh" "$@"
