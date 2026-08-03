#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIX_DATA_ROOT="${MIX_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix}"
SOURCE_VARIANT="${SOURCE_VARIANT:-with_enhance}"
REFERENCE_DATA_SCOPE="${REFERENCE_DATA_SCOPE:-bulk_interface}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_STAMP

FORCE_EVERY_N_STEPS="${FORCE_EVERY_N_STEPS:-5}"
export FORCE_EVERY_N_STEPS
export DATA_SCOPE="${DATA_SCOPE:-C_sparse_force_every${FORCE_EVERY_N_STEPS}}"
export REFERENCE_ROOT="${REFERENCE_ROOT:-${MIX_DATA_ROOT}/references/mix_further_ft/from_${SOURCE_VARIANT}/${REFERENCE_DATA_SCOPE}}"
export FORCE_TRAINING_STRATEGY="${FORCE_TRAINING_STRATEGY:-every_n}"
export VALIDATION_FORCE_MODE="${VALIDATION_FORCE_MODE:-all}"
export REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
export LR="${LR:-3e-5}"
export E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200}"
export F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export WANDB_NAME="${WANDB_NAME:-${RUN_STAMP}-C_sparse_force_every${FORCE_EVERY_N_STEPS}}"

exec bash "${SCRIPT_DIR}/train.sh" "$@"
