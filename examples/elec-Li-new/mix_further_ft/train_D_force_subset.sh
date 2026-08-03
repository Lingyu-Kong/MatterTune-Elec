#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MIX_DATA_ROOT="${MIX_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix}"
SOURCE_VARIANT="${SOURCE_VARIANT:-with_enhance}"
REFERENCE_DATA_SCOPE="${REFERENCE_DATA_SCOPE:-bulk_interface}"
CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
FORCE_SUBSET_FRACTION="${FORCE_SUBSET_FRACTION:-0.2}"
FORCE_SUBSET_SEED="${FORCE_SUBSET_SEED:-42}"
FORCE_SUBSET_KEY="${FORCE_SUBSET_KEY:-force_train_mask}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
export RUN_STAMP

safe_float_label() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

MERGED_TRAIN_FILE="${MERGED_TRAIN_FILE:-${MIX_DATA_ROOT}/train_mlpmd_bulk_interface.xyz}"
BULK_FILE="${BULK_FILE:-${MIX_DATA_ROOT}/train_mlpmd_bulk.xyz}"
INTERFACE_FILE="${INTERFACE_FILE:-${MIX_DATA_ROOT}/train_mlpmd_interface.xyz}"
if [[ ! -f "${MERGED_TRAIN_FILE}" || "${BULK_FILE}" -nt "${MERGED_TRAIN_FILE}" || "${INTERFACE_FILE}" -nt "${MERGED_TRAIN_FILE}" ]]; then
  tmp_merged="${MERGED_TRAIN_FILE}.tmp.$$"
  rm -f "${tmp_merged}"
  cat "${BULK_FILE}" "${INTERFACE_FILE}" > "${tmp_merged}"
  mv "${tmp_merged}" "${MERGED_TRAIN_FILE}"
fi

fraction_label="$(safe_float_label "${FORCE_SUBSET_FRACTION}")"
SUBSET_TRAIN_FILE="${SUBSET_TRAIN_FILE:-${MIX_DATA_ROOT}/train_mlpmd_bulk_interface_force_subset_f${fraction_label}_seed${FORCE_SUBSET_SEED}.xyz}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
PYTHONPATH=src python examples/elec-Li-new/mix_further_ft/prepare_force_subset_xyz.py \
  --input "${MERGED_TRAIN_FILE}" \
  --output "${SUBSET_TRAIN_FILE}" \
  --fraction "${FORCE_SUBSET_FRACTION}" \
  --seed "${FORCE_SUBSET_SEED}" \
  --key "${FORCE_SUBSET_KEY}"

export TRAIN_FILE="${TRAIN_FILE:-${SUBSET_TRAIN_FILE}}"
export DATA_SCOPE="${DATA_SCOPE:-D_force_subset_f${fraction_label}_seed${FORCE_SUBSET_SEED}}"
export REFERENCE_ROOT="${REFERENCE_ROOT:-${MIX_DATA_ROOT}/references/mix_further_ft/from_${SOURCE_VARIANT}/${REFERENCE_DATA_SCOPE}}"
export FORCE_TRAINING_STRATEGY="${FORCE_TRAINING_STRATEGY:-subset}"
export FORCE_SUBSET_KEY
export VALIDATION_FORCE_MODE="${VALIDATION_FORCE_MODE:-all}"
export REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
export REBUILD_TRAIN_FILE="${REBUILD_TRAIN_FILE:-0}"
export LR="${LR:-3e-5}"
export E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200}"
export F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export WANDB_NAME="${WANDB_NAME:-${RUN_STAMP}-D_force_subset_f${fraction_label}_seed${FORCE_SUBSET_SEED}}"

exec bash "${SCRIPT_DIR}/train.sh" "$@"
