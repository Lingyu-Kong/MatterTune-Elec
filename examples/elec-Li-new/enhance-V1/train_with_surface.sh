#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/enhance-V1/train_with_surface.sh [train.sh options]

Purpose:
  Merge the six lambda0/lambda1 parent/deleted datasets plus Li-surface.xyz,
  then run enhance-V1 energy/force training without a delta-E loss.

Environment overrides:
  DATA_ROOT=/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1
  MERGED_TRAIN_FILE=$DATA_ROOT/Li_electrolyte_V1_with_surface_all.xyz
  OUTPUT_PREFIX=Li_electrolyte_V1_with_surface
  OUTPUT_ROOT=$DATA_ROOT/local_runs/enhance-V1/with_enhance_surface
  FORCE_REBUILD_MERGED=1  rebuild even when the merged file is up to date
  PREPARE_ONLY=1          prepare/validate the merged file without training

All remaining command-line arguments are forwarded to enhance-V1/train.sh.
The merged train file, all-force strategy, positive force-loss weight, and
DELTA_E_LOSS_WEIGHT=0 are enforced after forwarded options.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SH="${SCRIPT_DIR}/train.sh"
DATA_ROOT="${DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1}"
MERGED_TRAIN_FILE="${MERGED_TRAIN_FILE:-${DATA_ROOT}/Li_electrolyte_V1_with_surface_all.xyz}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-Li_electrolyte_V1_with_surface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/local_runs/enhance-V1/with_enhance_surface}"
FORCE_REBUILD_MERGED="${FORCE_REBUILD_MERGED:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/envs/mattersim-elec/bin/python}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20.0}"

SOURCE_FILES=(
  "${DATA_ROOT}/Li_system_lambda0_del.xyz"
  "${DATA_ROOT}/Li_system_lambda0.xyz"
  "${DATA_ROOT}/Li_system_lambda1_del.xyz"
  "${DATA_ROOT}/Li_system_lambda1.xyz"
  "${DATA_ROOT}/Li-surface.xyz"
  "${DATA_ROOT}/Li-system-lambda0_mdv1_del.xyz"
  "${DATA_ROOT}/Li-system-lambda0_mdv1.xyz"
)

if [[ ! -f "${TRAIN_SH}" ]]; then
  echo "enhance-V1 train.sh not found: ${TRAIN_SH}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! awk -v value="${F_LOSS_WEIGHT}" 'BEGIN { exit !(value > 0.0) }'; then
  echo "F_LOSS_WEIGHT must be positive for train_without_delta_e: ${F_LOSS_WEIGHT}" >&2
  exit 2
fi
for source_file in "${SOURCE_FILES[@]}"; do
  if [[ ! -s "${source_file}" ]]; then
    echo "Training source is missing or empty: ${source_file}" >&2
    exit 1
  fi
done

needs_rebuild=0
if [[ "${FORCE_REBUILD_MERGED}" == "1" || ! -s "${MERGED_TRAIN_FILE}" ]]; then
  needs_rebuild=1
else
  for source_file in "${SOURCE_FILES[@]}"; do
    if [[ "${source_file}" -nt "${MERGED_TRAIN_FILE}" ]]; then
      needs_rebuild=1
      break
    fi
  done
fi

if [[ "${needs_rebuild}" == "1" ]]; then
  mkdir -p "$(dirname "${MERGED_TRAIN_FILE}")"
  merged_tmp="$(mktemp "${MERGED_TRAIN_FILE}.tmp.XXXXXX")"
  cleanup() {
    rm -f "${merged_tmp}"
  }
  trap cleanup EXIT

  echo "Merging ${#SOURCE_FILES[@]} XYZ files into ${MERGED_TRAIN_FILE}"
  for source_file in "${SOURCE_FILES[@]}"; do
    echo "  + ${source_file}"
    if [[ -s "${merged_tmp}" && "$(tail -c 1 "${merged_tmp}" | wc -l)" -eq 0 ]]; then
      printf '\n' >> "${merged_tmp}"
    fi
    cat "${source_file}" >> "${merged_tmp}"
  done
  mv "${merged_tmp}" "${MERGED_TRAIN_FILE}"
  trap - EXIT
else
  echo "Merged training file is up to date: ${MERGED_TRAIN_FILE}"
fi

"${PYTHON_BIN}" - "${MERGED_TRAIN_FILE}" <<'PY'
import sys
from pathlib import Path

import numpy as np
from ase.io import iread


path = Path(sys.argv[1])
n_structures = 0
n_atoms = 0
for index, atoms in enumerate(iread(path, index=":")):
    try:
        energy = float(atoms.get_potential_energy())
    except Exception as exc:
        raise RuntimeError(f"Missing energy at structure {index}") from exc
    try:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception:
        for key in ("forces", "force"):
            if key in atoms.arrays:
                forces = np.asarray(atoms.arrays[key], dtype=np.float64)
                break
        else:
            raise RuntimeError(f"Missing forces at structure {index}")
    if forces.shape != (len(atoms), 3):
        raise RuntimeError(
            f"Invalid force shape at structure {index}: {forces.shape}; "
            f"expected {(len(atoms), 3)}"
        )
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite label at structure {index}")
    n_structures += 1
    n_atoms += len(atoms)

print(f"Validated merged dataset: {n_structures:,} structures, {n_atoms:,} atoms")
PY

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "PREPARE_ONLY=1; training was not started."
  exit 0
fi

export DATA_ROOT
export DATA_VARIANT="with_enhance"
export TRAIN_FILE="${MERGED_TRAIN_FILE}"
export OUTPUT_PREFIX
export OUTPUT_ROOT
export PREPARE_PAIRS=0
export DELTA_E_LOSS_WEIGHT=0
export FORCE_TRAINING_STRATEGY=all
export F_LOSS_WEIGHT

exec bash "${TRAIN_SH}" "$@" \
  --train_file "${MERGED_TRAIN_FILE}" \
  --force_training_strategy all \
  --f_loss_weight "${F_LOSS_WEIGHT}" \
  --delta_e_loss_weight 0
