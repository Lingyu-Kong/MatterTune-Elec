#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATTERTUNE_DIR="${MATTERTUNE_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${MATTERTUNE_DIR}/.." && pwd)}"
MATTERSIM_DIR="${MATTERSIM_DIR:-${WORKSPACE_ROOT}/mattersim}"
LAMMPS_PYTHON_DIR="${LAMMPS_PYTHON_DIR:-${WORKSPACE_ROOT}/lammps/lammps/python}"

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
LMP_BIN="${LMP_BIN:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/envs/mattersim-elec/bin/lmp}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
RESTART_FILE="${RESTART_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
STEPS="${STEPS:-}"
STEPS_MODE="${STEPS_MODE:-total}"
RESTART_INTERVAL="${RESTART_INTERVAL:-1000}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
STRICT="${STRICT:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
KOKKOS_GPUS="${KOKKOS_GPUS:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-}}"
DRY_RUN="${DRY_RUN:-0}"
RUN_MD="${RUN_MD:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti_restart.sh --run-dir PATH [options]

Required:
  --run-dir PATH              Original run directory made by run_lammps_fep_ti.sh.

Step modes:
  --steps-mode total          Count saved restart steps and stop at --steps (default).
                              Without --steps, use the original run target.
  --steps-mode additional     Run --steps more steps from the selected restart.
                              Without --steps, use the original configured step count.
  --steps N                   Total target or additional count, according to --steps-mode.

Other options:
  --restart-file PATH         Restart to read. Default: newer lmp.restart.1/.2 in RUN_DIR.
  --output-dir PATH           Output location. Default: original RUN_DIR (same directory).
  --cuda-visible-devices IDS  Set CUDA_VISIBLE_DEVICES for export and LAMMPS.
  --prepare-only              Prepare files but do not launch LAMMPS.
  --dry-run                   Print commands without executing them.

LAMMPS keeps only lmp.restart.1 and lmp.restart.2 and alternates between them.
Energy and temperature CSVs continue in their original files. Continuation outputs
are numbered restart1, restart2, restart3, ... without overwriting earlier trajectories.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir|--source-run-dir) SOURCE_RUN_DIR="$2"; shift 2 ;;
    --restart-file|--restart) RESTART_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --steps-mode) STEPS_MODE="$2"; shift 2 ;;
    --total-steps) STEPS_MODE=total; STEPS="$2"; shift 2 ;;
    --additional-steps) STEPS_MODE=additional; STEPS="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --prepare-only|--no-run) RUN_MD=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${SOURCE_RUN_DIR}" ]]; then
  echo "--run-dir is required." >&2
  usage >&2
  exit 2
fi
if [[ "${STEPS_MODE}" != "total" && "${STEPS_MODE}" != "additional" ]]; then
  echo "--steps-mode must be total or additional, got: ${STEPS_MODE}" >&2
  exit 2
fi

SOURCE_METADATA="${SOURCE_RUN_DIR}/prepare_metadata.json"
SOURCE_CONFIG="${SOURCE_RUN_DIR}/run_lammps_fep_ti_config.txt"
if [[ ! -f "${SOURCE_METADATA}" || ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Missing prepare_metadata.json or run_lammps_fep_ti_config.txt under ${SOURCE_RUN_DIR}" >&2
  exit 1
fi

config_value() {
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "${SOURCE_CONFIG}"
}

latest_restart=""
restart_1="${SOURCE_RUN_DIR}/lmp.restart.1"
restart_2="${SOURCE_RUN_DIR}/lmp.restart.2"
if [[ -f "${restart_1}" && -f "${restart_2}" ]]; then
  if [[ "${restart_1}" -nt "${restart_2}" ]]; then latest_restart="${restart_1}"; else latest_restart="${restart_2}"; fi
elif [[ -f "${restart_1}" ]]; then
  latest_restart="${restart_1}"
elif [[ -f "${restart_2}" ]]; then
  latest_restart="${restart_2}"
fi

if [[ -z "${RESTART_FILE}" ]]; then RESTART_FILE="${latest_restart}"; fi
if [[ -z "${RESTART_FILE}" || ! -f "${RESTART_FILE}" ]]; then
  echo "No usable lmp.restart.1/.2 found (or --restart-file does not exist): ${RESTART_FILE}" >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then OUTPUT_DIR="${SOURCE_RUN_DIR}"; fi

CKPT="$(config_value checkpoint)"
LAMBDA_VALUE="$(config_value lambda_value)"
TARGET_TYPE="$(config_value target_type)"
TEMPERATURE="$(config_value temperature)"
TIMESTEP_FS="$(config_value timestep_fs)"
FRICTION_FS_INV="$(config_value friction_fs_inv)"
ORIGINAL_STEPS="$(config_value steps)"
THERMO_INTERVAL="$(config_value thermo_interval)"
DUMP_INTERVAL="$(config_value dump_interval)"
XTC_DUMP_INTERVAL="$(config_value xtc_dump_interval)"
ENERGY_LOG_INTERVAL="$(config_value energy_log_interval)"
SIGMA="$(config_value sigma)"
EPSILON="$(config_value epsilon)"
LJ_CUTOFF="$(config_value lj_cutoff)"
ENERGY_LOG_PATH="$(config_value energy_log_path)"
ENERGY_LOG_RAW_PATH="$(config_value energy_log_raw_path)"
TEMPERATURE_LOG_PATH="$(config_value temperature_log_path)"
if [[ -z "${STEPS}" ]]; then STEPS="${ORIGINAL_STEPS}"; fi
if ! [[ "${STEPS}" =~ ^[0-9]+$ ]]; then echo "--steps must be a non-negative integer" >&2; exit 2; fi

LAMBDA_TAG="$(python -c 'import sys; print(f"lambda{int(round(float(sys.argv[1]) * 1000)):03d}")' "${LAMBDA_VALUE}")"
max_continuation_index=0
shopt -s nullglob
for candidate in "${OUTPUT_DIR}"/log."${LAMBDA_TAG}".restart[0-9]*.lammps; do
  filename="${candidate##*/}"
  suffix="${filename#*.restart}"
  suffix="${suffix%.lammps}"
  if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
    suffix_number=$((10#${suffix}))
    if (( suffix_number > max_continuation_index )); then max_continuation_index=${suffix_number}; fi
  fi
done
shopt -u nullglob
continuation_index=$((max_continuation_index + 1))
CONTINUATION_TAG="restart${continuation_index}"

MODEL_PATH="${OUTPUT_DIR}/ghost-target-${LAMBDA_TAG}-${CONTINUATION_TAG}.pt"
INPUT_PATH="${OUTPUT_DIR}/in.${LAMBDA_TAG}.${CONTINUATION_TAG}.lammps"
METADATA_PATH="${OUTPUT_DIR}/prepare_${CONTINUATION_TAG}_metadata.json"
ENERGY_LOG_SEGMENT_PATH="${OUTPUT_DIR}/.fep-ti-energy-${CONTINUATION_TAG}.segment.csv"
FINAL_DATA_PATH="${OUTPUT_DIR}/final_${LAMBDA_TAG}_${CONTINUATION_TAG}.data"
DUMP_PATH="${OUTPUT_DIR}/traj_${LAMBDA_TAG}_${CONTINUATION_TAG}.lammpstrj"
XTC_PATH="${OUTPUT_DIR}/traj_${LAMBDA_TAG}_${CONTINUATION_TAG}.xtc"
LOG_PATH="${OUTPUT_DIR}/log.${LAMBDA_TAG}.${CONTINUATION_TAG}.lammps"
RESTART_PATH_1="${SOURCE_RUN_DIR}/lmp.restart.1"
RESTART_PATH_2="${SOURCE_RUN_DIR}/lmp.restart.2"

run_cmd() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

if [[ ! -f "${CONDA_SH}" ]]; then echo "Conda setup script not found: ${CONDA_SH}" >&2; exit 1; fi
if [[ ! -f "${CKPT}" ]]; then echo "Checkpoint not found: ${CKPT}" >&2; exit 1; fi
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
if [[ -d "${LAMMPS_PYTHON_DIR}/lammps/mliap" ]]; then
  export PYTHONPATH="${MATTERSIM_DIR}/src:${MATTERTUNE_DIR}/src:${LAMMPS_PYTHON_DIR}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${MATTERSIM_DIR}/src:${MATTERTUNE_DIR}/src:${PYTHONPATH:-}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"; fi

if [[ "${DRY_RUN}" != "1" ]]; then mkdir -p "${OUTPUT_DIR}"; else echo "[dry-run] would use ${OUTPUT_DIR}"; fi
echo "RESTART_FILE=${RESTART_FILE}"
echo "NEXT_RESTART_FILES=${RESTART_PATH_1},${RESTART_PATH_2}"
echo "STEPS_MODE=${STEPS_MODE}"
echo "STEPS=${STEPS}"
echo "XTC_PATH=${XTC_PATH}"
echo "ENERGY_LOG_PATH=${ENERGY_LOG_PATH}"
echo "TEMPERATURE_LOG_PATH=${TEMPERATURE_LOG_PATH}"

EXPORT_ARGS=()
if [[ "${STRICT}" == "1" ]]; then EXPORT_ARGS+=(--strict); fi
if [[ "${NO_COMPILE}" == "1" ]]; then EXPORT_ARGS+=(--no-compile); fi
run_cmd python "${MATTERTUNE_DIR}/examples/elec-Li-new/enhance-V1/export_mattertune_ghost_target_lammps_mliap.py" \
  --checkpoint "${CKPT}" --output "${MODEL_PATH}" --lambda-value "${LAMBDA_VALUE}" \
  --target-types "${TARGET_TYPE}" --epsilon "${EPSILON}" --sigma "${SIGMA}" \
  --lj-cutoff "${LJ_CUTOFF}" --energy-log-path "${ENERGY_LOG_SEGMENT_PATH}" \
  --energy-log-interval "${ENERGY_LOG_INTERVAL}" --energy-log-timestep-fs "${TIMESTEP_FS}" \
  --device "${EXPORT_DEVICE}" "${EXPORT_ARGS[@]}"

run_cmd python "${SCRIPT_DIR}/prepare_lammps_ghost_target_restart_input.py" \
  --restart "${RESTART_FILE}" --source-metadata "${SOURCE_METADATA}" \
  --input "${INPUT_PATH}" --model "${MODEL_PATH}" --metadata "${METADATA_PATH}" \
  --temperature "${TEMPERATURE}" --timestep-fs "${TIMESTEP_FS}" \
  --friction-fs-inv "${FRICTION_FS_INV}" --steps "${STEPS}" --steps-mode "${STEPS_MODE}" \
  --thermo-interval "${THERMO_INTERVAL}" --dump-interval "${DUMP_INTERVAL}" \
  --xtc-dump-interval "${XTC_DUMP_INTERVAL}" --final-data "${FINAL_DATA_PATH}" \
  --dump "${DUMP_PATH}" --xtc-dump "${XTC_PATH}" \
  --temperature-log "${TEMPERATURE_LOG_PATH}" --temperature-log-interval "${ENERGY_LOG_INTERVAL}" \
  --restart-interval "${RESTART_INTERVAL}" --restart-path-1 "${RESTART_PATH_1}" \
  --restart-path-2 "${RESTART_PATH_2}"

if [[ "${RUN_MD}" != "1" ]]; then echo "Prepared restart files; skipping MD because RUN_MD=${RUN_MD}."; exit 0; fi
if [[ ! -x "${LMP_BIN}" ]]; then echo "LAMMPS binary not executable: ${LMP_BIN}" >&2; exit 1; fi
run_cmd "${LMP_BIN}" -k on g "${KOKKOS_GPUS}" -sf kk -pk kokkos newton on neigh half \
  -in "${INPUT_PATH}" -log "${LOG_PATH}"

if [[ "${DRY_RUN}" != "1" && -f "${ENERGY_LOG_SEGMENT_PATH}" ]]; then
  run_cmd python "${SCRIPT_DIR}/merge_lammps_fep_ti_energy_log.py" \
    --energy-log-raw "${ENERGY_LOG_RAW_PATH}" --append-raw-segment "${ENERGY_LOG_SEGMENT_PATH}" \
    --temperature-log "${TEMPERATURE_LOG_PATH}" --output "${ENERGY_LOG_PATH}"
fi
