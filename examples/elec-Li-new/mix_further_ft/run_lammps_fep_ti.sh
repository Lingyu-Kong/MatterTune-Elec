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

CKPT="${CKPT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix/local_runs/mix_further_ft/from_with_enhance/all_mix/train_without_delta_e/model.ckpt}"

STRUCTURE="${STRUCTURE:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix/mixture_LHCE_system/top.pdb}"

CONFIG_TYPE="${CONFIG_TYPE:-system}"
RUN_ROOT="${RUN_ROOT:-${PWD}}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

RUN_DIR="${RUN_DIR:-}"

LAMBDA_VALUE="${LAMBDA_VALUE:-0.0}"

TARGET_INDICES="${TARGET_INDICES:-0}"
TARGET_TYPE="${TARGET_TYPE:-8}"

ELEMENT_ORDER="${ELEMENT_ORDER:-Li,F,S,N,O,C,H}"

STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"

TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TEMPERATURE="${TEMPERATURE:-298.15}"
FRICTION_FS_INV="${FRICTION_FS_INV:-0.02}"

THERMO_INTERVAL="${THERMO_INTERVAL:-100}"
DUMP_INTERVAL="${DUMP_INTERVAL:-100}"
XTC_DUMP_INTERVAL="${XTC_DUMP_INTERVAL:-500}"

RESTART_INTERVAL="${RESTART_INTERVAL:-1000}"
ENERGY_LOG_INTERVAL="${ENERGY_LOG_INTERVAL:-1}"

SEED="${SEED:-7}"
INIT_VELOCITIES="${INIT_VELOCITIES:-1}"

SIGMA="${SIGMA:-2.337}"
EPSILON="${EPSILON:-0.00694}"
LJ_CUTOFF="${LJ_CUTOFF:-10.0}"

EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"

STRICT="${STRICT:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

KOKKOS_GPUS="${KOKKOS_GPUS:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-}}"

DRY_RUN="${DRY_RUN:-0}"
RUN_MD="${RUN_MD:-1}"

usage() {
    cat <<'EOF'
Usage:
  run_lammps_fep_ti.sh --lambda-value VALUE [options]

Options:
  --checkpoint PATH
  --structure PATH
  --run-dir PATH
  --steps N
  --warmup-steps N
  --temperature K
  --timestep-fs FS
  --friction-fs-inv VALUE
  --energy-log-interval N
  --target-indices LIST
  --lj-cutoff A
  --cuda-visible-devices IDS
  --prepare-only
  --force-export
  --no-compile
  --dry-run
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lambda-value)
            LAMBDA_VALUE="$2"
            shift 2
            ;;
        --checkpoint|--ckpt)
            CKPT="$2"
            shift 2
            ;;
        --structure|--pdb)
            STRUCTURE="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --timestep-fs)
            TIMESTEP_FS="$2"
            shift 2
            ;;
        --friction-fs-inv)
            FRICTION_FS_INV="$2"
            shift 2
            ;;
        --energy-log-interval)
            ENERGY_LOG_INTERVAL="$2"
            shift 2
            ;;
        --target-indices)
            TARGET_INDICES="$2"
            shift 2
            ;;
        --lj-cutoff)
            LJ_CUTOFF="$2"
            shift 2
            ;;
        --cuda-visible-devices)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --prepare-only|--no-run)
            RUN_MD=0
            shift
            ;;
        --force-export)
            FORCE_EXPORT=1
            shift
            ;;
        --no-compile)
            NO_COMPILE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${LAMBDA_VALUE}" ]]; then
    echo "--lambda-value is required." >&2
    exit 2
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda setup script not found: ${CONDA_SH}" >&2
    exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
fi

if [[ ! -f "${STRUCTURE}" ]]; then
    echo "Structure PDB not found: ${STRUCTURE}" >&2
    exit 1
fi

PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_lammps_ghost_target_input.py"

if [[ ! -f "${PREPARE_SCRIPT}" ]]; then
    echo "Prepare script not found: ${PREPARE_SCRIPT}" >&2
    exit 1
fi

LAMBDA_TAG="$(
    python - "${LAMBDA_VALUE}" <<'PY'
import sys

value = float(sys.argv[1])

if value < 0.0 or value > 1.0:
    raise SystemExit("lambda must be in [0,1]")

print(f"lambda{int(round(value * 1000)):03d}")
PY
)"

if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR="${RUN_ROOT}/${CONFIG_TYPE}/${RUN_STAMP}-${LAMBDA_TAG}-lammps"
fi

mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

MODEL_PATH="${RUN_DIR}/ghost-target-${LAMBDA_TAG}-type${TARGET_TYPE}-rc${LJ_CUTOFF}.pt"

DATA_PATH="${RUN_DIR}/top_target_type${TARGET_TYPE}.data"

INPUT_PATH="${RUN_DIR}/in.${LAMBDA_TAG}.lammps"

METADATA_PATH="${RUN_DIR}/prepare_metadata.json"

ENERGY_LOG_PATH="${RUN_DIR}/fep-ti-energy.log"

ENERGY_LOG_RAW_PATH="${RUN_DIR}/fep-ti-energy.raw.csv"

TEMPERATURE_LOG_PATH="${RUN_DIR}/fep-ti-temperature.csv"

FINAL_DATA_PATH="${RUN_DIR}/final_${LAMBDA_TAG}.data"

DUMP_PATH="${RUN_DIR}/traj_${LAMBDA_TAG}.lammpstrj"

XTC_PATH="${RUN_DIR}/traj_${LAMBDA_TAG}.xtc"

RESTART_PATH_1="${RUN_DIR}/lmp.restart.1"
RESTART_PATH_2="${RUN_DIR}/lmp.restart.2"

LOG_PATH="${RUN_DIR}/log.${LAMBDA_TAG}.lammps"

CONFIG_PATH="${RUN_DIR}/run_lammps_fep_ti_config.txt"

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'

    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONNOUSERSITE=1

if [[ -d "${LAMMPS_PYTHON_DIR}/lammps/mliap" ]]; then
    export PYTHONPATH="${MATTERSIM_DIR}/src:${MATTERTUNE_DIR}/src:${LAMMPS_PYTHON_DIR}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${MATTERSIM_DIR}/src:${MATTERTUNE_DIR}/src:${PYTHONPATH:-}"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

#
# Store external inputs as they are.
# Store run-local files as basenames so the config itself can move
# between clusters.
#

if [[ "${DRY_RUN}" != "1" ]]; then
    cat > "${CONFIG_PATH}" <<EOF
created_at=$(date --iso-8601=seconds)
checkpoint=${CKPT}
structure=${STRUCTURE}
run_dir=.
lambda_value=${LAMBDA_VALUE}
target_indices_zero_based=${TARGET_INDICES}
target_type=${TARGET_TYPE}
element_order=${ELEMENT_ORDER}
temperature=${TEMPERATURE}
timestep_fs=${TIMESTEP_FS}
friction_fs_inv=${FRICTION_FS_INV}
warmup_steps=${WARMUP_STEPS}
steps=${STEPS}
thermo_interval=${THERMO_INTERVAL}
dump_interval=${DUMP_INTERVAL}
xtc_dump_interval=${XTC_DUMP_INTERVAL}
restart_interval=${RESTART_INTERVAL}
energy_log_interval=${ENERGY_LOG_INTERVAL}
sigma=${SIGMA}
epsilon=${EPSILON}
lj_cutoff=${LJ_CUTOFF}
model_path=$(basename "${MODEL_PATH}")
data_path=$(basename "${DATA_PATH}")
input_path=$(basename "${INPUT_PATH}")
metadata_path=$(basename "${METADATA_PATH}")
energy_log_path=$(basename "${ENERGY_LOG_PATH}")
dump_path=$(basename "${DUMP_PATH}")
xtc_path=$(basename "${XTC_PATH}")
restart_path_1=$(basename "${RESTART_PATH_1}")
restart_path_2=$(basename "${RESTART_PATH_2}")
energy_log_raw_path=$(basename "${ENERGY_LOG_RAW_PATH}")
temperature_log_path=$(basename "${TEMPERATURE_LOG_PATH}")
log_path=$(basename "${LOG_PATH}")
cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}
kokkos_gpus=${KOKKOS_GPUS}
EOF
else
    echo "[dry-run] would write ${CONFIG_PATH}"
fi

echo "RUN_DIR=${RUN_DIR}"
echo "LAMBDA_VALUE=${LAMBDA_VALUE}"
echo "TARGET_INDICES=${TARGET_INDICES}"
echo "TARGET_TYPE=${TARGET_TYPE}"

echo "CKPT=${CKPT}"
echo "STRUCTURE=${STRUCTURE}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PATH=${DATA_PATH}"

echo "DUMP_PATH=${DUMP_PATH}"
echo "XTC_PATH=${XTC_PATH}"

EXPORT_ARGS=()

if [[ "${STRICT}" == "1" ]]; then
    EXPORT_ARGS+=(--strict)
fi

if [[ "${NO_COMPILE}" == "1" ]]; then
    EXPORT_ARGS+=(--no-compile)
fi

if [[ -f "${MODEL_PATH}" &&
      "${FORCE_EXPORT}" != "1" &&
      ! -f "${ENERGY_LOG_RAW_PATH}" ]]; then

    echo "Existing model has no matching raw FEP-TI log."
    echo "Forcing model re-export."

    FORCE_EXPORT=1
fi

if [[ ! -f "${MODEL_PATH}" || "${FORCE_EXPORT}" == "1" ]]; then

    run_cmd \
        python \
        "${MATTERTUNE_DIR}/examples/elec-Li-new/enhance-V1/export_mattertune_ghost_target_lammps_mliap.py" \
        --checkpoint "${CKPT}" \
        --output "${MODEL_PATH}" \
        --lambda-value "${LAMBDA_VALUE}" \
        --target-types "${TARGET_TYPE}" \
        --epsilon "${EPSILON}" \
        --sigma "${SIGMA}" \
        --lj-cutoff "${LJ_CUTOFF}" \
        --energy-log-path "${ENERGY_LOG_RAW_PATH}" \
        --energy-log-interval "${ENERGY_LOG_INTERVAL}" \
        --energy-log-timestep-fs "${TIMESTEP_FS}" \
        --device "${EXPORT_DEVICE}" \
        "${EXPORT_ARGS[@]}"

else
    echo "Reusing existing model: ${MODEL_PATH}"
fi

INIT_ARGS=()

if [[ "${INIT_VELOCITIES}" == "1" ]]; then
    INIT_ARGS+=(--init-velocities)
else
    INIT_ARGS+=(--no-init-velocities)
fi

run_cmd \
    python \
    "${PREPARE_SCRIPT}" \
    --pdb "${STRUCTURE}" \
    --data "${DATA_PATH}" \
    --input "${INPUT_PATH}" \
    --model "${MODEL_PATH}" \
    --metadata "${METADATA_PATH}" \
    --target-indices "${TARGET_INDICES}" \
    --target-type "${TARGET_TYPE}" \
    --element-order "${ELEMENT_ORDER}" \
    --temperature "${TEMPERATURE}" \
    --timestep-fs "${TIMESTEP_FS}" \
    --friction-fs-inv "${FRICTION_FS_INV}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --steps "${STEPS}" \
    --thermo-interval "${THERMO_INTERVAL}" \
    --dump-interval "${DUMP_INTERVAL}" \
    --xtc-dump-interval "${XTC_DUMP_INTERVAL}" \
    --restart-interval "${RESTART_INTERVAL}" \
    --restart-path-1 "${RESTART_PATH_1}" \
    --restart-path-2 "${RESTART_PATH_2}" \
    --seed "${SEED}" \
    --final-data "${FINAL_DATA_PATH}" \
    --dump "${DUMP_PATH}" \
    --xtc-dump "${XTC_PATH}" \
    --temperature-log "${TEMPERATURE_LOG_PATH}" \
    --temperature-log-interval "${ENERGY_LOG_INTERVAL}" \
    "${INIT_ARGS[@]}"

if [[ "${RUN_MD}" != "1" ]]; then
    echo "Prepared files only; RUN_MD=${RUN_MD}"
    exit 0
fi

if [[ ! -x "${LMP_BIN}" ]]; then
    echo "LAMMPS binary not executable: ${LMP_BIN}" >&2
    exit 1
fi

#
# Run from the lambda directory.
#

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] LAMMPS working directory: ${RUN_DIR}"

    run_cmd \
        "${LMP_BIN}" \
        -k on g "${KOKKOS_GPUS}" \
        -sf kk \
        -pk kokkos newton on neigh half \
        -in "$(basename "${INPUT_PATH}")" \
        -log "$(basename "${LOG_PATH}")"
else
    (
        cd "${RUN_DIR}"

        echo "LAMMPS working directory: $(pwd)"

        "${LMP_BIN}" \
            -k on g "${KOKKOS_GPUS}" \
            -sf kk \
            -pk kokkos newton on neigh half \
            -in "$(basename "${INPUT_PATH}")" \
            -log "$(basename "${LOG_PATH}")"
    )
fi

if [[ "${DRY_RUN}" != "1" && -f "${ENERGY_LOG_RAW_PATH}" ]]; then
    run_cmd \
        python \
        "${SCRIPT_DIR}/merge_lammps_fep_ti_energy_log.py" \
        --energy-log-raw "${ENERGY_LOG_RAW_PATH}" \
        --temperature-log "${TEMPERATURE_LOG_PATH}" \
        --output "${ENERGY_LOG_PATH}"
fi

if [[ "${DRY_RUN}" != "1" && -f "${LOG_PATH}" ]]; then
    python - "${LOG_PATH}" <<'PY'
import re
import sys

path = sys.argv[1]

pattern = re.compile(
    r"Loop time of ([0-9.eE+-]+) on .* for (\d+) steps"
)

matches = []

with open(path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        match = pattern.search(line)

        if not match:
            continue

        seconds = float(match.group(1))
        steps = int(match.group(2))

        if steps > 0:
            matches.append((seconds, steps))

if matches:
    seconds, steps = matches[-1]

    print(
        "production_ms_per_step="
        f"{seconds * 1000.0 / steps:.6g}"
    )
else:
    print("production_ms_per_step=unavailable")
PY
fi
