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
RESTART_INTERVAL="${RESTART_INTERVAL:-10000}"

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
  bash run_lammps_mlp_md_restart.sh --run-dir PATH [options]

Required:
  --run-dir PATH

Step modes:
  --steps N
  --steps-mode total
  --steps-mode additional
  --total-steps N
  --additional-steps N

Other:
  --restart-file PATH
  --output-dir PATH
  --restart-interval N
  --cuda-visible-devices IDS
  --prepare-only
  --dry-run
EOF
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir|--source-run-dir)
            SOURCE_RUN_DIR="$2"
            shift 2
            ;;

        --restart-file|--restart)
            RESTART_FILE="$2"
            shift 2
            ;;

        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;

        --steps)
            STEPS="$2"
            shift 2
            ;;

        --steps-mode)
            STEPS_MODE="$2"
            shift 2
            ;;

        --total-steps)
            STEPS_MODE="total"
            STEPS="$2"
            shift 2
            ;;

        --additional-steps)
            STEPS_MODE="additional"
            STEPS="$2"
            shift 2
            ;;

        --restart-interval)
            RESTART_INTERVAL="$2"
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


if [[ -z "${SOURCE_RUN_DIR}" ]]; then
    echo "--run-dir is required." >&2
    exit 2
fi


if [[ "${STEPS_MODE}" != "total" &&
      "${STEPS_MODE}" != "additional" ]]; then

    echo \
        "--steps-mode must be total or additional: ${STEPS_MODE}" \
        >&2

    exit 2
fi


SOURCE_RUN_DIR="$(cd "${SOURCE_RUN_DIR}" && pwd)"


SOURCE_METADATA="${SOURCE_RUN_DIR}/prepare_metadata.json"

SOURCE_CONFIG="${SOURCE_RUN_DIR}/run_lammps_mlp_md_config.txt"


if [[ ! -f "${SOURCE_METADATA}" ]]; then
    echo "Missing ${SOURCE_METADATA}" >&2
    exit 1
fi


if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "Missing ${SOURCE_CONFIG}" >&2
    exit 1
fi


config_value() {
    awk \
        -F= \
        -v key="$1" \
        '$1 == key { sub(/^[^=]*=/, ""); print; exit }' \
        "${SOURCE_CONFIG}"
}


latest_restart=""


restart_1="${SOURCE_RUN_DIR}/lmp.restart.1"

restart_2="${SOURCE_RUN_DIR}/lmp.restart.2"


if [[ -f "${restart_1}" &&
      -f "${restart_2}" ]]; then

    if [[ "${restart_1}" -nt "${restart_2}" ]]; then
        latest_restart="${restart_1}"
    else
        latest_restart="${restart_2}"
    fi

elif [[ -f "${restart_1}" ]]; then
    latest_restart="${restart_1}"

elif [[ -f "${restart_2}" ]]; then
    latest_restart="${restart_2}"
fi


if [[ -z "${RESTART_FILE}" ]]; then
    RESTART_FILE="${latest_restart}"
fi


if [[ -z "${RESTART_FILE}" ||
      ! -f "${RESTART_FILE}" ]]; then

    echo "No usable restart file found." >&2
    exit 1
fi


RESTART_FILE="$(
    cd "$(dirname "${RESTART_FILE}")"

    printf \
        '%s/%s\n' \
        "$(pwd)" \
        "$(basename "${RESTART_FILE}")"
)"


if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SOURCE_RUN_DIR}"
fi


mkdir -p "${OUTPUT_DIR}"

OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"


STORED_CKPT="$(
    config_value checkpoint || true
)"

CKPT="${MATTERSIM_CKPT:-${CKPT:-${STORED_CKPT}}}"


TEMPERATURE="$(
    config_value temperature
)"

TIMESTEP_FS="$(
    config_value timestep_fs
)"

FRICTION_FS_INV="$(
    config_value friction_fs_inv
)"

ORIGINAL_STEPS="$(
    config_value steps
)"

THERMO_INTERVAL="$(
    config_value thermo_interval
)"

DUMP_INTERVAL="$(
    config_value dump_interval
)"

XTC_DUMP_INTERVAL="$(
    config_value xtc_dump_interval
)"


if [[ -z "${STEPS}" ]]; then
    STEPS="${ORIGINAL_STEPS}"
fi


if ! [[ "${STEPS}" =~ ^[0-9]+$ ]]; then
    echo "--steps must be a non-negative integer" >&2
    exit 2
fi


if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda setup script not found: ${CONDA_SH}" >&2
    exit 1
fi


if [[ -z "${CKPT}" ||
      ! -f "${CKPT}" ]]; then

    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
fi


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


max_continuation_index=0


shopt -s nullglob


for candidate in \
    "${OUTPUT_DIR}"/log.normal.restart[0-9]*.lammps
do
    filename="${candidate##*/}"

    suffix="${filename#log.normal.restart}"

    suffix="${suffix%.lammps}"

    if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
        suffix_number=$((10#${suffix}))

        if (( suffix_number > max_continuation_index )); then
            max_continuation_index="${suffix_number}"
        fi
    fi
done


shopt -u nullglob


continuation_index=$((max_continuation_index + 1))

CONTINUATION_TAG="restart${continuation_index}"


MODEL_PATH="${OUTPUT_DIR}/mattertune-mattersim-normal-${CONTINUATION_TAG}.pt"

INPUT_PATH="${OUTPUT_DIR}/in.normal.${CONTINUATION_TAG}.lammps"

METADATA_PATH="${OUTPUT_DIR}/prepare_${CONTINUATION_TAG}_metadata.json"

FINAL_DATA_PATH="${OUTPUT_DIR}/final_normal_${CONTINUATION_TAG}.data"

DUMP_PATH="${OUTPUT_DIR}/traj_normal_${CONTINUATION_TAG}.lammpstrj"

XTC_PATH="${OUTPUT_DIR}/traj_normal_${CONTINUATION_TAG}.xtc"

LOG_PATH="${OUTPUT_DIR}/log.normal.${CONTINUATION_TAG}.lammps"


RESTART_PATH_1="${OUTPUT_DIR}/lmp.restart.1"

RESTART_PATH_2="${OUTPUT_DIR}/lmp.restart.2"


run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'

    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}


echo "SOURCE_RUN_DIR=${SOURCE_RUN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESTART_FILE=${RESTART_FILE}"
echo "STEPS_MODE=${STEPS_MODE}"
echo "STEPS=${STEPS}"
echo "RESTART_INTERVAL=${RESTART_INTERVAL}"
echo "CKPT=${CKPT}"


EXPORT_ARGS=()


if [[ "${STRICT}" == "1" ]]; then
    EXPORT_ARGS+=(--strict)
fi


if [[ "${NO_COMPILE}" == "1" ]]; then
    EXPORT_ARGS+=(--no-compile)
fi


if [[ ! -f "${MODEL_PATH}" ||
      "${FORCE_EXPORT}" == "1" ]]; then

    run_cmd \
        python \
        "${MATTERTUNE_DIR}/examples/elec-Li-new/enhance-V1/export_mattersim_lammps_mliap.py" \
        "${CKPT}" \
        "${MODEL_PATH}" \
        --device "${EXPORT_DEVICE}" \
        "${EXPORT_ARGS[@]}"

else
    echo "Reusing existing model: ${MODEL_PATH}"
fi


LOCAL_RESTART="${OUTPUT_DIR}/$(basename "${RESTART_FILE}")"


if [[ "${RESTART_FILE}" != "${LOCAL_RESTART}" ]]; then
    run_cmd \
        cp \
        "${RESTART_FILE}" \
        "${LOCAL_RESTART}"
fi


run_cmd \
    python \
    "${SCRIPT_DIR}/prepare_lammps_normal_restart_input.py" \
    --restart "${LOCAL_RESTART}" \
    --source-metadata "${SOURCE_METADATA}" \
    --input "${INPUT_PATH}" \
    --model "${MODEL_PATH}" \
    --metadata "${METADATA_PATH}" \
    --temperature "${TEMPERATURE}" \
    --timestep-fs "${TIMESTEP_FS}" \
    --friction-fs-inv "${FRICTION_FS_INV}" \
    --steps "${STEPS}" \
    --steps-mode "${STEPS_MODE}" \
    --thermo-interval "${THERMO_INTERVAL}" \
    --dump-interval "${DUMP_INTERVAL}" \
    --xtc-dump-interval "${XTC_DUMP_INTERVAL}" \
    --final-data "${FINAL_DATA_PATH}" \
    --dump "${DUMP_PATH}" \
    --xtc-dump "${XTC_PATH}" \
    --restart-interval "${RESTART_INTERVAL}" \
    --restart-path-1 "${RESTART_PATH_1}" \
    --restart-path-2 "${RESTART_PATH_2}"


if [[ "${RUN_MD}" != "1" ]]; then
    echo "Prepared restart files; skipping MD."
    exit 0
fi


if [[ ! -x "${LMP_BIN}" ]]; then
    echo "LAMMPS binary not executable: ${LMP_BIN}" >&2
    exit 1
fi


if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ cd ${OUTPUT_DIR}"

    echo \
        "+ ${LMP_BIN} -k on g ${KOKKOS_GPUS} " \
        "-sf kk -pk kokkos newton on neigh half " \
        "-in $(basename "${INPUT_PATH}") " \
        "-log $(basename "${LOG_PATH}")"

else
    (
        cd "${OUTPUT_DIR}"

        "${LMP_BIN}" \
            -k on g "${KOKKOS_GPUS}" \
            -sf kk \
            -pk kokkos newton on neigh half \
            -in "$(basename "${INPUT_PATH}")" \
            -log "$(basename "${LOG_PATH}")"
    )
fi


echo
echo "Continuation finished."
echo "Target steps=${STEPS}"
echo "Steps mode=${STEPS_MODE}"
echo "Log=${LOG_PATH}"
echo "XTC=${XTC_PATH}"
echo "Restart files=${RESTART_PATH_1}, ${RESTART_PATH_2}"
