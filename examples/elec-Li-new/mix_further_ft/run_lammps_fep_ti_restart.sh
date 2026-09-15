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

#
# Internal implementation detail:
# refresh canonical FEP-TI logs every 2 minutes.
#
PERIODIC_MERGE_SECONDS=120
PERIODIC_MERGE_PID=""

usage() {
    cat <<'EOF'
Usage:
  bash run_lammps_fep_ti_restart.sh --run-dir PATH [options]

Required:
  --run-dir PATH

Step modes:
  --steps-mode total
  --steps-mode additional
  --steps N

Other options:
  --restart-file PATH
  --output-dir PATH
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

if [[ "${STEPS_MODE}" != "total" && "${STEPS_MODE}" != "additional" ]]; then
    echo "--steps-mode must be total or additional: ${STEPS_MODE}" >&2
    exit 2
fi

SOURCE_RUN_DIR="$(cd "${SOURCE_RUN_DIR}" && pwd)"

SOURCE_METADATA="${SOURCE_RUN_DIR}/prepare_metadata.json"
SOURCE_CONFIG="${SOURCE_RUN_DIR}/run_lammps_fep_ti_config.txt"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_lammps_fep_ti_energy_log.py"

if [[ ! -f "${SOURCE_METADATA}" || ! -f "${SOURCE_CONFIG}" ]]; then
    echo "Missing prepare_metadata.json or run_lammps_fep_ti_config.txt under ${SOURCE_RUN_DIR}" >&2
    exit 1
fi

if [[ ! -f "${MERGE_SCRIPT}" ]]; then
    echo "Merge script not found: ${MERGE_SCRIPT}" >&2
    exit 1
fi

config_value() {
    awk -F= -v key="$1" \
        '$1 == key { sub(/^[^=]*=/, ""); print; exit }' \
        "${SOURCE_CONFIG}"
}

#
# ----------------------------------------------------------------------
# Pick the newest available restart file.
# ----------------------------------------------------------------------
#

latest_restart=""

restart_1="${SOURCE_RUN_DIR}/lmp.restart.1"
restart_2="${SOURCE_RUN_DIR}/lmp.restart.2"

if [[ -f "${restart_1}" && -f "${restart_2}" ]]; then
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

if [[ -z "${RESTART_FILE}" || ! -f "${RESTART_FILE}" ]]; then
    echo "No usable restart file: ${RESTART_FILE}" >&2
    exit 1
fi

RESTART_FILE="$(
    cd "$(dirname "${RESTART_FILE}")"
    printf '%s/%s\n' "$(pwd)" "$(basename "${RESTART_FILE}")"
)"

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${SOURCE_RUN_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

#
# ----------------------------------------------------------------------
# Portable checkpoint handling.
#
# Priority:
#   1. MATTERSIM_CKPT from current cluster
#   2. CKPT exported by current job
#   3. checkpoint stored in old run config
# ----------------------------------------------------------------------
#

STORED_CKPT="$(config_value checkpoint || true)"
CKPT="${MATTERSIM_CKPT:-${CKPT:-${STORED_CKPT}}}"

#
# ----------------------------------------------------------------------
# Read scientific configuration.
# ----------------------------------------------------------------------
#

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

stored_energy_log_path="$(config_value energy_log_path || true)"
stored_energy_log_raw_path="$(config_value energy_log_raw_path || true)"
stored_temperature_log_path="$(config_value temperature_log_path || true)"

#
# Old configs may contain /projects/... or another cluster's paths.
# Only keep the run-local basename.
#

if [[ -n "${stored_energy_log_path}" ]]; then
    ENERGY_LOG_PATH="${OUTPUT_DIR}/$(basename "${stored_energy_log_path}")"
else
    ENERGY_LOG_PATH="${OUTPUT_DIR}/fep-ti-energy.log"
fi

if [[ -n "${stored_energy_log_raw_path}" ]]; then
    ENERGY_LOG_RAW_PATH="${OUTPUT_DIR}/$(basename "${stored_energy_log_raw_path}")"
else
    ENERGY_LOG_RAW_PATH="${OUTPUT_DIR}/fep-ti-energy.raw.csv"
fi

if [[ -n "${stored_temperature_log_path}" ]]; then
    TEMPERATURE_LOG_PATH="${OUTPUT_DIR}/$(basename "${stored_temperature_log_path}")"
else
    TEMPERATURE_LOG_PATH="${OUTPUT_DIR}/fep-ti-temperature.csv"
fi

if [[ -z "${STEPS}" ]]; then
    STEPS="${ORIGINAL_STEPS}"
fi

if ! [[ "${STEPS}" =~ ^[0-9]+$ ]]; then
    echo "--steps must be a non-negative integer" >&2
    exit 2
fi

#
# ----------------------------------------------------------------------
# Runtime environment.
# ----------------------------------------------------------------------
#

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda setup script not found: ${CONDA_SH}" >&2
    exit 1
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2

    if [[ -n "${STORED_CKPT}" ]]; then
        echo "Original config checkpoint: ${STORED_CKPT}" >&2
    fi

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

#
# ----------------------------------------------------------------------
# Recover orphan restart energy segments.
#
# If an earlier Slurm job died before finalization, any remaining
# .segment.csv file is merged automatically before starting another
# continuation.
# ----------------------------------------------------------------------
#

recover_orphan_segments() {
    local segment
    local restart_step_file
    local recovered=0

    shopt -s nullglob
    local segments=(
        "${OUTPUT_DIR}"/.fep-ti-energy-restart*.segment.csv
    )
    shopt -u nullglob

    if (( ${#segments[@]} == 0 )); then
        return 0
    fi

    while IFS= read -r segment; do
        [[ -n "${segment}" ]] || continue

        restart_step_file="${segment%.segment.csv}.restart-step"

        if [[ ! -s "${segment}" ]]; then
            rm -f "${segment}"
            continue
        fi

        if [[ ! -s "${restart_step_file}" ]]; then
            echo "WARNING: orphan FEP-TI segment has no restart-step file; keeping it:" >&2
            echo "  ${segment}" >&2
            continue
        fi

        echo "[RECOVERY] merging orphan FEP-TI segment: $(basename "${segment}")"

        python "${MERGE_SCRIPT}" \
            --energy-log-raw "${ENERGY_LOG_RAW_PATH}" \
            --append-raw-segment "${segment}" \
            --restart-step-file "${restart_step_file}" \
            --timestep-fs "${TIMESTEP_FS}" \
            --temperature-log "${TEMPERATURE_LOG_PATH}" \
            --output "${ENERGY_LOG_PATH}"

        recovered=$((recovered + 1))

    done < <(
        printf '%s\n' "${segments[@]}" | sort -V
    )

    if (( recovered > 0 )); then
        echo "[RECOVERY] recovered ${recovered} restart energy segment(s)."
    fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
    recover_orphan_segments
fi

#
# ----------------------------------------------------------------------
# Lambda naming.
# ----------------------------------------------------------------------
#

LAMBDA_TAG="$(
    python -c \
        'import sys; print(f"lambda{int(round(float(sys.argv[1])*1000)):03d}")' \
        "${LAMBDA_VALUE}"
)"

#
# ----------------------------------------------------------------------
# Determine next restart continuation number.
# ----------------------------------------------------------------------
#

max_continuation_index=0

shopt -s nullglob

for candidate in \
    "${OUTPUT_DIR}"/log."${LAMBDA_TAG}".restart[0-9]*.lammps
do
    filename="${candidate##*/}"

    suffix="${filename#*.restart}"
    suffix="${suffix%.lammps}"

    if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
        suffix_number=$((10#${suffix}))

        if (( suffix_number > max_continuation_index )); then
            max_continuation_index="${suffix_number}"
        fi
    fi
done

shopt -u nullglob

#
# IMPORTANT:
# Correct Bash arithmetic expansion.
#
continuation_index=$((max_continuation_index + 1))

CONTINUATION_TAG="restart${continuation_index}"

#
# ----------------------------------------------------------------------
# Current continuation files.
# ----------------------------------------------------------------------
#

MODEL_PATH="${OUTPUT_DIR}/ghost-target-${LAMBDA_TAG}-${CONTINUATION_TAG}.pt"

INPUT_PATH="${OUTPUT_DIR}/in.${LAMBDA_TAG}.${CONTINUATION_TAG}.lammps"

METADATA_PATH="${OUTPUT_DIR}/prepare_${CONTINUATION_TAG}_metadata.json"

ENERGY_LOG_SEGMENT_PATH="${OUTPUT_DIR}/.fep-ti-energy-${CONTINUATION_TAG}.segment.csv"

RESTART_STEP_PATH="${OUTPUT_DIR}/.fep-ti-energy-${CONTINUATION_TAG}.restart-step"

FINAL_DATA_PATH="${OUTPUT_DIR}/final_${LAMBDA_TAG}_${CONTINUATION_TAG}.data"

DUMP_PATH="${OUTPUT_DIR}/traj_${LAMBDA_TAG}_${CONTINUATION_TAG}.lammpstrj"

XTC_PATH="${OUTPUT_DIR}/traj_${LAMBDA_TAG}_${CONTINUATION_TAG}.xtc"

LOG_PATH="${OUTPUT_DIR}/log.${LAMBDA_TAG}.${CONTINUATION_TAG}.lammps"

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

if [[ "${DRY_RUN}" != "1" ]]; then
    rm -f "${RESTART_STEP_PATH}"
else
    echo "[dry-run] would use ${OUTPUT_DIR}"
fi

echo "RESTART_FILE=${RESTART_FILE}"
echo "NEXT_RESTART_FILES=${RESTART_PATH_1},${RESTART_PATH_2}"
echo "STEPS_MODE=${STEPS_MODE}"
echo "STEPS=${STEPS}"
echo "CKPT=${CKPT}"
echo "XTC_PATH=${XTC_PATH}"
echo "ENERGY_LOG_PATH=${ENERGY_LOG_PATH}"
echo "ENERGY_LOG_RAW_PATH=${ENERGY_LOG_RAW_PATH}"
echo "ENERGY_LOG_SEGMENT_PATH=${ENERGY_LOG_SEGMENT_PATH}"
echo "RESTART_STEP_PATH=${RESTART_STEP_PATH}"
echo "TEMPERATURE_LOG_PATH=${TEMPERATURE_LOG_PATH}"

#
# ----------------------------------------------------------------------
# Export continuation model.
# ----------------------------------------------------------------------
#

EXPORT_ARGS=()

if [[ "${STRICT}" == "1" ]]; then
    EXPORT_ARGS+=(--strict)
fi

if [[ "${NO_COMPILE}" == "1" ]]; then
    EXPORT_ARGS+=(--no-compile)
fi

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
    --energy-log-path "${ENERGY_LOG_SEGMENT_PATH}" \
    --energy-log-interval "${ENERGY_LOG_INTERVAL}" \
    --energy-log-timestep-fs "${TIMESTEP_FS}" \
    --device "${EXPORT_DEVICE}" \
    "${EXPORT_ARGS[@]}"

#
# ----------------------------------------------------------------------
# Generate restart input.
# ----------------------------------------------------------------------
#

run_cmd \
    python \
    "${SCRIPT_DIR}/prepare_lammps_ghost_target_restart_input.py" \
    --restart "${RESTART_FILE}" \
    --source-metadata "${SOURCE_METADATA}" \
    --input "${INPUT_PATH}" \
    --model "${MODEL_PATH}" \
    --metadata "${METADATA_PATH}" \
    --restart-step-file "${RESTART_STEP_PATH}" \
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
    --temperature-log "${TEMPERATURE_LOG_PATH}" \
    --temperature-log-interval "${ENERGY_LOG_INTERVAL}" \
    --restart-interval "${RESTART_INTERVAL}" \
    --restart-path-1 "${RESTART_PATH_1}" \
    --restart-path-2 "${RESTART_PATH_2}"

if [[ "${RUN_MD}" != "1" ]]; then
    echo "Prepared restart files; skipping MD because RUN_MD=${RUN_MD}."
    exit 0
fi

if [[ ! -x "${LMP_BIN}" ]]; then
    echo "LAMMPS binary not executable: ${LMP_BIN}" >&2
    exit 1
fi

#
# ----------------------------------------------------------------------
# Merge active restart segment.
#
# keep_segment=1:
#     periodic snapshot while LAMMPS is still writing
#
# keep_segment=0:
#     final merge after MD exits
# ----------------------------------------------------------------------
#

merge_energy_segment() {
    local keep_segment="${1:-0}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    if [[ ! -f "${ENERGY_LOG_SEGMENT_PATH}" ]]; then
        return 0
    fi

    if [[ ! -s "${ENERGY_LOG_SEGMENT_PATH}" ]]; then
        if [[ "${keep_segment}" != "1" ]]; then
            rm -f "${ENERGY_LOG_SEGMENT_PATH}"
        fi

        return 0
    fi

    if [[ ! -s "${RESTART_STEP_PATH}" ]]; then
        if [[ "${keep_segment}" != "1" ]]; then
            echo "WARNING: energy segment exists but restart step file is missing:" >&2
            echo "  ${RESTART_STEP_PATH}" >&2
            echo "Keeping segment for automatic recovery:" >&2
            echo "  ${ENERGY_LOG_SEGMENT_PATH}" >&2
        fi

        return 1
    fi

    local args=(
        python
        "${MERGE_SCRIPT}"
        --energy-log-raw
        "${ENERGY_LOG_RAW_PATH}"
        --append-raw-segment
        "${ENERGY_LOG_SEGMENT_PATH}"
        --restart-step-file
        "${RESTART_STEP_PATH}"
        --timestep-fs
        "${TIMESTEP_FS}"
        --temperature-log
        "${TEMPERATURE_LOG_PATH}"
        --output
        "${ENERGY_LOG_PATH}"
    )

    if [[ "${keep_segment}" == "1" ]]; then
        args+=(
            --keep-segment
            --quiet
        )
    fi

    "${args[@]}"
}

#
# ----------------------------------------------------------------------
# Periodic 2-minute refresh.
# ----------------------------------------------------------------------
#

periodic_merge_loop() {
    while true; do
        sleep "${PERIODIC_MERGE_SECONDS}"

        if [[ -s "${ENERGY_LOG_SEGMENT_PATH}" &&
              -s "${RESTART_STEP_PATH}" ]]; then

            merge_energy_segment 1 \
                || echo \
                    "WARNING: periodic restart FEP-TI energy refresh failed." \
                    >&2
        fi
    done
}

start_periodic_merge() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    periodic_merge_loop &

    PERIODIC_MERGE_PID=$!

    echo "FEP-TI energy log auto-refresh enabled."
}

stop_periodic_merge() {
    if [[ -z "${PERIODIC_MERGE_PID}" ]]; then
        return 0
    fi

    kill "${PERIODIC_MERGE_PID}" \
        2>/dev/null \
        || true

    wait "${PERIODIC_MERGE_PID}" \
        2>/dev/null \
        || true

    PERIODIC_MERGE_PID=""
}

#
# ----------------------------------------------------------------------
# Final merge on shell exit.
# ----------------------------------------------------------------------
#

finalize() {
    local status=$?
    local merge_status=0

    trap - EXIT
    set +e

    stop_periodic_merge

    echo "Finalizing FEP-TI energy log..."

    merge_energy_segment 0
    merge_status=$?

    if (( merge_status != 0 )); then
        echo \
            "WARNING: final energy merge failed with status ${merge_status}" \
            >&2

        if (( status == 0 )); then
            status="${merge_status}"
        fi
    fi

    if (( status != 0 )); then
        echo \
            "LAMMPS/restart workflow exited with status ${status}" \
            >&2
    fi

    exit "${status}"
}

trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

start_periodic_merge

#
# ----------------------------------------------------------------------
# Run LAMMPS from OUTPUT_DIR so generated input can use basenames only.
# ----------------------------------------------------------------------
#

if [[ "${DRY_RUN}" == "1" ]]; then

    run_cmd \
        "${LMP_BIN}" \
        -k on g "${KOKKOS_GPUS}" \
        -sf kk \
        -pk kokkos newton on neigh half \
        -in "$(basename "${INPUT_PATH}")" \
        -log "$(basename "${LOG_PATH}")"

else

    (
        cd "${OUTPUT_DIR}"

        echo "LAMMPS working directory: $(pwd)"

        "${LMP_BIN}" \
            -k on g "${KOKKOS_GPUS}" \
            -sf kk \
            -pk kokkos newton on neigh half \
            -in "$(basename "${INPUT_PATH}")" \
            -log "$(basename "${LOG_PATH}")"
    )

fi

exit 0
