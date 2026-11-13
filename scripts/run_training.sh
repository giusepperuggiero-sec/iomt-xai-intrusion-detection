#!/usr/bin/env bash
# ==============================================================================
# SINGLE-SEED TRAINING LAUNCHER FOR THE 3 MODELS (seed=0, T=10, F=6, 50 EPOCHS)
# Project: Group 6 - Marotta League - XAI for IoMT (DIETI UNINA)
# ==============================================================================
# Runs deterministic training and testing for the 3 models with Focal Loss
# (gamma=2, alpha=0.5) and STABLE protocol (gradient clipping, ReduceLROnPlateau
# keeping best_model weights, and learning rate warm-up for Transformers only).
#
# Usage:
#   bash scripts/run_training.sh --all --gpu
#   bash scripts/run_training.sh --model lopez17 --gpu
#   bash scripts/run_training.sh --model transformer --gpu
#   bash scripts/run_training.sh --model meddefender --gpu
#   bash scripts/run_training.sh --all --cpu --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
MODELS_DIR="${RESULTS_DIR}/models"

mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}/metrics" "${RESULTS_DIR}/figures"
cd "${REPO_ROOT}"

# Frozen experimental protocol parameters
SEED=0
NUM_PKTS=10
FIELDS="PL IAT DIR WIN TTL FLG"
NUM_CLASSES=13
BATCH_SIZE=64
NEPOCHS=50
FOCAL_GAMMA=2.0
FOCAL_ALPHA=0.5
WARMUP_EPOCHS=3
WARMUP_LR_FACTOR=0.1
CLIPPING=1.0
LR_PATIENCE=5
NUM_WORKERS=2

USE_GPU=1
DRY_RUN=0
TARGET_MODEL="all"

while [ $# -gt 0 ]; do
    case "$1" in
        --all)         TARGET_MODEL="all"; shift ;;
        --model)       TARGET_MODEL="$2"; shift 2 ;;
        --gpu)         USE_GPU=1; shift ;;
        --cpu)         USE_GPU=0; shift ;;
        --epochs)      NEPOCHS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --workers)     NUM_WORKERS="$2"; shift 2 ;;
        *)             echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

GPU_FLAG=""
if [ "${USE_GPU}" = "1" ]; then
    GPU_FLAG="--gpu"
fi

run_single_model() {
    local NET_NAME="$1"
    local EXP_NAME="$2"
    local EXTRA_NET_ARGS="$3"
    local LOG_FILE="${RESULTS_DIR}/train_${EXP_NAME}.log"

    echo ""
    echo "======================================================================"
    echo " STARTING BATCH TRAINING: ${NET_NAME} (Seed ${SEED}, T=${NUM_PKTS}, ${NEPOCHS} Epochs)"
    echo " Started:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Log:      ${LOG_FILE}"
    echo "======================================================================"

    local CMD="python src/main.py \
        --exp-name \"${EXP_NAME}\" \
        --results-path \"${RESULTS_DIR}\" \
        --datasets cic-iomt \
        --fields ${FIELDS} \
        --num-pkts ${NUM_PKTS} \
        --batch-size ${BATCH_SIZE} \
        --nepochs ${NEPOCHS} \
        --save-models \
        --network ${NET_NAME} \
        --approach scratch \
        --seed ${SEED} \
        --last-class -1 \
        --num-workers ${NUM_WORKERS} \
        --nc-first-task ${NUM_CLASSES} \
        --nc-incr-tasks 1 \
        --stop-at-task 1 \
        --focal-gamma ${FOCAL_GAMMA} \
        --focal-alpha ${FOCAL_ALPHA} \
        --clipping ${CLIPPING} \
        --lr-patience ${LR_PATIENCE} \
        ${GPU_FLAG} \
        ${EXTRA_NET_ARGS}"

    echo "Command: ${CMD}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY-RUN] No command executed."
        return 0
    fi

    local START_T=$(date +%s)
    eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
    local EXIT_CODE=${PIPESTATUS[0]}
    local DURATION=$(($(date +%s) - START_T))

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[SUCCESS] Training ${NET_NAME} completed in $((DURATION / 60))m $((DURATION % 60))s."
        echo "TRAINING_TIME_S=${DURATION}" >> "${LOG_FILE}"  # parsed by Notebook 02 (Table 1, Training Time)

        # Copy best validation checkpoint to results/models/{NET_NAME}/best_model.pth
        local CKPT_DIR="${RESULTS_DIR}/cic-iomt_scratch_${EXP_NAME}/models"
        if [ -d "${CKPT_DIR}" ]; then
            local LATEST_CKPT=$(find "${CKPT_DIR}" -name "*.ckpt" 2>/dev/null | sort -V | tail -n 1 || true)
            if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}" ]; then
                mkdir -p "${MODELS_DIR}/${NET_NAME}"
                cp -f "${LATEST_CKPT}" "${MODELS_DIR}/${NET_NAME}/best_model.pth"
                echo "[OK] Checkpoint saved as: ${MODELS_DIR}/${NET_NAME}/best_model.pth"
            fi
        fi
    else
        echo "[ERROR] Training ${NET_NAME} failed with exit code ${EXIT_CODE}."
        exit ${EXIT_CODE}
    fi
}

echo "======================================================================"
echo " BATCH JOB TRAINING PIPELINE - XAI FOR IoMT"
echo " Target: ${TARGET_MODEL} | GPU: ${USE_GPU} | Epochs: ${NEPOCHS}"
echo "======================================================================"

# Learning rate warm-up enabled for Transformer models only (AdamW)
TRANSFORMER_WARMUP="--warmup-nepochs ${WARMUP_EPOCHS} --warmup-lr-factor ${WARMUP_LR_FACTOR}"

case "${TARGET_MODEL,,}" in
    lopez17|cnn)
        run_single_model "Lopez17CNN" "S2_R01_Lopez17CNN_multi_10pkts_seed0" "--first-momentum 0.9"
        ;;
    transformer|biflow)
        run_single_model "BiflowTransformer" "S2_R02_BiflowTransformer_multi_10pkts_seed0" \
            "--nlayers 2 --nheads 4 --d-model 128 --dim-feedforward 512 ${TRANSFORMER_WARMUP}"
        ;;
    meddefender)
        run_single_model "MedDefenderTransformer" "S2_R03_MedDefender_multi_10pkts_seed0" \
            "--nlayers 1 --nheads 4 --d-model 128 --dim-feedforward 512 ${TRANSFORMER_WARMUP}"
        ;;
    all)
        run_single_model "Lopez17CNN" "S2_R01_Lopez17CNN_multi_10pkts_seed0" "--first-momentum 0.9"
        run_single_model "BiflowTransformer" "S2_R02_BiflowTransformer_multi_10pkts_seed0" \
            "--nlayers 2 --nheads 4 --d-model 128 --dim-feedforward 512 ${TRANSFORMER_WARMUP}"
        run_single_model "MedDefenderTransformer" "S2_R03_MedDefender_multi_10pkts_seed0" \
            "--nlayers 1 --nheads 4 --d-model 128 --dim-feedforward 512 ${TRANSFORMER_WARMUP}"
        ;;
    *)
        echo "[ERROR] Unrecognized model: ${TARGET_MODEL}. Choose from: lopez17, transformer, meddefender, all."
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo " BATCH JOB COMPLETED! All models are ready in: ${MODELS_DIR}"
echo " Open notebooks/02_train_benchmark.ipynb to inspect metrics and confusion matrices."
echo "======================================================================"
