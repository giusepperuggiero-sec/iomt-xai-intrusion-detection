#!/usr/bin/env bash
# ==============================================================================
# BATCH JOB ENTRYPOINT: Training the 3 Models (Open OnDemand / Jupyter Cluster)
# Project: Group 6 - Marotta League - XAI for IoMT (DIETI UNINA)
# ==============================================================================
# Zero-argument wrapper script to launch batch training for all 3 models
# (Lopez17CNN, BiflowTransformer, MedDefenderTransformer) on an L4 GPU.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}" || { echo "Directory non trovata: ${REPO_ROOT}"; exit 1; }

echo "======================================================================"
echo " AVVIO BATCH JOB: TRAINING 3 MODELLI (Seed 0, T=10, 50 Epoche)"
echo " Inizio: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Repo:   ${REPO_ROOT}"
echo "======================================================================"

bash "${SCRIPT_DIR}/run_training.sh" --all --gpu --epochs 50 "$@"
