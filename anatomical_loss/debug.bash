#!/bin/bash
set -euo pipefail

export nnUNet_raw="/fs5/p_masi/liux64/Eye_TOM500_specify/nnUNet_raw"
export nnUNet_preprocessed="/fs5/p_masi/liux64/Eye_TOM500_specify/nnUNet_preprocessed"
export nnUNet_results="/fs5/p_masi/liux64/Eye_TOM500_specify/nnUNet_topology_experiments"
export NNUNET_ANATOMICAL_CONFIG="/fs5/p_masi/liux64/Eye_TOM500_specify/nnUNet_topology_experiments/anatomical_config_debug.json"

LOG_DIR="/fs5/p_masi/liux64/Eye_TOM500_specify/nnUNet_topology_experiments/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_fold2_cuda2_$(date +%Y%m%d_%H%M%S).log"

echo "Using GPU: 2" | tee "$LOG_FILE"
echo "nnUNet_raw: $nnUNet_raw" | tee -a "$LOG_FILE"
echo "nnUNet_preprocessed: $nnUNet_preprocessed" | tee -a "$LOG_FILE"
echo "nnUNet_results: $nnUNet_results" | tee -a "$LOG_FILE"
echo "NNUNET_ANATOMICAL_CONFIG: $NNUNET_ANATOMICAL_CONFIG" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 806 3d_fullres 2 -tr nnUNetTrainer 2>&1 | tee -a "$LOG_FILE"