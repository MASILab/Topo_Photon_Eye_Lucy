#!/bin/bash
# Example training script.
# Usage: bash scripts/train.sh <DATASET_ID> <FOLD> <CONFIG_JSON> [GPU]
#   e.g. bash scripts/train.sh 806 0 configs/anatomical_topology.json 0   # topology-aware model
#        bash scripts/train.sh 806 0 configs/baseline.json 0              # baseline nnU-Net
set -euo pipefail

DATASET_ID=${1:?dataset id}
FOLD=${2:?fold}
CONFIG=${3:?config json}
GPU=${4:-0}

# Set these to your own nnU-Net folders
export nnUNet_raw="${nnUNet_raw:-/path/to/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/path/to/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/path/to/nnUNet_results}"
export NNUNET_ANATOMICAL_CONFIG="$(realpath "$CONFIG")"

LOG_DIR="$nnUNet_results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${DATASET_ID}_fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"

echo "GPU: $GPU | config: $NNUNET_ANATOMICAL_CONFIG" | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" -tr nnUNetTrainer 2>&1 | tee -a "$LOG_FILE"
