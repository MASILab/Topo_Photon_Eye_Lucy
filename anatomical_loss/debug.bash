#!/bin/bash
# Quick debug run: train one fold of the topology-aware model.
# Usage: bash anatomical_loss/debug.bash <DATASET_ID> [FOLD] [GPU]
# For the general training script, see scripts/train.sh.
set -euo pipefail

DATASET_ID=${1:?dataset id}
FOLD=${2:-0}
GPU=${3:-0}

# Set these to your own nnU-Net folders
export nnUNet_raw="${nnUNet_raw:-/path/to/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/path/to/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/path/to/nnUNet_results}"
export NNUNET_ANATOMICAL_CONFIG="$(cd "$(dirname "$0")/.." && pwd)/configs/anatomical_topology.json"

CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" -tr nnUNetTrainer
