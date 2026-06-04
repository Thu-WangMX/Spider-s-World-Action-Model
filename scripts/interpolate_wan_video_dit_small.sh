#!/usr/bin/env bash
set -euo pipefail

ROOT="/data73/mingxinwang"
PROJECT_DIR="${ROOT}/Spider-s-World-Action-Model"
CONDA_SH="/data73/envs/miniconda3/etc/profile.d/conda.sh"
LOG_DIR="${ROOT}/view_dino_guard_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/interpolate_wan_video_dit_small_${STAMP}.log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"
source "${CONDA_SH}"
conda activate spiderwam

export PYTHON="/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10"
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_DIR}/checkpoints"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="/tmp/matplotlib-cache"

"${PYTHON}" scripts/preprocess_wan_video_dit_small.py \
  --model-config configs/model/fastwam_smallvideo.yaml \
  --output checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16 \
  --apply-alpha-scaling true \
  2>&1 | tee "${LOG_FILE}"

ls -lh checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
echo "Log: ${LOG_FILE}"
