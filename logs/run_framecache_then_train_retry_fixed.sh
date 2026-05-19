#!/usr/bin/env bash
set -uo pipefail

cd /data11/wmx/Spider-s-World-Action-Model

export PYTHONPATH=/data11/wmx/Spider-s-World-Action-Model/src:/data11/wmx/Spider-s-World-Action-Model
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WAIT_FOR_GPUS=1
export WAIT_GPU_MAX_USED_MB=1000
export WAIT_GPU_MAX_UTIL=5
export WAIT_GPU_INTERVAL=60
export PYTHON=/data11/wmx/miniconda3/envs/fastwam/bin/python3.10
export OMP_NUM_THREADS=2

CACHE_DIR=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x2_frame_exact

run_cache() {
  local bs="$1"
  local emb="$2"
  echo "[stage 1] precompute DINO frame cache: batch=${bs}, encode_microbatch=${emb}"

  ${PYTHON} -m torch.distributed.run \
    --standalone --nproc_per_node=4 \
    scripts/precompute_dino_latents.py \
    task=libero_dino_s_smallvideo_2cam_224_1e-4 \
    dino_latent_cache_mode=frame \
    dino_latent_cache_dir=${CACHE_DIR} \
    +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
    dino_precompute_batch_size=${bs} \
    dino_precompute_num_workers=8 \
    model.dino_config.encode_microbatch_size=${emb} \
    2>&1 | tee -a logs/precompute_dino_frame_latents_FIXED_b${bs}_emb${emb}_$(date +%Y%m%d_%H%M%S).log
}

CACHE_OK=0
for spec in "128 128" "64 64" "32 32" "16 16"; do
  set -- $spec
  run_cache "$1" "$2"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    CACHE_OK=1
    break
  fi
  echo "[retry] cache failed with rc=${rc}; trying smaller batch..."
  sleep 20
done

if [ "$CACHE_OK" -ne 1 ]; then
  echo "[fatal] DINO cache failed for all retry settings."
  exit 1
fi

echo "[stage 1 done] cache size:"
du -sh "${CACHE_DIR}" || true

run_train() {
  local bs="$1"
  local ga="$2"
  echo "[stage 2] train smallvideo with frame cache: batch_size=${bs}, gradient_accumulation_steps=${ga}"

  bash scripts/train_zero1.sh 4 \
    task=libero_dino_s_smallvideo_2cam_224_1e-4 \
    +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
    data.train.dino_latent_cache_dir=${CACHE_DIR} \
    data.train.dino_latent_cache_mode=frame \
    data.train.dino_latent_cache_required=true \
    model.dino_config.load_backbone=false \
    model.dino_config.latent_spatial_pool=[1,2] \
    model.loss.lambda_video=0.05 \
    model.loss.lambda_action=5.0 \
    learning_rate=5e-5 \
    batch_size=${bs} \
    gradient_accumulation_steps=${ga} \
    wandb.enabled=true \
    wandb.project=fast-wam \
    wandb.name=libero_dino_s_smallvideo_framecache_FIXED_lv0.05_la5_bs${bs}ga${ga} \
    2>&1 | tee -a logs/train_libero_dino_s_smallvideo_framecache_FIXED_bs${bs}ga${ga}_$(date +%Y%m%d_%H%M%S).log
}

TRAIN_OK=0
for spec in "8 2" "4 4" "2 8" "1 16"; do
  set -- $spec
  run_train "$1" "$2"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    TRAIN_OK=1
    break
  fi
  echo "[retry] train failed with rc=${rc}; trying smaller batch..."
  sleep 30
done

if [ "$TRAIN_OK" -ne 1 ]; then
  echo "[fatal] training failed for all retry settings."
  exit 1
fi
