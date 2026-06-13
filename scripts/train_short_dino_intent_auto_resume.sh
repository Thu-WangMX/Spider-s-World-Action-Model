#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangmx2605/.conda/envs/spiderwam/bin/python}"
RUN_NAME="${RUN_NAME:-short_dino_intent_fromscratch_10ep_auto}"
RUN_ROOT="${RUN_ROOT:-/data32/mingxinwang/Spider-s-World-Action-Model/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/libero_dino_s_smallvideo_2cam_224_1e-4/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"
MAX_RETRIES="${MAX_RETRIES:-999}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0}"

DINO_CACHE_DIR="${DINO_CACHE_DIR:-${ROOT_DIR}/data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[auto_resume] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  echo "[auto_resume] Set PYTHON_BIN=/path/to/python before launching." >&2
  exit 1
fi

latest_state_dir() {
  local state_root="${OUTPUT_DIR}/checkpoints/state"
  if [[ ! -d "${state_root}" ]]; then
    return 1
  fi

  find "${state_root}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' -printf '%f\t%p\n' \
    | sort -t $'\t' -k1,1V \
    | tail -n 1 \
    | cut -f2-
}

should_exit=0
trap 'should_exit=1; echo "[auto_resume] received stop signal; exiting after current command."; exit 130' INT TERM

attempt=0
while (( attempt < MAX_RETRIES )); do
  attempt=$((attempt + 1))
  resume_arg="resume=null"
  if state_dir="$(latest_state_dir)"; then
    resume_arg="resume=${state_dir}"
    echo "[auto_resume] attempt=${attempt} using full training state: ${state_dir}"
  else
    echo "[auto_resume] attempt=${attempt} no state checkpoint found; starting from scratch."
  fi

  ts="$(date +%Y-%m-%d_%H-%M-%S)"
  log_file="${LOG_DIR}/train_attempt${attempt}_${ts}.log"
  echo "[auto_resume] output_dir=${OUTPUT_DIR}"
  echo "[auto_resume] log_file=${log_file}"

  set +e
  PYTHON="${PYTHON_BIN}" WAIT_FOR_GPUS="${WAIT_FOR_GPUS}" RUN_ID="${RUN_NAME}" \
    bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
      "task=libero_dino_s_smallvideo_2cam_224_1e-4" \
      "output_dir=${OUTPUT_DIR}" \
      "wandb.enabled=true" \
      "wandb.name=${RUN_NAME}" \
      "batch_size=6" \
      "num_workers=6" \
      "prefetch_factor=1" \
      "persistent_workers=true" \
      "learning_rate=5e-5" \
      "num_epochs=10" \
      "max_steps=null" \
      "gradient_accumulation_steps=2" \
      "weight_decay=1e-2" \
      "save_every=2000" \
      "eval_every=200" \
      "${resume_arg}" \
      "model.loss.lambda_video=0.05" \
      "model.loss.lambda_action=5.0" \
      "model.dino_config.load_backbone=false" \
      "model.dino_config.latent_spatial_pool=[1,1]" \
      "data.train.dino_latent_cache_dir=${DINO_CACHE_DIR}" \
      "data.train.dino_latent_cache_mode=frame_mmap" \
      "data.train.dino_latent_cache_required=true" \
      "+data.train.load_history_dino_latents=true" \
      "+data.train.history_dino_frame_offsets=[-8,-4,0]" \
      "+model.intent_config.enabled=true" \
      "+model.intent_config.history_offsets=[-8,-4,0]" \
      "+model.intent_config.max_history_frames=3" \
      "+model.intent_config.num_intent_tokens=8" \
      "+model.intent_config.resampler_dim=1024" \
      "+model.intent_config.num_resampler_layers=2" \
      "+model.intent_config.num_heads=8" \
      "+model.intent_config.mlp_ratio=4.0" \
      "+model.intent_config.dropout=0.0" \
      2>&1 | tee "${log_file}"
  status=${PIPESTATUS[0]}
  set -e

  if (( status == 0 )); then
    echo "[auto_resume] training finished successfully."
    exit 0
  fi

  if (( should_exit != 0 )); then
    exit "${status}"
  fi

  echo "[auto_resume] training exited with status=${status}."
  echo "[auto_resume] sleeping ${RETRY_SLEEP_SECONDS}s, then retrying from latest state if available."
  sleep "${RETRY_SLEEP_SECONDS}"
done

echo "[auto_resume] reached MAX_RETRIES=${MAX_RETRIES}; giving up." >&2
exit 1
