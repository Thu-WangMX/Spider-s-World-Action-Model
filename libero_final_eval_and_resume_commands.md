# LIBERO eval and resume commands

当前机器建议直接加载真实 conda hook。`source ~/.bashrc && conda activate spiderwam`
可能报 `CondaError: Run 'conda init' before 'conda activate'`。

通用环境头：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

## 1. Eval latest ckpt, 30 trials, 8 GPUs

评测某个 run 目录下最新的 `.pt` 权重。

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/finetune_fixed_from_step028930_lr1e-5_4k_2026-05-28_04-19-48
LATEST_CKPT=$(ls -1 "$RUN_DIR"/checkpoints/weights/step_*.pt | sort -V | tail -n 1)

$PYTHON experiments/libero/run_libero_manager.py \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  ckpt="$LATEST_CKPT" \
  EVALUATION.num_trials=30 \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="./evaluate_results/libero/nopool_latest_30trials_$(basename "$LATEST_CKPT" .pt)" \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=4 \
  model.dino_config.load_backbone=true \
  'model.dino_config.latent_spatial_pool=[1,1]'
```

## 2. Eval specific ckpt

用于评测指定权重，比如 `step_026000.pt`，或者 weight-only fine-tune 新 run 里的
`checkpoints/weights/step_002000.pt`。

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-24_10-27-57
CKPT="$RUN_DIR/checkpoints/weights/step_026000.pt"

$PYTHON experiments/libero/run_libero_manager.py \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  ckpt="$CKPT" \
  EVALUATION.num_trials=30 \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="./evaluate_results/libero/nopool_$(basename "$CKPT" .pt)_30trials" \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=4 \
  model.dino_config.load_backbone=true \
  'model.dino_config.latent_spatial_pool=[1,1]'
```

## 3. Weight-only resume / fine-tune

这是现在实际使用的 14 小时续训方式：加载 `step_028930.pt` 的模型权重，但重新创建
optimizer、scheduler 和 step counter。按之前速度约 `0.11 step/s`，`num_epochs=2`
约等于 `5786` optimizer steps，大概 `14-15` 小时。新 run 的 ckpt 会从
`step_000xxx` 重新编号，例如新 `step_002000` 约等于原始 `step_030930`。

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MASTER_PORT=29541
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-24_10-27-57
LATEST_CKPT="$RUN_DIR/checkpoints/weights/step_028930.pt"

RUN_ID=finetune_from_step028930_lr2e-5_2ep_$(date +%Y-%m-%d_%H-%M-%S) \
bash scripts/train_zero1.sh 8 \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  resume="$LATEST_CKPT" \
  num_epochs=2 \
  batch_size=6 \
  gradient_accumulation_steps=2 \
  num_workers=2 \
  learning_rate=2e-5 \
  wandb.enabled=true \
  wandb.name=libero_dino_s_nopool_from_step028930_lr2e-5_2ep \
  data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact \
  data.train.dino_latent_cache_mode=frame \
  data.train.dino_latent_cache_required=true \
  model.dino_config.load_backbone=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=16 \
  model.loss.lambda_video=0.05 \
  model.loss.lambda_action=5.0
```

日志里看到下面这行是正常的，说明是 weight-only：

```text
Loaded .pt weights only before optimizer/DeepSpeed initialization; optimizer/scheduler/step were not restored
```

## 4. Full-state resume

完整恢复 model、optimizer、scheduler、global step、epoch 和 dataloader progress。
适合训练中断后精确继续。

注意：如果原 run 已经到旧的 `max_steps`，还用旧 `num_epochs` 会直接结束。想继续往后训，
要把总 `num_epochs` 增大，或者显式设置更大的 `max_steps`。但 scheduler state 也会恢复，
如果原 run 已经到末尾，LR 可能仍然处在很低的位置，学习会比较慢。

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MASTER_PORT=29542
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-24_10-27-57
RESUME_STATE="$RUN_DIR/checkpoints/state/step_028930"

RUN_ID=fullstate_resume_from_step028930_to_12ep_$(date +%Y-%m-%d_%H-%M-%S) \
bash scripts/train_zero1.sh 8 \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  resume="$RESUME_STATE" \
  num_epochs=12 \
  batch_size=6 \
  gradient_accumulation_steps=2 \
  num_workers=2 \
  learning_rate=5e-5 \
  wandb.enabled=true \
  wandb.name=libero_dino_s_nopool_fullstate_resume_from_step028930_to_12ep \
  data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact \
  data.train.dino_latent_cache_mode=frame \
  data.train.dino_latent_cache_required=true \
  model.dino_config.load_backbone=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=16 \
  model.loss.lambda_video=0.05 \
  model.loss.lambda_action=5.0
```

日志里看到下面这行说明是 full-state：

```text
Resuming full training state from directory
```

## Notes

- Eval 要用 `model.dino_config.load_backbone=true`，因为 simulator 里要在线从图像算 DINO feature。
- 训练使用缓存好的 DINO latent，应保持 `model.dino_config.load_backbone=false`。
- 当前 batch 设置是 `batch_size=6`、`gradient_accumulation_steps=2`、8 GPU，所以 global batch 是 `96`。
