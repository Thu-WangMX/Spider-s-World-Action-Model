# SpiderWAM 最新实验上下文

更新时间：2026-06-03

这个文件从 2026-05-29 开始记录最新关键状态。后续新的训练、评测、修 bug 和结论优先追加到这里，旧的长上下文仍保留在 `PROJECT_CONTEXT.md`。

---

## 1. 当前主线

当前重点是 LIBERO 2cam DINO-S smallvideo + view-aware learnable token merge：

```text
task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4
```

核心结构：

```text
DINO-S no-pool latent: [D,T,H,W] = [384,9,14,28]
2 cameras along width: 14x28 -> 2 x 14x14
view-aware patch merge: [1,2,2]
tokens per frame: 392 -> 98
video target still unpatchifies back to [384,9,14,28]
```

初始化方式：

```text
model.video_dit_init_from_wan=false
model.video_dit_pretrained_path=checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
```

不要把 `model.video_dit_init_from_wan=true` 和 `model.video_dit_pretrained_path=...` 同时打开。

---

## 2. 当前 DataLoader 问题与结论

旧 `frame` cache 路径：

```text
data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact
```

规模：

```text
277713 个 frames/*.pt
约 79G
单帧 latent: [384,14,28] bf16, 约 0.287 MiB
每个训练 sample 读取 9 帧
```

多 worker 下容易触发：

```text
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
```

当前判断：

- 不是 viewpatch 模型逻辑直接导致。
- 主因更像是 `pyav` 视频解码 + DINO frame 小文件 `torch.load` 并发太高。
- `torchcodec` 当前不可用，会 fallback 到 `torchvision/pyav`；原因是环境缺 FFmpeg 动态库，如 `libavutil.so.*`。
- `num_workers=4` 快但不稳，`num_workers=3` 也不稳。
- 目前较稳的临时组合是：

```text
batch_size=16
num_workers=2
prefetch_factor=1
persistent_workers=true
```

---

## 3. 已新增 frame_mmap cache 模式

为解决 DINO 小文件 cache 问题，新增 `frame_mmap` 模式：

- 仍然只存每个全局 frame 一次，不像 `window` cache 那样膨胀约 9 倍。
- 把 27 万个 `.pt` 小文件转换成一个连续二进制文件 + `metadata.json`。
- 训练 worker 用 `np.memmap` 按 frame index 读取 9 帧。

相关文件：

```text
scripts/convert_dino_frame_cache_to_mmap.py
src/fastwam/datasets/lerobot/robot_video_dataset.py
```

已验证：

```text
py_compile OK
小样本 frame -> frame_mmap 转换 OK
DataLoader(num_workers=2) 读取 OK
读取 shape: [B,384,9,14,28]
dtype: bf16
与原 frame cache exact_equal=True
```

### 转换命令

建议在没有重要训练打满磁盘 IO 时运行：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

python scripts/convert_dino_frame_cache_to_mmap.py \
  --src ./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact \
  --dst ./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap
```

转换后训练使用：

```text
data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap
data.train.dino_latent_cache_mode=frame_mmap
data.train.dino_latent_cache_required=true
```

---

## 4. 当前稳定训练命令：旧 frame cache

当前正在跑的稳定版仍使用旧 `frame` cache，`num_workers=2`。如果要继续这条保守路线：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MASTER_PORT=29551
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_ID=viewpatch_1x2x2_waninit_bs16_ga1_10ep_$(date +%Y-%m-%d_%H-%M-%S) \
bash scripts/train_zero1.sh 8 \
  task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4 \
  batch_size=16 \
  gradient_accumulation_steps=1 \
  num_workers=2 \
  prefetch_factor=1 \
  persistent_workers=true \
  learning_rate=1e-4 \
  save_every=2000 \
  eval_every=200 \
  wandb.enabled=true \
  wandb.name=libero_dino_s_viewpatch_1x2x2_waninit_bs16_ga1_10ep \
  data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact \
  data.train.dino_latent_cache_mode=frame \
  data.train.dino_latent_cache_required=true \
  model.dino_config.load_backbone=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=16 \
  model.video_dit_init_from_wan=false \
  model.video_dit_pretrained_path=checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt \
  model.loss.lambda_video=0.05 \
  model.loss.lambda_action=5.0
```

---

## 5. 推荐新训练命令：frame_mmap + 更多 worker

转换 `frame_mmap` cache 后，用这条做 200 step 速度/稳定性试跑：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MASTER_PORT=29551
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_ID=viewpatch_1x2x2_mmap_bs16_w4_$(date +%Y-%m-%d_%H-%M-%S) \
bash scripts/train_zero1.sh 8 \
  task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4 \
  batch_size=16 \
  gradient_accumulation_steps=1 \
  num_workers=4 \
  prefetch_factor=1 \
  persistent_workers=true \
  learning_rate=1e-4 \
  save_every=2000 \
  eval_every=200 \
  wandb.enabled=true \
  wandb.name=libero_dino_s_viewpatch_1x2x2_mmap_bs16_w4 \
  data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap \
  data.train.dino_latent_cache_mode=frame_mmap \
  data.train.dino_latent_cache_required=true \
  model.dino_config.load_backbone=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=16 \
  model.video_dit_init_from_wan=false \
  model.video_dit_pretrained_path=checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt \
  model.loss.lambda_video=0.05 \
  model.loss.lambda_action=5.0
```

如果 `num_workers=4` 稳定且明显加速，再考虑试：

```text
num_workers=6, prefetch_factor=1
```

不要一上来恢复 `prefetch_factor=2`。

---

## 6. LIBERO 官方评测命令：viewpatch ckpt

评测不走训练 DINO cache，而是 simulator 在线图像 + DINO backbone。所以评测命令不需要传 `frame_mmap`。

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4/YOUR_RUN_DIR
LATEST_CKPT=$(ls -1 "$RUN_DIR"/checkpoints/weights/step_*.pt | sort -V | tail -n 1)

$PYTHON experiments/libero/run_libero_manager.py \
  task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4 \
  ckpt="$LATEST_CKPT" \
  EVALUATION.num_trials=30 \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="./evaluate_results/libero/viewpatch_1x2x2_30trials_$(basename "$LATEST_CKPT" .pt)" \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=4 \
  model.dino_config.load_backbone=true \
  'model.dino_config.latent_spatial_pool=[1,1]'
```

注意：

- 评测旧 no-pool ckpt 用旧 task：`libero_dino_s_smallvideo_2cam_224_1e-4`。
- 评测 viewpatch ckpt 用 viewpatch task：`libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4`。
- 两者不要混用，因为 `patch_embedding/head/view_embedding` 结构不同。

当前这版长跑 run 的 30 trials 评测命令：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4/viewpatch_1x2x2_mmap_bs16_w4_2026-05-29_10-05-46
LATEST_CKPT=$(ls -1 "$RUN_DIR"/checkpoints/weights/step_*.pt | sort -V | tail -n 1)

$PYTHON experiments/libero/run_libero_manager.py \
  task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4 \
  ckpt="$LATEST_CKPT" \
  EVALUATION.num_trials=30 \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="./evaluate_results/libero/viewpatch_1x2x2_mmap_bs16_w4_30trials_$(basename "$LATEST_CKPT" .pt)" \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=4 \
  model.dino_config.load_backbone=true \
  'model.dino_config.latent_spatial_pool=[1,1]'
```

---

## 7. 占卡与 guard

训练异常时可使用 guard 脚本自动启动占卡：

```bash
bash /data73/mingxinwang/run_view_dino_with_guard.sh
```

清理占卡：

```bash
bash /data73/mingxinwang/kill_occupy_gpu.sh
```

当前 guard 脚本已切换到 `frame_mmap`，并加入异常后的自动 full resume：

```text
data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap
data.train.dino_latent_cache_mode=frame_mmap
num_workers=4
prefetch_factor=1
```

逻辑：

- 训练异常退出后，先在当前 `RUN_ID` 对应的 `RUN_DIR/checkpoints/state/step_*` 中找最新完整 full-state checkpoint。
- 必须同时存在 `checkpoints/state/step_xxxxxx/trainer_state.json` 和 `checkpoints/weights/step_xxxxxx.pt`，才会执行 `resume=/.../checkpoints/state/step_xxxxxx`。
- 默认最多自动 full resume 3 次：`MAX_RESUME_ATTEMPTS=3`。
- 如果没有完整 state+weights，或超过最大重试次数，才启动 `/data73/mingxinwang/test_run.sh` 占卡。
- guard 脚本自身非正常退出时也有 `EXIT` 兜底，会尝试启动占卡；`SIGINT/SIGTERM` 也会启动占卡，除非显式设置 `DISABLE_OCCUPY_ON_SIGNAL=1` 或 `DISABLE_OCCUPY_ON_EXIT=1`。
- 注意：脚本只恢复当前 run 目录，不会自动接旧 run，避免误用旧结构/旧超参。

---

## 8. 2026-05-29 训练稳定性报错复盘

原计划是 2026-05-29 00:00 左右开始 viewpatch 训练，跑到当天晚上评测。但实际大量时间消耗在训练链路稳定性上，不是模型本身。

已遇到的主要问题：

1. **LIBERO 缺包**
   - 报错：`ModuleNotFoundError: No module named 'libero'`。
   - 原因：当前 conda 环境没有 official LIBERO package。
   - 处理：安装 LIBERO editable package，并补 `mujoco/robosuite/bddl/easydict/...`；注意避免按官方旧命令降级 torch。

2. **占卡/评测进程冲突**
   - 现象：占卡脚本、eval、自启动训练互相抢显存。
   - 处理：写了 `/data73/mingxinwang/kill_occupy_gpu.sh` 清占卡；guard 异常后自动启动 `/data73/mingxinwang/test_run.sh` 保卡。

3. **viewpatch 适配链路**
   - 新 task：`libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4`。
   - 注意：viewpatch ckpt 必须用 viewpatch task 评测；旧 no-pool ckpt 用旧 task。两者结构不同，不能混用。

4. **DataLoader worker 被杀/训练随机中断**
   - 老 cache 是 `frame` 模式：约 277713 个 `.pt` 小文件，每个 sample 读 9 帧 DINO latent，再叠加视频 decode。
   - worker 数增大后，小文件 IO、`torch.load`、pyav 解码、进程调度压力叠加，容易出现 worker 被杀或 rank 卡死。
   - 处理：新增 `frame_mmap`，把 DINO frame cache 转成连续 mmap，减少小文件读取压力。

5. **torchcodec / FFmpeg 不可用**
   - 现象：环境中有 `torchcodec`，但缺 FFmpeg 动态库，无法真正启用。
   - 处理：在 `video_utils.py` 中加入 codec 可用性检测与 fallback，当前实际走 pyav/torchvision decode。

6. **batch/worker 高吞吐但不稳定**
   - 记录：
     - `batch_size=24, num_workers=4`：报错。
     - `batch_size=24, num_workers=2`：约 40h。
     - `batch_size=16, num_workers=2`：约 38h。
     - `batch_size=16, num_workers=4`：一度约 24h，但仍需观察长跑稳定性。
     - `batch_size=24, num_workers=8`：吞吐可到约 `46-47 samples/s`，但长跑不稳定。
   - 结论：不能只看短期 `samples/s`，高 worker/high batch 可能在 1 小时后才暴露稳定性问题。

7. **NCCL ALLREDUCE timeout**
   - 典型报错：
     ```text
     Watchdog caught collective operation timeout:
     OpType=ALLREDUCE
     Timeout(ms)=600000
     ```
   - 时间线：曾经跑到 `step=1250`，`step=1000/1200` validation 都正常，然后某个 rank 卡住，其他 rank 等梯度同步 10 分钟后 NCCL timeout。
   - 判断：不是 validation 直接报错，也不是显存 OOM；更像某个 rank 被 DataLoader/视频解码/IO/系统调度拖死，最终拖死 DDP 同步。

8. **checkpoint 保存太稀导致白跑**
   - 之前 `save_every=2000`，训练死在 `step=1250`，没有可恢复 checkpoint。
   - 已改为 `save_every=500`。
   - 当前 run `viewpatch_1x2x2_mmap_bs16_w4_2026-05-29_10-05-46` 已确认 `step_000500` 完整保存：
     ```text
     checkpoints/weights/step_000500.pt
     checkpoints/state/step_000500/trainer_state.json
     ```

当前保守长跑建议：

```text
batch_size=16
num_workers=4 或 6
prefetch_factor=1
persistent_workers=true
save_every=500
eval_every=200
num_epochs=15
data.train.dino_latent_cache_mode=frame_mmap
```

训练命令优先通过 `/data73/mingxinwang/run_view_dino_with_guard.sh` 启动。原则是：能 full resume 就 full resume；不能 resume 就占卡，避免 GPU 裸奔。

---

## 9. 2026-05-30 从 step_023000 继续训到总 20 epoch

专用脚本：

```bash
bash /data73/mingxinwang/run_view_dino_resume_step023000_with_guard.sh
```

用途：

- 从 full-state checkpoint 接续：
  ```text
  /data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_viewpatch_1x2x2_1e-4/viewpatch_1x2x2_mmap_bs16_w4_2026-05-29_10-05-46/checkpoints/state/step_023000
  ```
- 已确认对应权重存在：
  ```text
  checkpoints/weights/step_023000.pt
  ```
- `trainer_state.json`：
  ```json
  {
    "global_step": 23000,
    "epoch": 10,
    "batch_in_epoch": 1300
  }
  ```
- 训练参数：`batch_size=16`，`num_workers=6`，`prefetch_factor=1`，`save_every=500`，`eval_every=200`，`num_epochs=20`。
- `num_epochs=20` 表示总目标 20 epoch，不是额外 20 epoch。
- 脚本异常后会优先找新 run 目录下最新完整 `state+weights` 自动 full resume；如果没有新 checkpoint，则回退到初始 `step_023000`；超过 `MAX_RESUME_ATTEMPTS=3` 或 checkpoint 缺失则启动 `/data73/mingxinwang/test_run.sh` 占卡。

---

## 10. 2026-05-30 weight-only resume 旧坑复盘

之前有一次从 no-pool `step_028930.pt` 做 weight-only finetune，跑满约 2 epoch 后评测严重掉到：

```text
evaluate_results/libero/finetune_from_028930_latest_30trials_step_005786

spatial=73.67
object=92.00
goal=64.33
libero_10=5.67
overall=58.92
```

对应 run：

```text
runs/libero_dino_s_smallvideo_2cam_224_1e-4/finetune_from_step028930_lr2e-5_2ep_2026-05-27_13-19-04
```

关键日志：

```text
Loaded .pt weights only; optimizer/scheduler/step were not restored
```

这次坏结果不是 weight-only 本身必然导致，而是当时 `.pt` 权重是在 `accelerator.prepare` / DeepSpeed ZeRO 包装之后加载，ZeRO 分片状态下加载普通 `.pt` 权重不可靠。训练日志也支持这个判断：一开始 `loss_action` 高到 `7.0 -> 6.1 -> 4.3...`，像从头训，不像从 `step_028930` 成熟权重正常初始化。

后来修复后的 run：

```text
runs/libero_dino_s_smallvideo_2cam_224_1e-4/finetune_fixed_from_step028930_lr1e-5_4k_2026-05-28_04-19-48
```

日志变为：

```text
Loaded .pt weights only before optimizer/DeepSpeed initialization; optimizer/scheduler/step were not restored.
```

这个修复版的 `step_004000.pt` 评测没有崩：

```text
evaluate_results/libero/nopool_latest_30trials_step_004000

spatial=96.67
object=99.67
goal=95.67
libero_10=82.33
overall=93.58
```

结论：

- 精确续训仍优先用 full-state `resume=.../checkpoints/state/step_xxxxxx`。
- 如果必须 weight-only，必须确保 `.pt` 权重在 DeepSpeed/Accelerate prepare 之前加载；日志应出现 `before optimizer/DeepSpeed initialization`。
- 只看到 `Loaded .pt weights only; ...` 这种旧日志时，不要信任该 run 的初始化质量。

补充：`train/loss_video` 日志值是已经乘过 `model.loss.lambda_video` 的加权项。当前 `lambda_video=0.05` 时，如果 wandb 显示 `loss_video=0.02`，未加权 video loss 约为 `0.4`；之前 no-pool 常见显示 `0.002`，未加权约为 `0.04`。因此 viewpatch 当前 video latent prediction 明显还没有学到 no-pool 后期的量级。

---

## 11. 2026-05-31 新对话交接摘要：当前进展、问题、后续方向

### 当前最好结果与主要对比

目前 no-pool 仍是最强结果：

```text
evaluate_results/libero/nopool_latest_30trials_step_028930

spatial=98.67
object=99.33
goal=92.33
libero_10=84.00
overall=93.58
```

view-aware patch merge `[1,2,2]` 的最好结果：

```text
viewpatch_1x2x2 step_032000

spatial=95.00
object=98.33
goal=96.33
libero_10=77.33
overall=91.75
```

观察：

- viewpatch 相比 no-pool：`goal` 明显更好，但 `spatial` 明显更差，`libero_10` 更差。
- 这说明 learnable token merge 不是纯粹“性能不变、速度变快”，它确实改变了模型可用的信息结构。
- `[1,2,2]` 每个 view 内同时压缩 H 和 W，可能破坏了 Spatial 任务需要的精细相对位置。
- Goal 任务更吃全局语义/目标状态，viewpatch 的压缩可能反而减少 token 噪声，所以表现更好。
- LIBERO-10/long 仍然偏弱，可能不只是 token merge 问题，还和缺少 memory、smallvideo 模型容量/视频先验有关。

### 新增 viewpatch `[1,1,2]` 配置

为缓解 `[1,2,2]` 对 spatial 的破坏，新增配置：

```text
configs/task/libero_dino_s_smallvideo_2cam_224_viewpatch_1x1x2_1e-4.yaml
```

核心含义：

```yaml
model:
  video_dit_config:
    latent_patch_size: [1, 1, 2]
    latent_patch_mode: "view"
    latent_num_views: 2
```

原始 2-camera DINO grid 是 `14x28`，view-aware 先拆成 `2 x 14 x 14`：

- `[1,2,2]`：每个 view `14x14 -> 7x7`，两视角共 `98 tokens/frame`。
- `[1,1,2]`：每个 view `14x14 -> 14x7`，两视角共 `196 tokens/frame`。
- `[1,1,2]` token 数和最早固定平均池化后的 token 数接近，但压缩是 learnable 的，并且保留了 H 方向精度。

### `[1,1,2]` 当前报错

第一次用下面组合启动：

```text
task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x1x2_1e-4
batch_size=16
gradient_accumulation_steps=1
num_workers=6
```

在 backward 阶段 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 332.00 MiB
GPU has ~267-309 MiB free
process has ~78.97 GiB memory in use
```

判断：

- 不是 shape/config bug，而是显存打满。
- `[1,1,2]` 比 `[1,2,2]` token 多一倍，attention 显存开销更高。
- `bs=16` 对 A800 80G 已经太紧。

建议重启参数：

```text
batch_size=12
gradient_accumulation_steps=1
num_workers=6
prefetch_factor=1
persistent_workers=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

如果必须保持 global batch 128，则用：

```text
batch_size=8
gradient_accumulation_steps=2
```

但会更慢。当前更建议先 `bs=12, ga=1` 跑通，优先确认 `[1,1,2]` 是否能恢复 spatial。

### 当前训练链路的重要经验

1. **checkpoint 必须频繁保存**
   - 之前 `save_every=2000` 时，多次在 1000-1250 step 附近死掉，没有 ckpt，白跑。
   - 现在长跑建议 `save_every=500`。

2. **full resume 优先**
   - 精确续训必须用 `checkpoints/state/step_xxxxxx`。
   - weight-only 可以做实验，但不能用于严格续训。

3. **weight-only 旧坑**
   - 旧版曾在 DeepSpeed/Accelerate prepare 后加载普通 `.pt`，导致初始化不可靠。
   - 正确日志应包含：
     ```text
     Loaded .pt weights only before optimizer/DeepSpeed initialization
     ```

4. **DINO cache 已改为 mmap**
   - 之前 frame cache 小文件 + 多 worker 容易 IO/解码/同步不稳定。
   - 已有 mmap cache 路线，训练命令应使用：
     ```text
     data.train.dino_latent_cache_mode=frame_mmap
     data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap
     ```

5. **num_workers 经验**
   - `bs=24, workers=4` 报错。
   - `bs=24, workers=2` 约 40h/10ep。
   - `bs=16, workers=2` 约 38h/10ep。
   - `bs=16, workers=4` 一度约 24h/10ep，但后来仍出现稳定性问题。
   - mmap 后 `bs=24, workers=8` 曾显示约 18h，但长跑仍可能出问题，需要 guard 和频繁 ckpt。

6. **guard 脚本原则**
   - `/data73/mingxinwang/run_view_dino_with_guard.sh` 和 resume 专用脚本的设计原则：训练异常退出后，优先找最近完整 `state+weights` 做 full resume；找不到就启动 `/data73/mingxinwang/test_run.sh` 占卡。
   - 原则是不让 GPU 裸奔。

### 对 LIBERO-10 差的当前判断

LIBERO-10/long 持续比 FastWAM 或 no-pool 理想值差，可能原因按优先级大致是：

```text
缺少 memory/history > smallvideo 1B 视频先验弱于原生 Wan 5B > token merge 压缩损失 > 继续训练不足
```

具体判断：

- WAM 本质利用 Wan 多帧视频预测中隐含的现实世界动力学先验。
- 当前 smallvideo 约 1B，且权重来自 Wan 插值/迁移，不等价于原生 5B Wan。
- 长程任务更依赖物体状态演化、接触变化、阶段记忆，因此 5B Wan 可能更有帮助。
- 但即使用 5B，如果推理没有 memory/history，也未必能根治 LIBERO-10。

建议后续验证顺序：

1. **先跑 `[1,1,2]` viewpatch**
   - 目标：看 spatial 能否从 `[1,2,2]` 的 95 附近回升，同时保住 goal。
   - 推荐先 `bs=12, ga=1`。

2. **再考虑 memory/history**
   - 这是最可能改善 LIBERO-10 的方向。
   - 需要确认当前 eval/inference 是否只看当前帧；如果只看当前帧，long task 很可能天然吃亏。

3. **小规模验证原生 5B Wan**
   - 保持数据、DINO、loss、eval 不变，只把 video DiT 换成原 FastWAM/Wan 5B。
   - 不必一开始训满，先 2k-4k steps 看 loss、速度、显存、LIBERO-10 早期趋势。

4. **调整 DINO loss 或 token loss 形式**
   - 当前 viewpatch 的 `loss_video` 明显高于 no-pool 后期量级，说明 dense DINO target 的重建/预测还没学好。
   - 可能需要考虑：先在 merged token space 做 loss，或对 dense loss 加位置/任务相关权重，而不是只靠 unpatchify 后全量 MSE。

### 新对话建议第一步

如果继续当前路线，新对话可以直接从这个命令开始试 `[1,1,2]`：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MASTER_PORT=29562
export MPLCONFIGDIR=/tmp/matplotlib-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=viewpatch_1x1x2_mmap_bs12_w6_10ep_$(date +%Y-%m-%d_%H-%M-%S) \
bash scripts/train_zero1.sh 8 \
  task=libero_dino_s_smallvideo_2cam_224_viewpatch_1x1x2_1e-4 \
  batch_size=12 \
  gradient_accumulation_steps=1 \
  num_workers=6 \
  prefetch_factor=1 \
  persistent_workers=true \
  learning_rate=1e-4 \
  lr_scheduler_type=cosine \
  num_epochs=10 \
  save_every=500 \
  eval_every=200 \
  wandb.enabled=true \
  wandb.name=libero_dino_s_viewpatch_1x1x2_mmap_bs12_w6_10ep \
  data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap \
  data.train.dino_latent_cache_mode=frame_mmap \
  data.train.dino_latent_cache_required=true \
  model.dino_config.load_backbone=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=16 \
  model.video_dit_init_from_wan=false \
  model.video_dit_pretrained_path=checkpoints/DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt \
  model.loss.lambda_video=0.05 \
  model.loss.lambda_action=5.0
```

---

## 12. 2026-06-03 新增 VAE smallvideo baseline

### 动机

当前 DINO 线主要使用 smallvideo 级别的 VideoDiT：

```text
hidden_dim=1024
ffn_dim=4096
num_layers=30
num_heads=24
attn_head_dim=128
```

而原始 FastWAM/Wan VAE 线默认使用 5B 形状：

```text
hidden_dim=3072
ffn_dim=14336
num_layers=30
num_heads=24
attn_head_dim=128
```

因此直接比较原始 VAE FastWAM 和 DINO smallvideo 不公平：差异同时包含 latent 表征和 VideoDiT 容量。新增 VAE smallvideo baseline 的目标是：

```text
同一 small VideoDiT 容量下，对比 VAE latent vs DINO token。
```

### 已新增文件

```text
scripts/preprocess_wan_video_dit_small.py
configs/model/fastwam_smallvideo.yaml
configs/task/libero_vae_smallvideo_2cam224_1e-4.yaml
/data73/mingxinwang/Spider-s-World-Action-Model/scripts/interpolate_wan_video_dit_small.sh
```

并修改：

```text
src/fastwam/models/wan22/wan_video_dit.py
src/fastwam/models/wan22/fastwam.py
src/fastwam/runtime.py
```

新增能力：

1. `WanVideoDiT.load_preprocessed_weights(video_dit_pretrained_path)`  
   允许 VAE VideoDiT 从预处理后的 small WanVideoDiT payload 加载。

2. `FastWAM.from_wan22_pretrained(..., video_dit_pretrained_path=...)`  
   当提供 `video_dit_pretrained_path` 时，先构造目标 small-shape WanVideoDiT，再加载预处理权重；不会把 5B 权重直接硬塞进 small shape。

3. `runtime.create_fastwam/create_fastwam_joint` 已透传 `video_dit_pretrained_path`。

### VAE smallvideo 目标结构

VAE smallvideo 保持 Wan VAE latent 输入输出：

```text
in_dim=48
out_dim=48
patch_size=[1,2,2]
```

同时把 VideoDiT backbone 缩到和 DINO smallvideo / ActionDiT 一致：

```text
hidden_dim=1024
ffn_dim=4096
num_layers=30
num_heads=24
attn_head_dim=128
text_dim=4096
freq_dim=256
```

也就是说：

```text
VAE latent + small WanVideoDiT + same ActionDiT shape
```

不是原始 5B FastWAM。

### 初始化方式

目标权重文件：

```text
checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
```

生成方式：

```text
Wan2.2-5B WanVideoDiT
-> sequential 1D linear interpolation
-> alpha scaling
-> target small WanVideoDiT full state_dict
```

这个 payload 包含完整 VAE VideoDiT 权重：

```text
patch_embedding.*
blocks.*
text_embedding.*
time_embedding.*
time_projection.*
head.*
```

不同于 DINO payload 只迁移 backbone、保留 DINO-specific input/output projection/head 随机。

### 生成 VAE smallvideo 初始化权重

当前 2026-06-03 检查时，8 张 GPU 正在跑：

```text
viewpatch_1x1x2_mmap_bs12_w6_weightonly_from_step024000_lr2e-5_5ep_2026-06-02_night
```

每张卡约占 60G，因此没有立即生成 VAE smallvideo payload。等训练结束或有空卡后运行：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
bash scripts/interpolate_wan_video_dit_small.sh
```

脚本内容等价于：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export DIFFSYNTH_MODEL_BASE_PATH=/data73/mingxinwang/Spider-s-World-Action-Model/checkpoints
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib-cache

$PYTHON scripts/preprocess_wan_video_dit_small.py \
  --model-config configs/model/fastwam_smallvideo.yaml \
  --output checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16 \
  --apply-alpha-scaling true
```

### VAE smallvideo 训练命令

生成 `checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt` 后，可用：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MASTER_PORT=29564
export MPLCONFIGDIR=/tmp/matplotlib-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID=vae_smallvideo_bs16_10ep_$(date +%Y-%m-%d_%H-%M-%S)
bash scripts/train_zero1.sh 8 \
  task=libero_vae_smallvideo_2cam224_1e-4 \
  output_dir=./runs/libero_vae_smallvideo_2cam224_1e-4/${RUN_ID} \
  batch_size=16 \
  gradient_accumulation_steps=1 \
  num_workers=8 \
  prefetch_factor=1 \
  persistent_workers=true \
  learning_rate=1e-4 \
  lr_scheduler_type=cosine \
  num_epochs=10 \
  save_every=2000 \
  eval_every=200 \
  wandb.enabled=true \
  wandb.name=libero_vae_smallvideo_bs16_10ep
```

### 与 DINO smallvideo / ActionDiT 初始化的关系

当前确认：

```text
ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
DinoVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
```

二者都是从 Wan2.2-5B 同源插值到 small shape，key 集合和 shape 一致：

```text
common keys = 820
action-only = 0
dino-only = 0
shape_diff = 0
```

但不保证逐元素完全相同。代码级原因是 ActionDiT 脚本和 DinoVideoDiT 脚本的 `alpha scaling` 与 `bf16 cast` 顺序不同：

```text
ActionDiT:
fp32 interpolate -> cast bf16 -> fp32 alpha scaling -> cast bf16 save

DinoVideoDiT:
fp32 interpolate -> fp32 alpha scaling -> cast bf16 save
```

这种差异是 bf16 rounding 级别，理论上对训练影响很小。后续 VAE smallvideo 应按 DINO 脚本更干净的顺序：

```text
fp32 interpolate -> fp32 alpha scaling -> final bf16 cast
```

### 实验解释口径

VAE smallvideo baseline 应表述为：

```text
VAE latent supervision under the same small VideoDiT capacity as DINO-small.
```

而不是原始 FastWAM 5B baseline。

如果结果：

1. VAE smallvideo 明显低于 DINO avg/no-pool  
   说明 DINO token 表征在同等 small backbone 容量下更有利于 action learning。

2. VAE smallvideo 接近 DINO  
   说明之前 DINO/VAE 差距可能主要来自容量、训练步数或具体 loss/token 设计，而不是 latent 类型本身。

3. VAE smallvideo 高于 DINO  
   说明 DINO 路线并不天然优于 VAE，需要重新考虑 DINO token dynamics 的训练目标和使用方式。

### 2026-06-03 VAE smallvideo 代码排查结论

新增 guard 脚本：

```bash
/data73/mingxinwang/run_vae_smallvideo_2cam224_10ep_with_guard.sh
```

今晚正确顺序：

```bash
# 1. 先生成 VAE-small VideoDiT 初始化 payload
cd /data73/mingxinwang/Spider-s-World-Action-Model
bash scripts/interpolate_wan_video_dit_small.sh

# 2. 确认生成成功
ls -lh /data73/mingxinwang/Spider-s-World-Action-Model/checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt

# 3. 再开训
bash /data73/mingxinwang/run_vae_smallvideo_2cam224_10ep_with_guard.sh
```

guard 脚本会在启动训练前检查：

```text
checkpoints/WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

如果 VAE-small video init payload 不存在，它会直接退出并提示先跑 preprocess，不会误开一个坏训练。

已经做过的校验：

```text
py_compile: OK
WanVideoDiT.load_preprocessed_weights strict loader roundtrip: OK
Hydra train compose: OK
Hydra sim_libero eval compose: OK
```

Hydra compose 关键结果：

```text
训练 task=libero_vae_smallvideo_2cam224_1e-4:
  model.load_text_encoder = false
  video hidden/ffn/layers = 1024 / 4096 / 30
  batch_size / num_epochs / lr = 16 / 10 / 1e-4
  save_every / eval_every = 2000 / 200

LIBERO rollout eval:
  base config = sim_libero.yaml
  model.load_text_encoder = true
  model.skip_dit_load_from_pretrain = true
  model.action_dit_pretrained_path = null
```

这个覆盖关系是合理的：

1. 训练时数据集提供 cached `context/context_mask`，不需要 text encoder，可以省显存。
2. 外部 LIBERO rollout eval 会传 prompt 字符串，所以 `sim_libero.yaml` 会把 `model.load_text_encoder=true` 打开。
3. eval 时 `skip_dit_load_from_pretrain=true` 和 `action_dit_pretrained_path=null` 不影响最终权重，因为模型实例化后马上会 `model.load_checkpoint(ckpt)` 加载训练 checkpoint。
4. 但 eval 仍然需要 `WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt` 存在，因为当前 `FastWAM.from_wan22_pretrained()` 会先用它构造同 shape 的 video expert，再加载训练 checkpoint。

训练中 eval 路径：

```text
Wan22Trainer.evaluate()
  -> model.training_loss(sample)
  -> model.infer_action(context/context_mask=sample context)
```

因此训练中 eval 不依赖 text encoder，和训练 batch 的 text embedding 来源一致。

外部 rollout eval 路径：

```text
experiments/libero/run_libero_manager.py
  -> sim_libero.yaml + task override
  -> eval_libero_single.py
  -> instantiate(cfg.model, load_text_encoder=true)
  -> model.load_checkpoint(ckpt)
  -> model.infer_action(prompt=DEFAULT_PROMPT.format(task=...))
```

VAE smallvideo eval 命令模板：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model
source /data73/envs/miniconda3/etc/profile.d/conda.sh
conda activate spiderwam

export PYTHON=/home/wangmx2605/.conda/envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAIT_FOR_GPUS=0
export MPLCONFIGDIR=/tmp/matplotlib-cache

RUN_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/runs/libero_vae_smallvideo_2cam224_1e-4/YOUR_RUN_DIR
LATEST_CKPT=$(ls -1 "$RUN_DIR"/checkpoints/weights/step_*.pt | sort -V | tail -n 1)

$PYTHON experiments/libero/run_libero_manager.py \
  task=libero_vae_smallvideo_2cam224_1e-4 \
  ckpt="$LATEST_CKPT" \
  EVALUATION.num_trials=30 \
  EVALUATION.dataset_stats_path="$RUN_DIR/dataset_stats.json" \
  EVALUATION.output_dir="./evaluate_results/libero/vae_smallvideo_30trials_$(basename "$LATEST_CKPT" .pt)" \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=4
```
