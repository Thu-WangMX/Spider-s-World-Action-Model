# SpiderWAM Short-DINO-Intent Plan

更新时间：2026-06-10

本文件记录当前围绕 FastWAM-DINO 加入历史视觉记忆 / intent tokens 的计划。它接在 `PROJECT_CONTEXT2.md` 之后，重点不是复盘 leaderboard，而是固定下一步架构实验的设计选择、代码改动范围和风险点。

---

## 1. 背景与当前约束

当前主线仍然是 `FastWAM_DINO` / no-pool DINO smallvideo 这套架构。短期不移动 1B DINO 大线，先在现有 no-pool DINO based FastWAM 上做一个轻量、可消融的 Short-DINO-Intent 实验。

我们现在面前有三类增强方向：

1. Motus-style：新增一条 VLM / understanding expert，增强语义 grounding 和指令跟随。
2. GaussianDream / IntentVLA-style：从历史帧提取视觉特征，再压缩成 intent / memory tokens。
3. Helios-style：更长程 memory，远历史更强压缩，先不加 VLM。

当前决定先做第 2 类的最小可行版本：

```text
past/current observations
  -> frozen/cached no-pool DINO tokens
  -> temporal position embedding
  -> learnable intent queries + Perceiver/MTA-lite resampler
  -> K Short-DINO-Intent tokens
  -> append after proprio in text context
  -> video/action experts read them through existing cross-attention
```

第一版 history offsets 采用 `[-8, -4, 0]`，因为当前 LIBERO smallvideo 配置是每 4 个 action/frame step 采一个 DINO video latent；这个间隔与现有 video 分支的时间粒度对齐，比直接照搬 GaussianDream 的 `[-10, -5, 0]` 更自然。

---

## 2. 外部 repo 参考结论

### GaussianDream

最值得借鉴。它真实使用历史/当前 context frames，并且用 learnable future/motion query tokens 去融合历史视觉特征。

关键点：

- 默认 temporal context 是 `[-2, -1, 0]`，最后一个必须是当前帧。
- LIBERO 配置实际用了 `temporal_context_offsets=(-10, -5, 0)`。
- eval 侧也维护 history deque，按 `(10, 5, 0)` 取过去帧。
- MTA block 不是简单 pooling，而是先做每帧内 token mixing，再做同 slot 的 temporal attention。
- query tokens 与历史 VGGT/MTA features 融合后产出 future/world tokens。

对应到我们这里，可以替换成：

```text
VGGT features -> no-pool DINO tokens
future/world tokens -> intent tokens
MTA / temporal fusion -> Perceiver/MTA-lite resampler
```

### Motus

Motus 的重点不是历史视觉帧。它的 `num_video_frames=8` 主要是要预测的未来 target video frames，dataset 返回 `first_frame + video_frames`，VLM 输入也主要来自 first frame。

Motus 对我们后续有价值的是：

- video/action/understanding 三专家 MoT 结构；
- VLM understanding tokens 参与 joint attention；
- 适合后续做 VLM branch，不适合作为 Short-DINO-Intent v1 的主要模板。

### LDA-1B

LDA 有历史 observation 的影子，但不是一个完整的历史 intent resampler。

有用点：

- 数据侧常见 `observation_indices=[-5,0]`，可以取过去一帧 + 当前帧。
- 有 `history_action_indices=range(-5,0)`，历史 action 接口比较明确。
- MMDiT 里对 image tokens 支持 3D RoPE / obs timesteps。
- 有 register / target vision tokens / learnable next obs token 的设计，可作为 learnable query 的旁证。

限制：

- 常见训练 yaml 里 `obs_horizon=1`，历史视觉窗口不长。
- 模型侧更像把多图/多视角统一作为 image list 喂给 QwenVL/DINO，而不是显式压缩历史视觉为 intent tokens。

---

## 3. 为什么 v1 选 context-after-proprio

目前 `FastWAM_DINO` 的 context 路径已经很适合做第一版：

```text
T5 text context
  -> append proprio token
  -> video/action pre_dit text_embedding
  -> MoT block cross-attn
```

代码现状：

- `src/fastwam/models/wan22/fastwam_dino.py`
  - `_append_proprio_to_context()` 把 proprio `[B,D]` 编码成 `[B,1,text_dim]`，拼到 text context 后面。
  - `build_inputs()` 里取 `proprio[:,0,:]`，然后 append 到 context。
  - `training_loss()` 里 video/action expert 都拿同一份 `context/context_mask`。
- `src/fastwam/models/wan22/dino_video_dit.py`
  - `pre_dit()` 把 `[B,L,text_dim]` context 投到 video hidden dim，并扩展 context mask。
- `src/fastwam/models/wan22/action_dit.py`
  - `pre_dit()` 同样把 context 投到 action hidden dim。
- `src/fastwam/models/wan22/mot.py`
  - 每个 expert 在 mixed attention 后通过 `block.cross_attn()` 读自己的 context payload。

因此 v1 直接做：

```text
context = [T5 tokens]
context = [T5 tokens, proprio token]
context = [T5 tokens, proprio token, intent tokens]
```

优点：

- 不改变 video token sequence length。
- 不影响 video RoPE / patch grid / `post_dit()` reshape。
- 不影响当前 `_build_mot_attention_mask()` 对 video/action token 的假设。
- 不影响 video prefill cache 和 action denoise cache。
- 可以做非常干净的 ablation：只开/关 intent context tokens。

暂时不选 video-prefix 的原因：

```text
[intent_prefix] + [video_tokens]
```

理论上更强，更像把历史记忆并入视觉世界模型；但它需要同时改：

- video token position / RoPE，intent prefix 没有自然 3D 网格位置；
- mixed attention mask，action 要能看见 intent prefix；
- `post_dit()`，输出 video target 前必须 strip intent prefix；
- inference video cache，prefix token 是否进入 cache 要非常明确；
- loss target，intent prefix 没有 video supervision，不能误进 video loss。

所以 video-prefix 放到 v2，在 v1 证明历史 intent 有效之后再做。

---

## 4. v1 具体代码改动计划

### 4.1 新增配置

在 DINO model config 下新增一个可关闭的 intent 配置，例如：

```yaml
model:
  intent_config:
    enabled: true
    source: history_dino_latents
    history_offsets: [-8, -4, 0]
    num_intent_tokens: 8
    resampler_dim: 1024
    num_resampler_layers: 2
    num_heads: 8
    dropout: 0.0
    append_position: after_proprio
```

默认 `enabled=false`，确保旧实验完全不受影响。

### 4.2 数据集返回 history DINO latents

当前 `RobotVideoDataset` 的 `dino_latents` 是预测视频窗口，对应：

```text
num_frames=33
action_video_freq_ratio=4
video_sample_indices=[0,4,8,...,32]
```

这些不是历史帧，不能拿未来预测窗口冒充 memory，否则会泄漏 future 信息。

需要在 `src/fastwam/datasets/lerobot/robot_video_dataset.py` 增加：

```text
history_frame_offsets: [-8, -4, 0]
load_history_dino_latents: true
```

并返回：

```text
sample["history_dino_latents"]: [D, T_h, H, W]
```

加载逻辑建议优先支持 `frame` / `frame_mmap` cache，因为这两个模式天然可以按任意 global frame index 取单帧。`window` cache 先不支持 history，避免为了实验一改一大片。

边界处理：

```text
global_idx + offset < episode_start -> clamp/repeat current frame
```

这与 GaussianDream eval 里 episode 起始处 history 不足时重复 current frame 的做法一致。

### 4.3 新增 intent resampler 模块

建议新增文件：

```text
src/fastwam/models/wan22/dino_intent.py
```

模块名：

```python
class DINOHistoryIntentResampler(nn.Module):
    ...
```

输入输出：

```text
input:  history_dino_latents [B, D, T_h, H, W]
output: intent_tokens        [B, K, text_dim]
```

内部第一版保持简单：

```text
history_dino_latents
  -> flatten to [B, T_h*H*W, D]
  -> add temporal embedding and optional spatial embedding
  -> project D -> resampler_dim
  -> K learnable queries [1,K,resampler_dim]
  -> 2-layer cross-attn/Perceiver block
  -> project resampler_dim -> text_dim
```

先不做很复杂的 multi-scale MTA。等 v1 有信号，再考虑 GaussianDream-style 的“frame 内 token mixing + slot temporal attention”。

### 4.4 接入 FastWAM_DINO

在 `FastWAM_DINO.__init__`：

```text
self.intent_encoder = DINOHistoryIntentResampler(...) if enabled else None
```

在 `build_inputs()`：

```text
context, context_mask = append_proprio(...)
if self.intent_encoder is not None:
    history = sample["history_dino_latents"]
    intent_tokens = self.intent_encoder(history)
    context, context_mask = self._append_intent_to_context(context, context_mask, intent_tokens)
```

新增 helper：

```python
def _append_intent_to_context(context, context_mask, intent_tokens):
    # intent_tokens: [B,K,text_dim]
    # mask: all True
```

顺序固定为：

```text
[text tokens, proprio token, intent tokens]
```

原因是 proprio 是当前机器人状态，intent 是视觉历史摘要；把 intent 放在 proprio 后面，语义上更像“额外条件 token”，也方便 mask/调试。

### 4.5 inference / LIBERO eval

`infer_action()` 第一版新增可选参数：

```python
history_dino_latents: Optional[torch.Tensor] = None
history_video: Optional[torch.Tensor] = None
```

优先级：

1. 如果传入 `history_dino_latents`，直接用。
2. 如果传入 `history_video`，用 frozen DINO encoder 在线编码。
3. 如果都没有，且 intent enabled，可以重复当前 input image 形成 `[t,t,t]`，但要在日志里 warning；正式 eval 不应长期用 fallback。

LIBERO runner 侧需要维护一个 per-env history buffer，按 offsets `[-8,-4,0]` 取帧。episode 起始不足时重复当前帧。

### 4.6 checkpoint save/load

需要把 intent encoder 加到 checkpoint：

```text
payload["intent_encoder"] = self.intent_encoder.state_dict()
```

load 时：

```text
如果 checkpoint 有 intent_encoder 且当前模型 enabled，则加载；
如果 checkpoint 没有 intent_encoder，则允许从旧 no-pool checkpoint 初始化其余权重，intent encoder 随机初始化。
```

这样可以从当前 no-pool DINO 最优点继续 finetune，而不是从头训练。

---

## 5. 验证顺序

### 5.1 纯 shape / import 验证

先做小 batch 前向，不跑完整训练：

```text
history_dino_latents [B,384,3,14,28]
intent_tokens [B,K,4096]
context length: 128 + 1 + K
video/action pre_dit context_mask shape 正常
```

### 5.2 配置关闭时回归

`intent_config.enabled=false` 时，旧模型行为应完全不变：

```text
不要求 history_dino_latents
context length 仍是 128 + 1
旧 checkpoint 可正常 load
```

### 5.3 小步训练

先跑 100-200 steps，看：

```text
loss 是否正常下降
显存增加是否可接受
DataLoader 是否因为额外 history cache 读取变慢/变不稳
```

### 5.4 LIBERO-10 重点观察

重点看当前怀疑的长程/阶段性任务：

```text
两个瓶子放盘子里后是否还会误以为没完成而继续抓
放置位置是否更准
任务完成判定前的动作是否更稳定
```

如果 v1 有正信号，再考虑：

```text
v2: intent tokens as video prefix
v3: longer Helios-style memory compression
v4: Motus-style VLM understanding branch
```

---

## 6. 风险与消融

主要风险：

1. history DINO tokens 只通过 cross-attn 注入，可能影响不够强。
2. LIBERO-10 当前错误可能更多是空间精度/成功判定边界，而不是记忆不足。
3. 额外读取 `history_dino_latents` 会增加 DataLoader IO。
4. 如果 offsets 取错，容易引入 future leakage。

必须做的消融：

```text
baseline: no-pool DINO current best, no intent
v1a: intent enabled, offsets [-8,-4,0], K=8
v1b: intent enabled, offsets [-2,-1,0], K=8
v1c: same params but history frames all replaced by current frame
```

其中 v1c 很关键：如果 v1a 提升但 v1c 也提升，说明收益可能来自额外 learnable context capacity，而不是历史信息本身。

---

## 7. 当前推荐执行顺序

第一阶段只做 context-after-proprio：

```text
1. dataset 支持 history_dino_latents
2. 新增 DINOHistoryIntentResampler
3. FastWAM_DINO append intent after proprio
4. infer_action 支持 history_dino_latents/history_video
5. shape test + 200-step smoke train
6. 从 no-pool DINO checkpoint finetune LIBERO
```

暂不做：

```text
video-prefix intent tokens
long-range Helios memory bank
VLM branch / understanding expert
history action conditioning
```

原因：先把“历史 DINO intent 是否有用”这个问题隔离出来。证明有效之后，再叠更复杂的 memory/VLM。

---

## 8. Short-DINO-Intent video_prefix 注入实现记录

更新时间：2026-06-13

本次在现有 Short-DINO-Intent / context-after-proprio 基础上，新增了严格可选的 `video_prefix` 注入方式：

```text
history frames [-8, -4, 0]
-> DINO history tokens
-> DINOHistoryIntentResampler
-> K intent tokens
-> prepend 到 VideoDiT video token sequence
```

配置开关：

```text
model.intent_config.enabled=true
model.intent_config.injection_mode=context_after_proprio | video_prefix
```

兼容性结论：

- 默认行为保持不变：不配置 `intent_config` 时不启用 intent；启用但不写 `injection_mode` 时默认等价于原来的 `context_after_proprio`。
- `context_after_proprio` 路径保持旧行为：intent tokens 追加到 text/proprio context 后面，作为 VideoDiT / ActionDiT 的 cross-attention context。
- `video_prefix` 路径只把 intent tokens prepend 到 VideoDiT video sequence；ActionDiT cross-attention context 不额外拼 intent。
- checkpoint 保存 `intent_encoder` 和 `intent_config`；加载时如果 checkpoint 有 intent encoder 但当前模型没启用 intent，或者 `injection_mode` 不一致，会直接报错，避免 eval 静默加载错误结构。

当前 `video_prefix` token/mask 设计：

```text
video branch  = [p, f0, f1..fh]
action branch = [a1..ah]

p      = Short-DINO-Intent prefix tokens
f0     = clean first-frame DINO tokens
f1..fh = 含噪后续 video/DINO tokens
a1..ah = 含噪 action tokens
```

训练 attention mask：

```text
             p      f0     f1..fh   a1..ah
p          [ ✓ ]   [ ✓ ]   [  ]     [  ]
f0         [ ✓ ]   [ ✓ ]   [  ]     [  ]
f1..fh     [ ✓ ]   [ ✓ ]   [ ✓ ]    [  ]
a1..ah     [ ✓ ]   [ ✓ ]   [  ]     [ ✓ ]
```

也就是说：

- prefix token 可以看 prefix 自己和 `f0`，不能看含噪后续 video token，避免从 noisy future video 泄漏。
- `f0` 可以看 prefix 和 `f0`，不能看含噪后续 video token。
- 含噪后续 video tokens 可以看 prefix、`f0` 和全部 video tokens，用于 video branch dynamics。
- action tokens 可以看 prefix、`f0` 和全部 action tokens，但不能看含噪后续 video tokens。
- video tokens 不看 action tokens。

RoPE / position 处理：

- prefix 是 resampler 产出的抽象 intent tokens，不对应真实 DINO 3D grid 位置。
- 当前实现给 prefix 使用 identity / zero-position RoPE，即 RoPE freqs 全 1；原始 video tokens 的 3D RoPE 完全保持原样。
- 暂不使用 negative RoPE，因为 history offsets 已经在 intent encoder 的 temporal embedding 中编码，给抽象 prefix 强行分配 `t<0` 的 video grid 位置会引入额外假设。

loss / shape 处理：

- prefix 在 DINO grid patchify 之后 prepend，进入 transformer sequence，不改原始 `[T,H,W]` grid metadata。
- `post_dit` 前会 strip 掉 prefix tokens，再 unpatchify 回原 DINO grid。
- video loss 只计算原始 video tokens；prefix 不参与 DINO target，也不参与 video loss。
- train forward 和 eval / `infer_action` 都使用同一个 `video_prefix` prepend helper，保证训推一致。

本次最小验证：

```text
python -m py_compile:
  src/fastwam/models/wan22/dino_video_dit.py
  src/fastwam/models/wan22/fastwam_dino.py
  experiments/libero/eval_libero_single.py
  src/fastwam/trainer.py

git diff --check:
  src/fastwam/models/wan22/dino_video_dit.py
  src/fastwam/models/wan22/fastwam_dino.py

Hydra compose:
  default config without intent_config OK
  explicit video_prefix intent_config OK

Tiny tensor smoke:
  context_after_proprio training_loss OK
  context_after_proprio infer_action OK
  video_prefix training_loss OK
  video_prefix infer_action OK
  checkpoint injection_mode mismatch guard OK
  explicit attention mask assertions OK
```

完整训练命令：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model

export PYTHON=/data73/mingxinwang/conda_envs/spiderwam/bin/python
export RUN_ID=short_dino_intent_video_prefix_10ep_$(date +%Y%m%d_%H%M%S)
export OUTPUT_DIR=/data32/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/${RUN_ID}
export DINO_CACHE_DIR=/data73/mingxinwang/Spider-s-World-Action-Model/data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact_mmap

WAIT_FOR_GPUS=1 RUN_ID="${RUN_ID}" bash scripts/train_zero1.sh 8 \
  "task=libero_dino_s_smallvideo_2cam_224_1e-4" \
  "output_dir=${OUTPUT_DIR}" \
  "wandb.enabled=true" \
  "wandb.name=${RUN_ID}" \
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
  "resume=null" \
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
  "+model.intent_config.injection_mode=video_prefix" \
  "+model.intent_config.history_offsets=[-8,-4,0]" \
  "+model.intent_config.max_history_frames=3" \
  "+model.intent_config.num_intent_tokens=8" \
  "+model.intent_config.resampler_dim=1024" \
  "+model.intent_config.num_resampler_layers=2" \
  "+model.intent_config.num_heads=8" \
  "+model.intent_config.mlp_ratio=4.0" \
  "+model.intent_config.dropout=0.0"
```

说明：

- 这条命令是 8 GPU、`batch_size=6`、`gradient_accumulation_steps=2`，global batch size 为 `96`，对齐当前 DINO no-pool / Short-DINO-Intent 的主要对照设置。
- `data.train.dino_latent_cache_mode` 已经存在于当前 config 中，所以这里使用普通 override，不使用 `+data.train.dino_latent_cache_mode=...`。
- 输出目录放在 `/data32/mingxinwang/...`；当前该目录已有写权限。

## 10. Short-DINO-Intent 评测命令与踩坑记录

### 10.1 重要注意

Short-DINO-Intent / no-pool DINO smallvideo / avgpool DINO smallvideo 的 LIBERO 评测必须使用：

```text
task=libero_dino_s_smallvideo_2cam_224_1e-4
```

不要误用：

```text
task=libero_dino_s_2cam_224_1e-4
```

原因：

- `libero_dino_s_smallvideo_2cam_224_1e-4` 会加载 `configs/model/fastwam_dino_s_smallvideo.yaml`，这是当前 1B smallvideo DINO / no-pool / avgpool / short-intent 系列使用的正确模型配置。
- `libero_dino_s_2cam_224_1e-4` 会加载 `configs/model/fastwam_dino_s.yaml`，这不是 smallvideo 路线，且当前这份配置里的 DINO 路径仍可能是旧的 `/data11/wmx/...`，会导致 eval worker 找不到本地 DINO 权重后 fallback 到 `torch.hub` / HuggingFace，离线机器会失败。
- 之前 avgpool 评测能正常跑，是因为使用了 smallvideo task，`model_path=checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors` 能在 worker `cd /data73/mingxinwang/Spider-s-World-Action-Model` 后正确解析。

另外，在没有 `python` 命令的机器上，manager 虽然可以用绝对路径启动，但 worker 仍会调用脚本里的 `PYTHON` 环境变量。因此需要显式设置：

```bash
export PYTHON=/data73/mingxinwang/conda_envs/spiderwam/bin/python3.10
```

### 10.2 1B Short-DINO-Intent context-after-proprio 评测命令

以下命令用于评测：

```text
/data32/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/short_dino_intent_fromscratch_10ep_auto/checkpoints/weights/step_022000.pt
```

使用 GPU `2,3,7`，每卡最多 4 个 LIBERO 子任务：

```bash
cd /data73/mingxinwang/Spider-s-World-Action-Model

export PYTHON=/data73/mingxinwang/conda_envs/spiderwam/bin/python3.10
export CUDA_VISIBLE_DEVICES=2,3,7
export LIBERO_ROOT=/data73/mingxinwang/LIBERO
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MPLCONFIGDIR=/tmp/matplotlib-cache
export WAIT_FOR_GPUS=0

$PYTHON experiments/libero/run_libero_manager.py \
  task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  ckpt=/data32/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/short_dino_intent_fromscratch_10ep_auto/checkpoints/weights/step_022000.pt \
  EVALUATION.num_trials=30 \
  EVALUATION.output_dir=/data73/mingxinwang/Spider-s-World-Action-Model/evaluate_results/libero/short_dino_intent_30trials_step_022000_3gpu237_mtp4_$(date +%Y%m%d_%H%M%S) \
  MULTIRUN.num_gpus=3 \
  MULTIRUN.max_tasks_per_gpu=4 \
  MULTIRUN.omp_num_threads=2 \
  MULTIRUN.mkl_num_threads=1 \
  MULTIRUN.openblas_num_threads=1 \
  MULTIRUN.numexpr_num_threads=1 \
  model.dino_config.load_backbone=true \
  model.dino_config.latent_spatial_pool=[1,1] \
  +model.intent_config.enabled=true \
  +model.intent_config.history_offsets=[-8,-4,0] \
  +model.intent_config.max_history_frames=3 \
  +model.intent_config.num_intent_tokens=8 \
  +model.intent_config.resampler_dim=1024 \
  +model.intent_config.num_resampler_layers=2 \
  +model.intent_config.num_heads=8 \
  +model.intent_config.mlp_ratio=4.0 \
  +model.intent_config.dropout=0.0
```

说明：

- `model.dino_config.load_backbone=true` 必须开启，因为 LIBERO eval 需要在线编码当前帧和 history frames。
- `model.dino_config.latent_spatial_pool=[1,1]` 必须覆盖，因为该 checkpoint 是 no-pool DINO latent；task 默认可能是 `[1,2]`。
- `+model.intent_config.*` 必须带上，否则模型不会构建 `intent_encoder`，checkpoint 中的 intent 权重也不会加载，评测会退化成非 intent 结构。
- 如果未来在某台机器上 DINO 权重路径解析仍失败，可以额外加：

```bash
  model.dino_config.model_path=/data73/mingxinwang/Spider-s-World-Action-Model/checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors \
```

## 11. Short-DINO-Intent context-after-proprio 当前评测与下一步决策

截至 2026-06-15，当前 1B Short-DINO-Intent / context-after-proprio 的 30-trial LIBERO 评测已经更新进：

```text
libero_dashboard/dashboard_data.json
DINO_TOKEN_PROCESSING_COMPARISON.md
```

评测结果：

| checkpoint | global bs | 等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `step_022000` | 96 | 7.60 | 92.33 | 100.00 | 95.00 | 75.00 | 90.58 |
| `step_024000` | 96 | 8.30 | 95.33 | 95.33 | 87.67 | 79.00 | 89.33 |
| `step_026000` | 96 | 8.99 | 93.00 | 97.67 | 90.67 | 75.00 | 89.08 |
| `step_028000` | 96 | 9.68 | 92.00 | 99.67 | 90.33 | 77.67 | 89.92 |
| `step_028930` | 96 | 10.00 | 93.33 | 98.67 | 90.33 | 77.67 | 90.00 |

对比 no-intent DINO no-pool fresh 10ep：

| route | checkpoint | global bs | 等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DINO no-pool fresh | `step_028930` | 96 | 10.00 | 98.67 | 99.33 | 92.33 | 84.00 | 93.58 |
| Short-DINO-Intent context-after-proprio endpoint | `step_028930` | 96 | 10.00 | 93.33 | 98.67 | 90.33 | 77.67 | 90.00 |
| difference | - | - | - | -5.34 | -0.66 | -2.00 | -6.33 | -3.58 |

当前判断：

- 10ep context-after-proprio 明显没有超过 no-intent no-pool fresh，尤其 LIBERO-10 从 `84.00` 掉到 `77.67`。
- 不能直接判死，因为它引入了 intent encoder / 新 context 分布，可能需要更充分训练；但继续用本机 8 卡 resume 同一条 context-after-proprio 的信息增量不高。
- AMD 卡那边已经有 context-after-proprio / 拼接 short intent 的 20ep 训练，明天评测可以回答“是不是只是 10ep 训练不足”。
- 本机 8 卡更建议今晚启动 `video_prefix` injection ablation：history offsets、K、resampler、loss、global bs、lr 都保持一致，只换注入位置，attribution 最干净。

建议执行顺序：

1. 等 AMD 20ep context-after-proprio eval：如果能明显回到 no-pool 10ep/20ep 附近，再考虑继续加训 context route。
2. 本机 8 卡先训 `model.intent_config.injection_mode=video_prefix`，完整命令见本文件第 8 节“完整训练命令”。
3. 后续对比时用同一组指标：`step_022000/024000/026000/028000/028930`，global bs `96`，等效 epoch 对齐。

## 12. video_prefix 当前训练归档注意事项

2026-06-15 检查 `dinointent` tmux 后确认：

```text
RUN_ID=short_dino_intent_video_prefix_10ep_20260614_083209
run_dir=/data32/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/short_dino_intent_video_prefix_10ep_20260614_083209
resume=/data32/mingxinwang/Spider-s-World-Action-Model/runs/libero_dino_s_smallvideo_2cam_224_1e-4/short_dino_intent_video_prefix_10ep_20260614_083209/checkpoints/state/step_006000
```

这个 run 不是 context-after-proprio 拼接版，而是 `video_prefix` 版：

```yaml
model.intent_config.enabled: true
model.intent_config.injection_mode: video_prefix
model.intent_config.history_offsets: [-8, -4, 0]
model.intent_config.num_intent_tokens: 8
data.train.load_history_dino_latents: true
data.train.history_dino_frame_offsets: [-8, -4, 0]
```

重要归档点：

- 这条线最早是在 `2026-06-14` 启动的 `video_prefix` run，中途被误以为是拼接版 intent 而 `C-c` 掉。
- 2026-06-15 重新启动 guard 时，tmux 环境里已有 `RUN_ID=short_dino_intent_video_prefix_10ep_20260614_083209`，所以 guard 正确进入同一个 `/data32` run 目录，并从最近完整 state `step_006000` 做 full-state resume。
- 虽然后续 resume 命令和新写出的 `config.yaml` 里显示 `learning_rate: 0.0001`，但 full-state resume 会恢复 optimizer/scheduler；日志中实际 LR 为 `lr=4.67e-05`，说明它实际延续的是原 `5e-5 cosine` 训练线，而不是 fresh `1e-4` 从零训练。
- 后续评测/leaderboard 命名不要写成 `video_prefix lr1e-4 fresh`。更准确写法是：

```text
Short-DINO-Intent video_prefix, lr5e-5 cosine, full resume from step_006000, run_id=short_dino_intent_video_prefix_10ep_20260614_083209
```

如果未来要做真正 fresh `1e-4` video_prefix 对照，需要新建不同 `RUN_ID`，且确保 `resume=null`，不要复用这个 run 目录。

## 13. 2026-06-21 handoff：LIBERO-Plus 后的新主线

本节记录 6/19-6/21 期间的新结论，方便新对话继续。

### 13.1 当前已经完成的关键评测

LIBERO-Plus 4 个主要 checkpoint 已经全量跑完，每个都是 `10030` 个 perturbation tasks、每个 task `1 trial`：

| 变体 | checkpoint | Original / LIBERO | Camera | Robot | Lang. | Light | BG | Noise | Layout | Plus Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B no-intent DINO no-pool，本机 | `step_028930` | 96.25 | 28.46 | 21.94 | 50.03 | 74.69 | 40.99 | 25.80 | 46.69 | 39.71 |
| 1B context-intent DINO，AMD 20ep | `step_010860` | 94.90 | 30.96 | 19.42 | 53.42 | 73.99 | 40.15 | 26.42 | 53.25 | 41.17 |
| 5B no-intent DINO，AMD 20ep | `step_010860` | 94.20 | 27.33 | 16.39 | 54.59 | 72.42 | 50.28 | 17.80 | 49.31 | 39.23 |
| 5B context-intent DINO，AMD 20ep | `step_010860` | 92.70 | 33.77 | 15.23 | 52.31 | 81.17 | 54.65 | 30.61 | 51.87 | 43.63 |

对应结果目录：

```text
evaluate_results/libero_plus/libero_plus_1b_nopool_nointent_20ep_step028930_10030tasks_1trial_8gpu_wpg4_20260618_084620
evaluate_results/libero_plus/libero_plus_amd_1b_context_intent_20ep_step010860_10030tasks_1trial_6gpu_wpg4_20260619_093915
evaluate_results/libero_plus/libero_plus_amd_5b_nointent_20ep_step010860_10030tasks_1trial_6gpu_wpg3_20260619_093915
evaluate_results/libero_plus/libero_plus_amd_5b_context_intent_step010860_10030tasks_1trial_6gpu_wpg3_20260619_093915
```

和论文表里的 Fast-WAM 对比：

| Model | Original | Camera | Robot | Lang. | Light | BG | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fast-WAM | 97.60 | 16.40 | 44.50 | 68.90 | 78.20 | 53.70 | 37.70 | 60.70 | 51.50 |
| Our best DINO-only per column | 96.25 | 33.77 | 21.94 | 54.59 | 81.17 | 54.65 | 30.61 | 53.25 | 43.63 |

核心判断：

- DINO-only 没有证明可以全面替代 Wan/VAE world model；Plus Total 仍低于 Fast-WAM。
- DINO-only 在 Camera、Light、BG 上有信号，符合 DINO 预训练对视角/光照/外观变化更稳定的直觉。
- Robot、Language、Noise、Layout 明显弱，说明纯 DINO dynamics 可能丢失了控制所需的精细几何、语言适配和像素扰动鲁棒性。
- Intent 在 Plus 上有正向作用：1B context-intent 比 1B no-intent `+1.46 Total`，5B context-intent 比 5B no-intent `+4.40 Total`。这比原始 LIBERO 上更明显，说明历史信息在 OOD perturbation 下可能更有价值。

### 13.2 LIBERO-Plus 子集含义

LIBERO-Plus 论文把 perturbation 分成 7 类：

- Camera Viewpoints：相机视角变化。
- Robot Initial States：机器人初始状态/姿态变化。
- Language Instructions：语言指令改写或扰动。
- Light Conditions：光照变化。
- Background Textures：背景/纹理变化。
- Sensor Noise：视觉传感器噪声或图像扰动。
- Objects Layout：物体布局、干扰物、目标位置等变化。

本地 `10030` 个任务的大致计数：

| subset | count |
|---|---:|
| Sensor Noise | 1601 |
| Camera Viewpoints | 1599 |
| Robot Initial States | 1550 |
| Language Instructions | 1537 |
| Objects Layout | 1525 |
| Light Conditions | 1142 |
| Background Textures | 1076 |

### 13.3 当前正在跑的训练

截至 2026-06-21，用户反馈有两条训练正在跑，预计约 58 小时后、周三上午出结果：

1. 本机 8 卡：`1B-DINO video_prefix Short-DINO-Intent` 继续训练。
   - 动机：`video_prefix` 10ep 的 best (`step_026000`, Overall 94.00) 已经超过 no-intent 10ep fresh (`93.58`)，但还没到 no-intent 20ep (`96.25`)。
   - 继续时应优先从 `step_026000` 这种 best checkpoint 小学习率 weight-only resume，而不是从 final `step_028930` 继续。
   - 参考 no-pool 从 10ep 到 20ep 的成功 recipe：weight-only、`lr=1e-5`、继续 10ep 左右。

2. 另一台 8 卡 H200：`3-branch MoT no-intent`。
   - 新配置：
     - `configs/model/fastwam_wan5b_dino_s_aux_mot.yaml`
     - `configs/task/libero_wan5b_dino_s_aux_mot_2cam_224_1e-4.yaml`
   - 结构：Wan/VAE video branch + DINO auxiliary branch + action branch。
   - VAE branch 使用原生 Wan 5B hidden/ffn，不使用插值出来的 1B smallvideo checkpoint。
   - DINO branch 仍使用当前 1B DINO route 的 token/loss 设计。
   - 默认 loss：`lambda_video=1.0, lambda_action=1.0, lambda_dino=0.02`。
   - 当前 no-intent 版本先验证“DINO 作为辅助监督是否能和 Wan/VAE 主干共存”。

### 13.4 3-branch MoT 设计判断

目前更合理的叙事已经从“DINO 替代 VAE”转为“DINO 辅助 Wan/VAE”：

- 纯 DINO 1B 在原始 LIBERO 上能接近 Fast-WAM，说明 DINO token dynamics 有可取之处。
- 但 Plus 上纯 DINO 的总鲁棒性不如 Fast-WAM，说明完全丢掉 VAE/Wan latent 不是最稳路线。
- 3-branch MoT 的目标是保留 Wan/VAE 的语言/生成式世界建模能力，同时用 DINO auxiliary loss 引入 Camera/Light/BG 这类 OOD 更稳定的语义特征。

如果周三 `3-branch MoT no-intent` 有提升：

1. 优先继续做 `3-branch MoT + Short-DINO-Intent`。
2. Intent 更建议先用已有更稳的 history encoder；注入方式可先从当前表现更好的 `video_prefix` 思路出发，但必须重新检查三分支 mask。
3. 评测顺序应同时看原始 LIBERO 和 LIBERO-Plus，不能只看原榜。

如果周三 `3-branch MoT no-intent` 没有提升：

1. 先排查 loss 权重和 branch interaction，不要直接否定 DINO auxiliary。
2. 可尝试更低 `lambda_dino` (`0.01`) 或 warmup 后再开 DINO loss。
3. 若 LIBERO 原榜掉分明显，说明辅助分支干扰了主 VAE/action dynamics，需要更弱耦合或 delayed auxiliary。

### 13.5 新对话优先接续项

1. 等周三两条训练结果：`video_prefix resume` 和 `3-branch MoT no-intent`。
2. 对新 checkpoint 先跑原始 LIBERO 30-trial，再挑 best 跑 LIBERO-Plus。
3. 若 3-branch MoT 有提升，立刻排 `3-branch MoT + intent`。
4. 若 `video_prefix` 继续提分，保留它作为 Short-DINO-Intent 主路线；如果只在 10ep 附近小幅超过 no-intent fresh，但追不上 no-intent 20ep，则不再单独投入太多 GPU。
5. Robotwin 是否要跑：可以作为下一阶段泛化验证，但先让 LIBERO-Plus 和 3-branch 结论稳定下来。
