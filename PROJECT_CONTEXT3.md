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
