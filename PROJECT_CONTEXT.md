# Spider's World Action Model — 项目总结

> 最后更新: 2026-05-15
> 仓库: https://github.com/Thu-WangMX/Spider-s-World-Action-Model.git

---

## 一、核心思想：用 DINO 特征替代 VAE Latent

### 背景

FastWAM 原始架构基于 Wan2.2 视频生成模型，用 VAE 将视频编码为 latent，然后在 latent 空间做 flow matching 来联合预测视频和动作。核心问题是：**VAE latent 空间是为视频生成优化的，不一定是最好的动作预测特征空间**。

### 参考论文核心思想

| 论文 | 核心思路 |
|---|---|
| **FastWAM** | Video Expert (Wan2.2 DiT) + Action Expert (小 DiT) 通过 MoT (Mixture of Transformers) 共享 self-attention，KV-cache 推理 |
| **Motus** | 在 FastWAM 基础上加入 Understanding Expert (VLM)，三路 MoT，强调"理解"驱动动作 |
| **LDA-1B** | 用 QwenVL + DINO 做视觉编码，DiT 做 action denoising，scaling law 实验 |

### 我们的改动

**用 frozen DINOv3-ViT-L/16 的特征空间替代 VAE latent 空间**，作为可选方案：

```
原始 FastWAM:
  Video: [B,3,T,H,W] → VAE → [B,48,T/4,H/8,W/8] → WanVideoDiT (Conv3d) → MoT

我们的 DINO 版本:
  Video: [B,3,T,H,W] → DINO (frozen) → [B,1024,T,14,14] → DinoVideoDiT (Linear) → MoT
```

**保持原始 FastWAM 代码完全不动**，通过配置切换。

---

## 二、架构细节

### DINO Feature 数据流

```
输入: [B, 3, T, H, W]  (如 [2, 3, 9, 384, 320])
  ↓ resize to 224×224, frozen ViT-L/16 (patch_size=16)
DINO Feature: [B, 1024, T, 14, 14]  (每帧 196 个 patch token, 1024 维)
  ↓ rearrange → [B, T×196, 1024]
  ↓ Linear(1024 → 3072)
Token 序列: [B, T×196, 3072]
  ↓ MoT (与 Action tokens 共享 self-attention, 30 层 DiTBlock)
  ↓ DinoHead(3072 → 1024)
输出: [B, 1024, T, 14, 14]  (预测的 velocity field)
```

### 维度对比

| 维度 | VAE latent | DINO feature |
|---|---|---|
| 通道 D | 48 | **1024** |
| 时域 T | T_video / 4 | **T_video (无压缩)** |
| 空域 | H/8 × W/8 | **14 × 14** (固定，因为 resize 到 224) |
| 每帧 token 数 | ~1920 | **196** |

### 组件初始化

| 组件 | 初始化方式 |
|---|---|
| DINO Encoder | frozen DINOv3 权重 (本地 safetensors) |
| DinoVideoDiT | 随机初始化 **或** 从 Wan2.2 DiT 迁移 Transformer 层 (可选) |
| ActionDiT | 从 Wan2.2 DiT 缩放得到的预训练权重 |
| MoT | 由上面两个 Expert 的 blocks 组成 |

---

## 三、文件清单

### 新增文件 (4个)

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/fastwam/models/wan22/dino_encoder.py` | 509 | DINO 编码器，零 remap 加载 safetensors |
| `src/fastwam/models/wan22/dino_video_dit.py` | 396 | DINO 空间的 Video Expert |
| `src/fastwam/models/wan22/fastwam_dino.py` | 894 | FastWAM-DINO 主模型 (训练+推理) |
| `configs/model/fastwam_dino.yaml` | 65 | DINO 模型配置 |
| `configs/task/robotwin_dino_3cam_224_1e-4.yaml` | 27 | RobotWin DINO 训练 task |

### 修改的原始文件 (1个)

| 文件 | 改动 |
|---|---|
| `src/fastwam/runtime.py` | 新增 `create_fastwam_dino()` 工厂函数 (+85 行) |

### 原始文件 (未修改)

- `src/fastwam/models/wan22/fastwam.py` — 原始 FastWAM 主模型
- `src/fastwam/models/wan22/wan_video_dit.py` — 原始 Video Expert
- `src/fastwam/models/wan22/action_dit.py` — Action Expert (两种模式共用)
- `src/fastwam/models/wan22/mot.py` — MoT (两种模式共用)

---

## 四、需要迁移的权重文件

| 文件 | 大小 | 用途 | 必需? |
|---|---|---|---|
| `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | 3.9 GB | ActionDiT 预训练权重 | **两种模式都需要** |
| DINOv3 safetensors (路径见 yaml 中 `dino_config.model_path`) | 1.2 GB | DINO 编码器权重 | DINO 模式需要 |
| `checkpoints/Wan-AI/` + `DiffSynth-Studio/` | ~30 GB | Wan2.2 VAE + Text Encoder | 原始 FastWAM 需要 |
| `data/` | ~35 GB+ | 训练数据 | 训练需要 |

**最小迁移 (只跑 DINO)**: ActionDiT 权重 (3.9G) + DINOv3 权重 (1.2G) + 训练数据

---

## 五、训练指令

### 基本命令

```bash
cd /path/to/FastWAM

# DINO 版本 — 8卡训练
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4

# 原始 FastWAM — 8卡训练 (对照组)
bash scripts/train_zero1.sh 8 task=robotwin_joint_3cam_384_1e-4
```

### 可选参数覆盖

```bash
# DinoVideoDiT 从 Wan2.2 继承 Transformer 层权重 (推荐，收敛更快)
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4 \
  model.video_dit_init_from_wan=true

# 调小 batch_size
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4 \
  batch_size=8

# 调学习率
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4 \
  learning_rate=5e-5

# 开 wandb 监控
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4 \
  wandb.enabled=true wandb.name=robotwin_dino

# 单卡调试
bash scripts/train_zero1.sh 1 task=robotwin_dino_3cam_224_1e-4

# 组合使用
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4 \
  model.video_dit_init_from_wan=true \
  batch_size=8 \
  learning_rate=5e-5 \
  wandb.enabled=true \
  wandb.name=robotwin_dino_wan_init
```

### 重要配置说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model.video_dit_init_from_wan` | false | 从 Wan2.2 DiT 迁移权重到 DinoVideoDiT |
| `model.wan_model_id` | Wan-AI/Wan2.2-TI2V-5B | Wan2.2 模型 ID (需要能下载) |
| `model.skip_dit_load_from_pretrain` | false | 跳过 ActionDiT 预训练加载 |
| `model.mot_checkpoint_mixed_attn` | true (yaml) / false (task) | MoT 梯度检查点 (省显存) |
| `batch_size` | 16 | 每卡 batch size |
| `learning_rate` | 1e-4 | 学习率 |
| `num_epochs` | 5 | 训练轮数 |

---

## 六、常用 Git 命令

```bash
# 日常提交推送
git add .
git commit -m "你的提交信息"
git push

# 查看状态
git status
git log --oneline -5

# 拉取最新代码
git pull

# 首次 clone
git clone https://github.com/Thu-WangMX/Spider-s-World-Action-Model.git
cd Spider-s-World-Action-Model
```

---

## 七、DINOv3 权重说明

- **模型**: DINOv3-ViT-L/16 (改进的自监督视觉模型)
- **格式**: safetensors (安全张量格式)
- **参数**: ~300M, embed_dim=1024, patch_size=16, 24层, 16 heads
- **Key 格式** (零 remap，直接匹配):
  - `embeddings.patch_embeddings.weight` [1024, 3, 16, 16]
  - `embeddings.cls_token` [1, 1, 1024]
  - `embeddings.register_tokens` [1, 4, 1024]
  - `layer.{i}.attention.q_proj.weight` [1024, 1024] (有 bias)
  - `layer.{i}.attention.k_proj.weight` [1024, 1024] (**无 bias**)
  - `layer.{i}.attention.v_proj.weight` [1024, 1024] (有 bias)
  - `layer.{i}.attention.o_proj.weight` [1024, 1024] (有 bias)
  - `layer.{i}.mlp.up_proj.weight` [4096, 1024] (GELU MLP)
  - `layer.{i}.mlp.down_proj.weight` [1024, 4096]
  - `layer.{i}.layer_scale1.lambda1` [1024]
- **无显式位置嵌入**
- **4 个 register tokens**

---

## 八、下一步实验计划

1. **Baseline**: 跑 DINO 版本 (随机初始化 DinoVideoDiT)，对比原始 FastWAM
2. **Wan2.2 初始化**: 开启 `video_dit_init_from_wan=true`，验证是否加速收敛
3. **涨点方向**:
   - 调整 loss 权重 (lambda_video vs lambda_action)
   - 探索不同 DINO 分辨率 (目前固定 224×224)
   - 探索 DINO feature 是否需要 normalize
   - 考虑时域建模 (当前逐帧独立编码)


4.一个潜在的调参方向
DINO 的 feature 维度是 1024（vs VAE 的 48），这意味着 DINO 的 video loss 数值天然会比 VAE 版本大很多（因为 MSE 是在通道维度上 mean 的，1024 维的特征方差通常比 48 维的 latent 大）。如果发现 video loss 远大于 action loss，可以试试：

bash
# 降低 video loss 权重，让 action loss 更受重视
bash scripts/train_zero1.sh 8 task=robotwin_dino_3cam_224_1e-4 \
  model.loss.lambda_video=0.1 model.loss.lambda_action=[1.0
训练时观察 wandb 里 loss_video 和 loss_action 的比例就知道要不要调了。🚀

可能有问题需要再次确认的点：
1. DINO backbone 会被 nn.Module.to() 递归移动

DinoVideoEncoder 是 nn.Module 子类，self.backbone = _DinoViT(...) 会自动注册为子模块。当 FastWAM_DINO.__init__ 最后调 self.to(self.device) 时，PyTorch 默认的递归 .to() 会先移动 backbone，然后你自定义的 to() 又手动 .to() 一次。结果是对的（backbone 在正确的 device 上），只是移了两次（无害）。

2. 训练时 DINO 编码每帧是串行的

Python
for frame_idx in range(num_frames):
    frame = video[:, :, frame_idx]
    features = self.encode_frames(frame)
如果帧数多（比如 9 帧），这会比 batch 编码慢。但因为 DINO 是 frozen + @torch.no_grad()，显存开销不大。如果训练速度是瓶颈，可以改成把所有帧拼成一个大 batch：

Python
# 优化版（以后需要时再改）
frames = rearrange(video, "b c t h w -> (b t) c h w")
features = self.encode_frames(frames)  # [(B*T), N, D]
features = rearrange(features, "(b t) n d -> b t n d", b=batch_size)
但现在的串行版本功能完全正确，只是效率差一点。不需要现在改。

3. loss_lambda_video 可能需要调

DINO 1024 维 feature 的 MSE loss 数值会比 VAE 48 维 latent 的大很多。训练时看 wandb 里 loss_video vs loss_action 的比例，如果差距太大就调 lambda_video。

结论
代码层面没有发现任何 bug 或错误。之前修复的 from_config 签名缺失是唯一的 bug，已经修复并提交。现在的代码可以放心开训！

去 8×A800 上冲吧！ 🕷️🔥🚀

---

## 九、2026-05-15 DINO/LDA-1B 复查补充

参考 LDA-1B 的 DINO latent 预测实现后，发现并修复了一个关键问题：

- DINOv3 没有 learned absolute position embedding，但 attention 内部需要 2D RoPE。
- 原先轻量 DINO loader 只做到权重 key/shape 对齐，没有在 Q/K 上应用 RoPE，且 token 顺序是 `CLS + patch + register`。
- 已修复为 DINOv3/HF/LDA 一致的实现：
  - token 顺序：`CLS + register + patch`
  - RoPE 只作用在 patch tokens，CLS/register prefix 不旋转
  - 输出仍然只返回 patch tokens，shape 为 `[B, 196, 1024]`

已验证：

```bash
/data11/wmx/miniconda3/envs/fastwam/bin/python -m py_compile \
  src/fastwam/models/wan22/dino_encoder.py \
  src/fastwam/models/wan22/fastwam_dino.py \
  src/fastwam/trainer.py \
  scripts/train.py

git diff --check
```

并做过最小 DINO 前向 sanity check：随机 224x224 单帧能输出 `features (1, 196, 1024) torch.float32 True`。

LDA-1B 值得参考但不建议开跑前再大改的点：

1. 它把多视角图像作为独立 view 编码，再拼接 tokens；我们当前 2cam 是横向拼接后 resize 到 224x224，会压缩两路相机。这是最有价值的后续 ablation，但现在改动较大，建议先跑 baseline。
2. 它有 policy / forward dynamics / inverse dynamics / video gen 四种任务 embedding；我们当前 FastWAM-DINO 已经做 video latent + action joint loss，LIBERO 首跑不需要加任务 embedding。
3. 它用 repeated diffusion steps 提高每个样本的 timestep 覆盖；我们可以后续加，但会增加显存/算力，首跑不建议改。
4. `model.video_dit_init_from_wan=true` 仍建议作为第二轮实验；第一轮保持 `false`，减少 Wan2.2 权重/加载链路带来的变量。

---

## 十、2026-05-16 训练提速与 DINO-S ablation 准备

在不打断已有训练的前提下，补了两个低风险提速项，下一次启动训练生效：

1. `DinoVideoEncoder` 增加 `encode_microbatch_size`。
2. DINO 视频编码从逐帧 Python loop 改成 `[B, C, T, H, W] -> [B*T, C, H, W]` 后按 microbatch 批量过 DINO。

DINO-L 默认配置：

```yaml
model.dino_config.encode_microbatch_size: 72
```

已验证 DINO-L：

```text
vitl missing 0 unexpected 0 keys 415
latent (2, 1024, 3, 14, 14) torch.float32 True
```

同时准备了 DINO-S ablation：

- 新模型配置：`configs/model/fastwam_dino_s.yaml`
- 新任务配置：`configs/task/libero_dino_s_2cam_224_1e-4.yaml`
- 权重：`checkpoints/dinov3_weights/dinov3_vits16_timm_lvd1689m.safetensors`
- 来源：公开可下载的 `timm/vit_small_patch16_dinov3.lvd1689m`

说明：

- 官方 `facebook/dinov3-vits16-pretrain-lvd1689m` 是 gated，未登录无法直接下载。
- timm 版权重 key 是 `blocks.*.attn.qkv.weight` 格式，已在 loader 中增加 timm->轻量 DINOv3 key remap。
- remap 会拆分 q/k/v，映射 `blocks/norm/mlp/gamma/reg_token/patch_embed`，并给 timm 省略的全零 Q/V bias 与 mask token 补零。

已验证 DINO-S：

```text
missing 0 []
unexpected 0 []
shape_mismatch 0 []
latent (2, 384, 3, 14, 14) torch.float32 True
```

DINO-S 运行命令示例：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train_zero1.sh 8 task=libero_dino_s_2cam_224_1e-4 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  wandb.enabled=true \
  wandb.project=fast-wam \
  wandb.name=libero_dino_s_2cam224_bs4ga4 \
  2>&1 | tee -a logs/libero_dino_s_2cam224_bs4ga4_$(date +%Y%m%d_%H%M%S).log
```

注意：DINO-S 主要减少 DINO encoder 与 DINO latent 通道开销；Video DiT/MoT 仍保持 5B 配置，所以不保证 `batch_size=8` 一定能过，只能作为下一轮 ablation 尝试。

---

## 十一、2026-05-16 2cam DINO spatial size 修正

复查时发现 LIBERO 2cam 数据路径会先把两路相机横向拼接，最终 `video_size` 是 `[224, 448]`。DINO encoder 不能再假设固定 `[224, 224] -> 14x14`，否则旧逻辑可能只取前 196 个 patch，等价于丢掉右半边相机信息。

已修正：

1. `DinoVideoEncoder.encode_video_to_latent()` 改为根据实际输入 `H/W` 动态计算 `H_grid/W_grid/num_patches`。
2. `configs/task/libero_dino_2cam_224_1e-4.yaml` 和 `configs/task/libero_dino_s_2cam_224_1e-4.yaml` 在 task 层覆盖：

```yaml
model.dino_config.input_resolution: [224, 448]
```

3. 后续发现 true 14x28 tokens 下 DINO-S `batch_size=4` 会 OOM；增加可选：

```yaml
model.dino_config.latent_spatial_pool: [1, 2]
model.dino_config.encode_microbatch_size: 16
```

这会先让 DINO 看完整 `224x448`，再把 DINO latent 从 `14x28` 池化成 `14x14`，保留双相机信息但把 VideoDiT token 数压回旧规模。

4. LIBERO DINO-L/DINO-S task 默认改成更保守的 `batch_size=4`、`gradient_accumulation_steps=4`、`model.mot_checkpoint_mixed_attn=true`。

已验证：

```text
DINO-L missing=0 unexpected=0 keys=415
DINO-S missing=0 unexpected=0 keys=211
DINO-L 224x448 latent: (1, 1024, 1, 14, 28)
DINO-S 224x448 latent: (1, 384, 2, 14, 28)
DINO-S 224x448 + latent_spatial_pool [1,2]: (1, 384, 2, 14, 14)
```

注意：已经在跑的训练进程不会自动加载这些代码改动；下一次重启才会使用 pooled 2cam DINO token。

---

## 十二、2026-05-16 DINO-S small-video 配置

考虑到 DINO 版不再依赖 Wan 视频权重，新增一套更轻的 video expert 配置，同时保留原来的大 video 配置：

- 大 video：`configs/model/fastwam_dino_s.yaml`
- 小 video：`configs/model/fastwam_dino_s_smallvideo.yaml`
- 小 video LIBERO 2cam task：`configs/task/libero_dino_s_smallvideo_2cam_224_1e-4.yaml`

设计：

1. ActionDiT 仍保持原来的 1B 配置：

```yaml
hidden_dim: 1024
ffn_dim: 4096
num_heads: 24
attn_head_dim: 128
num_layers: 30
```

这样可以继续加载 `checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`。

2. DinoVideoDiT 从 Wan2.2-5B 形状缩小到 ActionDiT-scale：

```yaml
video_dit_config:
  hidden_dim: 1024
  ffn_dim: 4096
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30
```

3. MoT 仍满足约束：video/action 的 `num_heads`、`attn_head_dim`、`num_layers` 一致。

已验证：

```text
bigvideo 5.00B
smallvideo 1.02B
Hydra task=libero_dino_s_smallvideo_2cam_224_1e-4:
  model=fastwam_dino_s_smallvideo
  DINO-S input_resolution=[224,448]
  latent_spatial_pool=[1,2]
  video hidden_dim=1024 ffn_dim=4096
  action hidden_dim=1024 ffn_dim=4096
```

运行示例：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train_zero1.sh 8 task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  wandb.enabled=true \
  wandb.project=fast-wam \
  wandb.name=libero_dino_s_smallvideo_2cam224_pool_bs4ga4 \
  2>&1 | tee -a logs/libero_dino_s_smallvideo_2cam224_pool_bs4ga4_$(date +%Y%m%d_%H%M%S).log
```

切回大 video 只需要使用 `task=libero_dino_s_2cam_224_1e-4`，或在命令行覆盖 `model=fastwam_dino_s`。

---

## 十三、2026-05-16 当前代码体检、LDA-1B 对比与可选参数

后续约定：关键代码修改、配置新增、实验结论和重要命令都追加到本文件末尾，避免睡醒之后断片。

### 1. 当前代码检查结果

已做基础检查：

```bash
git diff --check
/data11/wmx/miniconda3/envs/fastwam/bin/python -m py_compile \
  scripts/train.py \
  src/fastwam/runtime.py \
  src/fastwam/trainer.py \
  src/fastwam/models/wan22/dino_encoder.py \
  src/fastwam/models/wan22/fastwam_dino.py
```

结果：通过。

Hydra 配置展开也通过：

```bash
conda run -n fastwam python scripts/train.py --cfg job \
  task=libero_dino_s_smallvideo_2cam_224_1e-4
```

关键展开项确认：

- `batch_size: 4`
- `gradient_accumulation_steps: 4`
- `model=fastwam_dino_s_smallvideo`
- `video_dit_init_from_wan: false`
- `model.dino_config.model_name: dinov3-vits16-timm`
- `model.dino_config.input_resolution: [224, 448]`
- `model.dino_config.latent_spatial_pool: [1, 2]`
- `model.dino_config.encode_microbatch_size: 16`
- video/action expert 均为 `hidden_dim=1024, ffn_dim=4096, num_heads=24, attn_head_dim=128, num_layers=30`

结论：当前 small-video 配置能被 Hydra 正确解析，基础 Python 语法检查通过，可以作为下一轮轻量实验入口。尚未在本次检查中启动新的训练进程；正在跑的旧训练不会自动加载这些新代码/新配置。

### 2. LDA-1B 与当前 FastWAM-DINO 参数量对比

本地 `/data11/wmx/LDA-1B` 里和 action + DINO 预测相关的配置/代码显示，LDA-1B 不是 5B 级别的 video expert：

- `LDA_pretrain.yaml` / `LDA_robocasa.yaml` 使用：
  - `action_model_type: DiT-B`
  - `vision_encoder_type: dinov3`
  - `vision_encoder_size: s`
  - `num_target_vision_tokens: 32`
  - `obs_horizon: 1`
  - `num_views: 1`
  - `diffusion_model_cfg.num_layers: 8`
  - `diffusion_model_cfg.output_dim: 2560`
- `GR00T_ActionHeader_uwm_mmdit.py` 中 `DiT-B` 对应：
  - `input_embedding_dim: 768`
  - `num_attention_heads: 12`
  - `attention_head_dim: 64`
- `starvla_cotrain_libero.yaml` 使用更 LIBERO 向的设置：
  - `hidden_size: 1024`
  - `num_target_vision_tokens: 32`
  - `diffusion_model_cfg.num_layers: 16`
  - `diffusion_model_cfg.output_dim: 1024`

所以 LDA 的设计更像“Qwen/VLM + 小 MMDiT action head，同时预测少量 future DINO tokens”。它确实预测 action 和 DINO/视觉 token，但不是用 Wan2.2-5B 形状的完整 video DiT；参数规模和当前 `fastwam_dino_s.yaml` 的 5B video expert 差距很大。

当前 FastWAM-DINO 三种选择：

- DINO-L big-video：DINO-L encoder + Wan2.2-5B 形状 DinoVideoDiT。
- DINO-S big-video：DINO-S encoder + Wan2.2-5B 形状 DinoVideoDiT。
- DINO-S small-video：DINO-S encoder + 约 1B 级 DinoVideoDiT。

已经用 meta init 估算过 DinoVideoDiT 参数：

```text
bigvideo   5.00B
smallvideo 1.02B
```

因此，如果目标是更接近 LDA-1B 的轻量设定，优先跑 `task=libero_dino_s_smallvideo_2cam_224_1e-4`。

### 3. 现在主要可选参数

推荐入口：

```bash
task=libero_dino_s_smallvideo_2cam_224_1e-4
```

可切换 task：

- `task=libero_dino_2cam_224_1e-4`：DINO-L + big-video，容量最大、最慢、最吃显存。
- `task=libero_dino_s_2cam_224_1e-4`：DINO-S + big-video，只减小 DINO encoder/latent channel，VideoDiT 仍是 5B。
- `task=libero_dino_s_smallvideo_2cam_224_1e-4`：DINO-S + small-video，当前最建议的新实验。

显存/速度相关：

- `batch_size=4`：当前 task 默认值；如果 OOM，降到 `batch_size=2`。
- `gradient_accumulation_steps=4`：当前默认；若 `batch_size=2` 但想保持等效 batch，可用 `gradient_accumulation_steps=8`。
- `model.mot_checkpoint_mixed_attn=true`：建议保持开启，省显存。
- `model.dino_config.latent_spatial_pool=[1,2]`：DINO 看完整 224x448 双相机，再把 14x28 latent 池化为 14x14，建议默认开启。
- `model.dino_config.latent_spatial_pool=[1,1]`：不池化，保留 14x28 全 tokens，质量上限可能更高但显存/速度压力大。
- `model.dino_config.encode_microbatch_size=16`：LIBERO 2cam 默认；只影响 frozen DINO 编码峰值显存，不解决 MoT 主体 OOM。

初始化/结构相关：

- `model.video_dit_init_from_wan=false`：DINO-S 和 small-video 推荐值。
- `model.video_dit_init_from_wan=true`：只适合 big-video Wan2.2 形状，small-video 不能直接用。
- `model.dino_config.normalize_features=false`：当前保持原始 DINO feature；后续可做 ablation。

loss/训练长度相关：

- `model.loss.lambda_video=1.0`
- `model.loss.lambda_action=1.0`
- `num_epochs=10`：DINO-S big/small 当前默认。
- `num_epochs=30`：DINO-L task 当前默认。
- `max_steps=null`：按 epoch 跑；也可覆盖成固定步数。

建议 small-video 训练命令：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/train_zero1.sh 8 task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  wandb.enabled=true \
  wandb.project=fast-wam \
  wandb.name=libero_dino_s_smallvideo_2cam224_pool_bs4ga4 \
  2>&1 | tee -a logs/libero_dino_s_smallvideo_2cam224_pool_bs4ga4_$(date +%Y%m%d_%H%M%S).log
```

注意：如果当前 8 卡已有训练占用，不要同时在同一批 GPU 上启动这条命令。

---

## 十四、2026-05-16 01:38 训练进程确认与 eval 说明

当前仍在跑的训练进程：

```text
start: 2026-05-16 01:38:19/01:39:xx
task: libero_dino_s_2cam_224_1e-4
output_dir: runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20
log: logs/libero_dino_2cam224_bs4ga4_20260516_013820.log
batch_size: 2
gradient_accumulation_steps: 4
```

这版实际保存的 run config 中：

```yaml
model.dino_config:
  model_name: dinov3-vits16-timm
  input_resolution: [224, 448]
  feature_dim: 384
  encode_microbatch_size: 144
```

没有 `latent_spatial_pool` 字段。启动日志也确认：

```text
resolution=(224, 448)
grid=(14, 28)
num_patches=392
encode_microbatch_size=144
```

所以这版是 **未 token 池化的 DINO-S 2cam 全 token 版本**，不是后来默认的 `[1,2] pooled` 版本。

训练内 eval 已经在跑：

```text
eval_every: 200
step=4000 val_loss=0.0631 action_l2=0.0219 action_l1=0.0892
```

这里的 eval 是训练集/验证集上的 open-loop validation/action metric，不是 LIBERO simulator success rate。

当前已有 checkpoint：

```text
runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_002000.pt
runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_004000.pt
```

如果要对这版 checkpoint 做 LIBERO simulator eval，必须保持和训练时一致的 **无池化** DINO 设置。因为当前工作区里的 `configs/task/libero_dino_s_2cam_224_1e-4.yaml` 后来已经加入了 `latent_spatial_pool: [1,2]`，所以 eval 命令需要显式覆盖：

```bash
'model.dino_config.latent_spatial_pool=[1,1]'
'model.dino_config.encode_microbatch_size=144'
```

并保持：

```bash
EVALUATION.visualize_future_video=false
```

DINO 版目前用于 LIBERO success eval 的路径是 action-only `infer_action()`；不要打开 future video 可视化。

单任务 smoke eval 示例：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam

CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py \
  task=libero_dino_s_2cam_224_1e-4 \
  ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_004000.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  EVALUATION.visualize_future_video=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=144 \
  EVALUATION.output_dir=evaluate_results/libero_dino_s_fulltoken_step004000_smoke
```

注意：当前训练占满 8 卡时不要在同一批 GPU 上并行启动 simulator eval，否则会抢显存并影响训练。

---

## 十五、2026-05-16 DINO full-token 曲线判断

当前 01:38 这版 DINO-S full-token big-video 训练到约 6.8k step 时，日志显示：

```text
train/loss_video: 多数在 0.01-0.02
train/loss_action: 多数在 0.05-0.10，偶尔 0.03 或 0.15
eval/action_l1: 抖动较大，约 0.06-0.13
eval/action_l2: 抖动较大，约 0.01-0.06
```

和原始 FastWAM 对比：

- action loss 公式基本一致，都是 action flow-matching MSE。
- 原始 FastWAM 曾在约 21700 step 把 action loss 降到 `<0.001`，video loss 约 `0.07`。
- 当前 DINO 版 video loss 更低不代表视频/动作一定更好；DINO feature MSE 和 VAE latent MSE 数值不可直接按大小比较。
- 当前 full-token DINO 版是 `14x28=392` video tokens/frame，VideoDiT 仍是 5B 形状，且 DINO video target 对模型来说可能更容易快速拟合；action 分支可能被 full-token video/MoT 优化牵制。

判断：

- 不建议无脑跑满 `43400` step。
- 如果目标是 LIBERO action/success，当前曲线已经显示 action 学习效率明显弱于原始 FastWAM。
- 可以最多继续等到 `8000` 或 `10000` step 再看；如果 `train/loss_action` 仍长期在 `0.05-0.10`，应优先转实验。

优先后续实验：

1. 从当前或下一 checkpoint 开新 run，降低 video loss：

```bash
model.loss.lambda_video=0.1
model.loss.lambda_action=1.0
```

更激进可以试：

```bash
model.loss.lambda_video=0.0
model.loss.lambda_action=1.0
```

用于判断 action plateau 是否来自 video objective / MoT full-token 牵制。

2. 切到 pooled DINO-S：

```bash
'model.dino_config.latent_spatial_pool=[1,2]'
```

把 2cam DINO tokens 从 `14x28=392` 压回 `14x14=196`，减少 full-token video 对 action 的压力。

3. 新开 DINO-S small-video：

```bash
task=libero_dino_s_smallvideo_2cam_224_1e-4
```

这更接近 LDA-1B 的轻量思路，但不能直接无痛复用当前 5B video checkpoint。

---

## 十六、2026-05-16 对齐官方 FastWAM 的 LIBERO 评测协议

官方 `yuantianyuan01/FastWAM` README 中，训练后 LIBERO 评测入口是：

```bash
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

本地 `/data11/wmx/FastWAM` 与当前仓库的 `configs/sim_libero.yaml` / `experiments/libero/run_libero_manager.py` 基本一致。默认协议：

- suites:
  - `libero_10`
  - `libero_goal`
  - `libero_spatial`
  - `libero_object`
- `EVALUATION.num_trials: 50`
- `EVALUATION.replan_steps: 10`
- `EVALUATION.binarize_gripper: true`
- `EVALUATION.visualize_future_video: false`
- `EVALUATION.num_inference_steps: ${eval_num_inference_steps}`，当前 train 默认通常是 10
- 输出每个 task 的 `*_results.json`
- 汇总脚本：`python experiments/libero/summarize_results.py --output_dir <eval_output_dir>`

对当前 01:38 DINO-S full-token checkpoint 做官方协议评测时，必须显式对齐训练设置：

```bash
task=libero_dino_s_2cam_224_1e-4
ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt
'model.dino_config.latent_spatial_pool=[1,1]'
model.dino_config.encode_microbatch_size=144
EVALUATION.visualize_future_video=false
```

原因：当前工作区 task config 后来已经改成 pooled DINO；但这个 checkpoint 训练时没有 `latent_spatial_pool`，日志确认是 `grid=(14,28), num_patches=392`。

建议先 smoke：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python experiments/libero/eval_libero_single.py \
  task=libero_dino_s_2cam_224_1e-4 \
  ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10 \
  EVALUATION.visualize_future_video=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=144 \
  EVALUATION.output_dir=evaluate_results/libero_dino_s_fulltoken_step008000_smoke
```

如果 smoke 能跑，再按官方 manager 跑全量：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_list> python experiments/libero/run_libero_manager.py \
  task=libero_dino_s_2cam_224_1e-4 \
  ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt \
  EVALUATION.num_trials=50 \
  MULTIRUN.max_tasks_per_gpu=1 \
  EVALUATION.output_dir=evaluate_results/libero_dino_s_fulltoken_step008000_official \
  EVALUATION.visualize_future_video=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=144
```

`MULTIRUN.max_tasks_per_gpu=1` 比官方默认 `2` 更保守，因为当前 full-token + 5B video expert 显存压力大。若显存充足再考虑调回 `2`。

汇总：

```bash
python experiments/libero/summarize_results.py \
  --output_dir evaluate_results/libero_dino_s_fulltoken_step008000_official
```

---

## 十七、2026-05-16 LIBERO eval 缺包问题

运行：

```bash
python experiments/libero/run_libero_manager.py ...
```

如果报：

```text
ModuleNotFoundError: No module named 'libero'
```

原因是当前 `fastwam` conda 环境中没有安装 official LIBERO Python package。已确认：

```text
import libero -> ModuleNotFoundError
pip show libero hf-libero mujoco robosuite bddl -> not found
```

FastWAM README 也说明：跑 LIBERO benchmark 前需要先安装 official LIBERO environment，然后补：

```bash
pip install mujoco==3.3.2
```

注意：official LIBERO 老安装说明会安装 Torch 1.11；不要在当前 `fastwam` 环境里直接无脑执行会降级 torch 的命令。更稳做法：

```bash
cd /data11/wmx
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

source /data11/wmx/miniconda3/bin/activate fastwam
pip install -e . --no-deps
pip install mujoco==3.3.2 robosuite bddl easydict cloudpickle opencv-python h5py
```

安装后检查：

```bash
python - <<'PY'
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print("LIBERO import OK", benchmark.get_benchmark_dict().keys())
PY
```

如果后续报缺其他小依赖，再按报错逐个补，优先避免重装/降级 `torch`、`torchvision`。

---

## 十八、2026-05-16 训练与评测脚本增加等卡逻辑

新增通用等待器：

```text
scripts/wait_for_gpus.py
```

功能：

- 轮询 `nvidia-smi`
- 等到指定数量 GPU 同时满足空闲条件
- 输出选中的 GPU id 列表，例如 `0,1,2,3,4,5,6,7`
- 默认需要连续稳定 `2` 次检查才放行，避免刚释放瞬间被误判

已接入：

- `scripts/train_zero1.sh`
- `scripts/train_zero2.sh`
- `experiments/libero/run_libero_parallel_test.sh`
- `experiments/robotwin/run_robotwin_manager.py`

默认开启：

```bash
WAIT_FOR_GPUS=1
```

如需跳过：

```bash
WAIT_FOR_GPUS=0 bash scripts/train_zero1.sh 8 task=...
```

可调环境变量：

```bash
WAIT_GPU_MAX_USED_MB=1024      # 单卡显存占用 <= 1024 MiB 视为空闲
WAIT_GPU_MAX_UTIL=10           # GPU util <= 10% 视为空闲
WAIT_GPU_INTERVAL=30           # 每 30 秒检查一次
WAIT_GPU_STABLE_CHECKS=2       # 连续 2 次满足才启动
WAIT_GPU_TIMEOUT=0             # 0 表示无限等待；>0 表示超时秒数
```

如果用户已经指定：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5
```

等待器只会在这几个 GPU 里挑；如果不指定，则从全机器 GPU 中挑够数量。

训练示例：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam

WAIT_GPU_INTERVAL=15 WAIT_GPU_MAX_USED_MB=2048 \
bash scripts/train_zero1.sh 8 task=libero_dino_s_smallvideo_2cam_224_1e-4 \
  +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  wandb.enabled=true
```

LIBERO eval 示例：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam

WAIT_GPU_INTERVAL=15 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python experiments/libero/run_libero_manager.py \
  task=libero_dino_s_2cam_224_1e-4 \
  ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt \
  EVALUATION.num_trials=50 \
  MULTIRUN.max_tasks_per_gpu=1 \
  EVALUATION.visualize_future_video=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=144
```

已验证：

```bash
bash -n scripts/train_zero1.sh
bash -n scripts/train_zero2.sh
bash -n experiments/libero/run_libero_parallel_test.sh
python3 -m py_compile scripts/wait_for_gpus.py experiments/robotwin/run_robotwin_manager.py
git diff --check
```

并用宽松阈值 dry run：

```bash
python3 scripts/wait_for_gpus.py --count 1 --max-used-mb 999999 --max-util 100 --stable-checks 1 --timeout 1
```
## 十九、2026-05-17 LIBERO 评测环境与 DINO 评测入口修复

- 评测环境已在 `fastwam` conda 环境中补齐核心依赖：`mujoco==3.3.2`、`robosuite==1.4.1`、`bddl==1.0.1`、`gym==0.25.2`、`easydict==1.9`、`h5py`、`opencv-python-headless==4.6.0.66`、`matplotlib` 以及 MuJoCo/robosuite 所需的 `glfw/pyopengl/etils/scipy/numba/pynput/future` 等。
- 没有安装 LIBERO 的完整 `requirements.txt`，避免降级当前训练环境中的 `torch/hydra/numpy/wandb/transformers`。
- LIBERO 源码使用本机已有路径 `/data11/wx/openpi/third_party/libero`，通过环境中的 `.pth` 接入；`~/.libero/config.yaml` 已指向该 LIBERO 的 benchmark/bddl/init/assets 路径。
- 为避免重新下载大权重，已在本仓库添加软链接到 `/data11/wmx/FastWAM/checkpoints`：
  - `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors`
  - `checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`
  - `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl`
- 评测入口修复：
  - `experiments/libero/eval_libero_single.py` 现在强制把当前仓库 `src` 加到 `sys.path`，避免误用环境中其他 FastWAM 代码。
  - `experiments/libero/libero_utils.py` / `eval_libero_single.py` 默认设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，兼容 headless MuJoCo 和 PyTorch 2.6+ 加载 LIBERO 老 init states。
  - `eval_libero_single.py` 会按模型 `infer_action/infer_joint` 签名过滤 kwargs，避免 DINO 版不支持 `negative_prompt/text_cfg_scale/tiled` 等参数时报错。
  - `src/fastwam/models/wan22/helpers/loader.py` 新增 `load_dit/load_vae` 可选开关，默认不变；DINO text-only eval 路径关闭多余 Wan DiT/VAE 加载，避免不必要的 5B 随机 DiT 初始化和 VAE 下载。
  - `experiments/libero/run_libero_parallel_test.sh` 支持 `PYTHON=/path/to/python`，tmux 评测 worker 不再依赖 tmux server 里碰巧是什么 Python。
- 已完成 smoke eval：
  - 命令使用 GPU0，checkpoint：`runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt`
  - 显式对齐该 checkpoint 的 full-token 设置：`model.dino_config.latent_spatial_pool=[1,1]`、`model.dino_config.encode_microbatch_size=144`
  - `libero_spatial task_id=0 num_trials=1` 完整跑通，结果文件：
    `evaluate_results/libero_dino_s_fulltoken_step008000_smoke_gpu0/libero_spatial/gpu0_task0_results.json`
  - smoke 结果为 `0/1`，只表示流程已通，不代表正式指标。
- 若要用 4-7 四张卡跑官方对比评测，建议等当前 4-7 训练停掉后执行：

```bash
cd /data11/wmx/Spider-s-World-Action-Model
source /data11/wmx/miniconda3/bin/activate fastwam

PYTHON=/data11/wmx/miniconda3/envs/fastwam/bin/python \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
WAIT_FOR_GPUS=1 \
WAIT_GPU_MAX_USED_MB=2048 \
WAIT_GPU_MAX_UTIL=10 \
WAIT_GPU_INTERVAL=15 \
python experiments/libero/run_libero_manager.py \
  task=libero_dino_s_2cam_224_1e-4 \
  ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt \
  EVALUATION.dataset_stats_path=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/dataset_stats.json \
  EVALUATION.num_trials=50 \
  MULTIRUN.num_gpus=4 \
  MULTIRUN.max_tasks_per_gpu=1 \
  EVALUATION.output_dir=evaluate_results/libero_dino_s_fulltoken_step008000_official_4gpu \
  EVALUATION.visualize_future_video=false \
  'model.dino_config.latent_spatial_pool=[1,1]' \
  model.dino_config.encode_microbatch_size=144
```

## 二十、2026-05-17 DINO-S 8k LIBERO 评测全失败后的训练代码排查

- 现象：`libero_dino_s_fulltoken_step008000_official_4gpu` 的 rollout 目测基本失败，机械臂在抓取/摆放错误后姿态会继续漂移。该评测仍在 4-7 号卡上运行，当前进程命令对齐了 step_008000 full-token 训练设置：`task=libero_dino_s_2cam_224_1e-4`、`ckpt=runs/libero_dino_s_2cam_224_1e-4/2026-05-16_01-38-20/checkpoints/weights/step_008000.pt`、`model.dino_config.latent_spatial_pool=[1,1]`、`encode_microbatch_size=144`。
- 训练/推理链初查：DINO checkpoint 中 `mot` 同时包含 video/action experts，`proprio_encoder` 也保存；eval 日志显示 DINO-S timm 权重 keys 全部加载成功，step_008000 checkpoint 成功加载。训练 loss 和 original FastWAM 的 action flow-matching 结构一致，ActionDiT 的 denoise/infer 入口也在使用 checkpoint 覆盖后的 MoT/action weights。
- 当前更像是动作质量不够，而不是 checkpoint 没加载：8k 附近 train/eval action loss 仍在较高水平，`eval/action_l1` 约 0.09-0.12 raw action units，远高于原 FastWAM action loss 到 1e-3 量级时的状态；这种误差足以导致早期抓取/姿态偏差，并在 closed-loop rollout 里积累成漂移。
- 额外可疑点：用训练集里对应语言的 expert actions 直接在当前 LIBERO eval env/init 上回放也没有成功，说明还需要进一步核对 FastWAM 数据生成时的 LIBERO init/bddl/环境版本与当前 `/data11/wx/openpi/third_party/libero` eval 环境是否完全一致。这个问题会影响“专家轨迹 sanity check”，但官方 benchmark 仍应以当前 `.pruned_init` 为准。
- 代码改动：`src/fastwam/models/wan22/fastwam_dino.py` 的 `load_checkpoint()` 现在会在 non-strict MoT 加载时打印 missing/unexpected keys 数量和前 10 个 key，避免不同 DINO 版本/pooled 配置/小 video 配置加载错权重时静默失败。此改动不改变模型行为。

## 二十一、2026-05-17 DINO 训练/推理代码一致性审计结论

- 已对照 `/data11/wmx/FastWAM` 检查 LIBERO 训练/评测关键链路，暂未发现会直接解释“全失败”的硬 bug：动作反归一化、gripper 从 `[0,1]` 到 `[-1,1]`、`invert_gripper_action()`、`binarize_gripper`、LIBERO env action 执行顺序均与原版 FastWAM 保持一致。
- `experiments/libero/eval_libero_single.py` 相比原版的行为变化主要是：强制当前仓库 `src` 优先、EGL/headless 默认环境、按模型签名过滤 `infer_action/infer_joint` kwargs。没有改动作 postprocess 语义。
- DINO 版训练路径检查结果：
  - `FastWAM_DINO.training_loss()` 使用 DINO 编码完整视频，首帧作为 clean context，后续 DINO latent 做 video flow matching。
  - action 分支仍是原 FastWAM 的 action flow matching：随机时间步、`noise + t * (action - noise)`、mask 后 MSE，padding/mask 逻辑与原版一致。
  - MoT attention mask 仍是 video 看 video、action 看 action、action 看首帧 video；因此训练时 action 并不会偷看未来视频 latent。
- DINO 版推理路径检查结果：
  - `infer_action()` 只编码首帧，先 prefill video cache，再 denoise action。
  - 用一个小型随机 MoT 做过等价性测试：完整 MoT 前向与 `prefill_video_cache + _predict_action_noise_with_cache` 的 action 输出最大误差约 `5.96e-08`，说明 KV-cache 推理路径和训练式 full forward 在 action 可见信息上是一致的。
- checkpoint/权重检查：
  - DINO checkpoint 保存并加载 `mot`、`proprio_encoder`、`step`、`torch_dtype`。
  - `load_checkpoint()` 已增加 missing/unexpected keys warning，避免不同 DINO 版本、token pool、smallvideo/bigvideo 配置不匹配时静默加载。
  - 已确认 ActionDiT 预训练权重、DINO-L 权重、DINO-S timm 权重路径存在。
- DINO 编码配置风险：
  - 2cam 输入必须保持 `model.dino_config.input_resolution=[224,448]`，否则 base model config 可能把 224x448 双相机图像 resize 成 224x224，等价于挤压相机画面。当前 DINO task yaml 已覆盖为 `[224,448]`。
  - 老的 DINO-S full-token checkpoint 必须评测时显式覆盖 `'model.dino_config.latent_spatial_pool=[1,1]'`；当前新训练 task yaml 默认是 `[1,2]` token pool，不能混用。
  - `encode_microbatch_size` 只影响 DINO 编码分块和显存，不应改变数值语义。
- 目前更可疑的是训练目标/超参而不是训推代码 bug：
  - DINO run 的 video loss 很低，但 action loss/action L1 明显高于原 FastWAM 成功 checkpoint；接近物体但抓取失败也符合 action 精度不够导致 closed-loop rollout 误差积累的现象。
  - 下一轮建议优先做 action-focused ablation：降低或关闭 `lambda_video`、增加 action loss 权重、尝试 `eval_num_inference_steps=20/50`、对比 pooled DINO-S smallvideo 与 bigvideo、必要时短跑 action-only 版本确认 action loss 是否能降到原 FastWAM 量级。
