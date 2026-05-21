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
  wandb.name=libero_ddino_s_smallvideo_2cam224_pool_bs4ga4 \
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

## 二十二、2026-05-17 对比 LDA-1B 后的 DINO-FastWAM 风险判断

- 参考路径：`/data11/wmx/LDA-1B`。LDA 的核心不是“把原 video diffusion 换成 DINO latent”，而是专门为 DINO token/action token 设计的 MMDiT：
  - 当前 DINO tokens、action tokens 做 joint self-attention；
  - text/VLM tokens 分别 cross-attend 到 image/action streams；
  - timestep + task embedding 做 per-layer modulation；
  - 输出 action velocity 和 future DINO velocity。
- LDA 训练目标是四任务混训：`policy / forward_dynamics / inverse_dynamics / video_gen`。policy 样本中 future visual token 是 learnable placeholder，不是强迫 policy 每个样本都重建完整未来视频；forward/video-gen 样本才承担 DINO future 预测。
- LDA 默认配置更轻：
  - `obs_horizon=1`，`future_action_window_size=15`，action horizon 16；
  - DINO-S 224 单视图 token，含 CLS/register tokens；
  - MMDiT 配置常见为 8 层或 16 层，而不是继续使用 30 层 5B Wan-style video expert；
  - `num_inference_timesteps=4`，训练中可用 `repeated_diffusion_steps` 增加 timestep 覆盖。
- 我们当前 DINO-FastWAM 与 LDA 差异很大，可能导致“不 work/性价比差”的主因：
  1. 训练时对 33-step temporal window 做 32-step action flow matching，并对按 `action_video_freq_ratio` 子采样后的 2cam 224x448 DINO tokens 做 future video flow matching；若 full-token 且 ratio=4，则 DINO-S video token 量是 `9 * 14 * 28 = 3528` tokens/样本，仍明显重于 LDA policy 路径。
  2. action 只从首帧 video cache 取条件，而 video loss 占同等权重；优化很容易先把 DINO video loss 学低，但 action 精度仍不足。
  3. 仍沿用 Wan/FastWAM 的 30 层 video expert/MoT 框架，DINO 并没有带来结构上的轻量化。
  4. LDA 使用 CLS/register + patch tokens；我们目前默认只用 patch tokens，并把双相机横向拼接成一个 224x448 图像。这未必是 bug，但与 LDA 的 DINO token 使用方式不同。
  5. LDA 有显式 task embedding 和不同任务下的不同可见信息；我们只有同一个 joint video/action objective，缺少 inverse dynamics/forward dynamics 这种辅助约束分工。
- 直接建议：
  - 不建议继续重仓当前 “DINO full-token + 5B/big video expert + `lambda_video=lambda_action=1`”。
  - 下一轮 DINO 只做轻量验证：`smallvideo`、`latent_spatial_pool=[1,2]` 或更强 pooling、`lambda_video=0~0.1`、`lambda_action=5~10`。
  - 若要更像 LDA，应该新建一个 action-focused DINO policy 头：只编码当前 obs，future visual 用 learnable placeholder 或直接去掉 video loss，action/image tokens 用轻量 MMDiT joint attention，而不是沿用完整 FastWAM video expert。
  - 更稳的 AAAI 主线仍应保留原 FastWAM baseline；DINO 作为 ablation/扩展点，不宜替代主路线。

## 二十三、2026-05-17 关于 “33 帧 DINO future video” 和 2cam 拼接的一处澄清

- `configs/data/libero_2cam.yaml` 里的 `num_frames=33` 是原始 observation 时间窗长度；对应 action horizon 是 `num_frames - 1 = 32`。
- 训练样本里 action 分支仍预测 32 个 action step；评测时默认 `action_horizon=32`，通常每次执行 `replan_steps=10` 个 action 后重新规划。
- video/DINO 分支并不是直接吃满 33 张图。`RobotVideoDataset` 会先用 `video_sample_indices = range(0, num_frames, action_video_freq_ratio)` 对视频帧做子采样；LIBERO 2cam 配置若 `action_video_freq_ratio=4`，实际 DINO/video loss 的帧索引是 `[0,4,8,...,32]`，共 9 帧。因此更准确的说法是：33-step temporal window，32-step action flow matching，9-frame subsampled DINO future video flow matching。
- 双相机输入在训练和推理中保持一致：训练时按 `image,wrist_image` 顺序把两个 224x224 相机图沿宽度横向拼接成 224x448；推理评测的 `_obs_to_model_input()` 也按同样顺序把 agentview 和 wrist 图 resize/crop 到 224x224 后 `axis=1` 拼成 224x448，并检查 shape 是否等于 `data.train.video_size=[224,448]`。
- 2cam DINO 配置必须继续保持 `model.dino_config.input_resolution=[224,448]`，否则会把横向拼接后的双相机输入错误缩放成 224x224。

## 二十四、2026-05-17 新增离线 DINO latent cache 训练路径

- 新增 `scripts/precompute_dino_latents.py`：按训练 dataset index 读取同一套图像预处理、双相机拼接和视频子采样，然后用配置中的 DINO encoder 离线编码，保存每个样本的 `[D,T,H,W]` latent 到 `00000000.pt` 这种 index 文件。
- `RobotVideoDataset` 新增：
  - `dino_latent_cache_dir`：训练时读取离线 DINO latent；
  - `dino_latent_cache_required`：设为 true 时 cache 缺失/shape 错误会直接报错，不再随机换样本；
  - `load_text_context`：预计算 DINO 时可关掉文本 cache 读取，避免无关依赖。
- `FastWAM_DINO.build_inputs()` 现在优先使用 `sample["dino_latents"]`，缺失时回退到原来的在线 DINO 编码；因此默认配置不受影响。
- `model.dino_config.load_backbone=false` 可用于 cached 训练，完全不加载 DINO backbone，进一步省显存；推理或在线编码训练必须保持 true。
- cached 训练建议命令模式：
  - 预计算：`python scripts/precompute_dino_latents.py task=libero_dino_s_2cam_224_1e-4 dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x2 +data.train.pretrained_norm_stats=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
  - 训练：在原训练命令上追加 `data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x2 data.train.dino_latent_cache_required=true model.dino_config.load_backbone=false`。

## 二十五、2026-05-17 当前 DINO-S pooled big-video run 的判断

- 当前 run：`runs/libero_dino_s_2cam_224_1e-4/2026-05-17_03-08-16`，命令关键项为 `latent_spatial_pool=[1,2]`、`lambda_video=0.25`、`lambda_action=1.0`、`lr=5e-5`、`batch_size=4`、`gradient_accumulation_steps=4`。
- `step_008000.pt` 已保存；8200 step 的验证指标约为 `val_loss=0.0495, action_l2=0.0331, action_l1=0.0811`。
- action loss/action L1 与前一版失败 run 相比没有本质改善，video loss 已很低；继续把这个 big-video pooled run 训到很久，边际收益可能较低。
- 建议策略：保留 `step_008000.pt` 做一次快速/官方评测拿证据；随后优先切到 `fastwam_dino_s_smallvideo` + 离线 DINO latent cache，并尝试更 action-focused 的 loss（例如 `lambda_video=0~0.1`, `lambda_action=5~10`）。

## 二十六、2026-05-17 pooled big-video step8000 快速评测结论

- 当前 DINO-S pooled big-video run（`step_008000.pt`, `latent_spatial_pool=[1,2]`, `lambda_video=0.25`, `lr=5e-5`）不是完全失败：LIBERO Spatial 约 64% 成功率。
- 但仍显著低于原 FastWAM baseline（用户反馈低 30 多点），且 LIBERO-10 只从昨天的 0 成功提升到 1 个成功，说明泛化/长任务执行仍明显不够。
- 下一步不建议继续大力训练 big-video pooled 版本；应切到 `fastwam_dino_s_smallvideo`，配合离线 DINO latent cache，并把目标进一步 action-focused：优先尝试 `lambda_video=0.05`、`lambda_action=5.0`、`lr=5e-5`，观察 action loss 和快速 LIBERO Object/Spatial/10 指标是否改善。

## 二十七、2026-05-17 DINO latent 预计算加速修正

- 发现第一版 `scripts/precompute_dino_latents.py` 太慢的直接原因：按训练 window 串行 `dataset._get(idx)`，CPU 图像读取、相机拼接、resize/normalize 完全单进程执行；用户反馈全量缓存约 20 小时，不可接受。
- 已改为 DataLoader 多 worker 预取/预处理，新增顶层配置 `dino_precompute_num_workers`，默认 8；每个分布式 rank 只处理自己的 index 分片，并跳过已有 cache。
- 当前仍是 window-level cache，仍会重复编码重叠 window 中的同一底层帧；如果多 worker 后仍太慢，下一步应实现 frame-level/episode-level DINO cache，再由训练 window 组装 latent，以消除 stride=1 带来的重复编码。

## 二十八、2026-05-17 已实现 frame-level DINO cache

- 已新增 `dino_latent_cache_mode=frame`：
  - 预计算时按 LeRobot global frame index 读取单帧双相机图，拼成 224x448 后只编码一次；
  - 保存到 `CACHE_DIR/frames/00000042.pt`，每个文件是单帧 `[D,H,W]` DINO latent；
  - 训练时 `RobotVideoDataset` 根据当前 window 的 `[idx, idx+4, ..., idx+32]` 全局 frame indices 读取 frame cache，并 stack 成 `[D,T,H,W]`。
- 这样避免了 stride=1 window-level cache 对同一底层帧重复编码约多次的问题，同时 cache 体积也从 window latent 量级降到 frame latent 量级。
- 已做 data smoke test：frame-level loader 能返回 `[3,224,448]`、范围约 `[-1,1]` 的双相机拼接输入。
- 后续建议使用新的 cache 目录，例如 `./data/dino_latents_cache/libero_dino_s_2cam224_pool1x2_frame`，避免和旧 window-level cache 混用；训练时必须同时设置 `data.train.dino_latent_cache_mode=frame`。

## 二十九、2026-05-18 frame cache 失败原因与修复

- `run_cache_then_train_retry.sh` 没有进入训练，原因不是训练 OOM，而是 frame-level DINO cache 写盘失败。
- 具体失败：cache 目录 `data/dino_latents_cache/libero_dino_s_2cam224_pool1x2_frame` 膨胀到约 3.7T，随后报 `No space left on device`，脚本重试仍失败并退出；卡释放后被其他用户占用。
- 根因是 `torch.save(latents[local_i])` 保存了 batch tensor 的 storage view，每个单帧 cache 文件可能把整批 latent 的底层 storage 一起写入，导致单文件约 19MB，而不是预期的约 150KB。
- 已修复 `scripts/precompute_dino_latents.py`：保存 window/frame latent 前显式 `.clone().contiguous()`，避免 torch.save 写入整批 storage。
- 旧的 frame cache 文件已经膨胀且不可继续使用；需要清理旧 cache 目录后用修复后的脚本重建。清理属于删除操作，需用户明确确认后执行。

## 三十、2026-05-18 训前代码体检：frame cache 预处理一致性修正

- 重新检查了 DINO cache 生成、dataset 读取、cached training 和 LIBERO eval/inference 路径。
- 发现 frame-level cache 原先直接从底层 `hf_dataset` 读单帧并手写 resize/concat/normalize；shape 正确，但和正常训练 dataset 路径不是逐像素一致。实测同一帧 `mean_abs_diff≈0.01`，局部 `max_abs_diff≈0.5~0.65`。
- 已将 `scripts/precompute_dino_latents.py` 的 frame cache 单帧预处理改为复用正常训练路径里的 `FastWAMProcessor` image transforms，再做相同的相机拼接、resize/crop/normalize，确保 cache 里的 DINO 输入与在线训练 DINO 输入完全同源，同时避免为每个缓存帧加载完整 33-step window。
- 修正后 smoke test：sample 0 的 9 个视频采样帧经 frame-cache path 与正常训练 path 的 `max_abs_diff` 和 `mean_abs_diff` 全部为 0。
- 修正 `logs/run_framecache_then_train_retry_fixed.sh` 的等卡环境变量名：`WAIT_GPU_MAX_USED_MB / WAIT_GPU_MAX_UTIL / WAIT_GPU_INTERVAL`，与 `scripts/wait_for_gpus.py` 实际读取的名字一致。
- `py_compile` 已通过：`scripts/precompute_dino_latents.py`、`robot_video_dataset.py`、`fastwam_dino.py`、`eval_libero_single.py`。
- 当前 cached training 关键对齐项：`data.train.dino_latent_cache_mode=frame`、`data.train.dino_latent_cache_required=true`、`model.dino_config.load_backbone=false`、`model.dino_config.input_resolution=[224,448]`、`model.dino_config.latent_spatial_pool=[1,2]`。
- eval/inference 不应带 `model.dino_config.load_backbone=false`；LIBERO eval 通过 `configs/sim_libero.yaml` 设置 `model.load_text_encoder=true`，并按训练一致的 224x448 双相机横拼输入在线编码首帧 DINO。
- 发现旧 cache 目录 `data/dino_latents_cache/libero_dino_s_2cam224_pool1x2_frame` 中已有约 28G 文件，属于预处理一致性修正前生成的旧 cache；不能继续复用，否则新旧预处理会混合。
- `logs/run_framecache_then_train_retry_fixed.sh` 已改用新目录 `data/dino_latents_cache/libero_dino_s_2cam224_pool1x2_frame_exact`，避免删除旧文件且避免混用旧 cache。
- 训练阶段第一次进入 `bs=4, ga=4` 时失败原因是 `scripts/train_zero1.sh` 直接调用 `accelerate launch`，但 tmux/shell PATH 中没有 `accelerate` 命令；不是 OOM。
- 已修复 `scripts/train_zero1.sh`：改为 `${PYTHON:-python3} -m accelerate.commands.launch`，和脚本指定的 fastwam Python 环境绑定，不再依赖 conda activate/PATH。
- 当前自动 retry 已进入 `bs=2, ga=8` 并成功 launch：run 目录 `runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-18_14-50-08`，wandb run id `l5n3fbxn`，name `libero_dino_s_smallvideo_framecache_FIXED_lv0.05_la5_bs2ga8`。
- 观察到 cached smallvideo 训练显存约 29~31G/卡，`bs=2, ga=8` 速度约 `0.18 step/s, 1.44 samples/s`，明显偏保守。
- 已将 `logs/run_framecache_then_train_retry_fixed.sh` 的训练 retry 顺序改为保持全局 batch=64 的大 micro-batch 优先：`bs=16,ga=1 -> bs=8,ga=2 -> bs=4,ga=4 -> bs=2,ga=8 -> bs=1,ga=16`。
- 后续观察：`bs=16,ga=1` 会真实 OOM，不应再优先尝试；`bs=8,ga=2` 训练本身能跑，但在 step 200 内部 eval 时失败。
- `bs=8,ga=2` 失败根因不是 OOM，而是 `Wan22Trainer._to_batched_eval_sample()` 漏传了 `dino_latents` 和 pad masks，导致 cached training 的 eval batch 回退在线 DINO；此时配置里 `model.dino_config.load_backbone=false`，于是报 `RuntimeError: Backbone not loaded.`
- 已修复 `src/fastwam/trainer.py`：eval sample 现在会保留 `dino_latents`、`action_is_pad`、`image_is_pad`；smoke test 确认正常 training dataset 的样本含 `dino_latents (384,9,14,14)`，batched eval 后为 `(1,384,9,14,14)`。
- 已将 `logs/run_framecache_then_train_retry_fixed.sh` 的训练 retry 顺序改成 `bs=8,ga=2 -> bs=4,ga=4 -> bs=2,ga=8 -> bs=1,ga=16`，跳过已知 OOM 的 `bs=16,ga=1`。
- 当前如果已有旧进程在跑，它不会自动加载这次 trainer.py 修复；建议尽快重启脚本，以便从 `bs=8,ga=2` 重新开始并避免 step 200 eval 再炸。

## 三十一、2026-05-18 DINO 训推一致性再审计

- 参考 `/data11/wmx/FastWAM` 原始实现重新检查了 DINO 版的训练、内部 eval、LIBERO eval、frame cache、训练脚本启动路径。
- 主链路结论：DINO 版训练时用完整 9 帧 DINO latent 做 video flow matching，action 分支通过 MoT attention mask 只能看 first-frame video tokens；推理时在线 DINO 编码当前首帧，video expert prefill 一次 KV cache，再让 action expert denoise。这个设计与原 FastWAM 的 VAE first-frame cache 推理方式一致。
- 训练/推理 2cam 输入保持一致：训练与 frame cache 均为 `image,wrist_image` 先各自 224x224，再横拼 224x448；LIBERO eval 的 `_obs_to_model_input()` 也是同顺序、同尺寸横拼；`model.dino_config.input_resolution=[224,448]`、`latent_spatial_pool=[1,2]` 对应 cache shape `[384,9,14,14]`。
- 已修复一个潜在 cache 错配：`RobotVideoDataset._get()` 现在用 BaseLerobotDataset 实际返回的 `sample["idx"]` 作为 `dataset_idx` 和 DINO cache index。这样即使底层读取 retry 或 padding retry 换样本，也不会拿请求 idx 去读另一个样本的 cache。
- 已给 `FastWAM_DINO.infer_action()` 补齐原 FastWAM 同款检查：推理只允许 `video_attention_mask_mode="first_frame_causal"`，避免误用其它 video mask 时 action 推理语义不成立。
- 验证项已过：
  - `py_compile`：`robot_video_dataset.py`、`fastwam_dino.py`、`trainer.py`、`precompute_dino_latents.py`、`eval_libero_single.py`、`scripts/train.py`；
  - Hydra compose 当前 smallvideo + frame cache 训练覆盖项通过；
  - dataset/cache smoke：`video=(3,9,224,448)`、`action=(32,7)`、`proprio=(32,8)`、`dino_latents=(384,9,14,14)`；
  - frame-cache 预处理一致性 smoke：sample 0 的 9 个采样帧 cache path vs normal training path `max_diffs` 全部为 0。
- 当前正在跑的训练进程不会自动加载本段新补丁；但这次补丁主要是边缘 cache index 防护和推理配置早报错，当前 `skip_padding_as_possible=false` 下不影响已跑主流程。下次启动会自动生效。

## 三十二、2026-05-18 cached training 内部 eval 再修复

- `bs=8,ga=2` 和自动 retry 的 `bs=4,ga=4` 都在 step 200 内部 eval 报 `RuntimeError: Backbone not loaded. Call load_backbone() first.`。
- 根因：前一次修复只让 eval 的 `training_loss()` 能吃到 cached `dino_latents`；但 trainer 内部 eval 随后仍调用 `model.infer_action()` 计算 `action_l1/action_l2`，而 `infer_action()` 必须在线 DINO 编码首帧。cached training 配置里 `model.dino_config.load_backbone=false`，所以这里必炸。
- 已修复 `src/fastwam/trainer.py`：当模型有 `infer_action` 且 `dino_encoder._loaded=false` 时，内部 eval 只汇总/记录 cached `val_loss`，跳过需要在线 DINO backbone 的 action inference metrics。正式 LIBERO eval 不受影响，正式 eval 仍应加载 DINO backbone。
- 已清理误启动的 `bs=2,ga=8` 孤儿训练进程，重新在 tmux session `dino_framecache_train_restart_155252` 启动 `bs=8,ga=2`：
  - run dir：`runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-18_15-52-52`
  - log：`logs/train_libero_dino_s_smallvideo_framecache_FIXED_RESTART_bs8ga2_20260518_155252.log`
  - wandb name：`libero_dino_s_smallvideo_framecache_FIXED_RESTART_lv0.05_la5_bs8ga2`
- 验证：新 run 已通过 step 200 内部 eval，日志显示 `step=200 val_loss=1.3222`，随后继续到 step 210；没有再出现 Backbone traceback。当前速度约 `0.25 step/s, 8.04 samples/s`，4-7 卡显存约 55-56G。

## 三十三、2026-05-18 关于 DINO adapter 方向

- mentor 提出的关键判断：原 FastWAM 的效果很大程度来自复用 WAN 预训练 video DiT 的基座能力；如果 DINO 版只是保留 WAN-like 参数规模但随机初始化 video backbone，就等于用 LIBERO 这点数据从零训练一个基座级 DiT，成功概率很低。
- adapter 是后续值得优先尝试的方向：尽量冻结或半冻结 WAN 预训练 video blocks，只训练 DINO feature 到 WAN token/hidden space 的输入 adapter、输出 head/adapter，以及 MoT/action 相关少量参数。
- 目标不是单纯把 DINO feature 塞进一个随机大 DiT，而是最大化“蹭” WAN 已有时空生成先验，同时让 DINO 表征通过轻量 adapter 进入这个先验空间。

## 三十四、2026-05-18 DINO big/small/cache/推理代码审查补充

- 当前 smallvideo frame-cache 训练仍在跑，未打断；日志已到 3200+ step，4-7 卡显存约 55-56G/卡，速度约 `0.25 step/s`。
- Hydra compose 检查确认：
  - `libero_dino_2cam_224_1e-4`：DINO-L，`feature_dim=1024`，big video DiT `hidden_dim=3072, ffn_dim=14336`，`pool=[1,2]`，默认 `video_dit_init_from_wan=false`；
  - `libero_dino_s_2cam_224_1e-4`：DINO-S，`feature_dim=384`，big video DiT `hidden_dim=3072, ffn_dim=14336`，`pool=[1,2]`；
  - `libero_dino_s_smallvideo_2cam_224_1e-4`：DINO-S，small video DiT `hidden_dim=1024, ffn_dim=4096`，`pool=[1,2]`。
- `scripts/train.py` 和 `experiments/libero/eval_libero_single.py` 都会把本仓库 `src` 加到 `sys.path` 前面；即使用 fastwam conda，也优先跑当前仓库代码。
- 训推一致性复查：
  - cached/online training 都进入同一个 `FastWAM_DINO.build_inputs()`，区别只是 `dino_latents` 是否来自 cache；
  - 训练 action 分支通过 MoT mask 只能看 first-frame video tokens；
  - 推理时 `infer_action()` 在线编码当前观测首帧 DINO，video expert prefill 一次 KV cache，再 denoise action，语义与训练 action 可见信息一致；
  - LIBERO eval 输入仍是 `image + wrist_image` 横拼成 224x448，和训练/frame cache 的双相机横拼一致。
- cache 对齐复查：
  - `RobotVideoDataset._get()` 使用实际返回的 `sample["idx"]` 作为 `dataset_idx` 和 cache key；
  - frame cache 训练时按同一 episode 内 `[idx, idx+4, ..., idx+32]` 组装 9 帧 latent；
  - cache shape 会在模型里校验 D、H、W 是否与当前 DINO variant、resolution、pool 参数一致，避免 pool/版本混用。
- 本次新增一个 window-cache 防护：`scripts/precompute_dino_latents.py` 的 `_IndexedVideoDataset` 现在也使用样本里的真实 `dataset_idx` 写 cache，防止 window-level 预计算在 dataset retry 时把 latent 写到请求 idx。
- 本次新增一个 WAN init 防误用：`model.video_dit_init_from_wan=true` 现在要求 video DiT 保持 Wan2.2-5B 形状（`3072/14336/24 heads/128 head dim/30 layers`）；smallvideo 若误开该选项会提前报错。smallvideo 当前默认 false 是正确的，后续要蹭 WAN 需要显式 adapter 路线，而不是直接开这个 flag。
- 验证：`py_compile` 已通过 `fastwam_dino.py`、`precompute_dino_latents.py`、`scripts/train.py`、`eval_libero_single.py`；smallvideo + `video_dit_init_from_wan=true` 的错误路径 smoke test 能提前给出明确 ValueError。

## 三十五、2026-05-19 训练 resume 机制检查

- 训练 checkpoint 有两套：
  - `checkpoints/weights/step_xxxxxx.pt`：只保存模型权重和 step 字段，用于 eval 或 weight-only finetune；
  - `checkpoints/state/step_xxxxxx/`：Accelerate/DeepSpeed 完整训练状态，包含 ZeRO optimizer shard、scheduler、random states、`trainer_state.json`。
- 对当前 smallvideo run，完整 state 已存在，例如：
  - `runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-18_15-52-52/checkpoints/state/step_026000`
  - 其中 `trainer_state.json` 为 `global_step=26000, epoch=5, batch_in_epoch=8600`。
- 重要：如果要真正继续训练，应传 `resume=.../checkpoints/state/step_xxxxxx` 目录；传 `resume=.../weights/step_xxxxxx.pt` 只会加载权重，不会恢复 optimizer/scheduler/global_step，代码会明确 warning：`Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.`
- 已修复 dataloader resume 的一个小问题：
  - `load_training_state()` 现在会把 sampler epoch 设置成保存的 `epoch`；
  - 训练每次进入新 epoch 时也会更新 sampler epoch；
  - `ResumableEpochSampler` 的 batch offset skip 不再只限制在 `epoch==0`，因此从 `batch_in_epoch` 恢复时能正确跳过已消费 batch。
- `py_compile` 已通过：`trainer.py`、`samplers.py`。
- 完整 state resume 建议保持同样的 GPU 数/DeepSpeed 配置/模型配置/`batch_size`/`gradient_accumulation_steps`，尤其 ZeRO optimizer shards 通常依赖相同 world size；如果要改 GPU 数或 batch 设置，应使用 `.pt` 权重做 weight-only 启动，但这不等价于精确 resume。

## 三十六、2026-05-20 LIBERO eval 并发与 CPU 线程限制

- mentor 建议有道理：LIBERO/MuJoCo eval 是多进程仿真 + GPU policy inference 混合负载；默认 OpenMP/MKL/OpenBLAS 线程过多时，多个 eval worker 会抢 CPU，反而拖慢仿真。
- 已新增 eval worker 线程限制配置：
  - `MULTIRUN.omp_num_threads: 2`
  - `MULTIRUN.mkl_num_threads: 1`
  - `MULTIRUN.openblas_num_threads: 1`
  - `MULTIRUN.numexpr_num_threads: 1`
- `experiments/libero/run_libero_manager.py` 会把这些值写入 worker 环境变量；`experiments/libero/run_libero_parallel_test.sh` 在 tmux worker 里也会重新 export，防止 `source ~/.bashrc` 后丢失。
- `bash -n run_libero_parallel_test.sh`、`py_compile run_libero_manager.py` 已通过；Hydra compose 测试 `MULTIRUN.max_tasks_per_gpu=4` 正常。
- 当前 step 38k checkpoint 路径：
  - `runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-19_21-30-18/checkpoints/weights/step_038000.pt`
- 对 80G A800/A100，DINO-S smallvideo eval 可以尝试 `MULTIRUN.max_tasks_per_gpu=4`，即 4 卡同时最多 16 个 task worker；若出现 GPU OOM、CPU load 过高或 MuJoCo 不稳定，再降到 3 或 2。

## 三十七、2026-05-20 step 043400 eval 报错原因

- tmux session `dino` 中 30 trials eval 全部 worker 快速失败，不是 OOM，也不是 LIBERO 环境错误。
- 根因是 checkpoint 路径写错：命令使用了旧 run dir：
  - 错误：`runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-19_21-30-18/checkpoints/weights/step_043400.pt`
  - 该目录实际只到 `step_038000.pt`。
- resume 后新建了 run dir，最终权重实际在：
  - `runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-20_11-53-25/checkpoints/weights/step_043400.pt`
- 以后 eval 最终权重时优先 `find runs/libero_dino_s_smallvideo_2cam_224_1e-4 -path '*/checkpoints/weights/*.pt' | sort` 确认路径，避免旧 run dir/新 run dir 混用。

## 三十八、2026-05-20 smallvideo step 043400 30-trial eval 结果

- 使用正确最终权重：
  - `runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-20_11-53-25/checkpoints/weights/step_043400.pt`
- 评测目录：
  - `evaluate_results/libero_dino_s_smallvideo_framecache_step_043400_official_4gpu_30trials_fast_retry`
- 30 trials/task 结果：
  - `libero_spatial=95.67`
  - `libero_object=99.00`
  - `libero_goal=94.67`
  - `libero_10=82.00`
  - `Overall=92.83`
- 对比 step 038000 的 10 trials/task：
  - `libero_spatial=98.00`
  - `libero_object=100.00`
  - `libero_goal=93.00`
  - `libero_10=81.00`
  - `Overall=93.00`
- 结论：step 038000 到 step 043400 没有明显涨点；libero_goal/libero_10 小幅上升，spatial/object 小幅下降，overall 基本持平。考虑 trial 数不同，差异大概率在评测波动范围内。
- task-level 观察：
  - spatial/object 基本接近饱和，主要掉点来自少数 harder placement/grasp task：`libero_spatial_5=80%`、`libero_spatial_4=90%`、`libero_object_9=93.33%`；
  - goal 的弱项是抽屉/架子相关：`libero_goal_0=76.67%`、`libero_goal_9=80%`；
  - 真正限制 overall 的是 LIBERO-10：`libero_10_9=40%`、`libero_10_8=53.33%`、`libero_10_0=66.67%`、`libero_10_6=76.67%`；
  - 从 38k 到 43.4k，LIBERO-10 的两个原本最弱项有提升：`task8 30%->53.33%`、`task9 30%->40%`，但 `task0 90%->66.67%`、`task6 90%->76.67%` 回落，抵消了收益。

## 三十九、2026-05-20 smallvideo 40k/42k/43.4k 30-trial 对比

- 30 trials/task full eval：
  - `step_040000`: spatial `96.00`, object `98.33`, goal `93.33`, libero_10 `85.33`, overall `93.25`
  - `step_042000`: spatial `96.67`, object `99.33`, goal `95.33`, libero_10 `83.00`, overall `93.58`
  - `step_043400`: spatial `95.67`, object `99.00`, goal `94.67`, libero_10 `82.00`, overall `92.83`
- 当前 30-trial 最优：
  - 按 overall：`step_042000` 最好，`93.58%`
  - 按 LIBERO-10：`step_040000` 最好，`85.33%`
- 关键 task 变化：
  - `libero_10_0`: `80.00 -> 70.00 -> 66.67`，持续下降；
  - `libero_10_6`: `96.67 -> 80.00 -> 76.67`，持续下降；
  - `libero_10_9`: `53.33 -> 50.00 -> 40.00`，持续下降；
  - `libero_10_2`: `83.33 -> 93.33 -> 96.67`，持续上升；
  - `libero_goal_0`: `56.67 -> 83.33 -> 76.67`，42k 最好；
  - `libero_goal_9`: `90.00 -> 80.00 -> 80.00`，40k 最好。
- 结论：训练后期不是单调提升，而是在任务之间换分。若论文/汇报使用单模型 overall，优先报告 `step_042000`；若优先 LIBERO-10，`step_040000` 更强。

## 四十、2026-05-20 DINO PCA 可视化脚本

- 新增 `scripts/visualize_dino_pca.py`，用于对比模型 rollout 出来的 predicted DINO latent 和真实 DINO latent。
- 脚本行为：
  - 加载指定 `ckpt`；
  - 从训练集读取样本和 cached `dino_latents`；
  - 用真实首帧 DINO latent 作为条件，从 noise 经过 video scheduler denoise 出 predicted DINO clip；
  - 将 GT/pred DINO tokens 放在同一个 PCA basis 下投成 RGB；
  - 输出每个样本的 `RGB frame / GT DINO PCA / Pred DINO PCA / RMSE heat` 拼图；
  - 另存 LDA-style PCA 图：GT 和 Pred 各自按单帧独立 PCA、独立 min-max、双线性放大，更接近 `/data11/wmx/LDA-1B/eval/video_gen.py` 的 `visualize_dino()` 展示方式，肉眼更容易看空间结构，但颜色不再可严格跨图对齐；
  - 同时保存 `.npz`，包含 `gt`、`pred`、`mse_per_frame`，便于后续量化分析。
- 默认不加载 DINO backbone，依赖 frame/window cache，因此运行命令需要传：
  - `data.train.dino_latent_cache_dir=...`
  - `data.train.dino_latent_cache_mode=frame`
  - `data.train.dino_latent_cache_required=true`
  - `model.dino_config.load_backbone=false`
- 已通过 `python3 -m py_compile scripts/visualize_dino_pca.py`。
- 已用 fastwam conda smoke test `_fit_lda_style_pca_rgb()` 和共享 PCA 输出 shape/dtype 正常。

## 四十一、2026-05-20 step 042000 DINO PCA 可视化观察

- 可视化目录：`outputs/dino_pca/step042000_lda_style`
- LDA-style PCA 图的主要结论：
  - predicted DINO 不是完全坍缩，能保留桌面、柜体、盘子/碗等大块空间结构；
  - 但预测明显更平滑、更粗，末端执行器、小物体边界、抓取/接触附近细节不清楚；
  - 对动态大的样本，predicted future 的运动幅度偏小，表现为“知道大概区域，但跟不上真实交互变化”；
  - 对近静态样本，pred 反而可能有额外漂移/幻觉变化。
- 数值侧：
  - future frame cosine 大多在 `0.89~0.96`，说明 latent 大方向相似；
  - RMSE 随未来帧增大，动态样本末帧约 `0.15~0.16`；
  - predicted temporal delta 通常小于 GT temporal delta，例如 sample0 `0.076 vs 0.105`、sample10000 `0.080 vs 0.128`；
  - predicted spatial sharpness 是 GT 的约 `0.84~0.90`，定量支持“更平滑/更糊”的观察。
- 解释：这和 LIBERO-10 的瓶颈吻合。模型学到了语义/大布局，但对多阶段任务中小物体、末端姿态、接触和精细放置的 latent dynamics 不够准。单纯 DINO future MSE 低不代表足够支撑精细控制。
- 注意：当前 `latent_spatial_pool=[1,2]` 后只有 `14x14` token grid，天然比 LDA 例子里的 `40x30` 展示更粗；PCA 图只能看结构和坍缩趋势，不应期待像 RGB 一样直观。

## 四十二、2026-05-21 lambda_video=1.0 续训状态

- 当前续训 run：`runs/libero_dino_s_smallvideo_2cam_224_1e-4/2026-05-20_20-28-53`
- 续训设置：从 `step_040000` resume，`lambda_video=1.0`、`lambda_action=5.0`、`lr=1e-5`、`num_epochs=15`、frame cache、`batch_size=8`、`gradient_accumulation_steps=2`。
- 2026-05-21 11:15 左右状态：
  - 训练仍在运行，约 `epoch=12 step=53400/65100`；
  - 最新已保存权重：`checkpoints/weights/step_052000.pt`；
  - 近期训练速度约 `0.25 step/s, 8.08 samples/s`；
  - 近期 loss 大致：`loss_action=0.07~0.20`、`loss_video=0.02~0.03`；
  - `step=53400 val_loss=0.0948`。
- 建议：可以先评测 `step_052000.pt` 判断 `lambda_video=1.0` 是否带来收益；不要抢 4-7 上正在跑的训练，优先用 wait-for-gpus 在 0-3 或训练结束后的 4-7 排队评测。

## 四十三、2026-05-21 step 052000 lambda_video=1.0 评测结果

- 评测目录：`evaluate_results/libero_dino_s_smallvideo_lv1_step052000_official_4gpu_30trials_wait`
- 权重：`step_052000.pt`，续训设置为 `lambda_video=1.0`、`lambda_action=5.0`、`lr=1e-5`。
- 30 trials 官方四套结果：
  - spatial `98.33`
  - object `99.00`
  - goal `92.33`
  - libero_10 `84.33`
  - overall `93.50`
- 对比之前：
  - `step_040000`: overall `93.25`，LIBERO-10 `85.33`；
  - `step_042000`: overall `93.58`，LIBERO-10 `83.00`；
  - `step_043400`: overall `92.83`，LIBERO-10 `82.00`。
- 结论：`lambda_video=1.0` 续训到 52k 没有带来整体突破，overall 接近 42k 但略低；spatial 明显更好，LIBERO-10 比 42k/43.4k 回升但仍低于 40k；goal 掉点较明显。
- 分任务变化：涨点主要在 `libero_spatial_5`、`libero_10_8`、`libero_10_0`；掉点主要在 `libero_10_6`、`libero_goal_6`、`libero_10_2`、`libero_10_9`、`libero_goal_0`。
- DINO spatial pooling 实现：`latent_spatial_pool=[1,2]` 是对 `[B,D,T,H,W]` latent 用 `avg_pool3d(kernel=(1,1,2), stride=(1,1,2))`，即只沿宽度方向每相邻两个 patch token 平均；224x448 输入、patch16 时从 `14x28=392` tokens/frame 变成 `14x14=196` tokens/frame。
- 风险：这种 pooling 会丢掉横向细节和小物体/夹爪接触附近的精定位信息；双相机横向拼接时不会跨相机边界平均，但每个相机内部宽度从 14 列压成 7 列。

## 四十四、2026-05-21 no-pooling DINO frame cache

- 已生成 `latent_spatial_pool=[1,1]` 的 DINO-S frame-level cache：
  - 目录：`data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact`
  - 日志：`logs/precompute_dino_latents_pool1x1_frame_20260521_125905.log`
- 运行设置：
  - `task=libero_dino_s_smallvideo_2cam_224_1e-4`
  - `dino_latent_cache_mode=frame`
  - `model.dino_config.latent_spatial_pool=[1,1]`
  - 4 卡：`CUDA_VISIBLE_DEVICES=4,5,6,7`
  - `dino_precompute_batch_size=32`
  - `dino_precompute_num_workers=24`
- 校验结果：
  - metadata `total_samples=277713`
  - 实际 frame cache 文件数 `277713`
  - 抽样文件 shape 均为 `(384, 14, 28)`
  - dtype 为 `torch.bfloat16`
  - finite 检查通过
- 训练 no-pooling 模型时要保持：
  - `data.train.dino_latent_cache_dir=./data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact`
  - `data.train.dino_latent_cache_mode=frame`
  - `data.train.dino_latent_cache_required=true`
  - `model.dino_config.load_backbone=false`
  - `model.dino_config.latent_spatial_pool=[1,1]`

## 四十五、2026-05-21 当前评测结论与下一步

### 评测结果

当前最有参考价值的是 smallvideo + DINO-S + frame cache 这一组：

| checkpoint | spatial | object | goal | libero_10 | overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 040000 | 96.00 | 98.33 | 93.33 | 85.33 | 93.25 |
| step 042000 | 96.67 | 99.33 | 95.33 | 83.00 | 93.58 |
| step 043400 | 95.67 | 99.00 | 94.67 | 82.00 | 92.83 |
| step 052000 (`lambda_video=1.0`) | 98.33 | 99.00 | 92.33 | 84.33 | 93.50 |

- `step_042000` 是当前 overall 最好：`93.58`。
- `step_040000` 是当前 LIBERO-10 最好：`85.33`。
- `step_052000` 的 `lambda_video=1.0` 续训没有带来整体突破；spatial 变强，但 goal 掉点，LIBERO-10 仍低于 40k。
- 模型不是完全不 work：`spatial/object/goal` 均能到 90%+，说明训练、推理、坐标系、DINO cache 对齐等主逻辑基本是通的。
- 主要短板是 LIBERO-10，卡在 `82~85%`，和原 FastWAM 官方表现仍有明显差距。

### 当前判断

- DINO 表示是可用的，但当前 video future prediction 对精细交互不够强。
- PCA 可视化支持这个判断：
  - predicted DINO 没有坍缩；
  - 大布局和语义结构能对上；
  - 但小物体、夹爪接触、放置边界更平滑、更糊；
  - 动态较大的样本里 predicted future 的运动幅度偏小。
- 这和 LIBERO-10 瓶颈吻合：LIBERO-10 更依赖多阶段、接触、放置和细粒度状态。
- `latent_spatial_pool=[1,2]` 也是可疑因素：
  - 原始 DINO grid 为 `14x28=392 tokens/frame`；
  - pooling 后变成 `14x14=196 tokens/frame`；
  - 虽然不会跨双相机边界平均，但每个相机内部横向分辨率从 14 列变 7 列；
  - 这可能损失小物体/夹爪精定位信息。

### 下一步建议

1. 优先训练 no-pooling 版本：`latent_spatial_pool=[1,1]`
   - 这是最直接的 ablation，可以回答“spatial pooling 是否损害效果”。
   - no-pooling frame cache 已经生成好：`data/dino_latents_cache/libero_dino_s_2cam224_pool1x1_frame_exact`。
   - 代价是 token 数翻倍，训练更慢、更占显存。

2. 尝试更小/更合理的 video DiT 或 adapter
   - mentor 的判断有道理：原 FastWAM 蹭了 Wan 预训练视频生成能力；当前 DINO 版如果用较大 DiT 且随机初始化，相当于用 LIBERO 小数据训练一个基座级 video DiT，数据量可能不够。
   - adapter 或更小 backbone 可能更适合当前数据规模。

3. 对 action 分支做加强，而不是继续只调 video loss
   - `lambda_video=1.0` 没有明显突破，说明简单加大 video loss 权重不是核心解。
   - 可以考虑 action-only warmup、action loss schedule、或更稳定的 action 采样/监督。

4. 对 LIBERO-10 失败任务做 targeted 分析
   - 当前拖后腿的任务主要包括 `libero_10_9`、`libero_10_8`、`libero_10_6` 等多物体/开关/放置类任务。
   - rollout 视频和失败阶段分析比继续盯平均 loss 更有信息量。

5. 暂时不要继续无脑训当前 `[1,2] + lambda_video=1.0`
   - 从 40k 到 52k 已经说明没有稳定上升趋势。
   - 除非只是补曲线点，否则性价比不高。

一句话总结：当前方案已经证明 “DINO future + action” 能跑通并达到约 `93%` overall，但还没追上 FastWAM。下一步最值得尝试的是 no-pooling `[1,1]`，其次是 adapter/更小模型，把问题从“多训一点”转成“表示和模型容量是否匹配”。
