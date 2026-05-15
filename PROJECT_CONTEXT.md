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
