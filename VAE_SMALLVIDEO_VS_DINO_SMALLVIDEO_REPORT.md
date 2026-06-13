# VAE SmallVideo vs DINO SmallVideo LIBERO 最终结论报告

生成时间：2026-06-13

## 1. 最终结论

在当前 FastWAM / SpiderWAM 的 LIBERO 30-trial 评测里，**1B 级 DINO token route 明确强于 1B 级 VAE latent route**。这个结论在两组 VAE 对照下都成立：

1. VAE 原始 loss 权重：`lambda_video=1.0, lambda_action=1.0`
2. VAE 对齐 DINO loss 权重：`lambda_video=0.05, lambda_action=5.0`

最关键的结果如下：

| 路线 | checkpoint | loss 权重 | global bs | 等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| DINO no-pool + lr1e-5 weight-only | `step_028930` | `0.05 / 5.0` | 96 | 20.00 | 98.00 | 98.33 | 98.67 | 90.00 | 96.25 |
| DINO avgpool + lr1e-5 weight-only from `043400` 10ep base | `step_018000` | `0.05 / 5.0` | 128 | 18.29 | 95.33 | 97.33 | 98.00 | 90.00 | 95.17 |
| DINO no-pool fresh | `step_028930` | `0.05 / 5.0` | 96 | 10.00 | 98.67 | 99.33 | 92.33 | 84.00 | 93.58 |
| VAE smallvideo baseline | `step_021700` | `1.0 / 1.0` | 256 | 20.00 | 92.00 | 98.00 | 94.00 | 81.33 | 91.33 |
| VAE smallvideo baseline, lr1e-5 extra | `step_002000` | `1.0 / 1.0` | 256 | 21.84 | 89.67 | 97.67 | 95.00 | 81.67 | 91.00 |
| VAE loss-aligned, best | `step_054000` | `0.05 / 5.0` | 96 | 18.67 | 70.67 | 96.33 | 82.67 | 54.67 | 76.08 |
| VAE loss-aligned, final 20ep | `step_057860` | `0.05 / 5.0` | 96 | 20.00 | 71.67 | 94.67 | 83.33 | 53.67 | 75.83 |

一句话判断：**VAE 路线不是被 loss 权重拖累；把 loss 权重对齐到 DINO 后反而大幅变差。当前证据更支持“DINO 语义 token 表征本身更适合 LIBERO action prediction”。**

## 2. 主要对比

### 2.1 DINO no-pool vs VAE baseline

| Metric | DINO no-pool 20ep | VAE baseline 20ep | DINO - VAE |
|---|---:|---:|---:|
| Spatial | 98.00 | 92.00 | +6.00 |
| Object | 98.33 | 98.00 | +0.33 |
| Goal | 98.67 | 94.00 | +4.67 |
| LIBERO-10 | 90.00 | 81.33 | +8.67 |
| Overall | 96.25 | 91.33 | +4.92 |

VAE baseline 并不是完全不行，Object 很高，Goal 也不低；真正差距在 Spatial 和 LIBERO-10，说明它更容易在精细定位和长程闭环上掉分。

### 2.2 DINO no-pool vs VAE loss-aligned

| Metric | DINO no-pool 20ep | VAE loss-aligned 20ep | DINO - VAE aligned |
|---|---:|---:|---:|
| Spatial | 98.00 | 71.67 | +26.33 |
| Object | 98.33 | 94.67 | +3.66 |
| Goal | 98.67 | 83.33 | +15.34 |
| LIBERO-10 | 90.00 | 53.67 | +36.33 |
| Overall | 96.25 | 75.83 | +20.42 |

这个结果很关键：原来我们担心 DINO 高分可能只是因为 `lambda_action=5.0` 更偏 action。现在 VAE 也对齐到同样权重后，性能没有追上，反而从 baseline 的 `91.33` 掉到 `75.83`。所以这条对照基本排除了“只是 loss 权重导致 DINO 好”的解释。

## 3. VAE baseline 曲线

VAE baseline 使用 Wan VAE latent，small VideoDiT 从 `WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt` 初始化，loss 权重为 `lambda_video=1.0, lambda_action=1.0`。

| run | checkpoint | global bs | 等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh | `step_012000` | 256 | 11.06 | 92.00 | 97.67 | 88.67 | 78.33 | 89.17 |
| fresh | `step_014000` | 256 | 12.90 | 87.67 | 98.00 | 92.33 | 76.67 | 88.67 |
| full resume | `step_018000` | 256 | 16.59 | 90.00 | 97.67 | 94.67 | 76.67 | 89.75 |
| full resume | `step_020000` | 256 | 18.43 | 91.33 | 97.00 | 94.00 | 78.33 | 90.17 |
| full resume | `step_021700` | 256 | 20.00 | 92.00 | 98.00 | 94.00 | 81.33 | 91.33 |
| weight-only lr1e-5 | `step_002000` | 256 | 21.84 | 89.67 | 97.67 | 95.00 | 81.67 | 91.00 |
| weight-only lr1e-5 | `step_004000` | 256 | 23.69 | 89.67 | 97.67 | 93.00 | 79.67 | 90.00 |
| weight-only lr1e-5 | `step_005425` | 256 | 25.00 | 89.67 | 97.00 | 93.33 | 78.67 | 89.67 |

结论：

- VAE baseline 从 12-20ep 有增长：`88.67 -> 91.33`。
- 20ep 后继续 lr1e-5 weight-only 没有收益：`91.33 -> 91.00 -> 90.00 -> 89.67`。
- VAE baseline 的上限目前明显低于 DINO no-pool 20ep 的 `96.25`。

## 4. VAE loss-aligned 曲线

VAE loss-aligned 使用同样 VAE latent / small VideoDiT，但把 loss 权重对齐到 DINO：`lambda_video=0.05, lambda_action=5.0`。这套训练 global batch 为 `24 x 4 x 1 = 96`，`step_057860` 对应约 20ep。

| checkpoint | global bs | 等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `step_026000` | 96 | 8.99 | 70.67 | 87.33 | 69.00 | 40.67 | 66.92 |
| `step_030000` | 96 | 10.37 | 58.67 | 79.33 | 78.00 | 43.00 | 64.75 |
| `step_040000` | 96 | 13.83 | 72.33 | 94.00 | 77.33 | 53.00 | 74.17 |
| `step_048000` | 96 | 16.59 | 65.67 | 87.33 | 83.33 | 47.33 | 70.92 |
| `step_050000` | 96 | 17.28 | 69.33 | 91.00 | 81.67 | 55.00 | 74.25 |
| `step_052000` | 96 | 17.97 | 72.33 | 92.67 | 81.67 | 53.33 | 75.00 |
| `step_054000` | 96 | 18.67 | 70.67 | 96.33 | 82.67 | 54.67 | 76.08 |
| `step_056000` | 96 | 19.36 | 71.67 | 93.00 | 82.33 | 53.67 | 75.17 |
| `step_057860` | 96 | 20.00 | 71.67 | 94.67 | 83.33 | 53.67 | 75.83 |

结论：

- best 是 `step_054000`，`76.08 Overall / 54.67 LIBERO-10`。
- final 20ep 是 `75.83 Overall / 53.67 LIBERO-10`。
- 它比 VAE baseline 20ep 低 `15.50 Overall`，比 DINO no-pool 20ep 低 `20.42 Overall`。
- 这说明 VAE latent 在当前结构下不能简单复用 DINO 的 loss 权重。更高 action weight 没有让它更像控制模型，反而破坏了原先 VAE latent dynamics 的训练平衡。

## 5. DINO 三个代表变体

| DINO 变体 | checkpoint | global bs | 等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no-pool fresh | `step_028930` | 96 | 10.00 | 98.67 | 99.33 | 92.33 | 84.00 | 93.58 | 10ep 已超过 VAE baseline 20ep |
| no-pool lr1e-5 extra | `step_028930` | 96 | 20.00 | 98.00 | 98.33 | 98.67 | 90.00 | 96.25 | 当前最强 |
| avgpool lr1e-5 extra from 10ep base | `step_018000` | 128 | 18.29 | 95.33 | 97.33 | 98.00 | 90.00 | 95.17 | token 更省，长任务追平，但 Overall 低于 no-pool |

DINO 三个变体共同支持一个判断：**DINO feature space 对 LIBERO 更友好；保留更完整空间 token 的 no-pool 目前最强；avgpool 可作为效率路线保留。**

## 6. 论文/汇报上怎么讲

建议主张：

> In our FastWAM-style world-action model, predicting frozen DINO visual tokens yields substantially stronger LIBERO control performance than predicting Wan VAE latents under the same small-video DiT capacity. Aligning VAE loss weights with the DINO setting does not close the gap and instead significantly degrades performance, suggesting that the advantage comes from the semantic/geometric structure of DINO features rather than merely from action-heavy loss weighting.

中文表述：

> 在相同 1B 级 small-video DiT 容量下，DINO token dynamics 比 Wan VAE latent dynamics 更适合作为机器人控制的视觉世界模型监督。VAE baseline 在 20epoch 只能达到 `91.33`，而 DINO no-pool 达到 `96.25`；进一步把 VAE loss 权重对齐到 DINO 后，性能下降到 `75.83`，说明 DINO 的优势不是简单由 loss 权重造成，而更可能来自 DINO feature 对物体、局部几何和语义关系的表达。

需要避免的过度表述：

- 不要写成“VAE 天然不如 DINO”。更准确是“在当前 FastWAM 架构、small-video 容量和 LIBERO benchmark 下，DINO token route 明显更有效”。
- 不要把 1B DINO 讲成参数高效训练的核心贡献；当前证据更适合讲“表征空间选择”和“语义 token dynamics”。
- 不要说 loss 权重已经完全公平，因为 VAE baseline 和 DINO 的 global batch、token space、supervision geometry 仍不同；但 loss-aligned VAE 已经足够说明“权重不是 DINO 高分的主因”。

## 7. 下一步建议

1. **停止继续投入 1B VAE smallvideo 主线**：除非换 5B Wan 或引入更合理的 VAE-specific loss recipe，否则当前 1B VAE 已经不是最有希望路线。
2. **DINO no-pool 仍是主 baseline**：`96.25 Overall / 90.00 LIBERO-10` 是当前要守住的比较点。
3. **短期优先看 Short-DINO-Intent**：如果能把 LIBERO-10 从 `90.00` 往上推，就能讲“短时意图/历史 token”故事。
4. **若 Short-DINO-Intent 有效，下一步做长程 memory**：用 DINO 统一 current prediction、short intent、long memory，会比继续 VAE 更自然。
5. **5B DINO/Wan 初始化可以作为刷 SOTA 路线**：但 attribution 上要先保留 1B no-pool 和 1B short-intent 的清晰对照。

## 8. 数据来源

主要评测目录：

```text
evaluate_results/libero/vae_smallvideo_30trials_step_012000_4gpu_mtp4
evaluate_results/libero/vae_smallvideo_30trials_step_014000
evaluate_results/libero/vae_smallvideo_fullresume_step014000_to20ep_30trials_step_018000
evaluate_results/libero/vae_smallvideo_fullresume_step014000_to20ep_30trials_step_020000
evaluate_results/libero/vae_smallvideo_fullresume_step014000_to20ep_30trials_step_021700
evaluate_results/libero/vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_002000
evaluate_results/libero/vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_004000
evaluate_results/libero/vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_005425
evaluate_results/libero/vae_loss005_5_stepmatch_30trials_step_026000_3gpu_mtp4_20260611_115309
evaluate_results/libero/vae_loss005_5_stepmatch_30trials_step_030000_3gpu_mtp4_20260611_125935
evaluate_results/libero/vae_loss005_5_stepmatch_30trials_step_040000_3gpu_mtp4_20260611_141051
evaluate_results/libero/vae_loss005_5_fullresume_step046000_to57860_30trials_step_048000_8gpu_mtp4_20260612_181646
evaluate_results/libero/vae_loss005_5_fullresume_step046000_to57860_30trials_step_050000_8gpu_mtp4_20260612_185950
evaluate_results/libero/vae_loss005_5_fullresume_step046000_to57860_30trials_step_052000_8gpu_mtp4_20260612_193834
evaluate_results/libero/vae_loss005_5_fullresume_step046000_to57860_30trials_step_054000_8gpu_mtp4_20260612_201733
evaluate_results/libero/vae_loss005_5_fullresume_step046000_to57860_30trials_step_056000_8gpu_mtp4_20260612_205646
evaluate_results/libero/vae_loss005_5_fullresume_step046000_to57860_30trials_step_057860_8gpu_mtp4_20260612_213621
evaluate_results/libero/nopool_latest_30trials_step_028930
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_028930_8gpu_mtp4_delayed
evaluate_results/libero/avgpool_fullresume_30trials_step_018000_3gpu_mtp4_20260611_083924
```
