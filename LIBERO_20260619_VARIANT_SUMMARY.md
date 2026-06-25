# LIBERO 变体对比结论记录

日期：2026-06-19

## 结论摘要

当前结果不支持“5B 一定强于 1B”，也不支持“Short-DINO-Intent 已经稳定带来收益”。最稳的主 baseline 仍是 **1B DINO no-pool no-intent 20ep**：

| 变体 | checkpoint / step | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| 1B no-intent no-pool，本机 20ep | step_028930 | 98.00 | 98.33 | 98.67 | 90.00 | **96.25** |
| 1B context-intent，AMD 20ep / 本机评测 | step_010860 | 96.33 | 100.00 | 96.67 | 89.00 | 95.50 |
| 5B no-intent，AMD 20ep / 本机评测 | step_010860 | 95.67 | 99.33 | 97.33 | 87.67 | 95.00 |
| 5B context-intent，AMD 20ep / 本机评测 | step_010860 | 94.33 | 100.00 | 94.00 | 83.33 | 92.92 |
| 5B context-intent，AMD 最好截图结果 | step_10000 | 90.70 | 100.00 | 95.30 | 88.70 | 93.70 |

## 1. 1B 和 5B 是否有明显差距

目前没有看到 5B 明显领先。相反，当前最强点是 1B no-intent no-pool 20ep 的 `96.25 Overall / 90.00 LIBERO-10`。

5B no-intent 可以到 `95.00 Overall`，说明 5B 原生 Wan DiT 预测 DINO token 是可行的，但目前没有兑现参数量优势。5B context-intent 更低，在 `92-94 Overall` 区间。

## 2. AMD 评测和本机 NVIDIA 评测差距

同一 checkpoint 跨机器评测总体差距不大，通常在 `0-1 Overall` 左右，但 LIBERO-10 会有更明显抖动。

| 变体 | step | AMD Overall | 本机 Overall | 差值 |
|---|---:|---:|---:|---:|
| 1B context-intent | 010860 | 94.9 | 95.50 | +0.60 |
| 5B no-intent | 010860 | 94.2 | 95.00 | +0.80 |
| 5B context-intent | 010860 | 92.7 | 92.92 | +0.22 |

因此 AMD / NVIDIA 的评测差异不像主因。若要严谨比较某个边界点，建议以本机统一重评为准，尤其是 LIBERO-10。

## 3. AMD 10epoch 左右和本机 10epoch 的差距

本机 1B context-after-proprio 10ep：

| step | Spatial | Object | Goal | LIBERO-10 | Overall |
|---:|---:|---:|---:|---:|---:|
| 022000 | 92.33 | 100.00 | 95.00 | 75.00 | 90.58 |
| 028930 | 93.33 | 98.67 | 90.33 | 77.67 | 90.00 |

AMD 1B context-intent 约 10epoch 附近：

| step | Spatial | Object | Goal | LIBERO-10 | Overall |
|---:|---:|---:|---:|---:|---:|
| 5000 | 95.0 | 99.3 | 87.3 | 63.7 | 86.3 |
| 6000 | 88.0 | 99.3 | 86.7 | 78.0 | 88.0 |

本机 10ep Overall 高约 `2-4 points`，但 AMD `step_6000` 的 LIBERO-10 已接近本机。这个差异更像早期训练波动、step/epoch 对齐不完全、global batch 与硬件随机性的组合，不像“AMD 训练失败”。

本机 1B video-prefix 10ep：

| step | Spatial | Object | Goal | LIBERO-10 | Overall |
|---:|---:|---:|---:|---:|---:|
| 020000 | 91.67 | 100.00 | 87.67 | 68.33 | 86.92 |
| 026000 | 98.00 | 99.67 | 95.33 | 83.00 | 94.00 |
| 028000 | 97.33 | 99.33 | 94.67 | 81.33 | 93.17 |
| 028930 | 98.00 | 99.67 | 92.67 | 81.00 | 92.83 |

video-prefix 10ep 高于本机 context-intent 10ep，但仍低于 1B no-intent 20ep。

## 4. 是否还在提分，是否需要 20ep 后 resume

- **1B context-intent**：从 AMD `step_5000=86.3` 到 `step_010860=94.9/95.50` 明显提升，但 `step_9000=94.8` 后接近平台。20ep 后继续 resume 可能只有小收益，不是当前最高优先级。
- **5B no-intent**：从 `step_5000=79.2` 到 `step_010860=94.2/95.00` 提升明显，是 5B 里最值得继续验证上限的路线。
- **5B context-intent**：AMD best 在 `step_10000=93.7`，后续 `10500=92.2`、`10860=92.7`；本机也在 `92-93`。不建议优先继续 resume。
- **1B video-prefix**：`step_026000=94.00` 后下降到 `step_028930=92.83`。如果继续，建议从 best checkpoint 小学习率试，而不是直接从 final 继续。
- **1B no-intent no-pool**：仍是主 baseline，不建议为了短期收益随意改变。

## 5. Intent 是否有明显作用，是否和参数量有关

目前没有稳定正作用。

- 1B 上，context-intent 20ep 本机 `95.50`，接近但低于 no-intent `96.25`。
- 5B 上，intent 明显不如 no-intent：5B no-intent 本机 `95.00`，5B context-intent 本机 `92.92`，AMD best 也只有 `93.7`。
- Intent 有时会提高 LIBERO-10，例如 AMD 5B intent best 的 LIBERO-10 是 `88.7`，但同时 Spatial 下降，Overall 没有更好。

所以 intent 的问题不像是单纯参数量不足。5B 加 intent 反而更差，说明当前 intent 注入方式或训练 recipe 还没有稳定对齐。

## 下一步建议

短期先优先跑 LIBERO-plus，比较：

1. 1B no-intent no-pool 20ep
2. 1B context-intent 20ep
3. 5B no-intent 20ep

如果 LIBERO-plus 上 intent 明显提升鲁棒性，再继续优化 intent。若 LIBERO-plus 上也没有收益，下一阶段应优先考虑 DINO + RGB VAE auxiliary 或更合理的 dynamics supervision，而不是继续堆 Short-DINO-Intent。

## 6. 2026-06-21 LIBERO-Plus 全量结果更新

LIBERO-Plus 4 个主要 checkpoint 已经全量完成，每个都是 `10030` 个 perturbation tasks、每个 task `1 trial`。

| 变体 | checkpoint | Original | Camera | Robot | Lang. | Light | BG | Noise | Layout | Plus Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B no-intent DINO no-pool，本机 | `step_028930` | 96.25 | 28.46 | 21.94 | 50.03 | 74.69 | 40.99 | 25.80 | 46.69 | 39.71 |
| 1B context-intent DINO，AMD 20ep | `step_010860` | 94.90 | 30.96 | 19.42 | 53.42 | 73.99 | 40.15 | 26.42 | 53.25 | 41.17 |
| 5B no-intent DINO，AMD 20ep | `step_010860` | 94.20 | 27.33 | 16.39 | 54.59 | 72.42 | 50.28 | 17.80 | 49.31 | 39.23 |
| 5B context-intent DINO，AMD 20ep | `step_010860` | 92.70 | 33.77 | 15.23 | 52.31 | 81.17 | 54.65 | 30.61 | 51.87 | **43.63** |

Fast-WAM 论文表中对应结果：

| Model | Original | Camera | Robot | Lang. | Light | BG | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fast-WAM | 97.60 | 16.40 | 44.50 | 68.90 | 78.20 | 53.70 | 37.70 | 60.70 | 51.50 |
| Our best DINO-only per column | 96.25 | 33.77 | 21.94 | 54.59 | 81.17 | 54.65 | 30.61 | 53.25 | 43.63 |

结论：

- DINO-only 没有超过 Fast-WAM 的 LIBERO-Plus Total：当前最好是 `43.63`，低于 Fast-WAM `51.50`。
- DINO-only 在 Camera 上明显强于 Fast-WAM：best `33.77` vs `16.40`。
- 5B context-intent 在 Light / BG 上也能达到或略超 Fast-WAM：Light `81.17` vs `78.20`，BG `54.65` vs `53.70`。
- 主要短板是 Robot / Lang. / Noise / Layout，尤其 Robot：best `21.94` vs Fast-WAM `44.50`。
- 这说明“只用 DINO 替代 VAE latent”不是完整答案；DINO 对部分视觉 OOD 有价值，但不应完全丢掉 Wan/VAE 的语言、几何和生成式世界建模能力。

## 7. 对当前路线的更新判断

之前的目标曾经是用 1B DINO 直接超过 5B Wan/Fast-WAM。现在更合理的判断是：

1. **1B DINO-only 能在原始 LIBERO 接近 Fast-WAM，本身已经说明路线有价值。**
   - 参数量更小。
   - Wan 5B 原本是在 VAE latent/video generation space 里预训练，天然更适配 VAE target，不一定适配 DINO target。
   - 因此 1B DINO-only 不该被期待在所有维度上直接超过 5B Wan。

2. **LIBERO-Plus 暴露了 DINO-only 的边界。**
   - Camera / Light / BG 好，可能来自 DINO 预训练的数据增强和语义不变性，也可能来自它不需要重构背景/光照这类 nuisance detail。
   - Robot / Layout / Noise 差，说明 DINO feature 可能不够保留控制所需的精细空间、接触和像素扰动信息。
   - Lang. 差，可能是因为丢掉了 Wan 原有 text-video latent 对齐能力。

3. **Intent 在 Plus 上比在原始 LIBERO 上更有意义。**
   - 1B context-intent 比 1B no-intent：`41.17 - 39.71 = +1.46 Total`。
   - 5B context-intent 比 5B no-intent：`43.63 - 39.23 = +4.40 Total`。
   - 原始 LIBERO 接近饱和，history intent 不容易显著拉开；Plus 有更多扰动，历史信息更可能帮助 disambiguate 当前状态。

## 8. 当前最值得等的两个实验

截至 2026-06-21，用户反馈有两条训练在跑，预计周三上午出结果：

1. **1B-DINO video-prefix Short-DINO-Intent resume**
   - 本机 8 卡。
   - `video_prefix` 10ep best 已经超过 no-intent 10ep fresh：`94.00` vs `93.58`。
   - 现在要看继续训练后能否接近或超过 no-intent 20ep 的 `96.25`。

2. **3-branch MoT no-intent**
   - 另一台 8 卡 H200。
   - 结构是 Wan/VAE branch + DINO auxiliary branch + action branch。
   - 目标不是再让 DINO 替代 VAE，而是让 Wan/VAE 保留原本语言/生成式世界建模能力，同时用 DINO auxiliary supervision 补 Camera/Light/BG 这类 OOD 泛化。

## 9. 周三后的决策规则

如果 `3-branch MoT no-intent` 同时满足：

- 原始 LIBERO 不明显掉分；
- LIBERO-Plus Total 提升；
- Robot / Lang. / Layout 至少不比 DINO-only 更差；

那下一步应优先做：

```text
3-branch MoT + Short-DINO-Intent
```

如果 3-branch 只提升 Camera / Light，但 Robot / Lang. 仍弱，说明 DINO auxiliary 只是补视觉不变性，还没有补控制/语言核心问题。

如果 3-branch 掉分，不要立刻否定 DINO auxiliary；优先排查：

- `lambda_dino=0.02` 是否过大，可试 `0.01`；
- DINO loss 是否应该 delayed/warmup 后打开；
- VAE/action branch 是否被 DINO branch 的 mixed attention 干扰；
- 三分支 mask 是否过于宽松，让 future token 间的信息交互破坏了 attribution。

如果 `video_prefix resume` 继续涨：

- 保留它作为 Short-DINO-Intent 主注入方式。
- 后续做 3-branch + intent 时，优先参考 `video_prefix` 而不是 context-after-proprio。

如果 `video_prefix resume` 不涨：

- 不再单独投入太多 GPU 到纯 DINO intent。
- 把 intent 放到 3-branch MoT 上验证，避免在 DINO-only 上过拟合结论。
