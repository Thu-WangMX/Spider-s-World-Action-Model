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
