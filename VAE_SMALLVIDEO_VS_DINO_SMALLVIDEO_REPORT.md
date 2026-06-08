# VAE SmallVideo vs DINO SmallVideo LIBERO 对比报告

生成时间：2026-06-08

## 1. 结论

在当前已经完成的 LIBERO 30-trial 评测里，**同为 small VideoDiT / 1B 级 video backbone 的设定下，DINO no-pool 路线明显强于 VAE smallvideo 路线**。

最直接的对比是：

| 路线 | checkpoint | global bs | 累计等效 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE smallvideo | `step_021700` | 256 | 20.00 | 92.00 | 98.00 | 94.00 | 81.33 | 91.33 |
| DINO no-pool | `step_028930` | 96 | 10.00 | 98.67 | 99.33 | 92.33 | 84.00 | 93.58 |
| DINO no-pool + lr1e-5 weight-only | `step_020000` | 96 | 16.91 | 96.67 | 99.33 | 97.00 | 86.67 | 94.92 |

核心判断：

1. DINO no-pool 在约 10 epoch 就达到 `93.58`，已经超过 VAE smallvideo 20 epoch 的 `91.33`。
2. DINO no-pool 继续 weight-only 低学习率训练后，在约 16.91 累计 epoch 达到 `94.92`，比 VAE smallvideo 最好点高 `+3.59` Overall。
3. VAE smallvideo 的瓶颈主要在 Spatial 和 LIBERO-10：Object 一直很高，但 Spatial、长任务和综合分明显低于 DINO no-pool。
4. VAE smallvideo 20 epoch 后再 weight-only lr1e-5 续训没有提升，反而从 `91.33` 下降到 `91.00 / 90.00 / 89.67`。
5. 因此，如果目标是 LIBERO 分数，当前证据支持优先继续 DINO no-pool / DINO token 路线，而不是 VAE smallvideo 路线。

需要注意：这不是一个完全纯粹的“只换 latent 表征”的控制实验，因为两条线的 loss 权重也不同：

- DINO no-pool：`lambda_video=0.05`, `lambda_action=5.0`
- VAE smallvideo：`lambda_video=1.0`, `lambda_action=1.0`

所以更严谨的表述是：**在当前 FastWAM 训练设置和 small VideoDiT 容量下，DINO token route 比 VAE latent route 更有效。**

## 2. 等效 epoch 换算

不同实验的 global batch size 不同，不能直接用 raw step 比较训练量。这里统一使用当前 LIBERO 训练样本量近似：

```text
samples_per_epoch ~= 277713
等效 epoch = step * global_bs / 277713
```

对于 weight-only resume，使用累计等效 epoch：

```text
累计等效 epoch =
    base_step * base_global_bs / 277713
  + resumed_step * resumed_global_bs / 277713
```

当前主要 global batch：

| 路线 | per-GPU batch | grad accum | GPU 数 | global bs |
|---|---:|---:|---:|---:|
| VAE smallvideo | 32 | 1 | 8 | 256 |
| DINO no-pool fresh | 6 | 2 | 8 | 96 |
| DINO no-pool lr1e-5 weight-only | 6 | 2 | 8 | 96 |
| DINO pooled / framecache historical runs | varies | varies | varies | 64 or 128 |

## 3. VAE smallvideo 结果曲线

VAE smallvideo 使用 Wan VAE latent，VideoDiT backbone 缩到和 DINO smallvideo / ActionDiT 同级别，并从 `WanVideoDiT_smallvideo_from_Wan22_alphascale_1024hdim.pt` 初始化。

| run | checkpoint | global bs | run epoch | 累计 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh / full resume | `step_014000` | 256 | 12.90 | 12.90 | 87.67 | 98.00 | 92.33 | 76.67 | 88.67 |
| full resume | `step_018000` | 256 | 16.59 | 16.59 | 90.00 | 97.67 | 94.67 | 76.67 | 89.75 |
| full resume | `step_020000` | 256 | 18.43 | 18.43 | 91.33 | 97.00 | 94.00 | 78.33 | 90.17 |
| full resume | `step_021700` | 256 | 20.00 | 20.00 | 92.00 | 98.00 | 94.00 | 81.33 | 91.33 |
| weight-only lr1e-5 from `021700` | `step_002000` | 256 | 1.84 | 21.84 | 89.67 | 97.67 | 95.00 | 81.67 | 91.00 |
| weight-only lr1e-5 from `021700` | `step_004000` | 256 | 3.69 | 23.69 | 89.67 | 97.67 | 93.00 | 79.67 | 90.00 |
| weight-only lr1e-5 from `021700` | `step_005425` | 256 | 5.00 | 25.00 | 89.67 | 97.00 | 93.33 | 78.67 | 89.67 |

VAE smallvideo 趋势：

- 12.90 -> 20.00 epoch：`88.67 -> 91.33`，有稳定增长。
- 20.00 epoch 后 weight-only lr1e-5：`91.33 -> 91.00 -> 90.00 -> 89.67`，没有继续增长。
- Object 几乎一直很强，约 `97-98`。
- Spatial 从 `87.67` 涨到 `92.00`，但仍低于 DINO no-pool。
- LIBERO-10 从 `76.67` 涨到 `81.33`，仍明显低于 DINO no-pool 最新的 `86.67`。

## 4. DINO no-pool 结果曲线

DINO no-pool 使用 frozen DINO-S token，`latent_spatial_pool=[1,1]`，保留完整双相机 DINO grid：

```text
[384, T, 14, 28] -> 392 tokens/frame
```

| run | checkpoint | global bs | run epoch | 累计 epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh no-pool | `step_026000` | 96 | 8.99 | 8.99 | 96.67 | 99.67 | 92.33 | 83.33 | 93.00 |
| fresh no-pool | `step_028930` | 96 | 10.00 | 10.00 | 98.67 | 99.33 | 92.33 | 84.00 | 93.58 |
| weight-only lr1e-5 from `028930` | `step_004000` | 96 | 1.38 | 11.38 | 97.67 | 99.33 | 97.00 | 79.33 | 93.33 |
| weight-only lr1e-5 from `028930` | `step_008000` | 96 | 2.77 | 12.77 | 96.67 | 98.67 | 93.33 | 84.67 | 93.33 |
| weight-only lr1e-5 from `028930` | `step_012000` | 96 | 4.15 | 14.15 | 96.67 | 98.67 | 95.33 | 80.00 | 92.67 |
| weight-only lr1e-5 from `028930` | `step_016000` | 96 | 5.53 | 15.53 | 96.00 | 96.33 | 97.33 | 85.67 | 93.83 |
| weight-only lr1e-5 from `028930` | `step_020000` | 96 | 6.91 | 16.91 | 96.67 | 99.33 | 97.00 | 86.67 | 94.92 |
| weight-only lr1e-5 from `028930` | `step_022000` | 96 | 7.60 | 17.60 | 98.00 | 98.33 | 96.67 | 86.00 | 94.75 |

说明：

- `step_004000` 存在一次历史重复评测，曾得到 `93.58`；本表使用当前同一续训 run 下的 `30trials_step_004000` 结果 `93.33`。
- `step_012000` 有一次回落，但 `step_016000/020000/022000` 又继续上涨或维持高位，说明 DINO no-pool 续训并没有明显“训崩”，反而还有增长空间。

## 5. 同 epoch / 近似训练量对比

### 5.1 早期效率

| 对比点 | epoch | Overall |
|---|---:|---:|
| DINO no-pool `step_026000` | 8.99 | 93.00 |
| DINO no-pool `step_028930` | 10.00 | 93.58 |
| VAE smallvideo `step_014000` | 12.90 | 88.67 |

DINO no-pool 用更少训练量已经超过 VAE smallvideo 约 `+4.91` Overall。

### 5.2 接近 20 epoch 的效果

| 对比点 | epoch | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| VAE smallvideo `step_021700` | 20.00 | 92.00 | 98.00 | 94.00 | 81.33 | 91.33 |
| DINO no-pool `step_020000` | 16.91 | 96.67 | 99.33 | 97.00 | 86.67 | 94.92 |

虽然 DINO no-pool 的累计 epoch 更少，仍然高出：

| Metric | DINO - VAE |
|---|---:|
| Spatial | +4.67 |
| Object | +1.33 |
| Goal | +3.00 |
| LIBERO-10 | +5.34 |
| Overall | +3.59 |

### 5.3 VAE 继续训 vs DINO 继续训

| 路线 | resume 起点 | 后续训练 | Overall 变化 |
|---|---:|---:|---:|
| VAE smallvideo | 20.00 epoch | +5.00 epoch, lr1e-5 weight-only | 91.33 -> 89.67 |
| DINO no-pool | 10.00 epoch | +6.91 epoch, lr1e-5 weight-only | 93.58 -> 94.92 |

同样是低学习率 weight-only 继续训练，DINO no-pool 有收益，VAE smallvideo 没有收益。

## 6. 为什么 DINO 更强的可能原因

### 6.1 DINO token 更贴近 action 所需的语义和几何线索

DINO no-pool 保留完整 DINO patch token：

```text
2cam RGB 224x448
-> DINO-S patch16
-> [384, T, 14, 28]
-> 392 tokens/frame
```

这些 token 不是为了重建像素，而是来自自监督视觉表征，通常更偏物体、区域、语义对应关系和局部几何。LIBERO 的成功率主要依赖物体定位、相对位置、抓取对象识别和长程操作条件，这些可能更适合 DINO token。

### 6.2 VAE latent 更偏生成重建，不一定最适合动作决策

VAE smallvideo 监督的是 Wan VAE latent velocity，结构更接近原 FastWAM：

```text
RGB video
-> Wan VAE latent
-> small WanVideoDiT Conv3D patchify
-> predict VAE latent velocity
```

VAE latent 的优点是和视频生成预训练一致，但在当前 small VideoDiT 容量下，它没有转化成更强的 LIBERO action performance。

### 6.3 DINO no-pool 没有空间压缩，避免丢细粒度信息

DINO no-pool 不做 avg pooling，也不做 viewpatch merge，完整保留双相机 `14x28` grid。对 LIBERO 这种精细交互任务，完整空间 token 可能非常关键。

这也和 viewpatch / merged-token loss 的负结果一致：减少 token 或改变监督空间后，尤其容易伤 LIBERO-10。

## 7. 当前实验结论边界

这个对比支持 DINO，但还不能写成“VAE 表征天然不如 DINO 表征”的绝对结论，原因是：

1. DINO 和 VAE 的 loss 权重不同。
2. DINO no-pool 的 video loss 权重更小，action loss 权重更大，可能更偏向 action performance。
3. VAE smallvideo 的 `lambda_video=1.0` 可能让模型花更多容量拟合 video latent dynamics。
4. 当前只比较了 small VideoDiT 级别，原生 5B Wan backbone 下结论可能不同。

更稳妥的论文表述可以是：

> Under the same small VideoDiT capacity, DINO token dynamics yields substantially better LIBERO action performance than Wan VAE latent dynamics in our current FastWAM-style joint training setup.

## 8. 建议

优先级建议：

1. 继续把 DINO no-pool `step_024000` 之后的后续 ckpt 补齐评测，确认 `94.92` 附近是否稳定。
2. 当前 VAE smallvideo 可以作为 baseline 放进报告，但不建议继续主力投入。
3. 如果还要做 VAE 对照，建议只做一个更干净的 ablation：把 VAE smallvideo 的 action/video loss 权重对齐到 DINO，即 `lambda_video=0.05, lambda_action=5.0`，否则“表征差异”和“loss 权重差异”会纠缠。
4. DINO no-pool 已经是当前最强路线，下一步更值得做 memory/history 或 5B backbone，而不是继续 viewpatch merged-token loss。

## 9. 数据来源

主要评测目录：

```text
evaluate_results/libero/vae_smallvideo_30trials_step_014000
evaluate_results/libero/vae_smallvideo_fullresume_step014000_to20ep_30trials_step_018000
evaluate_results/libero/vae_smallvideo_fullresume_step014000_to20ep_30trials_step_020000
evaluate_results/libero/vae_smallvideo_fullresume_step014000_to20ep_30trials_step_021700
evaluate_results/libero/vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_002000
evaluate_results/libero/vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_004000
evaluate_results/libero/vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_005425
evaluate_results/libero/nopool_step026000_30trials
evaluate_results/libero/nopool_latest_30trials_step_028930
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_004000
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_008000_4gpu_serial
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_012000_4gpu_serial
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_016000_4gpu_serial
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_020000_4gpu_serial
evaluate_results/libero/nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_022000
```
