#!/usr/bin/env python3
"""Build dashboard_data.json for the LIBERO experiment history page."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "libero_dashboard" / "dashboard_data.json"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_TRAIN_SAMPLE_COUNT = 277713


MANUAL_NOTES: list[tuple[str, dict[str, Any]]] = [
    (
        "viewpatch_1x2x2_mergedloss_30trials_step_021700",
        {
            "valid": True,
            "resume_type": "fresh viewpatch train",
            "learning_rate": "1e-4 cosine",
            "global_batch": 128,
            "pooling": "view-aware patch merge [1,2,2], merged-token loss",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,2,2], output_patch_space=merged, loss on merged tokens, eval step021700",
        },
    ),
    (
        "vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_005425",
        {
            "valid": True,
            "resume_type": "weight-only restart from VAE-small full-resume step021700",
            "resume_base_step": 21700,
            "learning_rate": "1e-5 cosine restart",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, weight-only from step021700, lr1e-5, eval step005425",
        },
    ),
    (
        "vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_004000",
        {
            "valid": True,
            "resume_type": "weight-only restart from VAE-small full-resume step021700",
            "resume_base_step": 21700,
            "learning_rate": "1e-5 cosine restart",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, weight-only from step021700, lr1e-5, eval step004000",
        },
    ),
    (
        "vae_smallvideo_weightonly_from_step021700_lr1e-5_30trials_step_002000",
        {
            "valid": True,
            "resume_type": "weight-only restart from VAE-small full-resume step021700",
            "resume_base_step": 21700,
            "learning_rate": "1e-5 cosine restart",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, weight-only from step021700, lr1e-5, eval step002000",
        },
    ),
    (
        "vae_loss005_5_fullresume_step046000_to57860_30trials_step_",
        {
            "valid": True,
            "resume_type": "full-state resume from VAE loss-aligned step046000 to 20ep endpoint",
            "learning_rate": "1e-4 cosine resumed",
            "global_batch": 96,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "VAE small-video loss-aligned",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, loss weights aligned to DINO (lambda_video=0.05, lambda_action=5.0), bs24 x 4gpu, eval step048000-step057860",
        },
    ),
    (
        "vae_loss005_5_stepmatch_30trials_step_",
        {
            "valid": True,
            "resume_type": "fresh VAE loss-aligned train, step-matched 20ep schedule",
            "learning_rate": "1e-4 cosine",
            "global_batch": 96,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "VAE small-video loss-aligned",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, loss weights aligned to DINO (lambda_video=0.05, lambda_action=5.0), bs24 x 4gpu",
        },
    ),
    (
        "vae_smallvideo_fullresume_step014000_to20ep_30trials_step_021700",
        {
            "valid": True,
            "resume_type": "full-state resume from VAE-small step014000, trained to original 20ep endpoint",
            "learning_rate": "1e-4 cosine resumed",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, small VideoDiT 1024 hidden, full resume step014000 -> step021700",
        },
    ),
    (
        "vae_smallvideo_fullresume_step014000_to20ep_30trials_step_020000",
        {
            "valid": True,
            "resume_type": "full-state resume from VAE-small step014000",
            "learning_rate": "1e-4 cosine resumed",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, small VideoDiT 1024 hidden, full resume step014000 -> step020000",
        },
    ),
    (
        "vae_smallvideo_fullresume_step014000_to20ep_30trials_step_018000",
        {
            "valid": True,
            "resume_type": "full-state resume from VAE-small step014000",
            "learning_rate": "1e-4 cosine resumed",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, small VideoDiT 1024 hidden, full resume step014000 -> step018000",
        },
    ),
    (
        "vae_smallvideo_30trials_step_014000",
        {
            "valid": True,
            "resume_type": "fresh VAE-small train, eval step014000",
            "learning_rate": "1e-4 cosine",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, small VideoDiT 1024 hidden, base FastWAM first-frame action eval",
        },
    ),
    (
        "vae_smallvideo_30trials_step_012000_4gpu_mtp4",
        {
            "valid": True,
            "resume_type": "fresh VAE-small train, eval step012000",
            "learning_rate": "1e-4 cosine",
            "global_batch": 256,
            "pooling": "Wan VAE latent + Conv3D patchify [1,2,2]",
            "lambda_video": 1.0,
            "lambda_action": 1.0,
            "model": "VAE small-video baseline",
            "wan_init": "WanVideoDiT small from Wan2.2 interpolation",
            "variant": "VAE latent, small VideoDiT 1024 hidden, base FastWAM first-frame action eval",
        },
    ),
    (
        "viewpatch_1x1x2_lr2e-5_30trials_step_020000",
        {
            "valid": True,
            "resume_type": "weight-only resume from viewpatch_1x1x2_mmap_bs12_w6_30trials_step_024000",
            "resume_base_step": 24000,
            "learning_rate": "2e-5 constant",
            "global_batch": 96,
            "pooling": "view-aware patch merge [1,1,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,1,2], weight-only from step024000, lr2e-5, eval step020000",
        },
    ),
    (
        "viewpatch_1x1x2_weightinit_step024000_30trials_step_012000",
        {
            "valid": True,
            "warning": "unexpectedly low eval; logs confirm checkpoint step_012000 was loaded",
            "resume_type": "weight-only restart from viewpatch [1,1,2] step024000; later full-state resume from step009500",
            "resume_base_step": 24000,
            "learning_rate": "1e-4 cosine restart",
            "global_batch": 96,
            "pooling": "view-aware patch merge [1,1,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,1,2], weight-init step024000, eval step012000",
        },
    ),
    (
        "viewpatch_1x1x2_weightinit_step024000_30trials_step_016000",
        {
            "valid": True,
            "resume_type": "weight-only restart from viewpatch [1,1,2] step024000; later full-state resume from step009500",
            "resume_base_step": 24000,
            "learning_rate": "1e-4 cosine restart",
            "global_batch": 96,
            "pooling": "view-aware patch merge [1,1,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,1,2], weight-init step024000, eval step016000",
        },
    ),
    (
        "viewpatch_1x1x2_mmap_bs12_w6_30trials_step_024000",
        {
            "valid": True,
            "resume_type": "fresh viewpatch train",
            "learning_rate": "1e-4",
            "global_batch": 96,
            "pooling": "view-aware patch merge [1,1,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,1,2], mmap cache, step024000",
        },
    ),
    (
        "viewpatch_1x2x2_weightonly_from_step032000_lr2e-5_30trials_step_008000",
        {
            "valid": True,
            "resume_type": "weight-only restart from viewpatch step032000",
            "resume_base_step": 32000,
            "learning_rate": "2e-5 constant with warmup",
            "global_batch": 128,
            "pooling": "view-aware patch merge [1,2,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,2,2], weight-only restart, step008000",
        },
    ),
    (
        "viewpatch_1x2x2_weightonly_from_step032000_lr2e-5_30trials_step_014000",
        {
            "valid": True,
            "resume_type": "weight-only restart from viewpatch step032000",
            "resume_base_step": 32000,
            "learning_rate": "2e-5 constant with warmup",
            "global_batch": 128,
            "pooling": "view-aware patch merge [1,2,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,2,2], weight-only restart, step014000",
        },
    ),
    (
        "viewpatch_1x2x2_resume_step023000_30trials_step_032000",
        {
            "valid": True,
            "resume_type": "full-state resume from viewpatch step023000",
            "learning_rate": "1e-4 cosine, lr~1e-6 at step032000",
            "global_batch": 128,
            "pooling": "view-aware patch merge [1,2,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,2,2], mmap cache, step032000",
        },
    ),
    (
        "viewpatch_1x2x2_mmap_bs16_w4_30trials_step_023000",
        {
            "valid": True,
            "resume_type": "fresh viewpatch train",
            "learning_rate": "1e-4",
            "global_batch": 128,
            "pooling": "view-aware patch merge [1,2,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "viewpatch [1,2,2], mmap cache, step023000",
        },
    ),
    (
        "finetune_from_028930_latest_30trials_step_005786",
        {
            "valid": False,
            "warning": "invalid: old weight-only resume loaded .pt after DeepSpeed prepare; optimizer master weights likely overwrote model weights",
            "resume_type": "weight-only (buggy, do not compare)",
            "resume_base_step": 28930,
            "learning_rate": "2e-5",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "buggy weight-only fine-tune",
        },
    ),
    (
        "nopool_weightonly_from_step028930_lr1e-5_extra10ep_30trials_step_",
        {
            "valid": True,
            "resume_type": "fixed weight-only restart from no-pool step028930",
            "resume_base_step": 28930,
            "learning_rate": "1e-5 cosine restart",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "no-pool fixed weight-only from step028930, lr1e-5 extra10ep",
        },
    ),
    (
        "nopool_latest_30trials_step_004000",
        {
            "valid": True,
            "resume_type": "fixed weight-only restart from no-pool step028930",
            "resume_base_step": 28930,
            "learning_rate": "1e-5",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "no-pool fixed weight-only from step028930, eval step004000",
        },
    ),
    (
        "nopool_latest_30trials_step_028930",
        {
            "valid": True,
            "resume_type": "fresh no-pool train",
            "learning_rate": "1e-4",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "no-pooling, batch96",
        },
    ),
    (
        "aihub_1b_smallvideo_context_intent_30trials_step_",
        {
            "valid": True,
            "resume_type": "fresh Short-DINO-Intent context-after-proprio train on AMD",
            "learning_rate": "1e-4 cosine",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video + Short-DINO-Intent",
            "wan_init": "false",
            "variant": "AIHub/AMD 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is the 20ep endpoint",
            "manual_endpoint_step": 10860,
            "manual_endpoint_epoch": 20.0,
        },
    ),
    (
        "aihub_1b_smallvideo_context_intent_30trials_step_010860",
        {
            "manual_epoch": 20.0,
            "variant": "AIHub/AMD 1B Short-DINO-Intent context-after-proprio, 20ep endpoint, history offsets [-8,-4,0], K=8",
        },
    ),
    (
        "short_dino_intent_video_prefix_10ep_30trials_step_",
        {
            "valid": True,
            "resume_type": "fresh Short-DINO-Intent video-prefix train on local 8xGPU",
            "learning_rate": "5e-5 cosine (configured run, despite script name mentioning lr1e-4)",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video + Short-DINO-Intent",
            "wan_init": "false",
            "variant": "Short-DINO-Intent video_prefix, history offsets [-8,-4,0], K=8; local 10ep run",
        },
    ),
    (
        "short_dino_intent_video_prefix_weightonly_from_step026000_lr1e-5_extra15ep_30trials_step_",
        {
            "valid": True,
            "resume_type": "weight-only resume from local video_prefix step026000, lr1e-5 extra15ep schedule",
            "resume_base_step": 26000,
            "resume_base_global_batch": 96,
            "learning_rate": "1e-5 cosine restart",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video + Short-DINO-Intent",
            "wan_init": "false",
            "variant": "Short-DINO-Intent video_prefix, weight-only from local step026000, lr1e-5 extra15ep, history offsets [-8,-4,0], K=8",
        },
    ),
    (
        "aihub_5b_dino_s_nointent_30trials_step_",
        {
            "valid": True,
            "resume_type": "fresh AIHub/AMD 5B DINO-S no-intent train",
            "learning_rate": "1e-4 cosine",
            "global_batch": 128,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S 5B video expert + 1B action expert",
            "wan_init": "native Wan2.2 5B video init",
            "variant": "AIHub/AMD 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is the 20ep endpoint",
            "manual_endpoint_step": 10860,
            "manual_endpoint_epoch": 20.0,
        },
    ),
    (
        "aihub_5b_dino_s_nointent_30trials_step_010860",
        {
            "manual_epoch": 20.0,
            "variant": "AIHub/AMD 5B DINO-S no-intent, 20ep endpoint, latent_spatial_pool=[1,1]",
        },
    ),
    (
        "aihub_5b_dino_s_context_intent_30trials_step_",
        {
            "valid": True,
            "resume_type": "fresh AIHub/AMD 5B Short-DINO-Intent context-after-proprio train",
            "learning_rate": "1e-4 cosine",
            "global_batch": 128,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
            "wan_init": "native Wan2.2 5B video init",
            "variant": "AIHub/AMD 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is the 20ep endpoint",
            "manual_endpoint_step": 10860,
            "manual_endpoint_epoch": 20.0,
        },
    ),
    (
        "aihub_5b_dino_s_context_intent_30trials_step_010860",
        {
            "manual_epoch": 20.0,
            "variant": "AIHub/AMD 5B Short-DINO-Intent context-after-proprio, 20ep endpoint, history offsets [-8,-4,0], K=8",
        },
    ),
    (
        "aihub_5b_dino_s_context_intent_30trials_step_010000",
        {
            "valid": False,
            "warning": "invalid/partial eval: libero_object task0 was interrupted, summary uses 39/40 tasks",
            "variant": "AIHub/AMD 5B Short-DINO-Intent context-after-proprio, partial 39/40-task eval at step010000",
        },
    ),
    (
        "short_dino_intent_30trials_step_",
        {
            "valid": True,
            "resume_type": "fresh Short-DINO-Intent context-after-proprio train",
            "learning_rate": "5e-5 cosine",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video + Short-DINO-Intent",
            "wan_init": "false",
            "variant": "Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8, fresh 10ep",
        },
    ),
    (
        "nopool_step026000_30trials",
        {
            "valid": True,
            "resume_type": "fresh no-pool train",
            "learning_rate": "1e-4",
            "global_batch": 96,
            "pooling": "none [1,1]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "no-pooling, batch96",
        },
    ),
    (
        "libero_dino_s_smallvideo_lv1_step052000",
        {
            "valid": True,
            "resume_type": "full-state resume from step040000",
            "learning_rate": "1e-5",
            "global_batch": 128,
            "pooling": "fixed avg [1,2]",
            "lambda_video": 1.0,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "pooled, lambda_video=1.0",
        },
    ),
    (
        "avgpool_fullresume_30trials_step_",
        {
            "valid": True,
            "resume_type": "weight-only restart from avgpool step043400 (10epoch base)",
            "resume_base_step": 43400,
            "resume_base_global_batch": 64,
            "learning_rate": "1e-5 cosine restart",
            "global_batch": 128,
            "pooling": "fixed avg [1,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "avgpool [1,2], weight-only from step043400 10epoch base, lr1e-5 extra10ep",
        },
    ),
    (
        "libero_dino_s_smallvideo_framecache_step_",
        {
            "valid": True,
            "resume_type": "pooled training lineage",
            "learning_rate": "1e-4",
            "global_batch": 64,
            "pooling": "fixed avg [1,2]",
            "lambda_video": 0.05,
            "lambda_action": 5.0,
            "model": "DINO-S small-video",
            "wan_init": "false",
            "variant": "pooled frame-cache, batch64",
        },
    ),
    (
        "libero_dino_s_pool_step008000",
        {
            "valid": True,
            "resume_type": "fresh pooled train",
            "learning_rate": "5e-5",
            "global_batch": 128,
            "pooling": "fixed avg [1,2]",
            "lambda_video": 0.25,
            "lambda_action": 1.0,
            "model": "DINO-S big-video",
            "wan_init": "false",
            "variant": "early pooled big-video",
        },
    ),
]

MANUAL_EVALS: list[dict[str, Any]] = [
    {
        "id": "amd_manual_1b_context_intent_step_005000",
        "step": 5000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 95.0, "libero_object": 99.3, "libero_goal": 87.3, "libero_10": 63.7, "overall": 86.3},
    },
    {
        "id": "amd_manual_1b_context_intent_step_006000",
        "step": 6000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 88.0, "libero_object": 99.3, "libero_goal": 86.7, "libero_10": 78.0, "overall": 88.0},
    },
    {
        "id": "amd_manual_1b_context_intent_step_007000",
        "step": 7000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 95.3, "libero_object": 99.3, "libero_goal": 89.3, "libero_10": 79.0, "overall": 90.8},
    },
    {
        "id": "amd_manual_1b_context_intent_step_008000",
        "step": 8000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 95.3, "libero_object": 100.0, "libero_goal": 91.0, "libero_10": 79.3, "overall": 91.4},
    },
    {
        "id": "amd_manual_1b_context_intent_step_009000",
        "step": 9000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 97.3, "libero_object": 98.3, "libero_goal": 94.3, "libero_10": 89.0, "overall": 94.8},
    },
    {
        "id": "amd_manual_1b_context_intent_step_010000",
        "step": 10000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 96.3, "libero_object": 99.7, "libero_goal": 94.3, "libero_10": 85.3, "overall": 93.9},
    },
    {
        "id": "amd_manual_1b_context_intent_step_010860",
        "step": 10860,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "model": "DINO-S small-video + Short-DINO-Intent",
        "variant": "AMD manual 1B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 97.3, "libero_object": 99.7, "libero_goal": 96.3, "libero_10": 86.3, "overall": 94.9},
    },
    {
        "id": "amd_manual_5b_nointent_step_005000",
        "step": 5000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 91.7, "libero_object": 98.3, "libero_goal": 76.3, "libero_10": 50.7, "overall": 79.2},
    },
    {
        "id": "amd_manual_5b_nointent_step_006000",
        "step": 6000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 95.0, "libero_object": 99.7, "libero_goal": 91.7, "libero_10": 81.2, "overall": 91.9},
    },
    {
        "id": "amd_manual_5b_nointent_step_007000",
        "step": 7000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 91.3, "libero_object": 97.3, "libero_goal": 84.3, "libero_10": 77.0, "overall": 87.5},
    },
    {
        "id": "amd_manual_5b_nointent_step_008000",
        "step": 8000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 94.3, "libero_object": 99.7, "libero_goal": 93.3, "libero_10": 79.7, "overall": 91.8},
    },
    {
        "id": "amd_manual_5b_nointent_step_009000",
        "step": 9000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 94.7, "libero_object": 98.0, "libero_goal": 95.3, "libero_10": 81.3, "overall": 92.3},
    },
    {
        "id": "amd_manual_5b_nointent_step_010000",
        "step": 10000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 93.3, "libero_object": 98.7, "libero_goal": 97.3, "libero_10": 88.1, "overall": 94.3},
    },
    {
        "id": "amd_manual_5b_nointent_step_010860",
        "step": 10860,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert",
        "variant": "AMD manual 5B DINO-S no-intent, latent_spatial_pool=[1,1]; step010860 is 20ep",
        "scores": {"libero_spatial": 95.0, "libero_object": 99.3, "libero_goal": 97.7, "libero_10": 84.7, "overall": 94.2},
    },
    {
        "id": "amd_manual_5b_context_intent_step_005000",
        "step": 5000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 74.3, "libero_object": 95.0, "libero_goal": 81.0, "libero_10": 66.0, "overall": 79.1},
    },
    {
        "id": "amd_manual_5b_context_intent_step_006000",
        "step": 6000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 94.7, "libero_object": 98.7, "libero_goal": 88.7, "libero_10": 75.7, "overall": 89.4},
    },
    {
        "id": "amd_manual_5b_context_intent_step_007000",
        "step": 7000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 95.0, "libero_object": 98.3, "libero_goal": 90.3, "libero_10": 74.3, "overall": 89.5},
    },
    {
        "id": "amd_manual_5b_context_intent_step_008000",
        "step": 8000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 94.0, "libero_object": 99.7, "libero_goal": 95.0, "libero_10": 84.7, "overall": 93.3},
    },
    {
        "id": "amd_manual_5b_context_intent_step_009000",
        "step": 9000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 96.3, "libero_object": 100.0, "libero_goal": 91.3, "libero_10": 80.7, "overall": 92.1},
    },
    {
        "id": "amd_manual_5b_context_intent_step_010000",
        "step": 10000,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 90.7, "libero_object": 100.0, "libero_goal": 95.3, "libero_10": 88.7, "overall": 93.7},
    },
    {
        "id": "amd_manual_5b_context_intent_step_010500",
        "step": 10500,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 94.0, "libero_object": 99.7, "libero_goal": 92.3, "libero_10": 82.7, "overall": 92.2},
    },
    {
        "id": "amd_manual_5b_context_intent_step_010860",
        "step": 10860,
        "endpoint_step": 10860,
        "endpoint_epoch": 20.0,
        "global_batch": 128,
        "model": "DINO-S 5B video expert + 1B action expert + Short-DINO-Intent",
        "variant": "AMD manual 5B Short-DINO-Intent context-after-proprio, history offsets [-8,-4,0], K=8; step010860 is 20ep",
        "scores": {"libero_spatial": 93.0, "libero_object": 100.0, "libero_goal": 95.7, "libero_10": 82.0, "overall": 92.7},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_022000",
        "step": 22000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.01-0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-24",
        "eval_date": "06-24 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 97.33, "libero_object": 100.0, "libero_goal": 96.33, "libero_10": 96.33, "overall": 97.50},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_016000",
        "step": 16000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-25",
        "eval_date": "06-25 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 93.67, "libero_object": 98.67, "libero_goal": 95.67, "libero_10": 87.0, "overall": 93.75},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_020000",
        "step": 20000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-25",
        "eval_date": "06-25 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 95.0, "libero_object": 99.33, "libero_goal": 97.33, "libero_10": 94.33, "overall": 96.50},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_028000",
        "step": 28000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-25",
        "eval_date": "06-25 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 96.33, "libero_object": 99.33, "libero_goal": 97.33, "libero_10": 92.67, "overall": 96.42},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_032000",
        "step": 32000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-25",
        "eval_date": "06-25 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 98.0, "libero_object": 99.67, "libero_goal": 99.0, "libero_10": 92.67, "overall": 97.33},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_036000",
        "step": 36000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.01-0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-24",
        "eval_date": "06-24 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 96.67, "libero_object": 99.33, "libero_goal": 97.67, "libero_10": 94.67, "overall": 97.08},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_040000",
        "step": 40000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.01-0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-24",
        "eval_date": "06-24 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 98.67, "libero_object": 99.33, "libero_goal": 98.67, "libero_10": 93.0, "overall": 97.42},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_042000",
        "step": 42000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.01-0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-24",
        "eval_date": "06-24 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 98.33, "libero_object": 100.0, "libero_goal": 97.33, "libero_10": 95.33, "overall": 97.75},
    },
    {
        "id": "manual_wan5b_dino_s_aux_mot_nointent_step_043400",
        "step": 43400,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.01-0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "Manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-24",
        "eval_date": "06-24 manual",
        "eval_platform": "manual non-AMD",
        "source_note": "Non-AMD manual eval from user-provided table; suite-level only, no local summary.json/task videos.",
        "scores": {"libero_spatial": 98.0, "libero_object": 100.0, "libero_goal": 99.33, "libero_10": 93.0, "overall": 97.58},
    },
    {
        "id": "manual_plus_wan5b_dino_s_aux_mot_nointent_step_022000",
        "step": 22000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "LIBERO-Plus manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-25",
        "eval_date": "06-25 manual Plus",
        "eval_platform": "manual non-AMD",
        "eval_dir": "manual/wan5b_3mot_nointent_step_022000_plus10030_wpg5_20260625_033135",
        "source_note": "LIBERO-Plus eval copied from the completed turbo run; suite/group-level backup only, no local task videos on b3.",
        "resume_type": "manual LIBERO-Plus eval table",
        "scores": {"libero_spatial": 68.03, "libero_object": 86.26, "libero_goal": 53.07, "libero_10": 59.71, "overall": 66.77},
        "average_task_time_s": 107.60,
        "suite_times": {"libero_spatial": 92.76, "libero_object": 70.29, "libero_goal": 107.65, "libero_10": 158.98},
        "suite_trials": {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519},
        "plus_groups": {
            "Camera": {"score": 44.28, "tasks": 1599, "episodes": 1599, "successes": 708, "average_time_s": 116.58},
            "Robot": {"score": 55.94, "tasks": 1550, "episodes": 1550, "successes": 867, "average_time_s": 100.30},
            "Language": {"score": 75.34, "tasks": 1537, "episodes": 1537, "successes": 1158, "average_time_s": 76.92},
            "Light": {"score": 90.89, "tasks": 1142, "episodes": 1142, "successes": 1038, "average_time_s": 90.51},
            "Background": {"score": 69.14, "tasks": 1076, "episodes": 1076, "successes": 744, "average_time_s": 91.12},
            "Noise": {"score": 69.96, "tasks": 1601, "episodes": 1601, "successes": 1120, "average_time_s": 181.23},
            "Layout": {"score": 68.85, "tasks": 1525, "episodes": 1525, "successes": 1050, "average_time_s": 83.63},
            "Total": {"score": 66.65, "tasks": 10030, "episodes": 10030, "successes": 6685, "average_time_s": 107.60},
        },
    },
    {
        "id": "manual_plus_wan5b_dino_s_aux_mot_nointent_step_042000",
        "step": 42000,
        "endpoint_step": 43400,
        "endpoint_epoch": 20.0,
        "global_batch": 96,
        "learning_rate": "1e-4 cosine",
        "lambda_video": 1.0,
        "lambda_action": 1.0,
        "lambda_dino_aux": "0.02",
        "model": "3-branch MOT: Wan5B VAE video + DINO-S auxiliary video + action",
        "wan_init": "native Wan5B VAE branch + DINO-S auxiliary branch",
        "variant": "LIBERO-Plus manual non-AMD 3-branch MOT no-intent, VAE/action main branches plus DINO-S auxiliary branch; step043400 is 20ep",
        "source_date": "2026-06-25",
        "eval_date": "06-25 manual Plus",
        "eval_platform": "manual non-AMD",
        "eval_dir": "manual/libero_plus_user_provided_table",
        "source_note": "LIBERO-Plus manual eval from user-provided table; suite/group-level only, no local task videos.",
        "resume_type": "manual LIBERO-Plus eval table",
        "scores": {"libero_spatial": 68.48, "libero_object": 86.10, "libero_goal": 50.41, "libero_10": 61.10, "overall": 66.52},
        "average_task_time_s": 107.99,
        "suite_times": {"libero_spatial": 93.75, "libero_object": 70.68, "libero_goal": 111.49, "libero_10": 155.28},
        "plus_groups": {
            "Camera": {"score": 48.16, "tasks": 1599, "episodes": 1599, "successes": 770, "average_time_s": 113.68},
            "Robot": {"score": 54.13, "tasks": 1550, "episodes": 1550, "successes": 839, "average_time_s": 101.33},
            "Language": {"score": 72.41, "tasks": 1537, "episodes": 1537, "successes": 1113, "average_time_s": 81.19},
            "Light": {"score": 93.52, "tasks": 1142, "episodes": 1142, "successes": 1068, "average_time_s": 85.59},
            "Background": {"score": 71.00, "tasks": 1076, "episodes": 1076, "successes": 764, "average_time_s": 88.59},
            "Noise": {"score": 67.27, "tasks": 1601, "episodes": 1601, "successes": 1077, "average_time_s": 185.99},
            "Layout": {"score": 67.34, "tasks": 1525, "episodes": 1525, "successes": 1027, "average_time_s": 84.41},
            "Total": {"score": 66.38, "tasks": 10030, "episodes": 10030, "successes": 6658, "average_time_s": 107.99},
        },
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def parse_step(text: str) -> int | None:
    matches = re.findall(r"step[_-]?0*(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def read_yaml_light(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_nested(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    return path if path.is_absolute() else ROOT / path


def infer_train_sample_count(cfg: dict[str, Any]) -> int | None:
    dataset_dirs = get_nested(cfg, "data.train.dataset_dirs")
    if not isinstance(dataset_dirs, list):
        return DEFAULT_TRAIN_SAMPLE_COUNT

    total = 0
    for ds_dir in dataset_dirs:
        info_path = resolve_repo_path(str(ds_dir))
        if info_path is None:
            continue
        info_path = info_path / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            continue
        frames = as_int(info.get("total_frames"))
        if frames is not None:
            total += frames
    return total or DEFAULT_TRAIN_SAMPLE_COUNT


def estimate_epoch(step: int | None, metadata: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    global_batch = as_int(metadata.get("global_batch"))
    if step is None or global_batch is None or global_batch <= 0:
        return {
            "epoch": None,
            "run_epoch": None,
            "resume_base_epoch": None,
            "steps_per_epoch": None,
            "resume_base_steps_per_epoch": None,
            "train_sample_count": None,
        }

    train_sample_count = infer_train_sample_count(cfg)
    if train_sample_count is None or train_sample_count <= 0:
        return {
            "epoch": None,
            "run_epoch": None,
            "resume_base_epoch": None,
            "steps_per_epoch": None,
            "resume_base_steps_per_epoch": None,
            "train_sample_count": None,
        }

    steps_per_epoch = max((train_sample_count + global_batch - 1) // global_batch, 1)
    run_epoch = float(step) / float(steps_per_epoch)
    resume_base_step = as_int(metadata.get("resume_base_step"))
    resume_base_global_batch = as_int(metadata.get("resume_base_global_batch")) or global_batch
    resume_base_epoch = 0.0
    resume_base_steps_per_epoch = None
    if resume_base_step is not None and resume_base_step > 0 and resume_base_global_batch > 0:
        resume_base_steps_per_epoch = max(
            (train_sample_count + resume_base_global_batch - 1) // resume_base_global_batch,
            1,
        )
        resume_base_epoch = float(resume_base_step) / float(resume_base_steps_per_epoch)

    return {
        "epoch": round(resume_base_epoch + run_epoch, 2),
        "run_epoch": round(run_epoch, 2),
        "resume_base_epoch": round(resume_base_epoch, 2) if resume_base_step is not None else None,
        "steps_per_epoch": steps_per_epoch,
        "resume_base_steps_per_epoch": resume_base_steps_per_epoch,
        "train_sample_count": train_sample_count,
    }


def config_for_ckpt(ckpt: str | None) -> dict[str, Any]:
    if not ckpt:
        return {}
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    parts = ckpt_path.parts
    if "checkpoints" not in parts:
        return {}
    run_dir = Path(*parts[: parts.index("checkpoints")])
    return read_yaml_light(run_dir / "config.yaml")


def infer_metadata(eval_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    text = f"{eval_dir.as_posix()} {summary.get('run_id','')} {summary.get('ckpt','')}"
    metadata: dict[str, Any] = {
        "valid": True,
        "warning": "",
        "resume_type": "unknown",
        "learning_rate": None,
        "global_batch": None,
        "pooling": "unknown",
        "lambda_video": None,
        "lambda_action": None,
        "model": "unknown",
        "wan_init": "unknown",
        "variant": "unknown",
    }
    cfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    file_cfg = config_for_ckpt(summary.get("ckpt"))
    merged_cfg = file_cfg or cfg

    lr = get_nested(merged_cfg, "learning_rate")
    batch_size = get_nested(merged_cfg, "batch_size")
    grad_acc = get_nested(merged_cfg, "gradient_accumulation_steps")
    lambda_video = get_nested(merged_cfg, "model.loss.lambda_video")
    lambda_action = get_nested(merged_cfg, "model.loss.lambda_action")
    pool = get_nested(merged_cfg, "model.dino_config.latent_spatial_pool")
    wan_init = get_nested(merged_cfg, "model.video_dit_init_from_wan")

    if lr is not None:
        metadata["learning_rate"] = lr
    if batch_size is not None and grad_acc is not None:
        # All these LIBERO training runs used 8 GPUs unless noted manually.
        metadata["global_batch"] = int(batch_size) * int(grad_acc) * 8
    if lambda_video is not None:
        metadata["lambda_video"] = lambda_video
    if lambda_action is not None:
        metadata["lambda_action"] = lambda_action
    if pool is not None:
        metadata["pooling"] = f"none {pool}" if list(pool) == [1, 1] else f"fixed avg {pool}"
    if wan_init is not None:
        metadata["wan_init"] = str(wan_init).lower()

    if "smallvideo" in text:
        metadata["model"] = "DINO-S small-video"
    elif "dino_s_pool" in text or "dino_s_2cam" in text:
        metadata["model"] = "DINO-S big-video"

    if metadata["pooling"] == "unknown":
        if "nopool" in text or "pool1x1" in text:
            metadata["pooling"] = "none [1,1]"
        elif "pool" in text or "framecache" in text:
            metadata["pooling"] = "fixed avg [1,2]"

    # Apply known project-history notes last. Some reused config files do not
    # reflect the actual launched ablation settings we recorded in PROJECT_CONTEXT.
    for needle, override in MANUAL_NOTES:
        if needle in text:
            metadata.update(override)

    return metadata


def suite_score(stats: dict[str, Any]) -> float | None:
    trials = stats.get("total_trials")
    successes = stats.get("total_successes")
    if not trials:
        return None
    return float(successes) * 100.0 / float(trials)


def parse_video(path: Path, eval_dir: Path) -> dict[str, Any] | None:
    name = path.name
    m = re.search(r"episode=task(\d+)_trial(\d+)--success=(True|False)--task=(.*)\.mp4$", name)
    suite = path.parent.parent.name if path.parent.name == "videos" else ""
    if not m or not suite:
        return None
    task_id = int(m.group(1))
    trial_id = int(m.group(2))
    success = m.group(3) == "True"
    task_slug = m.group(4).replace("_", " ")
    return {
        "suite": suite,
        "task_id": task_id,
        "task_key": f"{suite}_{task_id}",
        "trial_id": trial_id,
        "success": success,
        "task_slug": task_slug,
        "path": rel(path),
        "url": "../" + rel(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "mtime": path.stat().st_mtime,
    }


def collect_videos(eval_dir: Path) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in eval_dir.glob("*/videos/*.mp4"):
        item = parse_video(path, eval_dir)
        if item:
            by_task[item["task_key"]].append(item)
    for items in by_task.values():
        items.sort(key=lambda x: (x["success"], x["mtime"]))
    return by_task


def load_eval(summary_path: Path) -> dict[str, Any] | None:
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return None

    eval_dir = summary_path.parent
    meta = infer_metadata(eval_dir, summary)
    step = parse_step(f"{summary.get('run_id','')} {summary.get('ckpt','')} {eval_dir.name}")
    cfg = config_for_ckpt(summary.get("ckpt"))
    if not cfg and isinstance(summary.get("config"), dict):
        cfg = summary["config"]
    epoch_info = estimate_epoch(step, meta, cfg)
    manual_endpoint_step = meta.get("manual_endpoint_step")
    manual_endpoint_epoch = meta.get("manual_endpoint_epoch")
    if step is not None and manual_endpoint_step is not None and manual_endpoint_epoch is not None:
        try:
            endpoint_step = float(manual_endpoint_step)
            if endpoint_step > 0:
                epoch = float(step) / endpoint_step * float(manual_endpoint_epoch)
                epoch_info["epoch"] = round(epoch, 2)
                epoch_info["run_epoch"] = round(epoch, 2)
        except (TypeError, ValueError):
            pass
    manual_epoch = meta.get("manual_epoch")
    if manual_epoch is not None:
        try:
            epoch_info["epoch"] = round(float(manual_epoch), 2)
            epoch_info["run_epoch"] = round(float(manual_epoch), 2)
        except (TypeError, ValueError):
            pass
    suites: dict[str, dict[str, Any]] = {}
    for suite in SUITES:
        stats = (summary.get("suite_stats") or {}).get(suite) or {}
        suites[suite] = {
            "score": suite_score(stats),
            "trials": stats.get("total_trials"),
            "successes": stats.get("total_successes"),
            "time_s": stats.get("total_time"),
            "max_time_s": stats.get("max_time"),
        }

    by_video_task = collect_videos(eval_dir)
    tasks = []
    for key, item in (summary.get("task_results") or {}).items():
        suite = key.rsplit("_", 1)[0]
        task_id = int(key.rsplit("_", 1)[1]) if key.rsplit("_", 1)[-1].isdigit() else None
        videos = by_video_task.get(key, [])
        fail_videos = [v for v in videos if not v["success"]][:3]
        tasks.append(
            {
                "key": key,
                "suite": suite,
                "task_id": task_id,
                "description": item.get("task_description") or "",
                "success_rate": item.get("success_rate"),
                "successes": item.get("successes"),
                "total_episodes": item.get("total_episodes"),
                "duration_s": item.get("duration"),
                "fail_videos": fail_videos,
                "video_count": len(videos),
                "failure_video_count": len([v for v in videos if not v["success"]]),
            }
        )
    tasks.sort(key=lambda x: (x["suite"], x["task_id"] if x["task_id"] is not None else 999))

    bad_tasks = [
        t
        for t in sorted(tasks, key=lambda x: (x["success_rate"] is None, x["success_rate"] or 0.0))
        if (t["success_rate"] is not None and t["success_rate"] < 85.0) or t["failure_video_count"] > 0
    ][:16]

    return {
        "id": summary.get("run_id") or eval_dir.name,
        "eval_dir": rel(eval_dir),
        "summary_path": rel(summary_path),
        "ckpt": summary.get("ckpt"),
        "step": step,
        "epoch": epoch_info["epoch"],
        "run_epoch": epoch_info["run_epoch"],
        "resume_base_epoch": epoch_info["resume_base_epoch"],
        "steps_per_epoch": epoch_info["steps_per_epoch"],
        "resume_base_steps_per_epoch": epoch_info["resume_base_steps_per_epoch"],
        "train_sample_count": epoch_info["train_sample_count"],
        "mtime": summary_path.stat().st_mtime,
        "mtime_text": datetime.fromtimestamp(summary_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "eval_date": datetime.fromtimestamp(summary_path.stat().st_mtime).strftime("%m-%d %H:%M"),
        "overall": (summary.get("overall") or {}).get("average_success_rate"),
        "average_task_time_s": (summary.get("overall") or {}).get("average_task_time"),
        "suites": suites,
        "tasks": tasks,
        "bad_tasks": bad_tasks,
        "meta": meta,
    }


def load_manual_eval(record: dict[str, Any], index: int) -> dict[str, Any]:
    scores = record["scores"]
    step = int(record["step"])
    endpoint_step = float(record.get("endpoint_step") or step)
    endpoint_epoch = float(record.get("endpoint_epoch") or 20.0)
    epoch = round(float(step) / endpoint_step * endpoint_epoch, 2) if endpoint_step > 0 else None
    source_date = str(record.get("source_date") or "2026-06-19")
    eval_date = str(record.get("eval_date") or "06-19 AMD")
    source_note = str(
        record.get("source_note")
        or "AMD machine eval from user-provided table; suite-level only, no local summary.json/task videos."
    )
    suites = {}
    for suite in SUITES:
        score = float(scores[suite])
        suite_times = record.get("suite_times") if isinstance(record.get("suite_times"), dict) else {}
        suite_trials = record.get("suite_trials") if isinstance(record.get("suite_trials"), dict) else {}
        trials = suite_trials.get(suite, 300)
        suites[suite] = {
            "score": score,
            "trials": trials,
            "successes": int(round(score * float(trials) / 100.0)) if trials else None,
            "time_s": suite_times.get(suite),
            "max_time_s": None,
        }

    mtime = datetime(2026, 6, 19, 0, 0, tzinfo=timezone.utc).timestamp() + index
    return {
        "id": record["id"],
        "eval_dir": record.get("eval_dir", "manual/amd_user_provided_screenshot"),
        "summary_path": None,
        "ckpt": None,
        "step": step,
        "epoch": epoch,
        "run_epoch": epoch,
        "resume_base_epoch": None,
        "steps_per_epoch": round(endpoint_step / endpoint_epoch) if endpoint_epoch > 0 else None,
        "resume_base_steps_per_epoch": None,
        "train_sample_count": DEFAULT_TRAIN_SAMPLE_COUNT,
        "mtime": mtime,
        "mtime_text": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
        "eval_date": eval_date,
        "overall": float(scores["overall"]),
        "average_task_time_s": record.get("average_task_time_s"),
        "suites": suites,
        "tasks": [],
        "bad_tasks": [],
        "meta": {
            "valid": True,
            "warning": source_note,
            "resume_type": record.get("resume_type", "AMD manual eval screenshot / user-provided table"),
            "learning_rate": record.get("learning_rate", "1e-4 cosine"),
            "global_batch": record["global_batch"],
            "pooling": "none [1,1]",
            "lambda_video": record.get("lambda_video", 0.05),
            "lambda_action": record.get("lambda_action", 5.0),
            "lambda_dino_aux": record.get("lambda_dino_aux"),
            "model": record["model"],
            "wan_init": record.get("wan_init", "see variant"),
            "variant": record["variant"],
            "manual_endpoint_step": int(endpoint_step),
            "manual_endpoint_epoch": endpoint_epoch,
            "manual_epoch": 20.0 if step == int(endpoint_step) else None,
            "eval_platform": record.get("eval_platform", "AMD"),
            "source": f"user-provided table on {source_date}",
            "plus_groups": record.get("plus_groups"),
        },
    }


def chronological_order(item: dict[str, Any]) -> tuple[float, str]:
    """Sort oldest to newest so the latest evaluation appears at the bottom."""
    return (float(item.get("mtime") or 0.0), item["id"])


def collect_training_state() -> dict[str, Any] | None:
    candidates = sorted(
        (p for p in (ROOT / "runs").glob("**/config.yaml") if "wandb" not in p.parts),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    latest_cfg = candidates[-1]
    run_dir = latest_cfg.parent
    log_candidates = list(run_dir.glob("*.log")) + list(run_dir.glob("output.log"))
    latest_log = max(log_candidates, key=lambda p: p.stat().st_mtime, default=None)
    latest_weight = max((run_dir / "checkpoints" / "weights").glob("step_*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    state = {
        "run_dir": rel(run_dir),
        "latest_weight": rel(latest_weight) if latest_weight else None,
        "latest_weight_step": parse_step(latest_weight.name) if latest_weight else None,
        "log": rel(latest_log) if latest_log else None,
        "last_line": "",
    }
    if latest_log:
        try:
            lines = latest_log.read_text(errors="ignore").splitlines()
            interesting = [line.strip() for line in lines if "loss=" in line or "Starting training" in line or "Loaded .pt weights" in line]
            state["last_line"] = interesting[-1] if interesting else lines[-1].strip() if lines else ""
        except Exception:
            pass
    return state


def build() -> dict[str, Any]:
    evals = []
    for summary_path in sorted((ROOT / "evaluate_results").glob("**/summary.json")):
        item = load_eval(summary_path)
        if item:
            evals.append(item)
    for index, record in enumerate(MANUAL_EVALS):
        evals.append(load_manual_eval(record, index))
    evals.sort(key=chronological_order)

    valid = [e for e in evals if e["meta"].get("valid", True)]
    best = max(valid, key=lambda x: x["overall"] or -1, default=None)
    latest = max(evals, key=lambda x: x["mtime"], default=None)
    latest_valid = max(valid, key=lambda x: x["mtime"], default=None)

    task_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in valid:
        for task in e["tasks"]:
            task_history[task["key"]].append(
                {
                    "eval_id": e["id"],
                    "step": e["step"],
                    "score": task["success_rate"],
                    "description": task["description"],
                    "suite": task["suite"],
                }
            )

    movers = []
    if len(valid) >= 2:
        previous, current = valid[-2], valid[-1]
        prev_tasks = {t["key"]: t for t in previous["tasks"]}
        for task in current["tasks"]:
            old = prev_tasks.get(task["key"])
            if not old or old.get("success_rate") is None or task.get("success_rate") is None:
                continue
            delta = task["success_rate"] - old["success_rate"]
            if abs(delta) >= 6.0:
                movers.append(
                    {
                        "key": task["key"],
                        "suite": task["suite"],
                        "description": task["description"],
                        "from_eval": previous["id"],
                        "to_eval": current["id"],
                        "from_score": old["success_rate"],
                        "to_score": task["success_rate"],
                        "delta": delta,
                    }
                )
        movers.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(ROOT),
        "suites": SUITES,
        "evals": evals,
        "best_eval_id": best["id"] if best else None,
        "latest_eval_id": latest["id"] if latest else None,
        "latest_valid_eval_id": latest_valid["id"] if latest_valid else None,
        "movers": movers[:24],
        "training_state": collect_training_state(),
    }


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT} with {len(data['evals'])} evaluations")


if __name__ == "__main__":
    main()
