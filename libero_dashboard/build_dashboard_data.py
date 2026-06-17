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
