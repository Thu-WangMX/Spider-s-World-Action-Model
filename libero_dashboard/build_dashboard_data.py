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


MANUAL_NOTES: list[tuple[str, dict[str, Any]]] = [
    (
        "viewpatch_1x1x2_lr2e-5_30trials_step_020000",
        {
            "valid": True,
            "resume_type": "weight-only resume from viewpatch_1x1x2_mmap_bs12_w6_30trials_step_024000",
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
    candidates = sorted((ROOT / "runs").glob("**/config.yaml"), key=lambda p: p.stat().st_mtime)
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
