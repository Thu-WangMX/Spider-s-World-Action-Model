#!/usr/bin/env python
"""Convert per-frame DINO .pt cache files into one mmap-friendly payload.

The original ``frame`` cache stores one ``frames/00000000.pt`` file per global
frame.  That is space-efficient but punishes DataLoader workers with thousands
of tiny ``torch.load`` calls per step.  This converter keeps the same one-frame
storage semantics while packing all frames into a contiguous binary file that
workers can read with ``np.memmap``.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def _atomic_json_save(payload: dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)


def _load_frame(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    latent = payload.get("dino_latent", payload.get("latent")) if isinstance(payload, dict) else payload
    if not torch.is_tensor(latent):
        raise TypeError(f"Expected tensor DINO latent in {path}, got {type(latent)}")
    if latent.ndim != 3:
        raise ValueError(f"Expected frame latent [D,H,W] in {path}, got {tuple(latent.shape)}")
    return latent.contiguous()


def _dtype_info(dtype: torch.dtype) -> tuple[str, np.dtype, str]:
    if dtype is torch.bfloat16:
        return "bf16", np.dtype("uint16"), "uint16"
    if dtype is torch.float16:
        return "fp16", np.dtype("float16"), "float16"
    if dtype is torch.float32:
        return "fp32", np.dtype("float32"), "float32"
    raise ValueError(f"Unsupported DINO frame dtype for mmap conversion: {dtype}")


def _tensor_to_storage_numpy(tensor: torch.Tensor, save_dtype: str) -> np.ndarray:
    if save_dtype == "bf16":
        return tensor.view(torch.uint16).numpy()
    return tensor.numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Existing frame cache directory.")
    parser.add_argument("--dst", required=True, type=Path, help="Output frame_mmap cache directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output payload.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Optional frame count override. Defaults to metadata total_samples or files in src/frames.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=8192,
        help="Flush mmap writes every N frames.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    frame_dir = src / "frames"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Missing source frame cache directory: {frame_dir}")

    src_meta_path = src / "metadata.json"
    src_meta: dict[str, Any] = {}
    if src_meta_path.exists():
        with src_meta_path.open("r", encoding="utf-8") as f:
            src_meta = json.load(f)

    if args.num_frames is not None:
        total_frames = int(args.num_frames)
    elif "total_samples" in src_meta:
        total_frames = int(src_meta["total_samples"])
    else:
        total_frames = len(list(frame_dir.glob("*.pt")))
    if total_frames <= 0:
        raise ValueError(f"Invalid total frame count: {total_frames}")

    first_path = frame_dir / "00000000.pt"
    if not first_path.exists():
        first_paths = sorted(frame_dir.glob("*.pt"))
        if not first_paths:
            raise FileNotFoundError(f"No frame cache files found in {frame_dir}")
        first_path = first_paths[0]
    first_latent = _load_frame(first_path)
    save_dtype, np_dtype, storage_dtype = _dtype_info(first_latent.dtype)
    latent_shape = tuple(int(v) for v in first_latent.shape)

    dst.mkdir(parents=True, exist_ok=True)
    mmap_name = f"frames.{save_dtype}.bin"
    mmap_path = dst / mmap_name
    tmp_mmap_path = dst / f".{mmap_name}.tmp"
    metadata_path = dst / "metadata.json"
    if (mmap_path.exists() or metadata_path.exists() or tmp_mmap_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists in {dst}. Pass --overwrite to replace {mmap_path.name}/metadata.json."
        )
    for path in (mmap_path, metadata_path, tmp_mmap_path):
        if path.exists():
            path.unlink()

    mmap = np.memmap(tmp_mmap_path, mode="w+", dtype=np_dtype, shape=(total_frames, *latent_shape))
    for idx in tqdm(range(total_frames), desc="Packing DINO frame cache", unit="frame", dynamic_ncols=True):
        frame_path = frame_dir / f"{idx:08d}.pt"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing source frame cache file: {frame_path}")
        latent = _load_frame(frame_path)
        if tuple(latent.shape) != latent_shape:
            raise ValueError(
                f"Shape mismatch in {frame_path}: expected {latent_shape}, got {tuple(latent.shape)}"
            )
        if latent.dtype != first_latent.dtype:
            raise ValueError(
                f"Dtype mismatch in {frame_path}: expected {first_latent.dtype}, got {latent.dtype}"
            )
        mmap[idx] = _tensor_to_storage_numpy(latent, save_dtype)
        if args.flush_every > 0 and (idx + 1) % args.flush_every == 0:
            mmap.flush()

    mmap.flush()
    del mmap
    os.replace(tmp_mmap_path, mmap_path)

    metadata = dict(src_meta)
    metadata.update(
        {
            "cache_mode": "frame_mmap",
            "source_cache_mode": src_meta.get("cache_mode", "frame"),
            "source_cache_dir": str(src),
            "mmap_file": mmap_name,
            "total_frames": total_frames,
            "latent_shape": list(latent_shape),
            "save_dtype": save_dtype,
            "storage_dtype": storage_dtype,
            "format_version": 1,
        }
    )
    _atomic_json_save(metadata, metadata_path)
    size_gib = mmap_path.stat().st_size / (1024**3)
    print(
        f"frame_mmap cache written: {dst}\n"
        f"  frames={total_frames} shape={latent_shape} save_dtype={save_dtype} "
        f"storage={storage_dtype} size={size_gib:.2f} GiB"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
