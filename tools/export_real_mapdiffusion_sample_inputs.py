import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")

# Make sure the original MapDiffusion plugin is importable.
sys.path.insert(0, str(MAPDIFF_ROOT))

import plugin  # noqa: F401


def unwrap(x):
    """Unwrap MMCV DataContainer / list wrappers."""
    if hasattr(x, "data"):
        x = x.data
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def as_numpy(x):
    x = unwrap(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def find_img(data):
    for key in ["img", "imgs", "image"]:
        if key in data:
            x = unwrap(data[key])
            if isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)

            # Common shapes:
            # [6, 3, 480, 800] or [1, 6, 3, 480, 800]
            if arr.shape == (6, 3, 480, 800):
                arr = arr[None]
            if arr.shape != (1, 6, 3, 480, 800):
                raise RuntimeError(f"Found {key}, but shape is {arr.shape}, expected [1,6,3,480,800]")
            return arr.astype(np.float32)

    raise RuntimeError(f"Could not find image tensor. Available keys: {list(data.keys())}")


def find_meta(data):
    for key in ["img_metas", "img_meta", "metas", "meta"]:
        if key in data:
            meta = unwrap(data[key])
            if isinstance(meta, (list, tuple)) and len(meta) == 1:
                meta = meta[0]
            return meta
    raise RuntimeError(f"Could not find img_metas. Available keys: {list(data.keys())}")


def find_ego2img(meta):
    # Print keys to help debug if needed.
    print("meta keys:", list(meta.keys()))

    candidates = [
        "ego2img",
        "lidar2img",
        "cam2img",
        "ego2img_aug",
        "img_aug_matrix",
    ]

    for key in candidates:
        if key in meta:
            arr = np.asarray(meta[key], dtype=np.float32)
            if arr.shape == (6, 4, 4):
                return arr[None].astype(np.float32), key
            if arr.shape == (1, 6, 4, 4):
                return arr.astype(np.float32), key

    # Sometimes stored per camera in a list.
    for key, val in meta.items():
        try:
            arr = np.asarray(val, dtype=np.float32)
        except Exception:
            continue

        if arr.shape == (6, 4, 4):
            return arr[None].astype(np.float32), key
        if arr.shape == (1, 6, 4, 4):
            return arr.astype(np.float32), key

    raise RuntimeError("Could not find ego2img/lidar2img-like [6,4,4] matrix in img_metas")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/018198687/Mapping/mapdiffusion/plugin/configs/new_mapdiff.py",
    )
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default="/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/cuda_bevformer/real_sample_inputs",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # Use validation set for realistic eval sample.
    dataset_cfg = cfg.data.val
    dataset_cfg.test_mode = True

    dataset = build_dataset(dataset_cfg)
    print("dataset length:", len(dataset))

    data = dataset[args.idx]
    print("sample keys:", list(data.keys()))

    img = find_img(data)
    meta = find_meta(data)
    ego2img, ego_key = find_ego2img(meta)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img.tofile(out / "img.fp32.bin")
    ego2img.tofile(out / "ego2img.fp32.bin")
    np.save(out / "img.npy", img)
    np.save(out / "ego2img.npy", ego2img)

    print("=" * 80)
    print("saved real sample inputs:", out)
    print("idx:", args.idx)
    print("img:", img.shape, img.dtype, float(img.min()), float(img.max()), float(img.mean()))
    print(f"ego2img from meta['{ego_key}']:", ego2img.shape, ego2img.dtype, float(ego2img.min()), float(ego2img.max()))
    print("=" * 80)


if __name__ == "__main__":
    main()
