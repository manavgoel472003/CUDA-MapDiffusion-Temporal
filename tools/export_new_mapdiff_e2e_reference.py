import sys
from pathlib import Path
import numpy as np
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

CUDA_BEV_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")
sys.path.insert(0, str(MAPDIFF_ROOT))

import plugin  # noqa: F401


CFG = "/home/018198687/Mapping/mapdiffusion/plugin/configs/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"

IN_DIR = CUDA_BEV_ROOT / "model/cuda_bevformer/e2e_pipeline_inmemory"
OUT_DIR = CUDA_BEV_ROOT / "model/cuda_bevformer/e2e_parity_new_mapdiff"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stat(name, x):
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    print(name, arr.shape, arr.dtype, float(arr.min()), float(arr.max()), float(arr.mean()))


def build_new_mapdiff():
    cfg = Config.fromfile(CFG)
    cfg.model.pretrained = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()
    return model


def extract_img_feats(model, img):
    """
    Recreate Engine A: img [1,6,3,480,800] -> FPN features.
    """
    bb = model.backbone
    B, N, C, H, W = img.shape

    x = img.reshape(B * N, C, H, W)

    feats = bb.img_backbone(x)
    if isinstance(feats, dict):
        feats = list(feats.values())

    feats = bb.img_neck(feats)
    feats = list(feats)

    outs = []
    for f in feats:
        outs.append(f.reshape(B, N, f.shape[1], f.shape[2], f.shape[3]).contiguous())

    return outs


def make_img_metas(ego2img):
    e = ego2img[0].astype(np.float32)
    return [{
        "ego2img": e,
        "lidar2img": e,
        "img_shape": [(480, 800, 3)] * 6,
        "ori_shape": [(480, 800, 3)] * 6,
        "pad_shape": [(480, 800, 3)] * 6,
        "scale_factor": 1.0,
        "flip": False,
    }]


def extract_bev_features(model, img, ego2img):
    """
    Prefer using the full backbone path so it matches new_mapdiff.
    If your backbone has a dedicated BEV method, this will print useful method names if it fails.
    """
    bb = model.backbone
    img_metas = make_img_metas(ego2img)

    # Try common BEVFormer-style forward signatures.
    attempts = [
        lambda: bb(img=img, img_metas=img_metas),
        lambda: bb.forward(img=img, img_metas=img_metas),
        lambda: bb.forward(img, img_metas),
    ]

    last_err = None
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, (list, tuple)):
                # Find tensor shaped like BEV.
                for x in out:
                    if isinstance(x, torch.Tensor) and x.numel() == 1 * 256 * 50 * 100:
                        return x.reshape(1, 256, 50, 100).contiguous()
                # Otherwise try first tensor.
                for x in out:
                    if isinstance(x, torch.Tensor):
                        return x
            if isinstance(out, torch.Tensor):
                if out.shape == (1, 50 * 100, 256):
                    return out.reshape(1, 50, 100, 256).permute(0, 3, 1, 2).contiguous()
                if out.shape == (1, 256, 50, 100):
                    return out.contiguous()
                if out.numel() == 1 * 256 * 50 * 100:
                    return out.reshape(1, 256, 50, 100).contiguous()
        except Exception as e:
            last_err = e

    print("Could not run backbone forward directly.")
    print("Last error:", repr(last_err))
    print("Backbone type:", type(bb))
    print("Useful attrs/methods:")
    for name in dir(bb):
        low = name.lower()
        if "bev" in low or "feat" in low or "forward" in low or "transformer" in low:
            print(" ", name)
    raise RuntimeError("BEV feature extraction failed")


def unwrap_head_tensor(x, name):
    # MapDiffusion often returns per-batch lists:
    #   lines:  [Tensor(100,40)]
    #   scores: [Tensor(100,3)]
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise RuntimeError(f"{name} is empty list/tuple")
        x = x[0]

    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{name} is {type(x)}, expected Tensor after unwrap")

    if x.dim() == 2:
        x = x.unsqueeze(0)

    return x.contiguous()


def run_head(model, query_coords, timestep, bev_features):
    head = model.head
    img_metas = [{}]

    out = head.forward_test(
        query_coords=query_coords,
        timestep=timestep,
        bev_features=bev_features,
        img_metas=img_metas,
    )

    # MapDiffusion forward_test returns list of decoder-layer dicts.
    final = out[-1] if isinstance(out, list) else out

    print("final head output keys:", list(final.keys()))
    for k, v in final.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k}: {type(v)} len={len(v)}")
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                print(f"    first: Tensor {tuple(v[0].shape)} {v[0].dtype}")
        else:
            print(f"  {k}: {type(v)}")

    line = final.get("lines", None)
    score = final.get("scores", None)

    if line is None or score is None:
        raise RuntimeError(f"Could not find lines/scores in final head output keys={final.keys()}")

    line = unwrap_head_tensor(line, "lines")
    score = unwrap_head_tensor(score, "scores")

    return line, score


def main():
    img = np.load(IN_DIR / "input_img.npy").astype(np.float32)
    ego2img = np.load(IN_DIR / "input_ego2img.npy").astype(np.float32)
    query_coords = np.load(IN_DIR / "input_query_coords.npy").astype(np.float32)
    timestep = np.load(IN_DIR / "input_timestep.npy").astype(np.float32)

    print("Loaded exact TRT inputs:")
    print("img", img.shape, img.dtype, img.min(), img.max())
    print("ego2img", ego2img.shape, ego2img.dtype, ego2img.min(), ego2img.max())
    print("query_coords", query_coords.shape, query_coords.dtype, query_coords.min(), query_coords.max())
    print("timestep", timestep.shape, timestep.dtype, timestep)

    model = build_new_mapdiff()

    img_t = torch.from_numpy(img).cuda()
    query_t = torch.from_numpy(query_coords).cuda()
    timestep_t = torch.from_numpy(timestep).cuda()

    with torch.no_grad():
        feats = extract_img_feats(model, img_t)
        bev = extract_bev_features(model, img_t, ego2img)
        line, score = run_head(model, query_t, timestep_t, bev)

    names = ["feat0", "feat1", "feat2"]
    for name, f in zip(names, feats[:3]):
        arr = f.detach().cpu().numpy().astype(np.float32)
        np.save(OUT_DIR / f"pt_{name}.npy", arr)
        stat("pt_" + name, arr)

    bev_arr = bev.detach().cpu().numpy().astype(np.float32)
    line_arr = line.detach().cpu().numpy().astype(np.float32)
    score_arr = score.detach().cpu().numpy().astype(np.float32)

    np.save(OUT_DIR / "pt_bev_features.npy", bev_arr)
    np.save(OUT_DIR / "pt_line_preds.npy", line_arr)
    np.save(OUT_DIR / "pt_cls_logits.npy", score_arr)

    stat("pt_bev_features", bev_arr)
    stat("pt_line_preds", line_arr)
    stat("pt_cls_logits", score_arr)

    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
