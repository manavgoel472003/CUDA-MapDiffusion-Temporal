import os
import sys
import inspect
import numpy as np
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.datasets.builder import build_dataloader
from mmdet3d.models import build_model

MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
if MAPDIFF_ROOT not in sys.path:
    sys.path.insert(0, MAPDIFF_ROOT)

import plugin

CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"
PARITY_DIR = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_routeA/parity_head"

torch.set_grad_enabled(False)

bev = torch.from_numpy(np.load(os.path.join(PARITY_DIR, "bev_features.npy"))).float().cuda()
query_coords = torch.from_numpy(np.load(os.path.join(PARITY_DIR, "query_coords.npy"))).float().cuda()
timestep = torch.from_numpy(np.load(os.path.join(PARITY_DIR, "timestep.npy"))).float().cuda()

cfg = Config.fromfile(CFG)

# Build one val sample just to get valid img_metas.
dataset = build_dataset(cfg.data.val)
loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False,
)
batch = next(iter(loader))
img_metas = batch["img_metas"].data[0]

model = build_model(cfg.model)
load_checkpoint(model, CKPT, map_location="cpu", strict=False)
model.cuda()
model.eval()

print("bev:", tuple(bev.shape), bev.dtype, float(bev.min()), float(bev.max()))
print("query_coords:", tuple(query_coords.shape), query_coords.dtype, float(query_coords.min()), float(query_coords.max()))
print("timestep:", tuple(timestep.shape), timestep.dtype, timestep.detach().cpu().numpy())

print("\n==== Finding forward_test head ====")

head = None
head_name = None

for name, module in model.named_modules():
    if hasattr(module, "forward_test"):
        try:
            sig = inspect.signature(module.forward_test)
            s = str(sig)
        except Exception:
            s = ""
        if "query_coords" in s and "timestep" in s and "bev_features" in s:
            head = module
            head_name = name
            print("selected:", name, type(module), s)
            break

if head is None:
    print("No exact forward_test match found. Candidates:")
    for name, module in model.named_modules():
        if hasattr(module, "forward_test"):
            try:
                print(name, type(module), inspect.signature(module.forward_test))
            except Exception as e:
                print(name, type(module), e)
    raise SystemExit(2)

with torch.no_grad():
    out = head.forward_test(query_coords, timestep, bev, img_metas)

print("raw output type:", type(out))

if isinstance(out, dict):
    print("dict keys:", out.keys())
    cls = out.get("cls_logits", None)
    line = out.get("line_preds", None)

    # fallback names
    if cls is None:
        for k in ["cls", "cls_scores", "all_cls_scores", "pred_logits"]:
            if k in out:
                cls = out[k]
                print("using cls key:", k)
                break

    if line is None:
        for k in ["line", "lines", "all_line_preds", "pred_lines", "pts_preds"]:
            if k in out:
                line = out[k]
                print("using line key:", k)
                break

elif isinstance(out, (tuple, list)):
    print("tuple/list len:", len(out))

    for i, x in enumerate(out):
        print(f" out[{i}]:", type(x))
        if isinstance(x, dict):
            print("  keys:", list(x.keys()))
            for k, v in x.items():
                if torch.is_tensor(v):
                    print(f"   {k}:", tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
                elif isinstance(v, (list, tuple)):
                    print(f"   {k}: list/tuple len={len(v)}")
                    for j, vv in enumerate(v[:5]):
                        if torch.is_tensor(vv):
                            print(f"     [{j}]:", tuple(vv.shape), vv.dtype, float(vv.min()), float(vv.max()))
                        else:
                            print(f"     [{j}]:", type(vv))
                else:
                    print(f"   {k}:", type(v))
        elif torch.is_tensor(x):
            print(f"  tensor:", tuple(x.shape), x.dtype, float(x.min()), float(x.max()))

    # MapDiffusion forward_test returns one dict per decoder layer.
    # Use final decoder layer for ONNX parity.
    final = out[-1]

    if not isinstance(final, dict):
        raise RuntimeError(f"Expected final output to be dict, got {type(final)}")

    cls = None
    line = None

    def collect_tensors(obj, prefix=""):
        found = []
        if torch.is_tensor(obj):
            found.append((prefix, obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                found.extend(collect_tensors(v, prefix + "." + str(k) if prefix else str(k)))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                found.extend(collect_tensors(v, prefix + f"[{i}]"))
        return found

    tensors = collect_tensors(final)

    print("\nFinal decoder tensor candidates:")
    for name, x in tensors:
        print(" ", name, tuple(x.shape), x.dtype, float(x.min()), float(x.max()))

    for name, x in tensors:
        if len(x.shape) >= 2 and x.shape[-1] == 3:
            cls = x
            print("selected cls:", name, tuple(x.shape))
        elif len(x.shape) >= 2 and x.shape[-1] == 40:
            line = x
            print("selected line:", name, tuple(x.shape))

    # forward_test returns per-sample outputs as [100,3] and [100,40].
    # ONNX/TRT outputs are batched: [1,100,3] and [1,100,40].
    if cls is not None and cls.dim() == 2:
        cls = cls.unsqueeze(0)
    if line is not None and line.dim() == 2:
        line = line.unsqueeze(0)
else:
    raise RuntimeError(f"Unexpected output type: {type(out)}")

if cls is None or line is None:
    raise RuntimeError("Could not identify cls_logits and line_preds")

cls = cls.detach().float().cpu().numpy()
line = line.detach().float().cpu().numpy()

print("cls:", cls.shape, cls.min(), cls.max())
print("line:", line.shape, line.min(), line.max())

np.save(os.path.join(PARITY_DIR, "pytorch_cls_logits.npy"), cls)
np.save(os.path.join(PARITY_DIR, "pytorch_line_preds.npy"), line)

print("saved pytorch outputs:", PARITY_DIR)
