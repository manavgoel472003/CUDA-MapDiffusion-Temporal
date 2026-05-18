import os
import sys
import inspect
import numpy as np
import torch
import onnx
from onnx import numpy_helper

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
ONNX = "model/mapdiffusion_routeA/mapdiffusion.head.onnx"
PARITY_DIR = "model/mapdiffusion_routeA/parity_head"

torch.set_grad_enabled(False)

bev = torch.from_numpy(np.load(os.path.join(PARITY_DIR, "bev_features.npy"))).float().cuda()
query_coords = torch.from_numpy(np.load(os.path.join(PARITY_DIR, "query_coords.npy"))).float().cuda()
timestep = torch.from_numpy(np.load(os.path.join(PARITY_DIR, "timestep.npy"))).float().cuda()

cfg = Config.fromfile(CFG)

dataset = build_dataset(cfg.data.val)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
batch = next(iter(loader))
img_metas = batch["img_metas"].data[0]

model = build_model(cfg.model)
load_checkpoint(model, CKPT, map_location="cpu", strict=False)
model.eval()

# Overwrite model weights with exact ONNX initializers where names match.
onnx_model = onnx.load(ONNX)
init = {x.name: numpy_helper.to_array(x) for x in onnx_model.graph.initializer}

sd = model.state_dict()
replaced = 0
for name, arr in init.items():
    if name in sd and tuple(sd[name].shape) == tuple(arr.shape):
        sd[name] = torch.from_numpy(arr).to(dtype=sd[name].dtype)
        replaced += 1
    elif "head." + name in sd and tuple(sd["head." + name].shape) == tuple(arr.shape):
        sd["head." + name] = torch.from_numpy(arr).to(dtype=sd["head." + name].dtype)
        replaced += 1

model.load_state_dict(sd, strict=False)
model.cuda()
model.eval()

print("loaded ONNX initializers into PyTorch:", replaced, "/", len(init))

head = model.head
print("head:", type(head))
print("forward_test:", inspect.signature(head.forward_test))

with torch.no_grad():
    out = head.forward_test(query_coords, timestep, bev, img_metas)

final = out[-1]
line = final["lines"][0]
score = final["scores"][0]

if line.dim() == 2:
    line = line.unsqueeze(0)
if score.dim() == 2:
    score = score.unsqueeze(0)

line_np = line.detach().float().cpu().numpy()
score_np = score.detach().float().cpu().numpy()

print("onnx-weight line:", line_np.shape, line_np.min(), line_np.max())
print("onnx-weight score:", score_np.shape, score_np.min(), score_np.max())

np.save(os.path.join(PARITY_DIR, "pytorch_onnx_weight_line_preds.npy"), line_np)
np.save(os.path.join(PARITY_DIR, "pytorch_onnx_weight_cls_logits.npy"), score_np)

print("saved:", PARITY_DIR)
