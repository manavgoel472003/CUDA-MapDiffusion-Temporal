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

dataset = build_dataset(cfg.data.val)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
batch = next(iter(loader))
img_metas = batch["img_metas"].data[0]

model = build_model(cfg.model)
load_checkpoint(model, CKPT, map_location="cpu", strict=False)
model.cuda()
model.eval()

head = model.head
print("head:", type(head))
print("forward_test:", inspect.signature(head.forward_test))

with torch.no_grad():
    out = head.forward_test(query_coords, timestep, bev, img_metas)

print("num decoder outputs:", len(out))

for i, d in enumerate(out):
    print("=" * 80)
    print("layer", i, "keys:", list(d.keys()))

    line = d["lines"][0]
    score = d["scores"][0]

    if line.dim() == 2:
        line = line.unsqueeze(0)
    if score.dim() == 2:
        score = score.unsqueeze(0)

    line_np = line.detach().float().cpu().numpy()
    score_np = score.detach().float().cpu().numpy()

    print(" line:", line_np.shape, line_np.min(), line_np.max())
    print(" score:", score_np.shape, score_np.min(), score_np.max())

    np.save(os.path.join(PARITY_DIR, f"pytorch_layer{i}_line_preds.npy"), line_np)
    np.save(os.path.join(PARITY_DIR, f"pytorch_layer{i}_cls_logits.npy"), score_np)

print("saved all layers:", PARITY_DIR)
