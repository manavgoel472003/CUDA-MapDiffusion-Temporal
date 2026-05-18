import sys
import numpy as np
import torch
import onnx
from onnx import numpy_helper

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
if MAPDIFF_ROOT not in sys.path:
    sys.path.insert(0, MAPDIFF_ROOT)

import plugin

CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"
ONNX = "model/mapdiffusion_routeA/mapdiffusion.head.onnx"

cfg = Config.fromfile(CFG)
model = build_model(cfg.model)
load_checkpoint(model, CKPT, map_location="cpu", strict=False)
model.eval()

sd = model.state_dict()

m = onnx.load(ONNX)
init = {x.name: numpy_helper.to_array(x) for x in m.graph.initializer}

print("num onnx initializers:", len(init))
print("num pytorch state_dict:", len(sd))

matched = []
missing = []
shape_mismatch = []

for name, arr in init.items():
    # ONNX may have exact PyTorch names.
    candidates = [
        name,
        "head." + name,
        name.replace("head.", ""),
    ]

    found_key = None
    for k in candidates:
        if k in sd:
            found_key = k
            break

    if found_key is None:
        missing.append(name)
        continue

    pt = sd[found_key].detach().cpu().numpy()

    if pt.shape != arr.shape:
        # Linear weights may sometimes be transposed, but first report it.
        shape_mismatch.append((name, found_key, arr.shape, pt.shape))
        continue

    diff = arr.astype(np.float32) - pt.astype(np.float32)
    matched.append((
        name,
        found_key,
        float(np.max(np.abs(diff))),
        float(np.mean(np.abs(diff))),
        arr.shape,
    ))

matched_sorted = sorted(matched, key=lambda x: x[2], reverse=True)

print("\nMatched initializers:", len(matched))
print("Missing from PyTorch:", len(missing))
print("Shape mismatch:", len(shape_mismatch))

print("\nTop matched diffs:")
for row in matched_sorted[:40]:
    print(row)

print("\nFirst 80 missing initializer names:")
for x in missing[:80]:
    print(" ", x)

print("\nShape mismatches:")
for row in shape_mismatch[:40]:
    print(row)
