import os
import sys
import importlib
from pathlib import Path

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")
CBF_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
sys.path.insert(0, str(MAPDIFF_ROOT))

CFG = os.environ.get(
    "TEMPORAL_CONFIG",
    str(CBF_ROOT / "model/mapdiffusion_temporal_routeB/temporal_config.py"),
)
CKPT = os.environ.get(
    "TEMPORAL_CKPT",
    "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth",
)
ONNX_OUT = Path(os.environ.get(
    "ONNX_OUT",
    str(CBF_ROOT / "model/mapdiffusion_temporal_routeB/onnx/stream_fusion_neck.temporal87000.fp32.onnx"),
))

cfg = Config.fromfile(CFG)

if getattr(cfg, "plugin", False):
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))

if hasattr(cfg, "custom_imports"):
    for imp in cfg.custom_imports.get("imports", []):
        importlib.import_module(imp)

model = build_model(cfg.model, test_cfg=cfg.get("test_cfg", None))
load_checkpoint(model, CKPT, map_location="cpu", strict=False)
model.cuda().eval()

print("model:", type(model))
print("streaming_bev:", getattr(model, "streaming_bev", None))
print("stream_fusion_neck:", model.stream_fusion_neck)

class StreamFusionWrapper(nn.Module):
    def __init__(self, neck):
        super().__init__()
        self.neck = neck

    def forward(self, prev_bev, curr_bev):
        out = self.neck(prev_bev, curr_bev)
        if out.dim() == 3:
            out = out.unsqueeze(0)
        return out.contiguous()

wrapper = StreamFusionWrapper(model.stream_fusion_neck).cuda().eval()

prev_bev = torch.randn(1, 256, 50, 100, dtype=torch.float32, device="cuda")
curr_bev = torch.randn(1, 256, 50, 100, dtype=torch.float32, device="cuda")

with torch.no_grad():
    fused = wrapper(prev_bev, curr_bev)

print("fused:", tuple(fused.shape), float(fused.min()), float(fused.max()), float(fused.mean()))

ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    wrapper,
    (prev_bev, curr_bev),
    str(ONNX_OUT),
    input_names=["prev_bev", "curr_bev"],
    output_names=["fused_bev"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes=None,
)

print("saved:", ONNX_OUT)
