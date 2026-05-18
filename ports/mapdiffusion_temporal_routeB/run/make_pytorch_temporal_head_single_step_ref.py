import os
import sys
import importlib
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
import importlib.util

EXPORTER_PATH = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/tools/temporal_routeB/export_temporal_head_onnx.py"
_spec = importlib.util.spec_from_file_location("export_temporal_head_onnx", EXPORTER_PATH)
_export_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_export_mod)
TemporalHeadExportWrapper = _export_mod.TemporalHeadExportWrapper
ManualTemporalQueryFusion = _export_mod.ManualTemporalQueryFusion

MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
CONFIG = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/temporal_config.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth"

TRACE_DIR = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/pytorch_trace_sample0")
OUT_DIR = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/parity_single_step_c")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, MAPDIFF_ROOT)


def import_plugin(cfg):
    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir
        if not isinstance(plugin_dirs, list):
            plugin_dirs = [plugin_dirs]
        for plugin_dir in plugin_dirs:
            module_path = os.path.dirname(plugin_dir).replace("/", ".")
            print("[plugin]", module_path)
            importlib.import_module(module_path)

    if hasattr(cfg, "custom_imports"):
        for imp in cfg.custom_imports.get("imports", []):
            print("[custom_import]", imp)
            importlib.import_module(imp)


def stat(name, x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    print(name, x.shape, x.dtype, float(x.min()), float(x.max()), float(x.mean()))


def main():
    cfg = Config.fromfile(CONFIG)
    import_plugin(cfg)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()

    head = model.head
    print("model:", type(model))
    print("head:", type(head))
    print("has temporal_query_fusion:", hasattr(head, "temporal_query_fusion"))

    # Match the ONNX export path exactly.
    if hasattr(model.head, "temporal_query_fusion"):
        original_tq = model.head.temporal_query_fusion
        model.head.temporal_query_fusion = ManualTemporalQueryFusion(original_tq).cuda().eval()
        print("replaced temporal_query_fusion with ManualTemporalQueryFusion")
        head = model.head

    bev = torch.from_numpy(np.load(TRACE_DIR / "bev_features.npy")).float().cuda()
    query = torch.from_numpy(np.load(TRACE_DIR / "step_00_query_coords.npy")).float().cuda()
    timestep = torch.tensor([1000.0], dtype=torch.float32, device="cuda")
    prev_query_feat = torch.zeros((1, 100, 512), dtype=torch.float32, device="cuda")
    prev_query_valid = torch.zeros((1, 100), dtype=torch.float32, device="cuda")

    wrapper = TemporalHeadExportWrapper(model.head).cuda().eval()

    with torch.no_grad():
        out = wrapper(
            bev_features=bev,
            query_coords=query,
            timestep=timestep,
            prev_query_feat=prev_query_feat,
            prev_query_valid=prev_query_valid,
        )

    print("raw output type:", type(out))

    if isinstance(out, dict):
        line_preds = out["line_preds"]
        cls_logits = out["cls_logits"]
        query_feat = out["query_feat"]
    elif isinstance(out, (list, tuple)):
        print("tuple/list len:", len(out))
        for i, v in enumerate(out):
            stat(f"out[{i}]", v)

        # Exported TRT binding order is:
        # OUT 4: query_feat
        # OUT 5: line_preds
        # OUT 6: cls_logits
        # But the PyTorch wrapper may return either order, so infer by shape.
        line_preds = None
        cls_logits = None
        query_feat = None
        for v in out:
            shape = tuple(v.shape)
            if shape == (1, 100, 40):
                line_preds = v
            elif shape == (1, 100, 3):
                cls_logits = v
            elif shape == (1, 100, 512):
                query_feat = v

        if line_preds is None or cls_logits is None or query_feat is None:
            raise RuntimeError("Could not infer wrapper outputs by shape")
    else:
        raise RuntimeError(f"Unexpected wrapper output type: {type(out)}")

    stat("pt_line_preds", line_preds)
    stat("pt_cls_logits", cls_logits)
    stat("pt_query_feat", query_feat)

    np.save(OUT_DIR / "pt_line_preds.npy", line_preds.detach().cpu().numpy())
    np.save(OUT_DIR / "pt_cls_logits.npy", cls_logits.detach().cpu().numpy())
    np.save(OUT_DIR / "pt_query_feat.npy", query_feat.detach().cpu().numpy())

    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
