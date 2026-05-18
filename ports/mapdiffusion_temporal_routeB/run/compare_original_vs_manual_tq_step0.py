import os
import sys
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
CONFIG = ROOT / "model/mapdiffusion_temporal_routeB/temporal_config.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth"
TRACE = ROOT / "model/mapdiffusion_temporal_routeB/pt_sampler_trace_sample0"

EXPORTER_PATH = ROOT / "tools/temporal_routeB/export_temporal_head_onnx.py"
spec = importlib.util.spec_from_file_location("export_temporal_head_onnx", str(EXPORTER_PATH))
export_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_mod)
ManualTemporalQueryFusion = export_mod.ManualTemporalQueryFusion

sys.path.insert(0, MAPDIFF_ROOT)


def import_plugins(cfg):
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
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    print(name, x.shape, float(x.min()), float(x.max()), float(x.mean()))


def run_head(head, bev, query, timestep, prev_query_feat, prev_query_valid):
    with torch.no_grad():
        preds = head(
            query,
            int(timestep.item()),
            bev,
            img_metas=[{}],
            prev_query_feat=prev_query_feat,
            prev_query_valid=prev_query_valid,
            return_loss=False,
        )
    last = preds[-1]
    line = torch.stack(last["lines"], dim=0).view(1, 100, 20, 2)
    cls = torch.stack(last["scores"], dim=0)
    qfeat = last["query_feat"]
    return line, cls, qfeat


def compare(name, a, b):
    a = a.detach().cpu().numpy().astype(np.float32)
    b = b.detach().cpu().numpy().astype(np.float32)
    diff = a - b
    cos = np.dot(a.reshape(-1), b.reshape(-1)) / (
        np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1)) + 1e-12
    )
    print("=" * 100)
    print(name)
    print("orig  :", a.shape, float(a.min()), float(a.max()), float(a.mean()))
    print("manual:", b.shape, float(b.min()), float(b.max()), float(b.mean()))
    print("max_abs:", float(np.max(np.abs(diff))))
    print("mean_abs:", float(np.mean(np.abs(diff))))
    print("rmse:", float(np.sqrt(np.mean(diff ** 2))))
    print("cosine:", float(cos))


def main():
    cfg = Config.fromfile(str(CONFIG))
    import_plugins(cfg)

    model_orig = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model_orig, CKPT, map_location="cpu")
    model_orig.cuda().eval()

    model_manual = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model_manual, CKPT, map_location="cpu")
    model_manual.cuda().eval()

    original_tq = model_manual.head.temporal_query_fusion
    model_manual.head.temporal_query_fusion = ManualTemporalQueryFusion(original_tq).cuda().eval()

    bev = torch.from_numpy(np.load(TRACE / "bev_features.npy")).float().cuda()
    query = torch.from_numpy(np.load(TRACE / "step_00_input_query_coords.npy")).float().cuda()
    timestep = torch.from_numpy(np.load(TRACE / "step_00_input_timestep.npy")).float().cuda()
    prev_query_feat = torch.from_numpy(np.load(TRACE / "prev_query_feat.npy")).float().cuda()

    # Use same valid state saved by PyTorch trace.
    valid_np = np.load(TRACE / "prev_query_valid.npy")
    if valid_np.ndim == 1:
        # TemporalQueryFusion expects [B] bool in original code.
        prev_query_valid = torch.from_numpy(valid_np > 0.5).cuda()
    else:
        prev_query_valid = torch.from_numpy(valid_np > 0.5).cuda()

    print("bev/query/timestep/prev_query_feat/valid")
    stat("bev", bev)
    stat("query", query)
    print("timestep", timestep.detach().cpu().numpy())
    stat("prev_query_feat", prev_query_feat)
    print("prev_query_valid", prev_query_valid.detach().cpu().numpy())

    orig_line, orig_cls, orig_qfeat = run_head(
        model_orig.head, bev, query, timestep, prev_query_feat, prev_query_valid
    )
    man_line, man_cls, man_qfeat = run_head(
        model_manual.head, bev, query, timestep, prev_query_feat, prev_query_valid
    )

    compare("line_preds", orig_line.reshape(1, 100, 40), man_line.reshape(1, 100, 40))
    compare("cls_logits", orig_cls, man_cls)
    compare("query_feat", orig_qfeat, man_qfeat)


if __name__ == "__main__":
    main()
