#!/usr/bin/env python3
import ctypes
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")

TRACE = ROOT / "model/mapdiffusion_temporal_routeB/pytorch_trace_sample0"
OUT = ROOT / "model/mapdiffusion_temporal_routeB/parity_sample0"

CONFIG = ROOT / "model/mapdiffusion_temporal_routeB/temporal_config.py"
CKPT = MAPDIFF_ROOT / "work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth"

ENGINE_A = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan"
ENGINE_B = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan"
ENGINE_D = ROOT / "model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan"
ENGINE_C = ROOT / "model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan"

PLUGIN_DCNV2 = ROOT / "build/plugins/libmmcv_dcnv2_trt.so"
PLUGIN_TSA = ROOT / "build/plugins/libbevformer_tsa_trt.so"
PLUGIN_SCA = ROOT / "build/plugins/libbevformer_sca_trt.so"
PLUGIN_MSDA = ROOT / "build/libmapdiffusion_msda.so"


# Reuse working TRT helper class from the temporal submission runner.
sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_val_submission.py"
spec = importlib.util.spec_from_file_location("temporal_runner", str(RUNNER))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

TrtModule = runner.TrtModule
load_engine = runner.load_engine
load_plugins = runner.load_plugins


def stats(name, arr):
    arr = np.asarray(arr)
    return (
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={float(arr.min()):.6f} max={float(arr.max()):.6f} mean={float(arr.mean()):.6f}"
    )


def compare(name, ref, test):
    ref = np.asarray(ref).astype(np.float32)
    test = np.asarray(test).astype(np.float32)

    if ref.shape != test.shape:
        print(f"[{name}] SHAPE MISMATCH ref={ref.shape} test={test.shape}")
        return None

    diff = test - ref
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))

    ref_flat = ref.reshape(-1).astype(np.float64)
    test_flat = test.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(ref_flat) * np.linalg.norm(test_flat)
    cosine = float(np.dot(ref_flat, test_flat) / denom) if denom > 0 else float("nan")

    print("=" * 100)
    print(name)
    print(stats("ref ", ref))
    print(stats("test", test))
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={mean_abs:.8f}")
    print(f"rmse={rmse:.8f}")
    print(f"cosine={cosine:.10f}")

    return {
        "name": name,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
    }


def load_trt_engines():
    logger = trt.Logger(trt.Logger.WARNING)
    load_plugins()

    A = TrtModule("EngineA_BackboneFPN", load_engine(ENGINE_A, logger))
    B = TrtModule("EngineB_BEVFormerEncoder", load_engine(ENGINE_B, logger))
    D = TrtModule("EngineD_StreamFusionNeck", load_engine(ENGINE_D, logger))
    C = TrtModule("EngineC_MapDiffusionHead", load_engine(ENGINE_C, logger))

    return A, B, D, C


def run_trt_abd(img, ego2img):
    A, B, D, _ = load_trt_engines()

    A.bind_host_input("img", img)
    A.allocate_outputs()

    B.bind_device_input("feat0", A.output_ptr("feat0"), A.output_shape("feat0"))
    B.bind_device_input("feat1", A.output_ptr("feat1"), A.output_shape("feat1"))
    B.bind_device_input("feat2", A.output_ptr("feat2"), A.output_shape("feat2"))
    B.bind_host_input("ego2img", ego2img)
    B.allocate_outputs()

    D.bind_device_input("prev_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.allocate_outputs()

    A.execute()
    B.execute()
    D.execute()

    b_out = B.copy_outputs_to_host()
    d_out = D.copy_outputs_to_host()

    return b_out["bev_features"], d_out["fused_bev"]


def run_trt_c(bev_features, query_coords, timestep, prev_query_feat):
    _, _, _, C = load_trt_engines()

    C.bind_host_input("bev_features", bev_features)
    C.bind_host_input("query_coords", query_coords)
    C.bind_host_input("timestep", timestep)
    C.bind_host_input("prev_query_feat", prev_query_feat)
    C.allocate_outputs()

    C.execute()
    return C.copy_outputs_to_host()


def import_mapdiff_plugins(cfg):
    sys.path.insert(0, str(MAPDIFF_ROOT))
    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir
        if not isinstance(plugin_dirs, list):
            plugin_dirs = [plugin_dirs]
        for plugin_dir in plugin_dirs:
            module_path = os.path.dirname(plugin_dir).replace("/", ".")
            importlib.import_module(module_path)

    if hasattr(cfg, "custom_imports"):
        for imp in cfg.custom_imports.get("imports", []):
            importlib.import_module(imp)


def run_pytorch_c(bev_features, query_coords, timestep, prev_query_feat):
    """
    Use the same TemporalHeadExportWrapper class from the ONNX exporter.
    This keeps the comparison aligned with the exported temporal head interface.
    """
    export_script = ROOT / "ports/mapdiffusion_temporal_routeB/export/export_temporal_head_onnx.py"
    if not export_script.exists():
        export_script = ROOT / "tools/temporal_routeB/export_temporal_head_onnx.py"

    if not export_script.exists():
        raise FileNotFoundError("Could not find export_temporal_head_onnx.py")

    spec = importlib.util.spec_from_file_location("temporal_head_export", str(export_script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "TemporalHeadExportWrapper"):
        raise AttributeError("TemporalHeadExportWrapper not found in export script")

    cfg = Config.fromfile(str(CONFIG))
    cfg.model.pretrained = None
    import_mapdiff_plugins(cfg)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg", None))
    load_checkpoint(model, str(CKPT), map_location="cpu", strict=False)
    model.cuda().eval()

    wrapper = mod.TemporalHeadExportWrapper(model.head).cuda().eval()

    bev_t = torch.from_numpy(bev_features.astype(np.float32)).cuda()
    q_t = torch.from_numpy(query_coords.astype(np.float32)).cuda()
    t_t = torch.from_numpy(timestep.astype(np.float32)).cuda()
    prev_t = torch.from_numpy(prev_query_feat.astype(np.float32)).cuda()

    # TemporalHeadExportWrapper.forward expects prev_query_valid.
    # Use valid=1 to match the exported temporal-head behavior that consumes prev_query_feat.
    prev_valid_np = np.ones((1,), dtype=np.float32)
    prev_valid_t = torch.from_numpy(prev_valid_np).cuda()

    with torch.no_grad():
        out = wrapper(bev_t, q_t, t_t, prev_t, prev_valid_t)

    if isinstance(out, dict):
        line_preds = out["line_preds"]
        cls_logits = out["cls_logits"]
        query_feat = out["query_feat"]
    elif isinstance(out, (list, tuple)) and len(out) == 3:
        line_preds, cls_logits, query_feat = out
    else:
        raise RuntimeError(f"Unexpected PyTorch C output type: {type(out)}")

    return {
        "line_preds": line_preds.detach().cpu().numpy(),
        "cls_logits": cls_logits.detach().cpu().numpy(),
        "query_feat": query_feat.detach().cpu().numpy(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    img = np.ascontiguousarray(np.load(TRACE / "img.npy").astype(np.float32))
    ego2img = np.ascontiguousarray(np.load(TRACE / "ego2img.npy").astype(np.float32))
    query = np.ascontiguousarray(np.load(TRACE / "step_00_query_coords.npy").astype(np.float32))

    pt_raw = np.ascontiguousarray(np.load(TRACE / "raw_bev.npy").astype(np.float32))
    pt_fused = np.ascontiguousarray(np.load(TRACE / "fused_bev.npy").astype(np.float32))
    pt_bev = np.ascontiguousarray(np.load(TRACE / "bev_features.npy").astype(np.float32))

    timestep = np.asarray([999.0], dtype=np.float32)
    prev_query_feat = np.zeros((1, 100, 512), dtype=np.float32)

    print("=" * 100)
    print("INPUTS")
    print(stats("img", img))
    print(stats("ego2img", ego2img))
    print(stats("query", query))
    print(stats("pt_raw", pt_raw))
    print(stats("pt_fused", pt_fused))
    print(stats("pt_bev", pt_bev))
    print("timestep:", timestep)
    print("=" * 100)

    print("\nRunning TRT A/B/D...")
    trt_raw, trt_fused = run_trt_abd(img, ego2img)

    np.save(OUT / "trt_raw_bev.npy", trt_raw)
    np.save(OUT / "trt_fused_bev.npy", trt_fused)

    compare("B raw_bev: PyTorch trace vs TRT B", pt_raw, trt_raw)
    compare("D fused_bev: PyTorch trace vs TRT D", pt_fused, trt_fused)
    compare("final bev_features: PyTorch trace vs TRT D fused", pt_bev, trt_fused)

    print("\nRunning TRT C with PyTorch trace bev_features...")
    trt_c = run_trt_c(pt_bev, query, timestep, prev_query_feat)
    for k, v in trt_c.items():
        np.save(OUT / f"trt_c_{k}.npy", v)
        print(stats(f"trt_c_{k}", v))

    print("\nRunning PyTorch C with same inputs...")
    try:
        pt_c = run_pytorch_c(pt_bev, query, timestep, prev_query_feat)
        for k, v in pt_c.items():
            np.save(OUT / f"pt_c_{k}.npy", v)
            print(stats(f"pt_c_{k}", v))

        for k in ["line_preds", "cls_logits", "query_feat"]:
            compare(f"C {k}: PyTorch head vs TRT C", pt_c[k], trt_c[k])

    except Exception as e:
        print("[WARN] PyTorch C parity skipped because wrapper failed:")
        print(repr(e))
        print("TRT C outputs were still saved.")

    print("=" * 100)
    print("saved parity outputs:", OUT)
    print("=" * 100)


if __name__ == "__main__":
    main()
