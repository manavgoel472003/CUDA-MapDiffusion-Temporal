import os
import sys
import random
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint

MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")
CBF_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")

CONFIG = os.environ.get(
    "TEMPORAL_CONFIG",
    str(CBF_ROOT / "model/mapdiffusion_temporal_routeB/temporal_config.py"),
)
CKPT = os.environ.get(
    "TEMPORAL_CKPT",
    "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth",
)
TRACE_DIR = Path(os.environ.get(
    "TRACE_DIR",
    str(CBF_ROOT / "model/mapdiffusion_temporal_routeB/pytorch_trace_sample0"),
))
ONNX_OUT = Path(os.environ.get(
    "ONNX_OUT",
    str(CBF_ROOT / "model/mapdiffusion_temporal_routeB/mapdiffusion.temporal_head.manual_tq.opset13.fp32.onnx"),
))
SEED = int(os.environ.get("SEED", "123"))

sys.path.insert(0, str(MAPDIFF_ROOT))

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def import_plugins(cfg):
    if getattr(cfg, "plugin", False):
        plugin_dir = cfg.plugin_dir
        if isinstance(plugin_dir, str):
            module_path = os.path.dirname(plugin_dir).replace("/", ".")
            print("[plugin] importing:", module_path)
            importlib.import_module(module_path)

    if hasattr(cfg, "custom_imports"):
        for imp in cfg.custom_imports.get("imports", []):
            print("[custom_import] importing:", imp)
            importlib.import_module(imp)



class ManualTemporalQueryFusion(nn.Module):
    """Fixed-shape export replacement for TemporalQueryFusion.

    This preserves the trained weights but avoids PyTorch nn.MultiheadAttention
    ONNX export, which creates dynamic [100, -1, 512] reshapes that TensorRT 8.5
    cannot solve.

    Expected fixed shapes:
      current_query:   [1, 100, 512]
      prev_query:      [1, 100, 512]
    """

    def __init__(self, original):
        super().__init__()

        self.embed_dims = int(original.embed_dims)
        self.num_heads = int(original.temporal_attn.num_heads)
        self.head_dim = self.embed_dims // self.num_heads
        self.scale = float(self.head_dim) ** -0.5
        self.use_ffn = bool(original.use_ffn)

        # Reuse trained LayerNorm modules.
        self.norm_q = original.norm_q
        self.norm_prev = original.norm_prev
        self.norm_after_attn = original.norm_after_attn
        self.norm_after_ffn = original.norm_after_ffn

        # Split trained MHA parameters ONCE at construction time.
        # Do NOT slice in forward(), because ONNX exports that as dynamic Slice
        # on weights and TensorRT 8.5 fails with axes.allValuesKnown().
        C = self.embed_dims
        in_w = original.temporal_attn.in_proj_weight.detach().clone()
        in_b = original.temporal_attn.in_proj_bias.detach().clone()

        self.w_q = nn.Parameter(in_w[0:C, :].contiguous())
        self.w_k = nn.Parameter(in_w[C:2*C, :].contiguous())
        self.w_v = nn.Parameter(in_w[2*C:3*C, :].contiguous())

        self.b_q = nn.Parameter(in_b[0:C].contiguous())
        self.b_k = nn.Parameter(in_b[C:2*C].contiguous())
        self.b_v = nn.Parameter(in_b[2*C:3*C].contiguous())

        self.out_proj_weight = nn.Parameter(
            original.temporal_attn.out_proj.weight.detach().clone().contiguous()
        )
        self.out_proj_bias = nn.Parameter(
            original.temporal_attn.out_proj.bias.detach().clone().contiguous()
        )

        # Reuse trained FFN linear modules.
        if self.use_ffn:
            self.ffn0 = original.ffn[0]
            self.ffn3 = original.ffn[3]
        else:
            self.ffn0 = None
            self.ffn3 = None

    def forward(self, current_query, prev_query=None, timestep=None, prev_valid=None):
        # Keep first-frame bypass for PyTorch use. During active temporal ONNX export
        # prev_query is always provided, so this branch is not in the exported graph.
        if prev_query is None:
            return current_query

        B = current_query.shape[0]
        N = current_query.shape[1]
        C = current_query.shape[2]

        q_in = self.norm_q(current_query)
        p_norm = self.norm_prev(prev_query.to(dtype=current_query.dtype))

        q = F.linear(q_in, self.w_q, self.b_q)
        k = F.linear(p_norm, self.w_k, self.b_k)
        v = F.linear(p_norm, self.w_v, self.b_v)

        # Fixed B=1, N=100, heads=8, head_dim=64 in this model.
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)

        temporal_context = F.linear(ctx, self.out_proj_weight, self.out_proj_bias)

        fused = self.norm_after_attn(current_query + temporal_context)

        if self.use_ffn:
            ff = self.ffn3(F.relu(self.ffn0(fused)))
            fused = self.norm_after_ffn(fused + ff)

        return fused



class TemporalHeadExportWrapper(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, bev_features, query_coords, timestep, prev_query_feat, prev_query_valid):
        # prev_query_valid is float32 [B, 100], 0 or 1.
        # If all invalid, bypass temporal fusion by passing None.
        # This branch is fixed during export, so for the ONNX graph we export the active temporal path.
        # For first-frame-only export, export a second wrapper later if needed.
        valid_bool = prev_query_valid > 0.5

        outputs = self.head.forward_test(
            query_coords=query_coords,
            timestep=timestep,
            bev_features=bev_features,
            img_metas=[{}],
            prev_query_feat=prev_query_feat,
            prev_query_valid=valid_bool,
        )

        # Last decoder layer is the final prediction.
        last = outputs[-1]
        line_preds = last["lines"][0].unsqueeze(0)
        cls_logits = last["scores"][0].unsqueeze(0)
        query_feat = last["query_feat"]

        return line_preds, cls_logits, query_feat


def main():
    cfg = Config.fromfile(CONFIG)
    import_plugins(cfg)

    print("=" * 100)
    print("CONFIG:", CONFIG)
    print("CKPT:", CKPT)
    print("TRACE_DIR:", TRACE_DIR)
    print("ONNX_OUT:", ONNX_OUT)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg", None))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()

    print("model type:", type(model))
    print("head type:", type(model.head))
    print("has temporal_query_fusion:", hasattr(model.head, "temporal_query_fusion"))

    # Replace PyTorch nn.MultiheadAttention-based temporal fusion with
    # a fixed-shape equivalent for TensorRT export.
    if hasattr(model.head, "temporal_query_fusion"):
        original_tq = model.head.temporal_query_fusion
        model.head.temporal_query_fusion = ManualTemporalQueryFusion(original_tq).cuda().eval()
        print("replaced temporal_query_fusion with ManualTemporalQueryFusion")

    wrapper = TemporalHeadExportWrapper(model.head).cuda().eval()

    bev_features = torch.from_numpy(np.load(TRACE_DIR / "bev_features.npy").astype(np.float32)).cuda()
    query_coords = torch.from_numpy(np.load(TRACE_DIR / "step_00_query_coords.npy").astype(np.float32)).cuda()
    timestep = torch.tensor([1000.0], dtype=torch.float32, device="cuda")

    # Export the ACTIVE temporal path.
    # Use dummy valid previous features so TemporalQueryFusion is in the graph.
    prev_query_feat = torch.zeros((1, 100, 512), dtype=torch.float32, device="cuda")
    prev_query_valid = torch.ones((1, 100), dtype=torch.float32, device="cuda")

    with torch.no_grad():
        line_preds, cls_logits, query_feat = wrapper(
            bev_features,
            query_coords,
            timestep,
            prev_query_feat,
            prev_query_valid,
        )

    print("line_preds:", tuple(line_preds.shape), float(line_preds.min()), float(line_preds.max()), float(line_preds.mean()))
    print("cls_logits:", tuple(cls_logits.shape), float(cls_logits.min()), float(cls_logits.max()), float(cls_logits.mean()))
    print("query_feat:", tuple(query_feat.shape), float(query_feat.min()), float(query_feat.max()), float(query_feat.mean()))

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (bev_features, query_coords, timestep, prev_query_feat, prev_query_valid),
        str(ONNX_OUT),
        input_names=[
            "bev_features",
            "query_coords",
            "timestep",
            "prev_query_feat",
            "prev_query_valid",
        ],
        output_names=[
            "line_preds",
            "cls_logits",
            "query_feat",
        ],
        opset_version=13,
        do_constant_folding=False,
        dynamic_axes=None,
        verbose=False,
    )

    print("saved ONNX:", ONNX_OUT)


if __name__ == "__main__":
    main()
