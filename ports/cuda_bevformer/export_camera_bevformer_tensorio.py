import os
import sys
import types
import torch
import onnx

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.datasets.builder import build_dataloader
from mmdet3d.models import build_model

MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
if MAPDIFF_ROOT not in sys.path:
    sys.path.insert(0, MAPDIFF_ROOT)

import plugin

from plugin.models.backbones.bevformer.spatial_cross_attention import SpatialCrossAttention
from plugin.models.backbones.bevformer.multi_scale_deformable_attn_function import (
    MultiScaleDeformableAttnFunction_fp32,
    MultiScaleDeformableAttnFunction_fp16,
)


CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"
OUT_ONNX = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/cuda_bevformer/camera.bevformer.onnx"

IMG_H = 480
IMG_W = 800






# -------------------------------------------------------------------------
# Export-safe dense SpatialCrossAttention
#
# Original BEVFormer SpatialCrossAttention dynamically selects visible BEV
# queries per camera using nonzero(), rebatches to max_len, and scatters back.
# TensorRT fails to statically compile that graph.
#
# This export-only version keeps all BEV queries for all cameras:
#   query_dense: [B, num_cams, 5000, 256]
#
# It keeps the same trained weights and uses the visibility mask only when
# averaging camera outputs back into BEV slots.
# -------------------------------------------------------------------------

def export_dense_spatial_cross_attention_forward(
    self,
    query,
    key,
    value,
    residual=None,
    query_pos=None,
    key_padding_mask=None,
    reference_points=None,
    spatial_shapes=None,
    reference_points_cam=None,
    bev_mask=None,
    level_start_index=None,
    flag='encoder',
    **kwargs,
):
    if key is None:
        key = query
    if value is None:
        value = key

    inp_residual = query if residual is None else residual

    if query_pos is not None:
        query = query + query_pos

    bs, num_query, _ = query.size()
    D = reference_points_cam.size(3)

    # key/value are [num_cams, length, bs, embed_dims]
    num_cams, l, bs_key, embed_dims = key.shape

    key = key.permute(2, 0, 1, 3).reshape(
        bs * self.num_cams, l, self.embed_dims
    )
    value = value.permute(2, 0, 1, 3).reshape(
        bs * self.num_cams, l, self.embed_dims
    )

    # Dense/static query expansion:
    # [bs, num_query, C] -> [bs, num_cams, num_query, C]
    query_dense = query[:, None, :, :].expand(
        bs, self.num_cams, num_query, self.embed_dims
    ).contiguous()

    # reference_points_cam is [num_cams, bs, num_query, D, 2]
    # Make it [bs, num_cams, num_query, D, 2]
    reference_points_dense = reference_points_cam.permute(
        1, 0, 2, 3, 4
    ).contiguous()

    queries = self.deformable_attention(
        query=query_dense.reshape(bs * self.num_cams, num_query, self.embed_dims),
        key=key,
        value=value,
        reference_points=reference_points_dense.reshape(
            bs * self.num_cams, num_query, D, 2
        ),
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    ).reshape(bs, self.num_cams, num_query, self.embed_dims)

    # bev_mask is [num_cams, bs, num_query, D]
    # visible is [bs, num_cams, num_query]
    visible = (bev_mask.sum(-1) > 0).permute(1, 0, 2).to(query.dtype)

    slots = (queries * visible[..., None]).sum(dim=1)

    count = visible.sum(dim=1)
    count = torch.clamp(count, min=1.0)

    slots = slots / count[..., None]
    slots = self.output_proj(slots)

    return self.dropout(slots) + inp_residual


def patch_spatial_cross_attention_dense_for_export(module):
    patched = 0
    for m in module.modules():
        if isinstance(m, SpatialCrossAttention):
            m.forward = types.MethodType(export_dense_spatial_cross_attention_forward, m)
            patched += 1
    print("[ONNX] patched dense SpatialCrossAttention modules:", patched)
    return patched


# -------------------------------------------------------------------------
# BEVFormer MSDA ONNX symbolic
#
# PyTorch 1.9 traces BEVFormer deformable attention as a PythonOp:
#   MultiScaleDeformableAttnFunction_fp32
#
# register_custom_op_symbolic does not reliably attach to this PythonOp.
# Therefore we attach a symbolic() method directly to the autograd Function
# class, same overall idea as the existing MapDiffusion custom TRT plugin path.
# -------------------------------------------------------------------------

def bevformer_msda_symbolic(
    g,
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
):
    return g.op(
        "bevformer::BEVFormerMultiScaleDeformableAttnPlugin",
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        outputs=1,
    )


def register_bevformer_msda_symbolic():
    MultiScaleDeformableAttnFunction_fp32.symbolic = staticmethod(bevformer_msda_symbolic)
    MultiScaleDeformableAttnFunction_fp16.symbolic = staticmethod(bevformer_msda_symbolic)
    print("[ONNX] patched symbolic on MultiScaleDeformableAttnFunction_fp32/fp16")

def tensor_point_sampling(self, reference_points, pc_range, img_metas):
    """
    Tensor-only replacement for BEVFormerEncoder.point_sampling.

    Original version reads:
      img_metas[0]['ego2img']
      img_metas[0]['img_shape']

    This version reads:
      img_metas[0]['ego2img_tensor']

    This makes ego2img a real ONNX input instead of Python metadata.
    """
    ego2img = img_metas[0]["ego2img_tensor"]

    # ego2img should be [B, N, 4, 4].
    if ego2img.dim() == 3:
        ego2img = ego2img.unsqueeze(0)

    reference_points = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1
    )

    reference_points = reference_points.permute(1, 0, 2, 3)

    D, B, num_query = reference_points.size()[:3]
    num_cam = ego2img.size(1)

    reference_points = (
        reference_points
        .view(D, B, 1, num_query, 4)
        .repeat(1, 1, num_cam, 1, 1)
        .unsqueeze(-1)
    )

    ego2img = (
        ego2img
        .view(1, B, num_cam, 1, 4, 4)
        .repeat(D, 1, 1, num_query, 1, 1)
    )

    reference_points_cam = torch.matmul(
        ego2img.to(torch.float32),
        reference_points.to(torch.float32)
    ).squeeze(-1)

    eps = 1e-5

    bev_mask = reference_points_cam[..., 2:3] > eps

    # Export-safe replacement for torch.maximum(z, eps).
    # PyTorch 1.9 ONNX exporter does not support aten::maximum reliably here.
    z = torch.clamp(reference_points_cam[..., 2:3], min=eps)
    reference_points_cam = reference_points_cam[..., 0:2] / z

    reference_points_cam[..., 0] = reference_points_cam[..., 0] / float(IMG_W)
    reference_points_cam[..., 1] = reference_points_cam[..., 1] / float(IMG_H)

    bev_mask = (
        bev_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
    )

    # Export-safe: bev_mask is already boolean here, so nan_to_num is unnecessary.
    bev_mask = bev_mask

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, bev_mask




def export_safe_bevformer_forward(self, img, img_metas, *args, prev_bev=None, only_bev=False, **kwargs):
    """
    ONNX-export-safe replacement for BEVFormerBackbone.forward.

    Same logic as original, but replaces:
      outs.unflatten(1, (bev_h, bev_w))
    with:
      outs.reshape(B, bev_h, bev_w, C)

    PyTorch 1.9 ONNX tracer fails on Tensor.unflatten().
    """
    mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)

    bs, num_cam, _, _, _ = mlvl_feats[0].shape
    dtype = mlvl_feats[0].dtype

    bev_queries = self.bev_embedding.weight.to(dtype)

    bev_mask = torch.zeros(
        (bs, self.bev_h, self.bev_w),
        device=bev_queries.device,
        dtype=dtype,
    )

    bev_pos = self.positional_encoding(bev_mask).to(dtype)

    outs = self.transformer.get_bev_features(
        mlvl_feats,
        bev_queries,
        self.bev_h,
        self.bev_w,
        grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
        bev_pos=bev_pos,
        img_metas=img_metas,
        prev_bev=prev_bev,
    )

    # Original:
    # outs = outs.unflatten(1, (self.bev_h, self.bev_w)).permute(0, 3, 1, 2).contiguous()
    #
    # Export-safe:
    b = outs.shape[0]
    c = outs.shape[2]
    outs = outs.reshape(b, self.bev_h, self.bev_w, c).permute(0, 3, 1, 2).contiguous()

    if self.upsample:
        outs = self.up(outs)

    return outs


class CameraBEVFormerTensorIO(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Patch BEVFormer forward to avoid Tensor.unflatten(), unsupported by PyTorch 1.9 ONNX tracer.
        self.backbone.forward = types.MethodType(export_safe_bevformer_forward, self.backbone)

        # Patch SpatialCrossAttention to remove dynamic nonzero/max_len/scatter path.
        patch_spatial_cross_attention_dense_for_export(self.backbone)

        # Patch encoder point_sampling to use tensor ego2img.
        enc = self.backbone.transformer.encoder
        enc.point_sampling = types.MethodType(tensor_point_sampling, enc)

    def forward(self, img, ego2img):
        # img:     [1, 6, 3, 480, 800]
        # ego2img: [1, 6, 4, 4]
        img_metas = [{
            "ego2img_tensor": ego2img,
            "img_shape": [(IMG_H, IMG_W, 3)] * 6,
        }]

        bev = self.backbone(img, img_metas=img_metas)

        # Expected [1, 256, 50, 100]
        if bev.dim() == 3:
            bev = bev.unsqueeze(0)

        return bev.contiguous()


def inspect_onnx(path):
    print("=" * 80)
    print("ONNX:", path)
    m = onnx.load(path)

    print("INPUTS:")
    for x in m.graph.input:
        shape = []
        for d in x.type.tensor_type.shape.dim:
            shape.append(d.dim_param or d.dim_value or "?")
        print(" ", x.name, shape)

    print("OUTPUTS:")
    for x in m.graph.output:
        shape = []
        for d in x.type.tensor_type.shape.dim:
            shape.append(d.dim_param or d.dim_value or "?")
        print(" ", x.name, shape)

    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1

    print("OPS:")
    for k, v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))[:80]:
        print(" ", k, v)


def main():
    torch.set_grad_enabled(False)

    cfg = Config.fromfile(CFG)

    print("============================================================")
    print("Build val sample")
    print("============================================================")
    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    batch = next(iter(loader))

    img = batch["img"].data[0].cuda()
    img_metas_original = batch["img_metas"].data[0]

    ego2img = torch.as_tensor(
        img_metas_original[0]["ego2img"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0)

    print("img shape:", tuple(img.shape))
    print("ego2img shape:", tuple(ego2img.shape))
    print("original img_shape:", img_metas_original[0]["img_shape"])

    print("============================================================")
    print("Build trained MapDiffusion model")
    print("============================================================")
    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu", strict=False)
    model.cuda()
    model.eval()

    wrapper = CameraBEVFormerTensorIO(model.backbone).cuda().eval()

    print("============================================================")
    print("Smoke forward")
    print("============================================================")
    bev = wrapper(img, ego2img)
    print("bev shape:", tuple(bev.shape))
    print("bev dtype:", bev.dtype)
    print("bev min/max:", float(bev.min()), float(bev.max()))

    assert tuple(bev.shape) == (1, 256, 50, 100), tuple(bev.shape)

    print("============================================================")
    print("Export ONNX")
    print("============================================================")
    register_bevformer_msda_symbolic()
    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (img, ego2img),
        OUT_ONNX,
        input_names=["img", "ego2img"],
        output_names=["bev_features"],
        opset_version=13,
        do_constant_folding=True,
        custom_opsets={"bevformer": 1},
        verbose=False,
    )

    print("saved:", OUT_ONNX)

    print("============================================================")
    print("Inspect ONNX")
    print("============================================================")
    inspect_onnx(OUT_ONNX)

    print("============================================================")
    print("EXPORT DONE")
    print("============================================================")


if __name__ == "__main__":
    main()
