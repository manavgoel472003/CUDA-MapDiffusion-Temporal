import os
import sys
import types
import importlib.util
import torch
import onnx
from torch.onnx import register_custom_op_symbolic

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
from plugin.models.backbones.bevformer.temporal_self_attention import TemporalSelfAttention

HELPER_SCRIPT = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/ports/cuda_bevformer/export_camera_bevformer_tensorio.py"
CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"
OUT_ONNX = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/cuda_bevformer/camera.bevformer_encoder.sca_plugin.onnx"

IMG_H = 480
IMG_W = 800


def load_helper():
    spec = importlib.util.spec_from_file_location("bevformer_export_helper", HELPER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod




# -------------------------------------------------------------------------
# TemporalSelfAttention plugin boundary
#
# This removes the existing MultiscaleDeformableAttnPlugin_TRT temporal path
# from ONNX. The current plugin is a skeleton boundary only; it is not
# numerically correct until the real TSA math is implemented in TensorRT.
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# TemporalSelfAttention plugin boundary with learned weights
#
# Local TSA details:
#   embed_dims=256
#   num_heads=8
#   num_levels=1
#   num_points=4
#   num_bev_queue=2
#
# This plugin receives all learned Linear weights needed for real TSA:
#   value_proj
#   sampling_offsets
#   attention_weights
#   output_proj
# -------------------------------------------------------------------------

class BEVFormerTemporalSelfAttentionPluginFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        identity,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        value_proj_weight,
        value_proj_bias,
        sampling_offsets_weight,
        sampling_offsets_bias,
        attention_weights_weight,
        attention_weights_bias,
        output_proj_weight,
        output_proj_bias,
    ):
        # Export placeholder only.
        # TensorRT plugin will implement the real TSA math.
        return query

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        identity,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        value_proj_weight,
        value_proj_bias,
        sampling_offsets_weight,
        sampling_offsets_bias,
        attention_weights_weight,
        attention_weights_bias,
        output_proj_weight,
        output_proj_bias,
    ):
        return g.op(
            "bevformer::BEVFormerTemporalSelfAttentionPlugin",
            query,
            key,
            value,
            identity,
            query_pos,
            reference_points,
            spatial_shapes,
            level_start_index,
            value_proj_weight,
            value_proj_bias,
            sampling_offsets_weight,
            sampling_offsets_bias,
            attention_weights_weight,
            attention_weights_bias,
            output_proj_weight,
            output_proj_bias,
            outputs=1,
        )


def export_tsa_plugin_forward(
    self,
    query,
    key=None,
    value=None,
    identity=None,
    query_pos=None,
    key_padding_mask=None,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    flag='decoder',
    **kwargs,
):
    # Match original TemporalSelfAttention.forward() input preparation.
    if value is None:
        assert self.batch_first
        bs, len_bev, c = query.shape
        value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

    if key is None:
        key = value

    if identity is None:
        identity = query

    if query_pos is None:
        query_pos = torch.zeros_like(query)

    return BEVFormerTemporalSelfAttentionPluginFn.apply(
        query,
        key,
        value,
        identity,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        self.value_proj.weight,
        self.value_proj.bias,
        self.sampling_offsets.weight,
        self.sampling_offsets.bias,
        self.attention_weights.weight,
        self.attention_weights.bias,
        self.output_proj.weight,
        self.output_proj.bias,
    )


def patch_temporal_self_attention_to_plugin(module):
    patched = 0
    for m in module.modules():
        if isinstance(m, TemporalSelfAttention):
            m.forward = types.MethodType(export_tsa_plugin_forward, m)
            patched += 1
    print("[ONNX] patched TemporalSelfAttention to plugin boundary:", patched)
    return patched



class BEVFormerSCAPluginFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        query_pos,
        reference_points_cam,
        bev_mask,
        spatial_shapes,
        level_start_index,
        value_proj_weight,
        value_proj_bias,
        sampling_offsets_weight,
        sampling_offsets_bias,
        attention_weights_weight,
        attention_weights_bias,
        output_proj_weight,
        output_proj_bias,
    ):
        # Export placeholder only.
        # Current TRT SCA skeleton copies query -> output.
        return query

    @staticmethod
    def symbolic(
        g,
        query,
        key,
        value,
        query_pos,
        reference_points_cam,
        bev_mask,
        spatial_shapes,
        level_start_index,
        value_proj_weight,
        value_proj_bias,
        sampling_offsets_weight,
        sampling_offsets_bias,
        attention_weights_weight,
        attention_weights_bias,
        output_proj_weight,
        output_proj_bias,
    ):
        return g.op(
            "bevformer::BEVFormerSpatialCrossAttentionPlugin",
            query,
            key,
            value,
            query_pos,
            reference_points_cam,
            bev_mask,
            spatial_shapes,
            level_start_index,
            value_proj_weight,
            value_proj_bias,
            sampling_offsets_weight,
            sampling_offsets_bias,
            attention_weights_weight,
            attention_weights_bias,
            output_proj_weight,
            output_proj_bias,
            outputs=1,
        )


def export_sca_plugin_forward(
    self,
    query,
    key=None,
    value=None,
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
    if residual is None:
        residual = query
    if query_pos is None:
        query_pos = torch.zeros_like(query)

    out = BEVFormerSCAPluginFn.apply(
        query,
        key,
        value,
        query_pos,
        reference_points_cam,
        bev_mask.to(query.dtype),
        spatial_shapes,
        level_start_index,
        self.deformable_attention.value_proj.weight,
        self.deformable_attention.value_proj.bias,
        self.deformable_attention.sampling_offsets.weight,
        self.deformable_attention.sampling_offsets.bias,
        self.deformable_attention.attention_weights.weight,
        self.deformable_attention.attention_weights.bias,
        self.output_proj.weight,
        self.output_proj.bias,
    )

    # Plugin output is expected to already include the SCA output path.
    return out


def patch_spatial_cross_attention_to_plugin(module):
    patched = 0
    for m in module.modules():
        if isinstance(m, SpatialCrossAttention):
            m.forward = types.MethodType(export_sca_plugin_forward, m)
            patched += 1
    print("[ONNX] patched SpatialCrossAttention to plugin boundary:", patched)
    return patched


class CameraBEVFormerEncoderSCAPluginTensorIO(torch.nn.Module):
    def __init__(self, backbone, helper):
        super().__init__()
        self.backbone = backbone
        self.helper = helper

        patch_temporal_self_attention_to_plugin(self.backbone)
        patch_spatial_cross_attention_to_plugin(self.backbone)

        enc = self.backbone.transformer.encoder
        enc.point_sampling = self.helper.types.MethodType(
            self.helper.tensor_point_sampling,
            enc,
        )

    def forward(self, feat0, feat1, feat2, ego2img):
        img_metas = [{
            "ego2img_tensor": ego2img,
            "img_shape": [(IMG_H, IMG_W, 3)] * 6,
        }]

        mlvl_feats = [feat0, feat1, feat2]

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.backbone.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.backbone.bev_h, self.backbone.bev_w),
            device=bev_queries.device,
            dtype=dtype,
        )

        bev_pos = self.backbone.positional_encoding(bev_mask).to(dtype)

        outs = self.backbone.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.backbone.bev_h,
            self.backbone.bev_w,
            grid_length=(
                self.backbone.real_h / self.backbone.bev_h,
                self.backbone.real_w / self.backbone.bev_w,
            ),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=None,
        )

        b = outs.shape[0]
        c = outs.shape[2]

        bev = outs.reshape(
            b,
            self.backbone.bev_h,
            self.backbone.bev_w,
            c,
        ).permute(0, 3, 1, 2).contiguous()

        if self.backbone.upsample:
            bev = self.backbone.up(bev)

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
        ops[(n.domain, n.op_type)] = ops.get((n.domain, n.op_type), 0) + 1

    print("OPS:")
    for (domain, op), v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0][1]))[:100]:
        if domain:
            print(f"  {domain}::{op} {v}")
        else:
            print(f"  {op} {v}")


def main():
    torch.set_grad_enabled(False)

    helper = load_helper()
    cfg = Config.fromfile(CFG)

    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    batch = next(iter(loader))
    img = batch["img"].data[0].cuda().contiguous()
    img_metas_original = batch["img_metas"].data[0]

    ego2img = torch.as_tensor(
        img_metas_original[0]["ego2img"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0).contiguous()

    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu", strict=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        feats = model.backbone.extract_img_feat(img=img, img_metas=img_metas_original)

    for i, f in enumerate(feats):
        print(f"feat{i}:", tuple(f.shape), f.dtype, float(f.min()), float(f.max()))

    wrapper = CameraBEVFormerEncoderSCAPluginTensorIO(model.backbone, helper).cuda().eval()

    print("============================================================")
    print("Smoke forward")
    print("============================================================")
    with torch.no_grad():
        bev = wrapper(feats[0], feats[1], feats[2], ego2img)

    print("bev shape:", tuple(bev.shape))
    print("bev min/max:", float(bev.min()), float(bev.max()))

    assert tuple(bev.shape) == (1, 256, 50, 100), tuple(bev.shape)

    print("============================================================")
    print("Export ONNX")
    print("============================================================")

    # Needed for TemporalSelfAttention, which still uses MultiScaleDeformableAttnFunction_fp32.
    helper.register_bevformer_msda_symbolic()

    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (feats[0], feats[1], feats[2], ego2img),
        OUT_ONNX,
        input_names=["feat0", "feat1", "feat2", "ego2img"],
        output_names=["bev_features"],
        opset_version=13,
        do_constant_folding=True,
        custom_opsets={"bevformer": 1},
        verbose=False,
    )

    print("saved:", OUT_ONNX)
    inspect_onnx(OUT_ONNX)
    print("EXPORT DONE")


if __name__ == "__main__":
    main()
