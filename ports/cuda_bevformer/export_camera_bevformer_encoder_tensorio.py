import os
import sys
import importlib.util
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

HELPER_SCRIPT = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/ports/cuda_bevformer/export_camera_bevformer_tensorio.py"
CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"
OUT_ONNX = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/cuda_bevformer/camera.bevformer_encoder.onnx"

IMG_H = 480
IMG_W = 800


def load_helper():
    spec = importlib.util.spec_from_file_location("bevformer_export_helper", HELPER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CameraBEVFormerEncoderTensorIO(torch.nn.Module):
    def __init__(self, backbone, helper):
        super().__init__()
        self.backbone = backbone
        self.helper = helper

        # Same export-safe patches used in the full BEVFormer export.
        self.helper.patch_spatial_cross_attention_dense_for_export(self.backbone)

        enc = self.backbone.transformer.encoder
        enc.point_sampling = self.helper.types.MethodType(
            self.helper.tensor_point_sampling,
            enc,
        )

    def forward(self, feat0, feat1, feat2, ego2img):
        # feat0:   [1,6,256,60,100]
        # feat1:   [1,6,256,30,50]
        # feat2:   [1,6,256,15,25]
        # ego2img: [1,6,4,4]

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

    img = batch["img"].data[0].cuda().contiguous()
    img_metas_original = batch["img_metas"].data[0]

    ego2img = torch.as_tensor(
        img_metas_original[0]["ego2img"],
        dtype=torch.float32,
        device="cuda",
    ).unsqueeze(0).contiguous()

    print("img shape:", tuple(img.shape))
    print("ego2img shape:", tuple(ego2img.shape))

    print("============================================================")
    print("Build trained MapDiffusion model")
    print("============================================================")

    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu", strict=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        feats = model.backbone.extract_img_feat(img=img, img_metas=img_metas_original)

    for i, f in enumerate(feats):
        print(f"feat{i}:", tuple(f.shape), f.dtype, float(f.min()), float(f.max()))

    wrapper = CameraBEVFormerEncoderTensorIO(model.backbone, helper).cuda().eval()

    print("============================================================")
    print("Smoke forward")
    print("============================================================")

    with torch.no_grad():
        bev = wrapper(feats[0], feats[1], feats[2], ego2img)

    print("bev shape:", tuple(bev.shape))
    print("bev dtype:", bev.dtype)
    print("bev min/max:", float(bev.min()), float(bev.max()))

    assert tuple(bev.shape) == (1, 256, 50, 100), tuple(bev.shape)

    print("============================================================")
    print("Export ONNX")
    print("============================================================")

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

    print("============================================================")
    print("Inspect ONNX")
    print("============================================================")
    inspect_onnx(OUT_ONNX)

    print("============================================================")
    print("EXPORT DONE")
    print("============================================================")


if __name__ == "__main__":
    main()
