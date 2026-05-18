import os
import sys
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

CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"
OUT_ONNX = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/cuda_bevformer/camera.backbone_fpn.onnx"


class CameraBackboneFPNTensorIO(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, img):
        # img: [1,6,3,480,800]
        img_metas = [{
            "img_shape": [(480, 800, 3)] * 6,
        }]

        feats = self.backbone.extract_img_feat(img=img, img_metas=img_metas)

        # Expected list of [B,N,C,H,W].
        # Return each level separately so TensorRT can build a clean backbone engine.
        return tuple(x.contiguous() for x in feats)


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
    for (domain, op), v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0][1]))[:80]:
        if domain:
            print(f"  {domain}::{op} {v}")
        else:
            print(f"  {op} {v}")


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
    img = batch["img"].data[0].cuda().contiguous()

    print("img shape:", tuple(img.shape))

    print("============================================================")
    print("Build trained MapDiffusion model")
    print("============================================================")

    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location="cpu", strict=False)
    model.cuda()
    model.eval()

    wrapper = CameraBackboneFPNTensorIO(model.backbone).cuda().eval()

    print("============================================================")
    print("Smoke forward")
    print("============================================================")

    feats = wrapper(img)
    for i, f in enumerate(feats):
        print(f"feat{i}:", tuple(f.shape), f.dtype, float(f.min()), float(f.max()))

    print("============================================================")
    print("Export ONNX")
    print("============================================================")

    os.makedirs(os.path.dirname(OUT_ONNX), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (img,),
        OUT_ONNX,
        input_names=["img"],
        output_names=[f"feat{i}" for i in range(len(feats))],
        opset_version=13,
        do_constant_folding=True,
        custom_opsets={"mmcv": 1},
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
