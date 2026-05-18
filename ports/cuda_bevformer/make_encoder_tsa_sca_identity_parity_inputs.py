import os
import sys
import types
import importlib.util
import numpy as np
import torch

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

HELPER_SCRIPT = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/ports/cuda_bevformer/export_camera_bevformer_tensorio.py"
CFG = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/new_mapdiff.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff/iter_83520.pth"

OUT_DIR = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/cuda_bevformer/parity_engine_b_tsa"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_H = 480
IMG_W = 800


def load_helper():
    spec = importlib.util.spec_from_file_location("bevformer_export_helper", HELPER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sca_identity_forward(
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
    flag="encoder",
    **kwargs,
):
    # Match current TRT SCA skeleton: output = input query.
    return query


def patch_sca_identity(module):
    patched = 0
    for m in module.modules():
        if isinstance(m, SpatialCrossAttention):
            m.forward = types.MethodType(sca_identity_forward, m)
            patched += 1
    print("patched SCA identity modules:", patched)


class EncoderReference(torch.nn.Module):
    def __init__(self, backbone, helper):
        super().__init__()
        self.backbone = backbone

        # Important: keep TSA real, patch only SCA to identity.
        patch_sca_identity(self.backbone)

        enc = self.backbone.transformer.encoder
        enc.point_sampling = helper.types.MethodType(helper.tensor_point_sampling, enc)

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

        return bev


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

    wrapper = EncoderReference(model.backbone, helper).cuda().eval()

    with torch.no_grad():
        ref = wrapper(feats[0], feats[1], feats[2], ego2img).float().cpu().numpy()

    names = ["feat0", "feat1", "feat2"]
    for name, tensor in zip(names, feats):
        arr = tensor.detach().float().cpu().numpy()
        arr.tofile(os.path.join(OUT_DIR, f"{name}.fp32.bin"))
        print(name, arr.shape, arr.min(), arr.max())

    ego = ego2img.detach().float().cpu().numpy()
    ego.tofile(os.path.join(OUT_DIR, "ego2img.fp32.bin"))

    np.save(os.path.join(OUT_DIR, "pytorch_bev_tsa_real_sca_identity.npy"), ref)

    print("ego2img", ego.shape, ego.min(), ego.max())
    print("ref", ref.shape, ref.min(), ref.max())
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
