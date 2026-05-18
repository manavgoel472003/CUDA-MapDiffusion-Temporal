import os
import sys
import json
import importlib
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")
CONFIG = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/temporal_config.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth"
OUT_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/e2e_trace_pytorch_2")

sys.path.insert(0, str(MAPDIFF_ROOT))
os.chdir(str(MAPDIFF_ROOT))

from plugin.models.utils.coef import compute_ddpm_coef, predict_noise_from_start


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


def unwrap(x):
    if hasattr(x, "data"):
        x = x.data
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def save(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    np.save(path, np.ascontiguousarray(x).astype(np.float32))


def denorm(line_preds):
    pts = line_preds.reshape(-1, 20, 2).astype(np.float32).copy()
    pts[..., 0] = pts[..., 0] * 60.0 - 30.0
    pts[..., 1] = pts[..., 1] * 30.0 - 15.0
    return pts


def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)

    cfg = Config.fromfile(CONFIG)
    import_plugins(cfg)

    dataset_cfg = cfg.data.test.copy()
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    print("[dataset]", len(dataset))

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()

    total_steps = int(getattr(cfg, "total_steps", 1000))
    scheduler = getattr(cfg, "scheduler", "cosine")
    coef = compute_ddpm_coef(total_steps, scheduler)

    eta = float(cfg.evaluation.get("eval_diffusion_eta", 0.5))
    sampling_timesteps = int(cfg.evaluation.get("eval_diffusion_sampling_timesteps", 5))
    query_threshold = float(cfg.evaluation.get("eval_diffusion_query_threshold", 0.5))

    times = torch.linspace(0, total_steps, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    print("[time_pairs]", time_pairs)

    submission = {"meta": {"output_format": "vector"}, "results": {}}

    for dataset_idx in [0, 1]:
        data = dataset[dataset_idx]
        batch = collate([data], samples_per_gpu=1)
        batch = scatter(batch, [0])[0]

        img = unwrap(batch["img"])
        img_metas = unwrap(batch["img_metas"])
        if isinstance(img_metas, dict):
            img_metas = [img_metas]
        if img.dim() == 4:
            img = img.unsqueeze(0)

        token = img_metas[0]["token"]
        scene = img_metas[0].get("scene_name", "unknown")

        out_dir = OUT_ROOT / f"debug_e2e_idx{dataset_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "meta.txt", "w") as f:
            f.write(f"dataset_idx={dataset_idx}\n")
            f.write(f"token={token}\n")
            f.write(f"scene={scene}\n")

        with torch.no_grad():
            save(out_dir / "00_img.npy", img)
            # PyTorch may not expose ego2img as separate tensor in same format, so skip unless present.

            # Backbone / BEV
            _bev_feats = model.backbone(img, img_metas)

            # Try to save A-like features if backbone returns list/tuple.
            if isinstance(_bev_feats, (list, tuple)):
                for i, feat in enumerate(_bev_feats):
                    save(out_dir / f"01_A_feat{i}.npy", feat)
            else:
                save(out_dir / "01_A_feat0.npy", _bev_feats)

            bev_feats = model.neck(_bev_feats)
            save(out_dir / "02_B_raw_bev_or_head_bev.npy", bev_feats)

            prev_query_feat, prev_query_valid = model._fetch_prev_query_feature(
                img_metas=img_metas,
                device=bev_feats.device,
                is_train=False,
            )

            if prev_query_feat is None:
                prev_query_feat = torch.zeros((1, 100, 512), device=bev_feats.device)
            if prev_query_valid is None:
                prev_query_valid = torch.zeros((1,), dtype=torch.bool, device=bev_feats.device)

            save(out_dir / "03_prev_query_feat.npy", prev_query_feat)
            save(out_dir / "03_prev_query_valid.npy", prev_query_valid.float())

            bs = bev_feats.shape[0]
            num_queries = model.head.num_queries
            num_points = model.head.num_points

            query_coords = torch.normal(
                mean=0.5,
                std=0.25,
                size=(bs, num_queries, num_points, 2),
                device=img.device,
            ).clip(0, 1)

            final_line = None
            final_cls = None
            final_qfeat = None

            for step_idx, (time, time_next) in enumerate(time_pairs):
                save(out_dir / f"04_C_step{step_idx:02d}_input_bev.npy", bev_feats)
                save(out_dir / f"04_C_step{step_idx:02d}_input_query_coords.npy", query_coords)
                save(out_dir / f"04_C_step{step_idx:02d}_input_timestep.npy", np.array([float(time)], dtype=np.float32))
                save(out_dir / f"04_C_step{step_idx:02d}_input_prev_query_feat.npy", prev_query_feat)

                preds_list = model.head(
                    query_coords,
                    time,
                    bev_feats,
                    img_metas=img_metas,
                    prev_query_feat=prev_query_feat,
                    prev_query_valid=prev_query_valid,
                    return_loss=False,
                )

                last = preds_list[-1]
                line_preds = torch.stack(last["lines"], dim=0).view(bs, -1, num_points, 2)
                cls_logits = torch.stack(last["scores"], dim=0)
                final_qfeat = last.get("query_feat", final_qfeat)

                save(out_dir / f"05_C_step{step_idx:02d}_output_line_preds.npy", line_preds.reshape(1, 100, 40))
                save(out_dir / f"05_C_step{step_idx:02d}_output_cls_logits.npy", cls_logits)
                if final_qfeat is not None:
                    save(out_dir / f"05_C_step{step_idx:02d}_output_query_feat.npy", final_qfeat)

                print(
                    f"[PT] idx={dataset_idx} step={step_idx} time={time} "
                    f"line_mean={float(line_preds.mean()):.4f} "
                    f"cls_minmax=({float(cls_logits.min()):.4f},{float(cls_logits.max()):.4f})"
                )

                if time_next == 0:
                    query_coords = line_preds
                    final_line = line_preds
                    final_cls = cls_logits
                    break

                pred_noise = predict_noise_from_start(coef, query_coords, time, line_preds)
                alpha = coef["alphas_cumprod"][time - 1]
                alpha_next = coef["alphas_cumprod"][time_next - 1]

                sigma = eta * np.sqrt(
                    (1 - alpha / alpha_next)
                    * (1 - alpha_next)
                    / (1 - alpha)
                )
                c = np.sqrt(1 - alpha_next - sigma ** 2)

                next_query_list = []
                for b in range(bs):
                    score_per_image = torch.sigmoid(cls_logits[b])
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                    keep_idx = value > query_threshold
                    num_remain = int(torch.sum(keep_idx).item())

                    pred_noise_b = pred_noise[b:b + 1, keep_idx, :, :]
                    x_start_b = line_preds[b:b + 1, keep_idx, :, :]

                    noise = torch.normal(
                        mean=0.0,
                        std=0.25,
                        size=x_start_b.shape,
                        device=img.device,
                    ).clip(0, 1)

                    query_b = (
                        x_start_b * np.sqrt(alpha_next)
                        + c * pred_noise_b
                        + sigma * noise
                    )

                    if num_remain < num_queries:
                        noise_new = torch.normal(
                            mean=0.5,
                            std=0.25,
                            size=(1, num_queries - num_remain, num_points, 2),
                            device=img.device,
                        ).clip(0, 1)
                        query_b = torch.cat([query_b, noise_new], dim=1)

                    next_query_list.append(query_b[:, :num_queries, :, :])

                query_coords = torch.cat(next_query_list, dim=0)

            final_line_np = final_line.detach().cpu().numpy().astype(np.float32)
            final_cls_np = final_cls.detach().cpu().numpy().astype(np.float32)
            scores = 1.0 / (1.0 + np.exp(-final_cls_np[0]))
            max_scores = scores.max(axis=-1)
            labels = scores.argmax(axis=-1)
            vectors = denorm(final_line_np[0])

            submission["results"][token] = {
                "vectors": vectors.tolist(),
                "scores": max_scores.astype(float).tolist(),
                "labels": labels.astype(int).tolist(),
                "prop_mask": [False] * len(max_scores),
            }

            print(
                f"[PT final] idx={dataset_idx} token={token} "
                f"score_max={float(max_scores.max()):.4f} score_mean={float(max_scores.mean()):.4f}"
            )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "submission_vector.json", "w") as f:
        json.dump(submission, f)
    print("[saved]", OUT_ROOT)


if __name__ == "__main__":
    main()
