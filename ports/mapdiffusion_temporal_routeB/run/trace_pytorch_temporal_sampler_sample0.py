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

MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
CONFIG = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/temporal_config.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth"
OUT_DIR = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/pt_sampler_trace_sample0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, MAPDIFF_ROOT)
os.chdir(MAPDIFF_ROOT)

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

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()

    total_steps = int(getattr(cfg, "total_steps", 1000))
    scheduler = getattr(cfg, "scheduler", "cosine")
    coef = compute_ddpm_coef(total_steps, scheduler)

    eta = float(cfg.evaluation.get("eval_diffusion_eta", 0.5))
    sampling_timesteps = int(cfg.evaluation.get("eval_diffusion_sampling_timesteps", 5))
    query_threshold = float(cfg.evaluation.get("eval_diffusion_query_threshold", 0.5))

    data = dataset[0]
    batch = collate([data], samples_per_gpu=1)
    batch = scatter(batch, [0])[0]

    img = unwrap(batch["img"])
    img_metas = unwrap(batch["img_metas"])
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    if img.dim() == 4:
        img = img.unsqueeze(0)

    with torch.no_grad():
        # Match MapDiffusionTemporal.forward_test up to BEV.
        if hasattr(model, "backbone"):
            _bev_feats = model.backbone(img, img_metas)
        else:
            raise RuntimeError("model has no backbone")

        bev_feats = model.neck(_bev_feats)

        prev_query_feat, prev_query_valid = model._fetch_prev_query_feature(
            img_metas=img_metas,
            device=bev_feats.device,
            is_train=False,
        )

        bs = bev_feats.shape[0]
        num_queries = model.head.num_queries
        num_points = model.head.num_points

        times = torch.linspace(0, total_steps, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        query_coords = torch.normal(
            mean=0.5,
            std=0.25,
            size=(bs, num_queries, num_points, 2),
            device=img.device,
        ).clip(0, 1)

        print("time_pairs:", time_pairs)
        print("bev_feats:", tuple(bev_feats.shape), float(bev_feats.min()), float(bev_feats.max()), float(bev_feats.mean()))
        print("prev_query_feat:", None if prev_query_feat is None else tuple(prev_query_feat.shape))
        print("prev_query_valid:", None if prev_query_valid is None else prev_query_valid.detach().cpu().numpy())

        np.save(OUT_DIR / "bev_features.npy", bev_feats.detach().cpu().numpy().astype(np.float32))
        if prev_query_feat is None:
            np.save(OUT_DIR / "prev_query_feat.npy", np.zeros((1, 100, 512), dtype=np.float32))
        else:
            np.save(OUT_DIR / "prev_query_feat.npy", prev_query_feat.detach().cpu().numpy().astype(np.float32))

        if prev_query_valid is None:
            np.save(OUT_DIR / "prev_query_valid.npy", np.zeros((1,), dtype=np.float32))
        else:
            np.save(OUT_DIR / "prev_query_valid.npy", prev_query_valid.detach().cpu().float().cpu().numpy())

        final_line = None
        final_cls = None
        final_qfeat = None

        for step_idx, (time, time_next) in enumerate(time_pairs):
            np.save(OUT_DIR / f"step_{step_idx:02d}_input_query_coords.npy", query_coords.detach().cpu().numpy().astype(np.float32))
            np.save(OUT_DIR / f"step_{step_idx:02d}_input_timestep.npy", np.array([float(time)], dtype=np.float32))

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

            np.save(OUT_DIR / f"step_{step_idx:02d}_pt_line_preds.npy", line_preds.detach().cpu().numpy().astype(np.float32))
            np.save(OUT_DIR / f"step_{step_idx:02d}_pt_cls_logits.npy", cls_logits.detach().cpu().numpy().astype(np.float32))
            if final_qfeat is not None:
                np.save(OUT_DIR / f"step_{step_idx:02d}_pt_query_feat.npy", final_qfeat.detach().cpu().numpy().astype(np.float32))

            print("=" * 100)
            print("step", step_idx, "time", time, "time_next", time_next)
            print("input query", tuple(query_coords.shape), float(query_coords.min()), float(query_coords.max()), float(query_coords.mean()))
            print("line", tuple(line_preds.shape), float(line_preds.min()), float(line_preds.max()), float(line_preds.mean()))
            print("cls", tuple(cls_logits.shape), float(cls_logits.min()), float(cls_logits.max()), float(cls_logits.mean()))

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
            np.save(OUT_DIR / f"step_{step_idx:02d}_next_query_coords.npy", query_coords.detach().cpu().numpy().astype(np.float32))

        if final_line is None:
            final_line = line_preds
            final_cls = cls_logits

        final_lines_np = final_line.detach().cpu().numpy().astype(np.float32)
        final_cls_np = final_cls.detach().cpu().numpy().astype(np.float32)
        np.save(OUT_DIR / "final_pt_line_preds.npy", final_lines_np)
        np.save(OUT_DIR / "final_pt_cls_logits.npy", final_cls_np)

        probs = 1.0 / (1.0 + np.exp(-final_cls_np[0]))
        scores = probs.max(axis=-1)
        labels = probs.argmax(axis=-1)
        vectors = denorm(final_lines_np[0])
        submission = {
            "meta": {"output_format": "vector"},
            "results": {
                img_metas[0]["token"]: {
                    "vectors": vectors.tolist(),
                    "scores": scores.astype(float).tolist(),
                    "labels": labels.astype(int).tolist(),
                    "prop_mask": [False] * len(scores),
                }
            }
        }
        with open(OUT_DIR / "submission_vector.json", "w") as f:
            json.dump(submission, f)

    print("saved trace:", OUT_DIR)


if __name__ == "__main__":
    main()
