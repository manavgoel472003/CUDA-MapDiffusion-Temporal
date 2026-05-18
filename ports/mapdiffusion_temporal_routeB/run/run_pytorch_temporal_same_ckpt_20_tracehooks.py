import os
import types
import sys
import json
import importlib
from pathlib import Path

import mmcv
import numpy as np
import torch
import plugin.models.utils.coef as coef_utils
from plugin.models.utils.coef import compute_ddpm_coef
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
CONFIG = "/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/temporal_config.py"
CKPT = "/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth"

OUT_DIR = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/pytorch_same_ckpt_20")
START = 0
LIMIT = 20

sys.path.insert(0, MAPDIFF_ROOT)
os.chdir(MAPDIFF_ROOT)


def import_plugin(cfg):
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


def to_builtin(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    return x



def unwrap_batch_value(x):
    if hasattr(x, "data"):
        x = x.data
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def extract_img_and_metas(batch):
    img = unwrap_batch_value(batch["img"])
    img_metas = unwrap_batch_value(batch["img_metas"])

    if torch.is_tensor(img) and img.dim() == 4:
        # [6,3,480,800] -> [1,6,3,480,800]
        img = img.unsqueeze(0)

    if isinstance(img_metas, dict):
        img_metas = [img_metas]

    return img, img_metas


def normalize_result(result, token):
    # Unwrap common test output wrappers.
    while isinstance(result, (list, tuple)) and len(result) == 1:
        result = result[0]

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected PyTorch result type: {type(result)}")

    print("[debug result keys]", result.keys())

    if not all(k in result for k in ["vectors", "scores", "labels"]):
        raise RuntimeError(f"Missing vectors/scores/labels in result keys: {result.keys()}")

    vectors = to_builtin(result["vectors"])
    scores = to_builtin(result["scores"])
    labels = to_builtin(result["labels"])

    n = len(vectors)
    prop_mask = result.get("prop_mask", result.get("prop", [True] * n))
    prop_mask = to_builtin(prop_mask)

    return {
        "token": token,
        "vectors": vectors,
        "scores": scores,
        "labels": labels,
        "prop_mask": prop_mask,
    }


def reset_temporal_state_if_available(model):
    # Best-effort reset hooks; harmless if absent.
    for obj in [model, getattr(model, "module", None), getattr(model, "head", None)]:
        if obj is None:
            continue
        for name in [
            "reset",
            "reset_temporal",
            "reset_temporal_state",
            "reset_streaming_state",
            "reset_memory",
        ]:
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    print(f"[reset] calling {obj.__class__.__name__}.{name}()")
                    fn()
                    return
                except TypeError:
                    pass



# ===== TRACE HOOK HELPERS =====
TRACE_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/pytorch_working_trace_2")

def _save_np(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import numpy as np
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    np.save(path, np.ascontiguousarray(x).astype(np.float32))

def install_head_forward_trace(model, dataset=None):
    import torch
    orig_forward_test = model.head.forward_test
    token_to_idx = {}
    if dataset is not None and hasattr(dataset, "samples"):
        token_to_idx = {sample["token"]: i for i, sample in enumerate(dataset.samples)}
    call_state = {"sample_idx": -1, "step": 0, "last_token": None, "token_steps": {}}

    def set_sample_idx(i):
        call_state["sample_idx"] = int(i)
        call_state["step"] = 0

    def traced_forward_test(self, query_coords, timestep, bev_features, img_metas, prev_query_feat=None, prev_query_valid=None, *args, **kwargs):
        # Derive sample idx from token instead of relying on patched outer loop.
        token = None
        try:
            meta0 = img_metas[0] if isinstance(img_metas, (list, tuple)) else img_metas
            token = meta0.get("token", None)
        except Exception:
            token = None

        sample_idx = token_to_idx.get(token, call_state["sample_idx"])

        if token not in call_state["token_steps"]:
            call_state["token_steps"][token] = 0
        step = call_state["token_steps"][token]

        out_dir = TRACE_ROOT / f"debug_e2e_idx{sample_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        _save_np(out_dir / f"04_C_step{step:02d}_input_bev.npy", bev_features)
        _save_np(out_dir / f"04_C_step{step:02d}_input_query_coords.npy", query_coords)

        if torch.is_tensor(timestep):
            t_arr = timestep.detach().float().view(-1).cpu().numpy()
        else:
            t_arr = np.array([float(timestep)], dtype=np.float32)
        _save_np(out_dir / f"04_C_step{step:02d}_input_timestep.npy", t_arr)

        if prev_query_feat is None:
            _save_np(out_dir / f"04_C_step{step:02d}_input_prev_query_feat.npy", np.zeros((1, 100, 512), dtype=np.float32))
        else:
            _save_np(out_dir / f"04_C_step{step:02d}_input_prev_query_feat.npy", prev_query_feat)

        if prev_query_valid is None:
            _save_np(out_dir / f"04_C_step{step:02d}_input_prev_query_valid.npy", np.zeros((1,), dtype=np.float32))
        else:
            _save_np(out_dir / f"04_C_step{step:02d}_input_prev_query_valid.npy", prev_query_valid.float())

        outputs = orig_forward_test(
            query_coords=query_coords,
            timestep=timestep,
            bev_features=bev_features,
            img_metas=img_metas,
            prev_query_feat=prev_query_feat,
            prev_query_valid=prev_query_valid,
            *args,
            **kwargs,
        )

        last = outputs[-1]
        line_preds = torch.stack(last["lines"], dim=0).view(1, -1, self.num_points, 2).reshape(1, self.num_queries, self.num_points * 2)
        cls_logits = torch.stack(last["scores"], dim=0)

        _save_np(out_dir / f"05_C_step{step:02d}_output_line_preds.npy", line_preds)
        _save_np(out_dir / f"05_C_step{step:02d}_output_cls_logits.npy", cls_logits)

        if "query_feat" in last:
            _save_np(out_dir / f"05_C_step{step:02d}_output_query_feat.npy", last["query_feat"])

        print(
            f"[PT HOOK] sample={sample_idx} step={step} "
            f"t={float(t_arr[0])} cls_min={float(cls_logits.min()):.4f} cls_max={float(cls_logits.max()):.4f}"
        )

        call_state["token_steps"][token] += 1
        call_state["step"] += 1
        return outputs

    model.head.forward_test = types.MethodType(traced_forward_test, model.head)
    model._trace_set_sample_idx = set_sample_idx
    print("[trace hooks installed]", TRACE_ROOT)
# ===== END TRACE HOOK HELPERS =====


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(CONFIG)
    import_plugin(cfg)

    total_steps = int(getattr(cfg, "total_steps", 1000))
    scheduler = getattr(cfg, "scheduler", "cosine")
    coef = compute_ddpm_coef(total_steps, scheduler)

    eval_eta = float(getattr(cfg.evaluation, "eval_diffusion_eta", 0.5)) if hasattr(cfg, "evaluation") else 0.5
    eval_sampling_timesteps = int(getattr(cfg.evaluation, "eval_diffusion_sampling_timesteps", 5)) if hasattr(cfg, "evaluation") else 5
    eval_query_threshold = float(getattr(cfg.evaluation, "eval_diffusion_query_threshold", 0.5)) if hasattr(cfg, "evaluation") else 0.5

    dataset_cfg = cfg.data.test.copy()
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    print("=" * 100)
    print("[coef_utils callables]", [x for x in dir(coef_utils) if not x.startswith("_")])
    print("CONFIG:", CONFIG)
    print("CKPT:", CKPT)
    print("total_steps:", total_steps)
    print("scheduler:", scheduler)
    print("eval_eta:", eval_eta)
    print("eval_sampling_timesteps:", eval_sampling_timesteps)
    print("eval_query_threshold:", eval_query_threshold)
    print("dataset size:", len(dataset))
    print("model:", type(model))
    print("head:", type(getattr(model, "head", None)))
    print("=" * 100)

    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda()
    model.eval()
    install_head_forward_trace(model, dataset)

    submission = {
        "meta": dict(output_format="vector"),
        "results": {},
    }
    results_list = []

    last_scene = None

    with torch.no_grad():
        for j, idx in enumerate(range(START, START + LIMIT), 1):
            sample = dataset.samples[idx]
            token = sample["token"]
            scene = sample.get("scene_name", "NA")

            if scene != last_scene:
                print(f"[scene reset] idx={idx} scene={scene}")
                reset_temporal_state_if_available(model)
                last_scene = scene

            data = dataset[idx]
            batch = collate([data], samples_per_gpu=1)
            batch = scatter(batch, [0])[0]

            img, img_metas = extract_img_and_metas(batch)

            # Match TRT sampler settings.
            # TRT chunk runner used timesteps [999, 749.25, 499.5, 249.75, 0].
            # The PyTorch forward_test API requires these sampler controls.
            result = model.forward_test(
                timestep=total_steps,
                eta=eval_eta,
                coef=coef,
                sampling_timesteps=eval_sampling_timesteps,
                query_threshold=eval_query_threshold,
                img=img,
                img_metas=img_metas,
            )

            pred = normalize_result(result, token)

            submission["results"][token] = {
                "vectors": pred["vectors"],
                "scores": pred["scores"],
                "labels": pred["labels"],
                "prop_mask": pred["prop_mask"],
            }
            results_list.append(pred)

            scores = np.asarray(pred["scores"], dtype=np.float32)
            labels = np.asarray(pred["labels"], dtype=np.int64)
            print(
                f"[{j:04d}/{LIMIT:04d}] idx={idx} scene={scene} token={token} "
                f"n={len(scores)} score_max={scores.max():.4f} "
                f"score_mean={scores.mean():.4f} labels={dict(zip(*np.unique(labels, return_counts=True)))}"
            )

    mmcv.dump(results_list, str(OUT_DIR / "pytorch_results_20.pkl"))
    with open(OUT_DIR / "submission_vector.json", "w") as f:
        json.dump(submission, f)

    print("=" * 100)
    print("saved:", OUT_DIR / "pytorch_results_20.pkl")
    print("saved:", OUT_DIR / "submission_vector.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
