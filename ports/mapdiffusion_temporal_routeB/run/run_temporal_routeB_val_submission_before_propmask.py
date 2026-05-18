#!/usr/bin/env python3
import argparse
import ctypes
import importlib
import json
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
import tensorrt as trt
from plugin.models.utils.coef import compute_ddpm_coef, predict_noise_from_start
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset


ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")

ENGINE_A = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan"
ENGINE_B = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan"
ENGINE_D = ROOT / "model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan"
ENGINE_C = ROOT / "model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan"

PLUGIN_DCNV2 = ROOT / "build/plugins/libmmcv_dcnv2_trt.so"
PLUGIN_TSA = ROOT / "build/plugins/libbevformer_tsa_trt.so"
PLUGIN_SCA = ROOT / "build/plugins/libbevformer_sca_trt.so"
PLUGIN_MSDA = ROOT / "build/libmapdiffusion_msda.so"

DEFAULT_CONFIG = ROOT / "model/mapdiffusion_temporal_routeB/temporal_config.py"

CATEGORIES = ["ped_crossing", "divider", "boundary"]


# ----------------------------
# CUDA runtime helpers
# ----------------------------
libcudart = ctypes.CDLL("libcudart.so")

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libcudart.cudaMalloc.restype = ctypes.c_int
libcudart.cudaFree.argtypes = [ctypes.c_void_p]
libcudart.cudaFree.restype = ctypes.c_int
libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libcudart.cudaMemcpy.restype = ctypes.c_int
libcudart.cudaDeviceSynchronize.argtypes = []
libcudart.cudaDeviceSynchronize.restype = ctypes.c_int


def cuda_check(code, msg):
    if int(code) != 0:
        raise RuntimeError(f"{msg}: cuda error code {code}")


def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    cuda_check(libcudart.cudaMalloc(ctypes.byref(ptr), int(nbytes)), "cudaMalloc")
    return ptr


def cuda_free(ptr):
    if ptr:
        libcudart.cudaFree(ptr)


def memcpy_h2d(dst_ptr, src_np):
    src_np = np.ascontiguousarray(src_np)
    cuda_check(
        libcudart.cudaMemcpy(
            ctypes.c_void_p(int(dst_ptr.value if hasattr(dst_ptr, "value") else dst_ptr)),
            src_np.ctypes.data_as(ctypes.c_void_p),
            int(src_np.nbytes),
            cudaMemcpyHostToDevice,
        ),
        "cudaMemcpy H2D",
    )


def memcpy_d2h(dst_np, src_ptr):
    dst_np = np.ascontiguousarray(dst_np)
    cuda_check(
        libcudart.cudaMemcpy(
            dst_np.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(int(src_ptr.value if hasattr(src_ptr, "value") else src_ptr)),
            int(dst_np.nbytes),
            cudaMemcpyDeviceToHost,
        ),
        "cudaMemcpy D2H",
    )


def cuda_sync():
    cuda_check(libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


# ----------------------------
# TensorRT wrapper
# ----------------------------
class TrtModule:
    def __init__(self, name, engine):
        self.name = name
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.bindings = [0] * engine.num_bindings
        self.owned_ptrs = {}
        self.owned_nbytes = {}
        self.output_host = {}

        print(f"\n[{self.name}] bindings")
        for i in range(engine.num_bindings):
            bname = engine.get_binding_name(i)
            mode = "IN " if engine.binding_is_input(i) else "OUT"
            dtype = engine.get_binding_dtype(i)
            shape = tuple(engine.get_binding_shape(i))
            print(f"  {mode} {i}: {bname} {shape} {dtype}")

    def _malloc_or_reuse(self, key, nbytes):
        old_ptr = self.owned_ptrs.get(key)
        old_nbytes = self.owned_nbytes.get(key, 0)
        if old_ptr is not None and old_nbytes >= nbytes:
            return old_ptr
        if old_ptr is not None:
            cuda_free(old_ptr)
        ptr = cuda_malloc(nbytes)
        self.owned_ptrs[key] = ptr
        self.owned_nbytes[key] = nbytes
        return ptr

    def bind_host_input(self, name, arr):
        idx = self.engine.get_binding_index(name)
        dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        arr = np.ascontiguousarray(arr.astype(dtype, copy=False))

        self.ctx.set_binding_shape(idx, tuple(arr.shape))
        ptr = self._malloc_or_reuse(f"input:{name}", arr.nbytes)
        memcpy_h2d(ptr, arr)

        self.bindings[idx] = int(ptr.value)
        return ptr

    def bind_device_input(self, name, ptr, shape):
        idx = self.engine.get_binding_index(name)
        self.ctx.set_binding_shape(idx, tuple(shape))
        self.bindings[idx] = int(ptr.value if hasattr(ptr, "value") else ptr)

    def allocate_outputs(self):
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                continue

            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.ctx.get_binding_shape(i))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"{self.name}: unresolved output shape for {name}: {shape}")

            host = np.empty(shape, dtype=dtype)
            ptr = self._malloc_or_reuse(f"output:{name}", host.nbytes)

            self.output_host[name] = host
            self.bindings[i] = int(ptr.value)

    def execute(self):
        ok = self.ctx.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError(f"{self.name}: execute_v2 failed")

    def copy_outputs_to_host(self):
        out = {}
        for name, host in self.output_host.items():
            ptr = self.owned_ptrs[f"output:{name}"]
            memcpy_d2h(host, ptr)
            out[name] = host.copy()
        return out

    def output_ptr(self, name):
        return self.owned_ptrs[f"output:{name}"]

    def output_shape(self, name):
        return tuple(self.output_host[name].shape)


# ----------------------------
# Loading helpers
# ----------------------------
def load_plugins():
    for p in [PLUGIN_DCNV2, PLUGIN_TSA, PLUGIN_SCA, PLUGIN_MSDA]:
        if not p.exists():
            raise FileNotFoundError(p)
        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        print("[plugin loaded]", p)

    trt.init_libnvinfer_plugins(None, "")


def load_engine(path, logger):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize engine: {path}")
    print("[engine loaded]", path)
    return engine


def import_mapdiff_plugins(cfg):
    sys.path.insert(0, str(MAPDIFF_ROOT))
    os.chdir(str(MAPDIFF_ROOT))

    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir
        if not isinstance(plugin_dirs, list):
            plugin_dirs = [plugin_dirs]
        for plugin_dir in plugin_dirs:
            module_path = os.path.dirname(plugin_dir).replace("/", ".")
            print("[plugin import]", module_path)
            importlib.import_module(module_path)

    if hasattr(cfg, "custom_imports"):
        for imp in cfg.custom_imports.get("imports", []):
            print("[custom import]", imp)
            importlib.import_module(imp)


# ----------------------------
# Dataset extraction
# ----------------------------
def unwrap_data_container(x):
    if hasattr(x, "data"):
        return x.data
    return x


def extract_meta(data):
    meta = unwrap_data_container(data["img_metas"])

    if isinstance(meta, list) and len(meta) == 1:
        meta = meta[0]
    if isinstance(meta, list) and len(meta) == 1:
        meta = meta[0]
    if isinstance(meta, dict):
        return meta

    raise TypeError(f"Unsupported img_metas format: {type(meta)}")


def extract_img(data):
    """
    Extract image tensor as [1, 6, 3, 480, 800].

    Different MapDiffusion dataset pipelines may expose the images under:
      img
      imgs
      img_inputs
      img_data
      images

    Some packed forms store the image tensor as the first element of a tuple/list.
    """
    candidate_keys = ["img", "imgs", "img_inputs", "img_data", "images"]

    found_key = None
    obj = None
    for k in candidate_keys:
        if k in data:
            found_key = k
            obj = unwrap_data_container(data[k])
            break

    if obj is None:
        raise KeyError(f"No image key found. Available keys: {list(data.keys())}")

    # Unwrap one-item lists from DataContainer.
    while isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]

    # Packed tuple/list case: image tensor is usually the first tensor-like item.
    if isinstance(obj, (list, tuple)):
        tensor_like = None
        for item in obj:
            item_unwrapped = unwrap_data_container(item)
            if hasattr(item_unwrapped, "shape"):
                tensor_like = item_unwrapped
                break
            if isinstance(item_unwrapped, (list, tuple)):
                for sub in item_unwrapped:
                    if hasattr(sub, "shape"):
                        tensor_like = sub
                        break
            if tensor_like is not None:
                break

        if tensor_like is None:
            raise TypeError(f"Could not find tensor-like image inside key {found_key}: {type(obj)}")

        obj = tensor_like

    if isinstance(obj, torch.Tensor):
        img = obj.detach().cpu().float().numpy()
    else:
        img = np.asarray(obj, dtype=np.float32)

    # Common shapes:
    # [6, 3, 480, 800] -> [1, 6, 3, 480, 800]
    # [1, 6, 3, 480, 800] unchanged
    if img.ndim == 4:
        img = img[None, ...]

    if img.shape != (1, 6, 3, 480, 800):
        raise RuntimeError(f"bad img shape from key {found_key}: {img.shape}")

    return np.ascontiguousarray(img.astype(np.float32))

def extract_ego2img(meta):
    ego2img = meta.get("ego2img", None)
    if ego2img is None:
        ego2img = meta.get("lidar2img", None)
    if ego2img is None:
        raise KeyError("meta has neither ego2img nor lidar2img")

    ego2img = np.asarray(ego2img, dtype=np.float32)
    if ego2img.ndim == 3:
        ego2img = ego2img[None, ...]

    if ego2img.shape != (1, 6, 4, 4):
        raise RuntimeError(f"bad ego2img shape: {ego2img.shape}")

    return np.ascontiguousarray(ego2img)


# ----------------------------
# Sampling / postprocess
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_timesteps(total_steps=1000, sampling_timesteps=5):
    # Match the older runner convention seen in logs: [999, 749, 499, 249, 0]
    return np.linspace(total_steps - 1, 0, sampling_timesteps, dtype=np.float32).tolist()


def denorm_vectors_60x30(line_preds):
    pts = line_preds.reshape(-1, 20, 2).astype(np.float32).copy()
    pts[..., 0] = pts[..., 0] * 60.0 - 30.0
    pts[..., 1] = pts[..., 1] * 30.0 - 15.0
    return pts


def format_result(line_preds, cls_logits, score_thr=0.0):
    if line_preds.ndim == 3:
        line_preds = line_preds[0]
    if cls_logits.ndim == 3:
        cls_logits = cls_logits[0]

    probs = sigmoid(cls_logits)
    labels = probs.argmax(axis=-1).astype(np.int64)
    scores = probs.max(axis=-1).astype(np.float32)
    vectors = denorm_vectors_60x30(line_preds)

    keep = scores >= float(score_thr)

    result = {
        "vectors": [vectors[i] for i in np.where(keep)[0]],
        "scores": [float(scores[i]) for i in np.where(keep)[0]],
        "labels": [int(labels[i]) for i in np.where(keep)[0]],
        "prop": [True for _ in np.where(keep)[0]],
    }

    return result, int(keep.sum()), float(scores.max()), float(scores.mean())



def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_initial_query_like_pytorch(rng, shape=(1, 100, 20, 2)):
    # PyTorch forward_test uses clipped normal noise, not uniform.
    q = rng.normal(0.5, 0.25, size=shape).astype(np.float32)
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def ddim_update_query_like_pytorch(
    query_coords,
    line_preds,
    cls_logits,
    time,
    time_next,
    coef,
    eta,
    query_threshold,
    rng,
):
    """NumPy version of MapDiffusionTemporal.forward_test query update."""
    x_start = np.ascontiguousarray(line_preds.reshape(1, -1, 20, 2).astype(np.float32))
    x_start_class = np.ascontiguousarray(cls_logits.reshape(1, -1, 3).astype(np.float32))

    if int(time_next) == 0:
        return x_start.astype(np.float32), int(x_start.shape[1]), 0

    t = int(time)
    tn = int(time_next)

    pred_noise = (
        coef["pred_coef1"][t - 1] * query_coords - x_start
    ) / coef["pred_coef2"][t - 1]

    alpha = coef["alphas_cumprod"][t - 1]
    alpha_next = coef["alphas_cumprod"][tn - 1]

    sigma = eta * np.sqrt(
        (1.0 - alpha / alpha_next)
        * (1.0 - alpha_next)
        / (1.0 - alpha)
    )
    c = np.sqrt(max(0.0, 1.0 - alpha_next - sigma ** 2))

    score = sigmoid_np(x_start_class).max(axis=-1)  # [1,100]
    keep = score[0] > query_threshold
    num_remain = int(keep.sum())

    pred_noise_keep = pred_noise[:, keep, :, :]
    x_start_keep = x_start[:, keep, :, :]

    noise = rng.normal(
        loc=0.0,
        scale=0.25,
        size=x_start_keep.shape,
    ).astype(np.float32)
    noise = np.clip(noise, 0.0, 1.0)

    query_keep = (
        x_start_keep * np.sqrt(alpha_next)
        + c * pred_noise_keep
        + sigma * noise
    )

    # Refill to 100 queries with clipped normal noise like PyTorch.
    num_refill = int(query_coords.shape[1] - num_remain)
    if num_refill > 0:
        refill = rng.normal(
            loc=0.5,
            scale=0.25,
            size=(1, num_refill, query_coords.shape[2], query_coords.shape[3]),
        ).astype(np.float32)
        refill = np.clip(refill, 0.0, 1.0)
        query_next = np.concatenate([query_keep, refill], axis=1)
    else:
        query_next = query_keep[:, :query_coords.shape[1], :, :]
        num_refill = 0

    query_next = np.clip(query_next, 0.0, 1.0).astype(np.float32)
    return query_next, num_remain, num_refill


def update_query_simple(line_preds, cls_logits, rng, query_threshold):
    """
    Stable approximation of the older feedback loop:
    keep high-confidence predicted x_start queries and refill the rest.
    """
    x_start = line_preds.reshape(1, 100, 20, 2).astype(np.float32)
    cls = cls_logits.reshape(1, 100, 3).astype(np.float32)

    scores = sigmoid(cls[0]).max(axis=-1)
    keep = scores >= float(query_threshold)
    keep_idx = np.where(keep)[0]

    next_q = np.empty_like(x_start)
    n_keep = min(len(keep_idx), 100)

    if n_keep > 0:
        next_q[:, :n_keep] = x_start[:, keep_idx[:n_keep]]

    if n_keep < 100:
        next_q[:, n_keep:] = rng.uniform(0.0, 1.0, size=(1, 100 - n_keep, 20, 2)).astype(np.float32)

    next_q = np.clip(next_q, 0.0, 1.0)
    return np.ascontiguousarray(next_q), int(n_keep), int(100 - n_keep)



def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.int32, np.int64, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--score-thr", type=float, default=0.0, help="filter threshold for saved submission; normally keep 0.0 and visualize with --thr")
    ap.add_argument("--query-threshold", type=float, default=0.5)
    ap.add_argument("--sampling-timesteps", type=int, default=5)
    ap.add_argument("--total-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--reset-each-scene", action="store_true", default=True)
    ap.add_argument("--debug-save-npz", action="store_true")
    args = ap.parse_args()

    old_cwd = os.getcwd()
    cfg = Config.fromfile(args.config)

    # Diffusion sampler coefficients for PyTorch-matching DDIM update.
    total_steps_cfg = int(getattr(cfg, 'total_steps', 1000))
    scheduler_cfg = getattr(cfg, 'scheduler', 'cosine')
    coef = compute_ddpm_coef(total_steps_cfg, scheduler_cfg)
    eval_cfg = getattr(cfg, 'evaluation', {})
    eta = float(eval_cfg.get('eval_diffusion_eta', 0.5))
    query_threshold = float(eval_cfg.get('eval_diffusion_query_threshold', 0.5))
    sampling_timesteps_cfg = int(eval_cfg.get('eval_diffusion_sampling_timesteps', 5))
    print('[sampler cfg] total_steps=', total_steps_cfg, 'scheduler=', scheduler_cfg, 'eta=', eta, 'sampling_timesteps=', sampling_timesteps_cfg, 'query_threshold=', query_threshold)
    import_mapdiff_plugins(cfg)

    # Use the original new_mapdiff TRT inference dataset path.
    # cfg.eval_config only returns img_metas/vectors for eval rendering;
    # cfg.data.test/val returns camera image tensors + ego2img.
    data_cfg = cfg.data.test if hasattr(cfg.data, "test") else cfg.data.val
    data_cfg = data_cfg.copy()
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    n_total = len(dataset)
    end = n_total if args.limit < 0 else min(n_total, args.start + args.limit)
    indices = list(range(args.start, end))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Temporal Route B Val Submission")
    print("config:", args.config)
    print("out_dir:", out_dir)
    print("dataset size:", n_total)
    print("start/end/count:", args.start, end, len(indices))
    print("ENGINE_A:", ENGINE_A)
    print("ENGINE_B:", ENGINE_B)
    print("ENGINE_D:", ENGINE_D)
    print("ENGINE_C:", ENGINE_C)
    print("=" * 100)

    logger = trt.Logger(trt.Logger.WARNING)
    load_plugins()

    engine_a = load_engine(ENGINE_A, logger)
    engine_b = load_engine(ENGINE_B, logger)
    engine_d = load_engine(ENGINE_D, logger)
    engine_c = load_engine(ENGINE_C, logger)

    A = TrtModule("EngineA_BackboneFPN", engine_a)
    B = TrtModule("EngineB_BEVFormerEncoder", engine_b)
    D = TrtModule("EngineD_StreamFusionNeck", engine_d)
    C = TrtModule("EngineC_MapDiffusionHead", engine_c)

    # Bind static shapes once with dummy inputs.
    dummy_img = np.zeros((1, 6, 3, 480, 800), dtype=np.float32)
    dummy_ego2img = np.zeros((1, 6, 4, 4), dtype=np.float32)
    dummy_bev = np.zeros((1, 256, 50, 100), dtype=np.float32)
    dummy_query = np.random.RandomState(args.seed).uniform(0.0, 1.0, size=(1, 100, 20, 2)).astype(np.float32)
    dummy_t = np.array([args.total_steps - 1], dtype=np.float32)
    dummy_qfeat = np.zeros((1, 100, 512), dtype=np.float32)

    A.bind_host_input("img", dummy_img)
    A.allocate_outputs()

    B.bind_device_input("feat0", A.output_ptr("feat0"), A.output_shape("feat0"))
    B.bind_device_input("feat1", A.output_ptr("feat1"), A.output_shape("feat1"))
    B.bind_device_input("feat2", A.output_ptr("feat2"), A.output_shape("feat2"))
    B.bind_host_input("ego2img", dummy_ego2img)
    B.allocate_outputs()

    D.bind_host_input("prev_bev", dummy_bev)
    D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.allocate_outputs()

    C.bind_host_input("bev_features", dummy_bev)
    C.bind_host_input("query_coords", dummy_query)
    C.bind_host_input("timestep", dummy_t)
    C.bind_host_input("prev_query_feat", dummy_qfeat)
    C.allocate_outputs()

    timesteps = get_timesteps(args.total_steps, args.sampling_timesteps)
    timesteps = [int(x) for x in np.linspace(total_steps_cfg, 0, sampling_timesteps_cfg + 1)]
    print("[sampling] timesteps:", timesteps)

    rng = np.random.RandomState(args.seed)

    submission = {
        "meta": cfg.eval_config.get("meta", getattr(dataset, "meta", {})),
        "results": {},
    }

    results_list = [None for _ in range(n_total)]

    prev_scene = None
    prev_bev_state = None
    prev_query_feat_state = np.zeros((1, 100, 512), dtype=np.float32)

    for count_i, dataset_idx in enumerate(indices):
        sample = dataset.samples[dataset_idx]
        token = sample["token"]
        scene = sample.get("scene_name", "")

        new_scene = scene != prev_scene
        if new_scene:
            print(f"[scene reset] idx={dataset_idx} scene={scene}")
            prev_bev_state = None
            prev_query_feat_state = np.zeros((1, 100, 512), dtype=np.float32)
            prev_scene = scene

        data = dataset[dataset_idx]
        meta = extract_meta(data)
        img = extract_img(data)
        ego2img = extract_ego2img(meta)

        A.bind_host_input("img", img)
        B.bind_host_input("ego2img", ego2img)

        A.execute()
        B.execute()

        if prev_bev_state is None:
            # First frame in scene: prev_bev = current raw BEV.
            D.bind_device_input("prev_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
        else:
            D.bind_host_input("prev_bev", prev_bev_state)

        D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
        D.execute()

        # Stable path: copy fused BEV to host and feed C.
        d_out = D.copy_outputs_to_host()
        fused_bev = np.ascontiguousarray(d_out["fused_bev"].astype(np.float32))
        if fused_bev.ndim == 3:
            fused_bev = fused_bev[None, ...]

        C.bind_host_input("bev_features", fused_bev)

        query_coords = make_initial_query_like_pytorch(rng, shape=(1, 100, 20, 2))
        final_out = None

        # Temporal query memory semantics:
        # Use previous-frame final query_feat as a fixed memory input for ALL
        # denoising steps of the current frame. Only update the frame-level
        # memory after the final diffusion step.
        prev_query_feat_input = np.ascontiguousarray(prev_query_feat_state.astype(np.float32))
        final_query_feat = prev_query_feat_input

        for step_i, t in enumerate(timesteps[:-1]):
            t_arr = np.array([t], dtype=np.float32)

            C.bind_host_input("query_coords", query_coords)
            C.bind_host_input("timestep", t_arr)
            C.bind_host_input("prev_query_feat", prev_query_feat_input)

            C.execute()
            out = C.copy_outputs_to_host()

            line_preds = np.ascontiguousarray(out["line_preds"].astype(np.float32))
            cls_logits = np.ascontiguousarray(out["cls_logits"].astype(np.float32))
            if "query_feat" in out:
                final_query_feat = np.ascontiguousarray(out["query_feat"].astype(np.float32))

            final_out = out

            if step_i < len(timesteps) - 1:
                time_next = timesteps[step_i + 1]
                query_coords, n_keep, n_refill = ddim_update_query_like_pytorch(
                    query_coords=query_coords,
                    line_preds=line_preds,
                    cls_logits=cls_logits,
                    time=t,
                    time_next=time_next,
                    coef=coef,
                    eta=eta,
                    query_threshold=query_threshold,
                    rng=rng,
                )

        final_line = np.ascontiguousarray(final_out["line_preds"].astype(np.float32))
        final_cls = np.ascontiguousarray(final_out["cls_logits"].astype(np.float32))

        result, n_saved, max_score, mean_score = format_result(final_line, final_cls, args.score_thr)

        submission["results"][token] = result
        results_list[dataset_idx] = result

        prev_bev_state = fused_bev
        prev_query_feat_state = final_query_feat

        if args.debug_save_npz:
            fdir = out_dir / f"debug_idx_{dataset_idx:06d}"
            fdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fdir / "trt_5step_outputs.npz",
                line_preds=final_line,
                cls_logits=final_cls,
                query_feat=prev_query_feat_state,
                fused_bev=fused_bev,
                timesteps=np.asarray(timesteps, dtype=np.float32),
            )

        print(
            f"[{count_i+1:04d}/{len(indices):04d}] "
            f"idx={dataset_idx} scene={scene} token={token} "
            f"saved={n_saved} score_max={max_score:.4f} score_mean={mean_score:.4f}"
        )

    results_pkl = out_dir / "trt_results.pkl"
    submission_json = out_dir / "submission_vector.json"

    mmcv.dump(submission, str(results_pkl))
    with open(submission_json, "w") as f:
        json.dump(to_jsonable(submission), f)

    print("=" * 100)
    print("saved results pkl:", results_pkl)
    print("saved submission:", submission_json)
    print("num results:", len(submission["results"]))
    print("=" * 100)

    os.chdir(old_cwd)


if __name__ == "__main__":
    main()
