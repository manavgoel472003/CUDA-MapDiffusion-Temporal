import os
import json
from pathlib import Path
import numpy as np
import tensorrt as trt
from cuda import cudart

from run_e2e_trt_mapdiffusion_5step_vis import (
    ROOT,
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    ENGINE_D,
    load_plugins,
    load_engine,
    TrtModule,
    create_inputs,
    memcpy_h2d,
    check_cuda,
    sync,
    make_event,
)

OUT_DIR = ROOT / "model/cuda_bevformer/e2e_pipeline_official_sampler"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_betas(total_steps, schedule="cosine"):
    if schedule == "linear":
        scale = 1000 / total_steps
        start, end = scale * 1e-4, scale * 2e-2
        return np.linspace(start, end, total_steps)

    if schedule == "cosine":
        betas = []
        for i in range(total_steps):
            t1 = i / total_steps
            t2 = (i + 1) / total_steps
            alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas)

    raise NotImplementedError(schedule)


def compute_ddpm_coef(total_steps=1000, schedule="cosine"):
    betas = make_betas(total_steps, schedule)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "pred_coef1": np.sqrt(1.0 / alphas_cumprod),
        "pred_coef2": np.sqrt(1.0 / alphas_cumprod - 1),
        "variance": variance,
        "posterior_log_variance_clipped": np.log(np.maximum(variance, 1e-20)),
        "posterior_mean_coef1": betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
    }


def predict_noise_from_start(coef, x_t, t, x0):
    return (coef["pred_coef1"][t - 1] * x_t - x0) / coef["pred_coef2"][t - 1]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    seed = int(os.environ.get("E2E_SEED", "123"))
    rng = np.random.default_rng(seed)

    total_steps = int(os.environ.get("E2E_TOTAL_STEPS", "1000"))
    sampling_timesteps = int(os.environ.get("E2E_SAMPLING_TIMESTEPS", "5"))
    eta = float(os.environ.get("E2E_ETA", "0.5"))
    query_threshold = float(os.environ.get("E2E_QUERY_THRESHOLD", "0.5"))
    scheduler = os.environ.get("E2E_SCHEDULER", "cosine")

    coef = compute_ddpm_coef(total_steps, scheduler)

    times = np.linspace(0, total_steps, sampling_timesteps + 1).astype(np.int32).tolist()
    times = list(reversed(times))
    time_pairs = list(zip(times[:-1], times[1:]))

    print("Official MapDiffusion TRT sampler")
    print("seed:", seed)
    print("total_steps:", total_steps)
    print("scheduler:", scheduler)
    print("eta:", eta)
    print("query_threshold:", query_threshold)
    print("time_pairs:", time_pairs)

    logger = trt.Logger(trt.Logger.WARNING)

    load_plugins()
    engine_a = load_engine(ENGINE_A, logger)
    engine_b = load_engine(ENGINE_B, logger)
    engine_c = load_engine(ENGINE_C, logger)
    engine_d = load_engine(ENGINE_D, logger)

    img, ego2img, initial_query_coords = create_inputs()

    A = TrtModule("EngineA_BackboneFPN", engine_a)
    B = TrtModule("EngineB_BEVFormerEncoder", engine_b)
    D = TrtModule("EngineD_StreamFusionNeck", engine_d)
    C = TrtModule("EngineC_MapDiffusionHead", engine_c)

    A.bind_host_input("img", img)
    A.allocate_outputs()

    B.bind_device_input("feat0", A.output_ptr("feat0"), A.output_shape("feat0"))
    B.bind_device_input("feat1", A.output_ptr("feat1"), A.output_shape("feat1"))
    B.bind_device_input("feat2", A.output_ptr("feat2"), A.output_shape("feat2"))
    B.bind_host_input("ego2img", ego2img)
    B.allocate_outputs()

    # DISABLED: Head C must consume fused BEV from EngineD, not raw EngineB BEV.

    query_buf = np.zeros((1, 100, 20, 2), dtype=np.float32)
    timestep_buf = np.zeros((1,), dtype=np.float32)

    query_ptr = C.bind_host_input("query_coords", query_buf)
    timestep_ptr = C.bind_host_input("timestep", timestep_buf)
    C.allocate_outputs()

    def run_once(verbose=False):
        A.execute()
        B.execute()

        # Streaming BEV first-frame path:
        # official PyTorch does stream_fusion_neck(curr_bev.detach(), curr_bev).
        # Optimized path: keep B -> D -> C fully on GPU.
        D.bind_device_input("prev_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
        D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
        D.allocate_outputs()
        D.execute()

        # IMPORTANT: Head C expects the streaming-fused BEV from EngineD, not raw EngineB BEV.
        C.bind_device_input("bev_features", D.output_ptr("fused_bev"), D.output_shape("fused_bev"))

        save_debug = os.environ.get("E2E_SAVE_DEBUG", "0") == "1"
        if save_debug:
            # Only copy for debugging/parity. Keep this disabled during latency runs.
            d_out = D.copy_outputs_to_host()
            fused_bev = np.ascontiguousarray(d_out["fused_bev"].astype(np.float32))
        debug_dir = Path(os.environ.get("E2E_DEBUG_DIR", str(OUT_DIR)))
        if save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            # Raw B output is optional and costs an extra copy.
            b_out = B.copy_outputs_to_host()
            np.save(debug_dir / "debug_raw_bev_features.npy", b_out["bev_features"].astype(np.float32))
            np.save(debug_dir / "debug_bev_features.npy", fused_bev.astype(np.float32))
            print("[debug] saved raw:", debug_dir / "debug_raw_bev_features.npy")
            print("[debug] saved fused:", debug_dir / "debug_bev_features.npy")

        # Use the initial query returned by create_inputs().
        # This may come from E2E_QUERY_COORDS, or from official clipped-normal random init.
        query_coords = np.ascontiguousarray(initial_query_coords.copy().astype(np.float32))
        print("[sampler] initial query used",
              query_coords.shape,
              float(query_coords.min()),
              float(query_coords.max()),
              float(query_coords.mean()))

        poly_class = None
        prop_mask = np.zeros((100,), dtype=np.bool_)

        step_records = []

        for step_idx, (time, time_next) in enumerate(time_pairs):
            timestep = np.array([float(time)], dtype=np.float32)

            if os.environ.get("E2E_SAVE_DEBUG", "0") == "1":
                debug_dir = Path(os.environ.get("E2E_DEBUG_DIR", str(OUT_DIR)))
                np.save(debug_dir / f"debug_step_{step_idx:02d}_query_coords.npy", query_coords.astype(np.float32))
                np.save(debug_dir / f"debug_step_{step_idx:02d}_timestep.npy", timestep.astype(np.float32))

            memcpy_h2d(query_ptr, query_coords.astype(np.float32))
            memcpy_h2d(timestep_ptr, timestep)

            C.execute()
            out = C.copy_outputs_to_host()

            line_preds = out["line_preds"].reshape(1, 100, 40).astype(np.float32)
            cls_logits = out["cls_logits"].reshape(1, 100, 3).astype(np.float32)

            if os.environ.get("E2E_SAVE_DEBUG", "0") == "1":
                debug_dir = Path(os.environ.get("E2E_DEBUG_DIR", str(OUT_DIR)))
                np.save(debug_dir / f"debug_step_{step_idx:02d}_trt_line_preds.npy", line_preds)
                np.save(debug_dir / f"debug_step_{step_idx:02d}_trt_cls_logits.npy", cls_logits)
                print("[debug] saved step", step_idx, "query/timestep/trt outputs")

            x_start = line_preds.reshape(1, -1, 20, 2)
            x_start_class = cls_logits
            poly_class = x_start_class

            if verbose:
                probs = sigmoid(x_start_class[0])
                max_scores = probs.max(axis=-1)
                print("=" * 80)
                print("step:", step_idx, "time:", time, "time_next:", time_next)
                print("query:", query_coords.shape, float(query_coords.min()), float(query_coords.max()), float(query_coords.mean()))
                print("x_start:", x_start.shape, float(x_start.min()), float(x_start.max()), float(x_start.mean()))
                print("cls:", x_start_class.shape, float(x_start_class.min()), float(x_start_class.max()), float(x_start_class.mean()))
                print("score max/mean:", float(max_scores.max()), float(max_scores.mean()))

            step_records.append({
                "step": step_idx,
                "time": time,
                "time_next": time_next,
                "line_preds": line_preds.copy(),
                "cls_logits": cls_logits.copy(),
            })

            if time_next == 0:
                query_coords = x_start.astype(np.float32)
                break

            pred_noise = predict_noise_from_start(coef, query_coords, time, x_start)

            score_per_image = sigmoid(x_start_class[0])
            value = score_per_image.max(axis=-1)
            keep_idx = value > query_threshold
            num_remain = int(keep_idx.sum())

            pred_noise = pred_noise[:, keep_idx, :, :]
            x_start_keep = x_start[:, keep_idx, :, :]

            alpha = coef["alphas_cumprod"][time - 1]
            alpha_next = coef["alphas_cumprod"][time_next - 1]

            sigma = eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = np.sqrt(1 - alpha_next - sigma ** 2)

            noise = rng.normal(0.0, 0.25, size=x_start_keep.shape).astype(np.float32)
            noise = np.clip(noise, 0.0, 1.0)

            query_coords = (
                x_start_keep * np.sqrt(alpha_next)
                + c * pred_noise
                + sigma * noise
            )
            query_coords = np.clip(query_coords, 0.0, 1.0).astype(np.float32)

            noise_new = rng.normal(0.5, 0.25, size=(1, 100 - num_remain, 20, 2)).astype(np.float32)
            noise_new = np.clip(noise_new, 0.0, 1.0)

            query_coords = np.concatenate([query_coords, noise_new], axis=1).astype(np.float32)

            if verbose:
                print("num_remain:", num_remain, "refill:", 100 - num_remain)
                print("next query:", query_coords.shape, float(query_coords.min()), float(query_coords.max()), float(query_coords.mean()))

        final_lines = query_coords.reshape(1, -1, 40).astype(np.float32)
        final_scores = poly_class.astype(np.float32)

        return final_lines, final_scores, prop_mask, step_records

    print("\n============================================================")
    print("Smoke official sampler")
    print("============================================================")
    final_lines, final_scores, prop_mask, step_records = run_once(verbose=True)
    sync()

    np.savez(
        OUT_DIR / "trt_official_sampler_outputs.npz",
        line_preds=final_lines,
        cls_logits=final_scores,
        prop_mask=prop_mask,
        time_pairs=np.array(time_pairs, dtype=np.int32),
    )

    print("final line_preds:", final_lines.shape, final_lines.dtype, float(final_lines.min()), float(final_lines.max()), float(final_lines.mean()))
    print("final cls_logits:", final_scores.shape, final_scores.dtype, float(final_scores.min()), float(final_scores.max()), float(final_scores.mean()))
    print("saved:", OUT_DIR / "trt_official_sampler_outputs.npz")

    if os.environ.get("E2E_SKIP_LATENCY", "0") == "1":
        print("E2E_SKIP_LATENCY=1, stopping after correctness smoke output.")
        return

    print("\n============================================================")
    print("Latency: official A+B+5C sampler")
    print("============================================================")

    warmup = int(os.environ.get("E2E_WARMUP", "5"))
    iters = int(os.environ.get("E2E_ITERS", "20"))

    for _ in range(warmup):
        run_once(verbose=False)
    sync()

    start = make_event()
    end = make_event()

    check_cuda(cudart.cudaEventRecord(start, 0), "record start failed")
    for _ in range(iters):
        run_once(verbose=False)
    check_cuda(cudart.cudaEventRecord(end, 0), "record end failed")
    check_cuda(cudart.cudaEventSynchronize(end), "event sync failed")

    err, total_ms = cudart.cudaEventElapsedTime(start, end)
    check_cuda(err, "elapsed failed")

    print("warmup:", warmup)
    print("iters:", iters)
    print(f"avg official sampler latency: {total_ms / iters:.3f} ms")
    print("Note: this correctness runner performs CPU-side sampler updates between TRT head calls.")


if __name__ == "__main__":
    main()
