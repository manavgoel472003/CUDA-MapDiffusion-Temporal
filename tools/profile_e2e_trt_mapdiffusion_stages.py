import os
import time
import numpy as np

from run_e2e_trt_mapdiffusion_5step_vis import (
    ENGINE_A, ENGINE_B, ENGINE_C, ENGINE_D,
    load_plugins, load_engine, TrtModule, create_inputs
)

import tensorrt as trt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_ddpm_coef(total_steps, schedule="cosine"):
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    if schedule == "linear":
        scale = 1000 / total_steps
        betas = np.linspace(scale * 1e-4, scale * 2e-2, total_steps)
    elif schedule == "cosine":
        betas = betas_for_alpha_bar(
            total_steps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(schedule)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    return {
        "alphas_cumprod": alphas_cumprod,
        "pred_coef1": np.sqrt(1.0 / alphas_cumprod),
        "pred_coef2": np.sqrt(1.0 / alphas_cumprod - 1),
    }


def predict_noise_from_start(coef, x_t, t, x0):
    return (coef["pred_coef1"][t - 1] * x_t - x0) / coef["pred_coef2"][t - 1]


def main():
    seed = int(os.environ.get("E2E_SEED", "123"))
    rng = np.random.default_rng(seed)

    total_steps = int(os.environ.get("E2E_TOTAL_STEPS", "1000"))
    scheduler = os.environ.get("E2E_SCHEDULER", "cosine")
    sampling_timesteps = int(os.environ.get("E2E_SAMPLING_TIMESTEPS", "5"))
    eta = float(os.environ.get("E2E_ETA", "0.5"))
    query_threshold = float(os.environ.get("E2E_QUERY_THRESHOLD", "0.5"))
    warmup = int(os.environ.get("E2E_WARMUP", "10"))
    iters = int(os.environ.get("E2E_ITERS", "100"))

    coef = compute_ddpm_coef(total_steps, scheduler)

    times = np.linspace(0, total_steps, sampling_timesteps + 1).astype(np.int32)
    times = list(reversed(times.tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    print("seed:", seed)
    print("total_steps:", total_steps)
    print("scheduler:", scheduler)
    print("sampling_timesteps:", sampling_timesteps)
    print("eta:", eta)
    print("query_threshold:", query_threshold)
    print("warmup:", warmup)
    print("iters:", iters)
    print("time_pairs:", time_pairs)

    logger = trt.Logger(trt.Logger.WARNING)
    load_plugins()

    engine_a = load_engine(ENGINE_A, logger)
    engine_b = load_engine(ENGINE_B, logger)
    engine_d = load_engine(ENGINE_D, logger)
    engine_c = load_engine(ENGINE_C, logger)

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

    # B -> D -> C stays on device.
    D.bind_device_input("prev_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.allocate_outputs()

    C.bind_device_input("bev_features", D.output_ptr("fused_bev"), D.output_shape("fused_bev"))
    C.allocate_outputs()

    def run_once():
        timing = {
            "A_ms": 0.0,
            "B_ms": 0.0,
            "D_ms": 0.0,
            "C_execute_total_ms": 0.0,
            "C_copy_total_ms": 0.0,
            "sampler_update_total_ms": 0.0,
            "total_ms": 0.0,
        }

        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        A.execute()
        timing["A_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        B.execute()
        timing["B_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        D.execute()
        timing["D_ms"] = (time.perf_counter() - t0) * 1000.0

        query_coords = np.ascontiguousarray(initial_query_coords.copy().astype(np.float32))
        poly_class = None
        prop_mask = np.zeros((100,), dtype=np.bool_)

        c_execs = []
        c_copies = []
        sampler_updates = []

        for step, (time_step, time_next) in enumerate(time_pairs):
            timestep = np.array([float(time_step)], dtype=np.float32)

            C.bind_host_input("query_coords", query_coords)
            C.bind_host_input("timestep", timestep)

            t0 = time.perf_counter()
            C.execute()
            c_exec_ms = (time.perf_counter() - t0) * 1000.0
            c_execs.append(c_exec_ms)

            t0 = time.perf_counter()
            out = C.copy_outputs_to_host()
            c_copy_ms = (time.perf_counter() - t0) * 1000.0
            c_copies.append(c_copy_ms)

            t0 = time.perf_counter()
            x_start = out["line_preds"].reshape(1, 100, 20, 2).astype(np.float32)
            x_start_class = out["cls_logits"].reshape(1, 100, 3).astype(np.float32)
            poly_class = x_start_class

            score_per_image = sigmoid(x_start_class[0])
            value = score_per_image.max(axis=-1)
            keep_idx = value > query_threshold
            prop_mask = np.zeros((100,), dtype=np.bool_)

            if time_next == 0:
                query_coords = x_start
                sampler_updates.append((time.perf_counter() - t0) * 1000.0)
                continue

            pred_noise = predict_noise_from_start(coef, query_coords, time_step, x_start)

            pred_noise = pred_noise[:, keep_idx, :, :]
            x_start_keep = x_start[:, keep_idx, :, :]

            alpha = coef["alphas_cumprod"][time_step - 1]
            alpha_next = coef["alphas_cumprod"][time_next - 1]
            sigma = eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = np.sqrt(1 - alpha_next - sigma ** 2)

            noise = rng.normal(0.0, 0.25, size=x_start_keep.shape).astype(np.float32)
            noise = np.clip(noise, 0.0, 1.0)

            query_coords = x_start_keep * np.sqrt(alpha_next) + c * pred_noise + sigma * noise
            query_coords = np.clip(query_coords, 0.0, 1.0).astype(np.float32)

            num_remain = int(keep_idx.sum())
            refill = 100 - num_remain
            noise_new = rng.normal(0.5, 0.25, size=(1, refill, 20, 2)).astype(np.float32)
            noise_new = np.clip(noise_new, 0.0, 1.0)
            query_coords = np.ascontiguousarray(np.concatenate([query_coords, noise_new], axis=1).astype(np.float32))

            sampler_updates.append((time.perf_counter() - t0) * 1000.0)

        timing["C_execute_total_ms"] = float(np.sum(c_execs))
        timing["C_copy_total_ms"] = float(np.sum(c_copies))
        timing["sampler_update_total_ms"] = float(np.sum(sampler_updates))
        timing["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
        timing["C_exec_each_ms"] = c_execs
        timing["C_copy_each_ms"] = c_copies
        timing["sampler_update_each_ms"] = sampler_updates

        scores = sigmoid(poly_class[0]).max(axis=-1)
        return timing, scores

    for _ in range(warmup):
        run_once()

    all_timings = []
    final_scores = None
    for _ in range(iters):
        timing, scores = run_once()
        all_timings.append(timing)
        final_scores = scores

    def avg(key):
        return float(np.mean([x[key] for x in all_timings]))

    print("=" * 100)
    print("Latency breakdown avg over", iters, "iters")
    for key in [
        "A_ms",
        "B_ms",
        "D_ms",
        "C_execute_total_ms",
        "C_copy_total_ms",
        "sampler_update_total_ms",
        "total_ms",
    ]:
        print(f"{key}: {avg(key):.3f} ms")

    c_each = np.asarray([x["C_exec_each_ms"] for x in all_timings], dtype=np.float32)
    copy_each = np.asarray([x["C_copy_each_ms"] for x in all_timings], dtype=np.float32)
    sampler_each = np.asarray([x["sampler_update_each_ms"] for x in all_timings], dtype=np.float32)

    print("C_execute_each_avg_ms:", np.mean(c_each, axis=0).round(3).tolist())
    print("C_copy_each_avg_ms:", np.mean(copy_each, axis=0).round(3).tolist())
    print("sampler_update_each_avg_ms:", np.mean(sampler_each, axis=0).round(3).tolist())

    print("=" * 100)
    print("final score max/mean:", float(final_scores.max()), float(final_scores.mean()))
    print("final >0.5:", int((final_scores > 0.5).sum()))


if __name__ == "__main__":
    main()
