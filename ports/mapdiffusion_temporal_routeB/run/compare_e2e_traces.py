from pathlib import Path
import numpy as np

PT_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/e2e_trace_pytorch_2")
TRT_ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/e2e_trace_trt_2")

def compare_arrays(name, p_pt, p_trt):
    if not p_pt.exists() or not p_trt.exists():
        print(name, "MISSING", p_pt.exists(), p_trt.exists())
        print(" PT :", p_pt)
        print(" TRT:", p_trt)
        return

    a = np.load(p_pt).astype(np.float32)
    b = np.load(p_trt).astype(np.float32)

    if a.shape != b.shape:
        print(name, "SHAPE", a.shape, b.shape)
        return

    diff = a - b
    denom = np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1)) + 1e-12
    cos = float(np.dot(a.reshape(-1), b.reshape(-1)) / denom)

    print("=" * 100)
    print(name)
    print("PT :", a.shape, float(a.min()), float(a.max()), float(a.mean()))
    print("TRT:", b.shape, float(b.min()), float(b.max()), float(b.mean()))
    print("max_abs:", float(np.max(np.abs(diff))))
    print("mean_abs:", float(np.mean(np.abs(diff))))
    print("rmse:", float(np.sqrt(np.mean(diff ** 2))))
    print("cosine:", cos)

for idx in [0, 1]:
    print("\n" + "#" * 140)
    print("IDX", idx)

    pt = PT_ROOT / f"debug_e2e_idx{idx:06d}"
    trt = TRT_ROOT / f"debug_e2e_idx{idx:06d}"

    print("PT dir exists:", pt.exists())
    print("TRT dir exists:", trt.exists())

    compare_arrays("00_img", pt / "00_img.npy", trt / "00_img.npy")

    # A features may have naming differences; compare what exists.
    for name in ["01_A_feat0", "01_A_feat1", "01_A_feat2"]:
        compare_arrays(name, pt / f"{name}.npy", trt / f"{name}.npy")

    # PyTorch saves final head BEV as 02_B_raw_bev_or_head_bev.
    compare_arrays("02_B_raw/head_bev", pt / "02_B_raw_bev_or_head_bev.npy", trt / "02_B_raw_bev.npy")
    compare_arrays("03_D_fused_bev", pt / "02_B_raw_bev_or_head_bev.npy", trt / "03_D_fused_bev.npy")

    for step in range(5):
        print("\n" + "-" * 100)
        print("STEP", step)

        engine_file = trt / f"04_C_step{step:02d}_engine.txt"
        if engine_file.exists():
            print("TRT engine:", engine_file.read_text().strip())

        compare_arrays(
            f"step{step}_C_input_bev",
            pt / f"04_C_step{step:02d}_input_bev.npy",
            trt / f"04_C_step{step:02d}_input_bev.npy",
        )
        compare_arrays(
            f"step{step}_C_input_query",
            pt / f"04_C_step{step:02d}_input_query_coords.npy",
            trt / f"04_C_step{step:02d}_input_query_coords.npy",
        )
        compare_arrays(
            f"step{step}_C_input_timestep",
            pt / f"04_C_step{step:02d}_input_timestep.npy",
            trt / f"04_C_step{step:02d}_input_timestep.npy",
        )
        compare_arrays(
            f"step{step}_C_input_prev_query_feat",
            pt / f"04_C_step{step:02d}_input_prev_query_feat.npy",
            trt / f"04_C_step{step:02d}_input_prev_query_feat.npy",
        )
        compare_arrays(
            f"step{step}_C_output_line",
            pt / f"05_C_step{step:02d}_output_line_preds.npy",
            trt / f"05_C_step{step:02d}_output_line_preds.npy",
        )
        compare_arrays(
            f"step{step}_C_output_cls",
            pt / f"05_C_step{step:02d}_output_cls_logits.npy",
            trt / f"05_C_step{step:02d}_output_cls_logits.npy",
        )
        compare_arrays(
            f"step{step}_C_output_query_feat",
            pt / f"05_C_step{step:02d}_output_query_feat.npy",
            trt / f"05_C_step{step:02d}_output_query_feat.npy",
        )
