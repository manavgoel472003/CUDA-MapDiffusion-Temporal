import argparse
from pathlib import Path
import numpy as np
import tensorrt as trt

from run_e2e_trt_mapdiffusion_5step_vis import (
    ENGINE_C,
    load_plugins,
    load_engine,
    TrtModule,
)


def metrics(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    d = a - b
    return {
        "max_abs": float(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d * d).mean())),
        "cosine": float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    trace = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    load_plugins()
    engine = load_engine(ENGINE_C, logger)

    bev = np.load(trace / "bev_features.npy").astype(np.float32)
    print("bev:", bev.shape, bev.min(), bev.max(), bev.mean())

    final_line = None
    final_cls = None

    for step in range(5):
        q = np.load(trace / f"step_{step:02d}_query_coords.npy").astype(np.float32)
        t = np.load(trace / f"step_{step:02d}_timestep.npy").astype(np.float32)

        C = TrtModule(f"EngineC_replay_step_{step}", engine)
        C.bind_host_input("bev_features", bev)
        C.bind_host_input("query_coords", q)
        C.bind_host_input("timestep", t)
        C.allocate_outputs()
        C.execute()
        pred = C.copy_outputs_to_host()

        line = pred["line_preds"].reshape(1, 100, 40).astype(np.float32)
        cls = pred["cls_logits"].reshape(1, 100, 3).astype(np.float32)

        np.save(out_dir / f"step_{step:02d}_trt_line_preds.npy", line)
        np.save(out_dir / f"step_{step:02d}_trt_cls_logits.npy", cls)

        pt_line = np.load(trace / f"step_{step:02d}_pt_line_preds.npy").astype(np.float32)
        pt_cls = np.load(trace / f"step_{step:02d}_pt_cls_logits.npy").astype(np.float32)

        score = sigmoid(cls[0]).max(axis=-1)

        print("=" * 100)
        print("step", step, "timestep", t.tolist())
        print("query:", q.shape, q.min(), q.max(), q.mean())
        print("TRT cls min/max/mean:", cls.min(), cls.max(), cls.mean())
        print("TRT score max/mean:", score.max(), score.mean(), ">0.5", int((score > 0.5).sum()))
        print("line metrics:", metrics(line, pt_line))
        print("cls  metrics:", metrics(cls, pt_cls))

        final_line = line
        final_cls = cls

    np.savez(
        out_dir / "trt_replay_official_trace_outputs.npz",
        line_preds=final_line.astype(np.float32),
        cls_logits=final_cls.astype(np.float32),
        prop_mask=np.zeros((100,), dtype=np.bool_),
    )

    print("saved:", out_dir / "trt_replay_official_trace_outputs.npz")


if __name__ == "__main__":
    main()
