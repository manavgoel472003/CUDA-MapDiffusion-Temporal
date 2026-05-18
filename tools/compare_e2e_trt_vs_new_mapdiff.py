import numpy as np
from pathlib import Path

TRT = Path("model/cuda_bevformer/e2e_pipeline_inmemory")
PT = Path("model/cuda_bevformer/e2e_parity_new_mapdiff")

pairs = [
    ("feat0", "trt_feat0.npy", "pt_feat0.npy"),
    ("feat1", "trt_feat1.npy", "pt_feat1.npy"),
    ("feat2", "trt_feat2.npy", "pt_feat2.npy"),
    ("bev_features", "trt_bev_features.npy", "pt_bev_features.npy"),
    ("line_preds", "trt_line_preds.npy", "pt_line_preds.npy"),
    ("cls_logits", "trt_cls_logits.npy", "pt_cls_logits.npy"),
]

def compare(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)

    if a.size != b.size:
        return None

    d = a - b
    return {
        "elems": int(a.size),
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "cosine": float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)),
        "trt_min": float(a.min()),
        "trt_max": float(a.max()),
        "pt_min": float(b.min()),
        "pt_max": float(b.max()),
        "trt_mean": float(a.mean()),
        "pt_mean": float(b.mean()),
    }

for label, trt_name, pt_name in pairs:
    print("=" * 100)
    print(label)

    trt_path = TRT / trt_name
    pt_path = PT / pt_name

    if not trt_path.exists():
        print("MISSING TRT:", trt_path)
        continue

    if not pt_path.exists():
        print("MISSING PT:", pt_path)
        continue

    trt = np.load(trt_path)
    pt = np.load(pt_path)

    print("trt shape:", trt.shape)
    print("pt  shape:", pt.shape)

    if trt.shape != pt.shape:
        print("SHAPE MISMATCH")
        print("trt elems:", trt.size)
        print("pt elems:", pt.size)

    m = compare(trt, pt)
    if m is None:
        print("Cannot compare: different element counts")
        continue

    for k, v in m.items():
        print(f"{k}: {v}")

print("=" * 100)
print("DONE")
