import json
import numpy as np
from pathlib import Path

d = Path("model/mapdiffusion_routeA/parity_head")
data = json.loads((d / "trt_head_output.json").read_text())

def extract_numbers(obj):
    out = []
    if isinstance(obj, (float, int)):
        out.append(float(obj))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(extract_numbers(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(extract_numbers(v))
    return out

def find_named(obj, target):
    if isinstance(obj, dict):
        name = str(obj.get("name", ""))
        if target in name:
            vals = extract_numbers(obj)
            if vals:
                return vals
        for v in obj.values():
            got = find_named(v, target)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = find_named(v, target)
            if got is not None:
                return got
    return None

trt_line_vals = find_named(data, "line_preds")
trt_cls_vals = find_named(data, "cls_logits")

if trt_line_vals is None or trt_cls_vals is None:
    all_vals = np.asarray(extract_numbers(data), dtype=np.float32)
    trt_line = all_vals[:4000]
    trt_cls = all_vals[4000:4300]
else:
    trt_line = np.asarray(trt_line_vals, dtype=np.float32).reshape(-1)
    trt_cls = np.asarray(trt_cls_vals, dtype=np.float32).reshape(-1)

def metrics(pt, trt):
    pt = pt.astype(np.float32).reshape(-1)
    trt = trt.astype(np.float32).reshape(-1)
    diff = trt - pt
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "cos": float(np.dot(pt, trt) / ((np.linalg.norm(pt) * np.linalg.norm(trt)) + 1e-12)),
        "pt_min": float(pt.min()),
        "pt_max": float(pt.max()),
        "trt_min": float(trt.min()),
        "trt_max": float(trt.max()),
    }

best_line = None
best_cls = None

for i in range(6):
    pt_line = np.load(d / f"pytorch_layer{i}_line_preds.npy").reshape(-1)
    pt_cls = np.load(d / f"pytorch_layer{i}_cls_logits.npy").reshape(-1)

    ml = metrics(pt_line, trt_line)
    mc = metrics(pt_cls, trt_cls)

    print("=" * 80)
    print("layer", i)
    print("line:", ml)
    print("cls :", mc)

    if best_line is None or ml["mean_abs"] < best_line[1]["mean_abs"]:
        best_line = (i, ml)
    if best_cls is None or mc["mean_abs"] < best_cls[1]["mean_abs"]:
        best_cls = (i, mc)

print("=" * 80)
print("BEST line layer:", best_line[0], best_line[1])
print("BEST cls  layer:", best_cls[0], best_cls[1])
