import json
import numpy as np
from pathlib import Path

d = Path("model/mapdiffusion_routeA/parity_head")

pt_cls_path = d / "pytorch_cls_logits.npy"
pt_line_path = d / "pytorch_line_preds.npy"
trt_json_path = d / "trt_head_output.json"

for p in [pt_cls_path, pt_line_path, trt_json_path]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

pt_cls = np.load(pt_cls_path).astype(np.float32).reshape(-1)
pt_line = np.load(pt_line_path).astype(np.float32).reshape(-1)

data = json.loads(trt_json_path.read_text())

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

if trt_cls_vals is None or trt_line_vals is None:
    print("Could not find named outputs. Extracting all numbers and splitting.")
    all_vals = np.asarray(extract_numbers(data), dtype=np.float32)

    n_line = pt_line.size
    n_cls = pt_cls.size

    # trtexec showed output order:
    # line_preds first, then cls_logits.
    trt_line = all_vals[:n_line]
    trt_cls = all_vals[n_line:n_line + n_cls]
else:
    trt_line = np.asarray(trt_line_vals, dtype=np.float32)
    trt_cls = np.asarray(trt_cls_vals, dtype=np.float32)

def report(name, pt, trt):
    print("=" * 80)
    print(name)
    print("pt elems:", pt.size)
    print("trt elems:", trt.size)

    if pt.size != trt.size:
        print("SIZE MISMATCH: trimming")
        n = min(pt.size, trt.size)
        pt = pt[:n]
        trt = trt[:n]

    diff = trt - pt

    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    cos = float(np.dot(pt, trt) / ((np.linalg.norm(pt) * np.linalg.norm(trt)) + 1e-12))

    print("max_abs_error:", max_abs)
    print("mean_abs_error:", mean_abs)
    print("rmse:", rmse)
    print("cosine:", cos)
    print("pt min/max:", float(pt.min()), float(pt.max()))
    print("trt min/max:", float(trt.min()), float(trt.max()))

report("line_preds", pt_line, trt_line)
report("cls_logits", pt_cls, trt_cls)
