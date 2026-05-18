import json
import numpy as np
from pathlib import Path

d = Path("model/mapdiffusion_routeA/parity_head")

pt_cls = np.load(d / "pytorch_onnx_weight_cls_logits.npy").astype(np.float32).reshape(-1)
pt_line = np.load(d / "pytorch_onnx_weight_line_preds.npy").astype(np.float32).reshape(-1)

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
        if target in str(obj.get("name", "")):
            return np.asarray(obj["values"], dtype=np.float32)
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

trt_cls = find_named(data, "cls_logits")
trt_line = find_named(data, "line_preds")

if trt_cls is None or trt_line is None:
    vals = np.asarray(extract_numbers(data), dtype=np.float32)
    trt_cls = vals[:pt_cls.size]
    trt_line = vals[pt_cls.size:pt_cls.size + pt_line.size]

trt_cls = trt_cls.reshape(-1)
trt_line = trt_line.reshape(-1)

def report(name, pt, trt):
    diff = trt - pt
    print("=" * 80)
    print(name)
    print("pt elems:", pt.size, "trt elems:", trt.size)
    print("max_abs_error:", float(np.max(np.abs(diff))))
    print("mean_abs_error:", float(np.mean(np.abs(diff))))
    print("rmse:", float(np.sqrt(np.mean(diff * diff))))
    print("cosine:", float(np.dot(pt, trt) / ((np.linalg.norm(pt) * np.linalg.norm(trt)) + 1e-12)))
    print("pt min/max:", float(pt.min()), float(pt.max()))
    print("trt min/max:", float(trt.min()), float(trt.max()))

report("line_preds", pt_line, trt_line)
report("cls_logits", pt_cls, trt_cls)
