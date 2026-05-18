import json
import re
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit("usage: python tools/trtexec_json_to_bins.py <trtexec_output.json> <out_dir>")

json_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(json_path.read_text())

def parse_dims(s):
    # TensorRT dimensions look like "(1x6x256x60x100)" or "1x6x..."
    if isinstance(s, list):
        return tuple(int(x) for x in s)
    s = str(s).strip().replace("(", "").replace(")", "")
    nums = re.split(r"[x,\s]+", s)
    return tuple(int(x) for x in nums if x)

def walk(obj):
    if isinstance(obj, dict):
        if "name" in obj and "values" in obj:
            yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

count = 0
for item in walk(data):
    name = str(item["name"])
    vals = np.asarray(item["values"], dtype=np.float32)

    dims = item.get("dimensions", None)
    if dims is not None:
        shape = parse_dims(dims)
        if np.prod(shape) == vals.size:
            vals = vals.reshape(shape)

    out_file = out_dir / f"{name}.fp32.bin"
    vals.astype(np.float32).tofile(out_file)

    npy_file = out_dir / f"{name}.npy"
    np.save(npy_file, vals.astype(np.float32))

    print(name, vals.shape, vals.dtype, float(vals.min()), float(vals.max()), "->", out_file)
    count += 1

print("converted tensors:", count)
