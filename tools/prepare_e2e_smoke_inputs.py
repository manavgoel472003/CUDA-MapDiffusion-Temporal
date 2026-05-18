import numpy as np
from pathlib import Path

out = Path("model/cuda_bevformer/e2e_pipeline_smoke")
out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(123)

img = rng.normal(0, 1, size=(1, 6, 3, 480, 800)).astype(np.float32)

ego2img = np.zeros((1, 6, 4, 4), dtype=np.float32)
for i in range(6):
    ego2img[0, i] = np.eye(4, dtype=np.float32)

# Use parity head inputs if they exist, otherwise deterministic smoke inputs.
parity = Path("model/mapdiffusion_routeA/parity_head")
qc = parity / "query_coords.npy"
ts = parity / "timestep.npy"

if qc.exists():
    query_coords = np.load(qc).astype(np.float32)
else:
    query_coords = rng.uniform(0, 1, size=(1, 100, 20, 2)).astype(np.float32)

if ts.exists():
    timestep = np.load(ts).astype(np.float32)
else:
    timestep = np.array([1.0], dtype=np.float32)

img.tofile(out / "img.fp32.bin")
ego2img.tofile(out / "ego2img.fp32.bin")
query_coords.tofile(out / "query_coords.fp32.bin")
timestep.tofile(out / "timestep.fp32.bin")

print("saved:", out)
for name, arr in [
    ("img", img),
    ("ego2img", ego2img),
    ("query_coords", query_coords),
    ("timestep", timestep),
]:
    print(name, arr.shape, arr.dtype, float(arr.min()), float(arr.max()))
