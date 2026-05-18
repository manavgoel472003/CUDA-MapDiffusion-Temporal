import argparse
import importlib
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset


MAPDIFF_ROOT = "/home/018198687/Mapping/mapdiffusion"
if MAPDIFF_ROOT not in sys.path:
    sys.path.insert(0, MAPDIFF_ROOT)


def import_plugin(cfg):
    sys.path.append(os.path.abspath(MAPDIFF_ROOT))
    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir
        if not isinstance(plugin_dirs, list):
            plugin_dirs = [plugin_dirs]
        for plugin_dir in plugin_dirs:
            module_dir = os.path.dirname(plugin_dir)
            module_path = module_dir.replace("/", ".")
            print("[plugin]", module_path)
            importlib.import_module(module_path)

    if hasattr(cfg, "custom_imports"):
        for imp in cfg.custom_imports.get("imports", []):
            print("[custom_import]", imp)
            importlib.import_module(imp)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def denorm_lines_60x30(lines):
    """
    TRT head outputs normalized xy in [0, 1].
    Convert to MapDiffusion local BEV meters:
      x: [-30, 30]
      y: [-15, 15]
    """
    pts = lines.reshape(-1, 20, 2).astype(np.float32).copy()
    pts[..., 0] = pts[..., 0] * 60.0 - 30.0
    pts[..., 1] = pts[..., 1] * 30.0 - 15.0
    return pts


def load_trt_outputs(path):
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        print("[npz keys]", keys)

        # Support common names from our runners.
        line_key = None
        cls_key = None
        for k in ["line_preds", "final_line_preds", "trt_line_preds", "step_04_trt_line_preds"]:
            if k in data:
                line_key = k
                break
        for k in ["cls_logits", "final_cls_logits", "trt_cls_logits", "step_04_trt_cls_logits"]:
            if k in data:
                cls_key = k
                break

        if line_key is None:
            line_candidates = [k for k in keys if "line" in k and data[k].shape[-1] == 40]
            if line_candidates:
                line_key = line_candidates[-1]
        if cls_key is None:
            cls_candidates = [k for k in keys if "cls" in k and data[k].shape[-1] in (3, 4)]
            if cls_candidates:
                cls_key = cls_candidates[-1]

        if line_key is None or cls_key is None:
            raise RuntimeError(f"Could not infer line/cls keys from {keys}")

        print("[using]", line_key, cls_key)
        line_preds = data[line_key]
        cls_logits = data[cls_key]
        return line_preds, cls_logits

    # Directory mode: find final debug npy files.
    if path.is_dir():
        candidates_line = sorted(path.glob("*line*.npy"))
        candidates_cls = sorted(path.glob("*cls*.npy"))
        print("[line candidates]", [str(x) for x in candidates_line])
        print("[cls candidates]", [str(x) for x in candidates_cls])
        if not candidates_line or not candidates_cls:
            raise RuntimeError(f"No line/cls npy files found in {path}")
        return np.load(candidates_line[-1]), np.load(candidates_cls[-1])

    raise RuntimeError(f"Unsupported input: {path}")


def make_single_result(line_preds, cls_logits):
    line_preds = np.asarray(line_preds).astype(np.float32)
    cls_logits = np.asarray(cls_logits).astype(np.float32)

    if line_preds.ndim == 3:
        line_preds = line_preds[0]
    if cls_logits.ndim == 3:
        cls_logits = cls_logits[0]

    vectors = denorm_lines_60x30(line_preds)
    probs = sigmoid(cls_logits)
    labels = probs.argmax(axis=-1).astype(np.int64)
    scores = probs.max(axis=-1).astype(np.float32)

    return {
        "vectors": [v for v in vectors],
        "scores": [float(s) for s in scores],
        "labels": [int(l) for l in labels],
        "prop": [True for _ in range(len(vectors))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trt-out", required=True, help="NPZ or debug dir containing final line_preds/cls_logits")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--thr", type=float, default=0.2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-pkl", default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    dataset = build_dataset(cfg.eval_config)
    print("[dataset len]", len(dataset))
    print("[idx]", args.idx)
    sample = dataset.samples[args.idx]
    print("[token]", sample.get("token"))
    print("[scene]", sample.get("scene_name"))

    line_preds, cls_logits = load_trt_outputs(args.trt_out)
    result = make_single_result(line_preds, cls_logits)

    token = sample.get("token")
    submission = {
        "meta": cfg.eval_config.get("meta", getattr(dataset, "meta", {})),
        "results": {
            token: result,
        },
    }

    if args.save_pkl:
        mmcv.dump(submission, args.save_pkl)
        print("[saved pkl]", args.save_pkl)

    out_dir = Path(args.out_dir).resolve()
    pred_dir = out_dir / "pred"
    gt_dir = out_dir / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # MapDiffusion renderer loads resources/car.png relative to cwd.
    old_cwd = os.getcwd()
    os.chdir(MAPDIFF_ROOT)
    try:
        print("[show_result] thr", args.thr, "out", pred_dir)
        dataset.show_result(
            submission=submission,
            idx=args.idx,
            score_thr=args.thr,
            out_dir=str(pred_dir),
        )

        print("[show_gt] out", gt_dir)
        dataset.show_gt(args.idx, str(gt_dir))
    finally:
        os.chdir(old_cwd)

    print("[done]")
    print("pred_dir:", pred_dir)
    print("gt_dir:", gt_dir)


if __name__ == "__main__":
    main()
