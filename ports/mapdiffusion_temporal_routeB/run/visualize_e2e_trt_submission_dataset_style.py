import argparse
import os
import sys
import json
import importlib
from pathlib import Path

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset


MAPDIFF_ROOT = Path("/home/018198687/Mapping/mapdiffusion")


def parse_args():
    p = argparse.ArgumentParser("Visualize TRT E2E MapDiffusion submission with dataset.show_result/show_gt")
    p.add_argument("--config", required=True)
    p.add_argument("--submission", required=True, help="submission_vector.json")
    p.add_argument("--idx", type=int, default=None, help="dataset idx to visualize")
    p.add_argument("--token", default=None, help="sample token to visualize")
    p.add_argument("--scene-index", type=int, default=None, help="visualize full scene by sorted scene index")
    p.add_argument("--max-frames", type=int, default=999999)
    p.add_argument("--thr", type=float, default=0.4)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def import_plugins(cfg):
    sys.path.insert(0, str(MAPDIFF_ROOT))

    if getattr(cfg, "plugin", False):
        plugin_dirs = cfg.plugin_dir
        if not isinstance(plugin_dirs, list):
            plugin_dirs = [plugin_dirs]
        for plugin_dir in plugin_dirs:
            module_path = os.path.dirname(plugin_dir).replace("/", ".")
            print("[plugin]", module_path)
            importlib.import_module(module_path)

    if hasattr(cfg, "custom_imports"):
        for imp in cfg.custom_imports.get("imports", []):
            print("[custom_import]", imp)
            importlib.import_module(imp)


def main():
    args = parse_args()

    # Renderer expects resources/car.png relative to mapdiffusion repo.
    os.chdir(str(MAPDIFF_ROOT))

    cfg = Config.fromfile(args.config)
    import_plugins(cfg)

    dataset = build_dataset(cfg.eval_config)
    print("[dataset len]", len(dataset))

    submission = json.load(open(args.submission))
    print("[submission tokens]", len(submission["results"]))

    token_to_idx = {}
    scene_to_indices = {}

    for i, sample in enumerate(dataset.samples):
        token = sample["token"]
        scene = sample.get("scene_name", "unknown_scene")
        token_to_idx[token] = i
        scene_to_indices.setdefault(scene, []).append(i)

    if args.token is not None:
        indices = [token_to_idx[args.token]]
    elif args.scene_index is not None:
        scenes = sorted(scene_to_indices.keys())
        scene = scenes[args.scene_index]
        indices = scene_to_indices[scene][: args.max_frames]
        print("[scene]", scene, "frames", len(indices))
    elif args.idx is not None:
        indices = [args.idx]
    else:
        # Default: visualize first available token in submission.
        first_token = next(iter(submission["results"].keys()))
        indices = [token_to_idx[first_token]]
        print("[default token]", first_token)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for local_i, idx in enumerate(indices):
        token = dataset.samples[idx]["token"]

        if token not in submission["results"]:
            print("[skip missing token]", idx, token)
            continue

        frame_dir = out_root / f"idx_{idx:06d}_{local_i:04d}"
        pred_dir = frame_dir / "pred"
        gt_dir = frame_dir / "gt"
        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        print("[render]", "idx", idx, "token", token, "thr", args.thr)

        dataset.show_result(
            submission=submission,
            idx=idx,
            score_thr=args.thr,
            out_dir=str(pred_dir),
        )

        dataset.show_gt(idx, str(gt_dir))

    print("[saved]", out_root)


if __name__ == "__main__":
    main()
