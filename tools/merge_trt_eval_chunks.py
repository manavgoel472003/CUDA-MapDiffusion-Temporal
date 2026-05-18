import argparse
from pathlib import Path
import re
import mmcv

def start_num(p):
    m = re.search(r"chunk_(\d+)", str(p))
    return int(m.group(1)) if m else 10**18

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = sorted(
        [p for p in chunks_dir.glob("chunk_*") if p.is_dir()],
        key=start_num
    )

    all_results = []
    merged_submission = None
    total_tokens = 0

    for d in chunk_dirs:
        pkl = d / "trt_results.pkl"
        js = d / "submission_vector.json"
        if not pkl.exists() or not js.exists():
            print("SKIP incomplete:", d)
            continue

        results = mmcv.load(str(pkl))
        sub = mmcv.load(str(js))

        print(d.name, "results:", len(results), "tokens:", len(sub["results"]))

        all_results.extend(results)

        if merged_submission is None:
            merged_submission = {
                "meta": sub["meta"],
                "results": {}
            }

        merged_submission["results"].update(sub["results"])
        total_tokens += len(sub["results"])

    out_pkl = out_dir / "trt_results.pkl"
    out_json = out_dir / "submission_vector.json"

    mmcv.dump(all_results, str(out_pkl))
    mmcv.dump(merged_submission, str(out_json))

    print("=" * 100)
    print("merged results:", len(all_results))
    print("merged tokens:", len(merged_submission["results"]))
    print("saved:", out_pkl)
    print("saved:", out_json)

if __name__ == "__main__":
    main()
