#!/usr/bin/env bash
set -euo pipefail

cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion

BASE_OUT=${BASE_OUT:-/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/trt_eval_full_val_seed123_temporal_routeB_chunks}
CONFIG=${CONFIG:-/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/temporal_config.py}

TOTAL=${TOTAL:-5981}
CHUNK=${CHUNK:-500}
SEED=${SEED:-123}

mkdir -p "$BASE_OUT"

echo "============================================================"
echo "Temporal Route B chunk inference"
echo "BASE_OUT=$BASE_OUT"
echo "CONFIG=$CONFIG"
echo "TOTAL=$TOTAL CHUNK=$CHUNK SEED=$SEED"
echo "Start: $(date)"
echo "============================================================"

for START in $(seq 0 "$CHUNK" $((TOTAL - 1))); do
  OUTDIR="$BASE_OUT/chunk_${START}"
  mkdir -p "$OUTDIR"

  if [ -f "$OUTDIR/trt_results.pkl" ] && [ -f "$OUTDIR/submission_vector.json" ]; then
    echo "Skipping completed chunk START=$START"
    continue
  fi

  LIMIT=$CHUNK
  REMAIN=$((TOTAL - START))
  if [ "$REMAIN" -lt "$CHUNK" ]; then
    LIMIT=$REMAIN
  fi

  echo "============================================================"
  echo "Running chunk START=$START LIMIT=$LIMIT"
  echo "Output: $OUTDIR"
  echo "Time: $(date)"
  echo "============================================================"

  CUDA_LAUNCH_BLOCKING=1 python ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_val_submission.py \
    --config "$CONFIG" \
    --out-dir "$OUTDIR" \
    --start "$START" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    2>&1 | tee "$OUTDIR/run.log"

  if [ ! -f "$OUTDIR/submission_vector.json" ] || [ ! -f "$OUTDIR/trt_results.pkl" ]; then
    echo "FAILED: missing output files in $OUTDIR"
    exit 1
  fi

  echo "Completed chunk START=$START at $(date)"
done

echo "============================================================"
echo "Completed all chunks at $(date)"
echo "============================================================"
