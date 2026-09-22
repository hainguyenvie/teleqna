#!/usr/bin/env bash
# Build T1/T2 ceiling sets for run-3 TARGET (CPU), then score both with Qwen3-8B on one card.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-6}"
ROOT=$HOME/projects/teleqna/runs/teleqna-8b
PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$HOME/projects/_shared/models/Qwen3-8B
cd "$ROOT"
[ -f data/eval/ceil_t2_kit.jsonl ] || "$PY" -u code/eg3/ceiling_sets.py || exit 1
for i in $(seq 1 180); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
  [ "$free" -gt 100000 ] && break; sleep 60
done
[ "$free" -gt 100000 ] || { echo "ABORT: card busy"; exit 14; }
for ARM in ceil_t1_raw ceil_t2_kit; do
  OUT=results/landscape/${ARM}_q3_8b_base_nothink512.json
  [ -f "$OUT" ] && { echo "skip $ARM"; continue; }
  echo "#### $ARM START $(date -Iseconds)"
  timeout --foreground 7200 "$PY" -u code/eval_dev_vllm.py \
    --base "$MODEL" --data "data/eval/${ARM}.jsonl" \
    --out-dir results/landscape --tag "${ARM}_q3_8b" --with-base \
    --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 16384
  echo "#### $ARM DONE rc=$? $(date -Iseconds)"
done
