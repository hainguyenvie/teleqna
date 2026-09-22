#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-6}"
ROOT=$HOME/projects/teleqna/runs/teleqna-8b
PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$HOME/projects/_shared/models/Qwen3-8B
cd "$ROOT"
[ -f data/eg2/windows.jsonl ] || { echo "ABORT: data/eg2/windows.jsonl missing"; exit 1; }
[ -f data/eval/ceil_t2b_kit12k.jsonl ] || "$PY" -u code/eg3/ceiling_sets2.py || { echo "ABORT: build failed"; exit 1; }
for ARM in ceil_t1b_full ceil_t1c_wins ceil_t2b_kit12k; do
  OUT=results/landscape/${ARM}_q3_8b_base_nothink512.json
  [ -f "$OUT" ] && { echo "skip $ARM"; continue; }
  echo "#### $ARM START $(date -Iseconds)"
  timeout --foreground 7200 "$PY" -u code/eval_dev_vllm.py --base "$MODEL" --data "data/eval/${ARM}.jsonl" \
    --out-dir results/landscape --tag "${ARM}_q3_8b" --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 16384
  echo "#### $ARM DONE rc=$? $(date -Iseconds)"
done
