#!/usr/bin/env bash
# T1 context ceiling for Qwen3-8B: the same 8 strong-RAG windows that gave Qwen3.5-9B 85.17.
# Output: results/landscape/ragstrong_q3_8b_base_nothink512.json (eval_dev_vllm naming).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-7}"
ROOT=$HOME/projects/teleqna/runs/teleqna-8b
PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$HOME/projects/_shared/models/Qwen3-8B
TAG=ragstrong_q3_8b
OUT=$ROOT/results/landscape/${TAG}_base_nothink512.json
[ -f "$OUT" ] && { echo "already done: $OUT"; exit 0; }
for i in $(seq 1 180); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1)
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 60
done
[ "$free" -gt 100000 ] || { echo "ABORT: card busy"; exit 14; }
echo "#### $TAG START card=$CUDA_VISIBLE_DEVICES free=${free}MiB $(date -Iseconds)"
timeout --foreground 14400 "$PY" -u "$ROOT/code/eval_dev_vllm.py" \
  --base "$MODEL" --data "$ROOT/data/eval/otfull_rag8_strong.jsonl" \
  --out-dir "$ROOT/results/landscape" --tag "$TAG" --with-base \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 24576
rc=$?
echo "#### $TAG DONE rc=$rc $(date -Iseconds)"
exit $rc
