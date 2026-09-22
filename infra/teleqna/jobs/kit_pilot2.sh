#!/usr/bin/env bash
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:-6}"
cd "$ROOT"
"$PY" -u code/kit/build_kit_eval2.py || { echo "ABORT: build failed"; exit 1; }
for f in data/eval/kit2_*.jsonl; do
  ARM=$(basename "$f" .jsonl); OUT=results/landscape/${ARM}_q3_8b_base_nothink512.json
  [ -f "$OUT" ] && { echo "skip $ARM"; continue; }
  echo "#### $ARM START $(date -Iseconds)"
  timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data "$f" \
    --out-dir results/landscape --tag "${ARM}_q3_8b" --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 32768 > logs/${ARM}_eval.log 2>&1
  echo "#### $ARM DONE rc=$? $(date -Iseconds)"
done
echo "#### KIT2_ALL_DONE"
