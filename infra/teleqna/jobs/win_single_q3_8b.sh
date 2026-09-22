#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-7}"
ROOT=$HOME/projects/teleqna/runs/teleqna-8b
PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
OUT=results/landscape/win_single_q3_8b_base_nothink512.json
[ -f "$OUT" ] && { echo "already done"; exit 0; }
[ -f data/eval/win_single.jsonl ] || "$PY" -u code/kit/build_win_single.py || exit 1
echo "#### win_single START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 14400 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" \
  --data data/eval/win_single.jsonl --out-dir results/landscape --tag win_single_q3_8b --with-base \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 8192
echo "#### win_single DONE rc=$? $(date -Iseconds)"
