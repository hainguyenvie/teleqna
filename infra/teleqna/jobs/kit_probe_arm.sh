#!/usr/bin/env bash
# After generation shard 1 (card 7) finishes: keep-only T1 eval (10k), then the probe watcher, all on card 7.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
until grep -q "TIER1 shard 1 DONE" logs/tier1_s1.log 2>/dev/null; do sleep 120; done
[ -f data/eval/keep_only.jsonl ] || "$PY" code/kit/select_windows_gold.py
if [ ! -f results/landscape/keep_only_q3_8b_base_nothink512.json ]; then
  echo "#### KEEP-ONLY T1 START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=7 timeout --foreground 7200 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data data/eval/keep_only.jsonl --out-dir results/landscape --tag keep_only_q3_8b --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 16384 > logs/eval_keep_only.log 2>&1
  echo "#### KEEP-ONLY T1 DONE rc=$? $(date -Iseconds)"
fi
CARDS=7 bash jobs/kit_probe_watch.sh
