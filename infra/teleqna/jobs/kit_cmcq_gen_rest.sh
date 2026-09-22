#!/usr/bin/env bash
# Contrastive-MCQ view for tiers 2-5 windows (~98k), on cards shared with the big run (gpu-mem 0.28, slow but free).
# Usage: CARDS=1 SHARD=0 NSHARDS=2 bash jobs/kit_cmcq_gen_rest.sh
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"
cd "$ROOT"
[ -f data/kit/windows_rest_all.jsonl ] || cat data/kit/windows_rest.jsonl data/kit/windows_tb_low.jsonl data/kit/tb_deep_windows.jsonl data/kit/tb_api_windows.jsonl > data/kit/windows_rest_all.jsonl
POOL="data/kit/windows_keep.jsonl,data/kit/windows_rest.jsonl,data/kit/windows_tb_low.jsonl,data/kit/tb_deep_windows.jsonl,data/kit/tb_api_windows.jsonl"
echo "#### CMCQ-REST shard ${SHARD:?}/${NSHARDS:?} START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 43200 "$PY" -u code/kit/gen_contrastive.py --windows-file data/kit/windows_rest_all.jsonl --pool "$POOL" --out data/kit/tier25c --shard "$SHARD" --nshards "$NSHARDS" --seed "$((800 + SHARD))" --gpu-mem 0.28 2>&1 | grep -vE "it/s\]|Processed prompts"
echo "#### CMCQ-REST shard $SHARD DONE rc=$? $(date -Iseconds)"
