#!/usr/bin/env bash
# After the LR sweep frees card 5: regenerate factviews at K=30 for the tier-1 windows (facts reused), 2 shards sequential on card 5.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/tier1_fv30
until grep -q "TIER1LR_CHAIN_DONE\|ABORT" logs/tier1lr_chain.log 2>/dev/null; do sleep 300; done
for i in $(seq 1 1440); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5); [ "$used" -le 1000 ] && break; sleep 60; done
for S in 0 1; do
  echo "#### FV30 shard $S START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=5 timeout --foreground 43200 "$PY" -u code/kit/gen_kit.py --only-factview data/kit/tier1/views_s$S.jsonl --nviews 30 --shard $S --nshards 2 --seed $((200+S)) --out data/kit/tier1_fv30 --gpu-mem 0.90 2>&1 | grep -vE "it/s\]|Processed prompts"
  echo "#### FV30 shard $S DONE $(date -Iseconds)"
done
echo "#### FV30_DONE"
