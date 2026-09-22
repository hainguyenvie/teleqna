#!/usr/bin/env bash
# HET generation: 4 styles x k=20 over the question-relevant windows (windows_keep + windows_rest), sharded over the given
# cards (each shard = one 8B vLLM on one card). Waits for the big4 chain to end (cards free), leaves card 0 to vd9/ens3 labels.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"; mkdir -p data/kit/het; CARDS="${CARDS:-1,2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l)
[ -f data/kit/het/windows.jsonl ] || cat data/kit/windows_keep.jsonl data/kit/windows_rest.jsonl > data/kit/het/windows.jsonl
until grep -q "BIG4_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/big4r_chain.log 2>/dev/null; do sleep 120; done
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### HET GEN START cards=$CARDS windows=$(wc -l < data/kit/het/windows.jsonl) $(date -Iseconds)"
for style in direct cloze reverse numeric; do
  s=0; for c in $(echo "$CARDS" | tr "," " "); do
    CUDA_VISIBLE_DEVICES=$c timeout --foreground 7200 "$PY" -u code/kit/gen_het.py --style $style --shard $s --nshards $NP --k 20 --seed 3 > logs/het_gen_${style}_s$s.log 2>&1 &
    s=$((s + 1)); sleep 3
  done; wait
  grep -h "train rows" logs/het_gen_${style}_s*.log | cut -c1-200; echo "#### HET $style done $(date -Iseconds)"
done
cat data/kit/het/rows_*_s*.jsonl > data/kit/het/rows_all.jsonl; cat data/kit/het/holdout_*_s*.jsonl > data/kit/het/holdout_all.jsonl
echo "#### HET_GEN_DONE rows=$(wc -l < data/kit/het/rows_all.jsonl) holdout=$(wc -l < data/kit/het/holdout_all.jsonl) $(date -Iseconds)"
