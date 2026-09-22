#!/usr/bin/env bash
# Recall-QA generation for the remaining ~98k windows (tiers 2-5) on idle cards, 6 shards. Cap 3h.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/recall_all; CARDS="${CARDS:-2,3,4,5,6,7}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l)
echo "#### RECALL-ALL GEN START cards=$CARDS $(date -Iseconds)"
i=0; for c in $(echo "$CARDS" | tr "," " "); do CUDA_VISIBLE_DEVICES=$c timeout --foreground 10800 "$PY" -u code/kit/gen_recall_qa.py --windows-file data/kit/windows_rest_all.jsonl --out data/kit/recall_all --shard $i --nshards $NP --seed $((1400+i)) --k 20 > logs/recall_all_s$i.log 2>&1 & i=$((i+1)); done; wait
grep -h "train rows\|Traceback" logs/recall_all_s*.log | cut -c1-200
echo "#### RECALL-ALL GEN DONE $(date -Iseconds)"
