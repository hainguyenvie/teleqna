#!/usr/bin/env bash
# Judge all hard synthetic MCQs with OTel-31B over cards 0-6 (one 31B per card), then concatenate.
cd ~/projects/teleqna/runs/teleqna-8b; export PATH=$HOME/venv-vllm-nightly/bin:$PATH HF_HUB_OFFLINE=1
CARDS="${CARDS:-0,1,2,3,4,5,6}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); echo "#### HARD JUDGE START cards=$CARDS $(date -Iseconds)"
s=0; for c in $(echo "$CARDS" | tr "," " "); do CUDA_VISIBLE_DEVICES=$c nohup python -u code/kit/hard_judge.py --shard $s --nshards $NP > logs/hard_judge_s$s.log 2>&1 & s=$((s+1)); sleep 3; done; wait
grep -h "verified" logs/hard_judge_s*.log | cut -c1-200; cat data/kit/utr/rows_hard_verified_s*.jsonl > data/kit/utr/rows_hard_verified.jsonl
echo "#### HARD_JUDGE_DONE rows=$(wc -l < data/kit/utr/rows_hard_verified.jsonl) $(date -Iseconds)"
