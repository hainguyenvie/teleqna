#!/usr/bin/env bash
# Serial GPU queue that survives card contention: each step waits for N free cards, runs, and the next step follows.
# Steps are defined inline below. Re-run this script after a container restart to resume (finished steps are skipped).
set -uo pipefail
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
free_cards() { for c in 0 1 2 3 4 5 6 7; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$u" -le 2000 ] && echo -n "$c "; done; }
wait_cards() { local n=$1 i; for i in $(seq 1 2880); do local f=($(free_cards)); [ "${#f[@]}" -ge "$n" ] && { echo "${f[@]:0:$n}" | tr " " ","; return 0; }; sleep 60; done; return 1; }
# step 1: deep-retrieval judging with the 31B (1 card)
if ! grep -q RETRIEVAL_CEILING_DONE logs/retrieval_ceiling_judge.log 2>/dev/null; then
  C=$(wait_cards 1); echo "#### QUEUE: retrieval-ceiling judge on card $C $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$C "$PY" -u code/analysis/retrieval_ceiling.py --tag wise_u3 --judge-only --gpu-mem 0.85 > logs/retrieval_ceiling_judge.log 2>&1
  grep -vE "^$|INFO|WARNING|it/s|EngineCore|Loading|Processed|Adding" logs/retrieval_ceiling_judge.log | tail -14 | cut -c1-220
fi
# step 2: on-policy cycle 2 from wise_o3 (sampling 1 card, judging 1 card, training 2 cards)
if ! grep -q "CYCLE 2 DONE" logs/cycle2.log 2>/dev/null; then
  C=$(wait_cards 2); echo "#### QUEUE: cycle 2 on cards $C $(date -Iseconds)"
  S=models/kit/wise_o3 N=2 CARDS=$C JCARDS=$(echo "$C" | cut -d, -f1) bash jobs/kit_onp_cycle.sh > logs/cycle2.log 2>&1
  grep -E "RESULT|ABORT" logs/cycle2.log | cut -c1-220
fi
echo "#### QUEUE_DONE $(date -Iseconds)"
