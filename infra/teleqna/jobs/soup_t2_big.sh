#!/usr/bin/env bash
# Model soup of two fine-tunes from the same base (tier-2 ep1 and big step24000): their correct sets differ (union 80.7 vs
# 76.0 / 77.8), so a 0.5 weight average may capture part of the union. Single 8B model. Eval on a card shared with the big run.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
echo "#### SOUP START $(date -Iseconds)"
$PY code/kit/wise_merge.py --base models/kit/tier2/ep1 --tuned models/kit/big/step24000 --alpha 0.5 --out models/kit/soup_t2_big50 > logs/soup_merge.log 2>&1; tail -1 logs/soup_merge.log
CARDS=1 GPUMEM=0.28 MEMWAIT=100000 CKPT=models/kit/soup_t2_big50 TAG=soup_t2_big50 bash jobs/eval_ckpt.sh > logs/eval_soup.log 2>&1; grep RESULT logs/eval_soup.log
echo "#### SOUP END $(date -Iseconds)"
