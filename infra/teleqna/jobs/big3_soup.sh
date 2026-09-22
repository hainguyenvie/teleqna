#!/usr/bin/env bash
# After big run 3: SWA-style uniform average of the late step checkpoints (>= step 3000) + ep1, eval on a shared card.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
until grep -q "TRAIN-BIG3 DONE rc=0" logs/big3_chain.log 2>/dev/null; do sleep 300; done
until [ -f models/kit/big3/ep1/config.json ]; do sleep 60; done
CK=$(ls -d models/kit/big3/step* 2>/dev/null | awk -F'step' '$2>=3000' | sort -t p -k3 -n | tr "\n" "," | sed "s/,$//"); CK="$CK,models/kit/big3/ep1"
echo "#### BIG3 SOUP START ckpts=$CK $(date -Iseconds)"
$PY code/kit/soup_n.py --out models/kit/big3_soup --ckpts "$CK" > logs/big3_soup_merge.log 2>&1; tail -1 logs/big3_soup_merge.log
CKPT=models/kit/big3_soup TAG=big3_soup CARDS=${CARDS:-1} GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_big3_soup.log 2>&1; grep RESULT logs/eval_big3_soup.log
echo "#### BIG3 SOUP END $(date -Iseconds)"
