#!/usr/bin/env bash
# After big run 4: uniform average of the late step checkpoints (>= step 2500) + ep1, eval on card 1 (vd9 labels use card 0).
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
until grep -q "TRAIN-BIG4 DONE rc=0" logs/big4r_chain.log 2>/dev/null; do sleep 300; done
until [ -f models/kit/big4/ep1/config.json ]; do sleep 60; done
CK=$(ls -d models/kit/big4/step* 2>/dev/null | awk -F'step' '$2>=2500' | sort -t p -k3 -n | tr "\n" "," | sed "s/,$//"); CK="$CK,models/kit/big4/ep1"
echo "#### BIG4 SOUP START ckpts=$CK $(date -Iseconds)"
$PY code/kit/soup_n.py --out models/kit/big4_soup --ckpts "$CK" > logs/big4_soup_merge.log 2>&1; tail -1 logs/big4_soup_merge.log
CKPT=models/kit/big4_soup TAG=big4_soup CARDS=${CARDS:-1} GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_big4_soup.log 2>&1; grep RESULT logs/eval_big4_soup.log
echo "#### BIG4 SOUP END $(date -Iseconds)"
