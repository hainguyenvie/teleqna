#!/usr/bin/env bash
# Merge sweep 3 around merge_f (0.2/0.3/0.5 vd6/vd8/ens = 81.79): g) 0.1/0.2/0.7, h) 0.2/0.2/0.6, i) 0.3/0.2/0.5.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
BASE=$HOME/projects/_shared/models/Qwen3-8B; echo "#### MERGE SWEEP 3 START $(date -Iseconds)"
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens/ep1 --out models/kit/merge_g --plain --weights 0.1,0.2,0.7 > logs/merge_g.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens/ep1 --out models/kit/merge_h --plain --weights 0.2,0.2,0.6 > logs/merge_h.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens/ep1 --out models/kit/merge_i --plain --weights 0.3,0.2,0.5 > logs/merge_i.log 2>&1
(CKPT=models/kit/merge_g TAG=merge_g CARDS=1 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_g.log 2>&1) &
(CKPT=models/kit/merge_h TAG=merge_h CARDS=2 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_h.log 2>&1) &
(CKPT=models/kit/merge_i TAG=merge_i CARDS=3 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_i.log 2>&1) &
wait; grep -h RESULT logs/eval_merge_g.log logs/eval_merge_h.log logs/eval_merge_i.log
echo "#### MERGE SWEEP 3 END $(date -Iseconds)"
