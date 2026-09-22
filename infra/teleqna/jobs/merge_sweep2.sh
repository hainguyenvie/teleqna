#!/usr/bin/env bash
# Merge sweep 2 with the ens checkpoint (ens_ep1 = 81.48 / rot1 80.58, vd8 lineage): d) 0.4 vd6 / 0.6 ens; e) 3-way 0.3 vd6 / 0.3 vd8 / 0.4 ens;
# f) 0.5 merge_a / 0.5 ens (== 0.2 vd6 / 0.3 vd8 / 0.5 ens). Evals on cards 1-3 while ens2 labels run on card 0.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
BASE=$HOME/projects/_shared/models/Qwen3-8B; echo "#### MERGE SWEEP 2 START $(date -Iseconds)"
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/ens/ep1 --out models/kit/merge_d --plain --weights 0.4,0.6 > logs/merge_d.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens/ep1 --out models/kit/merge_e --plain --weights 0.3,0.3,0.4 > logs/merge_e.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens/ep1 --out models/kit/merge_f --plain --weights 0.2,0.3,0.5 > logs/merge_f.log 2>&1
(CKPT=models/kit/merge_d TAG=merge_d CARDS=1 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_d.log 2>&1) &
(CKPT=models/kit/merge_e TAG=merge_e CARDS=2 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_e.log 2>&1) &
(CKPT=models/kit/merge_f TAG=merge_f CARDS=3 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_f.log 2>&1) &
wait; grep -h RESULT logs/eval_merge_d.log logs/eval_merge_e.log logs/eval_merge_f.log
echo "#### MERGE SWEEP 2 END $(date -Iseconds)"
