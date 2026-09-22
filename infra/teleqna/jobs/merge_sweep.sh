#!/usr/bin/env bash
# Merge sweep around the winning plain merge (vd6+vd8 mean = 81.36): a) 0.4 vd6 / 0.6 vd8, b) 0.6 / 0.4, c) mean scaled 1.25.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
BASE=$HOME/projects/_shared/models/Qwen3-8B; echo "#### MERGE SWEEP START $(date -Iseconds)"
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1 --out models/kit/merge_a --plain --weights 0.4,0.6 > logs/merge_a.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1 --out models/kit/merge_b --plain --weights 0.6,0.4 > logs/merge_b.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1 --out models/kit/merge_c --plain --lam 1.25 > logs/merge_c.log 2>&1
(CKPT=models/kit/merge_a TAG=merge_a CARDS=1 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_a.log 2>&1) &
(CKPT=models/kit/merge_b TAG=merge_b CARDS=2 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_b.log 2>&1) &
(CKPT=models/kit/merge_c TAG=merge_c CARDS=3 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_c.log 2>&1) &
wait; grep -h RESULT logs/eval_merge_a.log logs/eval_merge_b.log logs/eval_merge_c.log
echo "#### MERGE SWEEP END $(date -Iseconds)"
