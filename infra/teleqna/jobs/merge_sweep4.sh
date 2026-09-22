#!/usr/bin/env bash
# Merge sweep 4 after ens3 (big4 lineage): j) 0.2 vd6 / 0.3 vd8 / 0.5 ens3; k) 0.15/0.2/0.3/0.35 vd6/vd8/ens/ens3; l) 0.5 merge_f / 0.5 ens3.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
until grep -q "ENS3_CHAIN_DONE\|ABORT\|rc=[1-9]" logs/ens3_chain.log 2>/dev/null; do sleep 120; done
[ -f models/kit/ens3/ep1/config.json ] || { echo "ABORT no ens3"; exit 1; }
BASE=$HOME/projects/_shared/models/Qwen3-8B; echo "#### MERGE SWEEP 4 START $(date -Iseconds)"
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens3/ep1 --out models/kit/merge_j --plain --weights 0.2,0.3,0.5 > logs/merge_j.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1,models/kit/ens/ep1,models/kit/ens3/ep1 --out models/kit/merge_k --plain --weights 0.15,0.2,0.3,0.35 > logs/merge_k.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/merge_f,models/kit/ens3/ep1 --out models/kit/merge_l --plain --weights 0.5,0.5 > logs/merge_l.log 2>&1
(CKPT=models/kit/merge_j TAG=merge_j CARDS=1 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_j.log 2>&1) &
(CKPT=models/kit/merge_k TAG=merge_k CARDS=2 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_k.log 2>&1) &
(CKPT=models/kit/merge_l TAG=merge_l CARDS=3 GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_l.log 2>&1) &
wait; grep -h RESULT logs/eval_merge_j.log logs/eval_merge_k.log logs/eval_merge_l.log
echo "#### MERGE SWEEP 4 END $(date -Iseconds)"
