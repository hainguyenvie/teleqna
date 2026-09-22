#!/usr/bin/env bash
# Cross-lineage merge of vd6 (big1 line) and vd8 (big3 line): union of their correct sets is 86.0 vs 79.7/80.6 alone.
# Plain task-arithmetic mean and TIES (k=0.2) — CPU merge, eval on a shared/idle card.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
BASE=$HOME/projects/_shared/models/Qwen3-8B; echo "#### TIES START $(date -Iseconds)"
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1 --out models/kit/merge_plain --plain > logs/merge_plain.log 2>&1; tail -1 logs/merge_plain.log
$PY code/kit/ties_merge.py --base $BASE --ckpts models/kit/vd6/ep1,models/kit/vd8/ep1 --out models/kit/merge_ties --k 0.2 > logs/merge_ties.log 2>&1; tail -1 logs/merge_ties.log
CKPT=models/kit/merge_plain TAG=merge_plain CARDS=${CARDS:-3} GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_plain.log 2>&1; grep RESULT logs/eval_merge_plain.log
CKPT=models/kit/merge_ties TAG=merge_ties CARDS=${CARDS:-3} GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_merge_ties.log 2>&1; grep RESULT logs/eval_merge_ties.log
echo "#### TIES END $(date -Iseconds)"
