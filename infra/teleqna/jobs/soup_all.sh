#!/usr/bin/env bash
# Uniform average of ALL vd-lineage checkpoints (13) — the 7-ckpt lineage soup gave 79.19 (best), does more averaging help?
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
echo "#### SOUP-ALL START $(date -Iseconds)"
$PY code/kit/soup_n.py --out models/kit/soup_all --ckpts models/kit/vd/ep1,models/kit/vd2/ep1,models/kit/cm/ep1,models/kit/vd3/ep1,models/kit/cm_mlp/ep1,models/kit/cm2/ep1,models/kit/vd4/ep1,models/kit/num/ep1,models/kit/beh/ep1,models/kit/dpo/ep1,models/kit/dpo3/ep1,models/kit/style/ep1,models/kit/vd5/ep1 > logs/soup_all_merge.log 2>&1; tail -1 logs/soup_all_merge.log
CKPT=models/kit/soup_all TAG=soup_all CARDS=${CARDS:-4} GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_soup_all.log 2>&1; grep RESULT logs/eval_soup_all.log
echo "#### SOUP-ALL END $(date -Iseconds)"
