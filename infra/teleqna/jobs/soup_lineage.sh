#!/usr/bin/env bash
# SWA-style average of the vd3-lineage checkpoints (all within a small distance of vd3): does averaging denoise the
# fragile "flicker" rows (853 vd3 errors that some sibling checkpoint gets right)? CPU merge, eval on a shared card.
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
echo "#### SOUP-LINEAGE START $(date -Iseconds)"
$PY code/kit/soup_n.py --out models/kit/soup_lineage --ckpts models/kit/vd3/ep1,models/kit/vd4/ep1,models/kit/vd5/ep1,models/kit/dpo3/ep1,models/kit/cm_mlp/ep1,models/kit/num/ep1,models/kit/beh/ep1 > logs/soup_lineage_merge.log 2>&1; tail -1 logs/soup_lineage_merge.log
CKPT=models/kit/soup_lineage TAG=soup_lineage CARDS=${CARDS:-2} GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_soup_lineage.log 2>&1; grep RESULT logs/eval_soup_lineage.log
echo "#### SOUP-LINEAGE END $(date -Iseconds)"
