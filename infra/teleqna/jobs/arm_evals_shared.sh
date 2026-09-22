#!/usr/bin/env bash
# Evaluate arms A and B on cards shared with the big run (gpu-mem 0.28 fits beside the 96 GB training process).
cd ~/projects/teleqna/runs/teleqna-8b
CKPT=models/kit/tier1_armA/ep1 TAG=kit1armA_ep1 CARDS=1 GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_kit1armA.log 2>&1 &
CKPT=models/kit/tier1_armB/ep1 TAG=kit1armB_ep1 CARDS=2 GPUMEM=0.28 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_kit1armB.log 2>&1 &
wait
grep -h RESULT logs/eval_kit1armA.log logs/eval_kit1armB.log
echo "#### ARM EVALS DONE $(date -Iseconds)"
