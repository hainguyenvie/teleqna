#!/usr/bin/env bash
# Arm: the contrastive stage (same pack data/kit/cm/pack, from vd/ep1, lr 5e-6) updating ONLY the MLP weights — does
# freezing attention keep the recover (1181) while cutting the broke (491)? Then a consistency pass on top if it does.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; SUB="${SUB:-mlp}"; LR="${LR:-5e-6}"; TAG=cm_$SUB
for i in $(seq 1 480); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### $TAG TRAIN START lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 43200 torchrun --nproc_per_node=8 --master_port=29701 code/kit/train_tier1.py --pack data/kit/cm/pack --out models/kit/$TAG --init models/kit/vd/ep1 --epochs 1 --bs 2 --acc 2 --lr "$LR" --warm 50 --train-only "$SUB" > logs/${TAG}_train.log 2>&1
echo "#### $TAG TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${TAG}_train.log || { tail -5 logs/${TAG}_train.log; exit 1; }
CKPT=models/kit/$TAG/ep1 TAG=${TAG}_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_$TAG.log 2>&1; grep RESULT logs/eval_$TAG.log
echo "#### ${TAG}_CHAIN_DONE $(date -Iseconds)"
