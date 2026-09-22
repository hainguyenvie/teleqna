#!/usr/bin/env bash
# Last iteration of contrastive -> consistency: cm2 = cmcq pack, MLP-only, from vd3/ep1; then vd round 4 from cm2/ep1.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
for i in $(seq 1 480); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### cm2 TRAIN START $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 43200 torchrun --nproc_per_node=8 --master_port=29703 code/kit/train_tier1.py --pack data/kit/cm/pack --out models/kit/cm2 --init models/kit/vd3/ep1 --epochs 1 --bs 2 --acc 2 --lr 5e-6 --warm 50 --train-only mlp > logs/cm2_train.log 2>&1
echo "#### cm2 TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/cm2_train.log || exit 1
CKPT=models/kit/cm2/ep1 TAG=cm2_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_cm2.log 2>&1; grep RESULT logs/eval_cm2.log
ROUND=4 INIT=models/kit/cm2/ep1 N=100000 SEED=4 bash jobs/kit_vd_round.sh > logs/vd4_chain.log 2>&1; grep -E "RESULT|greedy|DONE" logs/vd4_chain.log
echo "#### CM2_VD4_DONE $(date -Iseconds)"
