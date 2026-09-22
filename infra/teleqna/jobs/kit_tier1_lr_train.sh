#!/usr/bin/env bash
# LR sweep on the existing tier-1 pack: 1 epoch each at lr 2e-5 and 3e-5 (epoch 2 added nothing at 1e-5).
# Waits for the r2 arm. Usage: CARDS=1,3,4,5 nohup bash jobs/kit_tier1_lr_train.sh > logs/tier1lr_chain.log 2>&1 </dev/null &
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:?}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
cd "$ROOT"
until grep -q "TIER1GF_CHAIN_DONE" logs/tier1gf_chain.log 2>/dev/null; do grep -q ABORT logs/tier1gf_chain.log 2>/dev/null && { echo "ABORT: r2 chain failed"; exit 1; }; sleep 300; done
for LR in 2e-5 3e-5; do
  TAGLR=$(echo $LR | tr -d '-'); OUTD=models/kit/tier1_lr$TAGLR
  for i in $(seq 1 1440); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
  echo "#### TRAIN-LR $LR START cards=$CARDS $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 86400 torchrun --nproc_per_node=$N --master_port=29634 code/kit/train_tier1.py --pack data/kit/tier1/pack --out $OUTD --epochs 1 --bs 2 --acc $ACC --lr $LR > logs/tier1_lr${TAGLR}_train.log 2>&1
  echo "#### TRAIN-LR $LR DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/tier1_lr${TAGLR}_train.log || continue
  CKPT=$OUTD/ep1 TAG=kit1lr${TAGLR}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_kit1lr${TAGLR}.log 2>&1; grep RESULT logs/eval_kit1lr${TAGLR}.log
done
echo "#### TIER1LR_CHAIN_DONE $(date -Iseconds)"
