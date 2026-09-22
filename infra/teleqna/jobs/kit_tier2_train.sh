#!/usr/bin/env bash
# Tier-2 train: waits for the pack, the LR sweep, and all listed cards; LR from data/kit/tier2/LR (default 2e-5); 1 epoch; eval + vote.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:?}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
cd "$ROOT"
until grep -q "PACK-MIXED DONE" logs/tier2_mixed_chain.log 2>/dev/null; do grep -q ABORT logs/tier2_mixed_chain.log 2>/dev/null && { echo "ABORT: mixed pack failed"; exit 1; }; sleep 300; done
until grep -q "TIER1LR_CHAIN_DONE\|ABORT" logs/tier1lr_chain.log 2>/dev/null; do sleep 300; done
for i in $(seq 1 2880); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
LR=$(cat data/kit/tier2/LR 2>/dev/null || echo 2e-5)
echo "#### TRAIN-T2 START cards=$CARDS nproc=$N acc=$ACC lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 172800 torchrun --nproc_per_node=$N --master_port=29640 code/kit/train_tier1.py --pack data/kit/tier2/pack_mixed --out models/kit/tier2 --epochs 1 --bs 2 --acc $ACC --lr $LR --save-every 3000 > logs/tier2_train.log 2>&1
echo "#### TRAIN-T2 DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/tier2_train.log || exit 1
EVC=$(echo "$CARDS" | cut -d, -f1)
CKPT=models/kit/tier2/ep1 TAG=kit2_ep1 CARDS=$EVC bash jobs/eval_ckpt.sh > logs/eval_kit2_ep1.log 2>&1; grep RESULT logs/eval_kit2_ep1.log
CUDA_VISIBLE_DEVICES=$EVC "$PY" -u code/kit/vote_eval.py --ckpt models/kit/tier2/ep1 --tag kit2_ep1 --gpu-mem 0.85 > logs/vote_kit2_ep1.log 2>&1; grep -E "^kit2|vote only" logs/vote_kit2_ep1.log
echo "#### TIER2_CHAIN_DONE $(date -Iseconds)"
