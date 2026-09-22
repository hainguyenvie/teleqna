#!/usr/bin/env bash
# Big run: 8 cards, 1 epoch, checkpoints every 6000 steps (evaluated along the way by kit_big_evalloop.sh), then ep1 eval + vote.
# Pack = data/kit/big/pack_masked if data/kit/big/USE_MASKED exists else data/kit/big/pack; LR from data/kit/big/LR (default 3e-5).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:-0,1,2,3,4,5,6,7}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
cd "$ROOT"
until grep -q "BIG-POST DONE" logs/big_post.log 2>/dev/null; do grep -q ABORT logs/big_post.log 2>/dev/null && { echo "ABORT: big post failed"; exit 1; }; sleep 120; done
for i in $(seq 1 2880); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
PACK=data/kit/big/pack; [ -f data/kit/big/USE_MASKED ] && PACK=data/kit/big/pack_masked
LR=$(cat data/kit/big/LR 2>/dev/null || echo 3e-5)
echo "#### TRAIN-BIG START cards=$CARDS nproc=$N acc=$ACC pack=$PACK lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 259200 torchrun --nproc_per_node=$N --master_port=29650 code/kit/train_tier1.py --pack $PACK --out models/kit/big --epochs 1 --bs 2 --acc $ACC --lr $LR --save-every 6000 > logs/big_train.log 2>&1
echo "#### TRAIN-BIG DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/big_train.log || exit 1
EVC=$(echo "$CARDS" | cut -d, -f1)
CKPT=models/kit/big/ep1 TAG=big_ep1 CARDS=$EVC bash jobs/eval_ckpt.sh > logs/eval_big_ep1.log 2>&1; grep RESULT logs/eval_big_ep1.log
CUDA_VISIBLE_DEVICES=$EVC "$PY" -u code/kit/vote_eval.py --ckpt models/kit/big/ep1 --tag big_ep1 --gpu-mem 0.85 > logs/vote_big_ep1.log 2>&1; grep -E "^big|greedy" logs/vote_big_ep1.log
echo "#### BIG_CHAIN_DONE $(date -Iseconds)"
