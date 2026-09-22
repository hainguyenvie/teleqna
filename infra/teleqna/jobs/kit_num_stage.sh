#!/usr/bin/env bash
# Numeric-drill stage: pack the 84k numeric Q/A rows (chat, <|im_end|>), short SFT from vd/ep1 (lr 5e-6), eval + vote.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; INIT="${INIT:-models/kit/vd/ep1}"; LR="${LR:-5e-6}"
until grep -q "CM_CHAIN_DONE\|ABORT" logs/cm_chain.log 2>/dev/null; do sleep 120; done
cat data/kit/num/rows_s0.jsonl data/kit/num/rows_s1.jsonl > data/kit/num/rows.jsonl
"$PY" -u code/kit/pack_chat.py --rows data/kit/num/rows.jsonl --out data/kit/num/pack > logs/num_pack.log 2>&1; tail -2 logs/num_pack.log
grep -q PACK_DONE logs/num_pack.log || { echo "ABORT num pack"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### NUM TRAIN START init=$INIT lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 43200 torchrun --nproc_per_node=8 --master_port=29695 code/kit/train_tier1.py --pack data/kit/num/pack --out models/kit/num --init "$INIT" --epochs 1 --bs 2 --acc 2 --lr "$LR" --warm 50 > logs/num_train.log 2>&1
echo "#### NUM TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/num_train.log || exit 1
CKPT=models/kit/num/ep1 TAG=num_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_num.log 2>&1; grep RESULT logs/eval_num.log
echo "#### NUM_CHAIN_DONE $(date -Iseconds)"
