#!/usr/bin/env bash
# Behaviour stage: pack behaviour MCQ rows (All-of-the-above / distractor / NOT), short SFT from the best checkpoint so far, eval.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; LR="${LR:-5e-6}"
until grep -q "NUM_CHAIN_DONE\|ABORT" logs/num_chain.log 2>/dev/null; do sleep 120; done
until grep -q "BEH shard 0 DONE" logs/beh_s0.log && grep -q "BEH shard 1 DONE" logs/beh_s1.log; do sleep 60; done
read -r INIT SCORE < <(bash jobs/best_ckpt.sh); echo "init=$INIT ($SCORE)"
cat data/kit/beh/rows_s0.jsonl data/kit/beh/rows_s1.jsonl > data/kit/beh/rows.jsonl
"$PY" -u code/kit/pack_chat.py --rows data/kit/beh/rows.jsonl --out data/kit/beh/pack > logs/beh_pack.log 2>&1; tail -2 logs/beh_pack.log
grep -q PACK_DONE logs/beh_pack.log || { echo "ABORT beh pack"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### BEH TRAIN START init=$INIT lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 43200 torchrun --nproc_per_node=8 --master_port=29697 code/kit/train_tier1.py --pack data/kit/beh/pack --out models/kit/beh --init "$INIT" --epochs 1 --bs 2 --acc 2 --lr "$LR" --warm 50 > logs/beh_train.log 2>&1
echo "#### BEH TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/beh_train.log || exit 1
CKPT=models/kit/beh/ep1 TAG=beh_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_beh.log 2>&1; grep RESULT logs/eval_beh.log
echo "#### BEH_CHAIN_DONE $(date -Iseconds)"
