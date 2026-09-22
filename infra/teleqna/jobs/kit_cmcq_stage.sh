#!/usr/bin/env bash
# Contrastive-MCQ stage from vd/ep1: generate cmcq for tiers 2-5 (2 cards, full speed), build chat rows from all
# evidence-gated cmcq (+ kit mcq), pack with <|im_end|>, short SFT (lr 5e-6), eval + vote.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/cm
INIT="${INIT:-models/kit/vd/ep1}"; LR="${LR:-5e-6}"
echo "#### CMCQ-REST GEN START $(date -Iseconds)"
sed -i 's/--gpu-mem 0.28/--gpu-mem 0.90/' jobs/kit_cmcq_gen_rest.sh
CARDS=1 SHARD=0 NSHARDS=2 bash jobs/kit_cmcq_gen_rest.sh > logs/cmcq_rest_s0.log 2>&1 &
CARDS=2 SHARD=1 NSHARDS=2 bash jobs/kit_cmcq_gen_rest.sh > logs/cmcq_rest_s1.log 2>&1 &
wait; grep -h "cmcq gate\|DONE" logs/cmcq_rest_s0.log logs/cmcq_rest_s1.log | cut -c1-160
"$PY" -u code/kit/mcq_rows.py --globs "data/kit/tier1c/views_s*.jsonl,data/kit/tier25c/views_s*.jsonl" --out data/kit/cm/rows.jsonl --rot 2 > logs/cm_rows.log 2>&1; tail -2 logs/cm_rows.log
"$PY" -u code/kit/pack_chat.py --rows data/kit/cm/rows.jsonl --out data/kit/cm/pack > logs/cm_pack.log 2>&1; tail -2 logs/cm_pack.log
grep -q PACK_DONE logs/cm_pack.log || { echo "ABORT cm pack"; exit 1; }
for i in $(seq 1 480); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### CM TRAIN START init=$INIT lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 43200 torchrun --nproc_per_node=8 --master_port=29680 code/kit/train_tier1.py --pack data/kit/cm/pack --out models/kit/cm --init "$INIT" --epochs 1 --bs 2 --acc 2 --lr "$LR" --warm 50 > logs/cm_train.log 2>&1
echo "#### CM TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/cm_train.log || exit 1
CKPT=models/kit/cm/ep1 TAG=cm_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_cm.log 2>&1; grep RESULT logs/eval_cm.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u code/kit/vote_eval.py --ckpt models/kit/cm/ep1 --tag cm_ep1 --gpu-mem 0.85 > logs/vote_cm.log 2>&1; grep -E "^cm|greedy" logs/vote_cm.log
echo "#### CM_CHAIN_DONE $(date -Iseconds)"
