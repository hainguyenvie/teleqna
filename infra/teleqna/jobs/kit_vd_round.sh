#!/usr/bin/env bash
# Vote-distillation round R (generic): ROUND=2 INIT=models/kit/vd/ep1 N=200000 SEED=2 bash jobs/kit_vd_round.sh
# Vote-distillation stage after the big run: self-consistent labels on 60k synthetic MCQs (card 0), pack, short
# full-weight SFT from the big checkpoint (8 cards, lr 5e-6, 1 epoch), eval + vote. Usage: INIT=models/kit/big/ep1 bash jobs/kit_vd.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROUND="${ROUND:?}"; VD=vd$ROUND; SEED="${SEED:-$ROUND}"
cd "$ROOT"; mkdir -p data/kit/$VD; CARDS="${CARDS:-0,1,2,3}"; NP=$(echo "$CARDS" | tr "," "\n" | wc -l); ACC=$(( 32 / (NP * 2) ))
# INIT = the better of big/ep1 and the best step checkpoint (first-letter metric), unless given.
EP1=$(grep RESULT logs/eval_big_ep1.log 2>/dev/null | sed -E "s/.*first-letter ([0-9.]+) .*/\1/"); BEST=$(cat models/kit/big/BEST 2>/dev/null || echo "0 none")
INIT="${INIT:?}"
LR="${LR:-5e-6}"; echo "ep1 first-letter=$EP1 best=$BEST -> INIT=$INIT"
for i in $(seq 1 480); do busy=0; for c in 0; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
if [ -f data/kit/$VD/pack_ids.npy ] && grep -q VD_DONE logs/${VD}_labels.log 2>/dev/null; then echo "labels+pack exist, skipping to train (init=$INIT)"; else
echo "#### VD LABELS START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 timeout --foreground 14400 "$PY" -u code/kit/vote_distill.py --ckpt "$INIT" --n "${N:-100000}" --seed "$SEED" --out data/kit/$VD/labels.jsonl > logs/${VD}_labels.log 2>&1; grep -E "gated|kept|Error" logs/${VD}_labels.log
grep -q VD_DONE logs/${VD}_labels.log || { echo "ABORT vd labels"; exit 1; }
"$PY" -u code/kit/pack_chat.py --rows data/kit/$VD/labels.jsonl --out data/kit/$VD/pack > logs/${VD}_pack.log 2>&1; grep -E "rows|Error" logs/${VD}_pack.log
grep -q PACK_DONE logs/${VD}_pack.log || { echo "ABORT vd pack"; exit 1; }
fi
for i in $(seq 1 480); do busy=0; for c in $(echo "$CARDS" | tr "," " "); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### VD TRAIN START lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$NP --master_port=29670 code/kit/train_tier1.py --pack data/kit/$VD/pack --out models/kit/$VD --init "$INIT" --epochs 1 --bs 2 --acc $ACC --lr "$LR" --warm 50 > logs/${VD}_train.log 2>&1
echo "#### VD TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/${VD}_train.log || exit 1
CKPT=models/kit/$VD/ep1 TAG=${VD}_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_$VD.log 2>&1; grep RESULT logs/eval_$VD.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u code/kit/vote_eval.py --ckpt models/kit/$VD/ep1 --tag ${VD}_ep1 --gpu-mem 0.85 > logs/vote_$VD.log 2>&1; grep -E "^vd|greedy" logs/vote_$VD.log
echo "#### VD_CHAIN_DONE round=$ROUND $(date -Iseconds)"
