#!/usr/bin/env bash
# Vote-distillation stage after the big run: self-consistent labels on 60k synthetic MCQs (card 0), pack, short
# full-weight SFT from the big checkpoint (8 cards, lr 5e-6, 1 epoch), eval + vote. Usage: INIT=models/kit/big/ep1 bash jobs/kit_vd.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"; mkdir -p data/kit/vd
until grep -q "BIG_CHAIN_DONE" logs/big_chain.log 2>/dev/null; do sleep 300; done
# INIT = the better of big/ep1 and the best step checkpoint (first-letter metric), unless given.
EP1=$(grep RESULT logs/eval_big_ep1.log 2>/dev/null | sed -E "s/.*first-letter ([0-9.]+) .*/\1/"); BEST=$(cat models/kit/big/BEST 2>/dev/null || echo "0 none")
if [ -z "${INIT:-}" ]; then if [ -n "$EP1" ] && [ "$(echo "$EP1 ${BEST%% *}" | awk "{print (\$1 >= \$2)}")" = 1 ]; then INIT=models/kit/big/ep1; else INIT=models/kit/big/${BEST#* }; fi; fi
LR="${LR:-5e-6}"; echo "ep1 first-letter=$EP1 best=$BEST -> INIT=$INIT"
for i in $(seq 1 240); do busy=0; for c in 0 1 2 3 4 5 6 7; do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
if [ -f data/kit/vd/pack_ids.npy ] && grep -q VD_DONE logs/vd_labels.log 2>/dev/null; then echo "labels+pack exist, skipping to train (init=$INIT)"; else
echo "#### VD LABELS START init=$INIT $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 timeout --foreground 14400 "$PY" -u code/kit/vote_distill.py --ckpt "$INIT" --n "${N:-60000}" > logs/vd_labels.log 2>&1; grep -E "gated|kept|Error" logs/vd_labels.log
grep -q VD_DONE logs/vd_labels.log || { echo "ABORT vd labels"; exit 1; }
"$PY" -u code/kit/pack_chat.py --rows data/kit/vd/labels.jsonl --out data/kit/vd/pack > logs/vd_pack.log 2>&1; grep -E "rows|Error" logs/vd_pack.log
grep -q PACK_DONE logs/vd_pack.log || { echo "ABORT vd pack"; exit 1; }
fi
echo "#### VD TRAIN START lr=$LR $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout --foreground 43200 torchrun --nproc_per_node=8 --master_port=29670 code/kit/train_tier1.py --pack data/kit/vd/pack --out models/kit/vd --init "$INIT" --epochs 1 --bs 2 --acc 2 --lr "$LR" --warm 50 > logs/vd_train.log 2>&1
echo "#### VD TRAIN DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/vd_train.log || exit 1
CKPT=models/kit/vd/ep1 TAG=vd_ep1 CARDS=0 bash jobs/eval_ckpt.sh > logs/eval_vd.log 2>&1; grep RESULT logs/eval_vd.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u code/kit/vote_eval.py --ckpt models/kit/vd/ep1 --tag vd_ep1 --gpu-mem 0.85 > logs/vote_vd.log 2>&1; grep -E "^vd|greedy" logs/vote_vd.log
echo "#### VD_CHAIN_DONE $(date -Iseconds)"
