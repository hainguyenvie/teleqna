#!/usr/bin/env bash
# One on-policy cycle from a student S: sample -> 31B-verified partially-known rows (weights ~ p) -> SFT (onpN) -> interpolate
# S <-> onpN at 0.7/0.3 and 0.5/0.5 -> eval. Usage: S=models/kit/wise_o3 N=2 CARDS=0,1,2,3 JCARDS=1,2,3 bash jobs/kit_onp_cycle.sh
set -uo pipefail
cd ~/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python; export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1
S="${S:?}"; N="${N:?}"; CARDS="${CARDS:-0,1,2,3}"; JCARDS="${JCARDS:-1,2,3}"; TAG=onp$N; RUN=data/kit/$TAG; C0=$(echo "$CARDS" | cut -d, -f1)
echo "#### CYCLE $N START student=$S $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$C0 "$PY" -u code/kit/utr_select.py --ckpt "$S" --n 300000 --seed $((10 + N)) --gpu-mem 0.85 --out $RUN --easy-keep 80000 > logs/${TAG}_select.log 2>&1
grep -q UTR_DONE logs/${TAG}_select.log || { echo "ABORT select"; exit 1; }; grep -E "^ALL|hard " logs/${TAG}_select.log | cut -c1-200
RUN=$RUN INIT="$S" TAG=$TAG JCARDS=$JCARDS CARDS=$CARDS DUP=4 bash jobs/kit_onp.sh
[ -f models/kit/$TAG/ep1/config.json ] || { echo "ABORT no $TAG"; exit 1; }
BASE=$HOME/projects/_shared/models/Qwen3-8B
$PY code/kit/ties_merge.py --base $BASE --ckpts "$S",models/kit/$TAG/ep1 --out models/kit/wise_${TAG}_3 --plain --weights 0.7,0.3 > logs/wise_${TAG}_3.log 2>&1
$PY code/kit/ties_merge.py --base $BASE --ckpts "$S",models/kit/$TAG/ep1 --out models/kit/wise_${TAG}_5 --plain --weights 0.5,0.5 > logs/wise_${TAG}_5.log 2>&1
(CKPT=models/kit/wise_${TAG}_3 TAG=wise_${TAG}_3 CARDS=$(echo "$CARDS" | cut -d, -f1) GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_wise_${TAG}_3.log 2>&1) &
(CKPT=models/kit/wise_${TAG}_5 TAG=wise_${TAG}_5 CARDS=$(echo "$CARDS" | cut -d, -f2) GPUMEM=0.85 MEMWAIT=100000 bash jobs/eval_ckpt.sh > logs/eval_wise_${TAG}_5.log 2>&1) &
wait; grep -h RESULT logs/eval_$TAG.log logs/eval_wise_${TAG}_3.log logs/eval_wise_${TAG}_5.log
echo "#### CYCLE $N DONE $(date -Iseconds)"
