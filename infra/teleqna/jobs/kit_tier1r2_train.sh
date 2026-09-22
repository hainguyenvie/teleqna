#!/usr/bin/env bash
# Arm R2: same generated views, kit restricted to the label-free majority-agreement windows.
# Runs after the label-free tier-1 chain on the same cards. Usage: CARDS=0,1,2,3,4,5 nohup bash jobs/kit_tier1r2_train.sh > logs/tier1gf_chain.log 2>&1 </dev/null &
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:?}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
cd "$ROOT"
until grep -q "TIER1GF_CHAIN_DONE" logs/tier1gf_chain.log 2>/dev/null; do grep -q ABORT logs/tier1gf_chain.log 2>/dev/null && { echo "ABORT: main chain failed"; exit 1; }; sleep 300; done
[ -f data/kit/windows_r2.json ] || "$PY" code/kit/select_windows_r2.py
echo "#### PACK-R2 START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py --windows data/kit/windows_r2.json --out data/kit/tier1/pack_r2 > logs/tier1r2_pack.log 2>&1
grep -q PACK_DONE logs/tier1r2_pack.log || { echo "ABORT pack-gf"; exit 1; }; grep -E "kit docs|replay|anchor|blocks" logs/tier1gf_pack.log
for i in $(seq 1 1440); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
echo "#### TRAIN-R2 START cards=$CARDS $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 172800 torchrun --nproc_per_node=$N --master_port=29633 code/kit/train_tier1.py --pack data/kit/tier1/pack_r2 --out models/kit/tier1r2 --epochs 2 --bs 2 --acc $ACC > logs/tier1r2_train.log 2>&1
echo "#### TRAIN-R2 DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/tier1r2_train.log || exit 1
EVC=$(echo "$CARDS" | cut -d, -f1)
for ep in ep1 ep2; do for set in otfull10000 otfull_p1; do
  TAG=kit1r2_${ep}_${set}; [ -f results/landscape/${TAG}_base_nothink512.json ] && continue
  CUDA_VISIBLE_DEVICES=$EVC timeout --foreground 7200 "$PY" -u code/eval_dev_vllm.py --base "models/kit/tier1r2/$ep" --data "data/eval/$set.jsonl" --out-dir results/landscape --tag "$TAG" --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 4096 > logs/eval_$TAG.log 2>&1
  echo "#### EVAL $TAG DONE rc=$? $(date -Iseconds)"
done; done
echo "#### TIER1R2_CHAIN_DONE $(date -Iseconds)"
