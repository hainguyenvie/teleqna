#!/usr/bin/env bash
# Full-weight CPT on the tier-1 pack, then score ep1/ep2 on ot-full (10k) and the rotated-choices control.
# Waits (up to 36 h) for the post-chain to finish and for every listed card to be at 0 MiB, so the cards
# can be freed at any time. Usage: CARDS=0,1,2,3,4,5,7 nohup bash jobs/kit_tier1_train.sh > logs/tier1_chain.log 2>&1 </dev/null &
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:?}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
cd "$ROOT"
until grep -q "POST DONE" logs/tier1_post.log 2>/dev/null; do grep -q ABORT logs/tier1_post.log 2>/dev/null && { echo "ABORT: post-chain failed"; exit 1; }; sleep 300; done
echo "#### pack ready $(date -Iseconds); waiting for cards $CARDS to be free"
for i in $(seq 1 2160); do
  busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done
  [ "$busy" = 0 ] && break; sleep 60
done
[ "$busy" = 0 ] || { echo "ABORT: cards never freed"; exit 14; }
echo "#### TRAIN START cards=$CARDS nproc=$N acc=$ACC $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 172800 torchrun --nproc_per_node=$N --master_port=29631 code/kit/train_tier1.py \
  --pack data/kit/tier1/pack --out models/kit/tier1 --epochs 2 --bs 2 --acc $ACC --save-every 2000 > logs/tier1_train.log 2>&1
rc=$?; echo "#### TRAIN DONE rc=$rc $(date -Iseconds)"; grep -q TRAIN_DONE logs/tier1_train.log || { tail -5 logs/tier1_train.log; exit 1; }
EVC=$(echo "$CARDS" | cut -d, -f1)
for ep in ep1 ep2; do
  for set in otfull10000 otfull_p1; do
    TAG=kit1_${ep}_${set}; OUT=results/landscape/${TAG}_base_nothink512.json; [ -f "$OUT" ] && continue
    echo "#### EVAL $TAG START $(date -Iseconds)"
    CUDA_VISIBLE_DEVICES=$EVC timeout --foreground 7200 "$PY" -u code/eval_dev_vllm.py --base "models/kit/tier1/$ep" --data "data/eval/$set.jsonl" \
      --out-dir results/landscape --tag "$TAG" --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 4096 > logs/eval_$TAG.log 2>&1
    echo "#### EVAL $TAG DONE rc=$? $(date -Iseconds)"
  done
done
echo "#### TIER1_CHAIN_DONE $(date -Iseconds)"
