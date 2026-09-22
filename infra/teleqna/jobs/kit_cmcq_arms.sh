#!/usr/bin/env bash
# Two isolated arms on the tier-1 (anchor-fix) pack, 1 epoch lr 1e-5, against anchor-fix 74.14:
#   A  pack_agree + mcq-gold rows from the tier-1 mcq view                      (cards 0,1,2)
#   B  pack_agree + mcq-gold rows from tier-1 mcq + contrastive MCQ (tier1c)      (cards 3,4,5)
# Runs while the big pack is being built; the big run waits for all cards, so this delays it by the arm length.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
ARM="${ARM:?}"; CARDS="${CARDS:?}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
if [ "$ARM" = A ]; then VIEWS="data/kit/tier1/views_s*.jsonl"; else VIEWS="data/kit/tier1/views_s*.jsonl,data/kit/tier1c/views_s*.jsonl"; fi
echo "#### PACK-$ARM START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py --views "$VIEWS" --anchor data/kit/tier1/anchor_selfreplay.jsonl --mcq-gold --out data/kit/tier1/pack_arm$ARM > logs/arm${ARM}_pack.log 2>&1
grep -q PACK_DONE logs/arm${ARM}_pack.log || { echo "ABORT pack-$ARM"; tail -3 logs/arm${ARM}_pack.log; exit 1; }; grep -E "chat-format|anchors:|^anchor |blocks of" logs/arm${ARM}_pack.log
for i in $(seq 1 240); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 30; done
echo "#### TRAIN-$ARM START cards=$CARDS nproc=$N acc=$ACC $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 43200 torchrun --nproc_per_node=$N --master_port=$((29660 + $(printf '%d' "'$ARM"))) code/kit/train_tier1.py --pack data/kit/tier1/pack_arm$ARM --out models/kit/tier1_arm$ARM --epochs 1 --bs 2 --acc $ACC --lr 1e-5 > logs/arm${ARM}_train.log 2>&1
echo "#### TRAIN-$ARM DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/arm${ARM}_train.log || exit 1
CKPT=models/kit/tier1_arm$ARM/ep1 TAG=kit1arm${ARM}_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_kit1arm$ARM.log 2>&1; grep RESULT logs/eval_kit1arm$ARM.log
echo "#### ARM-$ARM CHAIN DONE $(date -Iseconds)"
