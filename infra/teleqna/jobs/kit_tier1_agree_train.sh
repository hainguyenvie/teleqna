#!/usr/bin/env bash
# Isolated test of the anchor fix: tier-1 pack with agree-only anchors, same lr 1e-5, 1 epoch, on cards 0,2 (after tier-2 shards).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:-0,2}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) ))
cd "$ROOT"
until grep -q "SELFREPLAY_DONE" logs/tier1_selfreplay_v2.log 2>/dev/null; do sleep 60; done
grep -c '"agree": true' data/kit/tier1/anchor_selfreplay.jsonl
echo "#### PACK-AGREE START $(date -Iseconds)"
"$PY" -u code/kit/pack_tier1.py --out data/kit/tier1/pack_agree > logs/tier1agree_pack.log 2>&1
grep -q PACK_DONE logs/tier1agree_pack.log || { echo "ABORT pack"; tail -3 logs/tier1agree_pack.log; exit 1; }; grep -E "anchors:|anchor |blocks of" logs/tier1agree_pack.log
for i in $(seq 1 1440); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
echo "#### TRAIN-AGREE START cards=$CARDS acc=$ACC $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 86400 torchrun --nproc_per_node=$N --master_port=29641 code/kit/train_tier1.py --pack data/kit/tier1/pack_agree --out models/kit/tier1_agree --epochs 1 --bs 2 --acc $ACC --lr 1e-5 > logs/tier1agree_train.log 2>&1
echo "#### TRAIN-AGREE DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/tier1agree_train.log || exit 1
CKPT=models/kit/tier1_agree/ep1 TAG=kit1agree_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_kit1agree.log 2>&1; grep RESULT logs/eval_kit1agree.log
echo "#### TIER1AGREE_CHAIN_DONE $(date -Iseconds)"
