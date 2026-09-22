#!/usr/bin/env bash
# Arm: anchor-fix pack + masked-reconstruction rows (K=2 per chunk), 1 epoch lr 1e-5 — isolates the masked view against
# the 74.14 anchor-fix arm. Usage: CARDS=1,3,4,5,6 bash jobs/kit_tier1_masked.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARDS="${CARDS:-1,3,4,5,6}"; N=$(echo "$CARDS" | tr ',' '\n' | wc -l); ACC=$(( 32 / (N * 2) )); [ "$ACC" -ge 1 ] || ACC=1
cd "$ROOT"
until grep -q PACK_DONE logs/tier1masked_pack.log 2>/dev/null; do sleep 60; done
for i in $(seq 1 1440); do busy=0; for c in $(echo "$CARDS" | tr ',' ' '); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c); [ "$used" -le 1000 ] || busy=1; done; [ "$busy" = 0 ] && break; sleep 60; done
echo "#### TRAIN-MASKED START cards=$CARDS nproc=$N acc=$ACC $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=$CARDS timeout --foreground 86400 torchrun --nproc_per_node=$N --master_port=29643 code/kit/train_tier1.py --pack data/kit/tier1/pack_agree_masked --out models/kit/tier1_masked --epochs 1 --bs 2 --acc $ACC --lr 1e-5 > logs/tier1masked_train.log 2>&1
echo "#### TRAIN-MASKED DONE rc=$? $(date -Iseconds)"; grep -q TRAIN_DONE logs/tier1masked_train.log || exit 1
CKPT=models/kit/tier1_masked/ep1 TAG=kit1masked_ep1 CARDS=$(echo "$CARDS" | cut -d, -f1) bash jobs/eval_ckpt.sh > logs/eval_kit1masked.log 2>&1; grep RESULT logs/eval_kit1masked.log
echo "#### TIER1MASKED_CHAIN_DONE $(date -Iseconds)"
