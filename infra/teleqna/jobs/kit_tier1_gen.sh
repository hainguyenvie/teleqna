#!/usr/bin/env bash
# Tier-1 generation, one shard per card. Usage: CARDS=5 SHARD=0 NSHARDS=2 bash jobs/kit_tier1_gen.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"
cd "$ROOT"
for i in $(seq 1 180); do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1); [ "$free" -gt 100000 ] && break; sleep 60; done
[ "$free" -gt 100000 ] || { echo "ABORT: card $CUDA_VISIBLE_DEVICES busy"; exit 14; }
echo "#### TIER1 shard ${SHARD:?}/${NSHARDS:?} START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 43200 "$PY" -u code/kit/gen_kit.py --shard "$SHARD" --nshards "$NSHARDS" --seed "$SHARD" ${LIMIT:+--limit $LIMIT}
echo "#### TIER1 shard $SHARD DONE rc=$? $(date -Iseconds)"
