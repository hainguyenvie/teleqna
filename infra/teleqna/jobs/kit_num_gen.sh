#!/usr/bin/env bash
# Numeric drill generation on 2 free cards (3,4), hard cap 3h. Usage: CARDS=3 SHARD=0 NSHARDS=2 bash jobs/kit_num_gen.sh
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"; cd "$ROOT"
echo "#### NUM shard ${SHARD:?}/${NSHARDS:?} START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 10800 "$PY" -u code/kit/gen_numeric.py --shard "$SHARD" --nshards "$NSHARDS" --seed "$((900 + SHARD))" 2>&1 | grep -vE "it/s\]|Processed prompts"
echo "#### NUM shard $SHARD DONE rc=$? $(date -Iseconds)"
