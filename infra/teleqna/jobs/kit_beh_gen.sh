#!/usr/bin/env bash
# Behaviour-view generation (All-of-the-above / All-as-distractor / NOT items) on cards 3,4; cap 2h.
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"; cd "$ROOT"
echo "#### BEH shard ${SHARD:?}/${NSHARDS:?} START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 7200 "$PY" -u code/kit/gen_behavior.py --shard "$SHARD" --nshards "$NSHARDS" --seed "$((950 + SHARD))" --sample 24000 2>&1 | grep -vE "it/s\]|Processed prompts"
echo "#### BEH shard $SHARD DONE rc=$? $(date -Iseconds)"
