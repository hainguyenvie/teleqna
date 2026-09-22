#!/usr/bin/env bash
# Contrastive-MCQ view for the tier-1 keep windows, 2 shards on idle cards while big-post runs. Hard 3h cap so it
# never delays the big run (which waits for all cards to be free). Usage: CARDS=0 SHARD=0 NSHARDS=2 bash jobs/kit_cmcq_gen.sh
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:?}"
cd "$ROOT"
echo "#### CMCQ shard ${SHARD:?}/${NSHARDS:?} START card=$CUDA_VISIBLE_DEVICES $(date -Iseconds)"
timeout --foreground 10800 "$PY" -u code/kit/gen_contrastive.py --shard "$SHARD" --nshards "$NSHARDS" --seed "$((700 + SHARD))" --gpu-mem 0.90 2>&1 | grep -vE "it/s\]|Processed prompts"
echo "#### CMCQ shard $SHARD DONE rc=$? $(date -Iseconds)"
