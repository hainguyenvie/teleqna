#!/usr/bin/env bash
# Pilot: 2,000 windows through the 122B teacher, to measure keep-rate before
# committing to the other 34,945.
#
# The windows are the ones pulled by the 1,324 rows that are BOTH wrong
# closed-book AND have a single window carrying the answer. Every other targeting
# was tried and refuted: distill_v1 generated from windows chosen by a retriever
# that hit 37.2%, and bought +1.1.
#
# What this measures is not accuracy. It is the fraction of windows that yield an
# item passing the evidence gate - verbatim span, answer inside it, three usable
# distractors. If that fraction is low the corpus subset is wrong and no amount
# of training fixes it; if it is high the full run is worth ~18x this cost.
#
# Cards 4+6, TP=2. Card 0 is left alone at the user's request and card 1 is
# holding the KaLM rerank. TP=2 puts ~118 GiB of weights on each card against
# 143 GiB of HBM, so max_num_seqs comes down to 64 and the context to 4096 -
# a window is 400 words, so 4096 is roomy, and this is throughput-bound anyway.
set -euo pipefail
HOME_ROOT=/home/tensara
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CARDS="${CARDS:-4,6}"
IFS=',' read -ra IDX <<< "$CARDS"
for c in "${IDX[@]}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c has only ${free}MiB free"; exit 14; }
done
export CUDA_VISIBLE_DEVICES="$CARDS"

mkdir -p "$ROOT/data/synth"
echo "#### gen-rest START $(date -Iseconds)"
"$PY" "$ROOT/infra/gen_targeted_qa.py" \
  --chunks "$ROOT/data/chunks_rest.jsonl" \
  --test "$ROOT/data/otfull10000.jsonl" \
  --out "$ROOT/data/synth/rest_items.jsonl" \
  --model "$MODEL" \
  --mode targeted \
  --max-targets 4 \
  --max-chunk-chars 6000 \
  --tp 2 --gpu-mem 0.94 --max-model-len 4096 --max-tokens 1200 \
  --engine-kwargs '{"language_model_only": true, "disable_custom_all_reduce": true, "max_num_seqs": 64}'
echo "#### gen-rest DONE rc=$? $(date -Iseconds)"
