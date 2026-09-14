#!/usr/bin/env bash
# GRPO on the full 988-prompt band, from base.
#
# Justified by the smoke rather than by hope. On the unified vLLM stack, against
# base thinking 75.20, the smoke's three checkpoints read:
#
#   step20  74.50  -0.70   21 fixed / 28 broken
#   step40  75.90  +0.70   30 fixed / 23 broken
#   step50  76.40  +1.20   29 fixed / 17 broken   p=0.104, StdSpec 61.00 -> 66.00
#
# Monotonic, still climbing when it ran out of data, and the least destructive
# arm in the table — every other treatment churns ~50 rows to net the same +1.
# The smoke saw 200 of the 988 band prompts for one epoch, so the obvious
# experiment is to give it the rest.
#
# Checkpoints every 20 steps because the smoke's own curve is the evidence: what
# matters is where on the curve the gain peaks, and coarse saves would have
# hidden the trend that justified this run.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-grpo/bin/python"
PYV="$BASE_ROOT/venvs/venv/bin/python"
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR="${MODEL_DIR:-$BASE}"
DATA="${DATA:-$ROOT/data/grpo_band.jsonl}"
RUN="${RUN:-grpo2full}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EXTRA_SITE="$BASE_ROOT/venvs/venv/lib/python3.11/site-packages"

# 85GB, not 100. The first attempt aborted on a card that was perfectly usable:
# job 78 was SIGTERMed (rc143) and its vLLM child outlived the runner, so card 4
# holds a 50.9GB context at 100% util that this service account cannot reap —
# there is no kubectl exec into the pod. What is left is 92GB, and the run needs
# 43GB for the colocated vLLM (0.3 x 143.77) plus ~40GB for the LoRA trainer on
# an 8B. Demanding a pristine card was the bug, not the card.
NEED_MIB="${NEED_MIB:-85000}"
for i in $(seq 1 120); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt "$NEED_MIB" ] && break
  [ "$i" = 1 ] && echo "card busy (${free_mib}MiB free, need ${NEED_MIB}) — waiting $(date -Iseconds)"
  sleep 30
done
echo "card free=${free_mib}MiB  model=$MODEL_DIR  data=$DATA  $(date -Iseconds)"
[ "$free_mib" -gt "$NEED_MIB" ] || { echo "ABORT: card busy"; exit 14; }
wc -l "$DATA"

echo "#### $RUN GRPO START $(date -Iseconds)"
timeout 64800 "$PY" "$ROOT/infra/train_grpo.py" \
  --dataset "$DATA" \
  --model-path "$MODEL_DIR" \
  --output-dir "$ROOT/models/teleqna-$RUN" \
  --epochs 1 --learning-rate 1e-5 --beta 0.01 \
  --num-generations 8 --generation-batch-size 8 \
  --per-device-batch 8 --grad-accum 4 \
  --max-completion-length 2560 --temperature 1.0 \
  --lora-r 16 --vllm-gpu-mem 0.3 --save-steps 20
echo "#### $RUN GRPO DONE $(date -Iseconds)"

# Score every checkpoint on the vLLM stack — one model load, LoRA swapped per
# arm. This is ~4 minutes an arm against 3h25m on the transformers path, which
# is the only reason sweeping a dozen checkpoints is affordable at all.
ADAPTER_ARGS=()
for ckpt in "$ROOT"/models/teleqna-$RUN/checkpoint-* "$ROOT"/models/teleqna-$RUN/final; do
  [ -d "$ckpt" ] || continue
  ADAPTER_ARGS+=(--adapter "${RUN}$(basename "$ckpt" | sed 's/checkpoint-/s/;s/final/final/')=$ckpt")
done
echo "#### $RUN dev-1000 sweep START $(date -Iseconds)  arms=${#ADAPTER_ARGS[@]}"
timeout 21600 "$PYV" "$ROOT/infra/eval_dev_vllm.py" \
  --base "$BASE" --data "$ROOT/data/dev1000.jsonl" \
  --out-dir "$ROOT/results/vllm" --thinking "${ADAPTER_ARGS[@]}"
echo "#### $RUN ALL DONE $(date -Iseconds)"
