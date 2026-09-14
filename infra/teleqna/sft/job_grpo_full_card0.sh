#!/usr/bin/env bash
# GRPO on the full 988-prompt band, from base — on node card 0 this time.
#
# Same experiment as job_grpo_full.sh; only the card moved, and for a measured
# reason. On card 4 the run OOMed at step 0:
#
#   total 139.80 GiB, of which 11.26 GiB free
#   Process 78136 has 49.74 GiB in use     <- not ours to kill
#   this process        78.79 GiB in use
#
# That 49.74 GiB is the vLLM child of job 78, which was SIGTERMed (rc143) and
# outlived its runner; this service account has no exec into the pod, so it
# cannot be reaped. Colocated GRPO wants the trainer and a vLLM engine in one
# process and does not fit in what is left. Card 0 is clean (134 MiB, 0%), so
# the run goes there.
#
# The tradeoff being accepted: card 0 is reached through the runtime door and
# has no scheduler protection, so a neighbour can take it. --save-steps 20 is
# what makes that affordable — the most a preemption can cost is 20 steps.
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
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
BASE_ROOT=/workspace/telelogs-base
ROOT=/workspace/teleqna-spare4
SFT="$BASE_ROOT/runs/teleqna-sft"
INFRA="$SFT/infra"
PY="$BASE_ROOT/venvs/venv-grpo/bin/python"
PYV="$BASE_ROOT/venvs/venv/bin/python"
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR="${MODEL_DIR:-$BASE}"
DATA="${DATA:-$SFT/data/grpo_band.jsonl}"
RUN="${RUN:-grpo2full}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export EXTRA_SITE="$BASE_ROOT/venvs/venv/lib/python3.11/site-packages"

NEED_MIB="${NEED_MIB:-120000}"
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
timeout 64800 "$PY" "$INFRA/train_grpo.py" \
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
  ADAPTER_ARGS+=(--adapter "${RUN}$(basename "$ckpt" | sed 's/checkpoint-/s/')=$ckpt")
done
echo "#### $RUN dev-1000 sweep START $(date -Iseconds)  arms=${#ADAPTER_ARGS[@]}"
timeout 21600 "$PYV" "$INFRA/eval_dev_vllm.py" \
  --base "$BASE" --data "$SFT/data/dev1000.jsonl" \
  --out-dir "$ROOT/results/vllm" --thinking "${ADAPTER_ARGS[@]}"
echo "#### $RUN ALL DONE $(date -Iseconds)"
