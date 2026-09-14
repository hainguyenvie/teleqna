#!/usr/bin/env bash
# Re-measure every arm that matters on the vLLM stack, in one process per mode.
#
# Switching stacks invalidates every cross-stack comparison, so the point of
# this job is to rebuild the whole comparison set on one stack in one sitting:
# the base references, the incumbent best (dpo3@200), the GRPO smoke ladder and
# the distillation ladder. After this, "+1.4 over base" means something again.
#
# One model load per mode, LoRA swapped per arm — that is the whole reason this
# is affordable. The transformers path needed 3h25m per arm; here every arm in a
# mode shares a single load.
#
# Runs on card 4, this pod's own device-plugin allocation, which is free again
# now that the slow sweep is stopped.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYV="$BASE_ROOT/venvs/venv/bin/python"     # vllm 0.11.0
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEV="$ROOT/data/dev1000.jsonl"
OUT="$ROOT/results/vllm"
# distill1 lives in the other pod's tree, which is mounted read-only here.
SPARE="$BASE_ROOT/runs/teleqna-spare4/models/teleqna-distill1-checkpoints"
GRPO="$ROOT/models/teleqna-grpo1smoke"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Size vLLM to the memory that is actually free, instead of demanding the whole
# card. Two evals were killed earlier and their CUDA contexts have not handed
# the VRAM back — card 4 sat at 50.9GB used, unchanging, with no way to reap it
# from here (this service account cannot exec into the pod). Waiting for a clean
# card would mean waiting indefinitely; 8B weights plus a working KV cache fit
# comfortably in what is left.
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')
for i in $(seq 1 60); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card has only ${free_mib}MiB free — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free_mib" -gt 45000 ] || { echo "ABORT: under 45GB free on the card"; exit 14; }
# gpu_memory_utilization is a fraction of TOTAL, so convert, and leave 8GB of
# headroom so a neighbour's allocation does not make vLLM fail at startup.
GPU_MEM=$(python3 -c "print(round(max(0.25, min(0.85, ($free_mib - 8000) / $TOTAL_MIB)), 2))")
echo "card free=${free_mib}MiB of ${TOTAL_MIB}MiB -> gpu_memory_utilization=$GPU_MEM  $(date -Iseconds)"

ARMS=(
  "dpo3s200=$ROOT/models/teleqna-dpo3-checkpoints/checkpoint-200"
  "grpo1s20=$GRPO/checkpoint-20"
  "grpo1s40=$GRPO/checkpoint-40"
  "grpo1s50=$GRPO/checkpoint-50"
  "distill1s100=$SPARE/checkpoint-100"
  "distill1s236=$SPARE/checkpoint-236"
)
ADAPTER_ARGS=()
for a in "${ARMS[@]}"; do
  p="${a#*=}"
  [ -d "$p" ] && ADAPTER_ARGS+=(--adapter "$a") || echo "skip missing $p"
done

echo "#### THINKING sweep START $(date -Iseconds)"
timeout 21600 "$PYV" "$ROOT/infra/eval_dev_vllm.py" \
  --base "$BASE" --data "$DEV" --out-dir "$OUT" \
  --with-base --thinking --gpu-mem "$GPU_MEM" "${ADAPTER_ARGS[@]}"
echo "#### THINKING sweep DONE $(date -Iseconds)"

echo "#### NO-THINK sweep START $(date -Iseconds)"
timeout 21600 "$PYV" "$ROOT/infra/eval_dev_vllm.py" \
  --base "$BASE" --data "$DEV" --out-dir "$OUT" \
  --with-base --gpu-mem "$GPU_MEM" "${ADAPTER_ARGS[@]}"
echo "#### NO-THINK sweep DONE $(date -Iseconds)"
