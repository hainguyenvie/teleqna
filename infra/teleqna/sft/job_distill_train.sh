#!/usr/bin/env bash
# Track B: build the closed-book SFT file from the gated grounded items, train a
# deliberately small LoRA on it, and sweep dev-1000 at every checkpoint in both
# serving modes.
#
# The dose is set by what already failed. train_v3 at LORA_R=64 / LR=1e-4 /
# 2 epochs lost 5.4 points, and the gentler r16 / 2e-5 / 1 epoch rerun still
# regressed to 68.4%. Both trained on letters. This run keeps that gentle dose
# and changes the target instead — reasoning, over material the double gate
# proved the model cannot already produce. If the data was the problem, this is
# where it shows; if the dose was the problem, the ladder below will show a
# checkpoint that beats the bar before the drift sets in.
#
# Pinned to NODE card 0 — card 4 is Track A's GRPO, 6/7 are other tenants.
# telelogs-base is read-only in this pod, so PROJ_ROOT is the writable side.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
BASE=/workspace/telelogs-base
ROOT=/workspace/teleqna-spare4
INFRA="$BASE/runs/teleqna-sft/infra"
PYT="$BASE/venvs/venv-train/bin/python"
export BASE_MODEL="$BASE/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEV="$BASE/runs/teleqna-sft/data/dev1000.jsonl"
RUN="${RUN:-distill1}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
echo "node card 0 free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: node card 0 is not free"; exit 14; }

echo "#### build distill set $(date -Iseconds)"
"$PYT" "$INFRA/build_distill_set.py" \
  --items "$ROOT/results/grounded_full_kept.jsonl" \
  --out "$ROOT/results/distill_v1.jsonl"
wc -l "$ROOT/results/distill_v1.jsonl"

export PROJ_ROOT="$ROOT" TRAIN_DATA="$ROOT/results/distill_v1.jsonl"
export LORA_R=16 RUN_NAME="$RUN" EPOCHS=1 LR=2e-5 BATCH=16 ACCUM=2 MAXLEN=1024
# Many rungs, not three: the whole question is whether a point exists on the
# curve that gains before it drifts, and 300-step saves were too coarse to see it.
export SAVE_STEPS=25 SAVE_LIMIT=100

echo "#### $RUN TRAIN START $(date -Iseconds)"
timeout 21600 "$PYT" "$INFRA/train_teleqna.py"
echo "#### $RUN TRAIN DONE $(date -Iseconds)"

# Both serving modes at every rung. The A/A references on this stack are
# base no-think 74.00 and base thinking 75.10; thinking is the deployed mode,
# but a gain that appears in only one of them is a red flag worth seeing.
for ckpt in "$ROOT"/models/teleqna-$RUN-checkpoints/checkpoint-*; do
  [ -d "$ckpt" ] || continue
  step=$(basename "$ckpt" | cut -d- -f2)
  out="$ROOT/results/dev1000_${RUN}_step${step}.json"
  [ -e "$out" ] || timeout 3600 "$PYT" "$INFRA/eval_dev.py" \
    --base "$BASE_MODEL" --adapter "$ckpt" --batch 32 \
    --data "$DEV" --out "$out"
  grep -o '"accuracy": [0-9.]*' "$out" | head -1 | sed "s|^|no-think step$step |"
done
echo "#### $RUN NOTHINK SWEEP DONE $(date -Iseconds)"
