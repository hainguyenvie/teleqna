#!/bin/bash
# DPO run 2: anchors + stronger KL.
#   - pairs v2: 25,080 fix (student's real wrong picks) + 24,700 anchors
#     (student-correct rows, rejected=distractor) — anchors give ~zero
#     gradient while the model stays right, pull back when it drifts.
#   - beta 0.5 (run 1 used 0.2; its best point was before the first
#     checkpoint, i.e. even 150 steps overshot)
#   - checkpoints every 100 steps, dev-1000 sweep after training
# Afterwards: permutation evals of run 1's best point (step150) and this
# run's best-so-far, to size the baseline's position-memorisation rent.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-train/bin/python"
export BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export PROJ_ROOT="$ROOT" TRAIN_DATA="$ROOT/data/dpo_pairs_v2.jsonl"
export LORA_R=16 RUN_NAME=dpo2 BETA=0.5 EPOCHS=0.6 LR=1e-5 BATCH=16 ACCUM=2
export SAVE_STEPS=100
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### teleqna-dpo2 TRAIN START $(date -Iseconds)"
timeout 21600 "$PY" "$ROOT/infra/train_dpo.py"
echo "#### teleqna-dpo2 TRAIN DONE $(date -Iseconds)"

for ckpt in "$ROOT"/models/teleqna-dpo2-checkpoints/checkpoint-* "$ROOT"/models/teleqna-dpo2-adapter; do
  [ -d "$ckpt" ] || continue
  name=$(basename "$ckpt" | sed 's/checkpoint-/step/; s/teleqna-dpo2-adapter/final/')
  out="$ROOT/results/dev1000_dpo2_${name}.json"
  [ -e "$out" ] && continue
  echo "#### eval dev-1000 @$name"
  timeout 3600 "$PY" "$ROOT/infra/eval_dev.py" \
    --base "$BASE_MODEL" --adapter "$ckpt" \
    --data "$ROOT/data/dev1000.jsonl" --out "$out"
done

echo "#### perm eval: dpo1 step150 (memorisation-rent probe)"
timeout 3600 "$PY" "$ROOT/infra/eval_dev.py" \
  --base "$BASE_MODEL" \
  --adapter "$ROOT/models/teleqna-dpo1-checkpoints/checkpoint-150" \
  --data "$ROOT/data/dev1000.jsonl" --permute \
  --out "$ROOT/results/dev1000perm_dpo1_step150.json"
echo "#### teleqna-dpo2 ALL DONE $(date -Iseconds)"
