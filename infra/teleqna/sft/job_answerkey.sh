#!/usr/bin/env bash
# Distil the answer key into the weights, and measure what survives the trip.
#
# NOT SUBMITTABLE. Both adapters are trained on the benchmark's own
# `explanation` field. This job exists to measure the one assumption the whole
# teacher route rests on and that nobody has tested:
#
#   arm 0.1: a fact IN CONTEXT is worth +22.3 points (97.50 vs 75.20).
#   this job: what is the same fact worth once it is IN THE WEIGHTS?
#
# Give the procedure the perfect corpus - the exact facts the benchmark tests,
# one clean sentence each, no retrieval loss, no generation noise, no gate - and
# whatever comes out is the ceiling of distillation as currently built. Every
# honest number from the Tele-Data pipeline gets discounted against it.
#
#   fact_all   10,000 facts, stated bare: no question, no options, no letter.
#              The model cannot memorise an answer, only acquire the fact.
#              Scored on dev-1000, which is in-sample. => the ceiling.
#   fact_held  same procedure, same shape, the 9,000 rows NOT in dev-1000.
#              Those facts cannot help these questions, so this separates
#              "learned this fact" from "reading telecom prose helps".
#
#   fact_all - fact_held = the leakage budget of the distillation route,
#   the analogue of the +0.67 pp already reported for prompt tuning.
#
# LoRA rank 128, above every prior run here (16/64). A ceiling measurement must
# not be capacity-bound, or it reports the adapter's size rather than the
# method's limit. EPOCHS=8 for the same reason: the capacity literature puts
# knowledge acquisition at many exposures, and a one-epoch null result would be
# unreadable.
#
# Stack: vLLM 0.11.0, the same as base/dpo3/grpo/distill/ceiling, so the
# reference is the stored 75.20 and no arm needs re-measuring.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYT="$BASE_ROOT/venvs/venv-train/bin/python"
PYV="$BASE_ROOT/venvs/venv/bin/python"
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OUT="$ROOT/results/answerkey"
export BASE_MODEL="$BASE" PROJ_ROOT="$ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LORA_R=128 EPOCHS=8 LR=1e-4 BATCH=16 ACCUM=2 MAXLEN=256 SAVE_STEPS=1000
mkdir -p "$OUT"

for i in $(seq 1 120); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card busy (${free_mib}MiB free) — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free_mib" -gt 100000 ] || { echo "ABORT: card still busy after 60min"; exit 14; }
echo "card free=${free_mib}MiB  $(date -Iseconds)"

# mcq_all runs last and is the least informative: its answer is known in
# advance (a model trained on the test rows with their gold letters will score
# near 100 and mean nothing). It is here so the number exists in writing rather
# than as an assumption, and so it is the arm that gets dropped if the card is
# taken. Its rows carry a full MCQ prompt, so MAXLEN has to rise - the trainer
# hard-fails on an over-length row instead of truncating - and 3 epochs is
# already past the point where memorisation saturates.
for arm in fact_all fact_held mcq_all; do
  DATA="$ROOT/data/answerkey_${arm}.jsonl"
  [ -s "$DATA" ] || { echo "ABORT: missing $DATA"; exit 15; }
  ADAPTER="$ROOT/models/teleqna-ak-${arm}-adapter"
  case "$arm" in
    mcq_all) ARM_MAXLEN=1024; ARM_EPOCHS=3 ;;
    *)       ARM_MAXLEN=256;  ARM_EPOCHS=8 ;;
  esac
  if [ ! -d "$ADAPTER" ]; then
    echo "#### train $arm START $(date -Iseconds)  rows=$(wc -l < "$DATA") maxlen=$ARM_MAXLEN epochs=$ARM_EPOCHS"
    TRAIN_DATA="$DATA" RUN_NAME="ak-${arm}" ALLOW_OVERWRITE=1 \
      MAXLEN="$ARM_MAXLEN" EPOCHS="$ARM_EPOCHS" \
      timeout 21600 "$PYT" "$ROOT/infra/train_teleqna.py"
    echo "#### train $arm DONE $(date -Iseconds)"
  else
    echo "skip train $arm (adapter exists)"
  fi
done

# One model load, both adapters. --max-lora-rank must match LORA_R or vLLM
# refuses the adapter at load time rather than at first use.
echo "#### eval START $(date -Iseconds)"
timeout 10800 "$PYV" "$ROOT/infra/eval_dev_vllm.py" \
  --base "$BASE" --data "$ROOT/data/dev1000.jsonl" --out-dir "$OUT" \
  --tag ak --thinking --gpu-mem 0.85 --max-lora-rank 128 \
  --adapter "factall=$ROOT/models/teleqna-ak-fact_all-adapter" \
  --adapter "factheld=$ROOT/models/teleqna-ak-fact_held-adapter" \
  --adapter "mcqall=$ROOT/models/teleqna-ak-mcq_all-adapter"
echo "#### ALL DONE $(date -Iseconds)"
