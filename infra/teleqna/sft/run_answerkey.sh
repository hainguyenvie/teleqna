#!/usr/bin/env bash
# Exp B, run from the login pod instead of the job runner.
#
# NOT SUBMITTABLE — every adapter here is trained on the benchmark's own
# `explanation` field. See job_answerkey.sh for the full rationale; this is the
# same experiment relocated, because the runner's card (4) has ~107GB held by
# the orphaned vLLM child of the generation job we stopped, and this service
# account cannot exec into that pod to reap it.
#
# Relocating changes the serving stack from vLLM 0.11.0 to the nightly, so the
# reference is the base arm measured on the nightly by run_haystack.sh (75.10),
# not the stored 75.20. Both experiments then share one reference, which is the
# only way the haystack sweep and this can appear in one table.
#
# Cards are picked per stage and cards 0/1/4 are avoided: 0 is GRPO, 1 is the
# haystack sweep running concurrently, 4 is the held card.
set -euo pipefail
HOME_ROOT=/home/tensara
MODEL="$HOME_ROOT/projects/telelogs/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
OUT="$ROOT/results/answerkey"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export BASE_MODEL="$MODEL" PROJ_ROOT="$ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_R=128 LR=1e-4 BATCH=16 ACCUM=2 SAVE_STEPS=1000 SAVE_LIMIT=1
mkdir -p "$OUT"

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
NEED_MIB="${NEED_MIB:-60000}"     # LoRA r128 on an 8B in bf16, plus optimiser
pick_card() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ' | awk -F, -v need="$NEED_MIB" -v avoid="${AVOID:-0,1,4}" '
        BEGIN { split(avoid, a, ","); for (i in a) skip[a[i]] = 1 }
        !skip[$1] && $2 >= need && $2 > best { best = $2; idx = $1 }
        END { if (best) print idx, best }'
}
wait_for_card() {
  for try in $(seq 1 120); do
    read -r CARD FREE <<< "$(pick_card)"
    [ -n "${CARD:-}" ] && return 0
    [ "$try" = 1 ] && echo "no card with ${NEED_MIB}MiB free — waiting $(date -Iseconds)"
    sleep 30
  done
  echo "ABORT: no usable card after 60min"; exit 14
}

for arm in ${ARMS:-fact_all fact_held}; do
  DATA="$ROOT/data/answerkey_${arm}.jsonl"
  ADAPTER="$ROOT/models/teleqna-ak-${arm}-adapter"
  [ -s "$DATA" ] || { echo "ABORT: missing $DATA"; exit 15; }
  [ -d "$ADAPTER" ] && { echo "skip train $arm (adapter exists)"; continue; }
  case "$arm" in
    mcq_all) ARM_MAXLEN=1024; ARM_EPOCHS=3 ;;
    *)       ARM_MAXLEN=256;  ARM_EPOCHS=8 ;;
  esac
  wait_for_card
  export CUDA_VISIBLE_DEVICES="$CARD"
  echo "#### train $arm START $(date -Iseconds)  card=$CARD free=${FREE}MiB rows=$(wc -l < "$DATA")"
  set +e
  TRAIN_DATA="$DATA" RUN_NAME="ak-${arm}" ALLOW_OVERWRITE=1 \
    MAXLEN="$ARM_MAXLEN" EPOCHS="$ARM_EPOCHS" timeout 21600 "$PY" "$ROOT/infra/train_teleqna.py"
  rc=$?
  set -e
  [ $rc -ne 0 ] && { echo "#### train $arm FAILED rc=$rc $(date -Iseconds)"; continue; }
  echo "#### train $arm DONE $(date -Iseconds)"
done

ADAPTER_ARGS=()
for arm in ${ARMS:-fact_all fact_held}; do
  p="$ROOT/models/teleqna-ak-${arm}-adapter"
  [ -d "$p" ] && ADAPTER_ARGS+=(--adapter "ak${arm}=$p")
done
[ ${#ADAPTER_ARGS[@]} -gt 0 ] || { echo "ABORT: nothing trained"; exit 16; }

NEED_MIB=28000
wait_for_card
export CUDA_VISIBLE_DEVICES="$CARD"
GM=$(python3 -c "print(round(max(0.18, min(0.45, ($FREE - 6000) / $TOTAL)), 2))")
echo "#### eval START $(date -Iseconds)  card=$CARD gpu_mem=$GM"
# --max-lora-rank must match LORA_R; vLLM refuses the adapter at load, not use.
timeout 10800 "$PY" "$ROOT/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$ROOT/data/dev1000.jsonl" --out-dir "$OUT" \
  --tag ak --thinking --gpu-mem "$GM" --max-model-len 8192 \
  --max-lora-rank 128 "${ADAPTER_ARGS[@]}"
echo "#### ALL DONE $(date -Iseconds)"
