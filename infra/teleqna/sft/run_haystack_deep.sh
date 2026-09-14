#!/usr/bin/env bash
# The deep end of the haystack sweep: as many notes as the model can physically hold.
#
# NOT SUBMITTABLE — the notes are the benchmark's `explanation` field.
#
# Why this stops where it stops, and it is arithmetic rather than a choice:
#
#     all 10,000 notes   1,386,235 chars  ~= 346,558 tokens
#     all 1,000 dev notes  138,882 chars  ~=  34,720 tokens
#     Qwen3-8B context                        32,768 tokens
#
# "Put every note in the context" is short by a factor of ten, and even the
# notes belonging to the split being scored do not fit. K=512 is the largest
# arm that leaves room for a 6,000-token thinking budget: ~17.8k of notes plus
# ~6k of output inside a 32,768 window. The right note is then one of 512, all
# of them from the same subject — a precision of 0.2%.
#
# Run as a separate script rather than by editing run_haystack.sh: bash reads a
# script incrementally, so editing a file that is mid-execution corrupts the
# run that is already going.
set -euo pipefail
HOME_ROOT=/home/tensara
MODEL="$HOME_ROOT/projects/telelogs/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
INFRA="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/infra"
DATA="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/data"
OUT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft/results/haystack"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
# A 32k window over a long prompt needs real KV cache, so the bar is higher
# than the shallow sweep's 28GB.
NEED_MIB="${NEED_MIB:-55000}"
pick_card() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ' | awk -F, -v need="$NEED_MIB" -v avoid="${AVOID:-0,4}" '
        BEGIN { split(avoid, a, ","); for (i in a) skip[a[i]] = 1 }
        !skip[$1] && $2 >= need && $2 > best { best = $2; idx = $1 }
        END { if (best) print idx, best }'
}

for arm in hay256 hay512; do
  f="$DATA/dev1000_${arm}.jsonl"
  [ -s "$f" ] || { echo "SKIP $arm: missing $f"; continue; }
  [ -s "$OUT/${arm}_base_think.json" ] && { echo "skip $arm (done)"; continue; }
  case "$arm" in
    hay256) MML=16384 ;;
    hay512) MML=32768 ;;
  esac
  for try in $(seq 1 120); do
    read -r CARD FREE <<< "$(pick_card)"
    [ -n "${CARD:-}" ] && break
    [ "$try" = 1 ] && echo "no card with ${NEED_MIB}MiB free — waiting $(date -Iseconds)"
    sleep 30
  done
  [ -n "${CARD:-}" ] || { echo "ABORT: no usable card for $arm"; exit 14; }
  GM=$(python3 -c "print(round(max(0.25, min(0.55, ($FREE - 6000) / $TOTAL)), 2))")
  export CUDA_VISIBLE_DEVICES="$CARD"
  echo "#### $arm START $(date -Iseconds)  card=$CARD free=${FREE}MiB gpu_mem=$GM maxlen=$MML"
  set +e
  timeout 10800 "$PY" "$INFRA/eval_dev_vllm.py" \
    --base "$MODEL" --data "$f" --out-dir "$OUT" --tag "$arm" \
    --with-base --thinking --gpu-mem "$GM" --max-model-len "$MML"
  rc=$?
  set -e
  [ $rc -ne 0 ] && echo "#### $arm FAILED rc=$rc $(date -Iseconds)"
  echo "#### $arm DONE $(date -Iseconds)"
done
echo "#### DEEP DONE $(date -Iseconds)"
