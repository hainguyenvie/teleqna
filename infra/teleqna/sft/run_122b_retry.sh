#!/usr/bin/env bash
# Re-score the rows a 122B run left unparsed, with a much larger token ceiling.
#
# Every unparsed row on otlite was a truncation — no </think> anywhere in the
# completion, i.e. the model hit the ceiling mid-thought and never reached an
# ANSWER line. None had finished reasoning and then answered in prose. So this
# is a budget problem and not a format one, and re-running exactly those rows
# with more room is the entire fix. On otlite it took unparsed from 31 to 1 and
# the set from 81.50 to 82.80.
#
# Worth knowing before reading the gain: truncation selects hard questions. The
# otlite retry arm scored 41.94% against 81.50% for the set as a whole, and the
# completion lengths had said so in advance (correct 6.4k chars, wrong 12.3k,
# unparsed 21.3k). So recovering N rows buys far less than N times the base rate,
# and predicting otherwise once cost a 2.2pp estimate against a 1.3pp outcome.
#
# Parameterised over the source run. Defaults reproduce the otlite retry; ot-full
# is the same call with SRCRES/SRCSET/RETRYSET/TAG pointed at the 10,000-row run.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-1,5}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
HOME_ROOT=/home/tensara
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B}"
PY="${PY:-$HOME_ROOT/venv-vllm-nightly/bin/python}"
export PATH="$(dirname "$PY"):$PATH"
INFRA=$HOME_ROOT/projects/telelogs/runs/teleqna-sft/infra
DATA=$HOME_ROOT/projects/telelogs/runs/teleqna-sft/data
OUT="${OUT:-$HOME_ROOT/projects/telelogs/runs/teleqna-sft/results/vllm122b}"

SRCRES="${SRCRES:-$OUT/otlite_base_think.json}"
SRCSET="${SRCSET:-$DATA/otlite1000.jsonl}"
RETRYSET="${RETRYSET:-$DATA/otlite_retry.jsonl}"
TAG="${TAG:-otliteretry}"
MAX_NEW="${MAX_NEW:-16000}"
MAX_LEN="${MAX_LEN:-20480}"

# max_num_seqs has to come down as max_model_len goes up. The 122B's hybrid
# linear attention keeps one unpageable recurrent state block per in-flight
# sequence and the pool is fixed (~466 blocks measured), while KV per sequence
# scales with the window — 256 seqs fit at 8192 but will not at 36864 out of the
# ~38GB left over once 233GB of BF16 weights are resident. There are only a few
# hundred rows here anyway, so depth costs nothing.
SEQS="${SEQS:-64}"
ENGINE_KWARGS="${ENGINE_KWARGS:-{\"language_model_only\": true, \"disable_custom_all_reduce\": true, \"max_num_seqs\": $SEQS\}}"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

python3 "$INFRA/make_retry_set.py" "$SRCRES" "$SRCSET" "$RETRYSET"

IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
for c in "${IDX[@]}"; do
  for i in $(seq 1 90); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
    [ "$free" -gt 135000 ] && break
    [ "$i" = 1 ] && echo "card $c busy (${free}MiB) — waiting $(date -Iseconds)"
    sleep 10
  done
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c stuck at ${free}MiB"; exit 14; }
done

echo "#### $TAG START $(date -Iseconds)  max_new=$MAX_NEW max_len=$MAX_LEN seqs=$SEQS"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$MODEL" --data "$RETRYSET" --out-dir "$OUT" \
  --tag "$TAG" --with-base --thinking --tp 2 --gpu-mem 0.95 \
  --max-new "$MAX_NEW" --max-model-len "$MAX_LEN" \
  ${ENGINE_KWARGS:+--engine-kwargs "$ENGINE_KWARGS"}
echo "#### $TAG DONE $(date -Iseconds)"

python3 "$INFRA/merge_retry.py" "$SRCRES" "$OUT/${TAG}_base_think.json"
